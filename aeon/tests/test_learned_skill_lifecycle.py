"""Revision-bound learned-skill lifecycle regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeon.core.skills.lifecycle import (
    LearnedSkillError,
    LearnedSkillStore,
    skill_revision,
)


EVIDENCE = [{"note_id": f"note-{'a' * 32}", "revision": "b" * 64}]


def _ready_skill(
    tmp_path: Path, *, skill_name: str = "reliable_export"
) -> tuple[LearnedSkillStore, str, dict[str, object]]:
    store = LearnedSkillStore(tmp_path / "skills")
    revision = skill_revision("1. Export.\n2. Verify.")
    ready = store.save_protocol(
        category="conversion",
        skill_name=skill_name,
        content_revision=revision,
        evidence=EVIDENCE,
    )
    return store, revision, ready


@pytest.mark.parametrize(
    ("outcome", "expected_status", "counter"),
    [
        ("success", "ready", "successes"),
        ("adapted", "needs_review", "adaptations"),
        ("failed", "quarantined", "failures"),
    ],
)
def test_ready_activation_and_outcome_transitions(
    tmp_path: Path,
    outcome: str,
    expected_status: str,
    counter: str,
) -> None:
    store, revision, ready = _ready_skill(tmp_path, skill_name=outcome)

    assert ready["status"] == "ready"
    assert ready["metadata_stale"] is False
    assert ready["usage"] == {
        "activations": 0,
        "successes": 0,
        "adaptations": 0,
        "failures": 0,
        "not_applicable": 0,
    }

    activated = store.record_activation(
        category="conversion",
        skill_name=outcome,
        content_revision=revision,
    )
    assert activated["status"] == "ready"
    assert activated["usage"]["activations"] == 1

    final = store.record_outcome(
        category="conversion",
        skill_name=outcome,
        content_revision=revision,
        outcome=outcome,
        note=f"Observed {outcome} outcome.",
    )
    assert final["status"] == expected_status
    assert final["usage"]["activations"] == 1
    assert final["usage"][counter] == 1
    assert sum(
        final["usage"][key]
        for key in ("successes", "adaptations", "failures", "not_applicable")
    ) == 1
    assert final["last_outcome"] == {
        "outcome": outcome,
        "note": f"Observed {outcome} outcome.",
        "at": final["last_outcome"]["at"],
        "content_revision": revision,
    }

    if expected_status != "ready":
        with pytest.raises(LearnedSkillError, match=expected_status):
            store.record_activation(
                category="conversion",
                skill_name=outcome,
                content_revision=revision,
            )


def test_stale_content_digest_is_read_as_needs_review(tmp_path: Path) -> None:
    store, original_revision, _ready = _ready_skill(tmp_path)
    changed_revision = skill_revision("1. Export differently.\n2. Verify.")

    stale = store.read(
        "conversion",
        "reliable_export",
        current_content_revision=changed_revision,
    )

    assert stale is not None
    assert stale["content_revision"] == original_revision
    assert stale["metadata_stale"] is True
    assert stale["status"] == "needs_review"
    assert store.read("conversion", "reliable_export")["status"] == "ready"
    with pytest.raises(LearnedSkillError, match="needs_review"):
        store.record_activation(
            category="conversion",
            skill_name="reliable_export",
            content_revision=changed_revision,
        )


def test_remove_deletes_metadata_without_touching_protocol(tmp_path: Path) -> None:
    instance_dir = tmp_path / "skills"
    protocol = instance_dir / "conversion" / "reliable_export.txt"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("1. Export.\n2. Verify.\n", encoding="utf-8")
    store = LearnedSkillStore(instance_dir)
    revision = skill_revision(protocol.read_text(encoding="utf-8"))
    store.save_protocol(
        category="conversion",
        skill_name="reliable_export",
        content_revision=revision,
        evidence=EVIDENCE,
    )
    metadata = (
        instance_dir / ".skill-state" / "conversion" / "reliable_export.json"
    )
    assert metadata.is_file()

    store.remove("conversion", "reliable_export")

    assert store.read("conversion", "reliable_export") is None
    assert not metadata.exists()
    assert protocol.read_text(encoding="utf-8") == "1. Export.\n2. Verify.\n"
    store.remove("conversion", "reliable_export")


def test_outcome_metadata_refuses_secret_like_notes(tmp_path: Path) -> None:
    store, revision, _ready = _ready_skill(tmp_path)

    with pytest.raises(LearnedSkillError, match="credentials"):
        store.record_outcome(
            category="conversion",
            skill_name="reliable_export",
            content_revision=revision,
            outcome="failed",
            note="token=sk-" + "a" * 32,
        )

    assert store.read("conversion", "reliable_export")["status"] == "ready"


def test_secret_like_legacy_outcome_fails_closed_on_read(tmp_path: Path) -> None:
    store, revision, _ready = _ready_skill(tmp_path)
    metadata = (
        tmp_path
        / "skills"
        / ".skill-state"
        / "conversion"
        / "reliable_export.json"
    )
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["last_outcome"] = {
        "outcome": "failed",
        "note": "password=sk-" + "a" * 32,
        "at": 1.0,
        "content_revision": revision,
    }
    metadata.write_text(json.dumps(document) + "\n", encoding="utf-8")
    metadata.chmod(0o600)

    with pytest.raises(LearnedSkillError, match="outcome is invalid"):
        store.read("conversion", "reliable_export")
