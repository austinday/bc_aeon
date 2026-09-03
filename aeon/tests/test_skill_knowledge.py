"""Durable per-agent skill-wiki storage regressions."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from aeon.core.skills.knowledge import SkillKnowledgeError, SkillKnowledgeStore


def _learning_claim() -> dict[str, object]:
    return {
        "candidate_skill_path": "conversion/reliable_export",
        "procedure": "Inspect the failed export, correct its codec, then retry.",
        "verification": "Open the exported artifact and verify its codec metadata.",
        "procedure_stable": True,
        "uncertainty": "low",
    }


def _failure_then_success_experience() -> dict[str, object]:
    return {
        "request_id": "request-1",
        "attempt_count": 2,
        "failure_count": 1,
        "success_count": 1,
        "recovered_after_failure": True,
        "receipts": [
            {
                "tool": "export_artifact",
                "status": "failed",
                "error_code": "unsupported-codec",
                "summary_sha256": "a" * 64,
            },
            {
                "tool": "export_artifact",
                "status": "ok",
                "error_code": "",
                "summary_sha256": "b" * 64,
            },
        ],
    }


def test_note_lifecycle_is_private_and_revision_checked(tmp_path: Path) -> None:
    root = tmp_path / "skill-wiki"
    store = SkillKnowledgeStore(root)

    created = store.save_note(
        title="Reliable conversion",
        content="The verified sequence and its evidence.",
        related_skill_paths=["conversion/reliable_export"],
    )

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / f"{created['id']}.json").stat().st_mode) == 0o600
    assert store.read_note(created["id"])["content"] == created["content"]
    assert [note["id"] for note in store.list_notes()] == [created["id"]]

    updated = store.save_note(
        note_id=created["id"],
        expected_revision=created["revision"],
        title="Reliable conversion",
        content="The corrected, benchmarked sequence.",
        related_skill_paths=["conversion/reliable_export"],
    )
    assert updated["revision"] != created["revision"]
    with pytest.raises(SkillKnowledgeError, match="changed since it was loaded"):
        store.save_note(
            note_id=created["id"],
            expected_revision=created["revision"],
            title="Stale",
            content="Must not win.",
            related_skill_paths=[],
        )
    with pytest.raises(SkillKnowledgeError, match="changed since it was loaded"):
        store.delete_note(created["id"], expected_revision=created["revision"])

    store.delete_note(created["id"], expected_revision=updated["revision"])
    assert store.list_notes() == []


def test_note_store_refuses_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    root = tmp_path / "skill-wiki"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(SkillKnowledgeError, match="not private"):
        SkillKnowledgeStore(root).save_note(
            title="No redirect",
            content="This must not follow a symlink.",
            related_skill_paths=[],
        )


def test_note_validates_related_skill_paths(tmp_path: Path) -> None:
    with pytest.raises(SkillKnowledgeError, match="related_skill_paths"):
        SkillKnowledgeStore(tmp_path / "wiki").save_note(
            title="Unsafe relation",
            content="Evidence.",
            related_skill_paths=["../outside"],
        )


def test_note_refuses_secret_like_credentials(tmp_path: Path) -> None:
    with pytest.raises(SkillKnowledgeError, match="credentials"):
        SkillKnowledgeStore(tmp_path / "wiki").save_note(
            title="Unsafe note",
            content="api_key=sk-" + "a" * 32,
            related_skill_paths=[],
        )


def test_search_ranks_title_then_related_path_then_content(tmp_path: Path) -> None:
    store = SkillKnowledgeStore(tmp_path / "wiki")
    exact = store.save_note(
        title="Cache export",
        content="The exact verified procedure.",
    )
    related = store.save_note(
        title="Indexed procedure",
        content="The relationship carries the searchable terms.",
        related_skill_paths=["cache/export"],
    )
    body = store.save_note(
        title="Local observation",
        content="The cache remains available locally.",
    )
    unrelated = store.save_note(
        title="Network retry",
        content="Back off after a transport failure.",
    )

    matches = store.search_notes("cache export")

    assert [match["id"] for match in matches] == [
        exact["id"],
        related["id"],
        body["id"],
    ]
    assert [match["search_score"] for match in matches] == [26, 12, 1]
    assert unrelated["id"] not in {match["id"] for match in matches}
    assert [match["id"] for match in store.search_notes("cache export", limit=2)] == [
        exact["id"],
        related["id"],
    ]


def test_ordinary_update_preserves_learning_provenance(tmp_path: Path) -> None:
    store = SkillKnowledgeStore(tmp_path / "wiki")
    created = store.save_note(
        title="Recovered export",
        content="The first codec failed and the corrected codec succeeded.",
        related_skill_paths=["conversion/reliable_export"],
        learning=_learning_claim(),
        experience=_failure_then_success_experience(),
    )

    updated = store.save_note(
        note_id=created["id"],
        expected_revision=created["revision"],
        title="Recovered export, clarified",
        content="Clarified explanation without replacing the captured evidence.",
        related_skill_paths=["conversion/reliable_export"],
    )

    assert updated["origin"] == created["origin"] == {"kind": "agent-authored"}
    assert updated["learning"] == created["learning"]
    assert updated["experience"] == created["experience"]
    assert updated["skill_evidence_eligible"] is True
    assert updated["created_at"] == created["created_at"]
    assert updated["revision"] != created["revision"]


def test_learning_claim_can_be_retracted_by_revision_checked_update(tmp_path: Path) -> None:
    store = SkillKnowledgeStore(tmp_path / "wiki")
    created = store.save_note(
        title="Recovered export",
        content="A procedure that later proved too variable.",
        related_skill_paths=["conversion/reliable_export"],
        learning=_learning_claim(),
        experience=_failure_then_success_experience(),
    )

    corrected = store.save_note(
        note_id=created["id"],
        expected_revision=created["revision"],
        title="Recovered export, retracted",
        content="Current evidence shows the procedure is not stable enough for a skill.",
        related_skill_paths=["conversion/reliable_export"],
        clear_learning=True,
    )

    assert corrected["learning"] is None
    assert corrected["experience"] is None
    assert corrected["skill_evidence_eligible"] is False


def test_transferred_learning_is_not_local_skill_evidence(tmp_path: Path) -> None:
    note = SkillKnowledgeStore(tmp_path / "wiki").save_note(
        title="Transferred recovered export",
        content="Valid context copied from another agent.",
        related_skill_paths=["conversion/reliable_export"],
        origin={
            "kind": "transferred",
            "source_instance_id": "source-agent",
            "locally_earned": "false",
        },
        learning=_learning_claim(),
        experience=_failure_then_success_experience(),
    )

    assert note["origin"]["kind"] == "transferred"
    assert note["learning"] == _learning_claim()
    assert note["experience"] == _failure_then_success_experience()
    assert note["skill_evidence_eligible"] is False


def test_local_failure_then_success_is_eligible_skill_evidence(tmp_path: Path) -> None:
    note = SkillKnowledgeStore(tmp_path / "wiki").save_note(
        title="Locally recovered export",
        content="A failed attempt exposed the stable successful procedure.",
        related_skill_paths=["conversion/reliable_export"],
        learning=_learning_claim(),
        experience=_failure_then_success_experience(),
    )

    assert note["origin"] == {"kind": "agent-authored"}
    assert note["experience"]["failure_count"] == 1
    assert note["experience"]["success_count"] == 1
    assert note["experience"]["recovered_after_failure"] is True
    assert note["skill_evidence_eligible"] is True
