"""Bounded, digest-bound skill context regressions."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aeon.core.skills.manager import (
    INSTANCE_SKILLS_DIR_ENV,
    MAX_SKILL_CONTENT_BYTES,
    SkillContentError,
    SkillContentTooLarge,
    SkillsManager,
)
from aeon.core.skills.lifecycle import MAX_PRIVATE_SKILLS, skill_revision
from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.core.worker import Worker
from aeon.tests.test_worker_protocol import ScriptedLLM
from aeon.tools.skills_runtime import (
    ActivateSkillTool,
    CreateSkillTool,
    DeactivateSkillTool,
    DeleteSkillTool,
    RememberSkillKnowledgeTool,
)


def _skill_environment(root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["AEON_SKILLS_DIR"] = str(root)
    environment.pop(INSTANCE_SKILLS_DIR_ENV, None)
    environment.pop("AEON_REMOTE_INSTANCE_ID", None)
    environment.pop("AEON_CHAT_TRANSCRIPT_PATH", None)
    return environment


def test_oversized_skill_is_explicitly_refused() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        category = root / "review"
        category.mkdir()
        (category / "huge.txt").write_bytes(b"x" * (MAX_SKILL_CONTENT_BYTES + 1))
        with patch.dict(os.environ, _skill_environment(root), clear=True):
            with pytest.raises(SkillContentTooLarge, match="maximum"):
                SkillsManager().get_skill_content("review", "huge")
            worker = SimpleNamespace(
                active_skill=None,
                expanded_categories=set(),
                request_contract=SimpleNamespace(request_id="request-1"),
            )
            receipt = ActivateSkillTool(worker).execute("review/huge")

    assert "cannot be activated" in receipt
    assert worker.active_skill is None


def test_skill_symlink_is_not_loaded() -> None:
    with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
        root = Path(temporary)
        category = root / "review"
        category.mkdir()
        target = Path(other) / "outside.txt"
        target.write_text("outside", encoding="utf-8")
        (category / "linked.txt").symlink_to(target)
        # Keep one ordinary skill so the environment root is selected.
        (category / "ordinary.txt").write_text("ordinary", encoding="utf-8")
        with patch.dict(os.environ, _skill_environment(root), clear=True):
            with pytest.raises(SkillContentError, match="symlink"):
                SkillsManager().get_skill_content("review", "linked")


def test_activation_records_exact_content_digest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        category = root / "review"
        category.mkdir()
        content = "1. Inspect.\n2. Verify."
        (category / "bounded.txt").write_text(content, encoding="utf-8")
        worker = SimpleNamespace(
            active_skill=None,
            expanded_categories=set(),
            request_contract=SimpleNamespace(request_id="request-1"),
        )
        with patch.dict(os.environ, _skill_environment(root), clear=True):
            receipt = ActivateSkillTool(worker).execute("review/bounded")

    assert "ACTIVE" in receipt
    assert worker.active_skill["path"] == "review/bounded"
    assert worker.active_skill["content"] == content
    assert worker.active_skill["sha256"] == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    assert worker.active_skill["scope"] == "shared"
    assert worker.active_skill["paused"] is False
    assert "ADVISORY" in receipt


def test_second_activation_cannot_discard_first_skill_outcome() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        category = root / "review"
        category.mkdir()
        (category / "first.txt").write_text("First playbook.", encoding="utf-8")
        (category / "second.txt").write_text("Second playbook.", encoding="utf-8")
        worker = SimpleNamespace(
            active_skill=None,
            expanded_categories=set(),
            request_contract=SimpleNamespace(request_id="request-1"),
        )
        with patch.dict(os.environ, _skill_environment(root), clear=True):
            assert "ACTIVE" in ActivateSkillTool(worker).execute("review/first")
            refused = ActivateSkillTool(worker).execute("review/second")

    assert "Deactivate it with an honest outcome" in refused
    assert worker.active_skill["path"] == "review/first"


def test_create_skill_rejects_content_that_cannot_be_activated() -> None:
    worker = SimpleNamespace(active_skill=None, expanded_categories=set())
    receipt = CreateSkillTool(worker).execute(
        "review",
        "huge",
        "x" * (MAX_SKILL_CONTENT_BYTES + 1),
    )
    assert "maximum" in receipt


def test_learned_skill_catalog_is_deliberately_small() -> None:
    assert MAX_PRIVATE_SKILLS == 16


def test_compact_context_keeps_full_digest_bound_active_protocol() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        category = root / "fleet"
        category.mkdir()
        content = "1. Preserve fleet policy.\n2. Verify exact receipts."
        (category / "review.txt").write_text(content, encoding="utf-8")
        worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
        worker.persist_session = False
        worker.active_skill = {
            "path": "fleet/review",
            "content": content,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "scope": "shared",
            "status": "shared",
            "paused": False,
            "request_id": "",
        }
        with patch.dict(os.environ, _skill_environment(root), clear=True):
            compact = worker._compact_current_state("Audit Aeon")

    assert "ACTIVE SKILL GUIDANCE" in compact
    assert "ADVISORY" in compact
    assert content in compact
    assert worker.active_skill["sha256"] in compact


def test_active_skill_unpins_instead_of_adopting_dashboard_edit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        category = root / "fleet"
        category.mkdir()
        old = "old instructions"
        new = "new verified instructions"
        skill = category / "review.txt"
        skill.write_text(new, encoding="utf-8")
        worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
        worker.persist_session = False
        worker.active_skill = {
            "path": "fleet/review",
            "content": old,
            "sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
            "scope": "shared",
            "status": "shared",
            "paused": False,
            "request_id": "",
        }

        with patch.dict(os.environ, _skill_environment(root), clear=True):
            compact = worker._compact_current_state("Audit Aeon")

    assert new not in compact
    assert old not in compact
    assert "changed revision or origin" in compact
    assert worker.active_skill is None


def test_active_skill_unpins_after_dashboard_delete() -> None:
    worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
    worker.persist_session = False
    worker.active_skill = {
        "path": "fleet/review",
        "content": "deleted",
        "sha256": hashlib.sha256(b"deleted").hexdigest(),
        "scope": "shared",
        "status": "shared",
        "paused": False,
        "request_id": "",
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "other").mkdir()
        (root / "other" / "keep-root-selected.txt").write_text("keep", encoding="utf-8")
        with patch.dict(os.environ, _skill_environment(root), clear=True):
            compact = worker._compact_current_state("Audit Aeon")

    assert "missing or failed integrity checks" in compact
    assert worker.active_skill is None


def _private_skill(
    shared: Path, private: Path, *, content: str, name: str = "earned"
) -> dict[str, str]:
    (shared / "review").mkdir(parents=True)
    (shared / "review" / "base.txt").write_text("base", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "AEON_SKILLS_DIR": str(shared),
            INSTANCE_SKILLS_DIR_ENV: str(private),
        }
    )
    environment.pop("AEON_REMOTE_INSTANCE_ID", None)
    environment.pop("AEON_CHAT_TRANSCRIPT_PATH", None)
    with patch.dict(os.environ, environment, clear=True):
        manager = SkillsManager()
        manager.ensure_private_overlay()
        category = private / "review"
        category.mkdir(mode=0o700)
        skill_file = category / f"{name}.txt"
        skill_file.write_text(content + "\n", encoding="utf-8")
        skill_file.chmod(0o600)
        evidence = manager.knowledge_store().save_note(
            title="Recovered review procedure",
            content="The first approach failed and the revised procedure was verified.",
            related_skill_paths=[f"review/{name}"],
            learning={
                "candidate_skill_path": f"review/{name}",
                "procedure": "Use the revised review procedure.",
                "verification": "The focused check succeeds.",
                "procedure_stable": True,
                "uncertainty": "low",
            },
            experience={
                "request_id": "request-1",
                "attempt_count": 2,
                "failure_count": 1,
                "success_count": 1,
                "recovered_after_failure": True,
                "receipts": [
                    {
                        "tool": "open_file",
                        "status": "failed",
                        "error_code": "wrong_path",
                        "summary_sha256": "a" * 64,
                    },
                    {
                        "tool": "open_file",
                        "status": "ok",
                        "error_code": "",
                        "summary_sha256": "b" * 64,
                    },
                ],
            },
        )
        manager.learned_store().save_protocol(
            category="review",
            skill_name=name,
            content_revision=skill_revision(content),
            evidence=[{"note_id": evidence["id"], "revision": evidence["revision"]}],
        )
    return environment


def test_private_skill_text_never_enters_system_role(tmp_path: Path) -> None:
    secret_marker = "PRIVATE_PLAYBOOK_SENTINEL_7d6c"
    environment = _private_skill(
        tmp_path / "shared", tmp_path / "private", content=secret_marker
    )
    worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
    worker.persist_session = False
    worker._begin_protocol_request("Inspect one fixture")
    with patch.dict(os.environ, environment, clear=True):
        receipt = ActivateSkillTool(worker).execute("review/earned")
        assert "ACTIVE" in receipt
        system_message = worker._build_system_message(
            "Inspect one fixture", "", ""
        )
        current_state = worker._build_current_state_message(
            "", "", "", "", objective="Inspect one fixture"
        )
        messages, _ = worker._fit_protocol_messages(
            system_message, current_state, "Inspect one fixture", has_images=False
        )

    assert all(
        secret_marker not in str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    )
    assert any(
        secret_marker in str(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    )


def test_new_request_unpins_active_skill(tmp_path: Path) -> None:
    environment = _private_skill(
        tmp_path / "shared", tmp_path / "private", content="earned procedure"
    )
    worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
    worker.persist_session = False
    worker._begin_protocol_request("First task")
    with patch.dict(os.environ, environment, clear=True):
        ActivateSkillTool(worker).execute("review/earned")
        assert worker.active_skill is not None
        worker._begin_protocol_request("Unrelated second task")
    assert worker.active_skill is None


def test_failed_live_result_pauses_without_automatic_quarantine(tmp_path: Path) -> None:
    environment = _private_skill(
        tmp_path / "shared", tmp_path / "private", content="earned procedure"
    )
    worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
    worker.persist_session = False
    worker._begin_protocol_request("Use the earned procedure")
    with patch.dict(os.environ, environment, clear=True):
        ActivateSkillTool(worker).execute("review/earned")
        failed = ToolResult(
            tool_name="run_command",
            status=ToolStatus.FAILED,
            changed=False,
            summary="current state contradicts the procedure",
            error_code="precondition_changed",
            side_effect=SideEffect.READ_ONLY,
            call_id="call-1",
        )
        worker._record_protocol_tool_turn(
            {
                "intent": "check the procedure",
                "actions": [
                    {
                        "tool_name": "run_command",
                        "parameters": {},
                        "_call_id": "call-1",
                    }
                ],
            },
            [failed],
            1,
        )
        paused_record = SkillsManager().get_skill_record("review", "earned")
        deactivated = DeactivateSkillTool(worker).execute(
            outcome="not_applicable",
            note="The failed command was unrelated to this playbook.",
        )
        final_record = SkillsManager().get_skill_record("review", "earned")

    assert paused_record["lifecycle"]["status"] == "ready"
    assert "outcome=not_applicable" in deactivated
    assert final_record["lifecycle"]["status"] == "ready"
    assert worker.active_skill is None
    assert "PAUSED" in worker.last_observation


def test_skill_outcome_note_refuses_secret_without_unpinning(tmp_path: Path) -> None:
    environment = _private_skill(
        tmp_path / "shared", tmp_path / "private", content="earned procedure"
    )
    worker = Worker(ScriptedLLM(), print_func=lambda *_: None)
    worker.persist_session = False
    worker._begin_protocol_request("Use the earned procedure")
    with patch.dict(os.environ, environment, clear=True):
        assert "ACTIVE" in ActivateSkillTool(worker).execute("review/earned")
        refused = DeactivateSkillTool(worker).execute(
            outcome="failed", note="api_key=sk-" + "a" * 32
        )
        record = SkillsManager().get_skill_record("review", "earned")

    assert "COMMAND BLOCKED" in refused
    assert worker.active_skill is not None
    assert record["lifecycle"]["status"] == "ready"


def test_secret_like_legacy_skill_can_still_be_retired(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    private = tmp_path / "private"
    (shared / "review").mkdir(parents=True)
    (shared / "review" / "base.txt").write_text("base", encoding="utf-8")
    private.mkdir(mode=0o700)
    (private / "review").mkdir(mode=0o700)
    content = "Legacy example: api_key=sk-" + "a" * 32
    skill_file = private / "review" / "unsafe.txt"
    skill_file.write_text(content + "\n", encoding="utf-8")
    skill_file.chmod(0o600)
    environment = dict(os.environ)
    environment.update(
        {
            "AEON_SKILLS_DIR": str(shared),
            INSTANCE_SKILLS_DIR_ENV: str(private),
        }
    )
    environment.pop("AEON_REMOTE_INSTANCE_ID", None)
    environment.pop("AEON_CHAT_TRANSCRIPT_PATH", None)
    worker = SimpleNamespace(active_skill=None, expanded_categories=set())

    with patch.dict(os.environ, environment, clear=True):
        manager = SkillsManager()
        revision = manager.get_skill_record("review", "unsafe")["revision"]
        retired = DeleteSkillTool(worker).execute(
            "review/unsafe",
            expected_revision=revision,
            reason="Unsafe legacy credential example.",
        )
        notes = manager.knowledge_store().list_notes()

    assert "Retired skill" in retired
    assert not skill_file.exists()
    assert len(notes) == 1
    assert "api_key" not in notes[0]["content"]
    assert revision in notes[0]["content"]


def test_first_try_success_cannot_become_skill_but_recovery_can(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    (shared / "review").mkdir(parents=True)
    (shared / "review" / "base.txt").write_text("base", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "AEON_SKILLS_DIR": str(shared),
            INSTANCE_SKILLS_DIR_ENV: str(tmp_path / "private"),
        }
    )
    environment.pop("AEON_REMOTE_INSTANCE_ID", None)
    environment.pop("AEON_CHAT_TRANSCRIPT_PATH", None)
    result = lambda status: SimpleNamespace(
        tool_name="open_file",
        status=status,
        error_code="wrong_path" if status == "failed" else "",
        summary=f"{status} result",
    )
    worker = SimpleNamespace(
        active_skill=None,
        expanded_categories=set(),
        request_contract=SimpleNamespace(
            request_id="request-1", results=[result("ok")]
        ),
    )
    learning = {
        "candidate_skill_path": "review/recovered_path",
        "procedure": "1. Resolve the canonical path.\n2. Open that exact file.",
        "verification": "Confirm the expected file digest.",
        "procedure_stable": True,
        "uncertainty": "low",
    }
    protocol = (
        "# When to use\nThe canonical path is ambiguous.\n"
        "# Preconditions\nThe workspace root is known.\n"
        "# Procedure\n1. Resolve the canonical path.\n2. Open that exact file.\n"
        "# Verification\nConfirm the expected file digest.\n"
        "# Stop or adapt\nStop if the workspace identity changes."
    )
    with patch.dict(os.environ, environment, clear=True):
        refused_note = RememberSkillKnowledgeTool(worker).execute(
            title="First try",
            content="The file opened immediately.",
            related_skill_paths=["review/recovered_path"],
            learning=learning,
        )
        assert "first-try success is not eligible" in refused_note
        assert "evidence must contain" in CreateSkillTool(worker).execute(
            "review", "recovered_path", protocol, evidence=[]
        )

        worker.request_contract.results = [result("failed"), result("ok")]
        saved_receipt = RememberSkillKnowledgeTool(worker).execute(
            title="Recovered canonical path",
            content="A guessed path failed; resolving from the workspace root succeeded.",
            related_skill_paths=["review/recovered_path"],
            learning=learning,
        )
        assert "skill_evidence_eligible=true" in saved_receipt
        note = SkillsManager().knowledge_store().list_notes()[0]
        contradictory_protocol = protocol.replace(
            "1. Resolve the canonical path.\n2. Open that exact file.",
            "1. Guess a different path.\n2. Skip verification.",
        )
        contradicted = CreateSkillTool(worker).execute(
            "review",
            "recovered_path",
            contradictory_protocol,
            evidence=[{"note_id": note["id"], "revision": note["revision"]}],
        )
        created = CreateSkillTool(worker).execute(
            "review",
            "recovered_path",
            protocol,
            evidence=[{"note_id": note["id"], "revision": note["revision"]}],
        )
        duplicate_learning = {
            **learning,
            "candidate_skill_path": "review/duplicate_shortcut",
        }
        RememberSkillKnowledgeTool(worker).execute(
            title="Same recovery, second shortcut",
            content="This is another interpretation of the same recovery episode.",
            related_skill_paths=["review/duplicate_shortcut"],
            learning=duplicate_learning,
        )
        duplicate_note = next(
            item
            for item in SkillsManager().knowledge_store().list_notes()
            if item["learning"]
            and item["learning"]["candidate_skill_path"] == "review/duplicate_shortcut"
        )
        duplicate = CreateSkillTool(worker).execute(
            "review",
            "duplicate_shortcut",
            protocol,
            evidence=[
                {
                    "note_id": duplicate_note["id"],
                    "revision": duplicate_note["revision"],
                }
            ],
        )
        activated = ActivateSkillTool(worker).execute("review/recovered_path")

    assert "Created" in created
    assert "must exactly match" in contradicted
    assert "one recovery episode" in duplicate
    assert "ADVISORY" in activated


@pytest.mark.parametrize("non_failure", ["blocked", "no_change"])
def test_policy_refusal_or_noop_does_not_earn_skill_evidence(
    tmp_path: Path, non_failure: str
) -> None:
    shared = tmp_path / "shared"
    (shared / "review").mkdir(parents=True)
    (shared / "review" / "base.txt").write_text("base", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "AEON_SKILLS_DIR": str(shared),
            INSTANCE_SKILLS_DIR_ENV: str(tmp_path / "private"),
        }
    )
    environment.pop("AEON_REMOTE_INSTANCE_ID", None)
    environment.pop("AEON_CHAT_TRANSCRIPT_PATH", None)
    worker = SimpleNamespace(
        request_contract=SimpleNamespace(
            request_id="request-1",
            results=[
                SimpleNamespace(
                    tool_name="run_command",
                    status=non_failure,
                    error_code="policy_refusal" if non_failure == "blocked" else "",
                    summary=non_failure,
                ),
                SimpleNamespace(
                    tool_name="open_file",
                    status="ok",
                    error_code="",
                    summary="ordinary read succeeded",
                ),
            ],
        )
    )
    learning = {
        "candidate_skill_path": "review/not_earned",
        "procedure": "Do not infer recovery from a refusal or no-op.",
        "verification": "Require an actual failed tool result followed by success.",
        "procedure_stable": True,
        "uncertainty": "low",
    }

    with patch.dict(os.environ, environment, clear=True):
        result = RememberSkillKnowledgeTool(worker).execute(
            title="Not earned",
            content="No failed attempt was recovered in this request.",
            related_skill_paths=["review/not_earned"],
            learning=learning,
        )

    assert "first-try success is not eligible" in result
