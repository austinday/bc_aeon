"""Hermetic checks for the side-effect-free Aeon readiness report."""

from __future__ import annotations

from types import SimpleNamespace

from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.core.qwen_capabilities import STANDARD_IMAGE_ID
from aeon import doctor


def _fake_diagnostics(monkeypatch, *, opencode_ready: bool) -> dict[str, object]:
    monkeypatch.setattr(doctor, "_regular_owned", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "DAY2RTX6000PRO")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=STANDARD_IMAGE_ID + "\n", stderr=""
        ),
    )
    capability = SimpleNamespace(
        host="192.168.0.177",
        context_tokens=114688,
        runtime_adapter="reviewed",
    )
    monkeypatch.setattr(
        doctor,
        "enabled_qwen_runtime_capabilities",
        lambda: ((capability,), "a" * 64),
    )
    monkeypatch.setattr(doctor, "select_compute_backend", lambda: ("broker", "reviewed"))
    monkeypatch.setattr(doctor, "discover_workspace_instructions", lambda _path: ())
    monkeypatch.setattr(
        doctor,
        "opencode_status",
        lambda: {
            "ready": opencode_ready,
            "version": "1.18.27",
            "reason": "pinned artifact ready" if opencode_ready else "missing",
        },
    )
    # Artifact presence is orthogonal to the harness assertions here.
    monkeypatch.setattr(doctor.Path, "is_file", lambda _path: True)
    monkeypatch.setattr(
        doctor.Path,
        "read_text",
        lambda _path, **_kwargs: '{"complete":true,"status":"validated"}',
    )
    return doctor.collect_diagnostics()


def test_doctor_requires_pinned_opencode_and_reports_logical_model(monkeypatch) -> None:
    report = _fake_diagnostics(monkeypatch, opencode_ready=True)

    check = next(item for item in report["checks"] if item["name"] == "OpenCode harness")
    assert check["required"] is True
    assert check["ok"] is True
    assert "1.18.27" in check["detail"]
    releases = next(
        item for item in report["checks"] if item["name"] == "Qwen fleet releases"
    )
    assert "DAY2RTX6000PRO (192.168.8.111)" in releases["detail"]
    assert "192.168.0." not in releases["detail"]
    assert report["primary_model"] == AEON_DEFAULT_MODEL_NAME
    assert report["ok"] is True


def test_doctor_fails_closed_when_pinned_opencode_is_unavailable(monkeypatch) -> None:
    report = _fake_diagnostics(monkeypatch, opencode_ready=False)

    check = next(item for item in report["checks"] if item["name"] == "OpenCode harness")
    assert check["ok"] is False
    assert report["ok"] is False


def test_doctor_canonicalizes_host_shorthand_in_capability_error(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_regular_owned", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "DAY2RTX6000PRO")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=STANDARD_IMAGE_ID + "\n", stderr=""
        ),
    )
    monkeypatch.setattr(
        doctor,
        "enabled_qwen_runtime_capabilities",
        lambda: (_ for _ in ()).throw(
            doctor.QwenCapabilityError(".178 candidate evidence is unavailable")
        ),
    )
    monkeypatch.setattr(doctor, "select_compute_backend", lambda: ("broker", "reviewed"))
    monkeypatch.setattr(doctor, "discover_workspace_instructions", lambda _path: ())
    monkeypatch.setattr(
        doctor,
        "opencode_status",
        lambda: {"ready": True, "version": "1.18.27", "reason": "ready"},
    )
    monkeypatch.setattr(doctor.Path, "is_file", lambda _path: True)
    monkeypatch.setattr(
        doctor.Path,
        "read_text",
        lambda _path, **_kwargs: '{"complete":true,"status":"validated"}',
    )

    report = doctor.collect_diagnostics()

    releases = next(
        item for item in report["checks"] if item["name"] == "Qwen fleet releases"
    )
    assert releases["ok"] is False
    assert releases["detail"] == (
        "DAY2XRTX5000 (192.168.8.114) candidate evidence is unavailable"
    )
    assert ".178" not in releases["detail"]
