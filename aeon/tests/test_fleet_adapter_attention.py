import json
import subprocess
from types import SimpleNamespace
import unittest

from aeon.core import qwen_runtime
from aeon.core.fleet_adapter import AeonQwenFleetAdapter
from aeon.core.qwen_capabilities import (
    QwenCapabilityError,
    RTX5000_178_RELEASE_CAPABILITY_KEY,
    RTX5000_RELEASE_CAPABILITY_KEY,
    RTX5000_RELEASE_CANDIDATE_KEY,
    qwen_release_candidate_capability,
    qwen_runtime_capability,
)
from aeon.core.qwen_runtime import QwenRuntimeError


TRITON_DIGEST = (
    "9d18098b598382de6483b25f2dedcc843b4c7c9998fef5df1f371c4a20f29a0f"
)
FLASHINFER_DIGEST = (
    "70a81413019d8872091e8ab1a2791ca4f6d85ef402f6a0a4c324f56fb5fbf6a0"
)


def context(profile_id: str, digest: str | None):
    identity = {} if digest is None else {"attention_backend": digest}
    return SimpleNamespace(
        profile=SimpleNamespace(
            profile_id=profile_id,
            artifact_identity=identity,
        )
    )


def runtime_context(capability, manifest: str, profile_id: str):
    return SimpleNamespace(
        profile=SimpleNamespace(
            profile_id=profile_id,
            purpose="Hermetic Aeon Qwen runtime test",
            min_physical_vram_gb=capability.min_physical_vram_gb,
            exclusive=True,
            min_host_memory_gb=96,
            min_host_commit_gb=96,
            min_disk_free_gb=32,
            min_shm_free_gb=16,
            artifact_identity={
                "image": str(capability.image_id).removeprefix("sha256:"),
                "model_manifest": capability.model_manifest_sha256,
                "model_sha256s": capability.model_sha256s_sha256,
                "runtime_capabilities": manifest,
            },
        ),
        lease=SimpleNamespace(
            claim_id="gc-hermetic-lease",
            owner="owner-hermetic-aeon",
            host=capability.host,
            physical_gpu=0,
            gpu_uuid="GPU-12345678-abcd",
            model="NVIDIA RTX PRO 5000",
            memory_total_mib=48935,
            vram_budget_gb=capability.vram_budget_gb,
            exclusive=True,
            run_dir=(
                "/home/aday/.local/state/fleet-compute/runs/"
                "fr-0123456789abcdef0123456789abcdef"
            ),
        ),
    )


def verify_fake_coordinator_lease(lease):
    inventory = [
        {
            "host": lease["host"],
            "physical_gpu": lease["physical_gpu"],
            "uuid": lease["gpu_uuid"],
            "acl": "OPEN",
            "state": "RESERVED",
            "vast_watchdog_active": True,
            "memory_total_mib": lease["memory_total_mib"],
            "host_memory_available_mib": 128 * 1024,
            "host_commit_headroom_mib": 128 * 1024,
            "host_disk_available_mib": 128 * 1024,
            "host_shm_available_mib": 128 * 1024,
            "claims": [
                {
                    "claim_id": lease["claim_id"],
                    "owner": lease["owner"],
                    "run_dir": lease["run_dir"],
                    "gpu_uuid": lease["gpu_uuid"],
                    "vram_budget_mib": lease["vram_budget_mib"],
                    "exclusive": 1,
                }
            ],
        }
    ]
    return qwen_runtime.verify_coordinator_lease(
        lease,
        coord_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(inventory), ""
        ),
    )


class FleetAttentionBackendTests(unittest.TestCase):
    def test_stable_profile_selects_its_bound_backend(self):
        for profile_id in (
            "aeon-qwen38-standard",
            "aeon-qwen38-compact-workers",
        ):
            with self.subTest(profile_id=profile_id):
                self.assertEqual(
                    AeonQwenFleetAdapter._attention_backend(
                        context(profile_id, TRITON_DIGEST)
                    ),
                    "TRITON_ATTN",
                )

    def test_backend_profile_near_misses_fail_closed(self):
        refused = (
            context("aeon-qwen38-standard", None),
            context("aeon-qwen38-standard", FLASHINFER_DIGEST),
            context("aeon-qwen38-flashinfer-canary", TRITON_DIGEST),
            context("aeon-qwen38-flashinfer-canary", "0" * 64),
            context("another-profile", FLASHINFER_DIGEST),
            context("another-profile", TRITON_DIGEST),
        )
        for candidate in refused:
            with self.subTest(profile=candidate.profile.profile_id), self.assertRaises(
                QwenRuntimeError
            ):
                AeonQwenFleetAdapter._attention_backend(candidate)

    def test_stable_plan_has_no_canary_marker(self):
        _plan, stable = AeonQwenFleetAdapter._base_plan("TRITON_ATTN")
        self.assertEqual(stable["AEON_VLLM_ATTENTION_BACKEND"], "TRITON_ATTN")
        self.assertNotIn("AEON_ATTENTION_BACKEND_CANARY", stable)

    def test_178_promoted_capability_selects_for_compact_workers(self):
        promoted, manifest = qwen_runtime_capability(
            RTX5000_178_RELEASE_CAPABILITY_KEY
        )
        production = runtime_context(
            promoted, manifest, "aeon-qwen38-compact-workers"
        )

        capability, observed_manifest = AeonQwenFleetAdapter._capability(production)

        self.assertTrue(capability.enabled)
        self.assertEqual(capability.host, "192.168.0.178")
        self.assertEqual(capability.max_num_seqs, 8)
        self.assertEqual(observed_manifest, manifest)

    def test_promoted_178_lease_is_non_gate_and_coordinator_validated(self):
        adapter = AeonQwenFleetAdapter()
        promoted, manifest = qwen_runtime_capability(
            RTX5000_178_RELEASE_CAPABILITY_KEY
        )
        production = runtime_context(
            promoted, manifest, "aeon-qwen38-compact-workers"
        )
        selected, selected_manifest = adapter._capability(production)

        lease = adapter._lease(production, selected, selected_manifest)

        self.assertIs(lease["release_gate"], False)
        self.assertIs(verify_fake_coordinator_lease(lease)["release_gate"], False)

    def test_retired_release_gate_and_crossed_bindings_fail_closed(self):
        adapter = AeonQwenFleetAdapter()
        promoted, promoted_manifest = qwen_runtime_capability(
            RTX5000_178_RELEASE_CAPABILITY_KEY
        )
        released, released_manifest = qwen_runtime_capability(
            RTX5000_RELEASE_CAPABILITY_KEY
        )
        gate = runtime_context(
            promoted,
            promoted_manifest,
            "aeon-qwen38-compact-178-release-gate",
        )
        with self.assertRaises(QwenCapabilityError):
            qwen_release_candidate_capability(RTX5000_RELEASE_CANDIDATE_KEY)
        with self.assertRaises(QwenCapabilityError):
            adapter._capability(gate)
        with self.assertRaisesRegex(QwenRuntimeError, "binding changed"):
            adapter._lease(gate, promoted, promoted_manifest)
        with self.assertRaisesRegex(QwenRuntimeError, "binding changed"):
            adapter._lease(gate, released, released_manifest)


if __name__ == "__main__":
    unittest.main()
