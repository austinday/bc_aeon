"""Hermetic Fleet ticket contract tests for ComfyUI-backed tools."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.core.agent_protocol import ToolStatus
from aeon.tools.generate_image import ComfyUITool, EditImageTool, GenerateImageTool
from aeon.tools.generate_video import GenerateVideoTool


TICKET = "fd-" + "a" * 32


def active_ticket(
    ticket_id: str = TICKET,
    *,
    compute_state: str = "ready",
    endpoint: str | None = "http://127.0.0.1:8188",
    profile_id: str = "aeon-comfyui",
    service_id: str | None = None,
) -> dict:
    return {
        "ticket_id": ticket_id,
        "profile_id": profile_id,
        "service_id": service_id or profile_id,
        "consumer": "aeon-tool-fixture",
        "state": "active",
        "compute_state": compute_state,
        "endpoint": endpoint,
    }


def released_ticket(
    ticket_id: str = TICKET,
    *,
    profile_id: str = "aeon-comfyui",
    service_id: str | None = None,
) -> dict:
    return {
        "ticket_id": ticket_id,
        "profile_id": profile_id,
        "service_id": service_id or profile_id,
        "consumer": "aeon-tool-fixture",
        "state": "released",
        "compute_state": "inactive",
        "endpoint": None,
    }


class FakeBroker:
    def __init__(self):
        self.acquire_response = active_ticket()
        self.status_responses = [active_ticket()]
        self.renew_response = active_ticket()
        self.release_response = released_ticket()
        self.acquisitions: list[dict] = []
        self.statuses: list[str] = []
        self.renewals: list[tuple[str, int | None]] = []
        self.releases: list[str] = []

    def client_type(self):
        broker = self

        class Client:
            def __init__(self, _socket_path=None, *, timeout):
                self.socket_path = _socket_path
                self.timeout = timeout

            def acquire_service(self, **kwargs):
                broker.acquisitions.append(dict(kwargs))
                return dict(broker.acquire_response)

            def service_status(self, ticket_id):
                broker.statuses.append(ticket_id)
                response = (
                    broker.status_responses.pop(0)
                    if len(broker.status_responses) > 1
                    else broker.status_responses[0]
                )
                return dict(response)

            def renew_service(self, ticket_id, *, ttl_seconds=None):
                broker.renewals.append((ticket_id, ttl_seconds))
                return dict(broker.renew_response)

            def release_service(self, ticket_id):
                broker.releases.append(ticket_id)
                return dict(broker.release_response)

        return Client


class ComfyFleetContractTests(unittest.TestCase):
    def setUp(self):
        self.consumer = patch.object(
            ComfyUITool, "_new_fleet_consumer", return_value="aeon-tool-fixture"
        )
        self.consumer.start()
        self.addCleanup(self.consumer.stop)

    def _fleet_patch(self, broker: FakeBroker):
        return patch(
            "aeon.tools.generate_image.FleetBrokerClient", broker.client_type()
        )

    @staticmethod
    def _healthy(tool):
        return patch.object(tool, "_check_comfyui_health", return_value=True)

    def test_process_exit_has_no_legacy_comfy_container_cleanup(self):
        source = (Path(__file__).parents[1] / "main.py").read_text(encoding="utf-8")
        cleanup_body = source.split("def cleanup_transient_tools", 1)[1].split(
            "# =============================================================================",
            1,
        )[0]
        self.assertNotIn("aeon_comfyui", cleanup_body)
        self.assertNotIn("comfyui_registry", cleanup_body)

    def test_each_acquisition_uses_a_fresh_idempotency_key_and_exact_release(self):
        broker = FakeBroker()
        tool = GenerateImageTool()
        with self._fleet_patch(broker), self._healthy(tool):
            self.assertTrue(tool._ensure_comfyui_running(24))
            self.assertEqual(
                tool._finish_comfy_session(),
                {"state": "released", "compute_state": "inactive"},
            )
            self.assertTrue(tool._ensure_comfyui_running(24))
            tool._finish_comfy_session()

        self.assertEqual(broker.releases, [TICKET, TICKET])
        self.assertEqual(len(broker.acquisitions), 2)
        first, second = broker.acquisitions
        self.assertNotEqual(first["idempotency_key"], second["idempotency_key"])
        self.assertRegex(first["idempotency_key"], r"^aeon-comfyui/[0-9a-f]{32}$")
        self.assertEqual(first["profile"], "aeon-comfyui")
        self.assertEqual(first["ttl_seconds"], 900)

    def test_real_consumer_identity_is_unique_per_media_invocation(self):
        self.consumer.stop()
        first = ComfyUITool._new_fleet_consumer()
        second = ComfyUITool._new_fleet_consumer()
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^aeon/tool/comfy/[0-9]+/[0-9a-f]{32}$")

    def test_client_uses_the_configured_owner_socket(self):
        broker = FakeBroker()
        client_type = broker.client_type()
        with patch("aeon.tools.generate_image.FleetBrokerClient", client_type), patch.dict(
            "os.environ", {"AEON_FLEET_SOCKET": "/owner/private/fleet.sock"}
        ):
            client = GenerateImageTool()._fleet_client()

        self.assertEqual(client.socket_path, "/owner/private/fleet.sock")
        self.assertEqual(client.timeout, 15)

    def test_invalid_acquisition_proof_releases_the_known_ticket(self):
        broker = FakeBroker()
        broker.acquire_response = active_ticket(compute_state="mystery", endpoint=None)
        tool = GenerateImageTool()
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "unknown ComfyUI compute state"):
                tool._ensure_comfyui_running(24)
        self.assertEqual(broker.releases, [TICKET])
        self.assertIsNone(tool._fleet_ticket_id)

    def test_unowned_acquisition_identity_is_not_released(self):
        for field, replacement in (
            ("consumer", "aeon-tool-someone-else"),
            ("profile_id", "another-service"),
            ("service_id", "another-service"),
        ):
            with self.subTest(field=field):
                broker = FakeBroker()
                broker.acquire_response = {
                    **active_ticket(),
                    field: replacement,
                }
                tool = GenerateImageTool()
                with self._fleet_patch(broker):
                    with self.assertRaisesRegex(RuntimeError, "unowned"):
                        tool._ensure_comfyui_running(24)
                self.assertEqual(broker.releases, [])
                self.assertIsNone(tool._fleet_ticket_id)

    def test_comfy_endpoint_rejects_noncanonical_or_control_bearing_text(self):
        for endpoint in (
            " http://127.0.0.1:8188",
            "http://127.0.0.1:8188\n",
            "http://127.0.0.1:8188\r",
            "http://127.0.0.1:8188\t",
            "http://127.0.0.1:8188/;admin",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(RuntimeError):
                ComfyUITool._validate_comfy_endpoint(endpoint)

        self.assertEqual(
            ComfyUITool._validate_comfy_endpoint("http://[::1]:8188/"),
            "http://[::1]:8188",
        )

    def test_status_for_non_loopback_endpoint_fails_closed_and_releases(self):
        broker = FakeBroker()
        broker.acquire_response = active_ticket(
            compute_state="waiting_for_compute", endpoint=None
        )
        broker.status_responses = [active_ticket(endpoint="http://192.168.0.177:8188")]
        tool = GenerateImageTool()
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "non-loopback"):
                tool._ensure_comfyui_running(24)
        self.assertEqual(broker.statuses, [TICKET])
        self.assertEqual(broker.releases, [TICKET])

    def test_status_for_another_ticket_fails_and_releases_only_owned_ticket(self):
        broker = FakeBroker()
        broker.acquire_response = active_ticket(
            compute_state="waiting_for_compute", endpoint=None
        )
        broker.status_responses = [active_ticket("fd-" + "b" * 32)]
        tool = GenerateImageTool()
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "different ComfyUI ticket"):
                tool._ensure_comfyui_running(24)
        self.assertEqual(broker.releases, [TICKET])

    def test_status_consumer_drift_fails_and_releases_only_owned_ticket(self):
        broker = FakeBroker()
        broker.acquire_response = active_ticket(
            compute_state="waiting_for_compute", endpoint=None
        )
        broker.status_responses = [
            {**active_ticket(), "consumer": "aeon-tool-someone-else"}
        ]
        tool = GenerateImageTool()
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "consumer identity"):
                tool._ensure_comfyui_running(24)
        self.assertEqual(broker.releases, [TICKET])

    def test_active_job_renewal_requires_same_ready_endpoint(self):
        broker = FakeBroker()
        broker.renew_response = active_ticket(endpoint="http://127.0.0.1:8288")
        tool = GenerateImageTool()
        tool._fleet_ticket_id = TICKET
        tool._fleet_consumer_id = "aeon-tool-fixture"
        tool.comfy_url = "http://127.0.0.1:8188"
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "changed the ComfyUI endpoint"):
                tool._renew_comfy_ticket(require_ready=True)
            tool._finish_comfy_session()
        self.assertEqual(broker.renewals, [(TICKET, 900)])
        self.assertEqual(broker.releases, [TICKET])

    def test_malformed_release_proof_retains_exact_ticket_for_retry(self):
        broker = FakeBroker()
        broker.release_response = {
            **released_ticket(),
            "ticket_id": "fd-" + "b" * 32,
        }
        tool = GenerateImageTool()
        tool._fleet_ticket_id = TICKET
        tool._fleet_consumer_id = "aeon-tool-fixture"
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "prove exact ComfyUI ticket release"):
                tool._finish_comfy_session()
            self.assertEqual(tool._fleet_ticket_id, TICKET)
            broker.release_response = released_ticket()
            tool._finish_comfy_session()
        self.assertEqual(broker.releases, [TICKET, TICKET])
        self.assertIsNone(tool._fleet_ticket_id)

    def test_release_consumer_drift_retains_exact_ticket(self):
        broker = FakeBroker()
        broker.release_response = {
            **released_ticket(),
            "consumer": "aeon-tool-someone-else",
        }
        tool = GenerateImageTool()
        tool._fleet_ticket_id = TICKET
        tool._fleet_consumer_id = "aeon-tool-fixture"
        with self._fleet_patch(broker):
            with self.assertRaisesRegex(RuntimeError, "exact ComfyUI ticket release"):
                tool._finish_comfy_session()
        self.assertEqual(tool._fleet_ticket_id, TICKET)
        self.assertEqual(broker.releases, [TICKET])

    def test_generate_image_success_releases_its_ticket(self):
        broker = FakeBroker()
        tool = GenerateImageTool()

        class PromptResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"prompt_id": "prompt-1"}

        with tempfile.TemporaryDirectory() as output_dir, self._fleet_patch(broker), self._healthy(tool), patch(
            "aeon.tools.generate_image.enhance_prompt", side_effect=lambda *_a, **_k: _a[1]
        ), patch.object(tool, "_flux2_dev_te", return_value=None), patch.object(
            tool,
            "_flux1_models",
            return_value=("unet.gguf", "clip.safetensors", "t5.safetensors", "vae.safetensors"),
        ), patch("aeon.tools.generate_image.requests.post", return_value=PromptResponse()), patch.object(
            tool, "_await_comfy", return_value={"images": [{"filename": "image.png"}]}
        ), patch.object(tool, "_download_comfy_output"):
            result = tool.execute("a test image", output_dir)

        self.assertIn("Successfully generated image", result)
        self.assertEqual(broker.releases, [TICKET])
        self.assertIsNone(tool._fleet_ticket_id)

    def test_edit_failure_after_acquisition_releases_its_ticket(self):
        broker = FakeBroker()
        tool = EditImageTool()
        with tempfile.TemporaryDirectory() as output_dir:
            input_path = Path(output_dir) / "input.png"
            input_path.write_bytes(b"not decoded before the mocked upload")
            with self._fleet_patch(broker), self._healthy(tool), patch(
                "aeon.tools.generate_image.enhance_prompt", side_effect=lambda *_a, **_k: _a[1]
            ), patch.object(tool, "_upload_image", side_effect=RuntimeError("upload failed")):
                result = tool.execute(str(input_path), "edit it", output_dir)

        self.assertIn("upload failed", result)
        self.assertEqual(broker.releases, [TICKET])
        self.assertIsNone(tool._fleet_ticket_id)

    def test_video_failure_after_acquisition_releases_its_ticket(self):
        broker = FakeBroker()
        broker.acquire_response = active_ticket(profile_id="aeon-video-comfyui")
        broker.status_responses = [active_ticket(profile_id="aeon-video-comfyui")]
        broker.renew_response = active_ticket(profile_id="aeon-video-comfyui")
        broker.release_response = released_ticket(profile_id="aeon-video-comfyui")
        tool = GenerateVideoTool()
        with tempfile.TemporaryDirectory() as output_dir, self._fleet_patch(broker), self._healthy(tool), patch(
            "aeon.tools.generate_video.enhance_prompt", side_effect=lambda *_a, **_k: _a[1]
        ), patch.object(tool, "_resolve_ltx_model", return_value="video.gguf"), patch.object(
            tool, "_build_ltx_workflow", side_effect=RuntimeError("workflow failed")
        ):
            result = tool.execute(
                "text_to_video", output_dir, "a test video", renderer="ltx"
            )

        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("workflow failed", result.summary)
        self.assertEqual(broker.acquisitions[0]["profile"], "aeon-video-comfyui")
        self.assertRegex(
            broker.acquisitions[0]["idempotency_key"],
            r"^aeon-video-comfyui/[0-9a-f]{32}$",
        )
        self.assertEqual(broker.releases, [TICKET])
        self.assertIsNone(tool._fleet_ticket_id)


if __name__ == "__main__":
    unittest.main()
