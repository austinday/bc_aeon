"""Hermetic ownership and resource checks for the browser host service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from aeon.scripts import browser_service as service


def _release() -> dict:
    return {
        "schema": service.IMAGE_RECEIPT_SCHEMA,
        "image_id": "sha256:" + "b" * 64,
        "source_sha256": "c" * 64,
        "auth_version": service.AUTH_VERSION,
        "api_version": service.API_VERSION,
    }


def _receipt() -> dict:
    identity = "a" * 32
    return {
        "schema": service.SERVICE_RECEIPT_SCHEMA,
        "service_id": identity,
        "container_id": "d" * 64,
        "container_name": f"aeon-browser-{identity}",
        **{key: value for key, value in _release().items() if key != "schema"},
    }


def _inspection(profile: Path, token: Path, receipt: dict, *, running: bool = False) -> dict:
    environment = [
        "PORT=8030",
        "AEON_BROWSER_PROFILE=/profiles/default",
        "AEON_BROWSER_PROFILE_ROOT=/profiles",
        f"AEON_BROWSER_TOKEN_FILE={service.TOKEN_CONTAINER_PATH}",
        f"AEON_BROWSER_SERVICE_ID={receipt['service_id']}",
        "HOME=/profiles/.browser-home",
        "XDG_CACHE_HOME=/profiles/.browser-home/.cache",
        f"XDG_RUNTIME_DIR={service.XDG_RUNTIME_CONTAINER_PATH}",
        "PYTHONDONTWRITEBYTECODE=1",
    ] + [f"{name}={value}" for name, value in service._NO_ACCELERATOR_ENV.items()]
    return {
        "Id": receipt["container_id"],
        "Name": "/" + receipt["container_name"],
        "Image": receipt["image_id"],
        "Config": {
            "Image": receipt["image_id"],
            "User": f"{os.geteuid()}:{os.getegid()}",
            "Labels": {
                "owner": service._OWNER.pw_name,
                "com.bc_aeon.component": "browser",
                "com.bc_aeon.service-id": receipt["service_id"],
                "com.bc_aeon.browser.auth": service.AUTH_VERSION,
                "com.bc_aeon.browser.api": service.API_VERSION.replace("_", "-"),
                "com.bc_aeon.browser.source-sha256": receipt["source_sha256"],
            },
            "Env": environment,
        },
        "HostConfig": {
            "PortBindings": {
                "8030/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8030"}]
            },
            "PublishAllPorts": False,
            "Memory": service.MEMORY_BYTES,
            "MemorySwap": service.MEMORY_BYTES,
            "NanoCpus": service.NANO_CPUS,
            "PidsLimit": service.PIDS_LIMIT,
            "ShmSize": service.SHM_BYTES,
            "CpuShares": service.CPU_SHARES,
            "BlkioWeight": service.BLKIO_WEIGHT,
            "OomScoreAdj": service.OOM_SCORE_ADJ,
            "Privileged": False,
            "Devices": [],
            "DeviceRequests": [],
            "CapDrop": ["ALL"],
            "CapAdd": [],
            "SecurityOpt": ["no-new-privileges"],
            "ReadonlyRootfs": True,
            "NetworkMode": "bridge",
            "PidMode": "",
            "IpcMode": "private",
            "AutoRemove": False,
            "Init": True,
            "RestartPolicy": {"Name": "unless-stopped"},
            "LogConfig": {
                "Type": "json-file",
                "Config": {"max-file": "2", "max-size": "10m"},
            },
            "Tmpfs": {
                "/run": "rw,nosuid,nodev,noexec,size=67108864,mode=755",
                "/tmp": "rw,nosuid,nodev,size=1073741824,mode=1777",
            },
        },
        "Mounts": [
            {"Source": str(profile), "Destination": "/profiles", "RW": True},
            {
                "Source": str(token),
                "Destination": service.TOKEN_CONTAINER_PATH,
                "RW": False,
            },
        ],
        "State": {"Running": running},
    }


def _legacy_cuda_inspection(
    profile: Path, token: Path, receipt: dict, *, running: bool = False
) -> dict:
    document = _inspection(profile, token, receipt, running=running)
    environment = document["Config"]["Env"]
    environment[environment.index("CUDA_VISIBLE_DEVICES=void")] = (
        "CUDA_VISIBLE_DEVICES=-1"
    )
    return document


class _Response:
    def __init__(self, body: dict):
        self.status = 200
        self.headers = {"content-type": "application/json; charset=utf-8"}
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


class BrowserServiceTests(unittest.TestCase):
    def test_source_digest_is_deterministic_and_content_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one").write_text("first", encoding="utf-8")
            (root / "two").write_text("second", encoding="utf-8")
            with patch.object(service, "SOURCE_ROOT", root), patch.object(
                service, "SOURCE_FILES", ("one", "two")
            ):
                first = service.source_digest()
                self.assertEqual(first, service.source_digest())
                (root / "two").write_text("changed", encoding="utf-8")
                self.assertNotEqual(first, service.source_digest())

    def test_image_requires_exact_id_and_source_labels(self):
        release = _release()
        image = {
            "Id": release["image_id"],
            "Config": {
                "Labels": {
                    "com.bc_aeon.browser.auth": service.AUTH_VERSION,
                    "com.bc_aeon.browser.api": "human-v6",
                    "com.bc_aeon.browser.source-sha256": release["source_sha256"],
                }
            },
        }
        service._validate_image(
            image,
            image_id=release["image_id"],
            source_sha256=release["source_sha256"],
        )
        image["Config"]["Labels"]["com.bc_aeon.browser.source-sha256"] = "0" * 64
        with self.assertRaises(service.BrowserServiceError):
            service._validate_image(
                image,
                image_id=release["image_id"],
                source_sha256=release["source_sha256"],
            )

    def test_exact_receipted_cpu_container_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profiles"
            token = Path(temporary) / "token"
            receipt = _receipt()
            with patch.object(service, "PROFILE_ROOT", profile), patch.object(
                service, "TOKEN_PATH", token
            ):
                self.assertFalse(
                    service._validate_container(
                        _inspection(profile, token, receipt), receipt
                    )
                )
                self.assertTrue(
                    service._validate_container(
                        _inspection(profile, token, receipt, running=True), receipt
                    )
                )
                reordered = _inspection(profile, token, receipt)
                reordered["HostConfig"]["Tmpfs"]["/tmp"] = (
                    "mode=1777,size=1073741824,nodev,nosuid,rw"
                )
                self.assertFalse(service._validate_container(reordered, receipt))

    def test_device_port_and_resource_drift_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profiles"
            token = Path(temporary) / "token"
            receipt = _receipt()
            cases = []
            device = _inspection(profile, token, receipt)
            device["HostConfig"]["DeviceRequests"] = [{"Count": -1}]
            cases.append(device)
            port = _inspection(profile, token, receipt)
            port["HostConfig"]["PortBindings"]["8030/tcp"][0]["HostIp"] = "0.0.0.0"
            cases.append(port)
            memory = _inspection(profile, token, receipt)
            memory["HostConfig"]["Memory"] = 0
            cases.append(memory)
            proxy = _inspection(profile, token, receipt)
            proxy["Config"]["Env"].append(
                "HTTPS_PROXY=http://unreviewed-proxy.invalid:8080"
            )
            cases.append(proxy)
            with patch.object(service, "PROFILE_ROOT", profile), patch.object(
                service, "TOKEN_PATH", token
            ):
                for document in cases:
                    with self.subTest(document=document), self.assertRaises(
                        service.BrowserServiceError
                    ):
                        service._validate_container(document, receipt)

    def test_authenticated_health_is_bound_to_exact_service_identity(self):
        identity = "a" * 32
        good = _Response(
            {
                "status": "ok",
                "auth_required": True,
                "api_version": service.API_VERSION,
                "service_id": identity,
            }
        )
        opener = Mock()
        opener.open.return_value = good
        with patch.object(service, "build_opener", return_value=opener):
            self.assertTrue(service._healthy(identity, "secret-token" * 4))
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8030/health")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + "secret-token" * 4)

        wrong = _Response(
            {
                "status": "ok",
                "auth_required": True,
                "api_version": service.API_VERSION,
                "service_id": "b" * 32,
            }
        )
        opener.open.return_value = wrong
        with patch.object(service, "build_opener", return_value=opener):
            self.assertFalse(service._healthy(identity, "secret-token" * 4))

        server_source = (service.SOURCE_ROOT / "server.py").read_text(encoding="utf-8")
        self.assertIn('"service_id": BROWSER_SERVICE_ID', server_source)
        self.assertIn("AEON_BROWSER_SERVICE_ID", server_source)

    def test_benchmark_fixture_internal_errors_are_503_not_behavioral_misses(self):
        server_source = (service.SOURCE_ROOT / "server.py").read_text(encoding="utf-8")
        endpoint_source = server_source.split(
            '@app.post("/benchmark_fixture")', 1
        )[1].split('@app.post("/navigate")', 1)[0]

        self.assertIn("status_code=503", server_source)
        self.assertIn("benchmark_fixture_internal_failure", server_source)
        # State mismatches remain reachable 200/false model outcomes, while all
        # browser-engine/setup exception paths use the infrastructure response.
        self.assertIn('"passed": False', endpoint_source)
        self.assertGreaterEqual(
            endpoint_source.count("return _benchmark_fixture_internal_response(req)"),
            7,
        )
        self.assertNotIn("except PWError:\n            passed = False", endpoint_source)

    def test_docker_client_is_fixed_local_and_low_priority(self):
        completed = subprocess.CompletedProcess([], 0, "[]", "")
        with patch.object(
            service, "require_fleet_low_priority_wrapper", return_value="/verified/low"
        ), patch.object(service.subprocess, "run", return_value=completed) as run, patch.dict(
            os.environ,
            {
                "DOCKER_HOST": "tcp://untrusted.invalid:2375",
                "DOCKER_CONTEXT": "remote",
                "GPU_AGENT_CLAIM_ID": "secret",
            },
        ):
            service._docker(["container", "inspect", "d" * 64])
        command = run.call_args.args[0]
        self.assertEqual(
            command[:5],
            [
                "/verified/low",
                "/usr/bin/docker",
                "--host",
                service.DOCKER_HOST,
                "container",
            ],
        )
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("DOCKER_HOST", environment)
        self.assertNotIn("DOCKER_CONTEXT", environment)
        self.assertNotIn("GPU_AGENT_CLAIM_ID", environment)
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "void")

    def test_create_is_random_receipted_and_strictly_bounded(self):
        intent = {
            "schema": service.INTENT_SCHEMA,
            "service_id": "a" * 32,
            "container_name": "aeon-browser-" + "a" * 32,
            **{key: value for key, value in _release().items() if key != "schema"},
        }
        arguments = service._create_arguments(intent)
        joined = " ".join(arguments)
        self.assertIn("--cidfile", arguments)
        self.assertIn("127.0.0.1:8030:8030", arguments)
        self.assertIn("--read-only", arguments)
        self.assertIn("no-new-privileges", arguments)
        self.assertIn("--cap-drop ALL", joined)
        self.assertIn("--cpu-shares 2", joined)
        self.assertIn("--blkio-weight 10", joined)
        self.assertIn("--oom-score-adj 1000", joined)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", arguments)
        self.assertNotIn("--gpus", arguments)
        self.assertEqual(arguments[-1], intent["image_id"])

    def test_helper_has_no_discovery_or_unbounded_destructive_container_path(self):
        source = Path(service.__file__).read_text(encoding="utf-8")
        wrapper = (
            Path(service.__file__).with_name("start_browser.sh").read_text(encoding="utf-8")
        )
        for forbidden in (
            '["container", "list"',
            '["container", "ls"',
            '["container", "rm"',
            '["container", "kill"',
            '["container", "logs"',
            "docker ps",
            "docker inspect",
            "docker rm",
            "docker run",
        ):
            self.assertNotIn(forbidden, source + "\n" + wrapper)
        self.assertEqual(source.count('["container", "stop"'), 2)
        self.assertIn(
            '["container", "stop", "--time", "30", receipt["container_id"]]',
            source,
        )
        self.assertNotIn("/usr/bin/docker", wrapper)
        self.assertNotIn(" docker ", wrapper.lower())

    def test_legacy_cuda_migration_stops_only_receipted_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profiles"
            token = root / "token"
            receipt = _receipt()
            replacement = {**receipt, "service_id": "e" * 32, "container_id": "f" * 64}
            replacement["container_name"] = "aeon-browser-" + replacement["service_id"]
            running = _legacy_cuda_inspection(
                profile, token, receipt, running=True
            )
            stopped = _legacy_cuda_inspection(profile, token, receipt)
            completed = subprocess.CompletedProcess([], 0, "", "")
            with patch.object(service, "LOCK_PATH", root / "launch.lock"), patch.object(
                service, "CREATE_INTENT_PATH", root / "create-intent.json"
            ), patch.object(
                service, "PENDING_CID_PATH", root / "pending.cid"
            ), patch.object(
                service, "PROFILE_ROOT", profile
            ), patch.object(
                service, "TOKEN_PATH", token
            ), patch.object(
                service, "_prepare_service_state", return_value="secret"
            ), patch.object(
                service, "_secure_directory"
            ), patch.object(
                service, "_load_current_image_release", return_value=_release()
            ), patch.object(
                service, "_load_service_receipt", return_value=receipt
            ), patch.object(
                service, "_inspect_container", side_effect=[running, stopped]
            ), patch.object(
                service, "_docker", return_value=completed
            ) as docker, patch.object(
                service, "_retire_stopped_service_locked", return_value=receipt
            ) as retire, patch.object(
                service, "_ensure_service_locked", return_value=replacement
            ) as ensure:
                retired, created = service.migrate_legacy_cuda_sentinel()
            self.assertEqual(retired, receipt)
            self.assertEqual(created, replacement)
            docker.assert_called_once_with(
                ["container", "stop", "--time", "30", receipt["container_id"]],
                timeout=60,
            )
            retire.assert_called_once_with(require_legacy_cuda=True)
            ensure.assert_called_once_with("secret")

    def test_legacy_cuda_migration_refuses_current_or_drifted_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profiles"
            token = root / "token"
            receipt = _receipt()
            current = _inspection(profile, token, receipt, running=True)
            with patch.object(service, "LOCK_PATH", root / "launch.lock"), patch.object(
                service, "CREATE_INTENT_PATH", root / "create-intent.json"
            ), patch.object(
                service, "PENDING_CID_PATH", root / "pending.cid"
            ), patch.object(
                service, "PROFILE_ROOT", profile
            ), patch.object(
                service, "TOKEN_PATH", token
            ), patch.object(
                service, "_prepare_service_state", return_value="secret"
            ), patch.object(
                service, "_secure_directory"
            ), patch.object(
                service, "_load_current_image_release", return_value=_release()
            ), patch.object(
                service, "_load_service_receipt", return_value=receipt
            ), patch.object(
                service, "_inspect_container", return_value=current
            ), patch.object(service, "_docker") as docker:
                with self.assertRaises(service.BrowserServiceError):
                    service.migrate_legacy_cuda_sentinel()
            docker.assert_not_called()

    def test_current_image_replacement_stops_and_retires_only_receipted_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profiles"
            token = root / "token"
            receipt = _receipt()
            release = {**_release(), "image_id": "sha256:" + "e" * 64}
            replacement = {
                **receipt,
                "service_id": "f" * 32,
                "container_id": "1" * 64,
                "container_name": "aeon-browser-" + "f" * 32,
                "image_id": release["image_id"],
            }
            running = _inspection(profile, token, receipt, running=True)
            stopped = _inspection(profile, token, receipt)
            completed = subprocess.CompletedProcess([], 0, "", "")
            with patch.object(service, "LOCK_PATH", root / "launch.lock"), patch.object(
                service, "CREATE_INTENT_PATH", root / "create-intent.json"
            ), patch.object(
                service, "PENDING_CID_PATH", root / "pending.cid"
            ), patch.object(
                service, "PROFILE_ROOT", profile
            ), patch.object(
                service, "TOKEN_PATH", token
            ), patch.object(
                service, "_prepare_service_state", return_value="secret"
            ), patch.object(
                service, "_secure_directory"
            ), patch.object(
                service, "_load_current_image_release", return_value=release
            ), patch.object(
                service, "_load_service_receipt", return_value=receipt
            ), patch.object(
                service, "_inspect_container", side_effect=[running, stopped]
            ), patch.object(
                service, "_docker", return_value=completed
            ) as docker, patch.object(
                service, "_validate_retirement_identity"
            ) as validate_stopped, patch.object(
                service, "_retire_stopped_service_locked", return_value=receipt
            ) as retire, patch.object(
                service, "_ensure_service_locked", return_value=replacement
            ) as ensure:
                retired, created = service.replace_current_service()
            self.assertEqual(retired, receipt)
            self.assertEqual(created, replacement)
            docker.assert_called_once_with(
                ["container", "stop", "--time", "30", receipt["container_id"]],
                timeout=60,
            )
            validate_stopped.assert_called_once_with(stopped, receipt)
            retire.assert_called_once_with()
            ensure.assert_called_once_with("secret")

    def test_current_image_replacement_refuses_an_already_current_service(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = _receipt()
            with patch.object(service, "LOCK_PATH", root / "launch.lock"), patch.object(
                service, "CREATE_INTENT_PATH", root / "create-intent.json"
            ), patch.object(
                service, "PENDING_CID_PATH", root / "pending.cid"
            ), patch.object(
                service, "_prepare_service_state", return_value="secret"
            ), patch.object(
                service, "_secure_directory"
            ), patch.object(
                service, "_load_current_image_release", return_value=_release()
            ), patch.object(
                service, "_load_service_receipt", return_value=receipt
            ), patch.object(service, "_docker") as docker:
                with self.assertRaises(service.BrowserServiceError):
                    service.replace_current_service()
            docker.assert_not_called()

    def test_stopped_exact_service_receipt_is_retired_without_container_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            retired = state / "retired-services"
            profile = root / "profiles"
            token = root / "token"
            for directory in (state, profile):
                directory.mkdir(mode=0o700)
            token.write_text("x" * 48, encoding="utf-8")
            token.chmod(0o600)
            receipt = _receipt()
            current = state / "service.json"
            lock = state / "launch.lock"
            intent = state / "create-intent.json"
            pending = state / "pending.cid"
            with patch.object(service, "STATE_ROOT", state), patch.object(
                service, "RETIRED_SERVICE_ROOT", retired
            ), patch.object(service, "SERVICE_RECEIPT_PATH", current), patch.object(
                service, "LOCK_PATH", lock
            ), patch.object(service, "CREATE_INTENT_PATH", intent), patch.object(
                service, "PENDING_CID_PATH", pending
            ), patch.object(service, "PROFILE_ROOT", profile), patch.object(
                service, "TOKEN_PATH", token
            ), patch.object(
                service,
                "_inspect_container",
                return_value=_inspection(profile, token, receipt),
            ):
                service._atomic_json(current, receipt)
                result = service.retire_stopped_service()
            self.assertEqual(result, receipt)
            self.assertFalse(current.exists())
            archived = retired / f"{receipt['service_id']}.json"
            self.assertEqual(json.loads(archived.read_text(encoding="utf-8")), receipt)


if __name__ == "__main__":
    unittest.main()
