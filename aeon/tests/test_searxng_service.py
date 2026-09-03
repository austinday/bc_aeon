"""Hermetic identity/cap checks for the operator-owned search dependency."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from aeon.scripts import searxng_service as service


def _receipt() -> dict:
    identity = "a" * 32
    return {
        "schema": service.RECEIPT_SCHEMA,
        "service_id": identity,
        "container_id": "b" * 64,
        "container_name": f"aeon-searxng-{identity}",
        "image_id": service.IMAGE_ID,
        "image_ref": service.IMAGE_REF,
    }


def _inspection(settings: Path, receipt: dict, *, running: bool = False) -> dict:
    return {
        "Id": receipt["container_id"],
        "Name": "/" + receipt["container_name"],
        "Image": service.IMAGE_ID,
        "Config": {
            "Image": service.IMAGE_REF,
            "Labels": {
                "owner": "aday",
                "com.bc_aeon.component": "searxng",
                "com.bc_aeon.service-id": receipt["service_id"],
            },
            "Env": [
                "SEARXNG_BASE_URL=http://127.0.0.1:8095/",
                "CUDA_VISIBLE_DEVICES=void",
                "GPU_DEVICE_ORDINAL=-1",
                "HIP_VISIBLE_DEVICES=-1",
                "NVIDIA_VISIBLE_DEVICES=void",
                "ROCR_VISIBLE_DEVICES=-1",
            ],
        },
        "HostConfig": {
            "PortBindings": {
                "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8095"}]
            },
            "PublishAllPorts": False,
            "Devices": [],
            "DeviceRequests": [],
            "Privileged": False,
            "CapDrop": ["ALL"],
            "CapAdd": ["CHOWN", "SETGID", "SETUID"],
            "SecurityOpt": ["no-new-privileges"],
            "ReadonlyRootfs": False,
            "NetworkMode": "bridge",
            "PidMode": "",
            "IpcMode": "private",
            "AutoRemove": False,
            "Init": False,
            "LogConfig": {
                "Type": "json-file",
                "Config": {"max-file": "2", "max-size": "10m"},
            },
            "Memory": 512 * 1024 * 1024,
            "MemorySwap": 512 * 1024 * 1024,
            "NanoCpus": 1_000_000_000,
            "PidsLimit": 256,
            "CpuShares": 2,
            "BlkioWeight": 10,
            "OomScoreAdj": 1000,
            "ShmSize": 128 * 1024 * 1024,
            "RestartPolicy": {"Name": "unless-stopped"},
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(settings),
                "Destination": "/etc/searxng/settings.yml",
                "RW": False,
            },
            {
                "Type": "volume",
                "Name": "e" * 64,
                "Driver": "local",
                "Source": "/var/lib/docker/volumes/" + "e" * 64 + "/_data",
                "Destination": "/etc/searxng",
                "RW": True,
            },
            {
                "Type": "volume",
                "Name": "f" * 64,
                "Driver": "local",
                "Source": "/var/lib/docker/volumes/" + "f" * 64 + "/_data",
                "Destination": "/var/cache/searxng",
                "RW": True,
            },
        ],
        "State": {"Running": running},
    }


class SearxngServiceTests(unittest.TestCase):
    def test_health_requires_exact_instance_identity_and_semantics(self):
        healthy_config = json.dumps({
            "instance_name": service.INSTANCE_PREFIX + "a" * 32,
            "version": "test",
            "engines": [],
        }).encode()
        with patch.object(
            service,
            "_read_local",
            side_effect=[
                (200, b"OK", "text/plain; charset=utf-8"),
                (200, healthy_config, "application/json"),
            ],
        ):
            self.assertTrue(service._healthy("a" * 32))
        wrong_config = json.dumps({
            "instance_name": "SearXNG", "version": "test", "engines": []
        }).encode()
        with patch.object(
            service,
            "_read_local",
            side_effect=[
                (200, b"OK", "text/plain"),
                (200, wrong_config, "application/json"),
            ],
        ):
            self.assertFalse(service._healthy("a" * 32))

    def test_settings_are_bound_to_receipted_instance_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.yml"
            with patch.object(service, "SETTINGS_PATH", settings):
                self.assertFalse(service._ensure_settings("a" * 32))
                body = settings.read_text(encoding="utf-8")
                self.assertIn(service.INSTANCE_PREFIX + "a" * 32, body)
                self.assertFalse(service._ensure_settings("a" * 32))
                self.assertTrue(service._ensure_settings("b" * 32))
                self.assertIn(
                    service.INSTANCE_PREFIX + "b" * 32,
                    settings.read_text(encoding="utf-8"),
                )

    def test_settings_are_container_readable_only_inside_owner_private_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "private"
            parent.mkdir(mode=0o700)
            settings = parent / "settings.yml"
            settings.write_text("server: {}\n", encoding="utf-8")
            settings.chmod(0o600)
            service._secure_settings_file(settings)
            self.assertEqual(settings.stat().st_mode & 0o777, 0o644)
            self.assertEqual(parent.stat().st_mode & 0o777, 0o700)

    def test_exact_receipt_configuration_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.yml"
            receipt = _receipt()
            with patch.object(service, "SETTINGS_PATH", settings):
                self.assertFalse(
                    service._validate_container(_inspection(settings, receipt), receipt)
                )
                self.assertTrue(
                    service._validate_container(
                        _inspection(settings, receipt, running=True), receipt
                    )
                )
                daemon_normalized = _inspection(settings, receipt)
                daemon_normalized["HostConfig"]["CapAdd"] = [
                    "CAP_CHOWN",
                    "CAP_SETGID",
                    "CAP_SETUID",
                ]
                self.assertFalse(
                    service._validate_container(daemon_normalized, receipt)
                )

    def test_device_or_private_port_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.yml"
            receipt = _receipt()
            device_drift = _inspection(settings, receipt)
            device_drift["HostConfig"]["DeviceRequests"] = [{"Count": -1}]
            port_drift = _inspection(settings, receipt)
            port_drift["HostConfig"]["PortBindings"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"
            with patch.object(service, "SETTINGS_PATH", settings):
                with self.assertRaises(service.SearxngServiceError):
                    service._validate_container(device_drift, receipt)
                with self.assertRaises(service.SearxngServiceError):
                    service._validate_container(port_drift, receipt)

    def test_extra_mount_port_capability_network_or_claim_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.yml"
            receipt = _receipt()
            cases = []
            extra_mount = _inspection(settings, receipt)
            extra_mount["Mounts"].append({
                "Source": "/home/aday",
                "Destination": "/owner-home",
                "RW": False,
            })
            cases.append(extra_mount)
            extra_port = _inspection(settings, receipt)
            extra_port["HostConfig"]["PortBindings"]["9000/tcp"] = [
                {"HostIp": "127.0.0.1", "HostPort": "9000"}
            ]
            cases.append(extra_port)
            extra_capability = _inspection(settings, receipt)
            extra_capability["HostConfig"]["CapAdd"].append("SYS_ADMIN")
            cases.append(extra_capability)
            host_network = _inspection(settings, receipt)
            host_network["HostConfig"]["NetworkMode"] = "host"
            cases.append(host_network)
            claim = _inspection(settings, receipt)
            claim["Config"]["Env"].append("GPU_AGENT_CLAIM_ID=forged")
            cases.append(claim)
            with patch.object(service, "SETTINGS_PATH", settings):
                for document in cases:
                    with self.subTest(document=document), self.assertRaises(
                        service.SearxngServiceError
                    ):
                        service._validate_container(document, receipt)

    def test_docker_client_is_fixed_local_low_priority_and_scrubbed(self):
        completed = subprocess.CompletedProcess([], 0, "[]", "")
        with patch.object(
            service,
            "require_fleet_low_priority_wrapper",
            return_value="/verified/fleet-low-priority",
        ), patch.object(
            service.subprocess, "run", return_value=completed
        ) as run, patch.dict(
            os.environ,
            {
                "DOCKER_HOST": "tcp://untrusted.invalid:2375",
                "DOCKER_CONTEXT": "remote",
                "CONTAINER_HOST": "tcp://untrusted.invalid:1234",
                "GPU_AGENT_CLAIM_ID": "forged-claim",
            },
            clear=False,
        ):
            service._docker(["container", "inspect", "b" * 64])

        command = run.call_args.args[0]
        self.assertEqual(
            command[:5],
            [
                "/verified/fleet-low-priority",
                "/usr/bin/docker",
                "--host",
                service.DOCKER_HOST,
                "container",
            ],
        )
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("DOCKER_HOST", environment)
        self.assertNotIn("DOCKER_CONTEXT", environment)
        self.assertNotIn("CONTAINER_HOST", environment)
        self.assertNotIn("GPU_AGENT_CLAIM_ID", environment)
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "void")

    def test_create_contract_pins_network_and_bounded_logs(self):
        with patch.object(service, "_docker") as docker, patch.object(
            service, "_inspect", return_value={}
        ), patch.object(
            service, "_validate_container", return_value=False
        ), patch.object(
            service, "_atomic_json"
        ), patch.object(
            service, "_secure_file"
        ):
            docker.return_value = subprocess.CompletedProcess(
                [], 0, "b" * 64, ""
            )
            service._create_receipt("a" * 32)

        arguments = docker.call_args_list[0].args[0]
        joined = " ".join(arguments)
        self.assertIn("--network bridge", joined)
        self.assertIn("--log-driver json-file", joined)
        self.assertIn("--log-opt max-size=10m", joined)
        self.assertIn("--log-opt max-file=2", joined)

    def test_helper_never_lists_or_removes_containers(self):
        source = Path(service.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"container", "list"', source)
        self.assertNotIn('"container", "rm"', source)
        self.assertNotIn('"container", "stop"', source)


if __name__ == "__main__":
    unittest.main()
