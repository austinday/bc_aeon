from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import tempfile
import types
import unittest
from unittest.mock import patch

from aeon.tools.command_fleet_guard import (
    FLEET_LOW_PRIORITY,
    SYSTEMD_RUN,
    FleetCommandGuardError,
    discard_prepared_sandbox_boundary,
    guard_fleet_shell_command,
    launch_sandbox_service,
    prepare_fleet_shell_boundary,
    resolve_command_cwd,
    require_fleet_low_priority_wrapper,
    scrubbed_fleet_command_environment,
    scrubbed_service_controller_environment,
    trusted_guardrail_paths,
)
from aeon.tools.jobs import RunCommandAsync, jobs_base
from aeon.tools.system import RunCommandTool


class FleetCommandGuardTests(unittest.TestCase):
    def test_ordinary_cpu_and_inspection_commands_are_admitted(self):
        commands = (
            "git diff --stat",
            "python -m pytest -q",
            "make test",
            "ray status",
            "rg -n 'nvidia-smi' docs",
            "printf '%s\\n' CUDA_VISIBLE_DEVICES=0",
            "systemctl --user status owner-worker.service",
            "systemctl --user -p ActiveState show owner-worker.service",
            "systemctl --user list-units --state=running",
            "service owner-worker status",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(guard_fleet_shell_command(command), command)

    def test_direct_gpu_inventory_control_and_coordinator_paths_are_refused(self):
        commands = (
            "nvidia-smi --query-gpu=uuid --format=csv,noheader",
            "watch -n 1 /usr/bin/nvidia-smi",
            "python -c 'import pynvml; pynvml.nvmlInit()'",
            "python /home/aday/website_hosting/gpu_coord.py status",
            "/home/aday/website_hosting/gpu_coord.py status",
            "cat /dev/nvidia0",
            "bash -lc 'python -c \"import pynvml\"'",
            "echo \"$(nvidia-smi)\"",
            "echo `nvidia-smi`",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaisesRegex(
                    FleetCommandGuardError, "FLEET COMPUTE POLICY"
                ):
                    guard_fleet_shell_command(command)

    def test_lease_environment_overrides_are_refused_in_execution_contexts(self):
        commands = (
            "CUDA_VISIBLE_DEVICES=0 python train.py",
            "export CUDA_VISIBLE_DEVICES=0",
            "env GPU_AGENT_CLAIM_ID=fake python train.py",
            "sudo GPU_MEM_LIMIT_GB=99 python train.py",
            "docker run -e NVIDIA_VISIBLE_DEVICES=all image true",
            "bash -lc 'CUDA_VISIBLE_DEVICES=0 python train.py'",
            "CUDA_VISIBLE_DEVICES=0; python train.py",
            "printf ok\nCUDA_VISIBLE_DEVICES=0",
            "GPU_DEVICE_ORDINAL=0 python train.py",
            "GPU_PLANNED_VRAM_GB=48.7 python train.py",
            "GPU_RESERVE_GB=6 python train.py",
            "GPU_LEASE_EXCLUSIVE=1 python train.py",
            "CUDA_MPS_PIPE_DIRECTORY=/tmp/mps python train.py",
            "SLURM_JOB_GPUS=0 python train.py",
            "NVIDIA_DRIVER_CAPABILITIES=all python train.py",
            "AEON_GPU_MEM_UTIL=0.9 python train.py",
            "AEON_TOOL_GPU_POLICY=exclusive python train.py",
            "AEON_BROWSER_GPU=1 python browser.py",
            "AEON_LLM_VRAM_BUDGET_GB=48 python serve.py",
            "QWEN_GPU_MEMORY_UTILIZATION=0.9 python serve.py",
            "BASH_ENV=/tmp/restore-claims bash -c true",
            "DOCKER_HOST=tcp://worker.example:2375 python task.py",
            "KUBECONFIG=/tmp/admin.conf python task.py",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_container_gpu_passthrough_is_refused(self):
        commands = (
            "docker run --gpus all image true",
            "docker run --gpus=device=0 image true",
            "docker run --device=/dev/nvidia0 image true",
            "podman run --device nvidia.com/gpu=all image true",
            "docker run --runtime=nvidia image true",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_all_container_daemon_and_runtime_clients_are_refused(self):
        commands = (
            "docker ps",
            "docker inspect renter",
            "docker logs renter",
            "docker top renter",
            "docker stop renter",
            "docker kill renter",
            "docker rm -f renter",
            "docker update --cpus 1 renter",
            "docker run --rm image true",
            "docker build -t image .",
            "docker compose config",
            "docker container exec renter true",
            "docker container start renter",
            "podman ps",
            "nerdctl run image true",
            "crictl stop renter",
            "ctr task kill renter",
            "kubectl get pods",
            "buildah bud .",
            "runc run owner-work",
            "dockerd --host unix:///tmp/docker.sock",
            "lxc-attach -n renter -- true",
            "containerd-shim-runc-v2 -namespace moby",
            "helm list",
            "apptainer instance start image.sif owner",
            "curl --unix-socket /var/run/docker.sock http://localhost/containers/json",
            "curl --unix-socket /run/user/1000/systemd/private http://localhost/",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_service_lifecycle_signal_and_configuration_actions_are_refused(self):
        commands = (
            "systemctl --user start owner.service",
            "systemctl --user stop owner.service",
            "systemctl --user kill owner.service",
            "systemctl --user restart owner.service",
            "systemctl --user disable owner.service",
            "systemctl --user enable owner.service",
            "systemctl --user mask owner.service",
            "systemctl --user set-property owner.service CPUWeight=1",
            "systemctl --user daemon-reload",
            "systemctl -H worker.example status owner.service",
            "systemctl --host=worker.example show owner.service",
            "service owner start",
            "service owner stop",
            "service owner restart",
            "service owner reload",
            "kill -9 1234",
            "pkill python",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_dynamic_executable_and_privilege_scope_bypasses_are_refused(self):
        commands = (
            "cmd=systemd-run; $cmd --user --scope python train.py",
            "$(printf systemd-run) --user --scope python train.py",
            "/usr/bin/docke? stop renter",
            "sudo python train.py",
            "pkexec python train.py",
            "unshare --mount python train.py",
            "rg --pre 'systemd-run --user --scope python train.py' needle .",
            "printf '#!/bin/sh\\nexec nice -n 0 -- \"$@\"\\n' > "
            "/home/aday/bin/fleet-low-priority",
            "mv /tmp/replacement /home/aday/bin/fleet-low-priority",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_direct_distributed_and_gpu_launch_patterns_are_refused(self):
        commands = (
            "torchrun --nproc-per-node=2 train.py",
            "python -m torch.distributed.run train.py",
            "accelerate launch train.py",
            "deepspeed train.py",
            "ray start --head",
            "ray job submit -- python train.py",
            "srun --gres=gpu:1 train",
            "mpirun python -c 'import torch; torch.cuda.init()'",
            "ollama serve",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_obvious_scope_and_container_escape_paths_are_refused(self):
        commands = (
            "systemd-run --user --scope python train.py",
            "nsenter --target 1 --mount nvidia-smi",
            "docker compose up -d",
            "docker-compose run worker",
            "podman compose start",
            "docker run --privileged image true",
            "docker run --device-cgroup-rule='c 195:* rmw' image true",
            "docker run -v /var/run/docker.sock:/var/run/docker.sock image true",
            "docker exec owner-container python train.py",
            "docker start existing-container",
            "ssh worker.example python train.py",
            "ssh worker.example < launch.sh",
            "ssh worker.example",
            "ssh -oRemoteCommand='python train.py' worker.example",
            "ssh -o RemoteCommand='python train.py' worker.example",
            "systemctl --user start owner-gpu.service",
            "systemctl restart owner-worker.service",
            "crontab schedule.txt",
            "at now + 1 minute",
            "batch",
            "busctl --user call org.freedesktop.systemd1 "
            "/org/freedesktop/systemd1 org.freedesktop.systemd1.Manager "
            "StartUnit ss demo.service replace",
            "dbus-send --session --dest=org.freedesktop.systemd1 "
            "/org/freedesktop/systemd1 "
            "org.freedesktop.systemd1.Manager.StartUnit",
            "docker run -v /:/host image true",
            "docker run --volume=/run:/host-run image true",
            "docker run --mount=type=bind,source=/proc,target=/host-proc image true",
            "docker run --pid=host image true",
            "docker run -d image server",
            "sleep 30 &",
            "nohup python server.py",
            "setsid python server.py",
            "systemctl --user enable --now owner-worker.service",
            "crontab -l -r",
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(FleetCommandGuardError):
                    guard_fleet_shell_command(command)

    def test_malformed_shell_is_refused_without_exposing_parser_details(self):
        for command in ("echo 'unterminated", "echo $(date", "echo ok |"):
            with self.subTest(command=command):
                with self.assertRaisesRegex(FleetCommandGuardError, "malformed"):
                    guard_fleet_shell_command(command)

    def test_quoted_literals_and_shell_comments_do_not_create_false_positives(self):
        commands = (
            "echo '$(nvidia-smi)'",
            "echo '`nvidia-smi`'",
            "echo safe # nvidia-smi is documentation",
            "echo safe # ' deliberately unmatched comment quote",
            "systemctl --user status owner-worker.service",
            "crontab -l",
            "crontab -u aday -l",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(guard_fleet_shell_command(command), command)

    def test_low_priority_wrapper_contract_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = Path(temporary) / "fleet-low-priority"
            wrapper.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
            wrapper.chmod(0o700)
            with patch(
                "aeon.tools.command_fleet_guard.FLEET_LOW_PRIORITY", wrapper
            ):
                self.assertEqual(require_fleet_low_priority_wrapper(), str(wrapper))

    def test_accelerator_environment_is_scrubbed_without_mutating_source(self):
        source = {
            "PATH": "/bin",
            "CUDA_VISIBLE_DEVICES": "GPU-secret",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "1",
            "GPU_AGENT_CLAIM_ID": "claim-secret",
            "GPU_MEM_LIMIT_GB": "48",
            "GPU_DEVICE_ORDINAL": "0",
            "GPU_PLANNED_VRAM_GB": "48.7",
            "GPU_RESERVE_GB": "6",
            "GPU_LEASE_EXCLUSIVE": "1",
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps",
            "SLURM_JOB_GPUS": "0",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "AEON_GPU_MEM_UTIL": "0.9",
            "AEON_TOOL_GPU_POLICY": "exclusive",
            "AEON_BROWSER_GPU": "1",
            "AEON_LLM_VRAM_BUDGET_GB": "48",
            "QWEN_GPU_MEMORY_UTILIZATION": "0.9",
            "BASH_ENV": "/tmp/restore-claims",
            "DOCKER_HOST": "tcp://worker.example:2375",
            "KUBECONFIG": "/tmp/admin.conf",
            "FLEET_LEASE_ID": "lease-secret",
            "AEON_FLEET_TICKET": "ticket-secret",
            "UNRELATED": "kept",
        }
        scrubbed = scrubbed_fleet_command_environment(source)
        self.assertEqual(source["CUDA_VISIBLE_DEVICES"], "GPU-secret")
        self.assertEqual(scrubbed["CUDA_VISIBLE_DEVICES"], "void")
        self.assertEqual(scrubbed["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(scrubbed["HIP_VISIBLE_DEVICES"], "-1")
        self.assertEqual(scrubbed["ROCR_VISIBLE_DEVICES"], "-1")
        self.assertNotIn("GPU_AGENT_CLAIM_ID", scrubbed)
        self.assertNotIn("GPU_MEM_LIMIT_GB", scrubbed)
        self.assertEqual(scrubbed["GPU_DEVICE_ORDINAL"], "-1")
        self.assertNotIn("GPU_PLANNED_VRAM_GB", scrubbed)
        self.assertNotIn("GPU_RESERVE_GB", scrubbed)
        self.assertNotIn("GPU_LEASE_EXCLUSIVE", scrubbed)
        self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", scrubbed)
        self.assertNotIn("SLURM_JOB_GPUS", scrubbed)
        self.assertNotIn("NVIDIA_DRIVER_CAPABILITIES", scrubbed)
        self.assertNotIn("AEON_GPU_MEM_UTIL", scrubbed)
        self.assertNotIn("AEON_TOOL_GPU_POLICY", scrubbed)
        self.assertNotIn("AEON_BROWSER_GPU", scrubbed)
        self.assertNotIn("AEON_LLM_VRAM_BUDGET_GB", scrubbed)
        self.assertNotIn("QWEN_GPU_MEMORY_UTILIZATION", scrubbed)
        self.assertNotIn("BASH_ENV", scrubbed)
        self.assertNotIn("DOCKER_HOST", scrubbed)
        self.assertNotIn("KUBECONFIG", scrubbed)
        self.assertNotIn("FLEET_LEASE_ID", scrubbed)
        self.assertNotIn("AEON_FLEET_TICKET", scrubbed)
        self.assertEqual(scrubbed["UNRELATED"], "kept")

    def test_transient_service_preflight_is_unique_networkless_and_gated(self):
        boundary, manager_environment = prepare_fleet_shell_boundary(
            source_environment={"PATH": "/bin", "XDG_RUNTIME_DIR": "/run/user/1000"},
            runtime_max_seconds=45,
        )
        try:
            self.assertRegex(boundary.unit_name, r"^aeon-command-[0-9a-f]{32}\.service$")
            self.assertRegex(boundary.nonce, r"^[0-9a-f]{64}$")
            argv = boundary.argv(str(Path(boundary.control_dir) / "spec.json"))
            self.assertIn("--wait", argv)
            self.assertIn("--pipe", argv)
            self.assertIn("--service-type=exec", argv)
            self.assertNotIn("--scope", argv)
            properties = set(boundary.properties())
            self.assertIn("DevicePolicy=closed", properties)
            self.assertIn("Environment=CUDA_VISIBLE_DEVICES=void", properties)
            self.assertIn("Environment=GPU_DEVICE_ORDINAL=-1", properties)
            self.assertIn("Environment=HIP_VISIBLE_DEVICES=-1", properties)
            self.assertIn("Environment=NVIDIA_VISIBLE_DEVICES=void", properties)
            self.assertIn("Environment=ROCR_VISIBLE_DEVICES=-1", properties)
            self.assertNotIn("Environment=CUDA_VISIBLE_DEVICES=-1", properties)
            self.assertIn("RestrictAddressFamilies=AF_NETLINK", properties)
            self.assertIn("SystemCallFilter=~socket socketpair", properties)
            self.assertIn("SystemCallErrorNumber=EPERM", properties)
            self.assertIn("ProtectSystem=strict", properties)
            self.assertIn("ProtectHome=read-only", properties)
            self.assertIn("NoNewPrivileges=yes", properties)
            self.assertFalse(any(value.startswith("IPAddressAllow=") for value in properties))
            self.assertFalse(any(value.startswith("IPAddressDeny=") for value in properties))
            self.assertIn(str(Path.cwd().resolve()), boundary.guardrail_paths)
            self.assertIn("/home/aday/.aeon", boundary.inaccessible_paths)
            self.assertEqual(manager_environment["XDG_RUNTIME_DIR"], "/run/user/1000")
        finally:
            discard_prepared_sandbox_boundary(boundary)

    def test_transient_service_refuses_parent_child_source_version_drift(self):
        with patch(
            "aeon.tools.command_fleet_guard._launch_source_digest",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(
                FleetCommandGuardError,
                "source changed after this Aeon process started",
            ):
                prepare_fleet_shell_boundary(
                    source_environment={
                        "PATH": "/bin",
                        "XDG_RUNTIME_DIR": "/run/user/1000",
                    },
                    runtime_max_seconds=45,
                )

    def test_controller_environment_keeps_only_bus_and_validated_parent_slice(self):
        slice_name = "aeon_subagent_" + "a" * 32 + ".slice"
        clean = scrubbed_service_controller_environment(
            {
                "PATH": "/host/bin",
                "HOME": "/home/aday",
                "XDG_RUNTIME_DIR": "/run/user/1000",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
                "AEON_CPU_SANDBOX_SLICE": slice_name,
                "GPU_AGENT_CLAIM_ID": "secret",
                "OPENAI_API_KEY": "secret",
                "PYTHONPATH": "/untrusted",
            }
        )
        self.assertEqual(clean["AEON_CPU_SANDBOX_SLICE"], slice_name)
        self.assertIn("DBUS_SESSION_BUS_ADDRESS", clean)
        self.assertNotIn("GPU_AGENT_CLAIM_ID", clean)
        self.assertNotIn("OPENAI_API_KEY", clean)
        self.assertNotIn("PYTHONPATH", clean)
        self.assertNotEqual(clean["PATH"], "/host/bin")
        with self.assertRaisesRegex(FleetCommandGuardError, "exact aeon_subagent"):
            scrubbed_service_controller_environment(
                {"AEON_CPU_SANDBOX_SLICE": "user-controlled.slice"}
            )

    def test_coordinator_modules_and_helpers_are_refused(self):
        for command in (
            "python -m aeon.core.gpu status",
            "python -m aeon.core.gpu_queue status",
            "python -c 'from aeon.core.gpu import reserve_named_lease'",
            "python -c 'from helper import detect_gpus'",
            "python -c 'import coordinator_client'",
        ):
            with self.subTest(command=command), self.assertRaises(FleetCommandGuardError):
                guard_fleet_shell_command(command)


class ShellToolFleetBoundaryTests(unittest.TestCase):
    def test_command_cwd_can_narrow_to_an_exact_descendant_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            self.assertEqual(
                resolve_command_cwd("project", session_root=root),
                project.resolve(),
            )

    def test_command_cwd_refuses_paths_outside_the_launch_workspace(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            with self.assertRaisesRegex(FleetCommandGuardError, "outside this agent"):
                resolve_command_cwd(other, session_root=temporary)

    def test_command_cwd_refuses_an_explicit_protected_subtree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "private-credentials"
            protected.mkdir()
            with (
                patch(
                    "aeon.tools.command_fleet_guard.inaccessible_sandbox_paths",
                    return_value=(str(protected),),
                ),
                self.assertRaisesRegex(FleetCommandGuardError, "protected credential"),
            ):
                resolve_command_cwd(protected, session_root=root)

    def test_run_command_passes_the_exact_selected_cwd_to_the_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            metadata = project.stat()
            expected_identity = (metadata.st_dev, metadata.st_ino)
            previous = os.getcwd()
            os.chdir(root)
            try:
                with (
                    patch("aeon.tools.system.prepare_fleet_shell_boundary") as prepare,
                ):
                    prepare.side_effect = FleetCommandGuardError(
                        "stop after boundary preflight"
                    )
                    result = RunCommandTool().execute("printf ok", cwd=str(project))
            finally:
                os.chdir(previous)
        self.assertEqual(result, "stop after boundary preflight")
        prepare.assert_called_once()
        kwargs = prepare.call_args.kwargs
        self.assertEqual(kwargs["cwd"], project.resolve())
        self.assertEqual(kwargs["session_root"], root.resolve())
        self.assertEqual(kwargs["expected_cwd_identity"], expected_identity)

    def test_boundary_rejects_a_cwd_symlink_swap_after_admission(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            admitted = resolve_command_cwd(project, session_root=root)
            metadata = admitted.stat(follow_symlinks=False)
            identity = (metadata.st_dev, metadata.st_ino)
            moved = root / "project-before-swap"
            project.rename(moved)
            project.symlink_to(Path(other), target_is_directory=True)

            with self.assertRaisesRegex(
                FleetCommandGuardError, "outside this agent|changed after admission"
            ):
                prepare_fleet_shell_boundary(
                    cwd=project,
                    session_root=root,
                    expected_cwd_identity=identity,
                    runtime_max_seconds=30,
                )

    def test_launch_rejects_a_cwd_swap_after_boundary_preparation(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            boundary, manager_environment = prepare_fleet_shell_boundary(
                cwd=project,
                session_root=root,
                runtime_max_seconds=30,
            )
            moved = root / "project-before-launch-swap"
            project.rename(moved)
            project.symlink_to(Path(other), target_is_directory=True)
            try:
                with (
                    patch("aeon.tools.command_fleet_guard.subprocess.Popen") as popen,
                    self.assertRaisesRegex(
                        FleetCommandGuardError, "working directory changed before launch"
                    ),
                ):
                    launch_sandbox_service(
                        "pwd", boundary, manager_environment
                    )
                popen.assert_not_called()
            finally:
                project.unlink()
                moved.rename(project)
                discard_prepared_sandbox_boundary(boundary)

    def test_run_command_refuses_before_starting_a_process(self):
        with (
            patch("aeon.tools.system.prepare_fleet_shell_boundary") as prepare,
            patch("aeon.tools.system.subprocess.Popen") as popen,
        ):
            result = RunCommandTool().execute("nvidia-smi")
        self.assertIn("FLEET COMPUTE POLICY", result)
        self.assertIn("No process or background-job directory was created", result)
        prepare.assert_not_called()
        popen.assert_not_called()

    def test_run_command_async_refuses_before_creating_job_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                worker = types.SimpleNamespace(instance_id="guard-test", notified_jobs=set())
                with (
                    patch("aeon.tools.jobs.prepare_fleet_shell_boundary") as prepare,
                    patch("aeon.tools.jobs.subprocess.Popen") as popen,
                ):
                    result = RunCommandAsync(worker=worker).execute(
                        "docker run --gpus all image true"
                    )
                self.assertIn("FLEET COMPUTE POLICY", result)
                prepare.assert_not_called()
                popen.assert_not_called()
                self.assertFalse(jobs_base(worker).exists())
            finally:
                os.chdir(previous)

    def test_background_controller_is_low_priority_and_gets_no_principal_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                worker = types.SimpleNamespace(instance_id="wrapper-test", notified_jobs=set())

                class FailedController:
                    def __init__(self, argv):
                        job_dir = Path(argv[-1])
                        (job_dir / "startup_error.txt").write_text("fake failure")
                        (job_dir / "status.txt").write_text("FAILED")

                    def poll(self):
                        return None

                def fake_popen(argv, **kwargs):
                    self.assertEqual(argv[0], str(FLEET_LOW_PRIORITY))
                    self.assertTrue(str(argv[1]).endswith("python3"))
                    self.assertEqual(argv[2], "-I")
                    self.assertTrue(str(argv[3]).endswith("command_service_controller.py"))
                    self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
                    self.assertNotIn("GPU_AGENT_CLAIM_ID", kwargs["env"])
                    self.assertNotIn("PYTHONPATH", kwargs["env"])
                    self.assertTrue(kwargs["start_new_session"])
                    return FailedController(argv)

                with (
                    patch.dict(
                        os.environ,
                        {
                            "OPENAI_API_KEY": "secret",
                            "GPU_AGENT_CLAIM_ID": "secret",
                            "PYTHONPATH": "/untrusted",
                        },
                    ),
                    patch("aeon.tools.jobs.subprocess.Popen", side_effect=fake_popen),
                ):
                    result = RunCommandAsync(worker=worker).execute("printf done")
                self.assertIn("startup failed", result)
            finally:
                os.chdir(previous)

    def test_async_service_unavailable_leaves_no_job_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                worker = types.SimpleNamespace(instance_id="service-failure", notified_jobs=set())
                with (
                    patch(
                        "aeon.tools.jobs.prepare_fleet_shell_boundary",
                        side_effect=FleetCommandGuardError("service unavailable"),
                    ),
                    patch("aeon.tools.jobs.subprocess.Popen") as popen,
                ):
                    result = RunCommandAsync(worker=worker).execute("printf safe")
                self.assertEqual(result, "service unavailable")
                self.assertFalse(jobs_base(worker).exists())
                popen.assert_not_called()
            finally:
                os.chdir(previous)

    @unittest.skipUnless(
        Path(f"/run/user/{os.getuid()}/bus").exists()
        and Path("/dev/net/tun").exists()
        and SYSTEMD_RUN.exists(),
        "requires this host's safe user-systemd CPU sandbox",
    )
    def test_live_service_streams_and_denies_network_device_and_credentials(self):
        tool = RunCommandTool()
        self.assertIn("COMMAND SUCCESS", tool.execute("printf live-ok", timeout=30))
        self.assertIn(
            "COMMAND FAILED",
            tool.execute(
                "python -I -c 'import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)'",
                timeout=30,
            ),
        )
        self.assertIn("COMMAND FAILED", tool.execute("head -c 0 /dev/net/tun", timeout=30))
        self.assertIn(
            "COMMAND FAILED",
            tool.execute("head -c 0 /home/aday/.aeon/browser_api_token", timeout=30),
        )

    @unittest.skipUnless(
        Path(f"/run/user/{os.getuid()}/bus").exists()
        and Path("/dev/net/tun").exists()
        and SYSTEMD_RUN.exists(),
        "requires this host's safe user-systemd CPU sandbox",
    )
    def test_opaque_script_cannot_rewrite_any_fixed_guardrail(self):
        regular = [Path(path) for path in trusted_guardrail_paths() if Path(path).is_file()]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in regular}
        with tempfile.TemporaryDirectory(prefix="aeon-opaque-guard-") as temporary:
            script = Path(temporary) / "opaque.sh"
            script.write_text(
                "#!/usr/bin/bash\nset +e\n"
                + "\n".join(
                    f"printf x >> {shlex.quote(str(path))} 2>/dev/null || :"
                    for path in regular
                )
                + "\n",
                encoding="utf-8",
            )
            script.chmod(0o700)
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                result = RunCommandTool().execute("/usr/bin/bash opaque.sh", timeout=30)
            finally:
                os.chdir(previous)
        self.assertIn("COMMAND SUCCESS", result)
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in regular}
        self.assertEqual(after, before)

    @unittest.skipUnless(
        Path(f"/run/user/{os.getuid()}/bus").exists()
        and Path("/dev/net/tun").exists()
        and SYSTEMD_RUN.exists(),
        "requires this host's safe user-systemd CPU sandbox",
    )
    def test_nested_protected_fixture_is_immutable_without_blocking_external_cwd(self):
        with tempfile.TemporaryDirectory(prefix="aeon-landlock-fixture-") as temporary:
            root = Path(temporary)
            protected = root / "protected.txt"
            protected.write_text("original", encoding="utf-8")
            previous = os.getcwd()
            os.chdir(root)
            try:
                with patch(
                    "aeon.tools.command_fleet_guard.trusted_guardrail_paths",
                    return_value=(str(protected),),
                ):
                    denied = RunCommandTool().execute(
                        "printf changed > protected.txt", timeout=30
                    )
                allowed = RunCommandTool().execute(
                    "printf ordinary > ordinary.txt; cat ordinary.txt", timeout=30
                )
            finally:
                os.chdir(previous)
            self.assertIn("COMMAND FAILED", denied)
            self.assertEqual(protected.read_text(encoding="utf-8"), "original")
            self.assertIn("COMMAND SUCCESS", allowed)
            self.assertEqual((root / "ordinary.txt").read_text(), "ordinary")


if __name__ == "__main__":
    unittest.main()
