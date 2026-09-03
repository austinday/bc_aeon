"""Hermetic tests for bounded browser downloads and CPU helper cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import socket
import tempfile
import unittest
from unittest.mock import AsyncMock, call, patch

from aeon.services.browser import browser_util
from aeon.services.browser import media_safety as safety


def _public_answers(*_args, **_kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


class _Response:
    def __init__(self, chunks, *, url="https://example.com/media", headers=None):
        self._chunks = list(chunks)
        self._url = url
        self.headers = dict(headers or {})
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self._url

    def read(self, size):
        self.read_sizes.append(size)
        return self._chunks.pop(0) if self._chunks else b""


class _Opener:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return self.response


class BrowserMediaDownloadTests(unittest.TestCase):
    def test_benchmark_fixtures_are_closed_immutable_and_network_free(self):
        self.assertEqual(
            set(browser_util.BENCHMARK_FIXTURES),
            {"observe-v1", "form-v1", "session-v1", "vision-v1"},
        )
        for fixture_id in browser_util.BENCHMARK_FIXTURES:
            definition = browser_util.benchmark_fixture_definition(fixture_id)
            self.assertIsNotNone(definition)
            html, verification = definition
            self.assertIn("Content-Security-Policy", html)
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)
            self.assertTrue(verification)
        self.assertIsNone(browser_util.benchmark_fixture_definition("../../secret"))
        session_html = browser_util.BENCHMARK_FIXTURES["session-v1"].lower()
        self.assertNotIn("document.cookie", session_html)
        self.assertIn("localstorage", session_html)
        self.assertNotIn("sessionstorage", session_html)
        self.assertNotIn("password", session_html)
        self.assertNotIn("credential", session_html)
        valid = browser_util.validate_benchmark_fixture_request(
            "oc-" + "a" * 32,
            "benchmark",
            "benchmark-" + "b" * 12,
            "form-v1",
            "seed",
        )
        self.assertIsNotNone(valid)
        self.assertIsNotNone(
            browser_util.validate_benchmark_fixture_request(
                "oc-" + "a" * 32,
                "benchmark",
                "benchmark-" + "b" * 12,
                "session-v1",
                "reopen",
            )
        )
        self.assertIsNotNone(
            browser_util.validate_benchmark_fixture_request(
                "oc-" + "a" * 32,
                "benchmark",
                "benchmark-" + "b" * 12,
                "session-v1",
                "cleanup",
            )
        )
        for request in (
            ("default", "benchmark", "benchmark-" + "b" * 12, "form-v1", "seed"),
            ("oc-" + "a" * 32, "other", "benchmark-" + "b" * 12, "form-v1", "seed"),
            ("oc-" + "a" * 32, "benchmark", "default", "form-v1", "seed"),
            ("oc-" + "a" * 32, "benchmark", "benchmark-" + "b" * 12, "unknown", "seed"),
            ("oc-" + "a" * 32, "benchmark", "benchmark-" + "b" * 12, "form-v1", "other"),
            ("oc-" + "a" * 32, "benchmark", "benchmark-" + "b" * 12, "form-v1", "reopen"),
            ("oc-" + "a" * 32, "benchmark", "benchmark-" + "b" * 12, "form-v1", "cleanup"),
        ):
            with self.subTest(request=request):
                self.assertIsNone(browser_util.validate_benchmark_fixture_request(*request))
        with self.assertRaises(TypeError):
            browser_util.BENCHMARK_FIXTURES["other"] = "caller html"

    def test_benchmark_page_helpers_use_only_catalog_content_and_predicates(self):
        class FakePage:
            def __init__(self):
                self.content_calls = []
                self.wait_calls = []
                self.evaluate_calls = []

            async def set_content(self, html, **kwargs):
                self.content_calls.append((html, kwargs))

            async def wait_for_timeout(self, milliseconds):
                self.wait_calls.append(milliseconds)

            async def evaluate(self, script):
                self.evaluate_calls.append(script)
                return True

        page = FakePage()
        self.assertTrue(
            asyncio.run(browser_util.seed_benchmark_fixture_page(page, "form-v1"))
        )
        self.assertEqual(
            page.content_calls[0][0], browser_util.BENCHMARK_FIXTURES["form-v1"]
        )
        self.assertEqual(
            page.content_calls[0][1],
            {"wait_until": "domcontentloaded", "timeout": 10_000},
        )
        self.assertTrue(
            asyncio.run(browser_util.verify_benchmark_fixture_page(page, "form-v1"))
        )
        self.assertEqual(
            page.evaluate_calls,
            [browser_util.BENCHMARK_VERIFY_SCRIPTS["form-v1"]],
        )
        untouched = FakePage()
        self.assertFalse(
            asyncio.run(browser_util.seed_benchmark_fixture_page(untouched, "bad"))
        )
        self.assertEqual(untouched.content_calls, [])

    def test_session_fixture_uses_intercepted_origin_and_survives_new_page(self):
        session_id = "oc-" + "a" * 32
        storage = {"aeon-benchmark-session-v1:" + session_id: "authenticated"}

        class FakeRoute:
            def __init__(self):
                self.fulfilled = None

            async def fulfill(self, **kwargs):
                self.fulfilled = kwargs

        class FakeSessionPage:
            def __init__(self):
                self.routes = {}
                self.fulfilled = []
                self.url = "about:blank"
                self.wait_calls = []
                self.reload_calls = []

            async def route(self, url, handler):
                self.routes[url] = handler

            async def goto(self, url, **kwargs):
                self.url = url
                route = FakeRoute()
                await self.routes[url](route)
                self.fulfilled.append(route.fulfilled)
                return kwargs

            async def evaluate(self, script):
                self.assert_reset_script = script
                storage.pop("aeon-benchmark-session-v1:" + session_id, None)

            async def reload(self, **kwargs):
                self.reload_calls.append(kwargs)

            async def wait_for_timeout(self, milliseconds):
                self.wait_calls.append(milliseconds)

        first = FakeSessionPage()
        self.assertTrue(
            asyncio.run(
                browser_util.seed_benchmark_fixture_page(
                    first,
                    "session-v1",
                    session_id=session_id,
                    reset_session=True,
                )
            )
        )
        expected_url = browser_util.benchmark_fixture_page_url(
            "session-v1", session_id
        )
        self.assertEqual(first.url, expected_url)
        self.assertEqual(set(first.routes), {expected_url})
        self.assertEqual(
            first.fulfilled[0]["body"],
            browser_util.BENCHMARK_FIXTURES["session-v1"],
        )
        self.assertNotIn("authenticated", storage.values())

        # A sign-in is only a synthetic origin-scoped marker. A fresh Page
        # loading the same immutable intercepted URL sees the same context
        # storage without any network or real credential.
        storage["aeon-benchmark-session-v1:" + session_id] = "authenticated"
        reopened = FakeSessionPage()
        self.assertTrue(
            asyncio.run(
                browser_util.seed_benchmark_fixture_page(
                    reopened,
                    "session-v1",
                    session_id=session_id,
                    reset_session=False,
                )
            )
        )
        self.assertEqual(reopened.url, expected_url)
        self.assertEqual(
            storage["aeon-benchmark-session-v1:" + session_id], "authenticated"
        )

    def test_public_url_policy_rejects_ssrf_and_redirect_features(self):
        self.assertEqual(
            safety.validate_public_http_url(
                "https://example.com/media", resolver=_public_answers
            ),
            "https://example.com/media",
        )
        blocked = (
            "http://127.0.0.1/private",
            "http://169.254.169.254/latest/meta-data",
            "http://localhost/private",
            "http://service.internal/private",
            "https://user:secret@example.com/media",
            "https://example.com/media#fragment",
            "file:///etc/passwd",
        )
        for value in blocked:
            with self.subTest(value=value), self.assertRaises(
                safety.BrowserMediaSafetyError
            ):
                safety.validate_public_http_url(value, resolver=_public_answers)

        def mixed(*_args, **_kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 443)),
            ]

        with self.assertRaisesRegex(
            safety.BrowserMediaSafetyError, "exclusively"
        ):
            safety.validate_public_http_url(
                "https://example.com/media", resolver=mixed
            )

    def test_navigation_normalizes_only_public_http_and_validates_each_hop(self):
        self.assertEqual(
            safety.normalize_public_navigation_url(
                "example.com/path#section", resolver=_public_answers
            ),
            "https://example.com/path#section",
        )
        self.assertEqual(
            safety.validate_public_browser_request_url(
                "wss://example.com/events", resolver=_public_answers
            ),
            "wss://example.com/events",
        )
        for value in (
            "file:///profiles/browser_api_token",
            "data:text/plain,secret",
            "chrome://settings",
            "ftp://example.com/file",
            "http://127.0.0.1:8765/internal",
            "http://172.19.0.1:8765/internal",
            "http://169.254.169.254/latest/meta-data",
            "https://user:password@example.com/private",
        ):
            with self.subTest(value=value), self.assertRaises(
                safety.BrowserMediaSafetyError
            ):
                safety.normalize_public_navigation_url(
                    value, resolver=_public_answers
                )

    def test_download_streams_in_fixed_chunks_to_new_private_file(self):
        response = _Response(
            [b"abc", b"def"],
            headers={"content-length": "6", "content-type": "image/png"},
        )
        opener = _Opener(response)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "capture.tmp"
            content_type, size = safety.bounded_download(
                "https://example.com/media",
                target,
                max_bytes=10,
                timeout=7,
                resolver=_public_answers,
                opener=opener,
            )
            self.assertEqual((content_type, size), ("image/png", 6))
            self.assertEqual(target.read_bytes(), b"abcdef")
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        self.assertTrue(response.read_sizes)
        self.assertEqual(set(response.read_sizes), {safety.DOWNLOAD_CHUNK_BYTES})
        self.assertEqual(opener.calls[0][1], 7.0)

    def test_download_refuses_redirect_and_oversize_without_partial(self):
        cases = (
            _Response(
                [b"body"],
                url="https://redirected.example/media",
                headers={"content-length": "4"},
            ),
            _Response([b"123456"], headers={"content-length": "6"}),
        )
        for index, response in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "capture.tmp"
                with self.assertRaises(safety.BrowserMediaSafetyError):
                    safety.bounded_download(
                        "https://example.com/media",
                        target,
                        max_bytes=5,
                        timeout=2,
                        resolver=_public_answers,
                        opener=_Opener(response),
                    )
                self.assertFalse(target.exists())

    def test_default_download_uses_the_single_validated_ip_set_and_no_proxy(self):
        response = _Response(
            [b"safe"],
            headers={"content-length": "4", "content-type": "image/png"},
        )
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            safety, "_open_pinned_response", return_value=response
        ) as pinned:
            target = Path(temporary) / "capture.tmp"
            self.assertEqual(
                safety.bounded_download(
                    "https://example.com/media",
                    target,
                    max_bytes=8,
                    timeout=3,
                    resolver=_public_answers,
                ),
                ("image/png", 4),
            )
            identity = pinned.call_args.args[0]
            self.assertEqual(identity.hostname, "example.com")
            self.assertEqual(identity.addresses, ("93.184.216.34",))

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "capture.tmp"
            with self.assertRaisesRegex(
                safety.BrowserMediaSafetyError, "do not permit an upstream proxy"
            ):
                safety.bounded_download(
                    "https://example.com/media",
                    target,
                    max_bytes=8,
                    timeout=3,
                    resolver=_public_answers,
                    proxy_url="http://proxy.example:8080",
                )
            self.assertFalse(target.exists())

    def test_strict_opener_never_inherits_proxy_or_redirects(self):
        with patch.object(safety, "build_opener", return_value=object()) as build:
            safety.strict_url_opener()
            handlers = build.call_args.args
            self.assertEqual(handlers[0].proxies, {})
            self.assertIsInstance(handlers[1], safety._NoRedirect)

            safety.strict_url_opener("http://proxy.example:8080")
            handlers = build.call_args.args
            self.assertEqual(
                handlers[0].proxies,
                {
                    "http": "http://proxy.example:8080",
                    "https": "http://proxy.example:8080",
                },
            )
        for bad in (
            "socks5://proxy.example:1080",
            "http://proxy.example:8080/path",
            "http://proxy.example:8080/?query=1",
        ):
            with self.subTest(bad=bad), self.assertRaises(
                safety.BrowserMediaSafetyError
            ):
                safety.strict_url_opener(bad)


class BrowserHelperProcessTests(unittest.TestCase):
    def test_timeout_terminates_then_kills_exact_new_session_group(self):
        class Process:
            pid = 4242
            returncode = None

            async def wait(self):
                self.returncode = -signal.SIGKILL
                return self.returncode

        process = Process()

        async def timeout_once(awaitable, *, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        with patch.object(safety.asyncio, "wait_for", side_effect=timeout_once), patch.object(
            safety.os, "killpg"
        ) as killpg:
            asyncio.run(
                safety.terminate_exact_process_group(process, grace_seconds=0.1)
            )
        self.assertEqual(
            killpg.call_args_list,
            [call(4242, signal.SIGTERM), call(4242, signal.SIGKILL)],
        )

    def test_helper_has_new_session_bounded_stderr_and_scrubbed_authority(self):
        class Stream:
            def __init__(self):
                self.parts = [b"x" * 5000, b""]

            async def read(self, _size):
                return self.parts.pop(0)

        class Process:
            pid = 5151
            returncode = None
            stderr = Stream()

            async def wait(self):
                self.returncode = 0
                return 0

        process = Process()
        create = AsyncMock(return_value=process)
        source = {
            "http_proxy": "http://untrusted.invalid:8080",
            "HTTPS_PROXY": "http://untrusted.invalid:8080",
            "AEON_FLEET_SOCKET": "/owner/fleet.sock",
            "GPU_AGENT_CLAIM_ID": "claim",
            "CUDA_VISIBLE_DEVICES": "GPU-secret",
            "KEEP_ME": "yes",
        }
        with patch.object(safety.asyncio, "create_subprocess_exec", create):
            returncode, stderr = asyncio.run(
                safety.run_cpu_helper(
                    ["/usr/bin/ffmpeg", "-version"],
                    timeout=5,
                    environment=source,
                )
            )
        self.assertEqual(returncode, 0)
        self.assertEqual(len(stderr), safety.HELPER_STDERR_TAIL_BYTES)
        args = create.call_args.args
        kwargs = create.call_args.kwargs
        self.assertEqual(
            args[:5],
            ("/usr/bin/nice", "-n", "19", "/usr/bin/ffmpeg", "-version"),
        )
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["KEEP_ME"], "yes")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "void")
        for key in ("http_proxy", "HTTPS_PROXY", "AEON_FLEET_SOCKET", "GPU_AGENT_CLAIM_ID"):
            self.assertNotIn(key, kwargs["env"])

    def test_server_has_no_whole_body_media_fetch(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "services"
            / "browser"
            / "server.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("await resp.body()", source)
        self.assertNotIn("yt-dlp", source)
        self.assertIn('"/usr/bin/ffmpeg"', source)
        self.assertIn('await ctx.route("**/*", _route_public_network)', source)
        self.assertIn('service_workers="block"', source)
        self.assertIn("normalize_public_navigation_url", source)
        self.assertIn("await _require_public_page(page)", source)
        self.assertIn("await download.cancel()", source)
        self.assertNotIn("await download.save_as", source)
        self.assertNotIn("shutil.rmtree(_profile_dir(profile)", source)
        self.assertIn("inspected++ < 2000", source)
        self.assertIn("out.length < 80", source)
        self.assertIn("for cookie in cookies[:128]", source)


if __name__ == "__main__":
    unittest.main()
