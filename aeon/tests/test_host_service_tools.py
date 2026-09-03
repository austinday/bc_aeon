"""CPU host-service tools never acquire container-controller authority."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from aeon.tools import browser
from aeon.tools.search import SearchWebTool, _searxng_loopback_url


class _LLM:
    def summarize_text(self, *, text: str, query: str) -> str:
        return f"summary:{query}:{len(text)}"


class HostServiceBoundaryTests(unittest.TestCase):
    def test_browser_capture_accepts_only_unguessable_service_filenames(self):
        accepted = (
            "capture_img_123_" + "a" * 32 + ".png",
            "capture_vid_9_" + "0" * 32 + ".mp4",
        )
        for value in accepted:
            self.assertIsNotNone(browser._BROWSER_CAPTURE_FILENAME_RE.fullmatch(value))
        for value in (
            "../browser_api_token",
            "capture_img_123_guess.png",
            "download_123_" + "a" * 32 + ".bin",
            "capture_vid_9_" + "0" * 32 + ".mp4/../../secret",
        ):
            self.assertIsNone(browser._BROWSER_CAPTURE_FILENAME_RE.fullmatch(value))

    def test_browser_service_identity_requires_exact_private_receipt(self):
        identity = "a" * 32
        document = {
            "schema": 1,
            "service_id": identity,
            "container_id": "b" * 64,
            "container_name": f"aeon-browser-{identity}",
            "image_id": "sha256:" + "c" * 64,
            "source_sha256": "d" * 64,
            "auth_version": "required-v1",
            "api_version": browser.BROWSER_API_VERSION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            receipt = root / "service.json"
            receipt.write_text(json.dumps(document), encoding="utf-8")
            receipt.chmod(0o600)
            with patch.object(browser, "BROWSER_SERVICE_RECEIPT", receipt), patch.object(
                browser, "browser_service_source_digest", return_value="d" * 64
            ):
                self.assertEqual(browser._browser_service_identity(), identity)
                receipt.chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    browser._browser_service_identity()

    def test_browser_service_identity_rejects_stale_source_receipt(self):
        identity = "a" * 32
        document = {
            "schema": 1,
            "service_id": identity,
            "container_id": "b" * 64,
            "container_name": f"aeon-browser-{identity}",
            "image_id": "sha256:" + "c" * 64,
            "source_sha256": "d" * 64,
            "auth_version": "required-v1",
            "api_version": browser.BROWSER_API_VERSION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            receipt = root / "service.json"
            receipt.write_text(json.dumps(document), encoding="utf-8")
            receipt.chmod(0o600)
            with patch.object(browser, "BROWSER_SERVICE_RECEIPT", receipt), patch.object(
                browser, "browser_service_source_digest", return_value="e" * 64
            ):
                with self.assertRaisesRegex(RuntimeError, "stale for the current source"):
                    browser._browser_service_identity()

    def test_browser_never_discovers_or_deletes_stale_output_directories(self):
        with patch.object(browser.os, "listdir") as listdir, patch.object(
            browser.shutil, "rmtree"
        ) as rmtree:
            browser._pruned_stale_output = False
            browser._prune_stale_output_dirs()
        listdir.assert_not_called()
        rmtree.assert_not_called()

    def test_browser_unavailable_fails_without_starting_or_replacing_service(self):
        with patch.object(browser, "_browser_healthy", return_value=False), patch.object(
            browser, "_prune_stale_output_dirs"
        ):
            with self.assertRaisesRegex(RuntimeError, "do not inspect, start, replace, or stop"):
                browser.ensure_browser_running()

    def test_search_requires_managed_service_without_container_fallback(self):
        tool = SearchWebTool(_LLM())
        with patch("aeon.tools.search._service_identity", return_value="a" * 32), patch(
            "aeon.tools.search._local_get", side_effect=requests.ConnectionError("offline")
        ) as get:
            result = tool.execute("fleet policy")
        self.assertIn("operator-managed CPU-only SearXNG service is unavailable", result)
        get.assert_called_once_with("/healthz", timeout=2)

    def test_search_service_url_is_exact_loopback_and_validated(self):
        with patch.dict("os.environ", {"AEON_SEARXNG_PORT": "18443"}):
            self.assertEqual(_searxng_loopback_url(), "http://127.0.0.1:18443")
        for value in ("0", "65536", "8095@remote.invalid", "  "):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"AEON_SEARXNG_PORT": value}
            ):
                with self.assertRaises(RuntimeError):
                    _searxng_loopback_url()

    def test_search_summarizes_only_after_local_health_and_query(self):
        health = Mock(
            status_code=200,
            content=b"OK",
            headers={"content-type": "text/plain; charset=utf-8"},
        )
        config = Mock(
            status_code=200,
            content=b"{}",
        )
        config.json.return_value = {
            "instance_name": "Aeon SearXNG " + "a" * 32,
            "version": "test",
            "engines": [],
        }
        search = Mock(
            status_code=200,
            text="",
            content=b"{}",
        )
        search.json.return_value = {
            "results": [
                {"url": "https://example.test", "title": "Example", "content": "Result"}
            ]
        }
        tool = SearchWebTool(_LLM())
        with patch("aeon.tools.search._service_identity", return_value="a" * 32), patch(
            "aeon.tools.search._local_get", side_effect=[health, config, search]
        ) as get:
            result = tool.execute("fleet policy")
        self.assertIn("summary:fleet policy", result)
        self.assertIn("https://example.test", result)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(get.call_args_list[2].args[0], "/search")

    def test_search_rejects_unbound_health_response(self):
        health = Mock(
            status_code=200,
            content=b"OK",
            headers={"content-type": "text/plain"},
        )
        config = Mock(status_code=200, content=b"{}")
        config.json.return_value = {
            "instance_name": "some other service",
            "version": "test",
            "engines": [],
        }
        with patch("aeon.tools.search._service_identity", return_value="a" * 32), patch(
            "aeon.tools.search._local_get", side_effect=[health, config]
        ):
            result = SearchWebTool(_LLM()).execute("fleet policy")
        self.assertIn("operator-managed CPU-only SearXNG service is unavailable", result)


if __name__ == "__main__":
    unittest.main()
