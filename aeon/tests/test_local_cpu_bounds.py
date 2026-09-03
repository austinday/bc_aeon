"""Hermetic resource-contract tests for Aeon's in-process parsing tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image

from aeon.tools.analyzers import FileAnalyzer
from aeon.tools.composite_image import CompositeImageTool
from aeon.tools.file_io import OpenFileTool
from aeon.core.agent_protocol import ToolStatus


class _Worker:
    def __init__(self, workspace_root=None):
        self.open_files = {}
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        metadata = self.workspace_root.stat()
        self.workspace_root_identity = (int(metadata.st_dev), int(metadata.st_ino))

    def is_file_open(self, path):
        return os.path.abspath(path) in self.open_files

    def update_open_file(self, path, content):
        self.open_files[os.path.abspath(path)] = content


class CompositeResourceTests(unittest.TestCase):
    def test_normal_small_composite_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.png"
            overlay = root / "logo.png"
            Image.new("RGB", (40, 30), "white").save(base)
            Image.new("RGBA", (8, 4), (255, 0, 0, 128)).save(overlay)

            result = CompositeImageTool().execute(
                str(base), str(overlay), str(root), scale=0.25
            )

            self.assertIn("Composited", result)
            output = root / "base_composited.png"
            self.assertTrue(output.is_file())
            with Image.open(output) as rendered:
                self.assertEqual(rendered.size, (40, 30))

    def test_input_byte_limit_precedes_pillow_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.png"
            overlay = root / "overlay.png"
            base.write_bytes(b"12345")
            overlay.write_bytes(b"12")
            with patch("aeon.tools.composite_image.MAX_BASE_IMAGE_BYTES", 4), patch(
                "aeon.tools.composite_image.Image.open"
            ) as image_open:
                result = CompositeImageTool().execute(
                    str(base), str(overlay), str(root)
                )
        self.assertIn("refusing unsafe image composite", result)
        self.assertIn("byte", result)
        image_open.assert_not_called()

    def test_pixel_and_resize_limits_precede_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.png"
            overlay = root / "overlay.png"
            Image.new("RGB", (10, 10), "white").save(base)
            Image.new("RGBA", (1, 10), "red").save(overlay)
            with patch(
                "aeon.tools.composite_image.MAX_RESIZED_OVERLAY_PIXELS", 50
            ), patch.object(Image.Image, "convert") as convert:
                result = CompositeImageTool().execute(
                    str(base), str(overlay), str(root), scale=1.0
                )
        self.assertIn("resized overlay", result)
        convert.assert_not_called()


class AnalyzerResourceTests(unittest.TestCase):
    def test_non_regular_input_is_refused(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation unavailable")
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "input.txt"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular files"):
                FileAnalyzer(str(fifo))

    def test_large_code_uses_bounded_head_and_tail(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            FileAnalyzer, "MAX_FULL_CONTENT_BYTES", 8
        ), patch.object(FileAnalyzer, "MAX_TEXT_PREFIX_BYTES", 4):
            path = Path(directory) / "large.py"
            path.write_text("abcdefghij0123456789", encoding="utf-8")
            result = FileAnalyzer(str(path)).analyze()
        self.assertEqual(result["summary_type"], "bounded_text_preview")
        self.assertEqual(result["head_sample"], "abcd")
        self.assertEqual(result["tail_sample"], "6789")

    def test_json_and_notebook_byte_limits_precede_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "large.json"
            json_path.write_text('{"payload":"abcdef"}', encoding="utf-8")
            with patch.object(FileAnalyzer, "MAX_JSON_PARSE_BYTES", 4), patch(
                "aeon.tools.analyzers.handlers.json.json.loads"
            ) as loads:
                result = FileAnalyzer(str(json_path)).analyze()
            self.assertEqual(result["summary_type"], "bounded_json_preview")
            loads.assert_not_called()

            notebook = root / "large.ipynb"
            notebook.write_text(json.dumps({"cells": []}), encoding="utf-8")
            with patch.object(FileAnalyzer, "MAX_NOTEBOOK_PARSE_BYTES", 4), patch(
                "aeon.tools.analyzers.handlers.notebook.json.loads"
            ) as loads:
                result = FileAnalyzer(str(notebook)).analyze()
            self.assertEqual(result["summary_type"], "error")
            self.assertIn("Resource limit", result["error_message"])
            loads.assert_not_called()

    def test_pdf_page_limit_precedes_page_text_decode(self):
        class FakeDocument:
            page_count = 3

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def load_page(self, _page):
                raise AssertionError("page text must not be decoded")

        fake_fitz = SimpleNamespace(open=Mock(return_value=FakeDocument()))
        with tempfile.TemporaryDirectory() as directory, patch.object(
            FileAnalyzer, "MAX_PDF_PAGES", 2
        ), patch.dict(sys.modules, {"fitz": fake_fitz}):
            path = Path(directory) / "input.pdf"
            path.write_bytes(b"%PDF-small-fixture")
            result = FileAnalyzer(str(path)).analyze()
        self.assertEqual(result["summary_type"], "error")
        self.assertIn("3 pages", result["error_message"])

    def test_zip_member_limit_precedes_zipfile_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("one.txt", "1")
                archive.writestr("two.txt", "2")
            with patch.object(FileAnalyzer, "MAX_ARCHIVE_MEMBERS", 1), patch(
                "aeon.tools.analyzers.handlers.archive.zipfile.ZipFile"
            ) as zip_open:
                result = FileAnalyzer(str(path)).analyze()
        self.assertEqual(result["summary_type"], "error")
        self.assertIn("2 members", result["error_message"])
        zip_open.assert_not_called()

    def test_tar_decompression_stream_is_hard_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.txt"
            source.write_text("x" * 2048, encoding="utf-8")
            path = root / "data.tgz"
            with tarfile.open(path, "w:gz") as archive:
                archive.add(source, arcname="data.txt")
            with patch.object(FileAnalyzer, "MAX_ARCHIVE_STREAM_BYTES", 256):
                result = FileAnalyzer(str(path)).analyze()
        self.assertEqual(result["summary_type"], "error")
        self.assertIn("decompression exceeded", result["error_message"])

    def test_table_row_scan_is_bounded_and_truthful(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            FileAnalyzer, "MAX_TABULAR_SCAN_ROWS", 2
        ):
            path = Path(directory) / "rows.csv"
            path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
            result = FileAnalyzer(str(path)).analyze()
        self.assertEqual(result["summary_type"], "dataframe")
        self.assertTrue(result["row_scan_truncated"])
        self.assertEqual(result["row_count"], ">=1")

    def test_numpy_archive_reads_headers_without_numpy_load(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is optional")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.npz"
            np.savez_compressed(path, sample=np.zeros((2, 3), dtype="float32"))
            with patch.object(np, "load") as load:
                result = FileAnalyzer(str(path)).analyze()
        self.assertEqual(result["summary_type"], "numpy_archive_summary")
        self.assertEqual(result["arrays"]["sample"]["shape"], (2, 3))
        load.assert_not_called()


class OpenFileResourceTests(unittest.TestCase):
    def test_large_receipt_is_omitted_without_losing_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "aeon.tools.file_io.MAX_FILE_RECEIPT_BYTES", 4
        ):
            path = Path(directory) / "small.py"
            path.write_text("print('ok')\n", encoding="utf-8")
            worker = _Worker(directory)
            result = OpenFileTool(worker).execute(str(path))
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertIn("SHA256: omitted", result.summary)
        self.assertIn(str(path.resolve()), worker.open_files)

    def test_identity_drift_refuses_before_worker_cache_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.py"
            path.write_text("print('ok')\n", encoding="utf-8")
            worker = _Worker(directory)
            with patch.object(FileAnalyzer, "identity_is_current", return_value=False):
                result = OpenFileTool(worker).execute(str(path))
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("changed while it was being analyzed", result.summary)
        self.assertEqual(worker.open_files, {})


if __name__ == "__main__":
    unittest.main()
