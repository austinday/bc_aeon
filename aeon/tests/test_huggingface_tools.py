"""Hermetic regressions for public Hugging Face evidence tools."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from aeon.tools.huggingface import (
    HuggingFaceModelInfoTool,
    HuggingFaceModelSearchTool,
    HuggingFacePublicAPIError,
    HuggingFaceRepoFileTool,
    _same_origin_url,
)


class HuggingFaceToolTests(unittest.TestCase):
    def test_search_returns_unsummarized_metadata_and_explicit_absence_limit(self):
        response = [
            {
                "id": "owner/model-gguf",
                "sha": "a" * 40,
                "createdAt": "2026-08-27T00:00:00.000Z",
                "lastModified": "2026-08-28T00:00:00.000Z",
                "downloads": 123,
                "likes": 7,
                "tags": ["gguf", "base_model:owner/model"],
            }
        ]
        with patch(
            "aeon.tools.huggingface._public_json",
            return_value=(response, "https://huggingface.co/api/models?cursor=next"),
        ) as public_json:
            result = json.loads(
                HuggingFaceModelSearchTool().execute(
                    query="model", filter_tag="gguf", sort="downloads", limit=10
                )
            )

        self.assertEqual(result["results"][0]["id"], "owner/model-gguf")
        self.assertEqual(result["results"][0]["downloads"], 123)
        self.assertIn("not evidence", result["evidence_limits"][0])
        self.assertEqual(
            result["next_page_url"],
            "https://huggingface.co/api/models?cursor=next",
        )
        params = public_json.call_args.kwargs["params"]
        self.assertEqual(params["sort"], "downloads")
        self.assertEqual(params["direction"], -1)
        self.assertEqual(params["filter"], "gguf")

    def test_search_zero_results_never_claims_no_repository_exists(self):
        with patch("aeon.tools.huggingface._public_json", return_value=([], "")):
            result = json.loads(HuggingFaceModelSearchTool().execute(query="missing"))
        self.assertEqual(result["result_count"], 0)
        self.assertIn("not evidence", result["evidence_limits"][0])

    def test_info_preserves_identity_config_license_claim_and_files(self):
        metadata = {
            "id": "owner/model",
            "sha": "b" * 40,
            "createdAt": "2026-08-01T00:00:00.000Z",
            "tags": ["license:apache-2.0"],
            "config": {"model_type": "fixture", "architectures": ["FixtureModel"]},
            "cardData": {"license": "apache-2.0", "datasets": ["owner/data"]},
            "safetensors": {"total": 3_000_000_000},
            "siblings": [{"rfilename": "LICENSE"}, {"rfilename": "config.json"}],
        }
        with patch("aeon.tools.huggingface._public_json", return_value=(metadata, "")):
            result = json.loads(HuggingFaceModelInfoTool().execute("owner/model"))
        self.assertEqual(result["metadata"]["sha"], "b" * 40)
        self.assertEqual(result["config"]["model_type"], "fixture")
        self.assertEqual(result["card_data"]["license"], "apache-2.0")
        self.assertEqual(result["files"][0]["rfilename"], "LICENSE")
        self.assertIn("not a redistribution ruling", result["evidence_limits"][1])

    def test_repo_file_returns_revision_evidence_without_validating_claims(self):
        response = type(
            "Response",
            (),
            {"headers": {"x-repo-commit": "c" * 40, "etag": '"fixture"'}},
        )()
        with patch(
            "aeon.tools.huggingface._public_get",
            return_value=(b"Apache License fixture\n", response),
        ):
            result = json.loads(
                HuggingFaceRepoFileTool().execute(
                    "owner/model", "LICENSE", revision="c" * 40
                )
            )
        self.assertEqual(result["repo_commit"], "c" * 40)
        self.assertEqual(result["content"], "Apache License fixture\n")
        self.assertIn("does not independently validate", result["evidence_limits"][0])

    def test_invalid_ids_paths_and_continuations_fail_before_network(self):
        with patch("aeon.tools.huggingface._public_json") as public_json:
            self.assertTrue(HuggingFaceModelInfoTool().execute("../private").startswith("Error:"))
            self.assertTrue(
                HuggingFaceModelSearchTool().execute(
                    next_page_url="https://attacker.invalid/api/models"
                ).startswith("Error:")
            )
            public_json.assert_not_called()
        with patch("aeon.tools.huggingface._public_get") as public_get:
            self.assertTrue(
                HuggingFaceRepoFileTool().execute("owner/model", "../TOKEN").startswith("Error:")
            )
            public_get.assert_not_called()

    def test_same_origin_guard_rejects_credentials_ports_and_wrong_paths(self):
        accepted = _same_origin_url(
            "https://huggingface.co/api/models?cursor=abc",
            allowed_paths=("/api/models",),
        )
        self.assertEqual(accepted, "https://huggingface.co/api/models?cursor=abc")
        for value in (
            "http://huggingface.co/api/models",
            "https://user@huggingface.co/api/models",
            "https://huggingface.co:444/api/models",
            "https://huggingface.co/api/datasets",
            "https://huggingface.co/api/modelsevil",
            "https://huggingface.co.attacker.invalid/api/models",
        ):
            with self.subTest(value=value):
                with self.assertRaises(HuggingFacePublicAPIError):
                    _same_origin_url(value, allowed_paths=("/api/models",))


if __name__ == "__main__":
    unittest.main()
