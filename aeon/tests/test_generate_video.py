from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.core.agent_protocol import ToolStatus
from aeon.tools.generate_video import GenerateVideoTool


class GenerateVideoToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "models" / "comfyui" / "unet").mkdir(parents=True)
        (self.root / "models" / "comfyui" / "text_encoders").mkdir(parents=True)
        (self.root / "models" / "comfyui" / "vae").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def _tool(self) -> GenerateVideoTool:
        with patch.dict(os.environ, {"AEON_HOME": str(self.root)}, clear=False):
            return GenerateVideoTool()

    def test_release_selection_is_exact_and_prefers_q8(self):
        unet = self.root / "models" / "comfyui" / "unet"
        (unet / "10Eros_v1.210Eros_v1.2-Q8_0.gguf").touch()
        (unet / "10Eros_v1.5-Q4_K_M.gguf").touch()
        (unet / "10Eros_v1.5-Q8_0.gguf").touch()
        tool = self._tool()
        self.assertEqual(tool._resolve_video_model(), "10Eros_v1.5-Q8_0.gguf")

    def test_missing_reviewed_release_fails_closed(self):
        unet = self.root / "models" / "comfyui" / "unet"
        (unet / "10Eros_v1.210Eros_v1.2-Q8_0.gguf").touch()
        tool = self._tool()
        with self.assertRaisesRegex(RuntimeError, "reviewed uncensored video model"):
            tool._resolve_video_model()

    def test_h3_stack_requires_exact_uncensored_components(self):
        root = self.root / "models" / "comfyui"
        names = (
            ("unet", "10Eros_Max_h3_fl2va_beta2_pruned_nvfp4.safetensors"),
            ("text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
            ("vae", "minimax_h3_video_vae_fp16.safetensors"),
            ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
        )
        for subdir, filename in names:
            (root / subdir / filename).touch()
        tool = self._tool()
        self.assertEqual(tool._resolve_h3_stack(), tuple(filename for _, filename in names))

    def test_auto_renderer_routes_general_h3_and_mature_specialist_cases_to_ltx(self):
        tool = self._tool()
        self.assertEqual(
            tool._select_renderer("text_to_video", "a fox crosses a creek", None, 124, "auto"),
            "h3",
        )
        self.assertEqual(
            tool._select_renderer("text_to_video", "explicit nude sex scene", None, 124, "auto"),
            "ltx",
        )
        self.assertEqual(
            tool._select_renderer("edit_video", "make this look like film", None, 97, "auto"),
            "ltx",
        )
        boundary = [{"image": "a.png", "at_frame": 0}, {"image": "b.png", "at_frame": 123}]
        self.assertEqual(tool._select_renderer("keyframes", "walk forward", boundary, 124, "auto"), "h3")

    def test_h3_grid_and_prompt_ir_are_deterministic(self):
        tool = self._tool()
        self.assertEqual(tool._valid_h3_len(97), 107)
        self.assertEqual(tool._valid_h3_len(124), 124)
        self.assertEqual(tool._valid_h3_len(999), 362)
        prompt = tool._structured_h3_prompt(
            "A fox crosses a creek.",
            has_first=True,
            has_last=False,
            duration=124 / 24,
            negative_prompt="watermark",
        )
        self.assertTrue(prompt.startswith("For the target video, at 0.00 seconds"))
        self.assertIn("integrated_multimodal_description: [Shot 1]", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertIn("non_diegetic_music:", prompt)
        self.assertIn("Visual exclusions: watermark", prompt)

    def test_h3_workflow_keeps_native_audio_in_the_saved_mp4(self):
        tool = self._tool()
        with patch.object(
            tool,
            "_resolve_h3_stack",
            return_value=("h3.safetensors", "heretic.safetensors", "video.safetensors", "audio.safetensors"),
        ):
            workflow = tool._build_h3_workflow(
                "integrated_multimodal_description: [Shot 1] Test\n\noverall_soundscape: Test\n\nnon_diegetic_music: N/A",
                864,
                480,
                124,
                None,
                None,
                42,
            )
        self.assertEqual(workflow["1"]["class_type"], "UNETLoader")
        self.assertEqual(workflow["2"]["inputs"]["type"], "minimax")
        self.assertEqual(workflow["8"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(workflow["9"]["inputs"]["steps"], 20)
        self.assertEqual(workflow["14"]["inputs"]["audio"], ["13", 0])
        self.assertEqual(workflow["14"]["inputs"]["format"], "video/h264-mp4")

    def test_schema_is_strict_and_exposes_assembly(self):
        tool = self._tool()
        schema = tool.parameter_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("input_videos", schema["properties"])
        self.assertIn("renderer", schema["properties"])
        self.assertIn("enhance", schema["properties"])

    def test_dimensions_preserve_orientation_within_safe_area(self):
        tool = self._tool()
        landscape = tool._dimensions(1920, 1080)
        portrait = tool._dimensions(1080, 1920)
        self.assertGreater(landscape[0], landscape[1])
        self.assertGreater(portrait[1], portrait[0])
        self.assertLessEqual(landscape[0] * landscape[1], tool.MAX_PIXELS)
        self.assertLessEqual(portrait[0] * portrait[1], tool.MAX_PIXELS)

    def test_concatenate_returns_typed_final_artifact(self):
        tool = self._tool()
        with patch.object(tool, "_assemble", return_value=12345):
            with patch("aeon.tools.generate_video.resolve_output_dir") as resolve:
                resolve.return_value = self.root / "final.mp4"
                result = tool.execute(
                    mode="concatenate",
                    input_videos=["one.mp4", "two.mp4"],
                    output_dir=".",
                )
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.artifacts, [str(self.root / "final.mp4")])
        self.assertIn("2 verified clips", result.summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
