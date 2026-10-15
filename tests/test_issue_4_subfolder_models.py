import os
import sys
import tempfile
import unittest
import ast
from unittest.mock import MagicMock, patch

# Ensure ComfyUI and server environment mocks are in place for standalone testing
for mod_name in ["comfy_api", "comfy_api.latest", "folder_paths", "server", "aiohttp", "aiohttp.web"]:
    sys.modules.setdefault(mod_name, MagicMock())

# Add the parent directory containing the repo directory so package imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(repo_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import ComfyUI_HuggingFace_Downloader.model_discovery as md
import ComfyUI_HuggingFace_Downloader.web_api as web_api


class TestIssue4SubfolderModels(unittest.TestCase):

    def test_ast_list_comprehension_extraction(self):
        """Verify AST extractor handles list comprehensions from custom nodes like comfyui-impact-subpack."""
        code = """
def INPUT_TYPES(s):
    bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
    segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
    return {
        "required": {
            "model_name": (bboxs + segms, ),
        }
    }
"""
        tree = ast.parse(code)
        fn = tree.body[0]
        schema = md._extract_widget_schema_from_input_types_ast(fn)
        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["name"], "model_name")
        self.assertIn("ultralytics_bbox", schema[0]["folders"])
        self.assertIn("ultralytics_segm", schema[0]["folders"])

    def test_ast_set_and_generator_comprehensions(self):
        """Verify set comprehension and generator expression extraction."""
        code = """
def INPUT_TYPES(s):
    lora_gen = tuple(x for x in folder_paths.get_filename_list("loras"))
    return {
        "required": {
            "lora_name": (lora_gen,),
        }
    }
"""
        tree = ast.parse(code)
        fn = tree.body[0]
        schema = md._extract_widget_schema_from_input_types_ast(fn)
        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["name"], "lora_name")
        self.assertIn("loras", schema[0]["folders"])

    def test_choose_best_folder_candidate_with_subfolder_prefix(self):
        """Verify candidate choice matches subfolder prefix (bbox/ -> ultralytics_bbox)."""
        candidates = ["ultralytics_bbox", "ultralytics_segm"]
        chosen = md._choose_best_folder_candidate(
            candidates,
            widget_name="model_name",
            filename="bbox/face_yolov8m.pt"
        )
        self.assertEqual(chosen, "ultralytics_bbox")

        chosen_segm = md._choose_best_folder_candidate(
            candidates,
            widget_name="model_name",
            filename="segm/person_yolov8m-seg.pt"
        )
        self.assertEqual(chosen_segm, "ultralytics_segm")

    def test_check_model_files_with_registered_alias_key(self):
        """Verify check_model_files resolves 'ultralytics/bbox' when folder_paths has 'ultralytics_bbox'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bbox_dir = os.path.join(tmpdir, "models", "ultralytics", "bbox")
            os.makedirs(bbox_dir, exist_ok=True)
            model_file = os.path.join(bbox_dir, "face_yolov8m.pt")
            with open(model_file, "w") as f:
                f.write("weights")

            mock_folder_paths = MagicMock()
            mock_folder_paths.base_path = tmpdir
            mock_folder_paths.models_dir = os.path.join(tmpdir, "models")

            def get_paths(key):
                if key == "ultralytics_bbox":
                    return [bbox_dir]
                raise KeyError(key)

            mock_folder_paths.get_folder_paths.side_effect = get_paths

            with patch.object(md, "folder_paths", mock_folder_paths):
                found_models = [{
                    "filename": "face_yolov8m.pt",
                    "requested_path": "bbox/face_yolov8m.pt",
                    "suggested_folder": "ultralytics/bbox",
                }]
                missing, existing, path_mismatches = md.check_model_files(found_models)

                self.assertEqual(len(missing), 0, "Model should not be reported as missing")
                self.assertEqual(len(existing), 1, "Model should be reported as existing")
                self.assertEqual(len(path_mismatches), 0, "Subfolder model should not be false positive mismatch")
                self.assertEqual(os.path.realpath(existing[0]["found_path"]), os.path.realpath(model_file))
                self.assertEqual(existing[0]["clean_path"], "bbox/face_yolov8m.pt")

    def test_check_model_files_fallback_under_models_dir(self):
        """Verify fallback when folder_paths raises KeyError places search paths under models/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bbox_dir = os.path.join(tmpdir, "models", "ultralytics", "bbox")
            os.makedirs(bbox_dir, exist_ok=True)
            model_file = os.path.join(bbox_dir, "face_yolov8m.pt")
            with open(model_file, "w") as f:
                f.write("weights")

            mock_folder_paths = MagicMock()
            mock_folder_paths.base_path = tmpdir
            mock_folder_paths.models_dir = os.path.join(tmpdir, "models")
            mock_folder_paths.get_folder_paths.side_effect = KeyError("not found")

            with patch.object(md, "folder_paths", mock_folder_paths):
                found_models = [{
                    "filename": "face_yolov8m.pt",
                    "requested_path": "bbox/face_yolov8m.pt",
                    "suggested_folder": "ultralytics/bbox",
                }]
                missing, existing, path_mismatches = md.check_model_files(found_models)

                self.assertEqual(len(missing), 0, "Model in models/ultralytics/bbox should be found by fallback")
                self.assertEqual(len(existing), 1)
                self.assertEqual(len(path_mismatches), 0)
                self.assertEqual(os.path.realpath(existing[0]["found_path"]), os.path.realpath(model_file))

    def test_check_model_files_preserves_genuine_mismatches(self):
        """Verify that genuine relative path differences are still reported as mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = os.path.join(tmpdir, "models", "checkpoints")
            sub_dir = os.path.join(models_dir, "subfolder")
            os.makedirs(sub_dir, exist_ok=True)
            model_file = os.path.join(sub_dir, "model.safetensors")
            with open(model_file, "w") as f:
                f.write("weights")

            mock_folder_paths = MagicMock()
            mock_folder_paths.base_path = tmpdir
            mock_folder_paths.models_dir = os.path.join(tmpdir, "models")
            mock_folder_paths.get_folder_paths.side_effect = lambda k: [models_dir] if k == "checkpoints" else []

            with patch.object(md, "folder_paths", mock_folder_paths):
                found_models = [{
                    "filename": "model.safetensors",
                    "requested_path": "model.safetensors",  # Root requested, but file is in subfolder
                    "suggested_folder": "checkpoints",
                }]
                missing, existing, path_mismatches = md.check_model_files(found_models)

                self.assertEqual(len(missing), 0)
                self.assertEqual(len(existing), 1)
                self.assertEqual(len(path_mismatches), 1, "Genuine subfolder mismatch must be reported")

    def test_web_api_resolve_model_search_paths_aliases_and_fallback(self):
        """Verify _resolve_model_search_paths in web_api handles aliases and models_root prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = os.path.join(tmpdir, "models")
            ultralytics_bbox_dir = os.path.join(models_dir, "ultralytics", "bbox")
            os.makedirs(ultralytics_bbox_dir, exist_ok=True)

            mock_folder_paths = MagicMock()
            mock_folder_paths.models_dir = models_dir
            mock_folder_paths.base_path = tmpdir

            def get_paths(key):
                if key == "ultralytics_bbox":
                    return [ultralytics_bbox_dir]
                raise KeyError(key)

            mock_folder_paths.get_folder_paths.side_effect = get_paths

            with patch.object(web_api, "folder_paths", mock_folder_paths), \
                 patch.object(web_api, "is_path_within_allowed_roots", lambda p: True):
                paths = web_api._resolve_model_search_paths("ultralytics/bbox")
                self.assertIn(os.path.realpath(ultralytics_bbox_dir), paths)


if __name__ == "__main__":
    unittest.main()
