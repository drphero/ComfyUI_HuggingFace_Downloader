import os
import sys
import unittest
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

from huggingface_hub.hf_api import RepoFile, RepoFolder

import ComfyUI_HuggingFace_Downloader.file_manager as fm
import ComfyUI_HuggingFace_Downloader.web_api as web_api


FOLDER_TYPES = [
    "checkpoints", "loras", "vae", "text_encoders", "diffusion_models",
    "clip_vision", "controlnet", "upscale_models", "model_patches",
]


def _repo_file(path, size=100):
    return RepoFile(path=path, size=size, oid="x")


class TestSuggestModelFolder(unittest.TestCase):

    def test_root_file_has_no_suggestion(self):
        self.assertEqual(fm.suggest_model_folder("model.safetensors", FOLDER_TYPES), "")

    def test_top_level_model_folder(self):
        self.assertEqual(fm.suggest_model_folder("diffusion_models/a.safetensors", FOLDER_TYPES), "diffusion_models")
        self.assertEqual(fm.suggest_model_folder("vae/flux2-vae.safetensors", FOLDER_TYPES), "vae")

    def test_nested_split_files_layout(self):
        self.assertEqual(
            fm.suggest_model_folder("split_files/diffusion_models/wan.safetensors", FOLDER_TYPES),
            "diffusion_models",
        )
        self.assertEqual(
            fm.suggest_model_folder("split_files/clip_vision/clip_vision_h.safetensors", FOLDER_TYPES),
            "clip_vision",
        )

    def test_sub_path_below_match_is_kept(self):
        self.assertEqual(fm.suggest_model_folder("loras/wan/style.safetensors", FOLDER_TYPES), "loras/wan")
        self.assertEqual(
            fm.suggest_model_folder("split_files/loras/wan/v2/style.safetensors", FOLDER_TYPES),
            "loras/wan/v2",
        )

    def test_first_matching_segment_wins(self):
        self.assertEqual(fm.suggest_model_folder("checkpoints/vae/x.safetensors", FOLDER_TYPES), "checkpoints/vae")

    def test_legacy_aliases(self):
        self.assertEqual(fm.suggest_model_folder("unet/x.safetensors", FOLDER_TYPES), "diffusion_models")
        self.assertEqual(fm.suggest_model_folder("clip/t5xxl.safetensors", FOLDER_TYPES), "text_encoders")

    def test_case_insensitive(self):
        self.assertEqual(fm.suggest_model_folder("Split_Files/VAE/x.safetensors", FOLDER_TYPES), "vae")

    def test_unrelated_folders(self):
        self.assertEqual(fm.suggest_model_folder("example workflows_Wan2.1/t2v.json", FOLDER_TYPES), "")
        # Diffusers-style singular names are not ComfyUI folders.
        self.assertEqual(fm.suggest_model_folder("text_encoder/model.safetensors", FOLDER_TYPES), "")

    def test_default_folder_types_fallback(self):
        # folder_paths is mocked in tests, so the fallback list is used.
        self.assertEqual(fm.suggest_model_folder("split_files/text_encoders/t5.safetensors"), "text_encoders")


class TestListHfRepoFiles(unittest.TestCase):

    def _list(self, link, entries):
        api = MagicMock()
        api.list_repo_tree.return_value = entries
        with patch("huggingface_hub.HfApi", return_value=api), \
                patch.object(web_api, "get_token", return_value=None), \
                patch.object(web_api, "get_model_folder_types", return_value=FOLDER_TYPES), \
                patch.object(web_api, "resolve_target_dir", side_effect=lambda f: os.path.join("/nonexistent-models", f)):
            return web_api._list_hf_repo_files(link), api

    def test_shapes_files_and_suggestions(self):
        entries = [
            _repo_file(".gitattributes"),
            _repo_file("README.md"),
            RepoFolder(path="split_files", oid="t"),
            _repo_file("split_files/diffusion_models/wan_fp8.safetensors", size=16_000_000_000),
            _repo_file("split_files/vae/wan_vae.safetensors"),
            _repo_file("lora_at_root.safetensors"),
        ]
        result, api = self._list("Comfy-Org/Wan_2.1_ComfyUI_repackaged", entries)

        self.assertEqual(result["repo"], "Comfy-Org/Wan_2.1_ComfyUI_repackaged")
        self.assertEqual(result["revision"], "main")
        self.assertFalse(result["is_diffusers"])
        self.assertEqual(api.list_repo_tree.call_args.kwargs["path_in_repo"], None)

        by_path = {f["path"]: f for f in result["files"]}
        self.assertNotIn("split_files", by_path)  # folders are dropped
        self.assertFalse(by_path["README.md"]["is_model"])
        self.assertFalse(by_path[".gitattributes"]["is_model"])

        dm = by_path["split_files/diffusion_models/wan_fp8.safetensors"]
        self.assertTrue(dm["is_model"])
        self.assertEqual(dm["suggested_folder"], "diffusion_models")
        self.assertEqual(dm["dir"], "split_files/diffusion_models")
        self.assertEqual(dm["name"], "wan_fp8.safetensors")
        self.assertEqual(dm["size"], 16_000_000_000)
        self.assertFalse(dm["exists"])

        self.assertEqual(by_path["split_files/vae/wan_vae.safetensors"]["suggested_folder"], "vae")
        self.assertEqual(by_path["lora_at_root.safetensors"]["suggested_folder"], "")

    def test_tree_link_passes_revision_and_subfolder(self):
        entries = [_repo_file("split_files/vae/v.safetensors")]
        result, api = self._list(
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/dev/split_files", entries
        )
        self.assertEqual(result["revision"], "dev")
        self.assertEqual(result["subfolder"], "split_files")
        kwargs = api.list_repo_tree.call_args.kwargs
        self.assertEqual(kwargs["revision"], "dev")
        self.assertEqual(kwargs["path_in_repo"], "split_files")
        self.assertTrue(kwargs["recursive"])

    def test_diffusers_repo_disables_suggestions(self):
        entries = [
            _repo_file("model_index.json"),
            _repo_file("unet/diffusion_pytorch_model.safetensors"),
            _repo_file("vae/diffusion_pytorch_model.safetensors"),
        ]
        result, _ = self._list("someone/diffusers-repo", entries)
        self.assertTrue(result["is_diffusers"])
        self.assertTrue(all(f["suggested_folder"] == "" for f in result["files"]))


if __name__ == "__main__":
    unittest.main()
