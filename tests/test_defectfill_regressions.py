import ast
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("cv2") is None:
    sys.modules.setdefault("cv2", types.SimpleNamespace())

if importlib.util.find_spec("albumentations") is None:
    sys.modules.setdefault("albumentations", types.SimpleNamespace())
    sys.modules.setdefault("albumentations.pytorch", types.SimpleNamespace(ToTensorV2=object))

if importlib.util.find_spec("torch") is None:
    torch_module = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")


    class _Dataset:
        pass


    class _DataLoader:
        def __init__(self, *args, **kwargs):
            pass


    torch_utils_data.Dataset = _Dataset
    torch_utils_data.DataLoader = _DataLoader
    torch_utils.data = torch_utils_data
    torch_module.utils = torch_utils
    sys.modules.setdefault("torch", torch_module)
    sys.modules.setdefault("torch.utils", torch_utils)
    sys.modules.setdefault("torch.utils.data", torch_utils_data)

from data_loader import MVTecDefectDataset  # noqa: E402


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_standard_mvtec_test_defects_pair_with_ground_truth_masks(tmp_path):
    root = tmp_path / "MVTec"
    _touch(root / "hazelnut" / "test" / "hole" / "000.png")
    _touch(root / "hazelnut" / "ground_truth" / "hole" / "000_mask.png")

    dataset = MVTecDefectDataset(
        root_dir=str(root),
        object_class="hazelnut",
        split="train",
        defect_type="hole",
        transform=None,
    )

    assert dataset.images == [str(root / "hazelnut" / "test" / "hole" / "000.png")]
    assert dataset.masks == [str(root / "hazelnut" / "ground_truth" / "hole" / "000_mask.png")]


def test_attention_maps_are_not_detached_before_attention_loss():
    source = (ROOT / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "attention_maps"
            for target in node.targets
        )
    ]

    assert assignments, "expected AttentionStoreProcessor to store attention maps"
    for node in assignments:
        assert ".detach(" not in ast.get_source_segment(source, node)


def test_train_uses_object_branch_attention_and_defect_weighted_object_loss():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert 'attention_loss = obj_outputs.get("attention_loss"' in source
    assert "model.compute_object_loss(obj_outputs[\"noise_pred\"], noise, object_weight_latents" in source
    assert "attention_mask_latents=object_weight_latents" in source
    assert "obj_noise = torch.randn_like(latents)" not in source
    assert "obj_timesteps = torch.randint" not in source


def test_forward_accepts_separate_attention_mask_target():
    source = (ROOT / "model.py").read_text(encoding="utf-8")

    assert "attention_mask_latents: Optional[torch.Tensor] = None" in source
    assert "attention_mask_latents if attention_mask_latents is not None else mask_latents" in source


def test_local_base_model_path_is_wired_for_training_and_inference():
    model_source = (ROOT / "model.py").read_text(encoding="utf-8")
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")
    inference_source = (ROOT / "inference.py").read_text(encoding="utf-8")

    assert "base_model_path=None" in model_source
    assert "local_files_only=True" in model_source

    assert "--base_model_path" in train_source
    assert "base_model_path=args.base_model_path" in train_source
    assert "noise_scheduler = model.scheduler" in train_source

    assert "--base_model_path" in inference_source
    assert "base_model_path=args.base_model_path" in inference_source


def test_lpips_is_loaded_lazily_outside_model_initialization():
    model_source = (ROOT / "model.py").read_text(encoding="utf-8")
    inference_source = (ROOT / "inference.py").read_text(encoding="utf-8")

    assert "self.lpips_model = None" in model_source
    assert "def get_lpips_model" in model_source
    assert "self.lpips_model = lpips.LPIPS" in model_source
    assert "model.get_lpips_model()" in inference_source


def test_model_uses_paper_identifier_without_textual_inversion():
    model_source = (ROOT / "model.py").read_text(encoding="utf-8")

    assert 'placeholder_token="sks"' in model_source
    assert ".add_tokens(" not in model_source
    assert ".resize_token_embeddings(" not in model_source
    assert "token_embeds[self.placeholder_token_id]" not in model_source


def test_both_lora_branches_use_paper_dropout():
    model_source = (ROOT / "model.py").read_text(encoding="utf-8")

    assert "lora_dropout=0.1" in model_source
    assert model_source.count("lora_dropout=lora_dropout") == 2


def test_generation_delegates_to_standard_inpainting_pipeline():
    model_source = (ROOT / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(model_source)
    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    generate_source = ast.get_source_segment(model_source, generate)

    assert "self.pipeline(" in generate_source
    assert "output_type=\"pt\"" in generate_source
    assert "noise_for_bg" not in generate_source
    assert "latents_background" not in generate_source


def test_generation_preserves_background_with_one_persistent_noise_sample():
    model_source = (ROOT / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(model_source)
    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    generate_source = ast.get_source_segment(model_source, generate)

    assert generate_source.count("torch.randn(") == 1
    assert "latents=initial_noise" in generate_source
    assert "callback_on_step_end=preserve_background" in generate_source
    assert "scheduler.add_noise(" in generate_source
    assert "image_latents" in generate_source
    assert "initial_noise" in generate_source


def test_inference_consumes_the_manifest_heldout_partition():
    inference_source = (ROOT / "inference.py").read_text(encoding="utf-8")

    assert "load_split_manifest" in inference_source
    assert 'manifest["heldout"]' in inference_source
    assert "smart_crop_dynamic" not in inference_source
