import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_training_builds_manifest_before_allocating_model_and_uses_only_train_pairs():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert source.index("build_split_manifest(") < source.index("DefectFillModel(")
    assert 'manifest_path = os.path.join(args.output_dir, "split_manifest.json")' in source
    assert 'train_pairs=manifest["train"]' in source
    assert '"view_mode": "full"' in source


def test_training_defaults_match_the_paper_protocol_and_power_loss_policy():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--batch_size", type=int, default=1)' in source
    assert 'parser.add_argument("--gradient_accumulation_steps", type=int, default=4)' in source
    assert 'parser.add_argument("--seed", type=int, default=42' in source
    assert 'parser.add_argument("--split_seed", type=int, default=42' in source
    assert 'parser.add_argument("--save_steps", type=int, default=100)' in source
    assert 'parser.add_argument("--resume_from", type=str, default="auto")' in source
    assert 'parser.add_argument("--lora_dropout", type=float, default=0.1)' in source


def test_training_validates_and_restores_automatic_resume_state():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "find_latest_valid_checkpoint(" in source
    assert "expected_metadata=checkpoint_metadata" in source
    assert "restore_rng=True" in source
    assert "metadata=checkpoint_metadata" in source
    assert "torch.isfinite(total_loss)" in source
    assert "signal.SIGINT" in source
    assert "Interrupted; checkpoint saved" in source
