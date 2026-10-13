import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import prune_intermediate_checkpoints, save_checkpoint  # noqa: E402


class _TinyBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.zeros(1))


def _save(path, step):
    model = SimpleNamespace(placeholder_token="sks", pipeline=SimpleNamespace(text_encoder=_TinyBranch(), unet=_TinyBranch()))
    save_checkpoint(model, None, step, path)


def test_pruning_keeps_only_the_final_checkpoint(tmp_path):
    for step in (100, 200, 2000):
        _save(tmp_path / f"checkpoint_{step}.pt", step)
    _save(tmp_path / "checkpoint_final.pt", 2000)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")

    removed = prune_intermediate_checkpoints(tmp_path)

    assert sorted(Path(path).name for path in removed) == ["checkpoint_100.pt", "checkpoint_200.pt", "checkpoint_2000.pt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint_final.pt", "notes.txt"]


def test_pruning_refuses_without_a_readable_final_checkpoint(tmp_path):
    _save(tmp_path / "checkpoint_100.pt", 100)
    (tmp_path / "checkpoint_final.pt").write_bytes(b"truncated")

    with pytest.raises(Exception):
        prune_intermediate_checkpoints(tmp_path)

    assert (tmp_path / "checkpoint_100.pt").is_file()


def test_train_prunes_only_after_final_checkpoint_and_allows_opt_out():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert source.index("final_checkpoint_path,") < source.index("prune_intermediate_checkpoints(checkpoints_dir)")
    assert "--keep_intermediate_checkpoints" in source
