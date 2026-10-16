"""Record one finished DefectFill run into the experiments/ registry (small, git-tracked files).

Collects everything needed to compare runs later without the large artifacts:
code version, train config, loss summary (TensorBoard), inference settings, LFS scores,
evaluation rows (self and cross-evaluation), perf rows and mask-fill statistics.

Writes experiments/records/<run_id>/run.json, updates experiments/registry.json and regenerates
experiments/RESULTS.md. A run_id is never overwritten without --update, and the same inference
output cannot be registered twice under different ids.

Example (paths are local copies of the server runs, see the project guide):
  python tools/record_run.py --run_id 20261014_v2_hazelnut_hole_100img --version v2 --commit 784e8cd \
    --train_dir runs/hazelnut_hole_paper_2000step_v2 --generated_dir runs/hazelnut_hole_paper_generated_v2_100 \
    --metrics self runs/paper_protocol_metrics.csv defectfill_02_v2_100img \
    --metrics xeval_baseline_evaluator ../defect-fill/runs/compare/xeval_100.csv new_v2_out__baseline_eval_100img \
    --perf_csv runs/perf/perf_summary.csv
"""
import argparse
import csv
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_fill import mask_fill_stats  # noqa: E402

RUN_ID_PATTERN = re.compile(r"^\d{8}_[a-z0-9]+(_[a-z0-9]+)*$")
LOSS_TAGS = ("Loss/Defect", "Loss/Object", "Loss/Attention", "Loss/Total")
LOSS_WINDOW = 400
METRIC_FIELDS = ("config", "real_count", "generated_count", "kid_x1000", "kid_x1000_std", "ic_lpips",
                 "ic_lpips_cluster_std", "ic_lpips_valid_clusters", "ic_lpips_reference", "protocol_version",
                 "split_manifest_fingerprint", "timestamp")
PERF_FIELDS = ("start_time", "script", "tag", "status", "wall_seconds", "work_done", "throughput_per_s",
               "gpu_name", "proc_gpu_mem_max_mib", "git_commit", "output_dir")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def _to_number(value):
    for cast in (int, float):
        try:
            return cast(value)
        except (TypeError, ValueError):
            pass
    return value


def loss_summary(tensorboard_dir):
    """Mean of each loss over consecutive LOSS_WINDOW-step windows, plus loss-scale range."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"error": "tensorboard not installed"}
    accumulator = EventAccumulator(str(tensorboard_dir))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    summary = {"window_steps": LOSS_WINDOW}
    for tag in LOSS_TAGS:
        if tag not in tags:
            continue
        values = [event.value for event in accumulator.Scalars(tag)]
        windows = [sum(values[i:i + LOSS_WINDOW]) / len(values[i:i + LOSS_WINDOW]) for i in range(0, len(values), LOSS_WINDOW)]
        summary[tag.split("/")[1].lower()] = {"steps": len(values), "window_means": [round(v, 6) for v in windows]}
    if "LossScale" in tags:
        scales = [event.value for event in accumulator.Scalars("LossScale")]
        summary["loss_scale"] = {"min": min(scales), "max": max(scales), "last": scales[-1],
                                 "decreases": sum(b < a for a, b in zip(scales, scales[1:]))}
    return summary


def inference_record(generated_dir):
    log = _read_json(Path(generated_dir) / "inference_log.json")
    results = log.pop("results", [])
    scores = [result["lpips_score"] for result in results]
    fingerprint = hashlib.sha256(json.dumps(
        [(r["output_idx"], r["input_image"].replace("\\", "/").split("/")[-1], round(r["lpips_score"], 6)) for r in results]
    ).encode()).hexdigest()[:16]
    return {
        "settings": log,
        "outputs": len(results),
        "lfs_lpips": {"mean": sum(scores) / len(scores), "min": min(scores), "max": max(scores)} if scores else None,
        "fingerprint": fingerprint,
    }


def metric_rows(specs):
    rows = []
    for label, path, config in specs:
        with open(path, encoding="utf-8") as handle:
            matches = [row for row in csv.DictReader(handle) if row.get("config") == config]
        if not matches:
            raise SystemExit(f"No row with config={config!r} in {path}")
        row = matches[-1]
        rows.append({"label": label, "source": Path(path).as_posix(),
                     **{key: _to_number(row[key]) for key in METRIC_FIELDS if key in row and row[key] != ""}})
    return rows


def perf_rows(perf_csv, directories):
    names = {Path(d).name for d in directories if d}
    with open(perf_csv, encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if Path(row.get("output_dir", "")).name in names]
    return [{key: _to_number(row[key]) for key in PERF_FIELDS if key in row} for row in rows]


def regenerate_results(registry_dir, registry):
    lines = [
        "# Kết quả các run (tự sinh bởi `tools/record_run.py`, không sửa tay)",
        "",
        "Chỉ so các dòng có cùng số ảnh, và chỉ so IC-LPIPS khi cùng tâm cluster (`ref`). "
        "`fill_thin`/`fill_thick`: tỉ lệ pixel trong phần mảnh/dày của mask được vẽ lại; "
        "`drift_far`: thay đổi trung bình (0–255) ở nền xa mask, lý tưởng là 0. Định nghĩa trong `tools/mask_fill.py`. "
        "Run có hậu tố `_priority` dùng kế hoạch ưu tiên mask nét mảnh 3:1 (75/100 ảnh trên 5 mask), "
        "nên chỉ so với các run `_priority` khác.",
        "",
        "| run_id | phiên bản | commit | ảnh | KID×1000 (tự chấm) | IC-LPIPS (tự chấm, ref) | chấm chéo | fill_thick | fill_thin | drift_far | L_def đầu → cuối |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in registry["runs"]:
        run = _read_json(registry_dir / entry["path"])
        self_rows = [m for m in run["metrics"] if m["label"] == "self"]
        cross = [m for m in run["metrics"] if m["label"] != "self"]
        own = self_rows[0] if self_rows else {}
        fill = run.get("mask_fill") or {}
        defect_loss = ((run.get("train") or {}).get("loss_summary") or {}).get("defect", {}).get("window_means")
        fmt = lambda value, spec: format(value, spec) if isinstance(value, (int, float)) else "—"
        lines.append("| " + " | ".join([
            f"[{run['run_id']}]({entry['path']})", run["version"], f"`{run['commit']}`", str(run["inference"]["outputs"]),
            f"{fmt(own.get('kid_x1000'), '.2f')} (±{fmt(own.get('kid_x1000_std'), '.2f')})",
            f"{fmt(own.get('ic_lpips'), '.3f')} ({own.get('ic_lpips_reference', 'heldout')})",
            "; ".join(f"{m['label']}: KID {fmt(m.get('kid_x1000'), '.2f')}, IC {fmt(m.get('ic_lpips'), '.3f')} "
                      f"({m.get('ic_lpips_reference', 'heldout')})" for m in cross) or "—",
            fmt(fill.get("fill_thick"), ".1%"), fmt(fill.get("fill_thin"), ".1%"), fmt(fill.get("drift_far"), ".2f"),
            f"{defect_loss[0]:.4f} → {defect_loss[-1]:.4f}" if defect_loss else "— (dùng checkpoint có sẵn)",
        ]) + " |")
    (registry_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Record a finished run into experiments/")
    parser.add_argument("--run_id", required=True, help="YYYYMMDD_<version>_<object>_<defect>_<N>img[_note]")
    parser.add_argument("--version", required=True, help="Code version, e.g. baseline, v1, v2, v3")
    parser.add_argument("--commit", required=True, help="Git commit the run was produced with")
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--train_dir", help="Training run directory; omit when the run reuses another run's checkpoint")
    parser.add_argument("--reuses_checkpoint_of", help="run_id whose checkpoint this inference-only run used")
    parser.add_argument("--metrics", nargs=3, action="append", default=[], metavar=("LABEL", "CSV", "CONFIG"),
                        help="Evaluation row to attach; LABEL 'self' = the run's own evaluate.py")
    parser.add_argument("--perf_csv")
    parser.add_argument("--server_root", help="Where the full artifacts live, e.g. kindred:~/projects/defectfill_02")
    parser.add_argument("--notes", default="")
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parents[1] / "experiments"))
    parser.add_argument("--update", action="store_true", help="Overwrite an existing run_id")
    args = parser.parse_args()

    if not RUN_ID_PATTERN.match(args.run_id):
        raise SystemExit(f"run_id {args.run_id!r} must look like YYYYMMDD_version_object_defect_Nimg")
    registry_dir = Path(args.registry)
    registry_path = registry_dir / "registry.json"
    registry = _read_json(registry_path) if registry_path.exists() else {"runs": []}

    inference = inference_record(args.generated_dir)
    for entry in registry["runs"]:
        if entry["run_id"] == args.run_id and not args.update:
            raise SystemExit(f"{args.run_id} is already recorded; pass --update to overwrite it")
        if entry["run_id"] != args.run_id and entry.get("inference_fingerprint") == inference["fingerprint"]:
            raise SystemExit(f"The same inference output is already recorded as {entry['run_id']}")

    train = None
    if args.train_dir:
        train_dir = Path(args.train_dir)
        train = {"config": _read_json(train_dir / "train_config.json"),
                 "loss_summary": loss_summary(train_dir / "tensorboard")}
    run = {
        "run_id": args.run_id,
        "version": args.version,
        "commit": args.commit,
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "notes": args.notes,
        "reuses_checkpoint_of": args.reuses_checkpoint_of,
        "train": train,
        "inference": inference,
        "metrics": metric_rows(args.metrics),
        "mask_fill": mask_fill_stats(args.generated_dir),
        "perf": perf_rows(args.perf_csv, [args.train_dir, args.generated_dir]) if args.perf_csv else [],
        "artifacts": {"server_root": args.server_root, "train_dir": args.train_dir and Path(args.train_dir).name,
                      "generated_dir": Path(args.generated_dir).name},
    }
    run_path = Path("records") / args.run_id / "run.json"
    _write_json(registry_dir / run_path, run)

    registry["runs"] = sorted(
        [entry for entry in registry["runs"] if entry["run_id"] != args.run_id]
        + [{"run_id": args.run_id, "version": args.version, "commit": args.commit,
            "path": run_path.as_posix(), "inference_fingerprint": inference["fingerprint"]}],
        key=lambda entry: entry["run_id"],
    )
    _write_json(registry_path, registry)
    regenerate_results(registry_dir, registry)
    print(f"recorded {args.run_id} -> {registry_dir / run_path}")


if __name__ == "__main__":
    main()
