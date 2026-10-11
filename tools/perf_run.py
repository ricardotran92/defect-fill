#!/usr/bin/env python
"""Run a train/inference/evaluate command and save a performance report.

Records the machine configuration, a CPU/RAM/GPU snapshot before and after the
run, aggregated (mean/peak) usage while it runs, wall time and throughput.
Usage is sampled sparsely (default every 60 s) in a background thread of this
wrapper and only aggregates are kept, so the workload itself is not slowed.

    python tools/perf_run.py --gpu 2 -- python train.py --data_dir ... --output_dir runs/x
    python tools/perf_run.py --gpu 2 --tag hazelnut_hole -- python inference.py ...

Reports: <report-dir>/<timestamp>_<script>_<tag>.json (full) and
<report-dir>/perf_summary.csv (one row per run, appended).
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from importlib import metadata, util
from pathlib import Path

import psutil

GPU_FIELDS = [
    "index", "name", "memory.total", "memory.used", "utilization.gpu",
    "temperature.gpu", "power.draw", "power.limit", "driver_version", "pci.bus_id",
]
CSV_COLUMNS = [
    "start_time", "host", "script", "tag", "status", "exit_code", "gpu_index", "gpu_name",
    "wall_seconds", "work_unit", "work_done", "throughput_per_s", "seconds_per_unit",
    "images_processed", "images_per_s",
    "gpu_util_mean", "gpu_util_max", "gpu_mem_used_max_mib", "proc_gpu_mem_max_mib",
    "gpu_power_mean_w", "gpu_temp_max_c", "proc_cpu_percent_mean", "proc_cpu_percent_max",
    "proc_rss_max_gib", "sys_cpu_percent_mean", "sys_ram_used_max_gib",
    "before_gpu_mem_used_mib", "before_gpu_util", "before_sys_ram_used_gib",
    "samples", "git_commit", "output_dir", "report_json",
]
GIB = 1024 ** 3


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def query_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        row = dict(zip(GPU_FIELDS, [v.strip() for v in line.split(",")]))
        for key in ("index", "memory.total", "memory.used", "utilization.gpu",
                    "temperature.gpu", "power.draw", "power.limit"):
            row[key] = _num(row.get(key))
        if row["index"] is not None:
            row["index"] = int(row["index"])
        gpus.append(row)
    return gpus


def query_gpu_processes():
    """Compute processes on all GPUs: pid, bus id and used memory (MiB)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.strip().splitlines():
        pid, bus_id, mem = [v.strip() for v in line.split(",")]
        procs.append({"pid": int(pid), "bus_id": bus_id, "used_mib": _num(mem)})
    return procs


def package_versions():
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "peft",
                 "accelerate", "lpips", "albumentations", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    # Read torch's CUDA build without importing torch (slow, and would touch the GPU).
    spec = util.find_spec("torch")
    if spec and spec.origin:
        version_file = Path(spec.origin).parent / "version.py"
        text = version_file.read_text() if version_file.exists() else ""
        match = re.search(r"^cuda\s*(?::\s*\S+\s*)?=\s*'([^']*)'", text, re.M)
        versions["torch_cuda"] = match.group(1) if match else None
    return versions


def git_info(cwd):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def system_info():
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gib": round(vm.total / GIB, 2),
        "swap_total_gib": round(psutil.swap_memory().total / GIB, 2),
        "gpus": [{k: g[k] for k in ("index", "name", "memory.total", "power.limit",
                                    "driver_version", "pci.bus_id")} for g in query_gpus()],
        "packages": package_versions(),
    }


def snapshot(gpu_index):
    vm = psutil.virtual_memory()
    gpus = query_gpus()
    selected = next((g for g in gpus if g["index"] == gpu_index), None)
    on_selected = [p for p in query_gpu_processes()
                   if selected and p["bus_id"].lower() == selected["pci.bus_id"].lower()]
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "sys_cpu_percent": psutil.cpu_percent(interval=1.0),
        "load_avg_1_5_15": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
        "sys_ram_used_gib": round((vm.total - vm.available) / GIB, 2),
        "sys_ram_available_gib": round(vm.available / GIB, 2),
        "selected_gpu": selected,
        "selected_gpu_other_processes": len(on_selected),
        "selected_gpu_other_processes_mib": sum(p["used_mib"] or 0 for p in on_selected),
        "all_gpus": [{k: g[k] for k in ("index", "memory.used", "utilization.gpu")} for g in gpus],
    }


class Sampler(threading.Thread):
    """Sparse background sampling of the child process tree and the selected GPU."""

    def __init__(self, root_pid, gpu_index, interval):
        super().__init__(daemon=True)
        self.root_pid, self.gpu_index, self.interval = root_pid, gpu_index, interval
        self.stop_event = threading.Event()
        self.data = {k: [] for k in ("proc_cpu", "proc_rss", "proc_gpu_mem", "sys_cpu",
                                     "sys_ram", "gpu_util", "gpu_mem", "gpu_power", "gpu_temp")}
        self.procs = {}

    def _tree(self):
        try:
            root = psutil.Process(self.root_pid)
            members = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            return []
        for proc in members:
            if proc.pid not in self.procs:
                self.procs[proc.pid] = proc
                try:
                    proc.cpu_percent(None)  # prime the per-process counter
                except psutil.Error:
                    pass
        return members

    def sample(self):
        members = self._tree()
        cpu = rss = 0.0
        for proc in members:
            try:
                cpu += self.procs[proc.pid].cpu_percent(None)
                rss += proc.memory_info().rss
            except psutil.Error:
                pass
        pids = {p.pid for p in members}
        vm = psutil.virtual_memory()
        self.data["proc_cpu"].append(cpu)
        self.data["proc_rss"].append(rss / GIB)
        self.data["sys_cpu"].append(psutil.cpu_percent(None))
        self.data["sys_ram"].append((vm.total - vm.available) / GIB)
        gpu = next((g for g in query_gpus() if g["index"] == self.gpu_index), None)
        if gpu:
            for key, field in (("gpu_util", "utilization.gpu"), ("gpu_mem", "memory.used"),
                               ("gpu_power", "power.draw"), ("gpu_temp", "temperature.gpu")):
                if gpu[field] is not None:
                    self.data[key].append(gpu[field])
        self.data["proc_gpu_mem"].append(
            sum(p["used_mib"] or 0 for p in query_gpu_processes() if p["pid"] in pids))

    def run(self):
        self._tree()
        psutil.cpu_percent(None)
        # First sample after a short warm-up so short runs still get one sample.
        wait = min(self.interval, 15)
        while not self.stop_event.wait(wait):
            self.sample()
            wait = self.interval

    def summary(self):
        def agg(values):
            if not values:
                return {"mean": None, "max": None}
            return {"mean": round(statistics.fmean(values), 2), "max": round(max(values), 2)}
        return {"samples": len(self.data["sys_cpu"]), "interval_s": self.interval,
                **{key: agg(values) for key, values in self.data.items()}}


def arg_value(argv, name, default, cast=str):
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return cast(argv[i + 1])
        if token.startswith(name + "="):
            return cast(token.split("=", 1)[1])
    return default


def count_generated(directory):
    if not directory or not Path(directory).exists():
        return 0
    return sum(1 for _ in Path(directory).rglob("*_generated.png"))


def workload_before(script, argv):
    if script == "inference.py":
        return {"generated_before": count_generated(arg_value(argv, "--output_dir", None))}
    return {}


def workload_after(script, argv, before, output_text, wall):
    """Work done by THIS run (resume-aware) for the DefectFill scripts; defaults mirror their argparse."""
    info = {"unit": None, "done": None, "images": None}
    if script == "train.py":
        max_steps = arg_value(argv, "--max_train_steps", 2000, int)
        batch = arg_value(argv, "--batch_size", 1, int)
        accum = arg_value(argv, "--gradient_accumulation_steps", 4, int)
        resumed = re.findall(r"Resuming from step (\d+)", output_text)
        start = int(resumed[-1]) if resumed else 0
        steps = max(max_steps - start, 0)
        info.update(unit="optimizer_step", done=steps, images=steps * batch * accum,
                    resumed_from_step=start, max_train_steps=max_steps,
                    batch_size=batch, gradient_accumulation_steps=accum)
    elif script == "inference.py":
        after = count_generated(arg_value(argv, "--output_dir", None))
        made = after - before.get("generated_before", 0)
        candidates = arg_value(argv, "--num_samples", 8, int)
        info.update(unit="output_image", done=made, images=made * candidates,
                    generated_total=after, candidates_per_image=candidates,
                    diffusion_steps=arg_value(argv, "--steps", 50, int),
                    total_images_target=arg_value(argv, "--total_images", 1000, int))
    elif script == "evaluate.py":
        n = count_generated(arg_value(argv, "--generated_dir", None))
        info.update(unit="generated_image_scored", done=n, images=n)
    if info["done"] and wall:
        info["throughput_per_s"] = round(info["done"] / wall, 4)
        info["seconds_per_unit"] = round(wall / info["done"], 3)
    if info["images"] and wall:
        info["images_per_s"] = round(info["images"] / wall, 4)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", default=os.environ.get("GPU"),
                        help="Physical GPU index (sets CUDA_VISIBLE_DEVICES). Default: $GPU, else $CUDA_VISIBLE_DEVICES")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Seconds between background samples (default 60); 0 = only before/after snapshots")
    parser.add_argument("--report-dir", default="runs/perf")
    parser.add_argument("--tag", default="", help="Free-text label stored in the report")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command to run>")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command, e.g. -- python train.py ...")

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # make CUDA indices match nvidia-smi
    gpu = args.gpu if args.gpu is not None else env.get("CUDA_VISIBLE_DEVICES")
    if gpu is None or "," in str(gpu):
        parser.error("choose exactly one GPU with --gpu N (or export GPU=N)")
    gpu_index = int(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    py_index = next((i for i, t in enumerate(command) if t.endswith(".py")), None)
    script = Path(command[py_index]).name if py_index is not None else Path(command[0]).name
    script_args = command[py_index + 1:] if py_index is not None else command[1:]

    report = {
        "tag": args.tag,
        "command": shlex.join(command),
        "cwd": os.getcwd(),
        "cuda_visible_devices": str(gpu_index),
        "git": git_info(os.getcwd()),
        "system": system_info(),
    }
    print(f"[perf] snapshot before run (GPU {gpu_index}) ...", flush=True)
    report["before"] = snapshot(gpu_index)
    before_work = workload_before(script, script_args)

    start = time.time()
    report["start_time"] = dt.datetime.now().isoformat(timespec="seconds")
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sampler = Sampler(child.pid, gpu_index, args.interval) if args.interval > 0 else None
    if sampler:
        sampler.start()

    # Pass output through unchanged (tqdm \r updates included); keep a tail to detect resume.
    tail = b""
    status = "completed"
    try:
        while True:
            chunk = os.read(child.stdout.fileno(), 65536)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            tail = (tail + chunk)[-2_000_000:]
        exit_code = child.wait()
    except KeyboardInterrupt:
        status = "interrupted"
        child.send_signal(signal.SIGINT)
        exit_code = child.wait()
    wall = time.time() - start
    if sampler:
        sampler.stop_event.set()
        sampler.join(timeout=30)
    if status == "completed" and exit_code != 0:
        status = "failed"

    report.update({
        "end_time": dt.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "exit_code": exit_code,
        "wall_seconds": round(wall, 2),
        "wall_hms": str(dt.timedelta(seconds=round(wall))),
        "script": script,
        "script_args": script_args,
        "during": sampler.summary() if sampler else None,
        "workload": workload_after(script, script_args, before_work,
                                   tail.decode("utf-8", "replace"), wall),
    })
    print("[perf] snapshot after run ...", flush=True)
    report["after"] = snapshot(gpu_index)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(filter(None, [stamp, Path(script).stem, re.sub(r"\W+", "-", args.tag)]))
    json_path = report_dir / f"{name}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    during = report["during"] or {}
    work = report["workload"]
    sel = report["before"]["selected_gpu"] or {}

    def stat(key, field):
        return during.get(key, {}).get(field)

    row = {
        "start_time": report["start_time"], "host": report["system"]["hostname"],
        "script": script, "tag": args.tag, "status": status, "exit_code": exit_code,
        "gpu_index": gpu_index, "gpu_name": sel.get("name"),
        "wall_seconds": report["wall_seconds"], "work_unit": work.get("unit"),
        "work_done": work.get("done"), "throughput_per_s": work.get("throughput_per_s"),
        "seconds_per_unit": work.get("seconds_per_unit"),
        "images_processed": work.get("images"), "images_per_s": work.get("images_per_s"),
        "gpu_util_mean": stat("gpu_util", "mean"), "gpu_util_max": stat("gpu_util", "max"),
        "gpu_mem_used_max_mib": stat("gpu_mem", "max"),
        "proc_gpu_mem_max_mib": stat("proc_gpu_mem", "max"),
        "gpu_power_mean_w": stat("gpu_power", "mean"), "gpu_temp_max_c": stat("gpu_temp", "max"),
        "proc_cpu_percent_mean": stat("proc_cpu", "mean"),
        "proc_cpu_percent_max": stat("proc_cpu", "max"),
        "proc_rss_max_gib": stat("proc_rss", "max"),
        "sys_cpu_percent_mean": stat("sys_cpu", "mean"),
        "sys_ram_used_max_gib": stat("sys_ram", "max"),
        "before_gpu_mem_used_mib": sel.get("memory.used"),
        "before_gpu_util": sel.get("utilization.gpu"),
        "before_sys_ram_used_gib": report["before"]["sys_ram_used_gib"],
        "samples": during.get("samples", 0),
        "git_commit": report["git"]["commit"],
        "output_dir": arg_value(script_args, "--output_dir", None),
        "report_json": str(json_path),
    }
    csv_path = report_dir / "perf_summary.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print(f"[perf] {status} in {report['wall_hms']} | {work.get('done')} {work.get('unit')} "
          f"| {work.get('throughput_per_s')} /s | report: {json_path}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
