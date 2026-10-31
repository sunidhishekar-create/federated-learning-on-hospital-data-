"""Run the Docker deployment end to end and save measured results to results/docker/.

    python docker/run_experiment.py                  # all eval_seeds (42..46)
    python docker/run_experiment.py --seeds 42       # one seed
    python docker/run_experiment.py --down           # stop and remove the containers afterwards

Steps: start server + 5 hospital containers, wait until all hospitals have joined, measure the
idle network traffic, `flwr run` once per seed (measuring wall-clock time and network counters
of every container), verify the data isolation, and write results in the same layout as
results/federated*.json. Nothing outside results/docker/ is written.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                  # noqa: E402

from src.data_loader import load_config             # noqa: E402
from src.multiseed import METRICS, _ms              # noqa: E402

OUT = ROOT / "results" / "docker"
ENV = {"FLWR_HOME": str(ROOT / "docker" / "flwr_home")}
H = 5
SERVICES = ["superlink"] + [f"hospital-{i}" for i in range(H)]


def sh(cmd, check=True, **kw):
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check, **kw)


def dc(*args, check=True):
    return sh(["docker", "compose", *args], check=check)


# ---------------------------------------------------------------- measurement
def net_counters(service: str) -> dict:
    """Kernel byte counters of a container's network interfaces (everything except loopback)."""
    out = dc("exec", "-T", service, "cat", "/proc/net/dev").stdout
    rx = tx = 0
    for line in out.splitlines()[2:]:
        name, data = line.split(":", 1)
        if name.strip() != "lo":
            cols = data.split()
            rx, tx = rx + int(cols[0]), tx + int(cols[8])
    return {"rx": rx, "tx": tx}


def container_memory(service: str) -> dict:
    """Kernel cgroup counters of one container, in MB: memory now, peak since the container
    started, the anonymous (heap) part now, and peak swap."""
    out = dc("exec", "-T", service, "sh", "-c",
             "cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak "
             "/sys/fs/cgroup/memory.swap.peak; grep '^anon ' /sys/fs/cgroup/memory.stat").stdout.split()
    cur, peak, swap_peak, anon = int(out[0]), int(out[1]), int(out[2]), int(out[4])
    return {"current_mb": round(cur / 2**20), "peak_mb": round(peak / 2**20),
            "anon_now_mb": round(anon / 2**20), "swap_peak_mb": round(swap_peak / 2**20)}


def process_memory(service: str) -> list:
    """Resident + swapped memory of each process in a container (the idle Flower processes)."""
    script = ('for p in /proc/[0-9]*; do r=$(grep VmRSS $p/status 2>/dev/null | tr -s " " | cut -d" " -f2); '
              's=$(grep VmSwap $p/status 2>/dev/null | tr -s " " | cut -d" " -f2); '
              '[ -n "$r" ] && echo "$r $s $(tr "\\0" " " < $p/cmdline | cut -d" " -f2)"; done')
    procs = []
    for line in dc("exec", "-T", service, "sh", "-c", script).stdout.splitlines():
        rss, swap, cmd = line.split(maxsplit=2)
        if "python" in cmd or "flwr" in cmd or "flower" in cmd:
            procs.append({"process": cmd.split("/")[-1], "rss_mb": round(int(rss) / 1024),
                          "swap_mb": round(int(swap or 0) / 1024)})
    return procs


def memory_snapshot() -> dict:
    return {s: container_memory(s) for s in SERVICES}


def snapshot() -> dict:
    return {s: net_counters(s) for s in SERVICES}


def delta(a: dict, b: dict) -> dict:
    return {s: {k: b[s][k] - a[s][k] for k in ("rx", "tx")} for s in a}


# ---------------------------------------------------------------- start up
def start_stack():
    print("starting containers ...")
    dc("up", "-d", "--build", "--force-recreate")       # fresh containers: memory peaks start at 0
    deadline = time.time() + 180
    while time.time() < deadline:
        states = dc("ps", "--format", "{{.Service}} {{.State}}").stdout.split("\n")
        running = {line.split()[0] for line in states if line.endswith("running")}
        if set(SERVICES) <= running:
            break
        time.sleep(2)
    else:
        raise RuntimeError("containers did not reach the running state")
    # the SuperLink logs one "Activated node" line per hospital that has joined the federation
    deadline = time.time() + 180
    while time.time() < deadline:
        out = dc("logs", "superlink", check=False)
        if (out.stdout + out.stderr).count("Activated node") >= H:
            return
        time.sleep(3)
    raise RuntimeError("not all hospitals joined the SuperLink within 3 minutes")


def idle_traffic(seconds=20) -> dict:
    """Traffic with no run in progress (SuperNodes polling the SuperLink), per second."""
    a = net_counters("superlink")
    time.sleep(seconds)
    b = net_counters("superlink")
    return {"seconds": seconds, "rx_bytes_per_s": (b["rx"] - a["rx"]) / seconds,
            "tx_bytes_per_s": (b["tx"] - a["tx"]) / seconds}


def docker_info() -> dict:
    info = json.loads(sh(["docker", "info", "--format", "{{json .}}"]).stdout)
    images = {s: sh(["docker", "image", "inspect", f"hospital-fl-{r}:1.42.0", "--format",
                     "{{.Id}} {{.Size}}"]).stdout.split() for s, r in
              (("server", "superlink"), ("hospital", "supernode"))}
    return {"docker_server_version": info.get("ServerVersion"), "docker_cpus": info.get("NCPU"),
            "docker_memory_bytes": info.get("MemTotal"), "architecture": info.get("Architecture"),
            "images": {k: {"id": v[0], "size_bytes": int(v[1])} for k, v in images.items()}}


# ---------------------------------------------------------------- one run
def run_seed(seed: int) -> dict:
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["flwr", "run", ".", "docker-deployment", "--run-config", f"seed={seed}", "--stream"],
        cwd=ROOT, text=True, capture_output=True, env={**os.environ, **ENV})
    wall = time.perf_counter() - t0
    after = snapshot()
    (log_dir / f"flwr_run_seed{seed}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    result_file = OUT / "federated_runs" / f"seed_{seed}.json"
    if proc.returncode != 0 or not result_file.exists():
        raise RuntimeError(f"seed {seed} failed, see {log_dir / f'flwr_run_seed{seed}.log'}")
    d = delta(before, after)
    meta = {"seed": seed, "flwr_run_wall_seconds": round(wall, 2),
            "container_memory_after_run": memory_snapshot(),    # peak = since containers started
            "container_traffic_bytes": d,
            "server_traffic_bytes": d["superlink"]}
    (OUT / "federated_runs" / f"meta_seed_{seed}.json").write_text(json.dumps(meta, indent=2))
    r = json.loads(result_file.read_text())
    print(f"seed {seed}: ROC-AUC {r['global_test']['roc_auc']:.4f} | in-app {r['train_seconds']} s | "
          f"end-to-end `flwr run` {wall:.1f} s")
    return meta


# ---------------------------------------------------------------- isolation evidence
def check_isolation() -> dict:
    """Prove what each container can see. Written to results/docker/isolation_check.json."""
    evidence = {"mounts": {}, "visible_files": {}, "cross_hospital_reads": {}, "network": {}}
    for s in SERVICES:
        cid = dc("ps", "-q", s).stdout.strip()
        mounts = json.loads(sh(["docker", "inspect", cid, "--format", "{{json .Mounts}}"]).stdout)
        evidence["mounts"][s] = [{"host": m["Source"].replace(str(ROOT), "<project>"),
                                  "container": m["Destination"], "read_only": not m["RW"]}
                                 for m in mounts]
        evidence["visible_files"][s] = sorted(dc(
            "exec", "-T", s, "sh", "-c", "ls /app/data/processed").stdout.split())
        nets = json.loads(sh(["docker", "inspect", cid, "--format",
                              "{{json .NetworkSettings.Networks}}"]).stdout)
        evidence["network"][s] = sorted(n.replace("hospital-fl_", "") for n in nets)
    for i in range(H):                                   # can hospital i read anyone else's data?
        probe = {}
        for j in range(H):
            if j != i:
                r = dc("exec", "-T", f"hospital-{i}", "sh", "-c",
                       f"test -e /app/data/processed/hospital_{j}.npz && echo READABLE || echo absent")
                probe[f"hospital_{j}.npz"] = r.stdout.strip()
        for f in ("global_test.npz", "pool.npz"):
            r = dc("exec", "-T", f"hospital-{i}", "sh", "-c",
                   f"test -e /app/data/processed/{f} && echo READABLE || echo absent")
            probe[f] = r.stdout.strip()
        r = dc("exec", "-T", f"hospital-{i}", "sh", "-c",
               "touch /app/data/processed/x 2>&1 | head -1; echo done", check=False)
        probe["write_attempt"] = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        evidence["cross_hospital_reads"][f"hospital-{i}"] = probe
        # can hospital i reach another hospital at all?
        reach = {}
        for j in range(H):
            if j != i:
                r = dc("exec", "-T", f"hospital-{i}", "python", "-c",
                       f"import socket;socket.gethostbyname('hospital-{j}')", check=False)
                reach[f"hospital-{j}"] = "reachable" if r.returncode == 0 else "unreachable"
        evidence["network"][f"hospital-{i}_can_reach"] = reach
    server_sees_hospital = [f for f in evidence["visible_files"]["superlink"] if f.startswith("hospital_")]
    own = all(evidence["visible_files"][f"hospital-{i}"] == sorted(["meta.json", f"hospital_{i}.npz"])
              for i in range(H))
    never = all(v == "absent" for p in evidence["cross_hospital_reads"].values()
                for k, v in p.items() if k.endswith(".npz"))
    ro = all(m["read_only"] for s in SERVICES for m in evidence["mounts"][s]
             if m["container"].startswith("/app/data") or m["container"].endswith("config.yaml"))
    nets = all(all(v == "unreachable" for v in evidence["network"][f"hospital-{i}_can_reach"].values())
               for i in range(H))
    evidence["summary"] = {
        "each_hospital_sees_only_its_own_partition_and_meta": own,
        "no_hospital_can_read_another_hospital_or_pooled_or_test_data": never,
        "server_has_no_hospital_partition": not server_sees_hospital,
        "all_data_and_config_mounts_read_only": ro,
        "hospitals_cannot_reach_each_other_on_the_network": nets,
    }
    (OUT / "isolation_check.json").write_text(json.dumps(evidence, indent=2))
    for k, v in evidence["summary"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return evidence


# ---------------------------------------------------------------- collect
def collect(seeds: list, idle: dict, info: dict):
    from src.federated.run import summarize          # same summary layout as the simulation
    cfg = load_config()
    runs = [json.loads((OUT / "federated_runs" / f"seed_{s}.json").read_text()) for s in seeds]
    metas = [json.loads((OUT / "federated_runs" / f"meta_seed_{s}.json").read_text()) for s in seeds]
    for r, m in zip(runs, metas):
        r["flwr_run_wall_seconds"] = m["flwr_run_wall_seconds"]
        r["idle_traffic_per_second"] = idle
        r["container_traffic_bytes"] = m["container_traffic_bytes"]
        r["container_memory_after_run"] = m["container_memory_after_run"]
    if cfg["seed"] in seeds:
        (OUT / "federated.json").write_text(json.dumps(
            next(r for r in runs if r["seed"] == cfg["seed"]), indent=2))
    if len(runs) > 1:
        s = summarize(runs, cfg)
        s["total_wall_seconds"] = _ms([r["total_wall_seconds"] for r in runs])
        s["flwr_run_wall_seconds"] = _ms([r["flwr_run_wall_seconds"] for r in runs])
        s["measured_rx_bytes"] = _ms([r["measured_network"]["rx_bytes_total"] for r in runs])
        s["measured_tx_bytes"] = _ms([r["measured_network"]["tx_bytes_total"] for r in runs])
        s["round_seconds"] = {"mean_per_round": [float(np.mean([r["history"][i]["round_seconds"]
                              for r in runs])) for i in range(1, runs[0]["rounds"] + 1)]}
        s["idle_traffic_per_second"] = idle
        s["environment"] = info
        (OUT / "federated_seeds.json").write_text(json.dumps(s, indent=2))
    (OUT / "environment.json").write_text(json.dumps({**info, "idle_traffic": idle}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", help="default: eval_seeds from config")
    ap.add_argument("--down", action="store_true", help="stop and remove the containers at the end")
    a = ap.parse_args()
    seeds = a.seeds or load_config()["eval_seeds"]
    (OUT / "federated_runs").mkdir(parents=True, exist_ok=True)

    start_stack()
    info = docker_info()
    print("measuring idle traffic (20 s) ...")
    idle = idle_traffic()
    info["idle_memory"] = {"containers": memory_snapshot(),
                           "flower_processes": {s: process_memory(s) for s in SERVICES}}
    print("verifying data isolation ...")
    check_isolation()
    for seed in seeds:
        run_seed(seed)
    collect(seeds, idle, info)
    print("results written to", OUT)
    if a.down:
        dc("down")
