"""Memory profile of one ClientApp process, run INSIDE a hospital container image.

Flower starts a fresh `flwr-clientapp` process for every message, so a training message and an
evaluation message are measured in separate processes, each doing exactly what the ClientApp does
(same imports, same functions). Prints one JSON line.

Run it in a throwaway container with the same limits and mounts as hospital 1 (the largest):

    docker run --rm --cpus 1.0 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e FL_ROOT=/app \\
      -v $PWD/src:/app/src:ro -v $PWD/docker/measure_memory.py:/app/measure_memory.py:ro \\
      -v $PWD/configs/config.yaml:/app/configs/config.yaml:ro \\
      -v $PWD/data/processed/meta.json:/app/data/processed/meta.json:ro \\
      -v $PWD/data/processed/hospital_1.npz:/app/data/processed/hospital_1.npz:ro \\
      --entrypoint python hospital-fl-supernode:1.42.0 /app/measure_memory.py train 1

(and `evaluate 1`). Results before and after slimming the ClientApp: results/docker/memory_clientapp.json.
"""
import json
import sys


def mem() -> dict:
    """Current resident memory, its anonymous (heap) part, and the peak so far, in MB."""
    f = {l.split(":")[0]: int(l.split()[1]) // 1024
         for l in open("/proc/self/status") if l.startswith(("VmRSS", "RssAnon", "VmHWM"))}
    return {"rss_mb": f["VmRSS"], "anon_mb": f["RssAnon"], "peak_mb": f["VmHWM"]}


def main(kind: str, h: int) -> dict:
    stages = {"python started": mem()}
    import flwr.app, flwr.clientapp                          # noqa: F401  what flwr-clientapp loads
    stages["flwr imported"] = mem()
    from src.federated import client_app as ca               # the ClientApp module and its imports
    stages["ClientApp module imported"] = mem()
    cfg = ca.load_config()
    if kind == "train":
        X, y, _, _ = ca._data(h)
        stages["own partition loaded"] = mem()
        model = ca.build_model(ca.get_input_dim(), cfg)
        t = cfg["training"]
        ca.train(model, X, y, epochs=1, lr=t["learning_rate"], batch_size=t["batch_size"],
                 cfg=cfg, seed=1)
        stages["1 local epoch trained"] = mem()
    else:
        _, _, Xv, yv = ca._data(h)
        stages["own partition loaded"] = mem()
        model = ca.build_model(ca.get_input_dim(), cfg)
        p = ca.predict_proba(model, Xv)
        ca.compute_metrics(yv, p, cfg["training"]["threshold"])
        ca.bootstrap_auc_ci(yv, p)
        stages["validation metrics computed"] = mem()
    loaded = sorted(m for m in ("pandas", "sklearn", "scipy", "matplotlib", "seaborn", "joblib")
                    if m in sys.modules)
    return {"message": kind, "hospital": h, "stages": stages,
            "peak_mb": stages[list(stages)[-1]]["peak_mb"], "heavy_modules_loaded": loaded}


if __name__ == "__main__":
    sys.path.insert(0, "/app")
    print(json.dumps(main(sys.argv[1], int(sys.argv[2]))))
