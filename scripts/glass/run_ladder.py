"""The Glass Networks run queue: constructor runs on the AlgZoo ladder, with controls, several at once (one Luminal key
per worker, never two processes on one key). Job: produce the AlgZoo results without losing anything to an
interrupted runtime.

- The queue's state lives in results/glass/queue.json (job -> run id, status, attempts) and is rewritten after every
  change. Every run writes its own folder as it goes (lab/constructions/<target>/<run id>/).
- Restarting this script continues: finished jobs are skipped, and a job that was interrupted is resumed from its
  run folder (constructor.py --resume <run id>), up to 3 attempts.
- After each job, results/glass/RESULTS.md is regenerated (scripts/glass/results.py).

Run:  python3 scripts/glass/run_ladder.py [--workers 4] [--only job ...] [--dry]
Keys: LUMINAL_API_KEYS (comma-separated) in lab/.env.
"""
import argparse, json, os, re, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab"
RES = ROOT / "results" / "glass"
STATE = RES / "queue.json"
LOGS = RES / "logs"
LADDER = ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_2nd_argmax_4_4", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_8_8",
          "zoo_2nd_argmax_16_10"]
MAIN_STEPS, SHUFFLED_STEPS = 80, 30


def jobs():
    """Main blinded runs first, then the controls. Each job: (name, target, constructor args). The main runs start
    with the two smallest, then ARC's challenge network (the most informative, and the longest), then the rest."""
    first = ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_2nd_argmax_16_10"]
    out = [(f"{t}__main", t, ["--steps", str(MAIN_STEPS)]) for t in first + [t for t in LADDER if t not in first]]
    out += [(f"{t}__prior", t, ["--told", "--no-weights", "--one-shot"]) for t in LADDER]
    out += [(f"{t}__shuffled", t, ["--variant", "shuffled", "--steps", str(SHUFFLED_STEPS)]) for t in LADDER]
    return out


_lock = threading.Lock()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def keys():
    env = {}
    for line in (LAB / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    ks = [k for k in env.get("LUMINAL_API_KEYS", env.get("LUMINAL_API_KEY", "")).split(",") if k.strip()]
    if not ks:
        sys.exit("no Luminal keys in lab/.env")
    return ks


def run_status(target, run_id):
    p = LAB / "constructions" / target / run_id / "run.json"
    if not p.exists():
        return None, None
    rec = json.loads(p.read_text())
    rep = LAB / "constructions" / target / run_id / "report.json"
    v = json.loads(rep.read_text())["verdict"] if rep.exists() else None
    return rec.get("status"), v


def run_job(name, target, args, key, st):
    LOGS.mkdir(parents=True, exist_ok=True)
    with _lock:
        j = st.setdefault(name, {"target": target, "args": args, "status": "pending", "attempts": 0, "run_id": None})
        j["status"], j["attempts"], j["started"] = "running", j["attempts"] + 1, time.time()
        save_state(st)
        rid = j["run_id"]
    cmd = [sys.executable, str(LAB / "constructor.py"), target, *args, "--quiet"]
    if rid:
        cmd += ["--resume", rid]
    env = dict(os.environ, LUMINAL_API_KEY=key, PYTHONUNBUFFERED="1")
    with open(LOGS / f"{name}.log", "a") as log:
        log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {j['attempts']}: {' '.join(cmd[1:])}\n"); log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line); log.flush()
            m = re.match(r"\[start\] (\{.*\})", line.strip())
            if m and not rid:
                rid = json.loads(m.group(1))["id"]
                with _lock:
                    j["run_id"] = rid; save_state(st)
        p.wait()
    status, v = run_status(target, rid) if rid else (None, None)
    with _lock:
        j["exit_code"], j["finished"] = p.returncode, time.time()
        if status in ("done", "stopped") and p.returncode == 0:
            j["status"] = "done"
        else:
            j["status"] = "failed" if j["attempts"] >= 3 else "interrupted"
        j["run_status"], j["verdict"] = status, v
        save_state(st)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "glass" / "results.py")], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    st = load_state()
    for j in st.values():                       # a previous copy of this script died mid-job: resume those jobs
        if j["status"] == "running":
            j["status"] = "interrupted"
    save_state(st)
    todo = [x for x in jobs() if (not a.only or x[0] in a.only) and st.get(x[0], {}).get("status") not in ("done", "failed")]
    print(f"{len(todo)} jobs to run: " + ", ".join(n for n, _, _ in todo), flush=True)
    if a.dry:
        return
    ks = keys()
    nw = min(a.workers, len(ks))
    queue = list(todo)

    def worker(i):
        while True:
            with _lock:
                if not queue:
                    return
                name, target, args = queue.pop(0)
            print(f"{time.strftime('%H:%M:%S')} worker {i} starts {name}", flush=True)
            try:
                run_job(name, target, args, ks[i], st)
            except Exception as e:  # noqa: BLE001  (one job's crash must not stop the queue)
                print(f"worker {i}: {name} crashed: {e}", flush=True)
                with _lock:
                    st[name]["status"] = "interrupted"; save_state(st)
            print(f"{time.strftime('%H:%M:%S')} worker {i} finished {name}: {st[name]['status']}", flush=True)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(nw)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    left = [n for n, j in st.items() if j["status"] != "done"]
    print("== all jobs done" if not left else f"== queue empty; not done: {', '.join(left)}", flush=True)


if __name__ == "__main__":
    main()
