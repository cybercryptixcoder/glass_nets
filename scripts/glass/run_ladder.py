"""The Glass Networks run queue: constructor runs on the AlgZoo ladder, with controls, several at once (one Luminal key
per worker, never two processes on one key). Job: produce the AlgZoo results without losing anything to an
interrupted runtime.

- The queue's state lives in results/glass/queue.json (job -> run id, status, attempts) and is rewritten after every
  change. Every run writes its own folder as it goes (lab/constructions/<target>/<run id>/).
- Restarting this script continues: finished jobs are skipped, and a job that was interrupted is resumed from its
  run folder (constructor.py --resume <run id>), up to 3 attempts.
- After each job, results/glass/RESULTS.md is regenerated (scripts/glass/results.py).

Pass 2 (--pass2): every network whose first main run did not end certified (VERIFIED AND COMPLETE under the current
judge, re-judged here) gets one more blinded main run, "<target>__main2". Job: a second chance after judge changes
made during pass 1 (the copy rules), and a first measure of run-to-run variation.

Run:  python3 scripts/glass/run_ladder.py [--workers 4] [--only job ...] [--dry] [--pass2]
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


def certified_now(target, run_id):
    """Re-judge a run's construction with the current judge (writing report_current.*); True if complete."""
    rdir = LAB / "constructions" / target / (run_id or "_")
    if not (rdir / "construction.py").exists():
        return False
    sys.path.insert(0, str(LAB))
    import judge
    meta, W, X = judge.load_target(target)
    try:
        rep = judge.judge(meta["forward"], W, X, (rdir / "construction.py").read_text(), groups=meta.get("out_groups"),
                          labels=judge.load_labels(meta))
    except judge.ConstructionError:
        return False
    (rdir / "report_current.json").write_text(json.dumps(rep, indent=1, default=str))
    (rdir / "report_current.txt").write_text(judge.render(rep) + "\n")
    return bool(rep["verdict"]["complete"])


def pass2_jobs(st):
    out = []
    for t in LADDER:
        j = st.get(f"{t}__main")
        if j and j["status"] == "done" and not certified_now(t, j.get("run_id")):
            out.append((f"{t}__main2", t, ["--steps", str(MAIN_STEPS)]))
    return out


def jobs():
    """Main blinded runs first, then the controls. Each job: (name, target, constructor args). The main runs start
    with the two smallest, then ARC's challenge network (the most informative, and the longest), then the rest."""
    first = ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_2nd_argmax_16_10"]
    out = [(f"{t}__main", t, ["--steps", str(MAIN_STEPS)]) for t in first + [t for t in LADDER if t not in first]]
    out += [(f"{t}__prior", t, ["--told", "--no-weights", "--one-shot"]) for t in LADDER]
    out += [(f"{t}__shuffled", t, ["--variant", "shuffled", "--steps", str(SHUFFLED_STEPS)]) for t in LADDER]
    return out


def jobs2(st):
    """Pass-2 jobs already recorded in the state (so a restart resumes them without re-deciding)."""
    return [(n, j["target"], j["args"]) for n, j in st.items() if n.endswith("__main2")]


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
    ap.add_argument("--pass2", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    st = load_state()
    for j in st.values():                       # a previous copy of this script died mid-job: resume those jobs
        if j["status"] == "running":
            j["status"] = "interrupted"
    save_state(st)
    base = jobs() + ([x for x in pass2_jobs(st) if x[0] not in st] + [x for x in jobs2(st)] if a.pass2 else [x for x in jobs2(st)])
    todo = [x for x in base if (not a.only or x[0] in a.only) and st.get(x[0], {}).get("status") not in ("done", "failed")]
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
