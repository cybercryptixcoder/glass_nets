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
RES = ROOT / "results" / "glass" / os.environ.get("GLASS_ROUND", "")   # round 2: results/glass/round2
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


EXTEND = {"zoo_2nd_argmax_16_10": 160}     # networks whose pass-1 run is continued (same run, its notes kept) rather
                                         # than restarted: ARC's challenge network, where 80 steps is clearly short


def pass2_jobs(st):
    out = []
    for t in LADDER:
        j = st.get(f"{t}__main")
        if j and j["status"] == "done" and not certified_now(t, j.get("run_id")):
            if t in EXTEND:
                st[f"{t}__main2"] = {"target": t, "args": ["--steps", str(EXTEND[t])], "status": "pending", "attempts": 0,
                                     "run_id": j["run_id"], "note": "continues the pass-1 run"}
                out.append((f"{t}__main2", t, ["--steps", str(EXTEND[t])]))
            else:
                out.append((f"{t}__main2", t, ["--steps", str(MAIN_STEPS)]))
    return out


def jobs_round2():
    """Round 2 (docs/glass/PLAN_ROUND2.md): the longest-cycle discovery runs (reader v2) interleaved with the A/B on
    the 2nd-argmax ladder (round 1's reader v1 vs v2, two runs each, same budgets). Each job: (name, target, args, reader)."""
    lc = [("lc_3_4", 80), ("lc_4_5", 80), ("lc_6_5", 120), ("lc_8_6", 120)]
    zoo = [(t, 160 if t.endswith("16_10") else 80) for t in LADDER]
    J = []
    def ab(t, steps, rep):
        return [(f"{t}__{r}{rep}", t, ["--steps", str(steps)], r) for r in ("v1", "v2")]
    J += [(f"lc_3_4__main{k}", "lc_3_4", ["--steps", "80"], "v2") for k in "ab"]
    J += ab(*zoo[0], "a") + ab(*zoo[1], "a")
    J += [(f"{t}__main{k}", t, ["--steps", str(st)], "v2") for t, st in lc[1:3] for k in "ab"]
    J += ab(*zoo[2], "a") + ab(*zoo[3], "a")
    J += [(f"lc_8_6__main{k}", "lc_8_6", ["--steps", "120"], "v2") for k in "ab"]
    J += ab(*zoo[4], "a") + ab(*zoo[5], "a")
    for t, st in zoo:
        J += ab(t, st, "b")
    for t, _ in lc:
        J += [(f"{t}__prior", t, ["--told", "--no-weights", "--one-shot"], "v2"),
              (f"{t}__shuffled", t, ["--variant", "shuffled", "--steps", "30"], "v2")]
    return J


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


def run_job(name, target, args, key, st, reader="v2"):
    LOGS.mkdir(parents=True, exist_ok=True)
    with _lock:
        j = st.setdefault(name, {"target": target, "args": args, "status": "pending", "attempts": 0, "run_id": None})
        j["status"], j["attempts"], j["started"] = "running", j["attempts"] + 1, time.time()
        save_state(st)
        rid = j["run_id"]
    cmd = [sys.executable, str(LAB / "constructor.py"), target, *args, "--quiet"]
    if rid:
        cmd += ["--resume", rid]
    env = dict(os.environ, LUMINAL_API_KEY=key, PYTHONUNBUFFERED="1", GLASS_READER=reader)
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
    subprocess.run([sys.executable, str(ROOT / "scripts" / "glass" / "results.py")], cwd=ROOT, env=dict(os.environ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--pass2", action="store_true")
    ap.add_argument("--round2", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    st = load_state()
    for j in st.values():                       # a previous copy of this script died mid-job: resume those jobs
        if j["status"] == "running":
            j["status"] = "interrupted"
    save_state(st)
    if a.round2:
        base = jobs_round2()
    else:
        new2 = [x for x in pass2_jobs(st) if st.get(x[0], {}).get("status") in (None, "pending")] if a.pass2 else []
        base = [(*x, "v1") for x in jobs() + new2 + [x for x in jobs2(st) if x[0] not in {n for n, _, _ in new2}]]
    todo = [x for x in base if (not a.only or x[0] in a.only) and st.get(x[0], {}).get("status") not in ("done", "failed")]
    print(f"{len(todo)} jobs to run: " + ", ".join(x[0] for x in todo), flush=True)
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
                name, target, args, reader = queue.pop(0)
            print(f"{time.strftime('%H:%M:%S')} worker {i} starts {name}", flush=True)
            try:
                run_job(name, target, args, ks[i], st, reader)
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
