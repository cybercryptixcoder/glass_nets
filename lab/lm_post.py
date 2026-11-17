"""After a constructor run on an LM target: the analyses that turn a verified construction into findings.

  1. circuit   which heads and MLPs write each variable of the verified cuts (lab/lm_circuit.py)
  2. transfer  the construction, unchanged, on every transfer task the task file declares (TRANSFER = [...] in
               lab/lm_tasks/<file>.py; lab/lm_transfer.py): its first checkable claim beyond its own prompts
Results go into the run folder (circuit.json/.txt, transfer_<id>.json/.txt, post.json). Called by constructor.py
when an LM run ends with at least one verified stage; run it by hand with  python3 lab/lm_post.py <target> <run id>.
"""
import json, sys, time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))


def post(tid, rid, log=print):
    import lm_circuit, lm_transfer
    rdir = LAB / "constructions" / tid / rid
    meta = json.loads((LAB / "models" / tid / "meta.json").read_text())
    done = {"started": time.time(), "steps": []}
    src, _ = lm_transfer.source_of(tid, rid)
    for t2 in meta.get("transfer", []):
        if not (LAB / "models" / t2 / "meta.json").exists():
            log(f"transfer task {t2} is not built (python3 lab/lm_build.py lab/lm_tasks/<its file>.py); skipped")
            continue
        log(f"transfer to {t2} ...")
        rep = lm_transfer.transfer(tid, src, t2)
        (rdir / f"transfer_{t2}.json").write_text(json.dumps(rep, indent=1))
        (rdir / f"transfer_{t2}.txt").write_text(lm_transfer.render(rep) + "\n")
        done["steps"].append({"transfer": t2, "behaviour": rep["behaviour"]["match"]})
        log(lm_transfer.render(rep))
    log("circuit ...")
    res = lm_circuit.for_run(tid, rid)
    done["steps"].append({"circuit": len(res.get("variables", []))})
    import lm_writeup
    done["findings"] = str(lm_writeup.writeup(tid, rid))
    done["seconds"] = round(time.time() - done["started"], 1)
    (rdir / "post.json").write_text(json.dumps(done, indent=1))
    return done


if __name__ == "__main__":
    post(sys.argv[1], sys.argv[2])
