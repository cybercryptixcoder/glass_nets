"""Test for CHANGES T5 (the catalog shows unit maps). Written before the change; it must fail before and pass after.

  python3 scripts/glass/units/t5_catalog_test.py snapshot   # save the existing cards' verdicts (run BEFORE the change)
  python3 scripts/glass/units/t5_catalog_test.py            # the test
"""
import sys, json, os, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import catalog

SNAP = ROOT / "results/glass/units/t5_entries_before.json"
MAPS = ROOT / "results/glass/units/maps"
KEYS = ("target", "status", "stages", "behaviour")

def verdicts(E):
    return [{k: e.get(k) for k in KEYS} for e in E if e.get("status") != "unit map only"]

if sys.argv[1:] == ["snapshot"]:
    SNAP.write_text(json.dumps(verdicts(catalog.entries()), indent=0)); print("saved", SNAP); sys.exit()

fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)

E = catalog.entries(); by = {e["target"]: e for e in E}
# 1. the designed case and the held-out cases: a card with a unit map whose numbers equal the file
for t, held_out in (("zoo_argmedian_16_5", False), ("zoo_argmedian_16_7", True), ("zoo_2nd_argmax_8_8", True)):
    tag = f"{t}{' (held out)' if held_out else ''}"
    check(t in by, f"{tag}: has a card")
    u = by.get(t, {}).get("unit_map")
    check(bool(u), f"{tag}: card carries a unit map")
    if u:
        m = json.loads((MAPS / f"{t}.json").read_text())
        check(abs(u["eps95"] - m["map"]["eps95"]) < 1e-12 and abs(u["coverage_history"] - m["map"]["coverage_history"]) < 1e-12
              and abs(u["control_history"] - m["control"]["coverage_history"]) < 1e-12, f"{tag}: ε95 and coverage equal the file")
        check(len(u["per_unit"]) == len(m["map"]["per_unit"]) and len(u["cells"]) == len(m["map"]["cells"]), f"{tag}: every neuron and neuron-step present")
# 2. existing cards' verdicts and order unchanged
if SNAP.exists():
    check(verdicts(E) == json.loads(SNAP.read_text()), "existing cards: status, stages, behaviour and order unchanged")
else:
    check(False, "snapshot of the existing cards exists (run with 'snapshot' before the change)")
# 3. end to end in headless Chrome: the page renders the Parts table
port = "8799"; env = dict(os.environ, LAB_PORT=port)
srv = subprocess.Popen([sys.executable, str(ROOT / "lab/server.py")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(3)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True); pg = b.new_page()
        errors = []; pg.on("pageerror", lambda e: errors.append(str(e)))
        for t, n in (("zoo_argmedian_16_5", 16), ("zoo_argmedian_16_7", 16), ("zoo_2nd_argmax_8_8", 8), ("zoo_2nd_argmax_16_10", 16)):
            pg.goto(f"http://localhost:{port}/catalog?t={t}"); pg.wait_for_timeout(1500)
            rows = pg.locator("#parts table.neurons tbody tr").count()
            check(rows == n, f"{t}: page shows the Parts table with {n} neuron rows (got {rows})")
            check(pg.locator("#parts").count() > 0 and "ε95" in pg.locator("#parts").inner_text(), f"{t}: page states ε95")
        check(not errors, f"no JavaScript errors ({errors[:2]})")
        b.close()
finally:
    srv.terminate()
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
