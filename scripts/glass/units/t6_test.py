"""Test for CHANGES T6 (replaceable parts: unit map v4 + catalog). Fails before, passes after."""
import sys, json, os, time, subprocess, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
u18 = {}
for f in glob.glob(str(ROOT / "results/glass/units/u18_*.json")): u18.update(json.load(open(f)))
before = json.load(open(ROOT / "results/glass/units/t6_maps_before.json"))
for t, held in (("zoo_argmedian_16_5", False), ("zoo_argmedian_16_7", True), ("zoo_2nd_argmax_8_8", True)):
    m = json.load(open(ROOT / f"results/glass/units/maps/{t}.json"))["map"]; r = m.get("replaceable"); tag = t + (" (held out)" if held else "")
    check(bool(r) and {"n", "share_after_step1", "joint_fidelity", "task_rule"} <= set(r or {}), f"{tag}: map has 'replaceable'")
    if r:
        check(abs(r["joint_fidelity"] - u18[t]["fidelity"]) < 1e-9 and abs(r["share_after_step1"] - u18[t]["share_after_step1"]) < 1e-9, f"{tag}: joint fidelity and share equal U18")
for t, b in before.items():
    m = json.load(open(ROOT / f"results/glass/units/maps/{t}.json"))["map"]
    same = all(abs(m[k] - b[k]) < 1e-12 for k in ("coverage_history", "coverage_raw", "eps95") if b[k] is not None) and m["per_unit"] == b["per_unit"] and len(m["cells"]) == b["n_cells"]
    check(same, f"{t}: v3 fields unchanged")
port = "8797"; srv = subprocess.Popen([sys.executable, str(ROOT / "lab/server.py")], env=dict(os.environ, LAB_PORT=port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(3)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True); pg = b.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        for t in ("zoo_argmedian_16_5", "zoo_argmedian_16_7"):
            pg.goto(f"http://localhost:{port}/catalog?t={t}"); pg.wait_for_timeout(1500)
            txt = pg.locator("#parts").inner_text() if pg.locator("#parts").count() else ""
            check("Replaceable" in txt and "after step 1" in txt, f"{t}: page states the replaceable parts with the share after step 1")
        check(not errs, f"no JavaScript errors {errs[:1]}"); b.close()
finally:
    srv.terminate()
r = subprocess.run([sys.executable, str(ROOT / "scripts/glass/units/t5_catalog_test.py")], capture_output=True, text=True)
check(r.returncode == 0, "T5 test still passes")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
