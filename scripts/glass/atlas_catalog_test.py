"""Test for CHANGES A4 (atlas certificate + catalog). Fails before, passes after."""
import sys, json, os, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import catalog
E = {e["target"]: e for e in catalog.entries()}
targets = [t for t in ("zoo_2nd_argmax_8_5", "zoo_argmedian_8_5") if (ROOT / f"results/glass/atlas/{t}_d4.json").exists()]
for t in targets:
    a = json.loads((ROOT / f"results/glass/atlas/{t}_d4.json").read_text()); at = E.get(t, {}).get("atlas")
    check(bool(at), f"{t}: card carries the atlas")
    if at:
        check(abs(at["fidelity"]["rule_gated_trusted"] - a["fidelity"]["rule_gated_trusted"]) < 1e-12 and
              sum(len(m["mechanisms"]) for m in at["matrices"].values()) == sum(len(m["mechanisms"]) for m in a["matrices"].values()),
              f"{t}: atlas numbers equal the JSON")
    check((ROOT / f"results/glass/atlas/{t}_d4.md").exists(), f"{t}: markdown certificate exists")
port = "8795"; srv = subprocess.Popen([sys.executable, str(ROOT / "lab/server.py")], env=dict(os.environ, LAB_PORT=port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(3)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True); pg = b.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        for t in targets:
            a = json.loads((ROOT / f"results/glass/atlas/{t}_d4.json").read_text())
            n = sum(sum(x["trigger"]["rule"] != "always" for x in m["mechanisms"]) + any(x["trigger"]["rule"] == "always" for x in m["mechanisms"]) for m in a["matrices"].values())   # A34
            pg.goto(f"http://localhost:{port}/catalog?t={t}"); pg.wait_for_timeout(1500)
            rows = pg.locator("#atlas table.mechs tbody tr").count()
            check(rows == n, f"{t}: page shows one row per switched mechanism plus one per matrix with always-on ones ({rows} of {n})")
        check(not errs, f"no JavaScript errors {errs[:1]}"); b.close()
finally:
    srv.terminate()
for test in ("scripts/glass/units/t5_catalog_test.py", "scripts/glass/units/t6_test.py"):
    r = subprocess.run([sys.executable, str(ROOT / test)], capture_output=True, text=True); check(r.returncode == 0, f"{test} still passes")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
