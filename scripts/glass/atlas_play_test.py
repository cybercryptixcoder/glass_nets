"""Test for CHANGES A10 (atlas playground)."""
import sys, os, time, json, subprocess
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import atlas
if not hasattr(atlas, "play"):
    check(False, "atlas has play"); print(f"{len(fails)} FAILED"); sys.exit(1)
t = "zoo_2nd_argmax_8_5"
r = atlas.play(t, seed=1)
check("calls" in r and len(r["calls"]) > 0 and "network_decision" in r, "play returns calls and decisions")
# consistency: play's rebuilt decision equals the batch rebuilt decision
P = atlas._load_play(t); tgt, mods = P["tgt"], P["mods"]
xs = tgt.X[-200:]; batch = atlas.judge_decide(tgt.run_gated(mods, xs, P["gate"]) if hasattr(tgt, "run_gated") else atlas.run_gated(tgt, mods, xs, P["gate"]))
single = [atlas.play(t, x=xs[i, :, 0].tolist())["rebuilt_decision"] for i in range(len(xs))]
check(list(batch) == single, f"play's rebuilt decision equals the batch run on 200 inputs ({sum(int(a == b) for a, b in zip(batch, single))}/200)")
port = "8794"; srv = subprocess.Popen([sys.executable, str(ROOT / "lab/server.py")], env=dict(os.environ, LAB_PORT=port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(3)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True); pg = b.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://localhost:{port}/catalog?t={t}"); pg.wait_for_timeout(1500)
        btn = pg.locator("#atlas-play")
        check(btn.count() == 1, "page has the atlas playground button")
        if btn.count():
            btn.click(); pg.wait_for_timeout(4000)
            check(pg.locator("#atlas-out table tbody tr").count() >= 5, f"per-step table shown ({pg.locator('#atlas-out table tbody tr').count()} rows)")
        check(not errs, f"no JavaScript errors {errs[:1]}"); b.close()
finally:
    srv.terminate()
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
