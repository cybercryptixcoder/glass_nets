"""Regenerate docs/figures/month_line_vs_circle.png from lab/models/<target>/line_vs_circle.json (lab/line_vs_circle.py)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
models = [("gpt2_month_rollover", "GPT-2 small", 8), ("gpt2medium_month_rollover", "GPT-2 medium", 12), ("pythia160m_month_rollover", "Pythia-160m", 8)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
M = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
cols = ["#2a6f97", "#e07a5f", "#3d8b5f"]
for (t, name, d), c in zip(models, cols):
    r = json.load(open(f"lab/models/{t}/line_vs_circle.json"))
    u, v = np.array(r["line_positions"]["month1"]), np.array(r["line_positions"]["month2"])
    ax[0].plot(range(12), u / np.abs(u).max(), "-o", color=c, label=f"{name}: month1", ms=4)
    ax[0].plot(range(12), v / np.abs(v).max(), "--s", color=c, label=f"{name}: month2", ms=3, alpha=.7)
ax[0].set_xticks(range(12)); ax[0].set_xticklabels(M, fontsize=8); ax[0].axhline(0, color="#999", lw=.5)
ax[0].set_title("Each month's effect on 'next year' (normalised)", fontsize=10); ax[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.32))
kinds = [("line", "calendar line (fitted)"), ("calendar index", "calendar index (unfitted)"), ("random month code, mean", "unordered month code"), ("circle", "circle (cos, sin)"), ("random 2-D", "random plane")]
w = 0.16
for i, (t, name, d) in enumerate(models):
    r = json.load(open(f"lab/models/{t}/line_vs_circle.json"))["ablations"][f"month1 @ depth {d}"]
    for j, (k, lab) in enumerate(kinds):
        key = k + " margin_r2" if not k.startswith("random month code") else "random month code, mean margin_r2"
        ax[1].bar(i + (j - 2) * w, r[key], w, color=["#c1121f", "#e5989b", "#adb5bd", "#457b9d", "#ced4da"][j], label=lab if i == 0 else None)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"{n} (depth {d})" for _, n, d in models], fontsize=8)
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("share of the margin's variation kept", fontsize=9)
ax[1].set_title("Remove one code of month1: what survives?", fontsize=10)
ax[1].legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("docs/figures/month_line_vs_circle.png", dpi=150, bbox_inches="tight")
