"""
Builds the result figures in figures/ from the result files in this repository (no models or hidden states needed).
Run from ROOT:   python repro/analysis/make_figures.py
Each figure has a matching .md in figures/ explaining it.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("LDG_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results_sweep"; AO = ROOT / "repro" / "analysis_out"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "tinyllama-1.1b", "olmo2-1b"]
NAME = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B",
        "tinyllama-1.1b": "TinyLlama-1.1B", "olmo2-1b": "OLMo-2-1B"}
SELFILE = {"qwen2.5-0.5b": "qwen2.5-0.5b.json", "qwen2.5-1.5b": "qwen2.5-1.5b.json", "qwen2.5-3b": "qwen2.5-3b.json",
           "qwen2.5-7b": "qwen2.5-7b.json", "tinyllama-1.1b": "tinyllama.json", "olmo2-1b": "olmo.json"}
SHORT = {"qwen2.5-0.5b": "Qwen-0.5B", "qwen2.5-1.5b": "Qwen-1.5B", "qwen2.5-3b": "Qwen-3B", "qwen2.5-7b": "Qwen-7B", "tinyllama-1.1b": "TinyLlama", "olmo2-1b": "OLMo-2"}
HELDOUT = ["qwen2.5-3b", "qwen2.5-7b", "tinyllama-1.1b", "olmo2-1b"]
C = {"paper": "#C8642A", "als": "#2F6FB2", "pis": "#6D2E46", "mid": "#9AA7B8", "last": "#4A5568", "best": "#14213D", "gold": "#E8A262"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
log = []

def per_layer(m, feat="final"):
    return sorted([r for r in json.load(open(RES / m / "per_layer.json")) if r["feat"] == feat], key=lambda r: r["layer"])
def id_curve(rows):
    return np.array([np.mean(r["id_benign_boots"]) if r.get("id_benign_boots") else r["id_benign"] for r in rows]), \
           np.array([np.std(r["id_benign_boots"], ddof=1) if len(r.get("id_benign_boots") or []) > 1 else 0 for r in rows])
def sel(m):
    return json.load(open(AO / "selection" / SELFILE[m]))[m if m != "olmo2-1b" else list(json.load(open(AO / "selection" / SELFILE[m])))[0]]
def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); plt.close(fig); log.append(name); print("wrote", name)

# ---------------------------------------------------------------- 1. ID curves vs detection, all models
def fig1():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m in zip(axes.flat, MODELS):
        R = per_layer(m); idc, se = id_curve(R); au = np.array([r["squad_auroc_mean"] for r in R]); x = np.arange(1, len(R) + 1)
        pk = sel(m)["final"]["picks"]; pa, ad = pk["id:strict"]["layer"], pk["id:neartop(eps=0.1)"]["layer"]; best = int(np.argmax(au)) + 1
        ax.fill_between(x, idc - se, idc + se, color=C["als"], alpha=.18, lw=0); ax.plot(x, idc, color=C["als"], lw=2.2, label="Intrinsic dimension (benign prompts)")
        ax.set_xlabel("Layer"); ax.set_ylabel("Intrinsic dimension (TwoNN)", color=C["als"])
        a2 = ax.twinx(); a2.plot(x, au, color=C["best"], lw=1.5, ls="--", label="Probe AUROC on SQuAD"); a2.set_ylabel("Detection AUROC (SQuAD)"); a2.spines["top"].set_visible(False)
        for L, col, lab, y in ((pa, C["paper"], "ALS paper", .98), (ad, C["als"], "ALS adapted", .16), (best, C["best"], "Best", .07)):
            ax.axvline(L, color=col, lw=1.5, ls=":" if col == C["best"] else "-")
            ax.text(L, y, f" {lab} L{L}", color=col, fontsize=8, fontweight="bold", va="top", transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
        ax.set_title(NAME[m] + ("  (held out)" if m in HELDOUT else ""))
    h1, l1 = axes.flat[0].get_legend_handles_labels(); h2, l2 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("ALS: the intrinsic-dimension curve of benign prompts vs. where injection detection is best", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, .04, 1, 1)); save(fig, "fig1_id_curves_vs_detection")

# ---------------------------------------------------------------- 2. regret of each layer-choice rule, SQuAD and BIPIA
RULES = [("id:strict", "ALS, paper rule", C["paper"]), ("id:neartop(eps=0.1)", "ALS, adapted", C["als"]), ("pishield:val_acc", "PIShield (validation)", C["pis"]),
         ("fixed:0.5 depth", "Fixed mid-depth", C["mid"]), ("last", "Last layer", C["last"])]
def bipia_regret(m, feat, layer):
    b = json.load(open(AO / "bench_summary.json"))[m][feat]; pl = b["per_layer_mean_auroc"]
    pl = pl["std"] if isinstance(pl, dict) else pl
    return float(b["oracle_bench_auroc"] - pl[layer - 1])
def fig2():
    data = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, bench in zip(axes, ("SQuAD", "BIPIA")):
        w = .16; x = np.arange(len(MODELS))
        for i, (key, lab, col) in enumerate(RULES):
            vals = []
            for m in MODELS:
                p = sel(m)["final"]["picks"][key]
                vals.append(p["regret_squad"] if bench == "SQuAD" else bipia_regret(m, "final", p["layer"]))
            data[(bench, lab)] = vals
            ax.bar(x + (i - 2) * w, vals, w, color=col, label=lab)
        ax.set_xticks(x, [SHORT[m] for m in MODELS]); ax.set_title(f"Test domain: {bench}")
    axes[0].set_ylabel("Regret (AUROC lost vs. best layer, lower is better)")
    axes[0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.16), fontsize=9)
    fig.suptitle("Which layer to read? Regret of each layer-choice rule (last-token feature)", fontweight="bold", y=1.08)
    save(fig, "fig2_layer_choice_regret")
    return data

# ---------------------------------------------------------------- 3. cutoff sensitivity
def fig3():
    s = json.load(open(AO / "als_checks" / "sensitivity.json")); eps = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    out = {}
    for ax, metric, title in zip(axes, ("squad_regret", "bipia_regret"), ("SQuAD", "BIPIA")):
        for sub, lab, ls in ((list(s), "All 12 runs", "-"), ([k for k in s if k.split("/")[0] in HELDOUT], "8 held-out runs", "--")):
            y = [np.mean([s[k][f"eps={e:.2f}"][metric] for k in sub]) for e in eps]; p = np.mean([s[k]["paper"][metric] for k in sub])
            ax.plot([e * 100 for e in eps], y, marker="o", color=C["als"], ls=ls, lw=2, label=f"ALS adapted, {lab.lower()}")
            ax.axhline(p, color=C["paper"], ls=ls, lw=1.3, label=f"ALS paper rule, {lab.lower()}"); out[(title, lab)] = (y, p)
        ax.axvline(10, color="#999", lw=.8, ls=":"); ax.text(10.4, .135, "frozen rule (10%)", fontsize=8, color="#666")
        ax.set_xlabel("Cutoff: peak must be within this % of the curve's maximum"); ax.set_title(f"Mean regret on {title}")
    axes[0].set_ylabel("Mean regret (lower is better)"); axes[0].legend(fontsize=8.5, loc="center right", bbox_to_anchor=(1, .62))
    fig.suptitle("Is the 10% cutoff tuned? Regret stays low from 2% to 10%", fontweight="bold", y=1.02)
    save(fig, "fig3_cutoff_sensitivity"); return out

# ---------------------------------------------------------------- 4. ID vs detection correlation, with and without depth
def fig4():
    c = json.load(open(AO / "id_auroc_correlation.json")); keys = list(c)
    fig, ax = plt.subplots(figsize=(13, 4.4)); x = np.arange(len(keys)); w = .27
    ax.bar(x - w, [c[k]["rho_id_squad"] for k in keys], w, color=C["als"], label="ID vs AUROC (Spearman)")
    ax.bar(x, [c[k]["rho_depth_squad"] for k in keys], w, color=C["mid"], label="Depth vs AUROC")
    ax.bar(x + w, [c[k]["partial_squad"] for k in keys], w, color=C["best"], label="ID vs AUROC, depth removed (partial)")
    ax.axhline(0, color="#333", lw=.8); ax.set_ylim(-.2, 1)
    ax.set_xticks(x, [f"{SHORT[k.split('/')[0]]}\n{k.split('/')[1]}" for k in keys], fontsize=9)
    ax.set_ylabel("Rank correlation across layers (SQuAD)"); ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title("Layers with higher intrinsic dimension detect injections better, even after accounting for depth", pad=30)
    save(fig, "fig4_id_vs_detection_correlation")

# ---------------------------------------------------------------- 5. threshold collapse and recalibration
def fig5():
    rows = []
    for m in MODELS:
        s = sel(m)["final"]; L = s["picks"]["id:neartop(eps=0.1)"]["layer"]
        src_sq = next(r for r in per_layer(m) if r["layer"] == L)["squad_fpr_benign"]
        tgt_sq = next(v for k, v in s["recal"].items() if k.startswith("id:neartop"))["fpr_benign"]
        b = json.load(open(AO / "bench_summary.json"))[m]["final"]["rules"]["id:neartop(eps=0.1)"]["std"]
        src_bi = np.mean([b[d]["fpr_src"] for d in ("email", "table", "code")]); tgt_bi = np.mean([b[d]["fpr_tgt"] for d in ("email", "table", "code")])
        tpr_bi = np.mean([b[d]["tpr_tgt"] for d in ("email", "table", "code")])
        rows.append((m, L, src_sq, tgt_sq, src_bi, tgt_bi, tpr_bi))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True); x = np.arange(len(rows)); w = .38
    for ax, (i, j), title in zip(axes, ((2, 3), (4, 5)), ("SQuAD", "BIPIA (mean of email, table, code)")):
        ax.bar(x - w / 2, [r[i] * 100 for r in rows], w, color=C["paper"], label="Threshold set on metro (source)")
        ax.bar(x + w / 2, [r[j] * 100 for r in rows], w, color=C["als"], label="Threshold re-set on ~50 clean target documents")
        for k, r in enumerate(rows):
            ax.text(k - w / 2, r[i] * 100 + 1, f"{r[i]*100:.0f}", ha="center", fontsize=8); ax.text(k + w / 2, r[j] * 100 + 1, f"{r[j]*100:.0f}", ha="center", fontsize=8)
        ax.axhline(5, color="#333", lw=.8, ls=":"); ax.text(-.6, 6.5, "target 5%", fontsize=8, ha="left")
        ax.set_xticks(x, [f"{SHORT[r[0]]}\nL{r[1]}" for r in rows], fontsize=9); ax.set_title(title)
    axes[0].set_ylabel("False alarms on clean documents (%)"); axes[0].legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, 1.0))
    fig.suptitle("Finding 2: a detection threshold does not survive a new domain; re-setting it on clean documents fixes it", fontweight="bold", y=1.02)
    save(fig, "fig5_threshold_collapse"); return rows

# ---------------------------------------------------------------- 6. poisoned recalibration
def fig6():
    c = json.load(open(AO / "contamination_results.json")); es = [0.0, 0.02, 0.05, 0.1, 0.2]
    rl = [("naive", "Quantile (naive)", C["paper"], "-"), ("conservative", "Quantile (conservative)", C["gold"], "-"),
          ("trimmed", "Quantile (trimmed)", C["mid"], "-"), ("robust_affine", "Robust median/MAD", C["als"], "-")]
    ms = [m for m in MODELS if m in c]
    fig, axes = plt.subplots(2, len(ms), figsize=(4 * len(ms), 6.6), sharex=True)
    for j, m in enumerate(ms):
        for key, lab, col, ls in rl:
            axes[0, j].plot([e * 100 for e in es], [c[m][f"decoy|{e}"][key][1] * 100 for e in es], marker="o", color=col, lw=2, label=lab)
            axes[1, j].plot([e * 100 for e in es], [c[m][f"decoy|{e}"][key][0] * 100 for e in es], marker="o", color=col, lw=2, label=lab)
        axes[0, j].set_title(NAME[m]); axes[1, j].set_xlabel("Decoys planted in the calibration pool (%)")
        axes[0, j].set_ylim(-3, 100); axes[1, j].set_ylim(-1, 45); axes[1, j].axhline(5, color="#333", lw=.8, ls=":")
    axes[0, 0].set_ylabel("Attacks detected (%)"); axes[1, 0].set_ylabel("False alarms on clean docs (%)")
    axes[0, 0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Finding 3: an attacker who plants decoys in the clean pool can switch the detector off; a robust rule resists", fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "fig6_poisoned_recalibration")

# ---------------------------------------------------------------- 7. NotInject false alarms: trade-off of NotInject-aware training
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, sharex=True); out = []
    for ax, feat in zip(axes, ("final", "delta")):
        for k, m in enumerate(MODELS):
            L = sel(m)[feat]["picks"]["id:neartop(eps=0.1)"]["layer"]
            rows = json.load(open(AO / "notinject" / m / "ni_per_layer.json"))
            g = {r["variant"]: r for r in rows if r["feat"] == feat and r["layer"] == L}
            a, b = g["std"], g["ni_hard"]; col = plt.cm.tab10(k)
            ax.annotate("", xy=(b["ni_fpr_tgt"] * 100, b["email_tpr_tgt"] * 100), xytext=(a["ni_fpr_tgt"] * 100, a["email_tpr_tgt"] * 100),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.scatter([a["ni_fpr_tgt"] * 100], [a["email_tpr_tgt"] * 100], color=col, s=40, zorder=3, label=f"{NAME[m]} (L{L})")
            out.append((m, feat, L, a["ni_fpr_tgt"], b["ni_fpr_tgt"], a["email_tpr_tgt"], b["email_tpr_tgt"], a["ni_vs_attack_auroc"], b["ni_vs_attack_auroc"]))
        ax.set_title(f"Feature: {'last prompt token' if feat == 'final' else 'drift (document minus task-only)'}")
        ax.set_xlabel("False alarms on NotInject harmless text (%)  ← better"); ax.plot([0, 100], [0, 100], color="#ccc", lw=.8, ls=":")
    axes[0].set_ylabel("Email attacks detected (%)  ↑ better"); axes[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Finding 4: dot = standard probe, arrow tip = probe also trained on harmless trigger-word text", fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "fig7_notinject_tradeoff"); return out

# ---------------------------------------------------------------- 8. adversarial training rounds
def fig8():
    rs = [json.load(open(AO / "advtrain" / f"adv_round{i}.json")) for i in range(3)]; out = []
    for i, a in enumerate(rs):
        new = a[f"S{i}"]
        out.append((i, a["clean"]["or_14_22_tpr"], a["clean"]["or_14_22_fpr_benign"], new["or_14_22_tpr"], new["or_14_22_fpr_benign"],
                    [a[f"S{j}"]["or_14_22_tpr"] for j in range(i)]))
    fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(3); w = .2
    ax.bar(x - 1.5 * w, [o[1] * 100 for o in out], w, color=C["mid"], label="Clean injections caught")
    ax.bar(x - .5 * w, [o[3] * 100 for o in out], w, color=C["als"], label="Injections + that round's suffix caught")
    ax.bar(x + .5 * w, [o[4] * 100 for o in out], w, color=C["paper"], label="Harmless docs + that suffix flagged")
    ax.bar(x + 1.5 * w, [o[2] * 100 for o in out], w, color=C["gold"], label="Clean harmless docs flagged")
    for k, o in enumerate(out):
        for dx, v in ((-1.5, o[1]), (-.5, o[3]), (.5, o[4]), (1.5, o[2])): ax.text(k + dx * w, v * 100 + 1.5, f"{v*100:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Round 0: probes on clean data\nS0 = earlier 200-step suffix,\nre-scored (not a fresh attack)", "Round 1: + trained on S0\nS1 = fresh attack", "Round 2: + trained on S0, S1\nS2 = fresh attack"], fontsize=9)
    ax.set_ylabel("% (two-layer detector, L14 OR L22)"); ax.set_ylim(0, 112); ax.legend(ncol=2, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.2))
    ax.set_title("Adversarial training (Qwen2.5-0.5B): probes retrained on attacked inputs, then attacked again", pad=48)
    save(fig, "fig8_adversarial_training"); return out

# ---------------------------------------------------------------- 9. ALS in the reverse direction
def fig9():
    r = json.load(open(AO / "als_checks" / "reverse.json")); out = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True); x = np.arange(len(MODELS)); w = .16
    for ax, test in zip(axes, ("metro", "squad")):
        series = {"ALS on BIPIA curve, paper": [], "ALS on BIPIA curve, adapted": [], "Fixed mid-depth": [], "Average layer": [], "ALS on metro curve (forward)": []}
        for m in MODELS:
            rv = sorted(json.load(open(RES / "reverse" / m / "reverse_per_layer.json")), key=lambda q: q["layer"]); a = np.array([q[f"{test}_auroc"] for q in rv])
            g = r[m][f"regret_{test}"]
            series["ALS on BIPIA curve, paper"].append(g["paper"]); series["ALS on BIPIA curve, adapted"].append(g["adapted"])
            series["Fixed mid-depth"].append(float(a.max() - a[round(len(a) / 2) - 1])); series["Average layer"].append(float(a.max() - a.mean()))
            series["ALS on metro curve (forward)"].append(g["forward_als"])
        for i, (lab, col) in enumerate(zip(series, (C["paper"], C["als"], C["mid"], "#C9D3E3", C["best"]))):
            ax.bar(x + (i - 2) * w, series[lab], w, color=col, label=f"{lab} (mean {np.mean(series[lab]):.3f})")
        out[test] = {k: float(np.mean(v)) for k, v in series.items()}
        ax.set_xticks(x, [SHORT[m] for m in MODELS]); ax.set_title(f"Probe trained on BIPIA, tested on {'metro' if test == 'metro' else 'SQuAD'}")
        ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(0, .62)
    axes[0].set_ylabel("Regret (lower is better)")
    fig.suptitle("Reverse direction: ALS run on the new training domain's own curve does not find the best layer", fontweight="bold", y=1.02)
    save(fig, "fig9_reverse_direction"); return out

if __name__ == "__main__":
    fig1(); d2 = fig2(); d3 = fig3(); fig4(); d5 = fig5(); fig6(); d7 = fig7(); d8 = fig8(); d9 = fig9()
    # numbers quoted in the .md files
    summ = {"fig2_means": {f"{b}|{l}": float(np.mean(v)) for (b, l), v in d2.items()},
            "fig2_means_heldout": {f"{b}|{l}": float(np.mean([v[MODELS.index(m)] for m in HELDOUT])) for (b, l), v in d2.items()},
            "fig3": {f"{t}|{l}": {"adapted": [float(v) for v in y], "paper": float(p)} for (t, l), (y, p) in d3.items()},
            "fig5": [list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in d5],
            "fig7": [list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in d7],
            "fig8": [[o[0], o[1], o[2], o[3], o[4], o[5]] for o in d8], "fig9": d9}
    json.dump(summ, open(FIG / "figure_numbers.json", "w"), indent=1)
    print("done:", log)
