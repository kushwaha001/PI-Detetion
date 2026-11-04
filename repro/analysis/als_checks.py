"""
Three checks of ALS (intrinsic-dimension layer selection) for prompt-injection probes. CPU only, numpy + matplotlib.
  1. sensitivity  adapted rule near_top_peak(eps) for eps in EPS, and the paper rule (FEPoID), on all six models and
                  both features; regret on SQuAD (per_layer.json) and on BIPIA (bench_per_layer.json, variant std).
  2. figure       ID curve of benign prompts next to each layer's SQuAD AUROC, with the paper pick, the adapted pick
                  and the best layer marked (the analogue of the ALS paper's main figure).
  3. reverse      ALS run on the reverse direction's own training domain: TwoNN ID curve of benign BIPIA email+table
                  prompts (the data reverse_eval.py trains on), 10 subsamples of 80%; picks compared with the best
                  layers of reverse_per_layer.json (tests on metro and SQuAD) and with the forward ALS layer.
Run from ROOT:  python repro/analysis/als_checks.py      (LDG_ROOT / LDG_SWEEP override the folders)
Writes repro/analysis_out/als_checks/{sensitivity.json, sensitivity.md, reverse.json, reverse.md, als_figure.png}
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("LDG_ROOT", Path(__file__).resolve().parents[2]))
STATES = Path(os.environ.get("LDG_SWEEP", "D:/ldg-sweep"))
RES = ROOT / "results_sweep"; OUT = ROOT / "repro" / "analysis_out" / "als_checks"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "tinyllama-1.1b", "olmo2-1b"]
HELDOUT = {"qwen2.5-3b", "qwen2.5-7b", "tinyllama-1.1b", "olmo2-1b"}
EPS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
W = 5
rng = np.random.default_rng(0)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---- the two ALS rules (same code as repro/analysis/select_math.py) ----
def fepoid(idc, w=W):
    L = len(idc); peaks = [i for i in range(L) if (i == 0 or idc[i] > idc[i - 1]) and (i == L - 1 or idc[i] >= idc[i + 1])]
    for p in peaks:
        h = min(p + w, L - 1)
        mono = all(idc[j] < idc[j + 1] for j in range(p + 1, h)) if h > p + 1 else False
        if idc[p] < idc[h] and mono: continue
        return p + 1
    return (peaks[0] if peaks else 0) + 1

def near_top_peak(m, eps=0.10):
    m = np.asarray(m, float); thr = (1 - eps) * m.max(); L = len(m)
    for p in range(L):
        if m[p] >= thr and (p == 0 or m[p] > m[p - 1]) and (p == L - 1 or m[p] >= m[p + 1]): return p + 1
    return int(np.argmax(m)) + 1

# ---- TwoNN (same code as layer_sweep_analyze.py) ----
def knn_dist(X, k):
    X = np.asarray(X, np.float64); sq = (X * X).sum(1); D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(D2, np.inf); D2 = np.maximum(D2, 0)
    part = np.partition(D2, k - 1, axis=1)[:, :k]; part.sort(1); return np.sqrt(part)
def twonn(X, frac=0.9):
    d = knn_dist(X, 2); r1, r2 = d[:, 0], d[:, 1]; ok = r1 > 1e-8
    mu = np.sort(r2[ok] / r1[ok]); n = len(mu); F = np.arange(1, n + 1) / n; keep = int(n * frac)
    x = np.log(mu[:keep]); y = -np.log(1 - F[:keep]); return float((x @ y) / (x @ x))

def id_curve(rows):
    boots = [r.get("id_benign_boots") or [] for r in rows]
    if all(len(b) for b in boots): return np.array([np.mean(b) for b in boots]), np.array([np.std(b, ddof=1) if len(b) > 1 else 0 for b in boots])
    return np.array([r["id_benign"] for r in rows]), np.array([r.get("id_benign_sd", 0) for r in rows])

def check1_and_2():
    out = {}; md = ["# ALS sensitivity to the cutoff\n", "Regret = best layer's AUROC minus the chosen layer's (lower is better). "
          "Adapted rule = first peak within eps of the curve's maximum (eps = 0.10 is the frozen rule). Paper rule = FEPoID.\n"]
    fig_data = {}
    for m in MODELS:
        pl = json.load(open(RES / m / "per_layer.json")); bench = json.load(open(RES / "bench" / m / "bench_per_layer.json"))
        for feat in ("final", "delta"):
            rows = sorted([r for r in pl if r["feat"] == feat], key=lambda r: r["layer"])
            if not rows: continue
            mcur, se = id_curve(rows); sq = np.array([r["squad_auroc_mean"] for r in rows])
            br = {r["layer"]: np.nanmean([r[f"{d}_auroc"] for d in ("email", "table", "code")]) for r in bench if r["feat"] == feat and r["variant"] == "std"}
            bi = np.array([br.get(r["layer"], np.nan) for r in rows])
            def score(L):
                return {"layer": L, "squad_regret": float(np.nanmax(sq) - sq[L - 1]), "bipia_regret": float(np.nanmax(bi) - bi[L - 1]) if np.isfinite(bi).any() else None}
            res = {"best_squad": int(np.nanargmax(sq)) + 1, "best_bipia": int(np.nanargmax(bi)) + 1 if np.isfinite(bi).any() else None,
                   "paper": score(fepoid(list(mcur)))}
            for e in EPS: res[f"eps={e:.2f}"] = score(near_top_peak(mcur, e))
            out[f"{m}/{feat}"] = res
            if feat == "final": fig_data[m] = (mcur, se, sq, res)
            log(m, feat, {k: (v["layer"], round(v["squad_regret"], 3)) if isinstance(v, dict) else v for k, v in res.items()})
    # tables
    keys = ["paper"] + [f"eps={e:.2f}" for e in EPS]
    for metric in ("squad_regret", "bipia_regret"):
        md.append(f"\n## {'SQuAD' if metric == 'squad_regret' else 'BIPIA'} regret (chosen layer · regret)\n")
        md.append("| Model / feature | " + " | ".join(k.replace("eps=", "ε=") for k in keys) + " |"); md.append("|" + "---|" * (len(keys) + 1))
        for k, r in out.items():
            md.append(f"| {k}{' (held out)' if k.split('/')[0] in HELDOUT else ''} | " + " | ".join(f"L{r[c]['layer']} · {r[c][metric]:.3f}" if r[c][metric] is not None else "—" for c in keys) + " |")
        for name, sel in (("Mean, all 12", list(out)), ("Mean, held-out 8", [k for k in out if k.split('/')[0] in HELDOUT])):
            md.append(f"| **{name}** | " + " | ".join(f"**{np.nanmean([out[k][c][metric] for k in sel if out[k][c][metric] is not None]):.3f}**" for c in keys) + " |")
    json.dump(out, open(OUT / "sensitivity.json", "w"), indent=1); open(OUT / "sensitivity.md", "w", encoding="utf-8").write("\n".join(md))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = ["qwen2.5-1.5b", "qwen2.5-3b", "olmo2-1b", "qwen2.5-7b"]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.6))
    for ax, m in zip(axes, pick):
        mcur, se, sq, res = fig_data[m]; x = np.arange(1, len(mcur) + 1)
        ax.fill_between(x, mcur - se, mcur + se, color="#2F6FB2", alpha=.18, lw=0); ax.plot(x, mcur, color="#2F6FB2", lw=2, label="Intrinsic dimension (TwoNN)")
        ax.set_xlabel("Layer"); ax.set_ylabel("Intrinsic dimension", color="#2F6FB2"); ax.set_title(m, fontsize=11)
        ax2 = ax.twinx(); ax2.plot(x, sq, color="#555", lw=1.6, ls="--", label="SQuAD AUROC of the probe"); ax2.set_ylabel("SQuAD AUROC", color="#555")
        for L, c, lab, yy in ((res["paper"]["layer"], "#C8642A", "ALS paper rule", .99), (res["eps=0.10"]["layer"], "#2F6FB2", "ALS adapted", .17), (res["best_squad"], "#222", "Best layer", .08)):
            ax.axvline(L, color=c, lw=1.4, ls=":" if c == "#222" else "-", alpha=.9)
            ax.text(L, yy, f" {lab} L{L}", color=c, fontsize=7.5, va="top", fontweight="bold", transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
        for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[0].get_figure().axes[-1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / "als_figure.png", dpi=200); log("figure written")
    return out

def check3(forward, models=MODELS):
    prev = json.load(open(OUT / "reverse.json")) if (OUT / "reverse.json").exists() else {}
    R2 = [json.loads(l) for l in open(RES / "bench" / "records2.jsonl", encoding="utf-8")]
    dom = np.array([r["domain"] for r in R2]); fam = np.array([r["family"] for r in R2])
    out = dict(prev); md = ["# ALS in the reverse direction\n", "Probe trained on BIPIA email+table (reverse_eval.py). ALS reads the TwoNN curve of benign BIPIA email+table prompts "
          "(test and pool), 10 subsamples of 80%. Regret against the best layer for metro and for SQuAD.\n",
          "| Model | Best layer, metro | Best layer, SQuAD | ALS paper | ALS adapted | Forward ALS layer | Regret metro: paper / adapted / forward | Regret SQuAD: paper / adapted / forward |", "|---|---|---|---|---|---|---|---|"]
    for m in models:
        S = STATES / "bench" / m; meta = json.load(open(S / "meta.json")); L1, N, D = meta["L1"], meta["N"], meta["D"]
        F = np.memmap(S / "final.f16", np.float16, "r", shape=(L1, N, D)); ok = np.load(S / "done.npy") == 1
        idx = np.where(ok & np.isin(dom, ["email", "table"]) & (fam == "benign"))[0]
        subs = [np.sort(rng.choice(idx, int(.8 * len(idx)), replace=False)) for _ in range(10)]
        boots = np.zeros((10, L1 - 1))
        for l in range(1, L1):
            X = np.asarray(F[l][idx], np.float32); pos = {v: i for i, v in enumerate(idx)}
            for b, s in enumerate(subs): boots[b, l - 1] = twonn(X[[pos[v] for v in s]])
        mcur = boots.mean(0)
        rev = sorted(json.load(open(RES / "reverse" / m / "reverse_per_layer.json")), key=lambda r: r["layer"])
        met = np.array([r["metro_auroc"] for r in rev]); squ = np.array([r["squad_auroc"] for r in rev])
        pa, ad = fepoid(list(mcur)), near_top_peak(mcur, .10); fw = forward[f"{m}/final"]["eps=0.10"]["layer"]
        reg = lambda a, L: float(np.nanmax(a) - a[L - 1])
        res = {"n_benign": int(len(idx)), "id_curve": mcur.round(3).tolist(), "best_metro": int(np.nanargmax(met)) + 1, "best_squad": int(np.nanargmax(squ)) + 1,
               "paper": pa, "adapted": ad, "forward_als": fw,
               "regret_metro": {"paper": reg(met, pa), "adapted": reg(met, ad), "forward_als": reg(met, fw)},
               "regret_squad": {"paper": reg(squ, pa), "adapted": reg(squ, ad), "forward_als": reg(squ, fw)}}
        out[m] = res; log(m, "reverse", {k: v for k, v in res.items() if k != "id_curve"})
        json.dump(out, open(OUT / "reverse.json", "w"), indent=1)
    for m in [x for x in MODELS if x in out]:
        res = out[m]; pa, ad, fw = res["paper"], res["adapted"], res["forward_als"]
        rm, rs = res["regret_metro"], res["regret_squad"]
        md.append(f"| {m} | L{res['best_metro']} | L{res['best_squad']} | L{pa} | L{ad} | L{fw} | {rm['paper']:.3f} / {rm['adapted']:.3f} / {rm['forward_als']:.3f} | {rs['paper']:.3f} / {rs['adapted']:.3f} / {rs['forward_als']:.3f} |")
    for key in ("regret_metro", "regret_squad"):
        md.append(f"\nMean {key}: " + ", ".join(f"{r} {np.mean([out[m][key][r] for m in out]):.3f}" for r in ("paper", "adapted", "forward_als")))
    json.dump(out, open(OUT / "reverse.json", "w"), indent=1); open(OUT / "reverse.md", "w", encoding="utf-8").write("\n".join(md))

if __name__ == "__main__":
    # usage: als_checks.py            -> all checks
    #        als_checks.py forward            -> only checks 1 and 2
    #        als_checks.py reverse m1 m2 ...  -> only the reverse check for the named models (merges into reverse.json)
    if len(sys.argv) > 1 and sys.argv[1] == "forward":
        check1_and_2()
    elif len(sys.argv) > 1 and sys.argv[1] == "reverse":
        check3(json.load(open(OUT / "sensitivity.json")), sys.argv[2:] or MODELS)
    else:
        fwd = check1_and_2()
        check3(fwd)
    log("ALS CHECKS DONE")
