"""
Does the intrinsic-dimension (ID) curve track where prompt-injection detection is good, beyond plain depth?
For every model and feature, across layers:
  rho(ID, AUROC)            Spearman rank correlation between the benign ID curve and the probe's AUROC per layer
  rho(depth, AUROC)         the same for layer depth alone (the trivial explanation: "deeper is better")
  partial rho(ID, AUROC | depth)   ID vs AUROC after removing what depth explains from both
Reads results_sweep/<model>/per_layer.json (written by layer_sweep_analyze.py). numpy only.
Run from ROOT:  python repro/analysis/id_auroc_correlation.py   -> repro/analysis_out/id_auroc_correlation.{json,md}
"""
import json, os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("LDG_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results_sweep"; OUT = ROOT / "repro" / "analysis_out"
MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "tinyllama-1.1b", "olmo2-1b"]

def rank(a):                      # 0 for the smallest value, 1 for the next, ...
    return np.argsort(np.argsort(a)).astype(float)

def spearman(a, b):               # Pearson correlation of the ranks
    return float(np.corrcoef(rank(a), rank(b))[0, 1])

def partial_spearman(a, b, c):    # correlation of a and b after removing a straight-line effect of c from both
    ra, rb, rc = rank(a), rank(b), rank(c)
    resid = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return float(np.corrcoef(resid(ra), resid(rb))[0, 1])

out, md = {}, ["| Model | Feature | rho(ID, SQuAD AUROC) | rho(depth, SQuAD AUROC) | partial rho, SQuAD | partial rho, metro |", "|---|---|---|---|---|---|"]
for m in MODELS:
    rows_all = json.load(open(RES / m / "per_layer.json"))
    for feat in ("final", "delta"):
        R = sorted([r for r in rows_all if r["feat"] == feat], key=lambda r: r["layer"])
        idc = np.array([np.mean(r["id_benign_boots"]) if r.get("id_benign_boots") else r["id_benign"] for r in R])
        squad = np.array([r["squad_auroc_mean"] for r in R]); metro = np.array([r["metro_auroc_mean"] for r in R])
        depth = np.arange(1, len(R) + 1)
        res = {"rho_id_squad": spearman(idc, squad), "rho_id_metro": spearman(idc, metro), "rho_depth_squad": spearman(depth, squad),
               "partial_squad": partial_spearman(idc, squad, depth), "partial_metro": partial_spearman(idc, metro, depth)}
        out[f"{m}/{feat}"] = res
        md.append(f"| {m} | {feat} | {res['rho_id_squad']:+.2f} | {res['rho_depth_squad']:+.2f} | {res['partial_squad']:+.2f} | {res['partial_metro']:+.2f} |")
        print(m, feat, {k: round(v, 2) for k, v in res.items()})
json.dump(out, open(OUT / "id_auroc_correlation.json", "w"), indent=1)
open(OUT / "id_auroc_correlation.md", "w", encoding="utf-8").write("\n".join(md))
