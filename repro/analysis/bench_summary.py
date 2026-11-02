"""Aggregate cross-benchmark results: rule picks (frozen, computed from the metro ID curves) evaluated on BIPIA/NotInject,
hard-negative training, target recalibration, all-layer score ensemble, DeBERTa baseline."""
import json, sys, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import ROOT, SWEEP, REC1, REC2, OUT, SEL
UP = SWEEP; MATH = SEL
REC2 = [json.loads(l) for l in open(REC2, encoding="utf-8")]
dom = np.array([r["domain"] for r in REC2]); split = np.array([r["split"] for r in REC2]); fam = np.array([r["family"] for r in REC2])
MODELS = {"qwen2.5-7b": "qwen2.5-7b.json", "qwen2.5-3b": "qwen2.5-3b.json", "qwen2.5-1.5b": "qwen2.5-1.5b.json", "qwen2.5-0.5b": "qwen2.5-0.5b.json", "tinyllama-1.1b": "tinyllama.json", "olmo2-1b": "olmo.json"}
RULES = ["id:neartop(eps=0.1)", "id:sig(z=1.645)", "pishield:val_acc", "fixed:0.5 depth", "last"]
DOMS = ["email", "table", "code"]
def auroc(p, n): return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])) if len(p) and len(n) else float("nan")
def cthr(s, a=0.05):
    s = np.sort(s); n = len(s); return float(s[min(n - 1, int(math.ceil((n + 1) * (1 - a))) - 1)])
def metrics(s, s_cal):
    out = {}; thr = cthr(s_cal)
    for d in DOMS:
        te = (dom == d) & (split == "test"); b, a_ = s[te & (fam == "benign")], s[te & (fam == "attack")]; t2 = cthr(s[(dom == d) & (split == "pool")])
        out[d] = dict(auroc=auroc(a_, b), fpr_src=float((b > thr).mean()), tpr_src=float((a_ > thr).mean()), fpr_tgt=float((b > t2).mean()), tpr_tgt=float((a_ > t2).mean()))
    ni = s[dom == "notinject"]; out["notinject"] = dict(fpr_src=float((ni > thr).mean()), fpr_tgt=float((ni > cthr(s[(dom == "email") & (split == "pool")])).mean()),
                                                     auroc_vs_attacks=auroc(s[(dom == "email") & (fam == "attack")], ni))
    out["mean_auroc"] = float(np.mean([out[d]["auroc"] for d in DOMS])); return out
res = {}
for m, mf in MODELS.items():
    d = UP / "bench" / m
    if not (d / "bench_scores.npz").exists(): continue
    S = np.load(d / "bench_scores.npz"); sel = json.load(open(MATH / mf)); sel = list(sel.values())[0]
    res[m] = {}
    for feat in ("final", "delta"):
        L = sel[feat]["L"]; picks = {r: sel[feat]["picks"][r]["layer"] for r in RULES}
        per = {v: {l: metrics(S[f"{feat}_{v}_{l}"], S[f"{feat}_{v}_{l}_metrocal"]) for l in range(1, L + 1)} for v in ("std", "hardneg")}
        oracle = max(range(1, L + 1), key=lambda l: per["std"][l]["mean_auroc"])
        R = {"oracle_bench": oracle, "oracle_bench_auroc": per["std"][oracle]["mean_auroc"], "rules": {}}
        for r, l in picks.items():
            R["rules"][r] = {"layer": l, "std": per["std"][l], "hardneg": per["hardneg"][l], "regret": per["std"][oracle]["mean_auroc"] - per["std"][l]["mean_auroc"]}
        # all-layer ensemble: mean over layers of the score standardised by its metro-cal benign mean/std
        for v in ("std", "hardneg"):
            zs = [(S[f"{feat}_{v}_{l}"] - S[f"{feat}_{v}_{l}_metrocal"].mean()) / (S[f"{feat}_{v}_{l}_metrocal"].std() + 1e-9) for l in range(1, L + 1)]
            zc = [(S[f"{feat}_{v}_{l}_metrocal"] - S[f"{feat}_{v}_{l}_metrocal"].mean()) / (S[f"{feat}_{v}_{l}_metrocal"].std() + 1e-9) for l in range(1, L + 1)]
            R[f"ensemble_all_{v}"] = metrics(np.mean(zs, 0), np.mean(zc, 0))
            hi = range(L // 2, L + 1)    # upper half of the network
            R[f"ensemble_upper_{v}"] = metrics(np.mean([zs[l - 1] for l in hi], 0), np.mean([zc[l - 1] for l in hi], 0))
        R["per_layer_mean_auroc"] = {v: [per[v][l]["mean_auroc"] for l in range(1, L + 1)] for v in per}
        R["per_layer_ni_fpr_tgt"] = {v: [per[v][l]["notinject"]["fpr_tgt"] for l in range(1, L + 1)] for v in per}
        res[m][feat] = R
db = UP / "bench" / "deberta" / "deberta_scores.npz"
if db.exists():
    s = np.load(db)["bench"]; res["deberta"] = metrics(s, s[(dom == "email") & (split == "pool")])   # no metro-cal for the classifier: src thr = its own default 0 below
    res["deberta"]["default_threshold"] = {dd: dict(fpr=float((s[(dom == dd) & (split == "test") & (fam == "benign")] > 0).mean()), tpr=float((s[(dom == dd) & (split == "test") & (fam == "attack")] > 0).mean())) for dd in DOMS}
    res["deberta"]["default_threshold"]["notinject_fpr"] = float((s[dom == "notinject"] > 0).mean())
json.dump(res, open(OUT / "bench_summary.json", "w"), indent=1)
for m, R in res.items():
    if m == "deberta":
        print("\n== DeBERTa-v3 PI classifier:", {d: round(R[d]["auroc"], 3) for d in DOMS}, "| default thr:", {k: v for k, v in R["default_threshold"].items()}); continue
    for feat, F in R.items():
        print(f"\n== {m} [{feat}] oracle-on-bench L{F['oracle_bench']} mean AUROC {F['oracle_bench_auroc']:.3f}")
        for r, x in F["rules"].items():
            s_, h_ = x["std"], x["hardneg"]
            print(f"  {r:22s} L{x['layer']:>2} regret {x['regret']:.3f} | AUROC e/t/c {s_['email']['auroc']:.2f}/{s_['table']['auroc']:.2f}/{s_['code']['auroc']:.2f}"
                  f" | NI FPR src {s_['notinject']['fpr_src']:.2f}->{h_['notinject']['fpr_src']:.2f}(hardneg) tgt {s_['notinject']['fpr_tgt']:.2f}->{h_['notinject']['fpr_tgt']:.2f}"
                  f" | tgt TPR e/t/c {s_['email']['tpr_tgt']:.2f}/{s_['table']['tpr_tgt']:.2f}/{s_['code']['tpr_tgt']:.2f} FPR {s_['email']['fpr_tgt']:.2f}/{s_['table']['fpr_tgt']:.2f}/{s_['code']['fpr_tgt']:.2f}"
                  f" | src FPR e/t/c {s_['email']['fpr_src']:.2f}/{s_['table']['fpr_src']:.2f}/{s_['code']['fpr_src']:.2f}")
        for k in ("ensemble_all_std", "ensemble_all_hardneg", "ensemble_upper_std"):
            e = F[k]; print(f"  {k:22s}     mean AUROC {e['mean_auroc']:.3f} | AUROC e/t/c {e['email']['auroc']:.2f}/{e['table']['auroc']:.2f}/{e['code']['auroc']:.2f} | NI FPR tgt {e['notinject']['fpr_tgt']:.2f}")
