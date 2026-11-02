"""Document-level bootstrap CIs for regret differences between layer rules (SQuAD from metro probes, BIPIA)."""
import json, numpy as np, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from select_math import load_masks; from paths import SWEEP, REC1, REC2, OUT, SEL as SELDIR
from sklearn.metrics import roc_auc_score
UP = str(SWEEP); REC = str(REC1)
SEL = {"qwen2.5-7b": "qwen2.5-7b.json", "qwen2.5-3b": "qwen2.5-3b.json", "qwen2.5-1.5b": "qwen2.5-1.5b.json", "qwen2.5-0.5b": "qwen2.5-0.5b.json", "tinyllama-1.1b": "tinyllama.json", "olmo2-1b": "olmo.json"}
RULES = {"id": "id:neartop(eps=0.1)", "mid": "fixed:0.5 depth", "pis": "pishield:val_acc", "last": "last"}
B = int(__import__("os").environ.get("B", 500)); rng = np.random.default_rng(0)
R1 = [json.loads(l) for l in open(REC)]; fam1, ev, cal = load_masks(REC); doc1 = np.array([r["doc_id"] for r in R1])
R2 = [json.loads(l) for l in open(REC2, encoding="utf-8")]
dom2 = np.array([r["domain"] for r in R2]); sp2 = np.array([r["split"] for r in R2]); fam2 = np.array([r["family"] for r in R2]); ctx2 = np.array([r["ctx"] for r in R2])
def fast_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
out = {}; SKIP = set(__import__("os").environ.get("SKIP","").split(","))
for m, sf in SEL.items():
    if m in SKIP: continue
    sel = list(json.load(open(SELDIR / sf)).values())[0]
    S = np.load(f"{UP}/{m}/probe_scores.npz"); BS = np.load(f"{UP}/bench/{m}/bench_scores.npz")
    for feat in ("final", "delta"):
        L = sel[feat]["L"]; picks = {k: sel[feat]["picks"][v]["layer"] for k, v in RULES.items()}
        # SQuAD: per doc, keep index lists
        sq = ev["squad"] & np.isfinite(S[f"{feat}_1"][3]); docs = np.unique(doc1[sq])
        idx = {d: np.where(sq & (doc1 == d))[0] for d in docs}
        # BIPIA test contexts
        te = (sp2 == "test") & (dom2 != "notinject"); ok = np.isfinite(BS[f"{feat}_std_1"])
        ctxs = np.unique(ctx2[te & ok]); cidx = {c: np.where(te & ok & (ctx2 == c))[0] for c in ctxs}
        def sq_auc(l, sel_idx):
            s = S[f"{feat}_{l}"]; f = fam1[sel_idx]
            return np.nanmean([fast_auc(s[i][sel_idx][f == F], s[i][sel_idx][f == "benign"]) for i, F in enumerate("ABC")])
        def bi_auc(l, sel_idx):
            s = BS[f"{feat}_std_{l}"]; d = dom2[sel_idx]; f = fam2[sel_idx]; v = []
            for D in ("email", "table", "code"):
                mm = d == D; v.append(fast_auc(s[sel_idx][mm & (f == "attack")], s[sel_idx][mm & (f == "benign")]))
            return np.nanmean(v)
        res = {}
        for name, units, getidx, fn in (("squad", docs, idx, sq_auc), ("bipia", ctxs, cidx, bi_auc)):
            diffs = {k: [] for k in ("id-mid", "id-pis", "id-last")}; regs = {k: [] for k in RULES}
            for b in range(B):
                pick = rng.choice(units, len(units), replace=True); si = np.concatenate([getidx[u] for u in pick])
                a = np.array([fn(l, si) for l in range(1, L + 1)]); best = np.nanmax(a)
                r = {k: best - a[picks[k] - 1] for k in RULES}
                for k in RULES: regs[k].append(r[k])
                diffs["id-mid"].append(r["id"] - r["mid"]); diffs["id-pis"].append(r["id"] - r["pis"]); diffs["id-last"].append(r["id"] - r["last"])
            res[name] = {"regret": {k: (float(np.mean(v)), *map(float, np.percentile(v, [2.5, 97.5]))) for k, v in regs.items()},
                         "diff": {k: (float(np.mean(v)), *map(float, np.percentile(v, [2.5, 97.5]))) for k, v in diffs.items()}}
        out[f"{m}|{feat}"] = res
        print(m, feat, {n: {k: "%.3f [%.3f, %.3f]" % v for k, v in r["diff"].items()} for n, r in res.items()}, flush=True)
json.dump(out, open(OUT / "ci.json", "w"), indent=1)
