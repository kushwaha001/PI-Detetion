"""
NotInject false-alarm experiment (CPU). Same probes as cross_eval.py (metro half A, seen tasks), plus variants that add
benign NotInject documents as extra negatives. NotInject is split by context index: even ni<j> go to training, odd ni<j>
are held out, and every variant is scored only on the held-out half, so the comparison is fair.
Variants:  std        benign vs attack families A/B/C (as cross_eval.py)
           ni         std + NotInject train half as negatives
           ni_hard    ni + metro family D as negatives
           ni_bipia   ni + BIPIA clean pool contexts as negatives (target-domain benign text, no labels needed; the
                      email threshold is then set on contexts the probe saw in training, which is conservative)
Features:  final and delta (final minus task-only state).
Metrics on held-out NotInject: FPR at the threshold re-set on clean BIPIA email pool, and AUROC of email attacks vs NotInject;
on BIPIA: per-domain AUROC and TPR at the re-set threshold, so any loss of detection is visible.
Writes RES_DIR/ni_per_layer.json.
"""
import os, sys, json, math, time
from pathlib import Path
import numpy as np
try:
    import threadpoolctl as _tpc
    _o = _tpc._ThreadpoolInfo._make_module_from_path
    def _safe(self, *a, **k):
        try: return _o(self, *a, **k)
        except Exception: return None
    _tpc._ThreadpoolInfo._make_module_from_path = _safe
except Exception: pass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ["SRC_DIR"]); BEN = Path(os.environ["BENCH_DIR"]); RES = Path(os.environ["RES_DIR"]); RES.mkdir(parents=True, exist_ok=True)
REC1 = HERE / "cache_v2" / "records_full.jsonl"; REC2 = Path(os.environ.get("RECORDS", HERE / "results_sweep" / "bench" / "records2.jsonl"))
ALPHA = 0.05
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True)
    open(RES / "cross_eval.log", "a", encoding="utf-8").write(s + "\n")
def auroc(p, n):
    if len(p) == 0 or len(n) == 0: return float("nan")
    return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))
def cthr(s, a=ALPHA):
    s = np.sort(s); n = len(s); return float(s[min(n - 1, int(math.ceil((n + 1) * (1 - a))) - 1)])
m1 = json.load(open(SRC / "meta.json")); N1, L1, D = m1["N"], m1["L1"], m1["D"]
m2 = json.load(open(BEN / "meta.json")); N2 = m2["N"]; assert m2["L1"] == L1 and m2["D"] == D
F1 = np.memmap(SRC / "final.f16", np.float16, "r", shape=(L1, N1, D)); T1 = np.load(SRC / "task_final.npy")
F2 = np.memmap(BEN / "final.f16", np.float16, "r", shape=(L1, N2, D)); T2 = np.load(BEN / "task_states.npy"); tix2 = np.load(BEN / "task_idx.npy")
r1 = [json.loads(l) for l in open(REC1, encoding="utf-8")]; r2 = [json.loads(l) for l in open(REC2, encoding="utf-8")]
assert len(r1) == N1 and len(r2) == N2
fam1 = np.array([r["family"] for r in r1]); split1 = np.array([r["split"] for r in r1]); dom1 = np.array([r["domain"] for r in r1])
doc1 = np.array([r["doc_id"] for r in r1]); task1 = np.array([r["task_id"] for r in r1]); seen1 = np.array([r["task_seen"] for r in r1])
mt = sorted({d for d, a, b in zip(doc1, dom1, split1) if a == "metro" and b == "test"}, key=lambda s: int(s[1:])); halfA = set(mt[0::2])
ok1 = np.load(SRC / "done.npy") == 1
is_train = (split1 == "train") & ok1; is_cal = (split1 == "cal") & (fam1 == "benign") & ok1
is_pA = np.array([d in halfA for d in doc1]) & (split1 == "test") & seen1 & ok1
dom2 = np.array([r["domain"] for r in r2]); split2 = np.array([r["split"] for r in r2]); fam2 = np.array([r["family"] for r in r2])
ok2 = np.load(BEN / "done.npy") == 1
log(f"src {SRC} N1 {N1} | bench N2 {N2} ok {ok2.sum()} | L {L1 - 1} D {D}")
ctx2 = np.array([r["ctx"] for r in r2])
is_ni = (dom2 == "notinject") & ok2
ni_idx = np.array([int(c[2:]) if c.startswith("ni") else -1 for c in ctx2])
ni_tr = is_ni & (ni_idx % 2 == 0); ni_te = is_ni & (ni_idx % 2 == 1)
pool2 = (split2 == "pool") & ok2
log(f"NotInject train {ni_tr.sum()} held-out {ni_te.sum()} | BIPIA pool {pool2.sum()}")
VAR = {"std": (["benign", "A", "B", "C"], None), "ni": (["benign", "A", "B", "C"], ni_tr),
       "ni_hard": (["benign", "A", "B", "C", "D"], ni_tr), "ni_bipia": (["benign", "A", "B", "C"], ni_tr | pool2)}
rows, SC = [], {}
for kind in ("final", "delta"):
    for l in range(1, L1):
        X1 = np.asarray(F1[l], np.float32); X2 = np.asarray(F2[l], np.float32)
        if kind == "delta": X1 = X1 - T1[task1, l].astype(np.float32); X2 = X2 - T2[tix2, l].astype(np.float32)
        mu, sd = X1[is_train].mean(0), X1[is_train].std(0) + 1e-6
        Z1 = (X1 - mu) / sd; Z2 = (X2 - mu) / sd; Z2[~ok2] = 0
        for v, (fams, extra) in VAR.items():
            m = is_pA & np.isin(fam1, fams); y = np.isin(fam1[m], ["A", "B", "C"]).astype(int)
            Xt, yt = Z1[m], y
            if extra is not None: Xt = np.r_[Xt, Z2[extra]]; yt = np.r_[yt, np.zeros(int(extra.sum()), int)]
            clf = LogisticRegression(max_iter=3000, C=0.2).fit(Xt, yt)
            s1 = clf.decision_function(Z1[is_cal]); s2 = clf.decision_function(Z2)
            thr = cthr(s1); row = {"feat": kind, "variant": v, "layer": l}
            for d in ("email", "table", "code"):
                te = (dom2 == d) & (split2 == "test") & ok2; ben = s2[te & (fam2 == "benign")]; att = s2[te & (fam2 == "attack")]
                pool = s2[(dom2 == d) & (split2 == "pool") & ok2]; tt = cthr(pool)
                row.update({f"{d}_auroc": auroc(att, ben), f"{d}_fpr_src": float((ben > thr).mean()), f"{d}_tpr_src": float((att > thr).mean()),
                            f"{d}_fpr_tgt": float((ben > tt).mean()), f"{d}_tpr_tgt": float((att > tt).mean())})
            ni = s2[ni_te]; ep = s2[(dom2 == "email") & (split2 == "pool") & ok2]; ea = s2[(dom2 == "email") & (fam2 == "attack") & ok2]
            row.update({"ni_fpr_src": float((ni > thr).mean()), "ni_fpr_tgt": float((ni > cthr(ep)).mean()), "ni_vs_attack_auroc": auroc(ea, ni)})
            row["bench_auroc_mean"] = float(np.mean([row[f"{d}_auroc"] for d in ("email", "table", "code")]))
            rows.append(row)
        r_s = [r for r in rows if r["feat"] == kind and r["layer"] == l]
        log(f"{kind:5s} L{l:02d} " + " | ".join(f"{r['variant']}: AUROC {r['bench_auroc_mean']:.2f} TPR-email {r['email_tpr_tgt']:.2f} NI-FPR {r['ni_fpr_tgt']:.2f} NIvsAtt {r['ni_vs_attack_auroc']:.2f}" for r in r_s))
    json.dump(rows, open(RES / "ni_per_layer.json", "w"), indent=0)
log("NI EVAL DONE")
