"""Reverse direction: probe trained on BIPIA email+table (benign vs attack, test split) and tested on BIPIA code,
metro (half B) and SQuAD. z-scoring uses the BIPIA pool (clean training contexts). Last-prompt-token feature, all layers."""
import os, json, time
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
m1 = json.load(open(SRC / "meta.json")); N1, L1, D = m1["N"], m1["L1"], m1["D"]; m2 = json.load(open(BEN / "meta.json")); N2 = m2["N"]
F1 = np.memmap(SRC / "final.f16", np.float16, "r", shape=(L1, N1, D)); F2 = np.memmap(BEN / "final.f16", np.float16, "r", shape=(L1, N2, D))
r1 = [json.loads(l) for l in open(HERE / "cache_v2" / "records_full.jsonl", encoding="utf-8")]
r2 = [json.loads(l) for l in open(HERE / "results_sweep" / "bench" / "records2.jsonl", encoding="utf-8")]
ok1 = np.load(SRC / "done.npy") == 1; ok2 = np.load(BEN / "done.npy") == 1
fam1 = np.array([r["family"] for r in r1]); sp1 = np.array([r["split"] for r in r1]); dom1 = np.array([r["domain"] for r in r1]); doc1 = np.array([r["doc_id"] for r in r1])
mt = sorted({d for d, a, b in zip(doc1, dom1, sp1) if a == "metro" and b == "test"}, key=lambda s: int(s[1:])); hB = set(mt[1::2])
ev1 = {"metro": np.array([d in hB for d in doc1]) & (sp1 == "test") & ok1, "squad": (dom1 == "squad") & (sp1 == "test") & ok1}
dom2 = np.array([r["domain"] for r in r2]); sp2 = np.array([r["split"] for r in r2]); fam2 = np.array([r["family"] for r in r2])
tr = (sp2 == "test") & np.isin(dom2, ["email", "table"]) & ok2; pool = (sp2 == "pool") & ok2
def auc(p, n): return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])) if len(p) and len(n) else float("nan")
rows = []
for l in range(1, L1):
    X2 = np.asarray(F2[l], np.float32); X1 = np.asarray(F1[l], np.float32)
    mu, sd = X2[pool].mean(0), X2[pool].std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=0.2).fit((X2[tr] - mu) / sd, (fam2[tr] == "attack").astype(int))
    s2 = clf.decision_function((X2 - mu) / sd); s1 = clf.decision_function((X1 - mu) / sd)
    row = {"layer": l}
    te = (dom2 == "code") & (sp2 == "test") & ok2; row["code_auroc"] = auc(s2[te & (fam2 == "attack")], s2[te & (fam2 == "benign")])
    for dn, mk in ev1.items():
        row[f"{dn}_auroc"] = float(np.nanmean([auc(s1[mk & (fam1 == f)], s1[mk & (fam1 == "benign")]) for f in "ABC"]))
    rows.append(row); print(time.strftime("%H:%M:%S"), row, flush=True)
json.dump(rows, open(RES / "reverse_per_layer.json", "w"), indent=0); print("REVERSE DONE", flush=True)
