import json, numpy as np, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import REC1, SWEEP; from select_math import load_masks, conformal_thr, auroc
rec = str(REC1)
R = [json.loads(l) for l in open(rec)]
fam, ev, cal = load_masks(rec)
doc = np.array([r["doc_id"] for r in R]); dom = np.array([r["domain"] for r in R])
sq_docs = sorted({d for d in doc[dom == "squad"]}, key=lambda s: int(s[1:]))
h1 = np.isin(doc, sq_docs[0::2]) & ev["squad"]; h2 = np.isin(doc, sq_docs[1::2]) & ev["squad"]
S = np.load(SWEEP / "qwen2.5-1.5b" / "probe_scores.npz")
rng = np.random.default_rng(0)
for kind in ("final",):
  print(kind, "layer | source-calibrated: FPRben FPR_D TPR_C | target-recal (n clean target): FPRben FPR_D TPR_A TPR_B TPR_C")
  for l in (8, 14, 16, 17, 18, 22, 28):
    s = S[f"{kind}_{l}"][3]
    thr = conformal_thr(s[cal])
    b2 = s[h2 & (fam == "benign")]; d2 = s[h2 & (fam == "D")]
    src = [(b2 > thr).mean(), (d2 > thr).mean(), (s[h2 & (fam == "C")] > thr).mean()]
    out = []
    for n in (50, 200):
        pool = np.where(h1 & np.isin(fam, ["benign", "D"]))[0]
        vals = []
        for _ in range(20):
            idx = rng.choice(pool, n, replace=False); t = conformal_thr(s[idx])
            vals.append([(b2 > t).mean(), (d2 > t).mean()] + [(s[h2 & (fam == f)] > t).mean() for f in "ABC"])
        out.append((n, np.mean(vals, 0)))
    print(f"L{l:2d} | {src[0]:.2f} {src[1]:.2f} {src[2]:.2f} | " + " | ".join(f"n={n}: " + " ".join(f"{v:.2f}" for v in m) for n, m in out))
