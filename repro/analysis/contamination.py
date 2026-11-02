"""
Recalibrating the detector threshold from unlabeled target traffic that is contaminated with attacks.

Setup. Pool of n target scores: (1-e) benign (incl. benign documents with reader-directed instructions) and e contaminants.
A rule picks t = empirical q-quantile of the pool. Since F_mix = (1-e) F_ben + e F_con with F_con unknown,
        (q - e)/(1 - e) <= F_ben(t) <= q/(1 - e)    =>    FPR_ben(t) in [ (1-e-q)/(1-e) , (1-q)/(1-e) ].
Rules (e_max = assumed upper bound on contamination):
  naive        q = 1 - a                  FPR <= a/(1-e)           (inflated in the worst case)
  conservative q = 1 - a(1 - e_max)       FPR <= a for every e <= e_max, any attack distribution (distribution-free)
  trimmed      q = (1 - a)(1 - e_max)     resists decoys at the top, but FPR can reach (a + e_max - a e_max)/(1-e)
The width of the FPR interval, e/(1-e), does not depend on the rule: no threshold computed from the pool alone can both
guarantee FPR <= a and be immune to an attacker who controls where the contaminants score.
Experiments on real probe scores (target = SQuAD, pool from doc-half 1, rates on doc-half 2):
  natural   contaminants are attacks A/B/C drawn from half 1
  decoy     contaminants score above every benign document (attacker floods the pool with obvious decoys)
"""
import json, sys, math, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from select_math import load_masks; from paths import REC1, OUT
def q_thr(s, q):
    s = np.sort(s); n = len(s); return float(s[min(n - 1, max(0, int(math.ceil((n + 1) * q)) - 1))])
def run(scores, fam, doc, dom, ev, cal, a=0.05, n=200, e_max=0.10, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    sq = sorted({d for d in doc[dom == "squad"]}, key=lambda x: int(x[1:]))
    h1 = np.isin(doc, sq[0::2]) & ev["squad"]; h2 = np.isin(doc, sq[1::2]) & ev["squad"]
    ben1 = np.where(h1 & np.isin(fam, ["benign", "D"]))[0]; att1 = np.where(h1 & np.isin(fam, ["A", "B", "C"]))[0]
    ben2 = scores[h2 & np.isin(fam, ["benign", "D"])]; att2 = scores[h2 & np.isin(fam, ["A", "B", "C"])]
    rules = {"naive": 1 - a, "conservative": 1 - a * (1 - e_max), "trimmed": (1 - a) * (1 - e_max), "robust_affine": None}
    # robust affine transfer: benign score law assumed to shift by location/scale between domains; the source
    # (metro cal benign) fixes the standardised quantile k, the target pool gives median and MAD (50% breakdown point)
    c = scores[cal]; mad = lambda x: np.median(np.abs(x - np.median(x))) * 1.4826
    kq = (q_thr(c, 1 - a) - np.median(c)) / mad(c)
    out = {}
    for mode in ("natural", "decoy"):
        for e in (0.0, 0.02, 0.05, 0.10, 0.20):
            k = int(round(e * n)); res = {r: [] for r in rules}
            for _ in range(reps):
                pool = scores[rng.choice(ben1, n - k, replace=False)]
                con = scores[rng.choice(att1, k, replace=False)] if mode == "natural" else np.full(k, 1e6)
                pool = np.r_[pool, con]
                for r, q in rules.items():
                    t = q_thr(pool, q) if q is not None else float(np.median(pool) + kq * mad(pool))
                    res[r].append(((ben2 > t).mean(), (att2 > t).mean()))
            out[(mode, e)] = {r: tuple(np.mean(v, 0)) for r, v in res.items()}
    return out
if __name__ == "__main__":
    rec = str(REC1)
    R = [json.loads(l) for l in open(rec)]; fam, ev, cal = load_masks(rec)
    doc = np.array([r["doc_id"] for r in R]); dom = np.array([r["domain"] for r in R])
    allres = {}
    for model, (path, layer) in json.loads(sys.argv[1]).items():
        S = np.load(path); s = S[f"final_{layer}"][3]
        o = run(s, fam, doc, dom, ev, cal); allres[model] = {f"{m}|{e}": v for (m, e), v in o.items()}
        print(f"\n== {model} L{layer}  (FPR benign+D / TPR attacks; alpha=0.05, n=200, e_max=0.10)")
        for (m, e), v in o.items():
            print(f"  {m:7s} e={e:.2f}  " + "  ".join(f"{r}: {f:.3f}/{t:.3f}" for r, (f, t) in v.items()))
    json.dump(allres, open(OUT / "contamination_results.json", "w"), indent=1)
