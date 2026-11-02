"""
Layer-selection rules with explicit statistics, evaluated across models.

Inputs per model (from layer_sweep_analyze.py): per_layer.json (bootstrap ID curves, oracle outcomes) and
probe_scores.npz (held-out-family probe scores per layer). Records give the masks.

Rules (all label-free; they read only the benign ID curve of metro-train prompts):
  strict        ALS FEPoID on the bootstrap-mean TwoNN curve
  smooth        FEPoID on the 3-layer moving average (last night's heuristic)
  sig(z)        bootstrap-significant first peak: the first local maximum p of the mean curve m such that no layer in
                the forward horizon (p, p+w] exceeds it significantly:
                    max_j (m_j - m_p) / sqrt(se_p^2 + se_j^2) < z
                se_l = std of the B bootstrap replicates (subsampling SE). Replaces the ad hoc smoothing with a test.
  knee(eps)     saturation layer: min{ l : m_l >= (1 - eps) * max_j m_j }  -- the first layer at which the
                representation has reached (1-eps) of its maximum intrinsic complexity.
  each also on the Levina-Bickel MLE curve (lb_*), a larger-scale ID estimator.
Fusion (false alarms on benign instructions): z-score the probe scores of layer l* and of the last layer with the
metro-cal benign mean/std, combine with min (both layers must agree) or mean, calibrate a 5% conformal threshold on
metro-cal benign, and report SQuAD AUROC / benign FPR / family-D FPR / family-C TPR.
"""
import json, math, sys, os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ALPHA = 0.05
W = 5

def fepoid(idc, w=W):
    L = len(idc); peaks = [i for i in range(L) if (i == 0 or idc[i] > idc[i - 1]) and (i == L - 1 or idc[i] >= idc[i + 1])]
    for p in peaks:
        h = min(p + w, L - 1)
        mono = all(idc[j] < idc[j + 1] for j in range(p + 1, h)) if h > p + 1 else False
        if idc[p] < idc[h] and mono: continue
        return p + 1
    return (peaks[0] if peaks else 0) + 1

def smooth(c, k=3):
    c = np.asarray(c, float); pad = np.r_[c[0], c, c[-1]]
    return np.convolve(pad, np.ones(k) / k, mode="valid")

def sig_peak(boots, z=1.645, w=W):
    """boots: B x L matrix of bootstrap ID curves. Returns the first statistically effective peak (1-based layer)."""
    m = boots.mean(0); se = boots.std(0, ddof=1) + 1e-9; L = len(m)
    for p in range(L):
        if p > 0 and m[p] <= m[p - 1]: continue           # must be a rise into p
        if p < L - 1 and m[p] < m[p + 1]: continue        # local max of the mean curve
        later = range(p + 1, min(p + w, L - 1) + 1)
        if all((m[j] - m[p]) / math.sqrt(se[p] ** 2 + se[j] ** 2) < z for j in later):
            return p + 1
    return int(np.argmax(m)) + 1

def knee(m, eps=0.05):
    m = np.asarray(m, float); thr = (1 - eps) * m.max()
    return int(np.argmax(m >= thr)) + 1

def near_top_peak(m, eps=0.10):
    """First local maximum of the curve whose height is within eps of the global maximum (snaps the knee to a peak)."""
    m = np.asarray(m, float); thr = (1 - eps) * m.max(); L = len(m)
    for p in range(L):
        if m[p] >= thr and (p == 0 or m[p] > m[p - 1]) and (p == L - 1 or m[p] >= m[p + 1]): return p + 1
    return int(np.argmax(m)) + 1

def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))

def conformal_thr(s, alpha=ALPHA):
    s = np.sort(s); n = len(s); k = min(n - 1, int(math.ceil((n + 1) * (1 - alpha))) - 1); return float(s[k])

def load_masks(records):
    recs = [json.loads(l) for l in open(records, encoding="utf-8")]
    dom = np.array([r["domain"] for r in recs]); split = np.array([r["split"] for r in recs]); fam = np.array([r["family"] for r in recs])
    doc = np.array([r["doc_id"] for r in recs])
    mt = sorted({d for d, a, b in zip(doc, dom, split) if a == "metro" and b == "test"}, key=lambda s: int(s[1:]))
    halfB = set(mt[1::2])
    ev = {"metro": np.array([d in halfB for d in doc]) & (split == "test"), "squad": (dom == "squad") & (split == "test")}
    cal = (split == "cal") & (fam == "benign")
    return fam, ev, cal

def operating(sc_all, sc_fam, fam, ev, cal):
    """sc_all: N scores from the all-family probe; sc_fam: dict A/B/C -> held-out-family scores."""
    thr = conformal_thr(sc_all[cal]); out = {}
    for dn, msk in ev.items():
        ben = msk & (fam == "benign")
        out[f"{dn}_auroc"] = float(np.mean([auroc(sc_fam[f][msk & (fam == f)], sc_fam[f][ben]) for f in "ABC"]))
        out[f"{dn}_fpr_benign"] = float((sc_all[ben] > thr).mean())
        out[f"{dn}_fpr_D"] = float((sc_all[msk & (fam == "D")] > thr).mean())
        out[f"{dn}_tpr_C"] = float((sc_all[msk & (fam == "C")] > thr).mean())
    return out

def recal_target(s, fam, ev, doc, dom, n=50, reps=20, seed=0):
    """Threshold re-set on n clean target-domain docs (benign + reader-directed benign, i.e. normal traffic) from one
    half of SQuAD; rates measured on the other half. Returns mean over reps of FPR benign, FPR D, TPR A/B/C."""
    rng = np.random.default_rng(seed)
    sq = sorted({d for d in doc[dom == "squad"]}, key=lambda x: int(x[1:]))
    h1 = np.isin(doc, sq[0::2]) & ev["squad"]; h2 = np.isin(doc, sq[1::2]) & ev["squad"]
    pool = np.where(h1 & np.isin(fam, ["benign", "D"]))[0]; vals = []
    for _ in range(reps):
        t = conformal_thr(s[rng.choice(pool, n, replace=False)])
        vals.append([(s[h2 & (fam == f)] > t).mean() for f in ("benign", "D", "A", "B", "C")])
    m = np.mean(vals, 0)
    return dict(zip(["fpr_benign", "fpr_D", "tpr_A", "tpr_B", "tpr_C"], map(float, m)))

def zcal(s, cal):
    return (s - s[cal].mean()) / (s[cal].std() + 1e-9)

def analyse(res_dir, records, feats=("final", "delta")):
    rows = json.load(open(Path(res_dir) / "per_layer.json"))
    S = np.load(Path(res_dir) / "probe_scores.npz")
    fam, ev, cal = load_masks(records)
    fin = np.isfinite(S[[k for k in S.files][0]][3]); ev = {k: v & fin for k, v in ev.items()}; cal = cal & fin
    recs = [json.loads(l) for l in open(records, encoding="utf-8")]
    doc = np.array([r["doc_id"] for r in recs]); dom = np.array([r["domain"] for r in recs])
    out = {}
    for kind in feats:
        R = sorted([r for r in rows if r["feat"] == kind], key=lambda r: r["layer"])
        if not R: continue
        L = len(R)
        sq = np.array([r["squad_auroc_mean"] for r in R]); mt = np.array([r["metro_auroc_mean"] for r in R])
        best = int(np.argmax(sq)) + 1
        curves = {}
        for est in ("id", "lb"):
            if f"{est}_benign_boots" not in R[0]: continue
            B = np.array([r[f"{est}_benign_boots"] for r in R]).T          # boots x L
            curves[est] = B
        picks = {}
        for est, B in curves.items():
            m = B.mean(0)
            picks[f"{est}:strict"] = fepoid(list(m))
            picks[f"{est}:smooth"] = fepoid(list(smooth(m)))
            for z in (1.0, 1.645, 2.326): picks[f"{est}:sig(z={z})"] = sig_peak(B, z)
            for e in (0.03, 0.05, 0.10): picks[f"{est}:knee(eps={e})"] = knee(m, e)
            picks[f"{est}:neartop(eps=0.1)"] = near_top_peak(m, 0.10)
        picks["pishield:val_acc"] = max(R, key=lambda r: (r["val_acc"], -r["val_loss"]))["layer"]
        picks["fixed:0.5 depth"] = max(1, round(L * 0.5)); picks["last"] = L
        table = {p: {"layer": l, "squad_auroc": float(sq[l - 1]), "regret_squad": float(sq.max() - sq[l - 1]),
                     "regret_metro": float(mt.max() - mt[l - 1])} for p, l in picks.items()}
        # bootstrap stability of each rule (fraction of replicates landing within 1 layer of the full-data pick)
        stab = {}
        for est, B in curves.items():
            nb = B.shape[0]
            for name, fn in ((f"{est}:smooth", lambda b: fepoid(list(smooth(b)))), (f"{est}:knee(eps=0.05)", lambda b: knee(b, 0.05)),
                             (f"{est}:knee(eps=0.1)", lambda b: knee(b, 0.10)), (f"{est}:neartop(eps=0.1)", lambda b: near_top_peak(b, 0.10))):
                ls = [fn(B[i]) for i in range(nb)]; stab[name] = ls
            # sig needs replicates; leave-one-out over bootstraps
            ls = [sig_peak(np.delete(B, i, 0)) for i in range(nb)]; stab[f"{est}:sig(z=1.645)"] = ls
        # fusion with the last layer
        fus = {}
        def sc(l, k): return S[f"{kind}_{l}"][k]
        for p in ("id:knee(eps=0.05)", "id:sig(z=1.645)", "id:smooth"):
            if p not in picks: continue
            l = picks[p]
            single = operating(sc(l, 3), {f: sc(l, i) for i, f in enumerate("ABC")}, fam, ev, cal)
            fus[f"{p} alone (L{l})"] = single
            for how, op in (("min", np.minimum), ("mean", lambda a, b: (a + b) / 2)):
                a = {i: op(zcal(sc(l, i), cal), zcal(sc(L, i), cal)) for i in range(4)}
                fus[f"{p} L{l} {how} L{L}"] = operating(a[3], {f: a[i] for i, f in enumerate("ABC")}, fam, ev, cal)
        fus[f"last alone (L{L})"] = operating(sc(L, 3), {f: sc(L, i) for i, f in enumerate("ABC")}, fam, ev, cal)
        rc = {}
        for p in ("id:sig(z=1.645)", "id:knee(eps=0.1)", "id:neartop(eps=0.1)", "pishield:val_acc", "fixed:0.5 depth", "last"):
            l = picks[p]; rc[f"{p} (L{l})"] = recal_target(S[f"{kind}_{l}"][3], fam, ev, doc, dom)
        rc[f"oracle (L{best})"] = recal_target(S[f"{kind}_{best}"][3], fam, ev, doc, dom)
        rc_all = {l: recal_target(S[f"{kind}_{l}"][3], fam, ev, doc, dom, reps=10) for l in range(1, L + 1)}
        out[kind] = {"recal": rc, "recal_by_layer": rc_all, "L": L, "oracle_squad": best, "oracle_squad_auroc": float(sq.max()), "picks": table, "stability": stab,
                     "fusion": fus, "id_mean": {e: list(map(float, B.mean(0))) for e, B in curves.items()},
                     "id_se": {e: list(map(float, B.std(0, ddof=1))) for e, B in curves.items()}}
    return out

if __name__ == "__main__":
    records = sys.argv[1]; results = {}
    for d in sys.argv[2:]:
        name = Path(d).name; results[name] = analyse(d, records)
        for kind, r in results[name].items():
            print(f"\n=== {name} [{kind}]  L={r['L']}  oracle SQuAD L{r['oracle_squad']} ({r['oracle_squad_auroc']:.3f})")
            for p, t in r["picks"].items(): print(f"  {p:22s} L{t['layer']:>2}  auroc {t['squad_auroc']:.3f}  regret {t['regret_squad']:.3f}")
            print("  stability:", {k: v for k, v in r["stability"].items()})
            for k, v in r["recal"].items():
                print(f"  RECAL {k:28s} FPR ben {v['fpr_benign']:.2f}  FPR D {v['fpr_D']:.2f}  TPR A/B/C {v['tpr_A']:.2f}/{v['tpr_B']:.2f}/{v['tpr_C']:.2f}")
            bl = max(r["recal_by_layer"], key=lambda l: r["recal_by_layer"][l]["tpr_C"] + r["recal_by_layer"][l]["tpr_B"])
            print(f"  RECAL best layer by TPR B+C: L{bl}", r["recal_by_layer"][bl])
            for k, v in r["fusion"].items():
                print(f"  {k:40s} AUROC sq {v['squad_auroc']:.3f}  FPR ben sq {v['squad_fpr_benign']:.3f}  FPR D sq {v['squad_fpr_D']:.3f}  FPR D metro {v['metro_fpr_D']:.3f}  TPR C sq {v['squad_tpr_C']:.3f}")
    json.dump(results, open(os.environ.get("OUT", "select_math.json"), "w"), indent=1)
