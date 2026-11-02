"""Baseline: ProtectAI DeBERTa-v3 prompt-injection classifier on every document (BIPIA/NotInject records and the metro
/SQuAD Latent Drift Guard records). Long documents are scored with 512-token windows (stride 256), max over windows."""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE = Path(__file__).resolve().parent; RES = Path(os.environ["RES_DIR"]); RES.mkdir(parents=True, exist_ok=True)
M = os.environ.get("CLS_MODEL", "D:/ldg-models/deberta-v3-base-prompt-injection-v2")
tok = AutoTokenizer.from_pretrained(M); mdl = AutoModelForSequenceClassification.from_pretrained(M).to("cuda").eval().half()
inj = [i for i, l in mdl.config.id2label.items() if "INJ" in l.upper()][0]; print("labels", mdl.config.id2label, "inj idx", inj, flush=True)
@torch.no_grad()
def score(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]; W, S = 510, 256; best = -1e9
    for st in range(0, max(1, len(ids) - 1), S):
        chunk = [tok.cls_token_id] + ids[st:st + W] + [tok.sep_token_id]
        lo = mdl(input_ids=torch.tensor([chunk], device="cuda")).logits[0].float()
        best = max(best, float(lo[inj] - lo[1 - inj]))
        if st + W >= len(ids): break
    return best
out = {}
for name, path in (("bench", Path(os.environ.get("RECORDS", HERE / "results_sweep" / "bench" / "records2.jsonl"))), ("metro", HERE / "cache_v2" / "records_full.jsonl")):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]; t0 = time.time(); s = np.full(len(recs), np.nan, np.float32); cache = {}
    for i, r in enumerate(recs):
        if name == "metro" and not (r["split"] == "cal" or (r["split"] == "test")): continue
        d = r["doc"]; s[i] = cache[d] if d in cache else cache.setdefault(d, score(d))
        if i % 500 == 0: print(name, i, len(recs), f"{time.time() - t0:.0f}s", flush=True)
    out[name] = s
np.savez_compressed(RES / "deberta_scores.npz", **out); print("DEBERTA DONE", flush=True)
