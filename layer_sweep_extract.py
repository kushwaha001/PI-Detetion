"""
Layer sweep, pass 1 (GPU): all-layer hidden states for every cached Latent Drift Guard v2 record.

Reuses the records, prompts and model helpers of extract_v2.py unchanged, so the states line up 1:1 with
cache_v2/records_full.jsonl. What extract_v2 kept only at layers 9/14/19 is kept here at every layer:
  final       last prompt token (assistant header) - the PIShield read-out position
  data_last   last token of the document segment
  task_final  last prompt token of the task-only prompt, one per task (for the drift delta)
Output: float16 memmaps, layout [layer (0 = embeddings), record, hidden], in OUT_DIR (default D:/ldg-sweep).
Resumable: done.npy marks finished records (0 todo, 1 ok, 2 failed); a kill loses at most FLUSH records.
"""
import os, sys, json, time, shutil
from pathlib import Path
os.environ.setdefault("SCALE", "full")
os.environ.setdefault("MODEL", "D:/ldg-models/Qwen2.5-1.5B-Instruct")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import extract_v2 as X   # same prompts, chat template, embedding-on-CPU trick
import quant_loader; quant_loader.patch(X)
if os.environ.get("DTYPE") == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    X.DTYPE = torch.bfloat16      # OLMo-2 was trained in bf16; fp16 activations can overflow

OUT = Path(os.environ.get("OUT_DIR", "D:/ldg-sweep")); OUT.mkdir(parents=True, exist_ok=True)
FLUSH = 200
SAVE_DL = os.environ.get("SAVE_DATA_LAST", "0") == "1"   # off by default: weakest feature, halves the disk use
LOGDIR = Path(os.environ.get("LOG_DIR", HERE / "results_sweep")); LOGDIR.mkdir(parents=True, exist_ok=True)   # small log on C so progress is visible without D

def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    for d in (OUT, LOGDIR):
        with open(d / "extract.log", "a", encoding="utf-8") as fh: fh.write(s + "\n")

def main():
    if os.name == "nt":   # keep Modern Standby from suspending the run
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    recs = [json.loads(l) for l in open(X.OUT / f"records_{X.SCALE}.jsonl", encoding="utf-8")]
    tasks = json.load(open(X.OUT / f"meta_{X.SCALE}.json"))["tasks"]
    N = len(recs)
    tok, llm = X.load_llm()
    L1 = llm.config.num_hidden_layers + 1; D = llm.config.hidden_size
    need = (2 if SAVE_DL else 1) * L1 * N * D * 2 + L1 * len(tasks) * D * 2
    free = shutil.disk_usage(OUT).free
    log(f"model {X.MODEL} | blocks {L1 - 1} hidden {D} | records {N} | need {need / 1e9:.2f} GB, free {free / 1e9:.1f} GB on {OUT}")
    if not (OUT / "final.f16").exists() and free < need + 2e9:
        raise SystemExit(f"not enough free space in {OUT}")
    if X.DEVICE == "cuda":
        X.embed_to(llm, "cpu"); torch.cuda.empty_cache()

    shape = (L1, N, D)
    def mm(name):
        p = OUT / name
        return np.memmap(p, np.float16, "r+" if p.exists() else "w+", shape=shape)
    fin = mm("final.f16"); dl = mm("data_last.f16") if SAVE_DL else None
    done_fn = OUT / "done.npy"
    done = np.load(done_fn) if done_fn.exists() else np.zeros(N, np.int8)
    json.dump({"N": N, "L1": L1, "D": D, "model": X.MODEL, "records": str(X.OUT / f"records_{X.SCALE}.jsonl"),
               "layout": "[layer (0 = embeddings), record, hidden], float16",
               "files": ["final.f16"] + (["data_last.f16"] if SAVE_DL else []) + ["task_final.npy", "done.npy"]},
              open(OUT / "meta.json", "w"), indent=1)

    if not (OUT / "task_final.npy").exists():
        tf = np.zeros((len(tasks), L1, D), np.float16)
        for t, task in enumerate(tasks):
            _, enc, _, _ = X.build_prompt(tok, task, None)
            out = X.run_base(llm, enc, hidden=True)
            tf[t] = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).float().cpu().numpy()
            del out
        np.save(OUT / "task_final.npy", tf)
        log("task-only states done")

    sk = quant_loader.skip_mask(recs)
    if sk is not None:
        done[np.array(sk) & (done == 0)] = 3; log(f"subset: {int((done == 3).sum())} test records skipped")
    todo = np.where(done == 0)[0]
    log(f"{len(todo)} records to do")
    t0 = time.time(); nonfinite = 0
    log(f"dtype {X.DTYPE}")
    for k, i in enumerate(todo, 1):
        r = recs[i]
        _, enc, _, dix = X.build_prompt(tok, tasks[r["task_id"]], r["doc"])
        try:
            out = X.run_base(llm, enc, hidden=True)
            fin[:, i] = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).numpy()
            if not np.isfinite(fin[-1, i]).all(): nonfinite += 1
            if SAVE_DL: dl[:, i] = torch.stack([h[0, dix[-1]].float().cpu() for h in out.hidden_states]).numpy()
            done[i] = 1
            del out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); done[i] = 2; log("OOM, record", int(i))
        if k % FLUSH == 0 or k == len(todo):
            fin.flush(); dl is not None and dl.flush(); np.save(done_fn, done)
            rate = k / (time.time() - t0)
            log(f"  {int((done > 0).sum())}/{N} done  {rate:.2f} rec/s  eta {(len(todo) - k) / rate / 60:.0f} min  non-finite {nonfinite}")
    fin.flush(); dl is not None and dl.flush(); np.save(done_fn, done)
    log(f"EXTRACT DONE ok {int((done == 1).sum())} failed {int((done == 2).sum())}")

if __name__ == "__main__":
    main()
