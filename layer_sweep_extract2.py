"""
All-layer last-prompt-token states for an arbitrary records file (records2.jsonl: BIPIA + NotInject).
Same prompt construction as extract_v2.py (system prompt + "Task: ...\\n\\nDocument:\\n..."), so probes trained on the
metro states transfer directly. Output in OUT_DIR: final.f16 [L1, N, D], task_states.npy [U, L1, D], task_idx.npy [N],
done.npy, meta.json. Resumable.
"""
import os, sys, json, time, shutil
from pathlib import Path
os.environ.setdefault("SCALE", "full"); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import extract_v2 as X
import quant_loader; quant_loader.patch(X)
if os.environ.get("DTYPE") == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(): X.DTYPE = torch.bfloat16
OUT = Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
REC = Path(os.environ.get("RECORDS", HERE / "results_sweep" / "bench" / "records2.jsonl"))
LOGDIR = Path(os.environ["LOG_DIR"]); LOGDIR.mkdir(parents=True, exist_ok=True)
MAXTOK = int(os.environ.get("MAXTOK", "1024"))
def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True)
    for d in (OUT, LOGDIR):
        with open(d / "extract2.log", "a", encoding="utf-8") as fh: fh.write(s + "\n")
def main():
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    recs = [json.loads(l) for l in open(REC, encoding="utf-8")]; N = len(recs)
    tasks = sorted({r["task"] for r in recs}); tix = {t: i for i, t in enumerate(tasks)}
    tok, llm = X.load_llm(); L1 = llm.config.num_hidden_layers + 1; D = llm.config.hidden_size
    log(f"model {X.MODEL} dtype {X.DTYPE} | L {L1 - 1} D {D} | records {N} tasks {len(tasks)} | free {shutil.disk_usage(OUT).free / 1e9:.0f} GB")
    if X.DEVICE == "cuda": X.embed_to(llm, "cpu"); torch.cuda.empty_cache()
    p = OUT / "final.f16"; fin = np.memmap(p, np.float16, "r+" if p.exists() else "w+", shape=(L1, N, D))
    done = np.load(OUT / "done.npy") if (OUT / "done.npy").exists() else np.zeros(N, np.int8)
    np.save(OUT / "task_idx.npy", np.array([tix[r["task"]] for r in recs]))
    json.dump({"N": N, "L1": L1, "D": D, "model": X.MODEL, "records": str(REC), "tasks": tasks}, open(OUT / "meta.json", "w"), indent=1)
    if not (OUT / "task_states.npy").exists():
        ts = np.zeros((len(tasks), L1, D), np.float16)
        for i, t in enumerate(tasks):
            _, enc, _, _ = X.build_prompt(tok, t, None); out = X.run_base(llm, enc, hidden=True)
            ts[i] = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).float().cpu().numpy(); del out
        np.save(OUT / "task_states.npy", ts); log("task states done")
    if os.environ.get("RESET_SKIPPED") == "1": done[done == 3] = 0
    skip = set(filter(None, os.environ.get("SKIP_DOMAINS", "").split(",")))
    if skip:
        m = np.array([r["domain"] in skip for r in recs]) & (done == 0); done[m] = 3; log(f"skipping domains {sorted(skip)}: {int(m.sum())} records")
    todo = np.where(done == 0)[0]; t0 = time.time(); nonfinite = 0; truncated = 0
    for k, i in enumerate(todo, 1):
        r = recs[i]; doc = r["doc"]
        for _ in range(6):   # shorten very long documents from the end of the document (keeps the prompt tail intact)
            _, enc, _, dix = X.build_prompt(tok, r["task"], doc)
            if enc["input_ids"].shape[1] <= MAXTOK: break
            doc = doc[: int(len(doc) * 0.8)]; truncated += 1
        try:
            out = X.run_base(llm, enc, hidden=True)
            fin[:, i] = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).float().cpu().numpy(); done[i] = 1; del out
            if not np.isfinite(fin[-1, i]).all(): nonfinite += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); done[i] = 2; log("OOM", int(i))
        if k % 100 == 0 or k == len(todo):
            fin.flush(); np.save(OUT / "done.npy", done); rate = k / (time.time() - t0)
            log(f"  {int((done > 0).sum())}/{N}  {rate:.2f} rec/s  eta {(len(todo) - k) / rate / 60:.0f} min  non-finite {nonfinite} truncated {truncated}")
    fin.flush(); np.save(OUT / "done.npy", done)
    log(f"EXTRACT2 DONE ok {int((done == 1).sum())} failed {int((done == 2).sum())}")
if __name__ == "__main__": main()
