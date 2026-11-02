"""4-bit loading for models that do not fit a 4 GB GPU in fp16 (QUANT=4bit). Loads the base model only (no lm_head:
hidden states need none), NF4 weights with fp16 compute, layers that do not fit spill to CPU (GPU_MEM, default 3.3GiB).
Also: SUBSET_TEST_TASKS="0,1,..." marks metro/SQuAD test records of other tasks as skipped (done = 3)."""
import os, types
def patch(X):
    if os.environ.get("QUANT") != "4bit": return
    import torch
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    def load_llm():
        tok = AutoTokenizer.from_pretrained(X.MODEL)
        dq = os.environ.get("DOUBLE_QUANT", "1") == "1"
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=dq, llm_int8_enable_fp32_cpu_offload=True)
        if os.environ.get("DEVICE_MAP") == "layers_gpu":
            # 7B on 4 GB: only the (unquantized, 1 GB) embedding stays on CPU; all decoder layers on the GPU,
            # so no 4-bit module is ever offloaded (bitsandbytes cannot offload double-quantized weights).
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(X.MODEL)
            dm = {"embed_tokens": "cpu", "norm": 0, "rotary_emb": 0}
            dm.update({f"layers.{i}": 0 for i in range(cfg.num_hidden_layers)})
            base = AutoModel.from_pretrained(X.MODEL, quantization_config=q, device_map=dm, torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True).eval()
            # accelerate leaves a CPU-mapped module as a meta tensor; read the embedding matrix straight from the
            # checkpoint and do the lookup ourselves (an exact gather, as extract_v2 already does for small models)
            import json as _j, glob as _g
            from safetensors import safe_open
            key = "model.embed_tokens.weight"; W = None
            idx = os.path.join(X.MODEL, "model.safetensors.index.json")
            files = [os.path.join(X.MODEL, _j.load(open(idx))["weight_map"][key])] if os.path.exists(idx) else _g.glob(os.path.join(X.MODEL, "*.safetensors"))
            for f in files:
                with safe_open(f, framework="pt") as fh:
                    if key in fh.keys(): W = fh.get_tensor(key).to(torch.float16); break
            assert W is not None, "embedding not found in checkpoint"
            X._embed = lambda llm, ids: torch.nn.functional.embedding(ids.cpu(), W).to("cuda")
            print("embedding from checkpoint", tuple(W.shape), flush=True)
        else:
            base = AutoModel.from_pretrained(X.MODEL, quantization_config=q, device_map="auto", torch_dtype=torch.float16,
                                         max_memory={0: os.environ.get("GPU_MEM", "3.3GiB"), "cpu": "24GiB"}, low_cpu_mem_usage=True).eval()
        print("vram after load GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
        print("device map:", {k: str(v) for k, v in list(getattr(base, "hf_device_map", {}).items())[:3]}, "...",
              sorted({str(v) for v in getattr(base, "hf_device_map", {}).values()}), flush=True)
        return tok, types.SimpleNamespace(model=base, config=base.config)
    X.load_llm = load_llm
    X.embed_to = lambda llm, device: None          # accelerate already placed the embedding
def skip_mask(recs):
    s = os.environ.get("SUBSET_TEST_TASKS"); t = os.environ.get("SUBSET_TRAIN_TASKS")
    if not s and not t: return None
    keep = {int(x) for x in s.split(",")} if s else None; keep_tr = {int(x) for x in t.split(",")} if t else None
    return [(keep is not None and r["split"] == "test" and r["task_id"] not in keep) or
            (keep_tr is not None and r["split"] in ("train", "cal") and r["task_id"] not in keep_tr) for r in recs]
