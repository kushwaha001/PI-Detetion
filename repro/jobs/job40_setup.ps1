# 4-bit support (bitsandbytes) in the project venv + Qwen2.5-3B/7B-Instruct original weights to D: (ungated, safetensors only)
$py = "D:\ldg-venv\Scripts\python.exe"
$env:PIP_CACHE_DIR = "D:\pipcache"
& $py -m pip install --disable-pip-version-check "bitsandbytes>=0.45" accelerate 2>&1 | Select-Object -Last 3
& $py -c "import bitsandbytes, accelerate, torch, transformers; print('bnb', bitsandbytes.__version__, 'accelerate', accelerate.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
$env:HF_HOME = "D:\hf-cache"; $env:HF_HUB_OFFLINE = "0"; $env:TRANSFORMERS_OFFLINE = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
foreach ($m in @("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct")) {
    & $py -c "from huggingface_hub import snapshot_download as s; import os; p = s('Qwen/$m', local_dir='D:/ldg-models/$m', allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'tokenizer*']); print('downloaded', p, round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)/1e9, 2), 'GB')"
}
& $py -c "import shutil; print('free GB D:', round(shutil.disk_usage('D:/').free/1e9,1))"
exit 0
