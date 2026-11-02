# Environment report + model downloads (Hugging Face, ungated). Cache and files go to D: only.
$env:HF_HOME = "D:\hf-cache"; $env:HF_HUB_OFFLINE = "0"; $env:TRANSFORMERS_OFFLINE = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = "D:\ldg-venv\Scripts\python.exe"
& $py -c "import torch, transformers, huggingface_hub as h; print('torch', torch.__version__, '| transformers', transformers.__version__, '| hub', h.__version__); print('gpu', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported(), 'vram GB', round(torch.cuda.get_device_properties(0).total_memory/1e9,2))"
& $py -c "import shutil; print('free GB C:', round(shutil.disk_usage('C:/').free/1e9,1), ' D:', round(shutil.disk_usage('D:/').free/1e9,1))"
foreach ($m in @(@("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0"), @("allenai/OLMo-2-0425-1B-Instruct", "OLMo-2-0425-1B-Instruct"))) {
    & $py -c "from huggingface_hub import snapshot_download as s; p = s('$($m[0])', local_dir='D:/ldg-models/$($m[1])', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.model', '*.txt']); import os; print('downloaded', p, round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)/1e9, 2), 'GB')"
}
& $py -c "import shutil; print('free GB D: after', round(shutil.disk_usage('D:/').free/1e9,1))"
exit 0
