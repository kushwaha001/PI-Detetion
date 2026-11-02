# NotInject (benign prompts with trigger words; hard negatives) + ProtectAI DeBERTa-v3 prompt-injection classifier (baseline).
$env:HF_HOME = "D:\hf-cache"; $env:HF_HUB_OFFLINE = "0"; $env:TRANSFORMERS_OFFLINE = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = "D:\ldg-venv\Scripts\python.exe"
& $py -c "from huggingface_hub import snapshot_download as s; import os; p = s('leolee99/NotInject', repo_type='dataset', local_dir='D:/ldg-data/NotInject'); [print('FILE', os.path.join(r, f), os.path.getsize(os.path.join(r, f))) for r, _, fs in os.walk(p) for f in fs if not r.endswith('.cache') and '.cache' not in r]"
& $py -c "from huggingface_hub import snapshot_download as s; import os; p = s('protectai/deberta-v3-base-prompt-injection-v2', local_dir='D:/ldg-models/deberta-v3-base-prompt-injection-v2', allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer*', 'spm*', '*.txt']); print('downloaded', p, round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)/1e9, 2), 'GB')"
exit 0
