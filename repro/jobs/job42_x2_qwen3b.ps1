$env:MODEL = "D:/ldg-models/Qwen2.5-3B-Instruct"; $env:OUT_DIR = "D:/ldg-sweep/bench/qwen2.5-3b"; $env:LOG_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\qwen2.5-3b"; $env:QUANT = "4bit"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:RECORDS = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\records2.jsonl"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_extract2.py"
exit $LASTEXITCODE
