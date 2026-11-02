$env:MODEL = "D:/ldg-models/Qwen2.5-7B-Instruct"; $env:OUT_DIR = "D:/ldg-sweep/bench/qwen2.5-7b"; $env:LOG_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\qwen2.5-7b"; $env:QUANT = "4bit"; $env:DEVICE_MAP = "layers_gpu"; $env:DOUBLE_QUANT = "1"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:SKIP_DOMAINS = "notinject"; $env:RECORDS = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\records2.jsonl"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_extract2.py"
exit $LASTEXITCODE
