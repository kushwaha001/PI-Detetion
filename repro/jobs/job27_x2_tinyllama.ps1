$env:MODEL = "D:/ldg-models/TinyLlama-1.1B-Chat-v1.0"; $env:OUT_DIR = "D:/ldg-sweep/bench/tinyllama-1.1b"; $env:LOG_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\tinyllama-1.1b"; $env:DTYPE = "fp16"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:RECORDS = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\records2.jsonl"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_extract2.py"
exit $LASTEXITCODE
