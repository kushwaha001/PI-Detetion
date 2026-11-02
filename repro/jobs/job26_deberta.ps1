$env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\deberta"; $env:RECORDS = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\records2.jsonl"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\deberta_baseline.py"
exit $LASTEXITCODE
