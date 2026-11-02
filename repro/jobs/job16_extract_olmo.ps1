$env:MODEL = "D:/ldg-models/OLMo-2-0425-1B-Instruct"; $env:OUT_DIR = "D:/ldg-sweep/olmo2-1b"; $env:LOG_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\olmo2-1b"; $env:DTYPE = "bf16"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_extract.py"
exit $LASTEXITCODE
