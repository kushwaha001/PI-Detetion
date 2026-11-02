$env:MODEL = "D:/ldg-models/Qwen2.5-3B-Instruct"; $env:OUT_DIR = "D:/ldg-sweep/qwen2.5-3b"; $env:LOG_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\qwen2.5-3b"; $env:QUANT = "4bit"
$env:SUBSET_TEST_TASKS = "0,1,2,3,4,5,6,7,16,17"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_extract.py"
exit $LASTEXITCODE
