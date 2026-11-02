$env:MODEL = "D:/ldg-models/Qwen2.5-7B-Instruct"; $env:OUT_DIR = "D:/ldg-sweep/qwen2.5-7b"; $env:LOG_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\qwen2.5-7b"; $env:QUANT = "4bit"; $env:DEVICE_MAP = "auto"; $env:DOUBLE_QUANT = "0"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:SUBSET_TEST_TASKS = "0,1,16,17"; $env:SUBSET_TRAIN_TASKS = "0,1,2,3,4,5,6,7"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_extract.py"
exit $LASTEXITCODE
