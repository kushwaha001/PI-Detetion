$env:MODEL = "C:/Users/ASUS/latent-drift-guard/models/Qwen2.5-0.5B-Instruct"; $env:SRC_DIR = "D:/ldg-sweep/qwen2.5-0.5b"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\attack2\qwen2.5-0.5b-L22"
$env:ATTACK_LAYERS = "22"; $env:STEPS = "80"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\adaptive_attack2.py"
exit $LASTEXITCODE
