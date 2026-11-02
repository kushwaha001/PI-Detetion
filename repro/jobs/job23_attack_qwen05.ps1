$env:MODEL = "C:/Users/ASUS/latent-drift-guard/models/Qwen2.5-0.5B-Instruct"; $env:SRC_DIR = "D:/ldg-sweep/qwen2.5-0.5b"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\attack\qwen2.5-0.5b"
$env:ATTACK_LAYER = "14"; $env:STEPS = "40"; $env:NCAND = "32"; $env:NTRAIN = "6"; $env:NTEST = "80"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
& "D:\ldg-venv\Scripts\python.exe" -u "C:\Users\ASUS\latent-drift-guard\adaptive_attack.py"
exit $LASTEXITCODE
