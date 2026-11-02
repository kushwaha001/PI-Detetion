$env:CPU = "1"; $env:N_BOOT = "10"; $env:DEDUP = "1"; $env:FEATS = "final,delta"; $env:OBEYED = "1"
$env:OUT_DIR = "D:/ldg-sweep"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\qwen2.5-1.5b"
& "C:\Users\ASUS\anaconda3\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_analyze.py"
exit $LASTEXITCODE
