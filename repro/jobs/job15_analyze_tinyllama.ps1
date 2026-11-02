$env:CPU = "1"; $env:N_BOOT = "10"; $env:DEDUP = "1"; $env:FEATS = "final,delta"; $env:OBEYED = "0"
$env:OUT_DIR = "D:/ldg-sweep/tinyllama-1.1b"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\tinyllama-1.1b"
& "C:\Users\ASUS\anaconda3\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_analyze.py"
exit $LASTEXITCODE
