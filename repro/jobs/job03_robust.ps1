$env:CPU = "1"; $env:N_BOOT = "10"; $env:DEDUP = "1"; $env:FEATS = "final,delta"
$env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\robust_boot10_dedup"
& "C:\Users\ASUS\anaconda3\python.exe" -u "C:\Users\ASUS\latent-drift-guard\layer_sweep_analyze.py"
exit $LASTEXITCODE
