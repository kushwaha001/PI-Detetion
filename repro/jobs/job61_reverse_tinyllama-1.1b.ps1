$env:SRC_DIR = "D:/ldg-sweep/tinyllama-1.1b"; $env:BENCH_DIR = "D:/ldg-sweep/bench/tinyllama-1.1b"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\reverse\tinyllama-1.1b"
& "C:\Users\ASUS\anaconda3\python.exe" -u "C:\Users\ASUS\latent-drift-guard\reverse_eval.py"
exit $LASTEXITCODE
