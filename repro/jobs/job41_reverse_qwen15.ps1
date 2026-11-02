$env:SRC_DIR = "D:/ldg-sweep"; $env:BENCH_DIR = "D:/ldg-sweep/bench/qwen2.5-1.5b"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\reverse\qwen2.5-1.5b"
& "C:\Users\ASUS\anaconda3\python.exe" -u "C:\Users\ASUS\latent-drift-guard\reverse_eval.py"
exit $LASTEXITCODE
