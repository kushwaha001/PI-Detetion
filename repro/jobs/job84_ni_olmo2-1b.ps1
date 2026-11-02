$env:SRC_DIR = "D:/ldg-sweep/olmo2-1b"; $env:BENCH_DIR = "D:/ldg-sweep/bench/olmo2-1b"; $env:RES_DIR = "C:\Users\ASUS\latent-drift-guard\results_sweep\notinject\olmo2-1b"; $env:RECORDS = "C:\Users\ASUS\latent-drift-guard\results_sweep\bench\records2.jsonl"
& "C:\Users\ASUS\anaconda3\python.exe" -u "C:\Users\ASUS\latent-drift-guard\ni_eval.py"
exit $LASTEXITCODE
