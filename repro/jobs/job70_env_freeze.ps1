$d = "C:\Users\ASUS\latent-drift-guard\repro\environment"; New-Item -ItemType Directory -Force $d | Out-Null
& "D:\ldg-venv\Scripts\python.exe" -m pip freeze | Out-File -Encoding utf8 "$d\gpu_env_ldg-venv.txt"
& "D:\ldg-venv\Scripts\python.exe" --version | Out-File -Encoding utf8 "$d\gpu_env_python.txt"
& "C:\Users\ASUS\anaconda3\python.exe" -m pip freeze | Out-File -Encoding utf8 "$d\cpu_env_anaconda.txt"
& "C:\Users\ASUS\anaconda3\python.exe" --version | Out-File -Encoding utf8 "$d\cpu_env_python.txt"
nvidia-smi | Out-File -Encoding utf8 "$d\nvidia-smi.txt"
exit 0
