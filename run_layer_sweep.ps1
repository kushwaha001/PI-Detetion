# Layer sweep launcher: GPU extraction (all 29 layers) -> CPU analysis.
# Heavy outputs (~1.9 GB of hidden states) go to D:\ldg-sweep; the small results and logs go to results_sweep\ here.
# Re-running is safe: the extraction resumes from D:\ldg-sweep\done.npy.
$root = "C:\Users\ASUS\latent-drift-guard"
$res = "$root\results_sweep"; New-Item -ItemType Directory -Force $res | Out-Null
$log = "$res\run.log"
function Log($m) { "$(Get-Date -Format s) $m" | Out-File -Append -Encoding utf8 $log }
$venv = "D:\ldg-venv\Scripts\python.exe"           # GPU torch env used for extract_v2.py
$base = "C:\Users\ASUS\anaconda3\python.exe"        # sklearn / scipy / matplotlib env used for analyze_v2.py
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:TRANSFORMERS_VERBOSITY = "error"; $env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:SCALE = "full"; $env:MODEL = "D:/ldg-models/Qwen2.5-1.5B-Instruct"; $env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:OUT_DIR = "D:/ldg-sweep"
Set-Location $root
Log "start (pid $PID)"
if (-not (Test-Path $venv)) { Log "ERROR: $venv not found (is drive D: connected?)"; exit 1 }
if (-not (Test-Path "$($env:MODEL)/config.json")) { Log "ERROR: model not found at $($env:MODEL)"; exit 1 }

for ($i = 1; $i -le 5; $i++) {
    $doneFlag = (Test-Path "$res\extract.log") -and (Select-String -Path "$res\extract.log" -Pattern "EXTRACT DONE" -Quiet)
    if ($doneFlag) { break }
    Log "extraction attempt $i"
    & $venv -u "$root\layer_sweep_extract.py" *>> "$res\extract.out"
    Log "extraction exited with $LASTEXITCODE"
}
if (-not (Select-String -Path "$res\extract.log" -Pattern "EXTRACT DONE" -Quiet)) { Log "GAVE UP: extraction incomplete"; exit 1 }

$env:CPU = "1"
Log "analysis"
& $base -u "$root\layer_sweep_analyze.py" *>> "$res\analyze.out"
Log "analysis exited with $LASTEXITCODE"
Log "finished"
