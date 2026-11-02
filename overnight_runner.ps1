# Overnight job runner for the layer-sweep study.
# Runs every *.ps1 job that appears in results_sweep\queue\ (oldest first), one at a time, logging its output to
# queue\done\<job>.out. Claude drops jobs there through the linked-computer file tools; nothing else is executed.
# Stops when queue\STOP exists, or after 10 hours. Holds a power request so the laptop does not sleep meanwhile.
$root = "C:\Users\ASUS\latent-drift-guard"
$q = "$root\results_sweep\queue"; $doneDir = "$q\done"
New-Item -ItemType Directory -Force $q, $doneDir | Out-Null
$log = "$q\runner.log"
function Log($m) { "$(Get-Date -Format s) $m" | Out-File -Append -Encoding utf8 $log }
try {
    Add-Type -Namespace LDGR -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [LDGR.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (screen may turn off)
} catch { Log "power request failed: $($_.Exception.Message)" }
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:OUT_DIR = "D:/ldg-sweep"
$env:SCALE = "full"; $env:MODEL = "D:/ldg-models/Qwen2.5-1.5B-Instruct"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$deadline = (Get-Date).AddHours(10)
Log "runner started (pid $PID), stops at $deadline"
while ((Get-Date) -lt $deadline) {
    if (Test-Path "$q\STOP") { Log "STOP file found"; break }
    $job = Get-ChildItem "$q\*.ps1" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -First 1
    if ($job) {
        "$(Get-Date -Format s)" | Out-File -Encoding utf8 "$q\heartbeat.txt"
        Log "job $($job.Name) start"
        $out = "$doneDir\$($job.BaseName).out"
        Set-Location $root
        & powershell -NoProfile -ExecutionPolicy Bypass -File $job.FullName *>> $out
        Log "job $($job.Name) exit $LASTEXITCODE"
        Move-Item -Force $job.FullName "$doneDir\$($job.Name)"
    } else {
        "$(Get-Date -Format s) idle" | Out-File -Encoding utf8 "$q\heartbeat.txt"
        Start-Sleep -Seconds 30
    }
}
Log "runner finished"
