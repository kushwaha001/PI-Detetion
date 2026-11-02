@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run_layer_sweep.ps1"
echo Layer sweep finished. You can close this window.
pause
