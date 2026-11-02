@echo off
title Layer sweep overnight runner - leave this window open
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0overnight_runner.ps1"
echo Runner stopped.
pause
