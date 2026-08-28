$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& ".venv\Scripts\python.exe" "scripts\run_daily.py" @args

