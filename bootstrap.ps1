param(
    [switch]$WithNews
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
$Extras = if ($WithNews) { ".[research,dev,news]" } else { ".[research,dev]" }
& ".venv\Scripts\python.exe" -m pip install -e $Extras
& ".venv\Scripts\python.exe" -m ipykernel install --user --name market-state-lab --display-name "Python (Market State Lab)"
Write-Host "Environment ready: $Root\.venv"
