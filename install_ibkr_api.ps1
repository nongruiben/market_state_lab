param(
    [string]$PythonClientPath = "C:\TWS API\source\pythonclient"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$SetupPy = Join-Path $PythonClientPath "setup.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment is missing. Run .\bootstrap.ps1 first."
}

if (-not (Test-Path -LiteralPath $SetupPy)) {
    throw @"
Official IBKR Python source was not found at:
  $PythonClientPath

Download TWS API Latest for Windows from https://interactivebrokers.github.io/,
accept IBKR's license yourself, install it, and run this script again. You may
also pass a different extracted source path:
  .\install_ibkr_api.ps1 -PythonClientPath 'D:\path\IBJts\source\pythonclient'
"@
}

& $Python -m pip uninstall -y ibapi
& $Python -m pip install --no-build-isolation $PythonClientPath
& $Python -m pip show ibapi

