$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
$SitePackages = & ".venv\Scripts\python.exe" -c "import site; print(site.getsitepackages()[0])"
$SourcePath = Join-Path $Root "src"
$PthPath = Join-Path $SitePackages "market_state_lab_src.pth"
[System.IO.File]::WriteAllText(
    $PthPath,
    "$SourcePath$([Environment]::NewLine)",
    [System.Text.UTF8Encoding]::new($false)
)
& ".venv\Scripts\python.exe" -m ipykernel install --user --name market-state-lab --display-name "Python (Market State Lab)"
Write-Host "Environment ready: $Root\.venv"
