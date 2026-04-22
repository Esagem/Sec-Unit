# Sec-Unit pipeline runner (PowerShell wrapper of run.sh).
#
# Usage:
#   .\run.ps1                          # Run all 9 input combinations
#   .\run.ps1 <pdf1> <pdf2>            # Run a single pair
#   .\run.ps1 --build                  # Build the PyInstaller binary only

$ErrorActionPreference = "Stop"
$VenvDir = "comp5700-venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[+] Creating virtual environment in $VenvDir"
    python -m venv $VenvDir
}

$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Error "Could not find $Activate — venv may not have been created correctly."
    exit 1
}
. $Activate

Write-Host "[+] Installing dependencies"
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

$env:PYTHONIOENCODING = "utf-8"

if ($args.Count -ge 1 -and $args[0] -eq "--build") {
    Write-Host "[+] Building PyInstaller binary"
    python build.py
    Write-Host "[+] Binary at dist/sec-unit.exe"
    exit 0
}

if ($args.Count -eq 2) {
    Write-Host "[+] Running pipeline on $($args[0]) and $($args[1])"
    python main.py $args[0] $args[1]
}
else {
    Write-Host "[+] Running pipeline on all 9 input combinations"
    python main.py --all
}
