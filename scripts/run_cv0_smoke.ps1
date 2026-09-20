# Run the CV-0 smoke tests. Terminal output is ALSO captured to the run dir by the
# in-process tee; Start-Transcript here is an OS-level belt-and-suspenders copy.

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "runs\_transcripts" | Out-Null
Start-Transcript -Path "runs\_transcripts\smoke_$stamp.txt" -Force | Out-Null

Write-Host "== mirage smoke mock (GPU-free, byte-exact) =="
& $py -m mirage_persist.cli smoke mock
$mock = $LASTEXITCODE

if ($args -contains "--hf") {
    Write-Host "== mirage smoke hf (tiny real model on the GPU) =="
    & $py -m mirage_persist.cli smoke hf
}

Stop-Transcript | Out-Null
Write-Host "mock smoke exit code: $mock"
exit $mock
