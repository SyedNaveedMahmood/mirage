# MIRAGE-Persist environment setup (Windows PowerShell).
# Creates a venv that REUSES the system GPU stack (torch/transformers) so CUDA torch
# is not re-downloaded, then installs the pinned AgentDojo substrate + MIRAGE.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== Creating venv (--system-site-packages) =="
python -m venv .venv --system-site-packages

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "== Installing pinned AgentDojo substrate (editable) =="
& $py -m pip install -e prior_work_repos\agentdojo

Write-Host "== Installing MIRAGE-Persist (editable, with hf + dev extras) =="
& $py -m pip install -e ".[hf,dev]"

Write-Host "== Sanity checks =="
& $py -m mirage_persist.cli version
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
