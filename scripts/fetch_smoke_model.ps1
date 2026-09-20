# Fetch the tiny smoke model (Qwen2.5-1.5B-Instruct, ~3 GB) into the HF cache.
# Run once before `mirage smoke hf`.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$model = if ($args.Count -ge 1) { $args[0] } else { "Qwen/Qwen2.5-1.5B-Instruct" }

Write-Host "== Downloading $model into the HuggingFace cache =="
& $py -c @"
from transformers import AutoModelForCausalLM, AutoTokenizer
m = '$model'
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print('fetched', m)
"@
Write-Host "Done."
