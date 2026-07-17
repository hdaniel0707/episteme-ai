$ErrorActionPreference = "Stop"
Set-Location -LiteralPath 'C:\Users\extre\Git\EpistemeAI'
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
$env:EPAI_DB = "training"
$env:UMAP_OPEN_BROWSER = "false"
$env:PARALLEL_JOBS = "auto"
$env:PARALLEL_LOAD_TARGET = "0.90"
$env:EMBED_BATCH_SIZE = "64"

Write-Host "===== [1/4] bootstrap: fresh multi-source fetch -> raw.db ====="
Write-Host "      (resume-safe: re-running this step alone continues where it left off)"
uv run python -m workflow.src.main_extend_db bootstrap --target-rows 100000
if ($LASTEXITCODE -ne 0) { throw "bootstrap failed" }

Write-Host "===== [2/4] extend: raw.db + legacy DBs -> extended.db -> training.db ====="
uv run python -m workflow.src.main_extend_db extend
if ($LASTEXITCODE -ne 0) { throw "extend failed" }

Write-Host "===== [3/4] dataset statistics -> workflow/data/analysis/ ====="
uv run python -m workflow.src.main_analyse_dataset --db extended

Write-Host "===== [4/4] run_all_workflows: the training + benchmark queue ====="
uv run python -m workflow.run_all_workflows
if ($LASTEXITCODE -ne 0) { throw "run_all_workflows failed" }

Write-Host ""
Write-Host "===== PIPELINE COMPLETE $(Get-Date) ====="
Write-Host "results -> C:\Users\extre\Git\EpistemeAI\workflow\results\  (see results\summary\ for cross-run figures)"
