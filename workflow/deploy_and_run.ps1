<#
.SYNOPSIS
  Detached-by-default full pipeline launcher for Windows.

.DESCRIPTION
  Run this from inside the cloned repo. It keeps the pipeline running if the
  console is closed, and it writes logs under workflow\logs\.

.PARAMETER TargetRows
  Row target passed to the bootstrap step.
.PARAMETER Foreground
  Keep the run attached to the current console.
#>
[CmdletBinding()]
param(
    [int]$TargetRows = 100000,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoDir = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $ScriptDir "logs"

function Log($msg) { Write-Host "`n[deploy] $msg" -ForegroundColor Cyan }

if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    throw "Run this from a cloned repo checkout."
}

Set-Location -LiteralPath $RepoDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Log "installing uv (package manager)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
}
Log "uv sync - installing the project's pinned dependencies"
uv sync --frozen
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

if (-not (Test-Path ".env")) {
    Copy-Item ".env_sample" ".env"
    Log "created .env from .env_sample."
    Write-Host "        >>> Only needed if you enable keyed features (llm_judge baseline," -ForegroundColor Yellow
    Write-Host "        >>> naip_local weights download, MIRAI generated-paper scoring)." -ForegroundColor Yellow
    Write-Host "        >>> The default pipeline (bootstrap/extend/analyse/run_all_workflows" -ForegroundColor Yellow
    Write-Host "        >>> with specter2_knn + venue_ais baselines) needs NO keys at all." -ForegroundColor Yellow
    Write-Host "        >>> Edit $RepoDir\.env now if you want them, then re-run this script." -ForegroundColor Yellow
} else {
    Log ".env already present - leaving it untouched"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = Join-Path $LogDir "pipeline_$ts.log"
$runErr = Join-Path $LogDir "pipeline_$ts.err"
$inner = Join-Path $LogDir "pipeline_$ts.ps1"

@"
`$ErrorActionPreference = "Stop"
Set-Location -LiteralPath '$RepoDir'
`$env:PATH = "`$env:USERPROFILE\.local\bin;`$env:PATH"
`$env:EPAI_DB = "training"
`$env:UMAP_OPEN_BROWSER = "false"
`$env:PARALLEL_JOBS = "auto"
`$env:PARALLEL_LOAD_TARGET = "0.90"
`$env:EMBED_BATCH_SIZE = "64"

Write-Host "===== [1/4] bootstrap: fresh multi-source fetch -> raw.db ====="
Write-Host "      (resume-safe: re-running this step alone continues where it left off)"
uv run python -m workflow.src.main_extend_db bootstrap --target-rows $TargetRows
if (`$LASTEXITCODE -ne 0) { throw "bootstrap failed" }

Write-Host "===== [2/4] extend: raw.db + legacy DBs -> extended.db -> training.db ====="
uv run python -m workflow.src.main_extend_db extend
if (`$LASTEXITCODE -ne 0) { throw "extend failed" }

Write-Host "===== [3/4] dataset statistics -> workflow/data/analysis/ ====="
uv run python -m workflow.src.main_analyse_dataset --db extended

Write-Host "===== [4/4] run_all_workflows: the training + benchmark queue ====="
uv run python -m workflow.run_all_workflows
if (`$LASTEXITCODE -ne 0) { throw "run_all_workflows failed" }

Write-Host ""
Write-Host "===== PIPELINE COMPLETE `$(Get-Date) ====="
Write-Host "results -> $RepoDir\workflow\results\  (see results\summary\ for cross-run figures)"
"@ | Set-Content -Encoding utf8 $inner

if ($Foreground) {
    Log "running in the foreground (use -Foreground for an attached session)"
    powershell -NoProfile -ExecutionPolicy Bypass -File $inner 2>&1 | Tee-Object -FilePath $runLog
} else {
    Log "launching detached (hidden window) - safe to close this terminal / sleep the laptop"
    $psi = Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $inner
    ) -WindowStyle Hidden -RedirectStandardOutput $runLog -RedirectStandardError $runErr -PassThru
    $psi.Id | Out-File (Join-Path $LogDir "pipeline.pid")
    Log "started, PID $($psi.Id). Monitor with:  Get-Content -Wait $runLog"
}