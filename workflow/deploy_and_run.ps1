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
# Live status file: rewritten every tick to hold one current line per workflow (+ the AE
# footer), each line only once, with growing bars — a clean dashboard even for the
# detached run. View with:  Get-Content -Wait '$LogDir\progress.status'
`$env:PROGRESS_STATUS_FILE = '$LogDir\progress.status'

# Load big-feature toggles from .env.workflow (if present), then default unset ones to true.
`$envWf = Join-Path '$ScriptDir' '.env.workflow'
`$toggles = @{}
if (Test-Path `$envWf) {
    foreach (`$line in Get-Content `$envWf) {
        `$t = `$line.Trim()
        if (`$t -eq "" -or `$t.StartsWith("#")) { continue }
        `$kv = `$t -split '=', 2
        if (`$kv.Count -eq 2) {
            `$val = (`$kv[1] -split '#', 2)[0].Trim()
            `$toggles[`$kv[0].Trim()] = `$val
        }
    }
}
function Is-On([string]`$name) {
    `$v = `$toggles[`$name]
    if (`$null -eq `$v -or `$v -eq "") { return `$true }   # default ON
    return @('1','true','yes','on') -contains `$v.ToLower()
}

`$bootSources = @()
if (Is-On 'ENABLE_OPENALEX') { `$bootSources += 'openalex' }
if (Is-On 'ENABLE_CROSSREF') { `$bootSources += 'crossref' }
if (Is-On 'ENABLE_ARXIV')    { `$bootSources += 'arxiv' }
if (Is-On 'ENABLE_S2')       { `$bootSources += 's2' }

`$extSources = @('copy')
if (Is-On 'ENABLE_PDF')      { `$extSources += 'pdf' }
if (Is-On 'ENABLE_UEFISCDI') { `$extSources += 'uefiscdi' }
if (Is-On 'ENABLE_OPENALEX') { `$extSources += 'openalex' }
if (Is-On 'ENABLE_S2')       { `$extSources += 's2' }
if (Is-On 'ENABLE_FILL')     { `$extSources += 'fill' }
if ((Is-On 'ENABLE_LLM_FILL') -or (Is-On 'ENABLE_LLM_ABSTRACT_FILL')) { `$extSources += 'llm_fill' }

if ((Is-On 'ENABLE_BOOTSTRAP') -and `$bootSources.Count -gt 0) {
    `$bs = `$bootSources -join ','
    Write-Host "===== [1/4] bootstrap: fresh multi-source fetch -> raw.db (sources: `$bs) ====="
    Write-Host "      (resume-safe: re-running this step alone continues where it left off)"
    uv run python -m workflow.src.main_extend_db bootstrap --target-rows $TargetRows --sources `$bs
    if (`$LASTEXITCODE -ne 0) { throw "bootstrap failed" }
} else {
    Write-Host "===== [1/4] bootstrap SKIPPED (ENABLE_BOOTSTRAP=false or no sources enabled) ====="
}

if (Is-On 'ENABLE_EXTEND_DB') {
    `$es = `$extSources -join ','
    Write-Host "===== [2/4] extend: raw.db + legacy DBs -> extended.db -> training.db (sources: `$es) ====="
    uv run python -m workflow.src.main_extend_db extend --sources `$es
    if (`$LASTEXITCODE -ne 0) { throw "extend failed" }
} else {
    Write-Host "===== [2/4] database extension SKIPPED (ENABLE_EXTEND_DB=false) ====="
}

if (Is-On 'ENABLE_ANALYSE') {
    Write-Host "===== [3/4] dataset statistics -> workflow/data/analysis/ ====="
    uv run python -m workflow.src.main_analyse_dataset --db extended
} else {
    Write-Host "===== [3/4] dataset statistics SKIPPED (ENABLE_ANALYSE=false) ====="
}

if (Is-On 'ENABLE_TRAINING') {
    Write-Host "===== [4/4] run_all_workflows: the training + benchmark queue ====="
    uv run python -m workflow.run_all_workflows
    if (`$LASTEXITCODE -ne 0) { throw "run_all_workflows failed" }
} else {
    Write-Host "===== [4/4] training SKIPPED (ENABLE_TRAINING=false) ====="
}

Write-Host ""
Write-Host "===== PIPELINE COMPLETE `$(Get-Date) ====="
Write-Host "results -> $RepoDir\workflow\results\  (see results\summary\ for cross-run figures)"
"@ | Set-Content -Encoding utf8 $inner

if ($Foreground) {
    Log "running in the foreground (attached) - progress refreshes live in this console"
    # Run the inner script DIRECTLY to this console (a real TTY) so the live progress lines
    # refresh in place. Do NOT pipe through Tee-Object here: PowerShell's object pipeline
    # re-emits each line as a separate string with its own newline, which permanently breaks
    # the in-place carriage-return/ANSI refresh (that was the source of the appended,
    # repeating progress lines). FORCE_PROGRESS_ANSI=1 makes the display use ANSI redraw.
    # For a logged run, use the default (detached) mode instead, which writes to $runLog.
    $env:FORCE_PROGRESS_ANSI = "1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $inner
    Remove-Item Env:\FORCE_PROGRESS_ANSI -ErrorAction SilentlyContinue
} else {
    Log "launching detached (hidden window) - safe to close this terminal / sleep the laptop"
    # Detached to a plain log file (no console/TTY): the progress display auto-throttles to
    # one line per real state change instead of per-second ETA spam.
    $psi = Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $inner
    ) -WindowStyle Hidden -RedirectStandardOutput $runLog -RedirectStandardError $runErr -PassThru
    $psi.Id | Out-File (Join-Path $LogDir "pipeline.pid")
    $statusFile = Join-Path $LogDir 'progress.status'
    Log "started, PID $($psi.Id)."
    Write-Host "        Full scrolling log:      Get-Content -Wait $runLog"
    Write-Host "        Live progress dashboard: while(1){Clear-Host;Get-Content '$statusFile';Start-Sleep 1}"
    Write-Host "                                 (one line per workflow, updated in place, + AE footer)"
}