<#
.SYNOPSIS
  Stop the detached workflow pipeline and free best-effort RAM / VRAM.

.DESCRIPTION
  Kills the launcher process tree, any lingering workflow-related Python jobs,
  and any GPU compute consumers reported by nvidia-smi.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $PSCommandPath
$LogDir = Join-Path $ScriptDir "logs"
$PidFile = Join-Path $LogDir "pipeline.pid"

function Stop-ProcessTree {
    param([int]$ProcId)

    $children = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $ProcId } |
        Select-Object -ExpandProperty ProcessId
    foreach ($child in $children) {
        Stop-ProcessTree -ProcId $child
    }
    try { Stop-Process -Id $ProcId -Force -ErrorAction Stop } catch {}
}

if (Test-Path $PidFile) {
    $pidText = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($pidText -match '^[0-9]+$') {
        Stop-ProcessTree -ProcId ([int]$pidText)
        Write-Host "[stop] killed pipeline pid $pidText"
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'workflow[\\./](deploy_and_run|run_all_workflows|run_pipeline)|workflow[\\./]src[\\./](main_extend_db|main_analyse_dataset|main_train)') } |
    ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
    }

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuPids = & nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^[0-9]+$' } |
        Sort-Object -Unique
    foreach ($gpuPid in $gpuPids) {
        try { Stop-Process -Id ([int]$gpuPid) -Force -ErrorAction Stop } catch {}
    }
}

Write-Host "[stop] pipeline stopped; process memory should be released now."