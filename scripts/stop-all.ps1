<#
.SYNOPSIS
    Stop the Varnam OCR services listening on the API and web ports.

.DESCRIPTION
    Finds whatever process is listening on each port and stops it. Killing by
    port (instead of by image name) matters: killing every python.exe or
    node.exe would also take down unrelated projects on the machine.

.PARAMETER Ports
    Ports to free. Defaults to 8010 (API) and 3000 (web).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\stop-all.ps1
#>
param(
    [int[]]$Ports = @(8010, 3000)
)

$stopped = 0
foreach ($port in $Ports) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $listeners) {
        Write-Host "Port $port : nothing running" -ForegroundColor Gray
        continue
    }

    foreach ($listener in $listeners) {
        $processId = $listener.OwningProcess
        try {
            $name = (Get-Process -Id $processId -ErrorAction Stop).ProcessName
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "Port $port : stopped $name (pid $processId)" -ForegroundColor Green
            $stopped++
        } catch {
            Write-Host "Port $port : could not stop pid $processId (try an elevated window)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Stopped $stopped process(es)." -ForegroundColor Cyan
