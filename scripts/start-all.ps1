<#
.SYNOPSIS
    Start every Varnam OCR service in its own PowerShell window.

.DESCRIPTION
    Opens one window for the FastAPI backend and one for the Next.js frontend,
    then waits until the API reports ready before opening the browser.

    Why separate windows: both servers are long-running and print live logs. A
    single window would interleave them and make a crash in one look like a
    crash in the other.

.PARAMETER ApiPort
    Port for FastAPI. Defaults to 8010 because port 8000 is often already taken
    by another local project.

.PARAMETER WebPort
    Port for Next.js. Defaults to 3000.

.PARAMETER NoBrowser
    Skip opening the browser at the end.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
#>
param(
    [int]$ApiPort = 8010,
    [int]$WebPort = 3000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is <repo>\scripts, so the repo root is its parent.
$root = Split-Path -Parent $PSScriptRoot
$apiDir = Join-Path $root "services\api"
$python = Join-Path $apiDir ".venv\Scripts\python.exe"
$pnpm = Join-Path $env:APPDATA "npm\pnpm.cmd"

if (-not (Test-Path $python)) {
    Write-Host "Python environment not found: $python" -ForegroundColor Red
    Write-Host "Run the setup first (see README.md -> Quickstart)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $pnpm)) {
    Write-Host "pnpm not found: $pnpm" -ForegroundColor Red
    Write-Host "Install it once with:  cmd /c `"npm install -g pnpm@9`"" -ForegroundColor Red
    exit 1
}

# A stale listener on either port would make the new process exit immediately
# with "address already in use", which is confusing to debug later.
foreach ($port in @($ApiPort, $WebPort)) {
    $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        Write-Host "Port $port is already in use. Run scripts\stop-all.ps1 first." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Starting API on port $ApiPort ..." -ForegroundColor Cyan
# Backtick-escaped $env: keeps it literal so the CHILD process sets it, not this one.
$apiCommand = "cd '$apiDir'; `$env:PYTHONPATH='$root;$root\packages'; & '$python' -m uvicorn app.main:app --port $ApiPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCommand

Write-Host "Starting web on port $WebPort ..." -ForegroundColor Cyan
$webCommand = "cd '$root'; & '$pnpm' --filter @varnam/web dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $webCommand

Write-Host "Waiting for the API to become ready ..." -ForegroundColor Cyan
$ready = $false
for ($attempt = 1; $attempt -le 40; $attempt++) {
    try {
        $response = Invoke-WebRequest "http://127.0.0.1:$ApiPort/health/ready" -TimeoutSec 2 -UseBasicParsing
        Write-Host "API ready : $($response.Content)" -ForegroundColor Green
        $ready = $true
        break
    } catch {
        Start-Sleep -Milliseconds 800
    }
}

if (-not $ready) {
    Write-Host "API did not become ready. Check the API window for the error." -ForegroundColor Red
    exit 1
}

Write-Host "Waiting for the web server ..." -ForegroundColor Cyan
$webReady = $false
for ($attempt = 1; $attempt -le 40; $attempt++) {
    try {
        Invoke-WebRequest "http://localhost:$WebPort" -TimeoutSec 3 -UseBasicParsing | Out-Null
        $webReady = $true
        break
    } catch {
        Start-Sleep -Milliseconds 800
    }
}

if ($webReady) {
    Write-Host "Web ready : http://localhost:$WebPort/app" -ForegroundColor Green
} else {
    Write-Host "Web server did not answer yet - check the web window." -ForegroundColor Yellow
}

if (-not $NoBrowser -and $webReady) {
    Start-Process "http://localhost:$WebPort/app"
}

Write-Host ""
Write-Host "Both servers are running in their own windows. Close them to stop," -ForegroundColor Gray
Write-Host "or run:  powershell -ExecutionPolicy Bypass -File scripts\stop-all.ps1" -ForegroundColor Gray
