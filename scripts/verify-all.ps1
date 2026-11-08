<#
.SYNOPSIS
    Run every check in the project and print one summary table.

.DESCRIPTION
    Covers the four layers that can break independently:

      1. Artifacts  - model and charset actually exist and match
      2. Backend    - ruff, format, mypy, pytest
      3. Frontend   - tsc, eslint, vitest, build
      4. Runtime    - API alive, model loaded, one real OCR request

    Deployment is intentionally NOT covered: it depends on external accounts.
    Use scripts/smoke_test.sh against the deployed URL for that.

.PARAMETER SkipBuild
    Skip `next build` (it is the slowest step, ~40s).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify-all.ps1
#>
param(
    [switch]$SkipBuild
)

$root = Split-Path -Parent $PSScriptRoot
$api = Join-Path $root "services\api"
$python = Join-Path $api ".venv\Scripts\python.exe"
$pnpm = Join-Path $env:APPDATA "npm\pnpm.cmd"
$apiUrl = "http://127.0.0.1:8010"

$env:PYTHONPATH = "$root;$root\packages;$api"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
# ml.eval builds an engine from settings, and the default MODEL_DIR is relative to
# the working directory. Without this it looks in <repo>\models, finds nothing and
# silently falls back to the stub engine.
$env:MODEL_DIR = Join-Path $api "models"
$env:VARNAM_ENGINE = "onnx"

$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param([string]$Layer, [string]$Check, [bool]$Ok, [string]$Detail)
    $results.Add([pscustomobject]@{
        Layer  = $Layer
        Check  = $Check
        Status = $(if ($Ok) { "PASS" } else { "FAIL" })
        Detail = $Detail
    })
    $colour = $(if ($Ok) { "Green" } else { "Red" })
    Write-Host ("[{0}] {1,-10} {2,-22} {3}" -f $(if ($Ok) { "OK" } else { "--" }), $Layer, $Check, $Detail) -ForegroundColor $colour
}

# ---------------------------------------------------------------- 1. artifacts
Write-Host "`n=== 1. ARTIFACTS ===" -ForegroundColor Cyan

$model = Join-Path $api "models\rec.onnx"
$charset = Join-Path $api "models\charset.txt"

$modelOk = Test-Path $model
$modelSize = if ($modelOk) { [math]::Round((Get-Item $model).Length / 1MB, 1) } else { 0 }
Add-Result "artifacts" "rec.onnx" $modelOk "$modelSize MB"

$charsetOk = Test-Path $charset
$classCount = if ($charsetOk) { (Get-Content $charset).Count } else { 0 }
Add-Result "artifacts" "charset.txt" ($charsetOk -and $classCount -gt 10) "$classCount classes"

# The checkpoint must agree with the charset, or the model loads but predicts junk.
$checkpoint = Join-Path $root "artifacts\best.pt"
if (Test-Path $checkpoint) {
    $agreement = & $python -c "import torch,sys; ck=torch.load(r'$checkpoint',map_location='cpu',weights_only=True); print(ck['num_classes'], round(float(ck['val_cer'])*100,2))" 2>&1
    $parts = "$agreement" -split "\s+"
    $match = ($parts[0] -eq "$classCount")
    Add-Result "artifacts" "checkpoint <-> charset" $match "classes=$($parts[0]) valCER=$($parts[1])%"
}

# ------------------------------------------------------------------ 2. backend
Write-Host "`n=== 2. BACKEND ===" -ForegroundColor Cyan
Push-Location $api

& $python -m ruff check . 2>&1 | Out-Null
Add-Result "backend" "ruff" ($LASTEXITCODE -eq 0) "lint"

& $python -m ruff format --check . 2>&1 | Out-Null
Add-Result "backend" "ruff format" ($LASTEXITCODE -eq 0) "style"

& $python -m mypy app 2>&1 | Out-Null
Add-Result "backend" "mypy" ($LASTEXITCODE -eq 0) "types"

$pytestOut = & $python -m pytest -q -p no:cacheprovider 2>&1 | Out-String
$passed = $pytestOut -match "(\d+) passed"
$count = if ($passed) { $Matches[1] } else { "?" }
Add-Result "backend" "pytest" $passed "$count tests"

Pop-Location

# ----------------------------------------------------------------- 3. frontend
Write-Host "`n=== 3. FRONTEND ===" -ForegroundColor Cyan
Push-Location $root

& $pnpm --filter @varnam/web typecheck 2>&1 | Out-Null
Add-Result "frontend" "tsc --noEmit" ($LASTEXITCODE -eq 0) "types"

& $pnpm --filter @varnam/web lint 2>&1 | Out-Null
Add-Result "frontend" "eslint" ($LASTEXITCODE -eq 0) "lint"

$vitestOut = & $pnpm --filter @varnam/web test 2>&1 | Out-String
# Match the "Tests 7 passed" summary line, not "Test Files 1 passed" - matching the
# wrong line reports "1 test" and hides how many assertions actually ran.
$vitestOk = $vitestOut -match "Tests\s+(\d+)\s+passed"
$vitestCount = if ($vitestOk) { $Matches[1] } else { "?" }
Add-Result "frontend" "vitest" $vitestOk "$vitestCount tests"

if (-not $SkipBuild) {
    & $pnpm --filter @varnam/web build 2>&1 | Out-Null
    Add-Result "frontend" "next build" ($LASTEXITCODE -eq 0) "production bundle"
}

Pop-Location

# ------------------------------------------------------------------ 4. runtime
Write-Host "`n=== 4. RUNTIME ===" -ForegroundColor Cyan

$health = $null
try {
    $health = (Invoke-WebRequest "$apiUrl/health/ready" -TimeoutSec 3 -UseBasicParsing).Content
} catch { }
$apiUp = $null -ne $health
Add-Result "runtime" "API /health/ready" $apiUp $(if ($apiUp) { "up" } else { "not running" })

$engineOk = $apiUp -and ($health -match '"engine":"onnx"')
Add-Result "runtime" "engine onnx" $engineOk $(if ($apiUp) { ($health -replace '.*"engine":"([^"]+)".*', '$1') } else { "-" })

try {
    $webStatus = (Invoke-WebRequest "http://localhost:3000" -TimeoutSec 5 -UseBasicParsing).StatusCode
    Add-Result "runtime" "web :3000" ($webStatus -eq 200) "HTTP $webStatus"
} catch {
    Add-Result "runtime" "web :3000" $false "not running"
}

# Run a few real pages through the production pipeline and read the CER the
# evaluator prints. Calling ml.eval keeps this honest: it exercises exactly the
# code the API uses, instead of a bespoke snippet that could drift.
$manifest = Join-Path $root "data\eval_manifest.csv"
if ($apiUp -and (Test-Path $manifest)) {
    Push-Location $root
    $evalOut = & $python -m ml.eval --manifest $manifest --engine onnx --max-samples 8 --out (Join-Path $env:TEMP "varnam-verify.md") 2>&1 | Out-String
    Pop-Location
    $cerOk = $evalOut -match "overall CER=([\d\.]+)%"
    $cer = if ($cerOk) { $Matches[1] } else { "?" }
    Add-Result "runtime" "OCR pipeline" $cerOk "CER $cer% (8 samples)"
}

# ------------------------------------------------------------------- summary
Write-Host "`n================ SUMMARY ================" -ForegroundColor Cyan
$results | Format-Table Layer, Check, Status, Detail -AutoSize | Out-String | Write-Host

$failed = @($results | Where-Object { $_.Status -eq "FAIL" })
$total = $results.Count

if ($failed.Count -eq 0) {
    Write-Host "ALL $total CHECKS PASSED" -ForegroundColor Green
    exit 0
}

Write-Host "$($failed.Count) of $total CHECKS FAILED:" -ForegroundColor Red
$failed | ForEach-Object { Write-Host "  - $($_.Layer) / $($_.Check): $($_.Detail)" -ForegroundColor Red }
exit 1
