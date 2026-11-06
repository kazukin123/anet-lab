param([ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Label = 'final-r1')
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
Set-Location -LiteralPath $root
$outDir = Join-Path $root '.scratch/prd101'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$summaryPath = Join-Path $outDir "$Label-verification.json"
if (Test-Path -LiteralPath $summaryPath) { throw 'Use a fresh -Label to preserve existing verification evidence.' }
$results = @()
function Invoke-Prd101Check([string]$name, [scriptblock]$action) {
    $log = Join-Path $outDir "$Label-$name.log"
    Write-Output "START $name $([DateTime]::UtcNow.ToString('o'))"
    $started = [DateTime]::UtcNow
    & $action *> $log
    $code = $LASTEXITCODE
    $script:lastCode = $code
    $script:results += [pscustomobject]@{name=$name; exit_code=$code; seconds=([DateTime]::UtcNow-$started).TotalSeconds; log=$log}
    $script:results | ConvertTo-Json | Set-Content -Encoding UTF8 $summaryPath
    Write-Output "END $name exit=$code"
}
Invoke-Prd101Check 'build' { & cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug' }
$built = $script:lastCode -eq 0
Invoke-Prd101Check 'test-build' { & cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test' }
if ($script:lastCode -eq 0) {
    Invoke-Prd101Check 'related' { & core/anet-core/bin/Debug/anet-core-test.exe '[prd101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]' }
    Invoke-Prd101Check 'core' { & core/anet-core/bin/Debug/anet-core-test.exe }
} else {
    Write-Output 'SKIP tests: the final test executable was not built.'
}
if ($built) {
    Invoke-Prd101Check 'learning-series' { & ./.venv/Scripts/python.exe core/anet-core/testdata/prd101/run_validation.py stage2 --label $Label }
    Invoke-Prd101Check 'epsilon-switch' { & ./.venv/Scripts/python.exe core/anet-core/testdata/prd101/short_run.py epsilon --label "$Label-epsilon" }
    Invoke-Prd101Check 'learning-rate-switch' { & ./.venv/Scripts/python.exe core/anet-core/testdata/prd101/short_run.py learning_rate --label "$Label-learning-rate" }
} else {
    Write-Output 'SKIP runtime checks: the final runner was not built.'
}
if ($results.exit_code | Where-Object { $_ -ne 0 }) { exit 1 }
exit 0
