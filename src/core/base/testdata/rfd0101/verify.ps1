param([ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Label = 'final-r1')
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).Path
Set-Location -LiteralPath $root
$outDir = Join-Path $root '.scratch/rfd0101'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$summaryPath = Join-Path $outDir "$Label-verification.json"
if (Test-Path -LiteralPath $summaryPath) { throw 'Use a fresh -Label to preserve existing verification evidence.' }
$results = @()
function Invoke-Rfd0101Check([string]$name, [scriptblock]$action) {
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
Invoke-Rfd0101Check 'build' { & cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug' }
$built = $script:lastCode -eq 0
Invoke-Rfd0101Check 'test-build' { & cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-base-test anet-algo-dqn-test anet-algo-muzero-test anet-algo-imagecls-test' }
if ($script:lastCode -eq 0) {
    Invoke-Rfd0101Check 'related' { & out/test/Debug/anet-core-base-test.exe '[rfd0101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]' }
    Invoke-Rfd0101Check 'related-dqn' { & out/test/Debug/anet-algo-dqn-test.exe '[rfd0101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]' }
    Invoke-Rfd0101Check 'related-muzero' { & out/test/Debug/anet-algo-muzero-test.exe '[rfd0101]~[.],[profiled_value]~[.],[config]~[.],[replay_buffer]~[.],[diagnostic_no_wait],[actor_catalog]' }
    Invoke-Rfd0101Check 'related-imagecls' { & out/test/Debug/anet-algo-imagecls-test.exe '[config]~[.]' }
    Invoke-Rfd0101Check 'core' { & out/test/Debug/anet-core-base-test.exe }
    Invoke-Rfd0101Check 'dqn' { & out/test/Debug/anet-algo-dqn-test.exe }
    Invoke-Rfd0101Check 'muzero' { & out/test/Debug/anet-algo-muzero-test.exe }
    Invoke-Rfd0101Check 'imagecls' { & out/test/Debug/anet-algo-imagecls-test.exe }
} else {
    Write-Output 'SKIP tests: the final test executable was not built.'
}
if ($built) {
    Invoke-Rfd0101Check 'learning-series' { & ./.venv/Scripts/python.exe src/core/base/testdata/rfd0101/run_validation.py stage2 --label $Label }
    Invoke-Rfd0101Check 'epsilon-switch' { & ./.venv/Scripts/python.exe src/core/base/testdata/rfd0101/short_run.py epsilon --label "$Label-epsilon" }
    Invoke-Rfd0101Check 'learning-rate-switch' { & ./.venv/Scripts/python.exe src/core/base/testdata/rfd0101/short_run.py learning_rate --label "$Label-learning-rate" }
} else {
    Write-Output 'SKIP runtime checks: the final runner was not built.'
}
if ($results.exit_code | Where-Object { $_ -ne 0 }) { exit 1 }
exit 0
