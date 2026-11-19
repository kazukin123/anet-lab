$ErrorActionPreference = 'Stop'
chcp 65001 | Out-Null
[Console]::InputEncoding = [Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$toolsRoot = $PSScriptRoot
$appsRoot = Split-Path -Parent $toolsRoot
$runnerRoot = Join-Path $appsRoot 'runner'
$repoRoot = Split-Path -Parent $appsRoot
$runnerExe = Join-Path $runnerRoot 'bin\Debug\AnetRLRunner.exe'
$configPath = Join-Path $runnerRoot 'config\_main.txt'
$launcherNames = @('11_batch_run.bat', '12_batch_run.bat')
$shiftJis = [Text.Encoding]::GetEncoding(
    932,
    [Text.EncoderExceptionFallback]::new(),
    [Text.DecoderExceptionFallback]::new())
$testBase = [IO.Path]::GetFullPath((Join-Path $repoRoot 'out\test-tmp'))
$testRoot = [IO.Path]::GetFullPath((Join-Path $testBase "batchrun-fatal-error-handling-test-$([Guid]::NewGuid().ToString('N'))"))
if (-not $testRoot.StartsWith($testBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe test path: $testRoot"
}

function Write-ShiftJisBatch([string]$Path, [string]$Content) {
    $normalized = [regex]::Replace($Content, "\r?\n", "`r`n")
    if (-not $normalized.EndsWith("`r`n")) {
        $normalized += "`r`n"
    }
    [IO.File]::WriteAllBytes($Path, $shiftJis.GetBytes($normalized))
}

function Invoke-RunnerFatalSmoke(
    [string]$Override,
    [string]$ExpectedError,
    [string]$Scenario) {
    if (-not (Test-Path -LiteralPath $runnerExe)) {
        throw "Debug Runner is missing. Build it first: $runnerExe"
    }

    $stdoutPath = Join-Path $testRoot "runner-$Scenario.stdout.log"
    $stderrPath = Join-Path $testRoot "runner-$Scenario.stderr.log"
    $arguments = "--config `"$configPath`" `"app.$=app.batchrun`" `"app.runs_dir=$testRoot/runs`" `"$Override`""
    $process = Start-Process -FilePath $runnerExe `
        -ArgumentList $arguments `
        -WorkingDirectory $runnerRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru

    if (-not $process.WaitForExit(15000)) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
        throw 'batchrun Runner did not exit within 15 seconds; a modal dialog may be blocking it.'
    }
    $process.WaitForExit()
    if ($process.ExitCode -eq 0) {
        throw 'batchrun Runner returned zero for a fatal configuration error.'
    }

    $stderrText = [IO.File]::ReadAllText($stderrPath)
    if (-not $stderrText.Contains($ExpectedError)) {
        throw "Runner stderr does not contain the configuration error: $stderrText"
    }
}

function New-LauncherFixture([string]$LauncherName) {
    $fixtureRoot = Join-Path $testRoot ([IO.Path]::GetFileNameWithoutExtension($LauncherName) + '\apps')
    New-Item -ItemType Directory -Force (Join-Path $fixtureRoot 'runner') | Out-Null

    $sourcePath = Join-Path $appsRoot $LauncherName
    $content = $shiftJis.GetString([IO.File]::ReadAllBytes($sourcePath))
    $content = $content.Replace("`r`n", "`n").Replace("`r", "`n")
    $runnerPattern = '(?im)^SET (RUNNER(?:_OLD)?)="[^\n"]+"$'
    $waitPattern = '(?ims)^:waitprev\n.*?(?=^:skipwait$|^SET "SRC_EXE=)'
    $beepPattern = '(?im)^:beep_forever$'
    foreach ($pattern in @('(?im)^SET RUNNER="[^\n"]+"$', $waitPattern, $beepPattern)) {
        if ([regex]::Matches($content, $pattern).Count -ne 1) {
            throw "Expected exactly one fixture substitution in $sourcePath for pattern $pattern"
        }
    }

    # 一時コピーでは出荷実行体とマシンの空き資源に依存せず、既定キューと集計を通す。
    $content = [regex]::Replace($content, $runnerPattern, 'SET $1=call "%~dp0stub_runner.cmd"')
    $content = [regex]::Replace($content, $waitPattern, '')
    $content = [regex]::Replace($content, '(?im)^(?:if [^\n]* goto :no_exe|copy /Y [^\n]*|call :wait_disk)\n', '')
    $content = [regex]::Replace($content, '(?im)^call :check_ram$', 'SET "AVAIL_MB=40000"')
    $content = [regex]::Replace($content, $beepPattern, ":beep_forever`nexit /b 0")

    $launcherPath = Join-Path $fixtureRoot $LauncherName
    Write-ShiftJisBatch $launcherPath $content

    $stub = @'
@echo off
title %~nx0
set "CALL_COUNT=0"
if exist "%BATCH_TEST_COUNTER%" set /p CALL_COUNT=<"%BATCH_TEST_COUNTER%"
set /a CALL_COUNT+=1
> "%BATCH_TEST_COUNTER%" echo %CALL_COUNT%
if "%CALL_COUNT%"=="%BATCH_TEST_FAIL_INDEX%" exit /b 7
exit /b 0
'@
    Write-ShiftJisBatch (Join-Path $fixtureRoot 'stub_runner.cmd') $stub
    return $launcherPath
}

function Invoke-LauncherFixture([string]$LauncherPath, [int]$FailIndex) {
    $scenario = if ($FailIndex -eq 0) { 'success' } else { 'failure' }
    $fixtureRoot = Split-Path -Parent $LauncherPath
    $counterPath = Join-Path $fixtureRoot "$scenario.counter.txt"
    $env:BATCH_TEST_COUNTER = $counterPath
    $env:BATCH_TEST_FAIL_INDEX = $FailIndex
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $env:COMSPEC /d /c $LauncherPath 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
        Remove-Item Env:BATCH_TEST_COUNTER, Env:BATCH_TEST_FAIL_INDEX `
            -ErrorAction SilentlyContinue
    }

    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $counterPath)) {
        throw "Launcher fixture failed. exit_code=$exitCode output=$($output -join "`n")"
    }
    return [pscustomobject]@{
        Output = ($output -join "`n")
        CallCount = [int](Get-Content -LiteralPath $counterPath)
    }
}

function Assert-LauncherAggregation([string]$LauncherName) {
    $launcherPath = New-LauncherFixture $LauncherName

    $failed = Invoke-LauncherFixture $launcherPath 1
    if ($failed.CallCount -le 1) {
        throw "$LauncherName stopped instead of invoking a Run after the failure."
    }
    if (-not $failed.Output.Contains('[ERROR] RUN FAILED exit_code=7 args=')) {
        throw "$LauncherName did not print the failed Run line. output=$($failed.Output)"
    }
    $successCount = $failed.CallCount - 1
    if (-not $failed.Output.Contains("=== ALL DONE: $successCount SUCCEEDED, 1 FAILED ===")) {
        throw "$LauncherName did not print the aggregate failure count. output=$($failed.Output)"
    }

    $succeeded = Invoke-LauncherFixture $launcherPath 0
    if ($succeeded.CallCount -ne $failed.CallCount) {
        throw "$LauncherName ran different numbers of Runs in the success and failure scenarios."
    }
    if (-not $succeeded.Output.Contains("=== ALL DONE: $($succeeded.CallCount) SUCCEEDED ===")) {
        throw "$LauncherName did not print the all-success count. output=$($succeeded.Output)"
    }
}

try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null

    Invoke-RunnerFatalSmoke `
        'app.log_flush_interval_ms=-1' `
        'Invalid config key app.log_flush_interval_ms' `
        'log-flush-negative'
    Invoke-RunnerFatalSmoke `
        'app.drain_timeout_sec=0' `
        'Invalid config key app.drain_timeout_sec: value=0 (expected: positive integer > 0)' `
        'drain-timeout-zero'
    Invoke-RunnerFatalSmoke `
        'app.drain_timeout_sec=-1' `
        'Invalid config key app.drain_timeout_sec: value=-1 (expected: positive integer > 0)' `
        'drain-timeout-negative'
    Invoke-RunnerFatalSmoke `
        'app.drain_timeout_sec=1.5' `
        'ConfigData::Read failed. key=app.drain_timeout_sec value="1.5" expected=int' `
        'drain-timeout-non-integer'
    foreach ($launcherName in $launcherNames) {
        Assert-LauncherAggregation $launcherName
    }

    Write-Host 'batchrun fatal error handling tests passed.'
} finally {
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}
