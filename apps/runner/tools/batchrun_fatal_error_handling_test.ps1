$ErrorActionPreference = 'Stop'
chcp 65001 | Out-Null
[Console]::InputEncoding = [Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$toolsRoot = $PSScriptRoot
$runnerRoot = Split-Path -Parent $toolsRoot
$appsRoot = Split-Path -Parent $runnerRoot
$repoRoot = Split-Path -Parent $appsRoot
$runnerExe = Join-Path $runnerRoot 'bin\Debug\AnetRLRunner.exe'
$configPath = Join-Path $runnerRoot 'config\_main.txt'
$launcherNames = @('11_batch_run.bat', '12_batch_run.bat', '18_batch_run_atari5.bat')
$shiftJis = [Text.Encoding]::GetEncoding(
    932,
    [Text.EncoderExceptionFallback]::new(),
    [Text.DecoderExceptionFallback]::new())
$testBase = [IO.Path]::GetFullPath((Join-Path $repoRoot 'out\test-tmp'))
$testRoot = Join-Path $testBase 'batchrun-fatal-error-handling-test'
if (-not $testRoot.StartsWith($testBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe test path: $testRoot"
}

function Write-ShiftJisBatch([string]$Path, [string]$Content) {
    $normalized = [regex]::Replace($Content, "\r?\n", "`r`n")
    if (-not $normalized.EndsWith("`r`n")) {
        $normalized += "`r`n"
    }
    [IO.File]::WriteAllBytes($Path, $shiftJis.GetBytes($normalized))
}

function Invoke-RunnerFatalSmoke {
    if (-not (Test-Path -LiteralPath $runnerExe)) {
        throw "Debug Runner is missing. Build it first: $runnerExe"
    }

    $stdoutPath = Join-Path $testRoot 'runner.stdout.log'
    $stderrPath = Join-Path $testRoot 'runner.stderr.log'
    $arguments = "--config `"$configPath`" `"app.$=app.batchrun`" `"app.log_flush_interval_ms=-1`""
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
    if (-not $stderrText.Contains('Invalid config key app.log_flush_interval_ms')) {
        throw "Runner stderr does not contain the configuration error: $stderrText"
    }
}

function New-LauncherFixture([string]$LauncherName) {
    $fixtureRoot = Join-Path $testRoot ([IO.Path]::GetFileNameWithoutExtension($LauncherName))
    New-Item -ItemType Directory -Force (Join-Path $fixtureRoot 'runner') | Out-Null

    $sourcePath = Join-Path $appsRoot $LauncherName
    $content = $shiftJis.GetString([IO.File]::ReadAllBytes($sourcePath))
    $exePattern = '(?im)^SET EXE=.*$'
    if ([regex]::Matches($content, $exePattern).Count -ne 1) {
        throw "Expected exactly one active SET EXE line: $sourcePath"
    }

    # 一時コピーだけを stub へ差し替え、実 launcher の Run 一覧と制御フローを実行する。
    $content = $content.Replace("`r`n", "`n").Replace("`r", "`n")
    $filteredLines = $content.Split([char]10)
    $filteredLines = @($filteredLines | Where-Object {
        $_ -notmatch 'goto :no_exe' -and
        $_ -notmatch '^copy /Y ' -and
        $_ -notmatch '^pause\s*$'
    })
    $filteredLines = @($filteredLines | ForEach-Object {
        if ($_ -match '^SET EXE=') {
            'SET EXE=call "%~dp0stub_runner.cmd"'
        } else {
            $_
        }
    })
    $content = $filteredLines -join "`r`n"
    if ($content.Contains('goto :no_exe')) {
        throw "Failed to remove executable preflight from launcher fixture: $sourcePath"
    }
    if (-not $content.Contains('SET EXE=call "%~dp0stub_runner.cmd"')) {
        $actualExeLine = @($filteredLines | Where-Object { $_ -like '*EXE*' }) -join '; '
        throw "Failed to replace executable in launcher fixture: $sourcePath lines=$($filteredLines.Count) actual=$actualExeLine"
    }

    $launcherPath = Join-Path $fixtureRoot $LauncherName
    Write-ShiftJisBatch $launcherPath $content

    $stub = @'
@echo off
set "CALL_COUNT=0"
if exist "%BATCH_TEST_COUNTER%" set /p CALL_COUNT=<"%BATCH_TEST_COUNTER%"
set /a CALL_COUNT+=1
> "%BATCH_TEST_COUNTER%" echo %CALL_COUNT%
>> "%BATCH_TEST_CALLS%" echo %*
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
    $callsPath = Join-Path $fixtureRoot "$scenario.calls.txt"
    Remove-Item -LiteralPath $counterPath, $callsPath -Force -ErrorAction SilentlyContinue

    $env:BATCH_TEST_COUNTER = $counterPath
    $env:BATCH_TEST_CALLS = $callsPath
    $env:BATCH_TEST_FAIL_INDEX = $FailIndex
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $env:COMSPEC /d /c $LauncherPath 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
        Remove-Item Env:BATCH_TEST_COUNTER, Env:BATCH_TEST_CALLS, Env:BATCH_TEST_FAIL_INDEX `
            -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path -LiteralPath $callsPath)) {
        throw "Launcher fixture did not invoke the stub. output=$($output -join "`n")"
    }
    $calls = @(Get-Content -LiteralPath $callsPath)
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($output -join "`n")
        CallCount = $calls.Count
    }
}

function Assert-LauncherAggregation([string]$LauncherName) {
    $launcherPath = New-LauncherFixture $LauncherName

    $failed = Invoke-LauncherFixture $launcherPath 2
    if ($failed.ExitCode -eq 0) {
        throw "$LauncherName returned zero after a failed Run. output=$($failed.Output)"
    }
    if ($failed.CallCount -le 2) {
        throw "$LauncherName stopped instead of invoking a Run after the failure."
    }
    if (-not $failed.Output.Contains('[ERROR] RUN FAILED exit_code=7 args=')) {
        throw "$LauncherName did not print the failed Run line. output=$($failed.Output)"
    }
    if (-not $failed.Output.Contains('1 FAILED')) {
        throw "$LauncherName did not print the aggregate failure count. output=$($failed.Output)"
    }

    $succeeded = Invoke-LauncherFixture $launcherPath 0
    if ($succeeded.ExitCode -ne 0) {
        throw "$LauncherName returned non-zero when every Run succeeded. output=$($succeeded.Output)"
    }
    if (-not $succeeded.Output.Contains('0 FAILED')) {
        throw "$LauncherName did not print the all-success count. output=$($succeeded.Output)"
    }
}

try {
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $testRoot | Out-Null

    Invoke-RunnerFatalSmoke
    foreach ($launcherName in $launcherNames) {
        Assert-LauncherAggregation $launcherName
    }

    Write-Host 'batchrun fatal error handling tests passed.'
} finally {
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}
