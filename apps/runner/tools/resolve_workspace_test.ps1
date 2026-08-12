$ErrorActionPreference = 'Stop'

$toolsRoot = $PSScriptRoot
$runnerRoot = Split-Path -Parent $toolsRoot
$appsRoot = Split-Path -Parent $runnerRoot
$repoRoot = Split-Path -Parent $appsRoot
$sourceResolver = Join-Path $toolsRoot 'resolve_workspace.bat'
$launcherNames = @(
    '31_tb_bridge.bat',
    '32_start_tb.bat',
    '41_mlflow_bridge.bat',
    '42_start_mlflow.bat',
    '80_dot_to_png_all.bat',
    '81_dot_to_png_latest.bat',
    '90_to_mp4_all.bat',
    '91_to_mp4_latest.bat'
)
$shiftJis = [Text.Encoding]::GetEncoding(
    932,
    [Text.EncoderExceptionFallback]::new(),
    [Text.DecoderExceptionFallback]::new())
$testBase = [IO.Path]::GetFullPath((Join-Path $repoRoot 'out\test-tmp'))
$testRoot = Join-Path $testBase 'workspace-launcher-test'
if (-not $testRoot.StartsWith($testBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe test path: $testRoot"
}

function Add-Workspace([string]$Name) {
    $root = Join-Path $testRoot "runner\workspaces\$Name"
    New-Item -ItemType Directory -Force (Join-Path $root 'config'), (Join-Path $root 'runs') | Out-Null
    [IO.File]::WriteAllText((Join-Path $root 'config\_main.txt'), "# test`n", [Text.UTF8Encoding]::new($false))
    return $root
}

function Invoke-Resolver([string]$Argument = '') {
    $resolver = Join-Path $testRoot 'runner\tools\resolve_workspace.bat'
    $command = if ($Argument) {
        "call `"$resolver`" `"$Argument`" && echo RUNS_DIR=!RUNS_DIR!"
    } else {
        "call `"$resolver`" && echo RUNS_DIR=!RUNS_DIR!"
    }
    $output = & $env:COMSPEC /d /v:on /s /c $command 2>&1
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = ($output -join "`n") }
}

function Assert-BatchFileEncoding {
    $batchFiles = @(
        Get-ChildItem -LiteralPath $appsRoot -Filter '*.bat'
        Get-Item -LiteralPath $sourceResolver
    )
    foreach ($batchFile in $batchFiles) {
        $bytes = [IO.File]::ReadAllBytes($batchFile.FullName)
        try {
            $null = $shiftJis.GetString($bytes)
        } catch {
            throw "Batch file is not valid Shift_JIS: $($batchFile.FullName)"
        }
        for ($i = 0; $i -lt $bytes.Length; ++$i) {
            if ($bytes[$i] -eq 10 -and ($i -eq 0 -or $bytes[$i - 1] -ne 13)) {
                throw "Batch file does not use CRLF line endings: $($batchFile.FullName)"
            }
        }
    }
}

function Assert-LauncherWiring {
    $expectedCall = 'call "%~dp0runner\tools\resolve_workspace.bat" "%~1"'
    foreach ($launcherName in $launcherNames) {
        $launcherPath = Join-Path $appsRoot $launcherName
        $content = $shiftJis.GetString([IO.File]::ReadAllBytes($launcherPath))
        if (-not $content.Contains($expectedCall)) {
            throw "Workspace resolver call is missing: $launcherPath"
        }
    }
}

function Assert-MlflowBridgeStartupFeedback {
    $launcherPath = Join-Path $appsRoot '41_mlflow_bridge.bat'
    $content = $shiftJis.GetString([IO.File]::ReadAllBytes($launcherPath))
    if (-not $content.Contains("import importlib.metadata as metadata")) {
        throw 'MLflow bridge version check imports the full MLflow package.'
    }
    if (-not $content.Contains('[INFO] Starting MLflow bridge. Initial import may take a while...')) {
        throw 'MLflow bridge startup feedback is missing.'
    }
    if (-not $content.Contains('"%VENV_PYTHON%" -u ..\..\viewers\metrics-tools\mlflow_bridge.py')) {
        throw 'MLflow bridge Python output is buffered.'
    }
    if ($content -match '(?m)^pwd\r?$') {
        throw 'MLflow bridge contains the unsupported cmd command pwd.'
    }
}

Assert-BatchFileEncoding
Assert-LauncherWiring
Assert-MlflowBridgeStartupFeedback

try {
    New-Item -ItemType Directory -Force $testRoot | Out-Null
    $fixtureTools = Join-Path $testRoot 'runner\tools'
    New-Item -ItemType Directory -Force $fixtureTools | Out-Null
    Copy-Item -LiteralPath $sourceResolver -Destination $fixtureTools
    $trialRoot = Add-Workspace 'trial'
    $japaneseRoot = Add-Workspace '日本語 workspace'
    $defaultRoot = Add-Workspace '_default'
    $manualRoot = Join-Path $testRoot 'runner\workspaces\manual'
    New-Item -ItemType Directory -Force (Join-Path $manualRoot 'runs') | Out-Null
    $noRunsRoot = Join-Path $testRoot 'runner\workspaces\no-runs'
    New-Item -ItemType Directory -Force $noRunsRoot | Out-Null

    $explicit = Invoke-Resolver '  trial  '
    if ($explicit.ExitCode -ne 0 -or $explicit.Output -notlike "*$(Join-Path $trialRoot 'runs')*") {
        throw "Explicit workspace resolution failed: $($explicit.Output)"
    }

    $tabTrimmed = Invoke-Resolver "`ttrial`t"
    if ($tabTrimmed.ExitCode -ne 0 -or $tabTrimmed.Output -notlike "*$(Join-Path $trialRoot 'runs')*") {
        throw "Workspace tab trimming failed: $($tabTrimmed.Output)"
    }

    $utf8Explicit = Invoke-Resolver '日本語 workspace'
    if ($utf8Explicit.ExitCode -ne 0 -or $utf8Explicit.Output -notlike "*$(Join-Path $japaneseRoot 'runs')*") {
        throw "UTF-8 explicit workspace resolution failed: $($utf8Explicit.Output)"
    }

    $manual = Invoke-Resolver 'manual'
    if ($manual.ExitCode -ne 0 -or $manual.Output -notlike "*$(Join-Path $manualRoot 'runs')*") {
        throw "Workspace without config resolution failed: $($manual.Output)"
    }

    $noRuns = Invoke-Resolver 'no-runs'
    if ($noRuns.ExitCode -eq 0 -or (Test-Path (Join-Path $noRunsRoot 'runs'))) {
        throw 'Launcher created or accepted a workspace without runs.'
    }

    New-Item -ItemType Directory -Force (Join-Path $testRoot 'runner\appdata') | Out-Null
    [IO.File]::WriteAllText(
        (Join-Path $testRoot 'runner\appdata\last_workspace.txt'),
        $japaneseRoot,
        [Text.UTF8Encoding]::new($false))
    $utf8LastWorkspace = Invoke-Resolver
    if ($utf8LastWorkspace.ExitCode -ne 0 -or $utf8LastWorkspace.Output -notlike "*$(Join-Path $japaneseRoot 'runs')*") {
        throw "UTF-8 last workspace resolution failed: $($utf8LastWorkspace.Output)"
    }

    [IO.File]::WriteAllText(
        (Join-Path $testRoot 'runner\appdata\last_workspace.txt'),
        'missing-workspace',
        [Text.UTF8Encoding]::new($false))
    $fallback = Invoke-Resolver
    if ($fallback.ExitCode -ne 0 -or $fallback.Output -notlike "*$(Join-Path $defaultRoot 'runs')*") {
        throw "Default fallback failed: $($fallback.Output)"
    }

    foreach ($invalidPath in @('bad#name', '\\server\share', '//server/share', '\/server\share', '/\server/share')) {
        $invalid = Invoke-Resolver $invalidPath
        if ($invalid.ExitCode -eq 0) {
            throw "Invalid workspace path was accepted: $invalidPath"
        }
    }

    $missing = Invoke-Resolver 'not-created'
    if ($missing.ExitCode -eq 0 -or (Test-Path (Join-Path $testRoot 'runner\workspaces\not-created'))) {
        throw 'Launcher created or accepted a missing workspace.'
    }

    Write-Output 'Workspace launcher tests passed.'
} finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
