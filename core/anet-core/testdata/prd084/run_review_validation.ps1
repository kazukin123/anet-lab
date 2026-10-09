param([string]$OutputRoot = '', [int]$ExperienceBudget = 500000, [switch]$Resume)
$ErrorActionPreference = 'Stop'
$repo084 = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
if ($ExperienceBudget -lt 500000) { throw 'Acceptance requires at least 500000 experiences' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo084 ('.scratch/prd084/review-fix/after-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$exe084 = Join-Path $OutputRoot 'AnetRLRunner.exe'
if ($Resume) {
    if (-not (Test-Path -LiteralPath $exe084)) { throw 'Resume requires a saved executable' }
    if ((Get-FileHash -LiteralPath $exe084).Hash -ne (Get-FileHash -LiteralPath (Join-Path $repo084 'apps/runner/bin/Release/AnetRLRunner.exe')).Hash) { throw 'Resume executable differs from current build' }
} else {
if (Test-Path -LiteralPath $OutputRoot) { throw ('Output already exists: ' + $OutputRoot) }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $repo084 'apps/runner/config') -Destination (Join-Path $OutputRoot 'source-config') -Recurse
New-Item -ItemType Directory -Path (Join-Path $OutputRoot 'config') | Out-Null
Copy-Item -LiteralPath (Join-Path $repo084 'apps/runner/bin/Release/AnetRLRunner.exe') -Destination $exe084
foreach ($dll084 in Get-ChildItem -LiteralPath (Join-Path $repo084 'apps/runner/bin/Release') -Filter '*.dll') {
    New-Item -ItemType HardLink -Path (Join-Path $OutputRoot $dll084.Name) -Target $dll084.FullName | Out-Null
}
$oldSource084 = ($repo084.Replace('\','/') + '/.scratch/prd084/old/source-config')
$newSource084 = $OutputRoot.Replace('\','/') + '/source-config'
foreach ($case084 in @('atari_default','atari_efficiency','lunarlander_cpu')) {
    $oldConfig084 = Join-Path $repo084 ('.scratch/prd084/old/config/'+$case084+'.txt')
    $text084 = [System.IO.File]::ReadAllText($oldConfig084).Replace($oldSource084,$newSource084)
    $text084 = $text084.Replace($repo084.Replace('\','/')+'/.scratch/prd084/old/runs',$OutputRoot.Replace('\','/')+'/runs')
    [System.IO.File]::WriteAllText((Join-Path $OutputRoot ('config/'+$case084+'.txt')),$text084,[System.Text.UTF8Encoding]::new($false))
}
$chain084 = 'run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@rfit>run.@a5_metrics>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch'
foreach ($case084 in @('nz_learning','nz_reference')) {
    $text084 = @(
        ('$include <'+$newSource084+'/_main.txt>'),
        ('$include <'+$newSource084+'/Atari.txt>'),
        ('run.$ = '+$chain084+'>run.@'+$case084),
        'run.train.num_envs = 128','E1.game = breakout','run.seed = 1',
        'backend.$ = backend.@deterministic','backend.deterministic_warn_only = false',
        'app.$ = app.batchrun > P1','app.show_error_dialog = false',
        ('app.runs_dir = '+$OutputRoot.Replace('\','/')+'/runs')
    ) -join [char]10
    [System.IO.File]::WriteAllText((Join-Path $OutputRoot ('config/'+$case084+'.txt')),$text084+[char]10,[System.Text.UTF8Encoding]::new($false))
}
}
function Invoke-ReviewRun084([string]$case084,[string]$config084,[string[]]$extra084) {
    $path084 = Join-Path $OutputRoot ($case084+'.execution.json')
    if ($Resume -and (Test-Path -LiteralPath $path084)) {
        $previous084 = Get-Content -Raw -Encoding UTF8 -LiteralPath $path084 | ConvertFrom-Json
        if ($null -eq $previous084.exit_code) { throw ('Cannot resume unfinished run: '+$case084) }
        if ($previous084.exit_code -eq 0) {
            if ($previous084.sha256 -ne (Get-FileHash -LiteralPath $exe084).Hash -or $previous084.seed -ne 1) { throw 'Saved run does not match snapshot' }
            Write-Output ('Preserved successful '+$case084)
            return
        }
        foreach ($suffix084 in @('.execution.json','.stdout.log','.stderr.log')) {
            $failed084 = Join-Path $OutputRoot ($case084+$suffix084)
            if (Test-Path -LiteralPath $failed084) {
                $archive084=$failed084+'.failed-01'
                if (Test-Path -LiteralPath $archive084) { throw ('Failed-attempt archive exists: '+$archive084) }
                Move-Item -LiteralPath $failed084 -Destination $archive084
            }
        }
    }
    $existing084 = @(Get-ChildItem -LiteralPath (Join-Path $OutputRoot 'runs') -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    # OFF回帰3本は旧基準の500kに固定し、終了予算の延長は結合Runだけへ渡す。
    $caseBudget084 = if ($case084.StartsWith('nz_')) { $ExperienceBudget } else { 500000 }
    $args084 = @('--config',$config084,('app.run_name=run_{t}_prd084_review_'+$case084),
        ('app.exp_exit_step='+$caseBudget084)) + $extra084
    if ($case084.StartsWith('atari_') -or $case084.StartsWith('nz_')) {
        $args084 += 'run.eval_schedule.[greedy_dist].interval=0'
    }
    $actualBudget084 = $caseBudget084
    foreach ($argument084 in $args084) { if ($argument084 -match '^app.exp_exit_step=(\d+)$') { $actualBudget084 = [int]$Matches[1] } }
    $record084 = [ordered]@{case=$case084; executable=$exe084; sha256=(Get-FileHash -LiteralPath $exe084).Hash; arguments=$args084; seed=1; experience_budget=$actualBudget084; start=[DateTimeOffset]::Now.ToString('o')}
    $process084 = Start-Process -FilePath $exe084 -ArgumentList $args084 -WorkingDirectory (Join-Path $repo084 'apps/runner') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot ($case084+'.stdout.log')) -RedirectStandardError (Join-Path $OutputRoot ($case084+'.stderr.log'))
    $handle084 = $process084.Handle
    $record084.pid=$process084.Id
    $record084 | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $path084
    Write-Output ('Started '+$case084+' pid='+$process084.Id)
    while (-not $process084.WaitForExit(1000)) {}
    $process084.WaitForExit(); $process084.Refresh()
    $created084 = @(Get-ChildItem -LiteralPath (Join-Path $OutputRoot 'runs') -Directory | Where-Object { $_.Name.EndsWith('prd084_review_'+$case084) -and $_.FullName -notin $existing084 })
    if ($created084.Count -eq 1) { $record084.run=$created084[0].FullName }
    $record084.exit_code=$process084.ExitCode; $record084.end=[DateTimeOffset]::Now.ToString('o')
    $record084 | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $path084
    Write-Output ('Finished '+$case084+' exit='+$record084.exit_code)
    if ($record084.exit_code -ne 0) { throw ('Run failed: '+$case084+' exit='+$record084.exit_code) }
}
foreach ($case084 in @('atari_default','atari_efficiency','lunarlander_cpu','nz_learning','nz_reference')) {
    $extra084=@()
    if ($case084 -eq 'nz_reference') {
        $extra084=@('DefaultDQNAgent.nn_forward.[review_actor].noisy.mode=sample',
            'DefaultDQNAgent.nn_forward.[review_actor].noisy.hold=episode',
            'DefaultDQNAgent.nn_forward.[review_actor].noisy.share=sample',
            'DefaultDQNAgent.actor.[train].nn_forward_key=review_actor')
    }
    Invoke-ReviewRun084 $case084 (Join-Path $OutputRoot ('config/'+$case084+'.txt')) $extra084
}
$checkpoint084=Join-Path $repo084 '.scratch/prd084/old/runs/run_20261003-190410_prd084_old_lunarlander_cpu/agent_close.anet'
Invoke-ReviewRun084 'old_checkpoint' (Join-Path $OutputRoot 'config/lunarlander_cpu.txt') @(
    ('DefaultDQNAgent.auto_load_file='+$checkpoint084.Replace('\','/')),'app.exp_exit_step=256','app.batchrun.exp_exit_step=256')
exit 0
