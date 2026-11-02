param([string]$OutputRoot = '', [int]$ExperienceBudget = 224000, [int]$Repetitions = 2, [switch]$ListOnly, [string[]]$Pairs = @())
$ErrorActionPreference = 'Stop'
$repo084 = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
if ($ExperienceBudget -le 200000 -or $Repetitions -lt 2) { throw 'Performance runs require budget > 200000 and repetitions >= 2' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo084 ('.scratch/prd084/performance-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$chain084 = 'run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@rfit>run.@a5_metrics>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch'
$actor084 = 'DefaultDQNAgent.nn_forward.[perf_actor].noisy.'
$learner084 = 'DefaultDQNAgent.nn_forward.[perf_learner].noisy.'
$body084 = 'net.block.[AtariHeadFC512Def].'
$learnerBase084 = @(($learner084+'mode=sample'),($learner084+'hold=call'),'DefaultDQNAgent.learner.current.nn_forward_key=perf_learner','DefaultDQNAgent.learner.target.nn_forward_key=perf_learner')
$pair084 = @(
    @{name='off'; b_profile='nz_reference_off'; a=@(); b=@()},
    @{name='actor_share'; a=@(); b=@(($actor084+'share=sample'))},
    @{name='learner_share'; a=@($learnerBase084 + @($learner084+'share=batch')); b=@($learnerBase084 + @($learner084+'share=sample'))},
    @{name='hold_count32'; a=@(($actor084+'share=sample')); b=@(($actor084+'share=sample'),($actor084+'hold=count'),($actor084+'count=32'))},
    @{name='hold_episode'; a=@(($actor084+'share=sample')); b=@(($actor084+'share=sample'),($actor084+'hold=episode'))},
    @{name='sn_spectral'; a=@(); b=@(($body084+'weight_norm.mode=spectral'))},
    @{name='sn_cap'; a=@(); b=@(($body084+'weight_norm.mode=spectral_cap'))},
    @{name='precision'; a=@(); b=@(($body084+'force_fp32=false'))},
    @{name='diagnostics'; a=@(); b=@('metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2')}
)
foreach ($mode084 in @('none','spectral','spectral_cap')) {
    $base084 = @('net.block.[AtariHeadFC512].weight_norm.mode='+$mode084)
    $pair084 += @{name=('learning_sn_'+$mode084); profile='nz_learning'; a=@($base084 + @('net.block.[AtariHeadFC512].noisy.enabled=false','DefaultDQNAgent.head.projection.noisy.enabled=false')); b=$base084}
}
if ($Pairs.Count -gt 0) {
    foreach ($selected084 in $Pairs) {
        if ($selected084 -notin @($pair084 | ForEach-Object { $_.name })) { throw ('Unknown performance pair: ' + $selected084) }
    }
    $pair084 = @($pair084 | Where-Object { $_.name -in $Pairs })
}
if ($ListOnly) { $pair084 | ConvertTo-Json -Depth 6; return }
if (-not $ListOnly) {
if (Test-Path -LiteralPath $OutputRoot) { throw ('Output already exists: ' + $OutputRoot) }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $repo084 'apps/runner/config') -Destination (Join-Path $OutputRoot 'source-config') -Recurse
$exe084 = Join-Path $OutputRoot 'AnetRLRunner.exe'
Copy-Item -LiteralPath (Join-Path $repo084 'apps/runner/bin/Release/AnetRLRunner.exe') -Destination $exe084
foreach ($dll084 in Get-ChildItem -LiteralPath (Join-Path $repo084 'apps/runner/bin/Release') -Filter '*.dll') {
    New-Item -ItemType HardLink -Path (Join-Path $OutputRoot $dll084.Name) -Target $dll084.FullName | Out-Null
}
}

function Get-GpuStatus084 {
    $learners084 = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'AnetRLRunner|python' } | Select-Object ProcessId,Name,CommandLine)
    $gpu084 = @(& nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record GPU state' }
    $apps084 = @(& nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record GPU processes' }
    return [ordered]@{time=[DateTimeOffset]::Now.ToString('o'); gpu=$gpu084; compute_apps=$apps084; learning_processes=$learners084}
}

function Get-GpuUsed084 {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $value084 = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        if ($LASTEXITCODE -eq 0) { return [double]($value084 | Select-Object -First 1) }
    }
    return $null
}
foreach ($comparison084 in $pair084) {
    for ($repeat084 = 1; $repeat084 -le $Repetitions; ++$repeat084) {
        foreach ($side084 in @('a','b')) {
            $name084 = $comparison084.name + '-' + $repeat084 + '-' + $side084
            $profile084 = if ($side084 -eq 'b' -and $comparison084.b_profile) { $comparison084.b_profile } elseif ($comparison084.profile) { $comparison084.profile } else { 'nz_reference' }
            $config084Path = Join-Path $OutputRoot ($name084 + '.txt')
            $config084 = @(
                ('$include <' + ($OutputRoot.Replace('\','/')) + '/source-config/_main.txt>'),
                ('$include <' + ($OutputRoot.Replace('\','/')) + '/source-config/Atari.txt>'),
                ('run.$ = ' + $chain084 + '>run.@' + $profile084),
                'run.train.num_envs = 128','E1.game = breakout','run.seed = 1',
                'backend.$ = backend.@deterministic','backend.deterministic_warn_only = false',
                'app.$ = app.batchrun > P1','app.show_error_dialog = false',
                ('app.runs_dir = ' + $OutputRoot.Replace('\','/') + '/runs')
            ) -join "`n"
            [System.IO.File]::WriteAllText($config084Path, $config084 + "`n", [System.Text.UTF8Encoding]::new($false))
            $args084 = @('--config',$config084Path,('app.run_name=run_{t}_prd084_perf_'+$name084),('app.exp_exit_step='+$ExperienceBudget),('app.batchrun.exp_exit_step='+$ExperienceBudget),
                'run.eval_schedule.[eval].interval=0','run.eval_schedule.[eval_target].interval=0','run.eval_schedule.[greedy].interval=0',
                ($actor084+'mode=sample'),($actor084+'hold=call'),($actor084+'share=batch'),'DefaultDQNAgent.actor.[train].nn_forward_key=perf_actor') + $comparison084[$side084]
            $record084 = [ordered]@{pair=$comparison084.name; repeat=$repeat084; side=$side084; profile=$profile084; executable_sha256=(Get-FileHash -LiteralPath $exe084).Hash; arguments=$args084; seed=1; experience_budget=$ExperienceBudget; start=[DateTimeOffset]::Now.ToString('o'); gpu_before_mib=(Get-GpuUsed084)}
            if ($Pairs.Count -gt 0) {
                $record084.gpu_status_before=Get-GpuStatus084
                if ($record084.gpu_status_before.learning_processes.Count -gt 0) {
                    $record084.measurement_deferred='Another learning process is running; no performance run started'
                    $record084 | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputRoot ($name084+'.deferred.json'))
                    throw $record084.measurement_deferred
                }
            }
            $watch084 = [System.Diagnostics.Stopwatch]::StartNew()
            $process084 = Start-Process -FilePath $exe084 -ArgumentList $args084 -WorkingDirectory (Join-Path $repo084 'apps/runner') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot ($name084+'.stdout.log')) -RedirectStandardError (Join-Path $OutputRoot ($name084+'.stderr.log'))
            $handle084 = $process084.Handle
            $record084.pid=$process084.Id
            $record084 | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputRoot ($name084+'.execution.json'))
            Write-Output ('Started ' + $name084 + ' pid=' + $process084.Id)
            $peakPrivate084=0L; $peakWorking084=0L; $peakGpu084=$record084.gpu_before_mib; $gpuNext084=0.0
            while (-not $process084.WaitForExit(500)) {
                $process084.Refresh()
                if (-not $process084.HasExited) {
                    $peakPrivate084=[Math]::Max($peakPrivate084,$process084.PrivateMemorySize64)
                    $peakWorking084=[Math]::Max($peakWorking084,$process084.PeakWorkingSet64)
                }
                if ($watch084.Elapsed.TotalSeconds -ge $gpuNext084) {
                    $used084=Get-GpuUsed084
                    if ($null -ne $used084) { $peakGpu084=[Math]::Max($peakGpu084,$used084) }
                    $gpuNext084=$watch084.Elapsed.TotalSeconds+5
                }
            }
            $process084.WaitForExit(); $process084.Refresh(); $watch084.Stop()
            $record084.exit_code=$process084.ExitCode; $record084.end=[DateTimeOffset]::Now.ToString('o'); $record084.elapsed_seconds=$watch084.Elapsed.TotalSeconds
            $record084.peak_private_bytes=$peakPrivate084; $record084.peak_working_set_bytes=$peakWorking084
            $record084.peak_device_used_mib=$peakGpu084; $record084.device_extra_mib=$peakGpu084-$record084.gpu_before_mib
            if ($Pairs.Count -gt 0) { $record084.gpu_status_after=Get-GpuStatus084 }

            $record084 | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputRoot ($name084+'.execution.json'))
            Write-Output ('Finished ' + $name084 + ' exit=' + $record084.exit_code + ' seconds=' + $record084.elapsed_seconds)
            if ($record084.exit_code -ne 0) { exit $record084.exit_code }
            if ($Pairs.Count -gt 0 -and $record084.gpu_status_after.learning_processes.Count -gt 0) {
                throw 'Another learning process appeared during measurement; do not accept this performance run'
            }

        }
    }
}
$summaryArgs084 = @($OutputRoot)
if ($Pairs.Count -gt 0) { $summaryArgs084 += ($Pairs -join ',') }
& (Join-Path $repo084 '.venv/Scripts/python.exe') (Join-Path $PSScriptRoot 'summarize_performance.py') @summaryArgs084
exit $LASTEXITCODE
