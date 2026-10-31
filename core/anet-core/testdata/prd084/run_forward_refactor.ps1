param(
    [ValidateSet('off','on','on-after')][string]$Stage = 'off',
    [switch]$Resume,
    [ValidateSet('atari_default','atari_efficiency','lunarlander_cpu')][string[]]$Cases = @('atari_default','atari_efficiency','lunarlander_cpu'),
    [string]$Executable = (Join-Path $PSScriptRoot '../../../../apps/runner/bin/Release/AnetRLRunner.exe'),
    [ValidateNotNullOrEmpty()][string]$OutputRoot = '.scratch/prd084/forward-refactor/after'
)
$ErrorActionPreference = 'Stop'
$repo084 = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$root084 = if ([IO.Path]::IsPathRooted($OutputRoot)) {
    [IO.Path]::GetFullPath($OutputRoot)
} else {
    [IO.Path]::GetFullPath((Join-Path $repo084 $OutputRoot))
}
$before084 = Join-Path $repo084 '.scratch/prd084/forward-refactor/before'
$bin084 = Join-Path $root084 'bin'
New-Item -ItemType Directory -Force -Path $bin084,(Join-Path $root084 'config'),(Join-Path $root084 'runs') | Out-Null
$exe084 = Join-Path $bin084 'AnetRLRunner.exe'
$current084 = [IO.Path]::GetFullPath($Executable)
if (Test-Path -LiteralPath $exe084) {
    if ((Get-FileHash -LiteralPath $exe084).Hash -ne (Get-FileHash -LiteralPath $current084).Hash) { throw 'Saved executable differs from current Release' }
} else {
    Copy-Item -LiteralPath $current084 -Destination $exe084
    foreach ($dll084 in Get-ChildItem -LiteralPath (Split-Path $current084) -Filter '*.dll') {
        New-Item -ItemType HardLink -Path (Join-Path $bin084 $dll084.Name) -Target $dll084.FullName | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $repo084 'apps/runner/config') -Destination (Join-Path $root084 'source-config') -Recurse
}
$chain084 = 'run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@rfit>run.@a5_metrics>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch'
function Get-GpuUsed084 {
    $values084 = @(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $values084.Count -ne 1) { throw 'Measurement requires one observable GPU' }
    return [double]$values084[0]
}
function Test-WarmupReached084([string[]]$Lines) {
    foreach ($line084 in $Lines) {
        $match084 = [regex]::Match($line084, '"step"\s*:\s*(\d+)')
        if ($match084.Success -and [int64]$match084.Groups[1].Value -ge 200000) { return $true }
    }
    return $false
}
function Invoke-ForwardRun084([string]$Name,[string]$Executable,[string]$Config,[int]$Budget,[switch]$Noisy) {
    # CPU回帰はGPUの検査・計測へ依存させず、GPU Runだけ競合を検査する。
    $cpu084 = $Name -in @('lunarlander_cpu','old_checkpoint')
    $others084 = @(if (-not $cpu084) { Get-CimInstance Win32_Process -Filter "Name='AnetRLRunner.exe'" })
    if ($others084.Count -gt 0) { throw ('Measurement deferred: existing Runner PID '+($others084.ProcessId -join ',')) }
    $recordPath084 = Join-Path $root084 ($Name+'.execution.json')
    if (Test-Path -LiteralPath $recordPath084) {
        if (-not $Resume) { throw ('Existing run record: '+$recordPath084) }
        $saved084 = Get-Content -LiteralPath $recordPath084 -Encoding UTF8 | ConvertFrom-Json
        if ($saved084.exit_code -ne 0 -or $saved084.experience_budget -ne $Budget -or $saved084.sha256 -ne (Get-FileHash -LiteralPath $Executable).Hash) { throw ('Cannot resume incomplete/different run: '+$Name) }
        Write-Output ('Verified completed run: '+$Name)
        return
    }
    $args084 = @('--config',$Config,('app.run_name=run_{t}_prd084_forward_'+$Name),('app.exp_exit_step='+$Budget),('app.batchrun.exp_exit_step='+$Budget))
    if ($Noisy -or $Name.StartsWith('atari_')) { $args084 += 'run.eval_schedule.[greedy].interval=0' }
    if ($Noisy) {
        # プロファイルの選択宣言より後に、比較条件を CLI で明示する。
        $args084 += @('DefaultDQNAgent.nn_forward.[ref_actor].noisy.mode=sample','DefaultDQNAgent.nn_forward.[ref_actor].noisy.hold=episode','DefaultDQNAgent.nn_forward.[ref_actor].noisy.share=sample','DefaultDQNAgent.actor.[train].nn_forward_key=ref_actor','DefaultDQNAgent.nn_forward.[ref_learner].noisy.mode=sample','DefaultDQNAgent.nn_forward.[ref_learner].noisy.hold=call','DefaultDQNAgent.nn_forward.[ref_learner].noisy.share=sample','DefaultDQNAgent.learner.current.nn_forward_key=ref_learner','DefaultDQNAgent.learner.target.nn_forward_key=ref_learner','DefaultDQNAgent.learner.munchausen.enabled=true','DefaultDQNAgent.learner.munchausen.log_policy_mode=target')
    }
    if ($Name -eq 'old_checkpoint') { $args084 += ('DefaultDQNAgent.auto_load_file='+$before084.Replace('\','/')+'/old-off-checkpoint.anet') }
    $record084 = [ordered]@{case=$Name;executable=$Executable;sha256=(Get-FileHash -LiteralPath $Executable).Hash;arguments=$args084;seed=1;experience_budget=$Budget;start=[DateTimeOffset]::Now.ToString('o')}
    if (-not $cpu084) {
        $record084.gpu_identity = @(& nvidia-smi --query-gpu=index,uuid,name --format=csv)
        if ($LASTEXITCODE -ne 0) { throw 'Cannot record GPU identity' }
        $record084.gpu_before_mib = Get-GpuUsed084
    } else {
        $record084.gpu_before_mib = $null
    }
    $peakGpu084 = $record084.gpu_before_mib
    $peakPrivate084 = 0L
    $peakWorking084 = 0L
    $warmPrivate084 = 0L
    $warmWorking084 = 0L
    $warmGpu084 = 0.0
    $warmObserved084 = $null
    $gpuNext084 = 0.0
    $timer084 = [Diagnostics.Stopwatch]::StartNew()
    $process084 = Start-Process -FilePath $Executable -ArgumentList $args084 -WorkingDirectory (Join-Path $repo084 'apps/runner') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root084 ($Name+'.stdout.log')) -RedirectStandardError (Join-Path $root084 ($Name+'.stderr.log'))
    # 短いcheckpoint確認でも終了コードを取得できるよう、開始直後にhandleを保持する。
    $null = $process084.Handle
    $record084.pid=$process084.Id
    $record084 | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $recordPath084
    Write-Output ('Started '+$Name+' pid='+$process084.Id)
    while (-not $process084.WaitForExit(500)) {
        $process084.Refresh()
        if (-not $process084.HasExited) {
            $peakPrivate084 = [Math]::Max($peakPrivate084, $process084.PrivateMemorySize64)
            $peakWorking084 = [Math]::Max($peakWorking084, $process084.PeakWorkingSet64)
        }
        if (-not $cpu084 -and $timer084.Elapsed.TotalSeconds -ge $gpuNext084) {
            $used084 = Get-GpuUsed084
            $peakGpu084 = [Math]::Max($peakGpu084, $used084)
            if ($null -eq $warmObserved084) {
                $run084 = Get-ChildItem -LiteralPath (Join-Path $root084 'runs') -Directory | Where-Object { $_.Name.EndsWith('prd084_forward_'+$Name) } | Select-Object -First 1
                if ($run084) {
                    $metrics084 = Join-Path $run084.FullName 'metrics.jsonl'
                    if (Test-Path -LiteralPath $metrics084) {
                        # 性能の低頻度ログに依存せず、通常ログの経験数で warmup 到達を確認する。
                        $tail084 = @(Get-Content -LiteralPath $metrics084 -Tail 32 -Encoding UTF8)
                        if (Test-WarmupReached084 $tail084) { $warmObserved084 = $timer084.Elapsed.TotalSeconds }
                    }
                }
            }
            # 到達後は毎回の採取を peak に含め、低頻度ログが tail に無い場合も記録する。
            if ($null -ne $warmObserved084) {
                $warmPrivate084 = [Math]::Max($warmPrivate084, $process084.PrivateMemorySize64)
                $warmWorking084 = [Math]::Max($warmWorking084, $process084.WorkingSet64)
                $warmGpu084 = [Math]::Max($warmGpu084, $used084)
            }
            $gpuNext084 = $timer084.Elapsed.TotalSeconds + 5
        }
    }
    $process084.WaitForExit();$process084.Refresh();$timer084.Stop()
    $record084.exit_code=$process084.ExitCode;$record084.end=[DateTimeOffset]::Now.ToString('o');$record084.elapsed_seconds=$timer084.Elapsed.TotalSeconds
    $record084.warmup_observed_seconds = $warmObserved084
    $record084.post_warmup_peak_private_bytes = $warmPrivate084
    $record084.post_warmup_peak_working_set_bytes = $warmWorking084
    $record084.post_warmup_device_extra_mib = if ($null -ne $warmObserved084) { $warmGpu084 - $record084.gpu_before_mib } else { $null }
    $record084.peak_private_bytes = $peakPrivate084
    $record084.peak_working_set_bytes = $peakWorking084
    $record084.peak_device_used_mib = $peakGpu084
    $record084.device_extra_mib = $peakGpu084 - $record084.gpu_before_mib
    if (-not $cpu084) {
        $record084.gpu_processes_after = @(& nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv)
        if ($LASTEXITCODE -ne 0) { throw 'Cannot record final GPU processes' }
    }
    $others084 = @(if (-not $cpu084) { Get-CimInstance Win32_Process -Filter "Name='AnetRLRunner.exe'" })
    if ($others084.Count -gt 0) { throw ('Concurrent Run invalidates measurement: '+($others084.ProcessId -join ',')) }
    $runs084 = @(Get-ChildItem -LiteralPath (Join-Path $root084 'runs') -Directory | Where-Object { $_.Name.EndsWith('prd084_forward_'+$Name) })
    if ($runs084.Count -eq 1) { $record084.run=$runs084[0].FullName }
    $record084 | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $recordPath084
    if ($record084.exit_code -ne 0 -or $runs084.Count -ne 1) { throw ('Run failed: '+$Name) }
    $normalizeArgs084 = @((Join-Path $repo084 'core/anet-core/testdata/prd084/compare_metrics.py'),'normalize',$record084.run,(Join-Path $root084 ($Name+'.metrics.jsonl')))
    if ($Noisy -or $Name.StartsWith('atari_')) { $args084 += 'run.eval_schedule.[greedy].interval=0' }
    if ($Noisy) { $normalizeArgs084 += '--include-noisy' }
    & (Join-Path $repo084 '.venv/Scripts/python.exe') @normalizeArgs084
    if ($LASTEXITCODE -ne 0) { throw 'Metric normalization failed' }
    Write-Output ('Finished '+$Name+' seconds='+$record084.elapsed_seconds)
}
function Compare-ForwardRun084([string]$Old,[string]$New) {
    & (Join-Path $repo084 '.venv/Scripts/python.exe') (Join-Path $repo084 'core/anet-core/testdata/prd084/compare_metrics.py') compare $Old $New
    if ($LASTEXITCODE -ne 0) { throw ('Numeric regression differs: '+$New) }
}
if ($Stage -eq 'off') {
    foreach ($case084 in $Cases) {
        $text084 = [IO.File]::ReadAllText((Join-Path $before084 ('old-config/'+$case084+'.txt')))
        $text084 = $text084.Replace(($repo084.Replace('\','/')+'/.scratch/prd084/old/source-config'),($root084.Replace('\','/')+'/source-config'))
        $text084 = $text084.Replace(($repo084.Replace('\','/')+'/.scratch/prd084/old/runs'),($root084.Replace('\','/')+'/runs'))
        $config084 = Join-Path $root084 ('config/'+$case084+'.txt')
        [IO.File]::WriteAllText($config084,$text084,[Text.UTF8Encoding]::new($false))
        Invoke-ForwardRun084 $case084 $exe084 $config084 500000
        Compare-ForwardRun084 (Join-Path $repo084 ('.scratch/prd084/old/'+$case084+'.metrics.jsonl')) (Join-Path $root084 ($case084+'.metrics.jsonl'))
    }
    if ($Cases -contains 'lunarlander_cpu') {
        Invoke-ForwardRun084 'old_checkpoint' $exe084 (Join-Path $root084 'config/lunarlander_cpu.txt') 256
    }
} else {
    $source084=$root084.Replace('\','/')+'/source-config'
    $text084=@(
        ('$include <'+$source084+'/_main.txt>'),('$include <'+$source084+'/Atari.txt>'),('run.$='+$chain084+'>run.@nz_reference'),
        'run.train.num_envs=128','E1.game=breakout','run.seed=1','backend.$=backend.@deterministic','backend.deterministic_warn_only=false',
        'app.$=app.batchrun>P1','app.show_error_dialog=false',('app.runs_dir='+$root084.Replace('\','/')+'/runs'),
        'run.eval_schedule.[greedy].interval=0',
        'DefaultDQNAgent.nn_forward.[ref_actor].noisy.mode=sample','DefaultDQNAgent.nn_forward.[ref_actor].noisy.hold=episode',
        'DefaultDQNAgent.nn_forward.[ref_actor].noisy.share=sample','DefaultDQNAgent.actor.[train].nn_forward_key=ref_actor',
        'DefaultDQNAgent.nn_forward.[ref_learner].noisy.mode=sample','DefaultDQNAgent.nn_forward.[ref_learner].noisy.hold=call',
        'DefaultDQNAgent.nn_forward.[ref_learner].noisy.share=sample','DefaultDQNAgent.learner.current.nn_forward_key=ref_learner',
        'DefaultDQNAgent.learner.target.nn_forward_key=ref_learner','DefaultDQNAgent.learner.munchausen.enabled=true',
        'DefaultDQNAgent.learner.munchausen.log_policy_mode=target'
    ) -join [char]10
    $config084=Join-Path $root084 'config/representative_on.txt'
    [IO.File]::WriteAllText($config084,$text084+[char]10,[Text.UTF8Encoding]::new($false))
    if ($Stage -eq 'on-after') {
        # 再レビューでは最終実行体の単発だけを取り、保存済み代表Runとの数値一致を検証する。
        Invoke-ForwardRun084 'on_after_review' $exe084 $config084 500000 -Noisy
        Compare-ForwardRun084 (Join-Path $repo084 '.scratch/prd084/forward-refactor/after/on_before_1.metrics.jsonl') (Join-Path $root084 'on_after_review.metrics.jsonl')
    } else {
        for ($repeat084=1;$repeat084 -le 3;$repeat084++) {
            Invoke-ForwardRun084 ('on_before_'+$repeat084) (Join-Path $before084 'bin/AnetRLRunner.exe') $config084 500000 -Noisy
            Invoke-ForwardRun084 ('on_after_'+$repeat084) $exe084 $config084 500000 -Noisy
            Compare-ForwardRun084 (Join-Path $root084 ('on_before_'+$repeat084+'.metrics.jsonl')) (Join-Path $root084 ('on_after_'+$repeat084+'.metrics.jsonl'))
        }
    }
}
