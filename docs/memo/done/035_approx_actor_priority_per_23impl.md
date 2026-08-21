# completion責務とsampleable範囲分離 実装メモ

## 概要

PRD 035とADR 0012に従い、`DefaultReplayBuffer`が保持する初期優先度completionのstateと判断を、内部Module `InitialPriorityCompleter`へ移す。uniform ReplayBufferではcompleterとpending FIFOを構築せず、未消費entryが蓄積する経路をなくす。

公開`ReplayBuffer` interface、priority数値、source、metrics、fallback、completionタイミングは変更しない。

## 主な変更

- `InitialPriorityCompleter`がenv別pending FIFO、mode、Estimator、fallback判断、初期source決定、completion counterを所有する。
- `DefaultReplayBuffer`は`metadata_mutex_`、`ValidIndexManager`、`ReplayPriorityStore`を所有し、同じ同期区間でpending登録、sampleable化、completionを行う。
- production/test共通factoryはPERでだけcompleterを構築し、uniformでは`nullptr`を返す。
- `ValidIndexManager`の列挙、単点判定、上書き判定は同じ論理sampleable範囲を使う。

## Interfaceと責務

- `InitialPriorityCompleter::Enqueue`は、`ExperienceQueue`がn-step/系列を確定した後のreal transitionだけを受け取る。
- `InitialPriorityCompleter::CompleteReady`は、callerが`metadata_mutex_`を保持した状態で呼び、sampleableなFIFO先頭からpriority/sourceを適用する。
- completerはmutex、SumTree、source配列、generation、Learner更新stateを所有しない。
- 現在は開始itemごとにscalar priorityを1つ完成させる。MuZero等で系列内の複数stepを集約する場合は、Estimator入力とpending表現を拡張する。

## TDD

1. production factoryのPER/uniform構築契約をRED→GREENで追加する。
2. `ValidIndexManager`各経路のcharacterizationをGREENにし、range helper抽出後も維持する。
3. completerのsampleable待ち、各mode、terminal、truncated、nonfinite、契約違反、priority 0、counterをbehavior単位でRED→GREENにする。
4. `DefaultReplayBuffer`をcompleterへ委譲し、既存public経路のテストを回帰確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][per][initial_priority_completer]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][sampleability]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][per][actor_initial]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe

cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target anet-core-test'
core\anet-core\bin\RelWithDebInfo\anet-core-test.exe "[replay_buffer][per][initial_priority_completer][nonfinite]"
git diff --check
```

## 前提

- PRD 035記載の既知ReplayBuffer失敗5件は修正しない。
- WARN、`static once_flag`、generation、duplicate、統計、Serial/Pipeline契約は後続⑥で扱う。
- PER有効fixedの長時間Runで観測された速度低下は別調査とし、本変更へ推測修正を混ぜない。
- 既存`20impl`～`22impl`とその他のdirty/untracked filesには触れず、stageとcommitは行わない。
