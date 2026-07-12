# 近似Actor PER初期優先度 実装メモ

## 概要

PRD 035に従い、学習Actorが既存forwardから得たQヒントをReplayBufferへ運び、遷移のsampleable化境界で近似初期優先度を完成させる。`per_initial_priority_mode`の既定値は`fixed`とし、generation付きreplay item keyで上書き後のstale優先度更新を要素単位で棄却する。

## 主な変更

- `BatchActionInfo`へoptionalな`ActorQHint`を追加し、学習Actorの`actor_approx`時だけ連続`float32[B,2]`とCPU validityを生成する。評価・target・`fixed`・`max`では生成しない。
- `ExperienceSamples`はCPU `replay_item_keys`と`per_priority_sources`を返し、`UpdatePriorities`は`ReplayPriorityUpdateResult`を返す。
- ReplayBuffer内部のpriority操作を無効化、raw初期値、adjusted max初期値、Learner更新へ分離し、sourceとgenerationを追跡する。
- env別FIFOの初期優先度完成器を設け、非終端はbootstrap slotのActor state value、true terminalはbootstrap 0、truncatedはmax fallbackを使う。
- DQN推定器はLearnerとTBO・PER epsilon・clip規則を共有し、NetworkやRewardScalerを保持しない。
- source別sample/mass、Actor利用・fallback、stale drop、Actor/Learner比較メトリクスを追加し、設計文書とrunner設定を更新する。

## テスト

- Public interface / surface: `ActorQHint`、`ReplayBuffer::Push/Sample/UpdatePriorities`、DQN/Rainbow設定、`BatchUpdateResult` scalar、Serial/Pipeline経路。
- 優先 behavior: Actor hintから`actor_initial`までのtracer bullet、generation stale棄却、terminal/truncated、fixed/max、TBO、source/比較統計、prefetch戻り値、設定fail-fast。
- TDD順序: 各behaviorを1テストずつREDにし、最小実装でGREENへ戻してから次へ進む。refactorはGREEN後だけ行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[actor_initial]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per]"
core\anet-core\bin\Debug\anet-core-test.exe "[transfer]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][prefetch]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][tbo]"
core\anet-core\bin\Debug\anet-core-test.exe

cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target anet-core-test'
core\anet-core\bin\RelWithDebInfo\anet-core-test.exe "[nonfinite]"
git diff --check
```

## 前提

- `done && truncated`はtrue terminalを優先する。
- uniform modeは`fixed`だけを許可し、PER sourceは未定義、priority更新はdefault resultを返す。
- dirtyな`SumTree<double>`、double質量集計、追加テスト、LunarLanderの他設定、CONTEXTの他用語変更を保持する。
- PRDに列挙された既知ReplayBufferテスト5件だけをallowlistとし、追加テストは全pass、allowlist外失敗0を完了条件とする。
- 実装完了と実runでの採用判断は分離する。
