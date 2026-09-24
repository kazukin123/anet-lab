# config警告とpriority更新契約補強 実装メモ

## 概要

PRD 035の仕上げとして、Replay初期優先度設定のWARN発行単位、generation付きpriority更新、duplicate、Actor/Learner比較統計、Serial/Pipeline経路をテストで固定する。`per_clipped_count`はclip前priorityが上限を厳密に超え、実際に値が変わった件数へ修正する。

priority数値、completion、source、fallback、公開APIは変更しない。

## 責務と内部interface

- `ParseReplayInitialPriorityMode`は文字列から内部modeへの変換だけを行う。
- `ValidateReplayPriorityConfig`はLearner構築単位の設定検証とWARN発行を行う。
- generation付きkeyのchecked helperはencode/decodeの算術境界を隠蔽し、`DefaultReplayBuffer`は現在generationとの比較に専念する。
- Actor/Learner比較helperは正値倍率統計と平均rank Spearmanを一箇所で計算する。
- PER priority readbackはclipped priorityとclip件数を1つのpacked tensorで運び、物理D2Hを1本に保つ。

## 互換性

- `ReplayBuffer`、`InitialPriorityEstimator`、`ReplayPriorityUpdateResult`の公開契約は変更しない。
- stale/current/duplicateの適用順、priority source、最大優先度、completion/fallbackを維持する。
- `per_prio_clip_ratio`だけを、PRDどおり実際にclipされた割合へ補正する。

## TDD

1. config parseとvalidatorのWARN発行単位を1 behaviorずつRED→GREENにする。
2. key境界、batch atomicity、stale/current、duplicateを順にRED→GREENにする。
3. Actor/Learner比較統計の決定的な入力をRED→GREENにする。
4. clip未満・等値・超過とpacked readbackをRED→GREENにする。
5. Prefetching、Serial、Pipelineの運搬契約をRED→GREENにする。

## 変更ファイル一覧

- `core/anet-core/src/dqn_based_agent.hpp`: config validatorとpriority/countのbatch結果型、packed readback契約を宣言した。
- `core/anet-core/src/dqn_based_agent.cpp`: config検証・WARNをparseから分離し、strict clip countを単一packed D2Hで運ぶよう修正した。
- `core/anet-core/src/dqn_based_agent_test.cpp`: config境界、WARN、CPU/CUDA clip境界、ReplayBuffer更新結果のmetrics反映を固定した。
- `core/anet-core/src/replay_buffer_impl.hpp`: generation付きkey codecとActor/Learner比較統計の内部interfaceを宣言した。
- `core/anet-core/src/replay_buffer_impl.cpp`: checked key codec、batch先行検証、比較統計helperを実装した。
- `core/anet-core/src/replay_buffer_test.cpp`: generation、atomicity、stale/current、duplicate、最大優先度、統計、Prefetch透過返却を固定した。
- `core/anet-core/src/trainer_test.cpp`: Serial/Pipelineの両経路でopaqueなK=3 hintがReplayBuffer境界へ届くことを固定した。
- `docs/memo/035_approx_actor_priority_per_24impl.md`: 実装方針、実差分一覧、検証結果を記録した。

一覧は最終`git diff`と`git status`を照合して確定した。本タスク開始前から存在した無関係なdirty/untracked filesは含めていない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][config][per]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][per][generation]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][per][actor_comparison]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per][clip]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per]"
core\anet-core\bin\Debug\anet-core-test.exe "[trainer][replay_hint]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][prefetch][priority_update]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe

cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target anet-core-test --parallel 1'
core\anet-core\bin\RelWithDebInfo\anet-core-test.exe "[dqn][per][clip]"
git diff --check
```

## 検証結果

- TDDではconfig validator、key codec、比較統計helperを未宣言によるcompile REDから実装し、clip境界は旧`>=`判定が期待1件に対して2件を返すruntime REDを確認してからGREENにした。PrefetchとSerial/Pipelineは既存透過動作を固定するcharacterization testとして追加時からGREENだった。
- Debug `anet-core-test` build: 成功。
- Debug関連タグ:
  - `[dqn][config][per]`: 3 test cases / 33 assertions、全pass。
  - `[dqn][per][clip]`: 2 test cases / 48 assertions、全pass。CUDA利用可能環境でCPU/CUDA双方を実行した。
  - `[replay_buffer][per][generation]`: 8 test cases / 68 assertions、全pass。
  - `[replay_buffer][per][actor_comparison]`: 2 test cases / 23 assertions、全pass。
  - `[replay_buffer][prefetch][priority_update]`: 1 test case / 7 assertions、全pass。
  - `[trainer][replay_hint]`: Serial/Pipeline 2 sections / 6 assertions、全pass。
  - `[dqn][per]`: 16 test cases / 169 assertions、全pass。
- Debug `[replay_buffer]`: 73 test cases中68 pass、既知allowlist 5件だけfail。
- Debug全テスト: 293 test cases中288 pass、3186 assertions中3181 pass。失敗は次の既知allowlist 5件だけで、allowlist外失敗は0件だった。
  - `ReplayBuffer n-step returns stop at episode_start without done`
  - `ReplayBuffer excludes wrapped samples whose frame stack would read overwritten frames`
  - `ReplayBuffer PER samples only safe wrapped frame-stack indices`
  - `ReplayBuffer wrapped sampleability honors both frame stack and unroll horizons`
  - `ReplayBuffer frame stacking starts a new stack at episode_start without done`
- RelWithDebInfo `anet-core-test` build: 成功。
- RelWithDebInfo `[dqn][per][clip]`: 2 test cases / 48 assertions、全pass。CPU/CUDA双方のpriority/countとpacked readbackを確認した。
- `git diff --check`: pass。対象8ファイルはLF、末尾空白なし、EOF改行あり。
- 未実行検証はない。
- stageとcommitは行っていない。

## 前提

- PRD 035記載の既知ReplayBuffer失敗5件は修正しない。
- 無関係な`static once_flag`、長時間Runの速度低下、既存`20impl`～`23impl`は対象外とする。
- その他のdirty/untracked filesには触れず、stageとcommitは行わない。
