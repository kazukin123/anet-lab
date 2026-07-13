# 初期優先度hint運搬の汎用化 実装メモ

## 概要

PRD 035とADR 0012に従い、DQN固有の`ActorQHint`とCPU validityを汎用RL層から除去し、単一の`float32[B,K]`を不透明に運ぶ`ReplayInitialPriorityHint`へ置き換える。初期優先度の数式、completionタイミング、fallback、source、`ReplayBuffer::Push`の公開契約は変更しない。

## 主な変更

- `ReplayInitialPriorityHint`はdefined、rank 2、`B > 0`、`K > 0`、`float32`を検証し、detach済み連続tensorと遅延CPU cacheだけを保持する。`BatchActionInfo`はoptional carrierを解釈せず運ぶ。
- ReplayBuffer内部はCPU化した各行を`c10::SmallVector<float, 4>`へコピーし、開始・bootstrap hintを`std::span<const float>`として推定器へ同期的に渡す。
- `InitialPriorityEstimator`へ`ValidateHint`を追加し、schema違反とnonfiniteを区別する。truncatedでも開始hintを先に検証し、true terminalのbootstrap spanは空にする。
- DQN moduleへ`K = 2`の列定義とtensor/scalarのpack/decode helperを集約し、Actor、`DQNActionInfo::WithAction`、DQN推定器で共用する。
- `docs/design/110_agents_and_learning.jp.md`と`docs/design/140_data_pipeline.jp.md`の該当箇所を新しいcarrier契約へ同期する。

## テスト

- Public interface / surface: `ReplayInitialPriorityHint`、`BatchActionInfo`、`ReplayBuffer::Push/Sample`、`InitialPriorityEstimator`、DQN Actorと`DQNActionInfo::WithAction`。
- 優先behavior: carrier形式とCPU cache、任意`K`のReplay運搬、DQN `K = 2` schema、action差し替え後の再gather、hint有効化時のforward非増加、terminal/truncated/nonfinite分類。
- TDD順序: behaviorごとに1テストをREDにし、最小実装でGREENへ戻してから次へ進む。共通化と改名は関連テストがGREENになった後だけ行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][per][actor_initial]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per][actor_initial]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per]"
core\anet-core\bin\Debug\anet-core-test.exe

cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target anet-core-test'
core\anet-core\bin\RelWithDebInfo\anet-core-test.exe "[actor_initial][nonfinite]"
git diff --check
```

## 前提

- TBOとpriority確定式の共有は後続④、completion componentとsampleable境界の分離は後続⑤、WARNとpriority更新契約は後続⑥で扱う。
- `035_approx_actor_priority_per_20impl.md`は履歴として変更しない。
- PRD 035記載の既知ReplayBufferテスト5件だけをallowlistとし、追加テスト全pass、allowlist外失敗0を完了条件とする。
- stageとcommitは行わず、その他のdirty/untracked filesには触れない。
