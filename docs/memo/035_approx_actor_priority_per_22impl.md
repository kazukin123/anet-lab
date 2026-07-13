# 初期優先度計算のLearner数式共有 実装メモ

## 概要

PRD 035 §5に従い、DQN初期優先度推定器の独自scalar数式を廃止し、Learnerが使うtensor版と同名のscalar overloadへ統合する。数値仕様、設定、metrics、source、fallback、completionタイミングは変更しない。

## 主な変更

- DQN namespaceへ`TransformH`、`TransformHInv`、`MakePerRawPriority`のscalar overloadをtensor版と隣接定義し、符号、ゼロ、`per_eps`、clip境界を揃える。
- `DqnInitialPriorityEstimator`は独自scalar helperと手書きpriority確定式を廃止し、共有scalar helperだけを使用する。Push hot pathではtensorを生成しない。
- 内部DQN factory `CreateInitialPriorityEstimator(const LearnerConfig&)`を追加し、`SetupReplayBuffer`とテストで同じ推定器生成経路を使う。
- Learnerは既存tensor helperを引き続き使用する。n-step target合成は共通化せず、実`TDLearner`との数値一致テストで拘束する。

## テスト

- Public interface / surface: DQN namespaceのtensor/scalar数式helper、`InitialPriorityEstimator`、実`TDLearner::UpdateFromSamples`の更新結果。
- 優先behavior: TBO変換のtensor/scalar一致、raw priority確定policyの一致、1-step/n-step/terminal/TBO/QR平均Q/priority 0/clip/nonfinite、scalar DQN Learnerとの同一入力一致。
- TDD順序: 数式helper、production factory経由の推定器、実Learner比較の順に、各behaviorを1テストずつRED→GREENへ進める。refactorは関連テストがGREENになった後だけ行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][tbo]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per][actor_initial]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][per]"
core\anet-core\bin\Debug\anet-core-test.exe

cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target anet-core-test'
core\anet-core\bin\RelWithDebInfo\anet-core-test.exe "[dqn][per][actor_initial][math]"
core\anet-core\bin\RelWithDebInfo\anet-core-test.exe "[dqn][per][actor_initial][estimator]"
git diff --check
```

## 前提

- QR×TBOの完全一致は要求しない。completion componentとsampleable境界の分離は後続⑤、WARNと`per_clipped_count`を含むpriority更新契約は後続⑥で扱う。
- `035_approx_actor_priority_per_20impl.md`と`035_approx_actor_priority_per_21impl.md`は変更しない。
- PRD 035記載の既知ReplayBufferテスト5件だけをallowlistとし、追加テスト全pass、allowlist外失敗0を完了条件とする。
- stageとcommitは行わず、未追跡`docs/design/`を含むその他のdirty/untracked filesには触れない。
