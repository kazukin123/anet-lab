# Stack 次元ネイティブ対応 実装メモ

## 概要

`DefaultDQNAgent` が Network 構築用に生成する stack 対象の input spec を、特徴次元への乗算から
`[stack_count, *original_shape]` へ変更する。これにより `stack_count > 1` の dummy forward、
Actor forward、Learner forward で stack 軸の意味を一致させ、時間軸 Conv1D から
`Flatten > Reshape` の回避構成を除去する。

## 主な変更

- `DefaultDQNAgent` のローカルな `network_obs_spec` だけを先頭 stack 軸へ変更する。
  `stack_keys` 対象外、EnvSpec、ObservationNormalizer、ReplayBuffer、`NetworkBuilder`、RainbowAgentは変更しない。
- LunarLander の temporal Conv を `Conv1D_Permute > TConv64 > ...` へ移行し、`ReS4F8` を削除する。
  共通設定の回避用 `ReS4` 定義も削除する。
- 汎用 `Reshape` module type の実装・登録は維持し、deprecated WARN は追加しない。
- 離散 Grid は spec/raw Network 入力では stack 軸を保持し、既存どおり one-hot 前処理時に
  channel へ統合する。DropMerge の CNN 設定は変更しない。
- NN/DQN 設計文書へ stack 軸契約と `Flatten`、`StackMerge`、temporal Conv、離散 Grid の
  前処理境界を記録する。`CONTEXT.md` と ADR は変更しない。
- 公開 C++ API、設定キー、`TensorSpec` 型は変更しない。GraphViz の input spec 表示は
  `[stack * feature]` から `[stack, feature]` へ変わる。

## テスト

- Public interface / surface: `DefaultDQNAgent` の public constructor、`CreateActor()`、
  `Actor::MakeAction()`、ActionInfo の NN trace を通して観測可能な shape と forward 成功を検証する。
- 優先 behavior:
  1. stack=4 の vector と Reshape なしの `Permute > Conv1d` が構築・行動選択でき、branch 入力が `(B,4,F)` になる。
  2. `Flatten > Linear` の MLP が stack 入力を従来どおり処理できる。
  3. 連続 Grid の `StackMerge > Conv2d` が dummy と実入力を `(B,S,C,H,W)` として扱える。
  4. 離散 Grid が one-hot 後に従来どおり `(B,S*C,H,W)` で CNN へ入る。
  5. `stack_keys` 対象外の入力には stack 軸が追加されない。
- TDD 順序: tracer bullet の1テストを RED にし、最小の spec 変更で GREEN にする。
  以後は上記 behavior ごとに1テスト追加、失敗確認、最小実装または既存挙動確認を繰り返す。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe "[native_stack]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- 対象は `use_stacker=true && stack_count>1`。`stack_count==1` の Actor/Replay 軸差は今回扱わない。
- 長時間学習、reward 曲線比較、bit 単位の Run 比較は完了条件に含めない。
- PRD と無関係な未コミット変更を保持する。
