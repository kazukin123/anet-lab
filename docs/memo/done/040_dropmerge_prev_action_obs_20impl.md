# DropMergeEnv 直前行動観測 実装メモ

## 概要

`DropMergeEnv` 内だけで prev-action trio、DROP 列マーカー、固執監視 metric を実装する。
2 つの観測 flag は default false とし、PRD039 の既存実装を保持する。
Agent、NN、ReplayBuffer、既存 snapshot は変更しない。

## 主な変更

- `DropMergeEnvConfig` に `obs_include_prev_action` と `obs_prev_drop_marker` を追加し、
  config 読み込みと dump へ反映する。いずれかが ON の move 系は構築時に fail-fast する。
- 選択された直前命令を `last_action_` に保持し、Reset は未行動へ戻す。
  trio ON 時は既存 vector 末尾へ `[valid, noop, drop_x]` を追加し、
  spec、buffer、実データ次元を一致させる。
- marker ON 時は直前 DROP 命令列を top row の class 12 で描画する。
  Reset／NOOP では描画せず、grid spec は変更しない。
  busy 中に棄却された DROP も選択命令として扱う。
- direct 系の successive DROP 命令列から `ep_same_drop_col_ratio` を算出し、
  episode 終端時だけ返す。NOOP は直前 DROP 列を消去せず、move 系では NaN を返す。
- `DropMerge.txt` には両 flag のコメント例と metric 6 本を追加する。
  PRD039 の番号を保持するため、新規番号は `42_env/27,73`、
  `51_eval1/81,88`、`52_eval2/81,88` とする。
- `CONTEXT.md` は既に用語定義済みのため変更せず、新規 ADR も作成しない。

## テスト

- Public interface / surface: `DropMergeEnvConfig`、`GetSpec()`、
  `Reset()`／`Step()` が返す Observation、
  `GetScalar("ep_same_drop_col_ratio")`、runner config。
- 優先 behavior:
  1. config 読み込みと direct_noop の trio spec／Reset／NOOP／DROP。
  2. direct mode と move／move_fast の fail-fast。
  3. marker の DROP／NOOP／Reset と grid spec 不変。
  4. ON/OFF 同 seed 並走、終端 state、train/eval prefix の batch `ValidateObservation`。
  5. ratio の既知列、DROP 2 未満、NOOP を挟む列、move 系 NaN。
  6. busy 中の棄却命令を `[i,j,j]` のような列で観測と ratio の双方から検証し、
     果物数が増えないことも確認する。
- TDD 順序: 既存 default-OFF 契約を characterization test で固定した後、
  上記を 1 behavior ずつ test 追加、RED 確認、最小実装、GREEN 確認の順で進める。
  private 実装や test-only API は検査せず、整理は全 GREEN 後に限る。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target DropMergeEnv-test'
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe --anet-test-failure-dialog=off
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
git diff --check
```

追加 metric の番号・完全 tag 重複、対象ファイルの LF、
無関係な差分が変更されていないことも確認する。

## 前提

- flag ON は新 observation 契約なので既存 `.anet` snapshot を読み込まず、
  run 名 `_pa` の新規 Run とする。
- runner smoke、100M A/B、700M 長期 Run、commit はユーザー実施であり、
  コード実装完了条件には含めない。
- ユーザーの無関係な未コミット変更を保持する。
