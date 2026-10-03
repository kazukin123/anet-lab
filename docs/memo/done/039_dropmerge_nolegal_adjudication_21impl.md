# PRD039 Phase 1b・Phase 2 NoLegal 裁定 実装メモ

## 概要

`039_dropmerge_nolegal_adjudication_10prd.md` の Phase 1b と Phase 2 を実装する。
`DropMergeEnvConfig` の既定値は裁定 OFF・`no_legal_min_blocked_frames = 60` とし、
active `E` overlay では裁定 ON・N=60・`no_drop_timeout_gameover_penalty = -10` を新規 Run から適用する。
settled な NoLegal candidate は従来どおり即時受理し、blocked persistence を受理上限保証として OR 追加する。

## 主な変更

- episode 終端時の未解消 `blocked_candidate_frames_` を `ep_terminal_blocked_frames` として確定し、
  解消済み blocked run 数を `ep_blocked_run_count` として公開する。episode 外では NaN を返す。
- `DropMergeEnvConfig` に `use_no_legal_adjudication = false` と
  `no_legal_min_blocked_frames = 60` を追加する。
- Env 構築時に N >= 1、および裁定 ON・timeout done ON・timeout 有効時の
  N < `no_drop_timeout_steps` を `ANET_SYSTEM_ERROR` で検証する。
- NoLegal 受理を settled fast-path または persistence >= N とする。
  既存の終了判定順、報酬、Observation、RNG、物理ループは変更しない。
- fast-path は既存ログを維持し、persistence 経路は英語の専用 verbose log を出す。
  両方成立した場合は fast-path を優先する。
- `DropMerge.txt` の active `E` を裁定 ON・N=60・timeout 罰 -10 に切り替える。
  PH1b raw metrics は train の 25/26、eval1/eval2 の 79/80 を使い、EMA は追加しない。
- `CONTEXT.md` と ADR 0014 は変更しない。

## テスト

- Public interface / surface: `DropMergeEnvConfig`、`DropMergeEnv::Step()`、
  `SingleState.done` / `truncated`、step reward、`DropMergeEnv::GetScalar()`、
  verbose log、`DropMerge.txt`。
- 優先 behavior:
  1. settled NoLegal 終端で terminal blocked frames が正になり、episode 中は新 scalar が NaN。
  2. legal timeout と blocked 解消後の終端では terminal blocked frames が 0。
     解消済み run は `ep_blocked_run_count` に載る。
  3. config の既定値・読み込み・不正 N・timeout 競合が契約どおり。
  4. 裁定 OFF の unsettled blocked 盤面は従来どおり timeout。
  5. 裁定 ON では settled 即時受理、unsettled 盤面の N-1 非終端・N 到達受理、
     N 未満での legal 復活、legal 盤面の従来 timeoutが成立。
  6. persistence 受理は done=true・truncated=false・NoLegal 無罰で専用ログを出す。
- TDD 順序: 上記を 1 behavior ずつ、テスト追加 -> RED 確認 -> 最小実装 -> GREEN 確認で進める。
  共通 fixture の整理は GREEN 後だけ行い、production 本体へ test-only API を追加しない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target DropMergeEnv-test'
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe --anet-test-failure-dialog=off
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
git diff --check
```

active metric ID の重複も確認する。追加コストは既存の
`DropMergeEnv::Step` / `isNoLegalCandidateState` / `hasAnyLegalDropForCurrentFruit`
の Profile でスモーク確認し、新しい細粒度 ProfileRange は追加しない。

## 前提

- N=60 は今回の実装値として確定し、PH1b metrics は導入後の受理経路確認と将来の再調整に使う。
- active `E` の変更は新規 Run 専用とし、進行中・resume Run へ途中適用しない。
- fast-forward、timeout 再分類、追加裁定上限、Agent / Replay、既知の metrics ID 不整合、
  長時間 Run / A/B は対象外。
- ユーザーの無関係な未コミット変更を保持する。
