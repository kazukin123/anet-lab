# ReplayBuffer観測履歴整合性修復 実装メモ

## 概要

Push時の`BatchState::episode_start`をslot単位の履歴開始として保存し、`obs`と`next_obs`のframe stackを同じ規則で復元する。公開`ReplayBuffer` API、設定、sampleable集合、PER・prefetch契約は変更しない。

## 主な変更

- `ReplayExperienceStorage`へゼロ初期化した`std::vector<uint8_t> history_starts_`を追加する。内部`Push(..., bool history_start)`で観測と同時に上書きし、dummyは必ずfalse、`DumpToLog()`にも出力する。レイアウトを隠す`IsHistoryStart(env_idx, physical_idx)`だけをextractorへ公開する。
- 内部helper `FindStackStart(storage, env_idx, latest_logical_idx, stack_count, capacity)`は、最新slotから古い方向へ履歴開始を探し、見つからなければstack幅の先頭を返す。`stack_count == 1`は走査せず、`obs`は`t`、`next_obs`は`t + actual_n`を最新位置として共用する。`terminals_`と`actual_n_steps_`は境界判定から外す。
- `DefaultReplayBuffer`に`std::vector<bool> lane_expects_episode_start_`をtrue初期値で持たせる。Push冒頭で全laneをpreflightし、書込み前に`state.episode_start`との一致を検証する。不一致は`ReplayBuffer::Push episode_start mismatch. lane=... logical_index=... expected=... actual=...`でfail-fastする。正常書込み後は`done || truncated`を次回期待値にし、dummyは検証対象にしない。
- matrixを`stack {1,2,4} × n_step {1,2,3,5} × lane {1,4,16,128} × capacity {17,31} × Uniform/PER × direct/CPU Prefetch`の384条件へ拡張する。matrixだけを`[.][integrity_assay]`とし、可視回帰テストは`[replay_buffer][frame_stack][history_start]`へ移す。
- `run_integrity_assay.py`は`--exe`、`--cases`（`N`または包含範囲`A-B`のカンマ区切り）、`--output`、`--seed`、`--timeout-seconds`を受ける。既定値はDebug実行体、全384件、時刻付き`.scratch/prd078/`、`20260919`、300秒とする。case別ログ、`results.csv`、`report.md`を生成し、timeout・非0終了・完了標記欠落を失敗として最後まで続行する。
- matrixは`Catch::getSeed()`をReplayBufferとprobe RNGへ渡す。可視テストは固定seedを維持する。CSVにはcase、状態、終了コード、秒数、最後の完了地点、失敗地点、ログパスを記録する。
- `docs/design/150_replay_buffer.jp.md`と英語版を新契約へ同期し、実装コメントと`AGENTS.md`の標準実行手順を更新する。既存の`CONTEXT.md`とstage済みADR 0044は保持し、履歴PRDは書き換えない。

## テスト

- Public interface / surface: `ReplayBuffer::Push`、`SampleUniqueUniform`、`Sample`、`Size`、PrefetchingReplayBufferの同期境界、整合性アッセイrunner。
- 優先behavior: 初回満杯の起動padding、n-step未確定中のbootstrap履歴、wrap後の旧metadata非干渉、episode_start整合性の全lane事前検証、短episodeとtruncationの終端観測、384条件の全地点完走。
- TDD順序:
  1. 初回満杯の既存再現をRED確認し、履歴開始の保存と`obs`復元だけでGREENにする。
  2. pending/wrapped metadata再現をRED確認し、同じhelperを`next_obs`へ適用してGREENにする。
  3. 初回push不整合テストをRED確認し、全lane preflightを実装する。その後、終端後の開始欠落・終端なし開始・書込み非発生を追加確認する。
  4. 容量到達前後、n-step=2否定制御、旧slot三種、短episode、長さ1のdone/truncation、metadata確定前後不変性を公開sampleで確認する。
- 旧`[!shouldfail]` 2件はfail-fastテストへ置換する。PRD指定のReplayBuffer 6件とDQN 3件は初回入力だけ`episode_start=true`へ移行し、期待値は変えない。Storage初期値テストには未書込み履歴開始がfalseであることを追加する。
- 固定条件325〜328を修正前後で同条件実行し、既存ProfileRangeの`DefaultReplayBuffer::Push`、`DefaultReplayBuffer::Sample`、extractor範囲についてcount・total・meanを比較する。新しい性能基盤や合否率は追加しない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
```

```powershell
core\anet-core\bin\Debug\anet-core-test.exe "[history_start]"
```

```powershell
.\.venv\Scripts\python.exe core\anet-core\testdata\prd078\run_integrity_assay.py
```

```powershell
core\anet-core\bin\Debug\anet-core-test.exe
```

matrixが既定suiteと`"[replay_buffer]"`に含まれず、全384条件・全検査地点が完走し、通常suiteが失敗0・期待失敗0であることを完了条件とする。

## 前提

- ReplayBufferはserialize対象外なので保存データmigrationは不要。
- true terminalの`next_obs`値は保証対象外だが、新しい履歴開始規則により決定的に復元する。
- CUDA転送経路、学習スコア改善、PinnedThreadPool修正、汎用test runner化は対象外。
- 無関係なbat変更、未追跡ファイル、ADR 0044のstage状態には触れない。
