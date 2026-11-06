# PRD078 検証

ReplayBuffer の観測履歴と bootstrap stack の整合性（履歴開始）を検証する資材。可視の単体テストと、hidden の網羅 matrix の 2 層からなる。

## 可視テスト

最小再現 3 本と境界の単体テストは通常タグ `[replay_buffer][frame_stack][history_start]` を持ち、既定スイートと `"[replay_buffer]"` で走る。

```bash
core\anet-core\bin\Debug\anet-core-test.exe "[history_start]"
```

## 384 条件 matrix（hidden）

matrix の TEST_CASE は `[.][integrity_assay]` だけをタグに持ち、既定スイートと `"[replay_buffer]"` には含まれない。条件は stack `{1,2,4}` × n_step `{1,2,3,5}` × lane数 `{1,4,16,128}` × 実lane容量 `{17,31}` × Uniform/PER × direct/CPU Prefetch。ランナーが 1 条件 1 プロセスで順次実行し、失敗しても続行する。通常の Debug ビルド済みで、リポジトリルートから実行する。

```bash
.\.venv\Scripts\python.exe core\anet-core\testdata\prd078\run_integrity_assay.py
```

- 個別再現は `--cases 325-328` のように単独番号または包含範囲を指定する。
- `--seed`（既定 `20260919`）は Catch2 の `--rng-seed` として渡り、matrix の TEST_CASE が `Catch::getSeed()` で読んで ReplayBuffer の抽選と unique probe に使う。入力履歴は固定の生成規則で、seed では変わらない。可視テストは固定 seed のまま。
- `--timeout-seconds`（既定 300）を超えた case は失敗として記録し、次へ進む。hang による全体停止を防ぐ上限で、性能の合格基準ではない。
- 出力は既定で `.scratch/prd078/<timestamp>/`。case 別ログ `case-NNN.log`、`results.csv`（case、状態、終了コード、秒、最後に完了した検査地点、失敗地点、ログパス）、`report.md`（成功/失敗数、合計時間、検査地点数、延べ key 数、照合サンプル数）。
- 失敗が 1 件でもあればランナーの終了コードは非 0。時間上限超過、クラッシュ、完了標記の欠落はすべて失敗に数える。

受入は全 384 条件の全検査地点完走。Debug で約 2 時間かかる。

## 実行が必要になるとき

ReplayBuffer の Storage 書込み、extractor の走査、`ValidIndexManager` の ready / history margin / dummy 除外、n-step queue と builder、`Push` の dummy 挿入と lane 状態、Prefetch の write-behind 順序を変更したとき。Env 継ぎ目の `episode_start` 契約や Runner の state 受け渡しを変えたときも同様。

## 関連

- 設計: [ReplayBuffer](../../../../docs/design/150_replay_buffer.jp.md) §2.3、§8
- 判断: [ADR 0044](../../../../docs/adr/0044-replay-frame-history-start-from-episode-start-at-push.md)
- 要求: PRD 078（`docs/memo` 配下、完了後は `done/`）§7.2〜§7.4
