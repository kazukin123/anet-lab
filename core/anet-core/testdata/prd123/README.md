# PRD123 の検証

仕様は [PRD123](../../../../docs/memo/123_rnd_intrinsic_reward_scale_10prd.md)、判断・結果は [20impl](../../../../docs/memo/123_rnd_intrinsic_reward_scale_20impl.md) に残す。以下はリポジトリルートで実行する。

## 実装前の保存

コードを変更する前に、`AGENTS.md` の MSVC 初期化経由で Debug 全体をビルドする。新しい `.scratch/prd123/baseline-20261111/` を作り、Runner と DLL を `baseline-bin/`、設定一式を `baseline-config/` に複製する。既存の保存先は上書きしない。

既存テスト `[prd093-config-snapshot]` に、`ANET_PRD093_CONFIG_ROOT` で保存した設定の絶対パス、`ANET_PRD093_OUTPUT` で同じ保存先の `resolved-before/` の絶対パスを渡し、7 環境の全キー・文字列値を採取する。実行体の SHA256 とビルド結果も実装記録に残す。

## 前後の scalar 比較

`run_validation.py` は PRD098 の短い決定的 Run と比較の道具を使う。RND 無効の GridMaze・LunarLander・Atari・DropMerge は各8,704件、RND 有効の collect_errors と variance_feedback は LunarLander で各100,000件を採取する。後者は Atari の run.@rnd_taiga の8値を明示する。構成・CLI・実行体と設定の hash・Run の場所・時間・終了コードを条件ごとの execution.json に記録する。各条件が失敗しても後続は続け、最後に失敗を返す。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd123/run_validation.py before
```

実装後に Debug を全体ビルドする。実行体が変わる再実行には別の label を指定し、記録済みの Run を上書きしない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd123/run_validation.py after --label after-r1
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd123/run_validation.py compare --label after-r1
```

時刻・資源の群だけを除き、全 scalar 点の checksum と `metrics.scalar.defs` 全体を比べる。比較対象の基準は今回保存した実行体で、過去の PRD116 の記録は使わない。

## 変更前の checkpoint

before の collect 条件は `agent_close.anet` も保存する。次の load は、その checkpoint を実装後の同じ collect_errors 設定で読み、16件進めて正常終了することを確かめる。学習進行は checkpoint に保存されないため、終了条件は新しい Run の件数である。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd123/run_validation.py load --label after-r1 --case collect
```

## 単体・結合テスト

`anet-core-test.exe "[prd123]"` は Factory と facade の公開経路で、warm-up、EMA の手計算・batch 分割・時定数1、中心化と clip、設定の ERROR と通知、checkpoint の継続と識別子、done/truncated の学習報酬を確かめる。既存の `[prd116],[scaler],[shipped_config]` も回す。テストの通常ログを含む出力は `.scratch/prd123/` へ保存する。

設定変更時は PRD072 の `check_default_leaves.py` と、設定 snapshot の全キー・文字列値の前後比較で、新しい5キー以外に差が無いことを確かめる。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd123/compare_config.py
```

PRD116 の smoke 判定は `smoke_validation_test.py` で確認する。合成指標だけを使い、warm-up 中も learn_count が増える場合の点の選び方と、KFM の件数の明示を検証する。Runner は起動しない。

## 実験と設定の時機

KFM の10M×10本は本スクリプトの対象外。PRD §7.1 の段取りを別に示し、承認後にサブ機で実行する。Atari.txt の warm-up 200,000 行も、メイン機の Montezuma キューの stage20 終了を確認してから入れる。未確認ならその行と断定するテスト・文書は足さず、課題へ残件を記す。
