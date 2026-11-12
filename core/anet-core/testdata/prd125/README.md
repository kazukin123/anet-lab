# PRD125 検証資材

正本は [PRD125](../../../../docs/memo/125_dqn_learner_update_skeleton_10prd.md)、実施結果と未実施の項目は [20impl](../../../../docs/memo/125_dqn_learner_update_skeleton_20impl.md) に記録する。

`run_validation.py` は PRD098 の短縮条件、PRD109 の隔離した process 実行と失敗ケース採取、PRD084 の scalar 正規化を呼ぶ。設定と実行体の保存先は `.scratch/prd125/`。既存の label と保存先の上書きは拒否する。

## 保存と入力

- 変更前の Debug・Release を、AGENTS.md の MSVC 初期化を経由してビルドしてから `snapshot` を実行する。実行体・DLL・設定・core の Debug テスト実行体・既存 DQN テスト 2 ファイルを `baseline/` へコピーする。commit、Git の変更の有無、ファイルごとの SHA-256 は `baseline/manifest.json` に保存する。実行体のコピーを終えるまで DQN 本体を編集しない。
- `prepare` は、保存した GUI テストの `[prd109_config]` で凍結設定を解決する。条件の定義は [run_validation.py](run_validation.py) の `VARIANTS`・`definitions`・`expected_values`。既定 4 組、LunarLander の 3 分布 × 9 組が FP32 の 31 組である。
- BF16 は LunarLander の target+TBO+skip を 3 分布で、Atari の既定を 1 組。LunarLander は CPU を先に試し、一致しないときだけ CUDA も試す。どちらでも一致しない BF16 の組を明示して除外し、既存の `[prd103]` で確かめる。FP32 の組は除外しない。
- 現在の凍結入力は `inputs-r2/`。最初の試行で保存した `inputs/` と、その Run・失敗ログは当時の記録として残す。r2 は TD/QR の配線へ切り替えるとき、Agent 全体の選択も切り替える。変種の IQN 診断を含む既定の M1 を保持し、必要な replay fit・Munchausen・skip の群を追加する。

## 比較

seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価停止、決定的 backend。GridMaze/LunarLander は CPU、Atari qbert/DropMerge は CUDA。CPU 条件だけ最大 4 組を並行し、CUDA 条件は逐次実行する。

`repeat` は保存した Debug の実行体で 2 回ずつ流す。Run の `config/config_data.txt` で指定値を検査し、変種の replay fit、Munchausen 有効時の診断を検査する。IQN は全群で 7 tag の定義を確認し、PER 有効群は 7 tag、PER 無効群は 01〜03 の実点と少なくとも 1 つの有限値を確認する。PER 無効群の未成立の 04〜07 は実点を求めない。正規化した全点を比較し、除くのは `90_perf/`・`91_resource/`・`92_resource/` だけ。Noisy の点も含む。

`compare` は基準の 1 回目と変更後の Debug を、同じ凍結設定で比べる。基準の config SHA-256 と現在の入力を照合し、指定値と診断点も再検査する。各 label の `-summary.json` に点数・一致・ハッシュ・参照元を残す。独立した組の失敗後もほかの組を確認する。

`suite` は既定の core suite を JUnit と process log へ保存する。`baseline-suite` は保存した実行体を使い、変更後の label ではその失敗ケース集合との一致を検査する。

## 性能

`performance` は他の Runner、core test、ビルドが動いていれば開始しない。Release、Atari qbert/BTR、IQN、Munchausen target、BF16、prefetch、128 lane、batch 256、skip 無効、warmup 256、82,176 experience。凍結入力の実効値も検査する。

捨て Run のあとに保存した実行体→変更後を 3 巡以上流す。既定は 3 巡で、`--rounds` で開始前に巡数を指定できる（3 未満は拒否する）。learn step 64〜256 の時間差を 192 で割り、各 Run の秒/更新、基準の中央値と最大値、変更後の中央値、合否を `-summary.json` に残す。変更後の中央値が基準の最大値以下であることを受入にする。

## 式のテストと破壊検査

`dqn_munchausen_test.cpp` の `[prd125]` は、通常 Q/分位点 × bonus 有無 × TBO 有無 × 終端指定の 16 組を、Network の出力を使わず手計算と比較する。2 行に異なる return・n-step 数・終端旗を置き、終端でも bonus が残ることと、勾配を作らないことを確認する。

`run_mutations.py` は、共通の式の h の欠落、終端マスクの欠落、終端時 bonus の欠落を一つずつ入れる。各変更で `[prd125]` が失敗し、元の式へ戻してビルドし直すと成功することを確認する。実行中は同じソースの編集・ビルド・テストを重ねない。対象の一行だけを書き換え、例外時も復元と再ビルドを行う。label の出力先を再利用せず、各 build と test の終了値・JUnit・ログと、`summary.json` を `.scratch/prd125/` 以下へ保存する。

## 実行済みの入口

リポジトリルートの PowerShell から実行する。保存と採取は一度だけ行い、追加の検証は新しい label を使う。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd125/run_validation.py snapshot
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd125/run_validation.py repeat --label baseline --workers 4
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd125/run_validation.py suite --label baseline-suite
```
