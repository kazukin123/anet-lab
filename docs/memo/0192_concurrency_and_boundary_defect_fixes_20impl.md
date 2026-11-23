# PRD192 監査で見つかった不具合 5 件の実装計画

## 概要

[PRD192](0192_concurrency_and_boundary_defect_fixes_10rfd.md) の D1〜D17、受入 A1〜A15、§4 の 5 課題。実装前に現行の Debug Runner とテスト実行体 10 本をビルドする。

## 主な変更

Notifier の登録と配信を並行して呼べるようにする。動画は画像の大きさを照合してから書き込む。DQN の強制行動では、ヒントの計算だけで Q と同じ device を使う。同名の直接登録は拒否し、公開初期化関数は登録結果を保存する。QValuePanel は平滑化半径を正整数で検証し、加工側は重みから半径を得る。

PRD §6.7 の設計文書 5 本を更新する。QValuePanel の設定契約は「アプリケーションとツール」に追記する。

## テスト

変更の前後でテストケースごとの結果を採取する。以下の各行は確認対象を小さく分け、対象のテストで RED、最小実装で GREEN、そのあとに整理する。既存の期待値は維持する。

| 順 | 公開の入口と観測する挙動 | 実装と検証 |
|---|---|---|
| 1 | `Notifier` に複数の Observer を登録し、通知中に自分を外しても、登録順で残りが呼ばれ、次の通知に反映される | 最初に `[notifier][contract]` でこの経路を通す。変更時にだけ Observer の列を複製し、通知中は取得した一覧を共有する。通知中の登録、別スレッドからの解除後の寿命、Train・Learn の同時実行、残る読み出しと変更の入口を順に検証する。ストレス版は 2,000,000 回。キャッシュの呼び出し条件と Runner の 8 か所をコメントに記録する（A1〜A3・A15） |
| 2 | `MetricsLogger` の画像記録へ大きさ違いと無効な画像を渡す | `[metrics][video][frame_size]` で tag ごとの WARN と文言を確認し、ffmpeg で動画を読み戻して有効な画像だけが記録されたことを確かめる。動画へ書く直前に検証する（A4） |
| 3 | DQN の `WithAction` に CPU の行動を渡し、CUDA のヒントを取り直す | `[dqn][forced_action][device]` でヒントの比較と、ヒントの有無に共通する結果の device を CUDA で確認する。取り直しに使うテンソルだけを移す（A5・A6・A12） |
| 4 | Repository の重複登録と、10 個の初期化関数の再呼び出し | Repository ごとに `[repository][duplicate]` で例外と先に登録した factory を確認する。初期化関数ごとに `[init][idempotent]` で再呼び出しと登録結果を確認する。初期化中の例外は記録し、同じ関数の次回呼び出しでも送出する。GridMaze の既存テストは初期化関数を使う。既定スイートで回帰を確認する（A7〜A9） |
| 5 | QValuePanel の設定読み込みと `ComputeQValueHeatMap` | `[q_value_smooth]` で無効値の拒否・有効値の受理を確認する。加工は重みだけから半径と有効状態を決め、空・奇数・偶数個を確認する。既存の加工テストは期待値を保つ（A10・A11） |

## 検証

- Visual Studio の `VsDevCmd.bat` で x64 環境を初期化し、CMake の `x64-Debug` preset で Debug Runner と PRD §7 のテスト実行体 10 本（GUI を明示指定）をビルドする。
- 既定スイートを前後で実行し、テストケースごとの結果を `.scratch/prd192` に保存する。前に通ったケースの後退が 0 件か照合する。追加タグと隠しストレスを別に実行する。
- PRD103 の検証資材と同じ 6 構成・短縮条件を使う。変更前に保存した Runner と変更後の Runner で実行し、PRD103 の正規化方法を使って A13 のスカラーをバイト比較する。入出力と結果は `.scratch/prd192` に置く。
- A11・A12 の指定検索、A14 の手動操作（学習中に両 Panel を各 10 回開閉、GPU 評価で強制行動を 10 回）、A15 の設計・コード照合を実施する。未実施の項は理由とともに記録する。

主な実行コマンドは以下のとおり。各コマンドの標準出力と標準エラーは `.scratch/prd192` の個別ログに保存し、終了値と集計を確認する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AnetRLRunner AnetRLRunner-gui-test anet-core-base-test anet-algo-dqn-test anet-algo-muzero-test anet-algo-imagecls-test anet-env-atari-test anet-env-dropmerge-test anet-env-gridmaze-test anet-env-imagecls-test anet-env-lunarlander-test'
.\out\test\Debug\anet-core-base-test.exe '[notifier_stress]'
.\out\test\Debug\anet-algo-dqn-test.exe '[init_failure]'
.\.venv\Scripts\python.exe .scratch\prd192\suites.py after
.\.venv\Scripts\python.exe .scratch\prd192\suites.py compare
.\.venv\Scripts\python.exe .scratch\prd192\metrics.py after
.\.venv\Scripts\python.exe .scratch\prd192\metrics.py compare
.\.venv\Scripts\python.exe .scratch\prd192\suites.py final
.\.venv\Scripts\python.exe .scratch\prd192\suites.py compare-final
.\.venv\Scripts\python.exe .scratch\prd192\metrics.py final
.\.venv\Scripts\python.exe .scratch\prd192\metrics.py compare-final
```

## 前提

- 現行では `WithAction` が存在し、PRD161 の単位 A は未実装。PRD146 の課題は別工程。ADR 0082 は既に記録されている。
- 出荷設定の平滑化半径は 5、設定を省略したときは 4。強制行動の結果は呼び出し側から渡された device に置く。
- PRD の設計判断は確定済み。CUDA 対応 GPU（RTX 5090）と ffmpeg を確認した。CUDA の実行可否はテストで確かめる。
- 登録処理の例外を捕捉する共通処理は既存の `anet/init.hpp` の `detail` に置き、10 個の初期化関数はそれぞれ最初の登録結果を保持する。DQN では 2 つ目の登録を意図的に失敗させる隠しテストで、再呼び出しが同じ失敗を返すことも確かめる。
- 通知のロックは 1 回の通知あたり約 0.1 µs の見積もりで、通常は 1 step につき最大 2 回である（PRD §1.3）。CUDA への行動転送は EvalPanel の強制操作ごとに小さな整数テンソル 1 回で、学習と通常評価の step には入らない。短いホスト・デバイス転送として 1 回数十 µs 程度を見込む。

## 経過

- 2026-11-20 未決事項監査。ユーザー判断が必要な項目は 0 件。5 課題の `prd` が PRD192 を指すことを確認した。
- 2026-11-20 保存済み計画と Notifier の最初の変更を引き継いだ。既存の RED・GREEN に加え、`[notifier]` が 4 ケース、隠しストレスが 1 ケース通過した。
- 2026-11-20 `[metrics][video][frame_size]` は修正前に動画パイプへの書き込みで失敗し、修正後は既存の動画テストと合わせて 10 ケース通過した。`[forced_action][device]` は `cpu` と `cuda:0` の不一致、基盤の `[repository][duplicate],[init][idempotent]` は上書きで修正前に失敗した。
- 2026-11-20 `[q_value_smooth]` は修正前に半径 0・-1・1.5 と偶数個の重みを受理し、2 ケースとも失敗した。修正後は既存の加工テストを含む GUI の対象検証 7 ケースが通った。
- 2026-11-20 初回の修正後比較は、既定スイート 10 実行体の 1,210 ケースのうち 1,209 件が通り、修正前と同じ 1 件がスキップされた。修正前に通ったケースの後退は 0 件。短縮 Run は 6 構成で 178,659 点がバイト一致した。
- 2026-11-20 初回の独立レビューは差し戻し（原文 `.scratch/prd192/review-20261120-0420.md`）。ffmpeg 異常検査をフレームの判定より先に行い、正常フレームの記録を動画の読み戻しで確認するようにした。QValuePanel の設定を設計文書に追記し、同期境界コメント、include、定数と実装計画を直した。
- 2026-11-20 再レビューは差し戻し（原文 `.scratch/prd192/review-20261120-0435.md`）。実装計画の章順、テストの無名名前空間・定数宣言、Atari テストの include 順を直した。
- 2026-11-20 開発者がエージェントによる実画面の確認結果を A14 の受入に採用した。
- 2026-11-20 3 回目の独立レビューは計画・実装ともに OK（原文 `.scratch/prd192/review-20261120-0439.md`）。章順、テストの宣言と include、A14 の受入記録、最終 Runner の 6 構成の一致を確認した。

## 検証結果

- Debug の Runner と受入対象の実行体 10 本を、`VsDevCmd.bat` で x64 を初期化した後に `cmake --build --preset x64-Debug --target` でビルドした。コマンドとログは `.scratch/prd192/debug-final-build.log` と各対象ログを参照。
- A1〜A8・A10〜A12。基盤の対象タグ 18 ケース、変更後の基盤実行体での Train・Learn 各 200 万回のストレステスト、CUDA を実行した DQN の対象タグ 2 ケース、GUI の対象タグ 7 ケースが通った。残りの Agent・Env の初期化関数 7 本も各 1 ケースが通った。部分登録の失敗を繰り返す隠しテスト `[init_failure]` は、マッチャの寿命を直して再ビルド後に 1 ケース通った。A11・A12 の指定検索はともに 0 件。
- A4 の再検証では、ffmpeg で動画を読み戻し、サイズ違いを挟んだタグに正常フレームが 2 枚、無効画像を続けたタグに正常フレームが 1 枚あると確認した。基盤の対象タグを再実行して 18 ケース、74 アサーションが通った（`.scratch/prd192/final-core-contracts-retry.log`）。
- A9。レビューで動作を直した後の既定スイートは 10 実行体がすべて終了値 0。1,209 ケース通過、修正前と同じ 1 ケースがスキップ、修正前に通ったケースの後退は 0（`.scratch/prd192/suites-final/results.json`、`.scratch/prd192/suites-compare-final.log`）。最後のテストコードの規約修正後に基盤・DQN・Atari を再ビルドし、対象の計 9 ケースが通った（`.scratch/prd192/test-style-build.log`、`.scratch/prd192/style-*.log`）。
- A13。ffmpeg の検査順を直した最終 Runner でも、GridMaze・LunarLander の Skip 無効・有効、Atari・DropMerge の Skip 無効の計 6 構成で、正規化した 178,659 点が修正前とバイト一致した（`.scratch/prd192/metrics-compare-final.log`）。
- A14。エージェントが Runner の実画面を操作し、開発者が結果を受入に採用した。LunarLander の CUDA 評価（`actor_approx`）に矢印キーを 10 回送り、表示が Step 10 に進んだ。学習を再開して Conv2d を 10 回開閉し、学習 step の増加を確認した。LunarLander の IQN では HeatMap の既定の選択が `taus` の不足で止まったため、GridMaze の通常 DQN で HeatMap を 10 回開閉し、学習継続と窓の正常終了を確認した。Run 名は `run_20261120-033611_prd192_ui` と `run_20261120-040149_prd192_heatmap`。修正前の Runner を同設定で起動した `run_20261120-041017_prd192_iqn_baseline` でも `taus` が不足する停止を再現した。
- A15。ADR 0082、PRD 指定の設計文書 5 本、Notifier の 7 種の操作、キャッシュ 3 か所の宣言、Runner の Observer 登録 8 か所を PRD と照合した。

### 実装上の補足

動画の書き込みが無効画像を検出したときにも元のタグを診断に使えるよう、`VideoLogger::WriteFrame` に `MetricsLogger` からタグを渡した。動画のファイル名に使う変換済みのタグで警告の回数を管理すると、同じ文字列に変換されるタグを区別できないためである。登録ヘルパーと動画書き込みの引数を置く公開ヘッダは `anet/init.hpp` と `anet/metrics_logger.hpp`。既存の画像記録の呼び口は従来どおり使える。

## 決まらなかったこと

PRD192 の設計判断は確定済み。PRD146 と最初の動画画像が無効な場合の扱いは PRD §9 の範囲による。

LunarLander の IQN で HeatMap を既定の `policy-net.forward` で開くと学習が止まる問題は、[issue-20261120-7ac5fa](../management/issues/issue-20261120-7ac5fa.md) に記録した。

QValuePanel の `hist_bins` に値域検査がない問題は、[issue-20261120-1979f6](../management/issues/issue-20261120-1979f6.md) に記録した。

## 完了報告

PRD192 の 5 件を修正し、A1〜A15 の受入を確認した。独立レビューは 3 回目で計画・実装がともに OK。対応した 5 課題の AI 判定は残件無しとした。範囲外で見つかった 2 件は開発者の選択を受けて起票した。
