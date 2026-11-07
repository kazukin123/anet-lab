# RND の採点と補助学習を独立させる実装メモ

## 概要

[PRD116](116_rnd_lifelong_novelty_10prd.md) §5.1・§5.3・§6.2 の2026-11-06夜の改訂と、ユーザー指定の `.scratch/prd116/request-20261106-interface.md` に対応する。基準コミットは `08bf1e2e62a16cafa2e44c5ed4ae70015d87acf2`。初回実装の経過は [20impl](116_rnd_lifelong_novelty_20impl.md) に保持する。

## 主な変更

- `include/anet/intrinsic.hpp`、`src/intrinsic_rnd.hpp/.cpp`、`src/default_dqn_agent.cpp`: `Multiplier` を `Estimate`、`Update` を `Learn` へ改名する。採点・統計更新・補助学習の意味を明確にし、計算グラフの持ち越しを削除する。
- `Learn` は渡された観測からlaneを抽選し、現在の観測統計で正規化して、選択分だけforward・backward・AdamWの1stepを行う。統計更新は `Estimate` だけが行う。非有限誤差は学習時もoptimizerを進める前に拒否する。
- 入力deviceと計算deviceを分け、倍率は入力deviceへ返す。既存CPU正規化・統計を利用し、CUDA入力は内部でCPUへ移す。Learnは選択後の観測だけCPUへ移し、正規化後に計算deviceへ送る。
- `src/intrinsic_test.cpp`: 操作の独立、入力に従うloss、入力deviceへの返却、呼び出し間のSave/Loadを公開APIで検証する。
- `docs/design/212_rnd_lifelong_novelty.jp.md`: §3.4〜3.8・§4.4末尾・§4.5・§5.1〜5.2を追随させ、§3.6をsequenceDiagramへ変更する。§5.4と `core/anet-core/testdata/prd116/README.md` の案内に、初回実装20implとAPI見直し21implを区別して載せる。200日英は旧操作名が無いことを検索する。
- 初回実装記録へ続編への案内を追記し、索引 `issue-20261106-16b7af` の対応内容へ結果を1行追加する。PRD本文は別枠の変更を保持する。

## 未決事項監査と前提

- ユーザー判断が必要なブロッカー: 0。公開名・学習の入力・device・保存の時点は改訂PRDで確定済み。
- `Estimate` は既存の `InitialPriorityEstimator::Estimate`（`src/dqn_based_agent.cpp`）と同じく入力から評価値を推定して返す意味で使う。`Learn` は既存の `Update`（統計更新）や `Train`（モード・用途）の多義性を避けるため、改訂PRDが選んだ補助学習の動詞である。
- facadeは `RewardScaler → Estimate → 報酬合成 → Learn → 内側のLearner` を維持する。依頼書の図の列挙はLearnerが先だが、「facadeの呼び順を変えない」とPRD§5.3を優先し、図も実装と一致させる。
- `warmup_steps=0` では初回Learn単独から学習する。正のwarm-upは従来どおり最初の指定回数のEstimate直後まで学習を休止する。Learnは採点回数を進めない。
- 設定名と既定値、無効時経路、保存形式、targetの固定は変えない。Learnerへの移設はPRD924の範囲。
- 性能は測定しない。追加転送の概算: B=64、float画像84×84ならEstimateの入力約1.72MiB、Learnはfraction=.25で期待約0.43MiB。CPU入力・CUDA計算のLearnで約0.03〜0.15msのH2Dと選択誤差D2H約0.01〜0.10msが1 batch stepに最大1回ずつ加わる想定。CUDA入力ではさらにEstimateの全入力D2H、Learnの選択入力D2H、選択indexとαのH2Dが加わる。合計概算0.2〜0.7ms/batch（forward/backward・CPU正規化を除く、共有GPU負荷で変動）。転送はlaneループ外でまとめ、計測区間を残す。単価は未測定の仮定であり性能保証ではない。
- 観測正規化の二重計算に対する統計更新専用APIの要否は、既存の合意どおり測定後に判断する。

## テスト

Public interface: Factory、Estimate、Learn、Diagnostics、Save/Load、facadeのUpdateFromBatch。

TDDは1項目ずつRED→GREENで進める。

1. warm-up無しでLearn単独が引数の観測から有限lossを得て学習する。既存APIで失敗を確認してから独立化・改名する。
2. Estimateを連続しても学習に依存しない。異なる観測でLearnしたlossが、その時点の統計を使う独立した採点結果と一致し、Learnは統計を更新しない。
3. CPU/CUDA入力とCPU/CUDA計算の組合せでαのdevice・値を比較する（CUDA無しはskip）。
4. 初回・Estimate直後・Learn直後にSave/Loadを挟んだ継続結果が一致する。既存のtarget固定100回・warm-up・byte観測・facadeテストも継続する。

## 検証

リポジトリルートのPowerShellで、用途に応じて全体ビルドかTDD用ビルドを選び、成功後に関連テストを実行する。前工程で実行を確認したビルド・テストに、再実行用のログ保存先を付けて示す。必要な行をコピーして貼り付ける。

```powershell
# Debug全体ビルド
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug -j 2' > .scratch/prd116/interface-recheck-build.log 2>&1
# TDD中のテスト用ビルド
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test -j 2' > .scratch/prd116/interface-recheck-target-build.log 2>&1
# 関連テスト
core/anet-core/bin/Debug/anet-core-test.exe '[prd116],[scaler]' > .scratch/prd116/interface-recheck-test.log 2>&1
```

- 各実行の標準出力・標準エラーを上記ログへ保存し、終了後の `$LASTEXITCODE` と末尾の要約のみ返す。実施済みの保存先は「経過と結果」「最終確認とレビュー」に記録する。
- 無効時4環境checksumはMZ Runと競合するため保留。smokeは不要。実行中Runの比較には既存15:56版exeを使うため、その保存済み実行体を変更しない。
- 旧名・pending状態の現用参照、UTF-8/LF/BOM、差分空白、文書リンクを検査し、全品質項目で自己点検後に独立レビューを行う。

## 経過と結果

- 2026-11-06: 指定依頼・改訂PRD・既存実装とテストを照合。実装・検証はこれから。
- TDD1 RED: Debugテストビルド成功後、`[rnd_independent]` が `nanf == Approx(4.4697461128)` で失敗（終了42）。採点なしのUpdateが引数を使わず休止していたことを確認。ログ: `.scratch/prd116/interface-red-build.log`、`interface-red-test.log`。
- 採点からグラフ保持を除去し、Learnの引数から選択・forward・backwardを行う形へ変更。公開API・facade・全既存テストをEstimate/Learnへ移行。Debugテストビルド成功（`interface-green-build.log`）。212の図・保存時点・学習時の統計・費用を追随。
- TDD1 GREEN: `[prd116],[scaler]` 全18 tests /580 assertions成功（`interface-green-test.log`）。
- TDD device RED: CPU/CUDA計算の両方でCUDA入力に対してCPUのαが返り、2 assertions失敗（終了42、`interface-device-red-test.log`）。入力をCPUへ集約し、αを入力deviceへ戻す処理を追加。
- TDD device GREEN: CPU/CUDA計算の各々でCPU/CUDA入力を4回ずつ採点・学習し、入力device・αの一致・勾配なし・loss一致を全40 assertionsで確認（`interface-device-green-test.log`、終了0）。
- 独立化で成立した保存境界の回帰を追加: 初回前・Estimate直後・Learn直後をCPU/CUDAで生成し、復元後はEstimateの前にLearnを呼んで継続を比較する。独立学習には採点と異なる3laneの観測を渡し、lossと観測正規化・誤差統計の不変を確かめる。これらは独立化後の追加確認であり、別のREDを得たとは扱わない。
- Debug全体ビルド成功。非有限入力をLearnからも拒否する確認を加え、`[prd116],[scaler]` 全21 tests /998 assertions成功（終了0）。ログ: `.scratch/prd116/interface-verified-build.log`、`interface-verified-test.log`。CUDAは利用可能であり、deviceと保存境界のCUDA分も実行した。

## 自己点検と受入条件の照合

- `docs/quality/impl.md` の全項目を適用範囲に従って確認した。公開APIの改名、旧状態の除去、入力device、統計と勾配の分離、選択laneだけの学習、保存境界をコードとテストに照合した。新しい汎用機構・クラス・設定・借用・UIロック・GetScalar変更・skill変更はなく、これら固有の観点は対象外。
- 操作の独立: warm-up 0でLearn単独のlossと反復学習を確認。Estimateを2回続けた後に別の3lane入力でLearnし、独立した同seed推定器が算出する当該入力の誤差とlossを比較する。観測正規化は公開checkpointから復元した正規化結果を比較し、誤差の平均・標準偏差・直近値も不変を確認する。
- device: CPU/CUDA計算ごとにCPU/CUDA入力を4回ずつ比較し、αのdevice・値・勾配なし・lossを確認。保存: CPU/CUDAそれぞれで初回前・Estimate直後・Learn直後の3境界から12回継続し、復元後の最初をLearnにして欠落を検出する。
- 既存の固定教師100回・warm-up・byte観測・不正shape・facade経路の確認も同じスイートに含む。非有限入力はEstimateとLearnの双方で拒否し、未成立のlossをNaNのままとする。無効時経路はfacade差分が有効分岐内の2操作名の変更だけであることを確認したが、4環境checksumの再測定は未実施。
- 計測範囲・日本語コメント・英語エラー・集成体の指示付き初期化・リテラルを確認した。同期転送はlaneループの外にまとめ、費用の未測定の仮定を「前提」へ記録した。
- `.scratch/prd116/verify_interface.py` は品質観点のBOM・文字列検査コマンドを実行し、UTF-8/LF、文書のリンク・fence、`git diff --check` も確認する。重複候補の `rnd.warmup_steps`・`rnd.train_fraction` は定数化対象外の設定キー。現用コード・設定・日英設計文書・CONTEXT・wikiの地図に旧操作名とpending状態が残っていないことを検索した。
- 標準出力と標準エラーは各ログへまとめ、50秒程度の待機と終了時の短い要約で確認した。全体ビルドの実行はPowerShellから `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug -j 2'`、テストは `core/anet-core/bin/Debug/anet-core-test.exe '[prd116],[scaler]'`。
- PRDの入力blobは `7f60c39ff28d57a7cb4400b3a23bd08a9c16d52f` のまま保持した。212日本語版だけを更新し、200日英は旧操作名が無く変更不要と確認した。計画の変更対象と実差分を照合し、索引issueへ対応を追記した。
- 未実施: MZ Runと競合する無効時4環境checksum再測定、ユーザー指示による性能測定。smokeは今回の依頼で不要。これらを今回の成功結果へ含めない。

## 最終確認とレビュー

- テスト補助のStateSpecを指示付き初期化へ整理した後も、Debug全体ビルドと21 tests /998 assertionsが成功（終了0）。最終ログ: `.scratch/prd116/interface-review-build.log`、`interface-review-test.log`。
- 独立した `anet-review-agent` の1往復で、計画・実装ともOK。修正要求・範囲外の検出・開発者への確認事項はなし。原文: `.scratch/prd116/review-20261106-2339.md`。
- レビューでは、facade順序、Learn単独と別入力、統計の不変、CPU/CUDA入力、操作間の保存、warm-up境界を重点確認した。checksum再測定と性能測定は成功結果に含めない。根拠が無く見送った指摘はない。
- 索引issueの判定は実装済み、判定日は2026-11-06のまま。今回対象となる子issueはない。INDEXと全16地図の課題区画を再計算し、現行内容との一致を確認したため、書き戻しは不要だった。全mapsの再計算で出る別作業のファイル一覧等の差分は保持した。

## 完了報告

- 実施内容: Estimate/Learnの独立化、任意device入力、操作間の保存再開、受入テスト、212の説明と図の更新。
- 変更ファイル: コード・テスト5ファイル、212文書、検証資材README、20implの案内、本21impl、索引issueの計10ファイル。
- レビュー結果: 初回の独立レビューは計画・実装ともOK（1往復）。最終Debug全体ビルド・関連21 tests /998 assertions成功。その後の2モデルレビューと対応は次節に記録する。
- 決まらなかったこと: なし。未実施の検証は上記理由で保留し、実施済みと区別した。
- 判断してほしいこと: なし。
- 次にできること: MZ Runの切れ目で無効時4環境checksumを再確認し、性能測定可能になったら転送・再forward・二重正規化の内訳を確認する。

## 2モデルレビュー後の文書修正

- 2026-11-06: `cc-review` の初回はCodexが計画・実装ともOK、Claudeが計画の「検証」節のコマンド不足で差し戻し、実装はOK。Claudeは範囲外として、212 §5.4と検証資材READMEが20implだけを案内する点も報告した。原文: `.scratch/prd116/review-20261106-2344.md`、`review-20261106-2344-claude.md`。
- ユーザーが指摘と範囲外のリンク2箇所の修正・再レビューを指定。「検証」節へMSVC初期化を含む全体ビルド・テスト用ビルド・テスト実行のコマンドをまとめ、2箇所の案内は20implと21implの役割を区別した。主な変更と変更ファイル数にもREADMEを反映した。
- コード・設定の変更はない。`.scratch/prd116/verify_interface.py` にREADMEを加えて実行し、文書の相対リンク、コードブロック、UTF-8/LF、既存9ファイルのBOM保持（新規1ファイルは比較外）、差分空白を確認した。PowerShell Parserによるコマンドの構文検査と、実行ファイル・MSVC初期化のパスの存在確認も成功。実装品質観点の全項目を適用範囲に従って自己点検し、コード・設定等を変えていない項目は対象外とした。ビルド・テストは前工程の成功結果を引き継ぎ、今回は再実行していない。
- 再レビューは、同じCodexレビューエージェントとClaudeのセッションへ、上記修正をまとめて依頼する。
- 2026-11-07: 再レビューでCodex・Claudeとも計画・実装がOK。コマンド不足とリンク2箇所の解消を双方が確認し、修正要求・範囲外の検出・開発者への確認事項はなし。2モデルレビューは今回で2往復。原文: `.scratch/prd116/review-20261106-2356.md`、`review-20261106-2356-claude.md`。ClaudeのAPI報告額は約2.75 USD。Claudeはコマンド構文・BOM等の機械検査を直接実行せず、親側の静的検査記録を参照した。ビルド・テストの成功は前工程の記録を引き継ぐ。
