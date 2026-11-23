# PRD193 重み初期化と設定読み取りの実装メモ

## 概要

[PRD193](0193_variance_scaling_weight_init_10rfd.md) に従い、fan_in で分散を決める重み初期化と、初期化設定の共通の読み口を実装する。既存方式は初期値と乱数消費順を保つ。変更前の比較基準は `90ee34b2` の作業ツリー。ステージ57の引数は現在の `apps/13_batch_run.bat` から採る。

## 主な変更

- `WeightInitConfig` に `variance_scale = 1.0` を追加し、`WeightInitializer` が `variance_scaling` で範囲 ±√(3 × scale / fan_in) の一様分布と bias 0 を適用する。fan_in は重みの入力軸とカーネル軸から求める。
- `variance_scale` の設定読み取りは有限正数を受け付ける。適用時の重みは出力軸と入力軸を持つ float16、bfloat16、float32、float64 を対象とする。空でない重みは、求めた半幅が要素型の最小正値から最大値の半分の範囲に収まることも検査する。抽選に渡す半幅は、指定の半幅以下で要素型が表現できる最大値とする。型、形状、幅の条件を満たさない場合は、倍率と条件を示して書き込み前に止める。適用時の検査と半幅の計算は `nn_head.cpp` に置く。複数の層を持つ block も、使う層すべての条件を最初の forward で登録前に確かめ、検査で発生した例外は後続の forward にも返す。
- `ConfigReader<WeightInitConfig>` の `Read` と `ReadFields` で D15 の順に8キーを読み、既存の記録と接頭辞を使う。block 9か所・DQN 2か所・Rainbow は読んだ直後に検証する。価値分解の Head と RND は読み取りと検証を分けて既存の順序を保つ。
- `ValidateWeightInitConfigMode` を `ValidateWeightInitConfigChoices` に改名し、初期化方式と活性化名の列挙を検証することを名前で示す。数値の検証順と診断は §8.1、直接指定の検証は D10 に従う。Agent と RND は完全キー、block は相対キーを示す。
- DQN と Rainbow の Head は `linear` を既定にし、skip は Head の全8値を既定にする。価値分解が無効なときの既存の通知に `nonlinearity` と `variance_scale` を加える。
- 重みの登録前検査の診断には block の相対キーを出す。公開 API の直接呼び出しでは `WeightInitConfig` を診断名に使う。
- `nn.txt` と `agent.txt`、設計文書130・200の日英版を更新し、翻訳元情報を照合する。

## テスト

公開の Factory、`WeightInitializer`、Agent・RND の設定コンストラクタを通す。各挙動で RED → GREEN を確認してから次へ進む。

1. 最初のテストは Linear block の設定から初回 forward まで通し、新しい分布と bias を観測する。Linear 256→512 と Conv2d 4→32・8×8 について、倍率 0.5773502691896258 と 1.0 の上限、標本分散、bias を §9.2 の許容幅で確かめる。
2. §8.1 の拒否条件、14か所で倍率0を拒否する時点と診断、価値分解の列挙と数値の検査順を確かめる。
3. DQN・Rainbow の Head の既定値と明示値、設定の記録、skip の継承と個別指定、価値分解を使わない場合の WARN の件数を確かめる。
4. 初期化指定を省いた各 block の初期値と RND の既定を §9.2 の11に沿って確かめる。ResBlock の shortcut を含み、block の各層は8,192要素以上とする。
5. Factory を使い、Linear、Conv1d、Conv2d、ResBlock の主経路と shortcut、CNBlock の最終 Conv で型に合わない倍率を適用する。登録前に相対キー付きの診断が出ることと、後続の forward で失敗を再送出することを確かめる。公開の `WeightInitializer` では rank 0・1 の重みを拒否し、float16 の非正規化数と bfloat16 の丸め境界で抽選範囲を守ることを確かめる。

## 検証

- 編集前に MSVC 環境で Release Runner と Debug DQN テストをビルドし、実行体と依存 DLL を `.scratch/prd193/before/` に保存する。旧・新の Run は同じマシン、device、seed、`backend.@deterministic` と引数で流す。LunarLander 100,000 exp step、Breakout ステージ57 30,000 exp step の正常終了と学習更新を確認し、PRD084 の `compare_metrics.py normalize` と `compare` で既定除外群以外の全 scalar を照合する。
- 旧・新の `[prd061-capture]` を異なる `ANET_PRD061_OUTPUT` に採取する。全対象入力の `values`・`typed` は、追加したキーだけ除いて比較する。
- Debug ビルド、core・DQN のテストと `[shipped_config]` を実行する。§5.2 の検索で初期化設定の個別読み取りが0行であることを確かめる。
- ステージ57に §7 の12指定を加え、30,000 exp step の `tmp` Run を流す。最初の `train_step=`、入力写しの12行、Agent の設定記録の方式と倍率、未読キー一覧を確認する。ログと比較結果は `.scratch/prd193/` に保存する。

## 前提

- ユーザー判断が必要な未決事項は0件。PRD194 の現状では block の診断は相対キーとする。5M の実験の段取りと結果の読みは Atari 側へ渡す。
- 倍率の既定値は `1.0`。DQN と Rainbow の Head の活性化名の既定は `linear`。DQN の skip は DQN の Head の初期化設定全8値を継承する。
- 対象課題は 533820 と d4feb6。自己点検と独立レビューを実施し、各課題に対応内容・検証結果を記録する。受入の結果に基づき判定を記録する。

## 経過

- 2026-11-21 実装計画を保存。
- 2026-11-21 対象2件を対応中に変更し、課題一覧と関連3カテゴリの課題区画を再生成。
- 2026-11-21 設計文書130・200の日英でPRD193の該当箇所を同期した。両英語版の既存の翻訳元blobより日本語版が先行しているため、文書全体が最新訳であることを示すblobの更新は既存差分の翻訳時に行う。
- 2026-11-21 変更前の Release Runner と Debug DQN テストをビルドして保存した。Runner は LunarLander 100,000 exp step と Breakout ステージ57 30,000 exp step を正常終了し、正規化済み scalar はそれぞれ39,054点と17,578点だった。
- 2026-11-21 固定 manifest の `[prd061-capture]` は変更前から `run.@nature_dqn` が現行設定に無いため停止する。作業用の manifest で廃止された Atari の2構成を現行の Nature・rainbow_like 構成に差し替え、RainbowAgent の入力を追加した。同じテスト実行体で変更前の14入力を採取できた。固定 manifest は履歴資料として保持する。
- 2026-11-21 PRD072 の既定キー検査は旧ディレクトリ `core` を参照して停止する。作業用ラッパーで現行の `src/core` に読み替えると、変更前から `apps/tools/metrics_viewer.py` の4件だけが指摘された。
- 2026-11-21 Linear の設定と初回 forward を通す最初のテストは、実装前に未知の方式で失敗した。読み口と初期化方式を実装したあと、同じテストの3断定が通った（`red-linear.log`・`green-linear.log`）。
- 2026-11-21 Block Factory のテストで、移行前の倍率の読み落としと構築時の診断漏れを再現した。移行後は `[nn][init]` の8件・104断定が通った（`red-blocks.log`・`green-blocks.log`）。DQN・Rainbow と RND も読み口の切り替え前に追加テストで失敗を再現した（`red-heads.log`）。
- 2026-11-21 レビュー1回目は計画が OK、実装は2件差し戻し。Conv1D の方式一覧を更新した。Float32 の重みで極端な倍率を使ったときの下側の丸めと上側の汎用例外をテストで再現し、要素型に応じた初期化時の検査を追加した（`review-20261121-1250.md`・`red-dtype-bound.log`）。
- 2026-11-21 レビュー2回目は1回目の2件が解消し、計画は OK、実装は新たに2件差し戻し。複素重みを初期化してしまうことを再現し、対応型を実浮動小数点4種類に揃えた。関数名を `InitializeVarianceScaling` に変更した。活性化名の防御的な診断から制御が戻ると MSVC が推定する警告 C4715 も修正した（`review-20261121-1258.md`・`red-complex-weight.log`）。
- 2026-11-21 レビュー3回目で計画・実装の両方が OK（`review-20261121-1306.md`）。
- 2026-11-21 Codex と Claude の再確認で、block の初期化失敗後の状態、入力軸が無い重み、低精度型の抽選範囲、識別子の定数化、計画の前提・改名、設定を読む順序のコメントに指摘を受けた。修正前のテストでは5種類の block で子層が登録され、rank 0・1 で期待した診断が出ず、Float16 と BFloat16 で抽選範囲を超えた（`cc-red-atomic.log`・`cc-red-bounds.log`）。block の層サイズと入力型が揃う初回 forward で、同じ半幅計算を使って登録前に検証する。抽選の端点は要素型が内側で表現できる値に揃え、範囲の保証を日英の設計文書に記した。

- 2026-11-21 二モデルの再レビューで、Codex は登録前検査が対象層の設定を示さない点、Claude は DQN 全テストと設定再採取の結果が計画に無い点を指摘した（`review-20261121-1351.md`・`review-20261121-1346-claude.md`）。相対キーの期待を加えたテストで診断の不一致を再現した（`cc-red-diagnostic.log`）。実際に使う層の設定接頭辞を検査に渡し、同じ形状の ResBlock の2層でも診断を区別できるようにした。

## 検証結果

- Debug で `anet-core-base-test` と `anet-algo-dqn-test` をビルドした。core の初期化・RND 設定テスト12件・132断定、DQN の Head 設定テスト8件・71断定、`[shipped_config]` 9件・954断定がすべて通った。分布、14か所の診断、継承、記録順、WARN、block と RND の既定を含む。
- §9.2 の5の検索で個別読み取りは94行から0行になった。改名前の `ValidateWeightInitConfigMode` は現用範囲で0件。設定ファイルの追加2キーは作業用の既定キー検査を通り、既存の指摘4件は前後で同じだった。
- §9.2 の10で指定された固定 manifest は変更前後とも廃止されたプロファイルで中断した。作業用の現行構成14入力は、追加した初期化設定を除き `typed` と `values` がすべて一致した。採取先は `.scratch/prd193/before/capture-current` と `.scratch/prd193/after/capture-current`。
- レビュー1回目の実行体で、Run の scalar は従来方式の LunarLander 39,054点、Breakout 17,578点がそれぞれ前後で完全一致した。新方式の Breakout も正常終了し、初回の学習ログ、入力写し12行、Head の記録2行、未読キー一覧を確認した（`compare.log`・`after-runs.log`）。
- レビュー1回目の通常テストは core 647件合格・CUDA利用可能時の1件スキップ、DQN 364件・161,171断定が合格した（`core-suite.log`・`dqn-suite.log`）。
- 差し戻し対応後の Debug core で、初期化・RND・出荷設定テスト22件・1,092断定が通った。Float32 で `1e-100` と `1e80` を拒否し、診断に倍率、型、`fan_in`、許容半幅を示すことと、重み・bias がそのままであることを確かめた（`green-dtype-bound.log`）。
- 半幅検査を追加した実行体でも、従来方式の2 Run の完全一致、現行14入力の設定一致、Dopamine 倍率の Breakout の正常終了と12行の配線を再確認した（`reviewed-compare.log`・`reviewed-runs.log`）。
- レビュー2回目の core 通常テストでは、初期化を使わない `TestAgent` の評価セッションテスト1件で終了ログが3件となり、6断定が失敗した（`reviewed-core-suite.log`）。合格647件、CUDA利用可能時のスキップ1件。
- 最後の Debug ビルドで初期化・RND・出荷設定23件・1,095断定とDQN設定8件・71断定が通った。評価セッションテストは単独の再実行3回とも通った。C4715 は修正前の `real-dtype-debug-build.log` にあり、修正後の `final-debug-build.log` と `real-dtype-release-build.log` に無い。
- 最後の core 通常テストは649件・33,692断定が合格した。CUDA利用可能時の1件はスキップ。評価セッションのログの失敗はこの再実行では出なかった（`final-core-suite.log`）。
- 最後の Release Runner で新方式の Breakout を30,000 exp stepまで実行し、終了コード0、最初の学習ログ、入力写し12行、Head の方式と倍率、未読キー一覧を確認した（`final-smoke.log`、Run は `.scratch/prd193/after-final/breakout_vs/runs/`）。
- 二モデルの指摘を修正した後、Debug の core・DQN と Release Runner を再ビルドした。PRD193 の回帰テストは、失敗後に子層やパラメータを公開しないことと同じ例外を返すこと、rank 0・1 で重みと bias を変えずに止まること、CPU・CUDA の両方で Float16・BFloat16 の抽選範囲を守ることを確かめた。core の対象25件・1,142断定と DQN の対象8件・71断定に合格した（`cc-core-contract-scoped.log`・`cc-dqn-contract.log`）。
- core 通常テストは651件合格・1件スキップ・1件失敗。失敗した `Main View common default is Eval` は、基準 `3805a114` で `eval` だった `common.txt` の並行差分が `train` を指定したため。対象25件の再実行ではこの1件を除いた（`cc-core-suite.log`・`cc-core-contract.log`）。
- 従来方式の LunarLander と Breakout は、修正後の実行体でも比較対象 scalar が39,054点・17,578点すべて一致した。新方式の Breakout は30,000 exp stepで正常終了し、最初の学習ログ、入力写し12行、Head の方式と倍率、未読キー一覧を確認した（`cc-runs.log`）。
- 対象18ファイルの BOM の保持と、実装計画を含む19ファイルが LF であることを確かめた。旧関数名と個別キー読み取りは現用範囲で0件。差分の空白検査も通過した。
- DQN の通常テストを、登録前検査と端点処理を直した版で再実行し、364件・161,171断定に合格した（`cc-dqn-suite.log`）。
- 作業ツリーの構成14入力の採取は成功し、照合は並行差分の `app.main_view` により不一致になった（`cc-capture.log`・`cc-compare.log`）。比較用コピーだけに基準の `common.txt` を配置して再採取すると、14入力の `values` と `typed` は PRD193 の追加キーを除きすべて一致した（`cc-reference-capture.log`・`cc-reference-compare.log`）。固定 manifest は引き続き廃止された構成で停止した。
- 相対キーの診断を加えた最終版で Debug の core・DQN を再ビルドし、core 対象25件・1,154断定、DQN 対象8件・71断定に合格した（`cc-final-core-contract.log`・`cc-final-dqn-contract.log`）。最終版の差分は診断名とそのテスト・説明の追加で、通常テスト全件および従来方式の学習結果の比較は、一つ前の版の結果を使った。
- 最終版の Release Runner をビルドした。新方式の Breakout は30,000 exp stepで正常終了し、最初の学習ログ、入力写し12行、Head の方式と倍率、未読キー一覧を確認した（`cc-final-release-build.log`・`cc-final-smoke.log`）。

### 実行したコマンド

リポジトリのルートで実行した。以下のログの保存先は `.scratch/prd193/`。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-base-test anet-algo-dqn-test --parallel 6'
```

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner --parallel 3'
```

ビルドの stdout と stderr は `after-debug-build.log`・`after-release-build.log` に保存した。変更前は同じ MSVC 初期化でそれぞれ `--target anet-algo-dqn-test --parallel 4` と `--target AnetRLRunner --parallel 4` を使った。

| 確認 | 実行 | ログ |
|---|---|---|
| 初期化と RND | `.\out\test\Debug\anet-core-base-test.exe '[nn][init],[prd116][intrinsic][config]'` | `init-contract.log` |
| DQN Head と検査順 | `.\out\test\Debug\anet-algo-dqn-test.exe '[prd193],[prd121-review-config],[review-fix-rainbow]'` | `dqn-head-contract.log` |
| 出荷設定 | `.\out\test\Debug\anet-core-base-test.exe '[shipped_config]'` | `shipped-config.log` |
| 半幅・型の変更後の core | `.\out\test\Debug\anet-core-base-test.exe '[nn][init],[prd116][intrinsic][config],[shipped_config]'` | `green-dtype-bound.log`・`real-dtype-contract.log` |
| 最後の Debug 向け確認 | 上の core と DQN の対象フィルタで再実行。ビルドは同じターゲットで `--parallel 4` | `final-debug-build.log`・`final-core-contract.log`・`final-dqn-contract.log` |
| 評価セッションテストの単独再実行 | `.\out\test\Debug\anet-core-base-test.exe 'EvalRunner RunSession emits adopted episodes then one session event'` を3回 | `eval-session-probe.log` |
| 通常のテスト | `.\out\test\Debug\anet-core-base-test.exe` と `.\out\test\Debug\anet-algo-dqn-test.exe` | `core-suite.log`・`dqn-suite.log` |
| 前後の設定採取 | `.\.venv\Scripts\python.exe .scratch/prd193/verify.py capture-before` と `capture-after` | `before-capture-current.log`・`after-capture.log`、各採取先の `capture*.execution.json` と stdout・stderr |
| Runner | `.\.venv\Scripts\python.exe .scratch/prd193/verify.py before` と `after` | `before-runs.log`・`after-runs.log`、各 Run の `runner.execution.json` と stdout・stderr |
| 差し戻し対応後の Run | `.\.venv\Scripts\python.exe .scratch/prd193/verify.py after --post-dir after-review` と `compare --post-dir after-review` | `reviewed-runs.log`・`reviewed-compare.log` |
| 実浮動小数点に揃えた後のスモーク | `.\.venv\Scripts\python.exe .scratch/prd193/verify.py smoke --post-dir after-final` | `final-smoke.log` と Run の `runner.execution.json`・stdout・stderr |
| 既定キー | `.\.venv\Scripts\python.exe .scratch/prd193/check_default_leaves.py before` と `after` | `before-default-leaves.log`・`after-default-leaves.log` |
| 二モデルの指摘を修正した後のビルド | 上記の Debug 2ターゲットを `--parallel 4`、Release Runner を `--parallel 3` | `cc-debug-build.log`・`cc-release-build.log` |
| 二モデルの指摘を修正した後のテスト | 上記の対象フィルタと、引数なしの core・DQN 全テスト。追加で core のフィルタ `[nn][init],[prd116][intrinsic][config],[shipped_config]~"Main View common default is Eval"` | `cc-core-contract.log`・`cc-core-contract-scoped.log`・`cc-dqn-contract.log`・`cc-core-suite.log`・`cc-dqn-suite.log` |
| 二モデルの指摘を修正した後の Run と設定比較 | `.\.venv\Scripts\python.exe .scratch/prd193/verify.py after --post-dir after-cc`、`capture-after --post-dir after-cc`、`compare --post-dir after-cc` | `cc-runs.log`・`cc-capture.log`・`cc-compare.log` |
| 比較用コピーでの設定再採取 | `.\.venv\Scripts\python.exe .scratch/prd193/verify.py capture-after --post-dir after-cc --reference-common` と同じ引数を付けた `compare` | `cc-reference-capture.log`・`cc-reference-compare.log` |
| 相対キーの診断を加えた最終版 | 上記の MSVC 初期化と Debug 各ターゲット（`--parallel 4`）、Release Runner（`--parallel 3`）、core・DQN の対象フィルタ。スモークは `.\.venv\Scripts\python.exe .scratch/prd193/verify.py smoke --post-dir after-cc-final` | `cc-final-debug-build.log`・`cc-final-dqn-build.log`・`cc-final-release-build.log`・`cc-final-core-contract.log`・`cc-final-dqn-contract.log`・`cc-final-smoke.log` |

## 完了報告

- §9.2 の1〜9・11を確認した。§9.2 の10は固定 manifest の廃止済み構成で止まるため、現行構成14入力を同じ共通設定で前後比較した。
- Codex と Claude の最終再レビューで、実装計画と実装の両方が OK（`review-20261121-1404.md`・`review-20261121-1400-claude.md`）。
- 対象課題 533820・d4feb6 の対応と確かめを記録し、AI判定を残件無しとした。
- 固定 manifest の現用採取への流用と、既定キー検査の旧パスおよび Viewer の偽陽性は検証資材に残る。5M の実験設計と結果の読みは PRD §9.1 の材料を Atari 側へ渡す。
