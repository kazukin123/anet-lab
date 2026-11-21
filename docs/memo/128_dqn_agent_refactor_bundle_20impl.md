# PRD128 DQN 系 Agent リファクタリング実装メモ

## 概要

[PRD128](128_dqn_agent_refactor_bundle_10prd.md) の確定事項 D1〜D24 と受入 1〜14 に従い、5 件をステージ 0〜5 の順に実装する。PRD129 は a3447045 で実装済みなので、その後のコードを全ステージの比較基準とする。設定キー、正常入力の計算、乱数列、保存形式を維持し、意図した変更は PRD §6.2 に限定する。

## 主な変更

1. ステージ 0: 変更前の Debug・Release と DLL、設定、テスト実行体を `.scratch/prd128/` に保存する。`core/anet-core/testdata/prd128/` に PRD125 の道具を呼ぶ検証入口と README を作る。FP32 31 組、RND 2 組、全診断購読を凍結し、scalar・設定記録・WARN/ERROR を比較する。既存テストの被覆を確認し、不足する特性テストを追加する。
2. ステージ 1（288879）: `rl.hpp` の ExperienceSamples 内にフィールド名・転送先を渡す非公開の共通走査を置き、`To`、`ForEachTensor`、`ToString` で使う。公開関数と aggregate 初期化を維持する。TensorDict も同じ列挙へ載せる。CPU metadata 2 件だけ CPU へ送り、未定義 tensor はそのまま返す。不要な stream guard を除く。
3. ステージ 2（e33797）: `default_dqn_agent.hpp` の設定読み取りを関心ごとの private 関数へ分け、実行順を維持する。語・数値の検証と接頭辞下の記録を集約する。方策と τ の検証は `agent.hpp` へ移す。エラー文は D6 の形へそろえる。
4. ステージ 3（08d557）: `IntrinsicRewardCoordinator` を DefaultDQN の実装グループへ置き、7 メンバ、採点、補助学習、診断、保存・読み込み、Trace の写しと採点を移す。共通のボーナス式を学習と Trace から呼ぶ。PRD129 の構築時・学習停止中の β 評価も移す。Agent の排他区間と保存順を維持する。
5. ステージ 4（24d885）: `dqn_based_agent.hpp/.cpp` に DiagnosticDemandTable と共通 scalar 読み出しを置く。購読設定は振り分け・検証・資源確保へ分ける。可塑性の 3 チャネルと attention の層別要求は U1 で適合を調べ、合わない部分は固有構造を維持する。指標は enum 等の名前とキーの対応を constexpr で検証し、並び替えで static_assert が失敗する形にする。Agent の基の seed から Learner が probe 用乱数 5 個を作り、購読再設定で作り直さない。直接 Learner を作るテストも移す。
6. ステージ 5（72d640）: DqnComponentAssembler を `dqn_based_agent.hpp/.cpp` に置き、Head、方策、Actor、Learner の生成と scalar の振り分けを共通化する。具象 Agent の Config は受け取らず、任意の部品を任意の入力で受ける。Rainbow の制限は Config の検証へ置き、読み取り時と DTO を使う構築時に呼ぶ。
7. 設計文書 `010`・`110`・`150`・`200`・`212` の jp と en を追随させる。200 から新しい検証 README へ参照する。対象課題の対応内容・判定を更新し、派生一覧を再生成する。他の作業による差分は保持する。
8. 追加の2モデルレビューでは、PRDのU2の記録をそろえ、検証コマンドを計画へ補い、検証のメッセージと直接構築テストの入力を既存/名前付きの定数へ寄せ、切り出したコメントの意図と所有者を正す。現在の値・文言・seedは変えない。
9. 開発者が追加で選択した2件として、readbackのMunchausen診断サイズを既存のCountへそろえ、200英語版§9.1の重複文を除く。PRD128の契約は変えず、選択された小さな修正として検証と同じ2モデルレビューへ含める。

## テスト

- Public interface / surface: ExperienceSamples の公開関数、ConfigData と DTO からの Agent 構築、Agent の UpdateFromBatch・GetScalar・Save/Load・Trace、Learner の購読と学習結果、Runner の成果物。
- 優先 behavior: 全フィールドの転送と走査、未定義保持、内的報酬の保存往復と Trace 一致、Rainbow の固定 seed 学習と不正方策検出、6 Head/3 Learner の選択、診断購読の発火間隔、学習停止中の診断の既知/未知/NaN と WARN 回数。
- TDD 順序: ステージ 0 の特性テストは現状の期待値で通し、各リファクタリングの直前にも確認する。意図して変える挙動は 1 テストの RED を確認してから最小実装で GREEN にし、次の挙動へ進む。最初は全フィールドを入れた ExperienceSamples の CPU 転送から走査・文字列化までを通す。次に未定義保持を RED→GREEN で確認する。Rainbow の方策制限・τ 検証も個別に RED→GREEN とする。
- 既存の内的報酬 checkpoint と PRD127/129 の Trace テストは期待値を変えず再利用する。probe の直接注入テストは基の seed を渡す形へ移し、同じ named seed の列を維持する。U1/U2/U3 の結論と追加の構成は検証記録に残す。
- 並び替えの破壊検査は共通型に載る群と載らない群をそれぞれ選び、隣接 2 指標を交換するとビルドが失敗し、戻すと成功することを確認する。

## 検証

- ビルドは AGENTS.md の VsDevCmd.bat 経由で Debug と Release を使用する。
- ステージ 0 で基準の既定 suite と各構成を採り、各ステージ後に同じ凍結入力で PRD §7 の 1〜3 を検査する。時間・資源の 3 prefix 以外は全 scalar を比較し、NaN 同士だけを等価にする。
- ステージ 1・3・4 は PRD125 の performance と同条件で、捨て Run と交互 3 巡以上を実行する。ステージ 3 は RND を重ねる。他の実行との競合を確認してから開始する。
- RND の LunarLander checkpoint から基準と変更後をそれぞれ 8,704 experience 続け、scalar を比較する。
- 最後に既定 core suite、PRD の検索検査、全品質確認観点の自己点検、独立コンテキストの実装計画・実装レビューを行う。PRD121 の足す先を関数名で表に残す。
- 実際の入口、終了値、ログ、比較結果、未実施と理由は以下へ追記する。

### 検証コマンド（PowerShell、リポジトリルート）

ビルドとテストの入口は次のとおり。通常ログもファイルへ保存し、終了値とXML/summaryだけを要約する。コードの枠はAGENTS.mdの指定により `bash` とし、実行シェルはPowerShellとする。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug > .scratch\prd128\cc-fix-debug-build.log 2>&1'
```

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release > .scratch\prd128\cc-fix-release-build.log 2>&1'
```

```bash
.\core\anet-core\bin\Debug\anet-core-test.exe '[prd128],[dqn][config],[dqn][scalar],[dqn][munchausen],[dqn][replay_fit]' --reporter junit --out .scratch/prd128/cc-fix-related.xml > .scratch/prd128/cc-fix-related.log 2>&1
```

既定suiteは比較用の基準suiteを保存したあとに実行する。次の未使用labelは日時から作り、過去の結果を上書きしない。

```bash
$prd128Label = 'suite-' + (Get-Date -Format yyyyMMdd-HHmmss); .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py suite --label $prd128Label > ".scratch/prd128/$prd128Label.log" 2>&1
```

基準の採取は本体を変更する前のステージ0だけで行う。`snapshot` は基準ディレクトリがある場合は実行せず、`prepare` も凍結済み入力がある場合は実行しない。変更後の実行体で基準を採り直す手順としては使わない。

```bash
if (-not (Test-Path -LiteralPath .scratch/prd128/acceptance/baseline)) { .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py snapshot > .scratch/prd128/snapshot.log 2>&1 } else { Write-Output 'PRD128 baseline already exists; preserved.' }
```

```bash
if (-not (Test-Path -LiteralPath .scratch/prd128/acceptance/inputs/cases.json)) { .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py prepare > .scratch/prd128/prepare.log 2>&1 } else { Write-Output 'PRD128 frozen inputs already exist; preserved.' }
```

保存した変更前の実行体による反復、変更後との比較、旧checkpointの継続の入口は次のとおり。CPUは最大4並列、CUDAは道具の中で逐次実行する。比較の基準は実際の受入で採用した `baseline-checked` を使う。

```bash
$prd128Label = 'repeat-' + (Get-Date -Format yyyyMMdd-HHmmss); .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py repeat --label $prd128Label --workers 4 > ".scratch/prd128/$prd128Label.log" 2>&1
```

```bash
$prd128Label = 'compare-' + (Get-Date -Format yyyyMMdd-HHmmss); .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py compare --label $prd128Label --reference baseline-checked --workers 4 > ".scratch/prd128/$prd128Label.log" 2>&1
```

```bash
$prd128Label = 'checkpoint-' + (Get-Date -Format yyyyMMdd-HHmmss); .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py checkpoint --label $prd128Label --reference baseline-checked > ".scratch/prd128/$prd128Label.log" 2>&1
```

性能はほかのRunner・test・buildが動いていないことを確認してから1コマンドずつ実行する。ステージ1・4は通常の入力、ステージ3は `--rnd` を使い、いずれも3巡以上とする。

```bash
$prd128Label = 'performance-' + (Get-Date -Format yyyyMMdd-HHmmss); .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py performance --label $prd128Label --rounds 3 > ".scratch/prd128/$prd128Label.log" 2>&1
```

```bash
$prd128Label = 'performance-rnd-' + (Get-Date -Format yyyyMMdd-HHmmss); .\.venv\Scripts\python.exe core/anet-core/testdata/prd128/run_validation.py performance --label $prd128Label --rounds 3 --rnd > ".scratch/prd128/$prd128Label.log" 2>&1
```

## 前提

- 監査時の基準コミットは `2f3e8d81`。PRD129 は既に実装済みで、未実装とする PRD128 §10 の記述は時点差として扱う。PRD129 の機能変更を今回加えない。
- ユーザー判断が必要なブロッカーは 0。U1〜U3 は PRD が定めた実測事項で、結果に応じた分岐も同 PRD に従う。
- ステージ中に他の変更が対象コードへ入った場合は、前後比較の成立を確認し、必要なら切れ目で基準を保存し直す。
- 新しい設定キー・メトリクス名・互換層を追加しない。新規 ADR と CONTEXT の変更は不要（D21）。既存の検証資材は変更せず呼び出す。
- 計画保存時に対象 5 課題を対応中へ更新する。status を解決にするのは今回の工程に含めない。

## U1 の照合とステージ 4 の適用範囲

| 群 | 現在の購読・間隔・要求 | 共通型への載せ方 |
|---|---|---|
| gradient noise | 指標ごとの gate と固定長の要求、購読から計算資源の種類を導出 | 共通型で解析・間隔・要求を持ち、資源の導出を群に残す |
| noisy | 指標ごとの gate。前半の統計と後半の probe で必要な資源が違う | 共通型で要求を持ち、名前で示した統計/probe の範囲を使う |
| policy churn | 指標ごとの gate と要求、同期の位相に関する起動時 WARN | 共通型で要求を持ち、位相の検証と forward 順序を群に残す |
| replay fit | 学習有効時だけ登録し、指標ごとの gate から計数・抽選・採点の要否を導出 | 共通型で要求を持ち、必要な計算の解決を群に残す |
| plasticity | online/target/probe の 3 チャネル、最小間隔で先に振り落としてから個別 gate を進める | 現行の固有構造を維持する。共通 scalar 読み出しと名前での指標参照、並びの検証は適用する |
| attention | 層番号が動的な指標と層ごとの gate | 現行の層別要求を維持する。共通 scalar 読み出しを適用し、層は解析した番号を使う |

指標の並びの破壊検査は、共通型に載る noisy と、固有構造を残す plasticity の名前付きの並びで行う。キー文字列は定義に 1 回だけ置き、指標 ID と位置が一致することを static_assert で検査する。並びの隣接 2 項目だけを交換するとこの検査が失敗する形にする。これは実装上の手段であり、指標名と要求が立つ更新の時機は変えない。

## ステージ 2 の分割と記録の方針

設定を読む順序は現状のままとし、内的報酬、Head の射影と構築設定、forward のカタログ、Actor のカタログ、target policy の既定値、Learner、報酬スケーラ、観測の正規化、選択値、skip、Actor の最終設定、組み合わせの検証へ分ける。Actor の読み取りスコープはカタログ読み取りの関数から返し、後半の最終設定と未読キーの検査まで同じ追跡状態を使う。メンバへ新しい可変状態を足さない。

接頭辞の下を記録する処理は、相対キーを渡す関数 1 つから RecordConfigValue を呼ぶ。写す先の ConfigData のキーの形と、値の型（opaque な欄は文字列）は保つ。skip の net.block/net.config_profile のコピーには、共通カタログの参照を解決するためのコピーであることをコメントする。

語の検証はキー・値・候補、数値の範囲の検証は条件・キー・値・期待範囲を渡す private の関数へ集める。方策と τ の検証は D5 のとおり本文と文言を変えず agent.hpp へ移す。複数の設定の組み合わせで判定する検査と参照の解決は、既存の条件を保ち、単一値の範囲検証へ混ぜない。

## 経過

- 2026-11-14: PRD と現行コードを照合。Git 状態はサンドボックス内で失敗するため、サンドボックス外の読み取りで確認した。他の作業の変更を確認し、対象コードに既存変更が無いことを確認した。変更前 Debug ビルドは成功（no work to do）、Release ビルドを開始した。
- 2026-11-14 ステージ 0: Debug・Release ビルド成功。`run_validation.py snapshot` により基準を `.scratch/prd128/acceptance/baseline/` に保存した。PRD125 の 31 FP32 組と追加 3 組の入力を生成し、基準の反復 Run と既定 core suite を実行中。
- U2: `[prd128]` の特性テストで、未定義の actions・target_returns・next_state.terminals のそれぞれが CPU 転送で例外になることを確認した。捕捉時に ERROR のログは出なかった。ステージ 1 で未定義保持へ期待値を変更する。全 18 tensor を入れた CPU/CUDA 転送・走査・metadata の CPU 保持は成功（`.scratch/prd128/characterization-samples-before-r2.xml`、47 assertions、失敗 0）。
- 比較の道具: 実ログのレベル表記が `[W]`・`[E]` であることを確認し、採取する式を合わせた。基準反復の完了後に `audit-reference` で保存済みの全ログを再検査する。比較の道具自身の 3 テストは成功し、警告の変更・設定行順の変更・JSON の変更をそれぞれ検出した。
- 特性テスト: 内的報酬の checkpoint、Trace と学習のボーナスの一致、学習停止中の intrinsic 診断の保存は既存の `[prd116]`・`[prd127]`・`[prd129]` を再利用する。診断の gate は既存の plasticity・policy_churn・replay_fit・noisynet・attention の demand/gate テストを使い、停止中の 6 群の代表キーを既存の disabled Learner テストへ追加した。Rainbow は通常 Q/QR の 2 構成を追加し、全 lane に同じ終端遷移を入れて抽選・target の乱数に期待値を依存させず、損失列と線形 Head の全 weight/bias を基底出力から復元した checksum を比べる。
- RND の短縮構成: 8 lane に対し intrinsic.learn_batch_size の既定が 32 のため、そのままでは成立しない。新しい検証入力 `rnd_LunarLander_b8`・`rnd_Atari_b8` に限り batch=8、RND warm-up=256 を明示し、補助学習の実行と有限の診断値も短い Run 内で確認する。最初の入力は保存して書き換えない。基準の既定 core suite は終了値 0、失敗 0（592.44 秒）。
- GPU の採取: 基準の全構成の入口と RND 追加の入口が GPU の構成で重なった。CUDA 条件を逐次実行する契約に合わせ、この GPU 結果は基準へ採らず、両入口の完走後に GPU の現用 4 構成を 1 つの入口から逐次で採り直す。CPU の成功結果と、今回の全ログは保持する。
- 特性テスト最終確認: `.scratch/prd128/characterization-before-final.xml` は終了値 0、172 assertions、失敗 0。Rainbow の通常 Q/QR は Agent と torch の seed をどちらも 128 に固定し、別 process の結果が一致することも確認した。
- PRD の追随: U2 の確認結果を PRD の最新改訂行・§6.2・§8 へ反映し、成果物キー prd で独立レビューを受けた。結果は OK、指摘なし（`.scratch/prd128/review-20261114-1236-prd.md`）。
- U3 の試行: 指定の @full を重ねた最初の DropMerge は、取得先の無い旧 Agent 指標 loss_ema が Env まで探索され、集約接頭辞の検査で停止した。検証用の別入力 `diagnostics_DropMerge_explicit_sources` では @full の選択を保ち、取得先の無い Agent 指標だけ `$agent` を明示する。未提供の旧キーは既存の未知キー WARN として扱い、その文言と回数も前後比較に含める。出荷設定・GetScalar の契約は変更しない。最初の失敗の実行・入力・ログは保持する。
- 採取と TDD の実行順: 基準の Run は保存した実行体だけを使い、現在の本体の変更と独立しているため、GPU 基準の逐次採取を続けながらステージ 1 の TDD を進める。基準実行体と凍結入力は書き換えない。ステージ 1 の完了判定は、34 構成の基準反復・U3 の確認と、同じ入力での変更後との比較がすべて通ってから行う。
- ステージ 1 RED: 未定義保持の期待値へ差し替え、CPU の 3 例が例外、CUDA の 3 例は成功することを確認した（`.scratch/prd128/stage1-undefined-red.xml`、終了値 1）。まず CPU の 3 つの素の .to だけを安全な既存 anet::To へ替え、GREEN 後に並びの共通化を行う。
- ステージ 1 GREEN: CPU の 3 行を anet::To へ替えた最小実装で、関連特性テスト群は 187 assertions、失敗 0 になった（`.scratch/prd128/stage1-undefined-green.xml`、終了値 0）。次に全フィールドの文字列化を 1 behavior として RED→GREEN にし、その後に共通走査へリファクタリングする。
- U3: 取得先を明示した診断組の 1 回目が成功し、可塑性・policy churn・replay fit・noisy・attention・gradient noise の 6 群のそれぞれに有限点があることを audit で確認した。6 群は 1 組で足りる（`.scratch/prd128/acceptance/baseline-gpu-serial_1/diagnostics_DropMerge_explicit_sources/`）。
- ステージ 1 の文字列化 RED: 18 フィールドの名前を確認するテストで 9 件の失敗（省略されていた 8 フィールドと actions の表記）を確認した（`.scratch/prd128/stage1-string-red.xml`、終了値 1）。
- ステージ 1 の文字列化 GREEN: 8 フィールドの表示と actions の表記を加えた最小実装で、関連 205 assertions が成功した（`.scratch/prd128/stage1-string-green.xml`、終了値 0）。その後、ExperienceSamples の private static の VisitFields へ名前・転送先を伴う並びを集め、To・ForEachTensor・ToString を共通走査へ載せた。公開の関数、集成体の初期化、フィールドの順序は保持し、転送元全体の一時コピーも足していない。Debug・Release の全ビルドを開始した（`.scratch/prd128/stage1-refactor-build.log`）。
- 共通走査のリファクタリング後: Debug 全ビルド成功、関連 205 assertions が成功（`.scratch/prd128/stage1-refactor-green.xml`、終了値 0）。Release 全ビルドを継続中。
- 設計文書の追随（ステージ 1）: 150 の §2.2 に共通のフィールド一覧、未定義保持、定義済みだけの走査、全フィールドの文字列化を書き、§8 に検証資材を追加した。日英の対応と、見出し・表・コードブロック・英語本文・相対リンクの検査は成功。英訳の blob は確認済みの `363082e5a8c67800b61e9b6a1afcd610e9cf17b1` に更新した。glossary の追加と drift 候補はなし。§7.7・§7.8 の probe RNG の記述はステージ 4 で追随する。
- ステージ 0 の基準確定: GPU を 1 つの入口から逐次で採り直した 4 構成もすべて一致した。`audit-reference --label baseline-checked --sources baseline baseline-rnd-b8 baseline-gpu-serial` で現用 34 構成をまとめて再検査し、34/34 成功。基準は `.scratch/prd128/acceptance/baseline-checked-summary.json`。旧 RND 2 構成と旧診断組の失敗・重複した GPU の結果は原記録として保持し、基準には採っていない。
- ステージ 1 の Debug・Release 全ビルドは成功（終了値 0）。他の Run・ビルドが終了した状態で、Release の速度比較を開始した（`stage1-performance-launcher.log`）。
- ステージ 1 の速度: 捨て Run 後に基準→変更後を交互に 3 巡測定し、合格。更新 1 回あたりの実時間は基準の中央値 0.0413645786466077 秒、最大値 0.04255728854332119 秒、変更後の中央値 0.04046355024911463 秒（`.scratch/prd128/acceptance/stage1-performance-summary.json`）。改善効果は断定せず、PRD のブレ幅以内という受入として扱う。
- ステージ 1 の前後比較: `compare --label stage1 --reference baseline-checked --workers 4` を開始した。比較が済むまで Runner の本体を次のステージへ変更しない。
- ステージ 1 の受入完了: 34/34 構成の scalar・設定ファイル・設定 JSON・WARN/ERROR がすべて一致した（`.scratch/prd128/acceptance/stage1-summary.json`）。速度の受入も合格しており、ステージ 2 を開始する。
- ステージ 2 の最初の RED: 公開の設定読み込みで `learner.noisy_probe.num_draws=0` のエラー文を D6 の形で検査し、旧文言との差で 1 assertion が失敗した（`.scratch/prd128/stage2-numeric-red.xml`）。本体の最小修正はステージ 1 の比較完了後に行う。
- ステージ 2 の数値検証 GREEN: 対象のエラー文 1 か所だけを直した最小修正でテスト成功（`.scratch/prd128/stage2-numeric-green.xml`、終了値 0）。次に選択肢 2 件の既存テストの期待値を D6 へ替え、3 件の設定読み込みテストを足して RED を確かめる。
- ステージ 2 の選択肢 RED: 2 候補（skip forward mode）の 2 例、3 候補（quantile mode）の 1 例が旧文言との差で失敗し、先の数値検証は成功した（`.scratch/prd128/stage2-choice-red.xml`、終了値 42）。対象 2 か所の文言だけを直して GREEN を確かめる。
- ステージ 2 の選択肢 GREEN: 対象 2 か所の最小修正で 4 assertions が成功した（`.scratch/prd128/stage2-choice-green.xml`、終了値 0）。その後、読み込みを順序を保った 13 の関心ごとの関数へ分け、語と数値の検証を各 1 関数へ、接頭辞の写しと記録を `CopyAndRecordPrefix` へまとめた。方策・τ の検証は文言も判定も変えず agent.hpp へ移した。ANET_READ_CONFIG の 119 件の並びは変更前と同一。Debug 全ビルドと関連テストを行う。
- ステージ 2 の追随漏れ: Debug 全ビルドは成功した。関連 757 assertions のうち、policy churn の既存テストが旧文言 value=0 を断定する 2 assertions で失敗した（`.scratch/prd128/stage2-refactor-green.xml`）。D6 が許可した変更として期待値を直し、PRD の §2.4・D6・§6.6 の既存テスト数を 2 本へ補正した。判定・値域・有効な設定の挙動は変更しない。
- PRD の再確認: §6.2 のテスト数も含めて整合を確かめ、同じ PRD レビューエージェントへ再提出した。結果は OK、指摘・確認事項・範囲外の検出なし（`.scratch/prd128/review-20261114-1338-prd.md`）。追随後の関連 757 assertions は成功（`stage2-refactor-r2.xml`）。skip.max_length の指定有無と有効化の組み合わせ検証は数値の範囲だけの検証とは別で、specified の情報も含めた旧文言を保つ。これを含む最終ビルドと関連テストを確認中（`stage2-refactor-r3-build.log`）。
- ステージ 2 の最終確認: Debug 全ビルド成功、関連 757 assertions がすべて成功（`.scratch/prd128/stage2-refactor-r3.xml`、終了値 0）。`compare --label stage2 --reference baseline-checked --workers 4` で 34 構成の比較を開始する。比較が終わるまで次のステージの本体は変更・ビルドしない。
- ステージ 3 の実装の細部: Coordinator は Agent の private な入れ子とし、unique_ptr で所有する。設定は不変な値として保持する。β の更新、採点、報酬への加算、補助学習の順と intrinsic_reward の計測範囲を保つため、AdvanceBeta → IsLearningEnabled → ScoreBonus → Agent の加算 1 行 → Learn と委ねる。ボーナスの共通式は CalculateBonus、Trace 固有の NaN 処理は ScoreTrace に置く。Trace の callback は Agent と同じ制御ブロックを持つ Coordinator の aliasing shared_ptr を捕捉し、Agent を採点へ参照せず、今と同じ寿命とロックを保つ。下書きから適用済みで、以後は本体を正本にする。前方宣言した型の破棄は専用 deleter を cpp で定義して閉じ、Agent の公開の特殊メンバは変えない。
- ステージ 3 の準備確認: CPU の既存 PRD129 テストは、本体を分離する前に 134 assertions すべて成功した（`.scratch/prd128/stage3-prd129-before.xml`）。200（852 行、blob 27589dcbd235883fda60dc486511c1b595b48ccb）と 212（551 行、blob 20a3be0a2af54d32e5ef7cc1bda662470d746db2）の日本語正本を全文読んだ。文書の追随時は、並行変更の有無をこの版と照合する。
- 実行資源の確認（13:52）: 別作業の PRD127 cost 用 Release Runner が動いている。ステージ 2 の比較は値・設定・ログの一致だけを判定して続ける。後続の Release ビルドと性能測定は、ほかの Runner の終了を確認してから行う。別作業の実行は停止しない。
- ステージ 2 の受入完了: 34/34 構成で全比較項目が一致し、比較の process 終了値も 0（`.scratch/prd128/acceptance/stage2-summary.json`）。内的報酬の分離を本体へ適用し、Debug 全ビルドと関連テストを開始した（`.scratch/prd128/stage3-build.log`）。
- 追加の特性テスト: Agent を 2 個同じ seed で作り、probe の抽選を進めたあとに片方だけ購読し直し、続く policy churn と学習損失の一致を確認するテストを追加した。分離前は 41 assertions 成功（`.scratch/prd128/probe-resubscribe-before.xml`）。非ゼロの更新も確認する断定を足して、後続の検証にも含める。取り込み・credit・抽選・checkpoint・β の既存 CPU テスト群も分離前に 545 assertions が成功した（`stage3-intake-before.xml`）。
- ステージ 3 のビルド: 最初の案では、前方宣言した型の破棄に伴って default の move constructor を明示し、既存の複数 Module 基底の変換が曖昧になる C2594 でビルドが止まった（`stage3-build.log`）。公開の特殊メンバを足す案をやめ、private な入れ子と専用 deleter へ直した。Debug 全ビルドと関連 745 assertions が成功（`stage3-r2-build.log`・`stage3-intake-after.xml`、終了値 0）。
- ステージ 3 の受入開始: 別作業の Runner の終了を確認し、Release 全ビルド、34 構成の比較、旧 RND checkpoint の継続比較を開始した。CPU の Run は比較を 3 並列、checkpoint を 1 つずつにして最大 4 に保つ。性能はこれらの終了後に行う。
- ステージ 3 の途中結果: Release 全ビルド成功。旧 RND checkpoint から新旧を各 8,704 experience 続け、31,156 scalar 点・設定・ログがすべて一致した（`.scratch/prd128/acceptance/stage3-checkpoint-summary.json`）。
- ステージ 3 の前後比較完了: 34/34 構成でscalar・設定ファイル・設定JSON・WARN/ERRORが一致し、process終了値も0（`.scratch/prd128/acceptance/stage3-summary.json`）。全intrinsicテストを確認してから、他のRunner・test・buildが動いていない状態でRND有効のRelease性能を測る。
- ステージ 3 の設計文書: 200 のクラス表・構造図・取り込み・設定読み込み・所有・保存・検証の参照、212 の担当・処理順・図・コード案内・保存・Trace を追随した。212 へ別作業が追加した GUI 費用の測定と英訳も保持した。日英の見出し、表、コードブロックの内容、日本語残り、相対リンクの検査は 150・200・212 とも成功（`.scratch/prd128/stage3-translation-check.json`）。英訳の確認済み blob は 200 が 1274683f6e3b1b1617a3f7b3eef1dd0b3786c03f、212 が 3e19e24825d44b926293b7aea7ac2878df4fc6e8。glossary への追加はなし。既存の 212 §5.1 の Atari 費用が未測定という文は §5.2 の測定記録と食い違う drift 候補で、翻訳では原文どおり保った。
- 並行作業の確認（2026-11-14 13:17）: Atari の RND 設定、200・212 の設計文書などに別作業の変更が加わった。比較は保存済みの入力を使い続ける。設計文書を追随するときは、この時点以降の現物を読み、別作業の変更を保つ。AGENTS.md の追加規約（コミットの状態を記憶から言わない）も確認した。

## 実施した検証

| 検証 | 結果 | 記録 |
|---|---|---|
| 変更前 Debug・Release ビルド（VsDevCmd.bat 経由） | 成功 | 基準の manifest.json に実行体の SHA-256 を保存 |
| 全フィールド転送と U2（本体変更前） | 成功 | `.scratch/prd128/characterization-samples-before-r2.xml` |
| 比較道具の単体テスト `run_validation_test.py` | 3/3 成功 | コンソールの unittest 出力 |
| 追加した特性テストと停止中診断（本体変更前） | 成功（172 assertions、失敗 0） | `.scratch/prd128/characterization-before-final.xml` |
| 基準反復 34 構成 | 成功（34/34） | `.scratch/prd128/acceptance/baseline-checked-summary.json` |
| 基準の既定 core suite | 成功（終了値 0、失敗 0、592.44 秒） | `.scratch/prd128/acceptance/baseline-suite/` |
| ステージ 1 の Debug・Release 全ビルド | 成功 | `.scratch/prd128/stage1-refactor-build.log` |
| ステージ 1 の関連特性テスト | 成功（205 assertions、失敗 0） | `.scratch/prd128/stage1-refactor-green.xml` |
| ステージ 1 の速度（3 巡） | 合格（変更後の中央値 ≤ 基準の最大値） | `.scratch/prd128/acceptance/stage1-performance-summary.json` |
| ステージ 1 の 34 構成の前後比較 | 成功（34/34、全比較項目が一致） | `.scratch/prd128/acceptance/stage1-summary.json` |
| ステージ 2 の Debug 全ビルドと関連テスト | 成功（757 assertions、失敗 0） | `.scratch/prd128/stage2-refactor-r3-build.log`・`stage2-refactor-r3.xml` |
| ステージ 2 の 34 構成の前後比較 | 成功（34/34、全比較項目が一致） | `.scratch/prd128/acceptance/stage2-summary.json` |
| ステージ 3 の Debug・Release 全ビルド | 成功 | `.scratch/prd128/stage3-r2-build.log`・`stage3-release-build.log` |
| ステージ 3 の関連テスト | 成功（745 assertions、失敗 0） | `.scratch/prd128/stage3-intake-after.xml` |
| ステージ 3 の旧 RND checkpoint 継続 | 成功（全比較項目が一致） | `.scratch/prd128/acceptance/stage3-checkpoint-summary.json` |
| ステージ 3 の 34 構成の前後比較 | 成功（34/34、全比較項目が一致） | `.scratch/prd128/acceptance/stage3-summary.json` |
| ステージ 3 の全 intrinsic と速度 | 成功（4,707 assertions、変更後中央値70.30 ms/更新 ≤ 基準最大73.32 ms/更新） | `stage3-intrinsic.xml`・`acceptance/stage3-performance-summary.json` |
| ステージ 4 の Debug・Release 全ビルドと DQN 関連 | 成功（142,325 assertions、失敗0） | `stage4-r5-build.log`・`stage4-release-build.log`・`stage4-dqn.xml` |
| ステージ 4 の指標の並びの破壊検査 | noisy・weight norm ともコンパイルで検出、復元成功 | `stage4-noisy-order-negative.log`・`stage4-plasticity-order-negative.log`・`stage4-order-restored-build.log` |
| ステージ 4 の34構成と速度 | 成功（34/34、変更後中央値39.63 ms/更新 ≤ 基準最大41.91 ms/更新） | `acceptance/stage4-summary.json`・`stage4-performance-summary.json` |
| 最終版の Debug・Release ビルド | 成功 | `stage5-final-checked-debug-build.log`・`stage5-final-checked-release-build.log` |
| 最終版の既定 core suite | 成功（179,995 assertions、失敗0、基準と同じskip1） | `acceptance/final-suite-r2/suite.json`・`core.xml` |
| ステージ 5 の34構成の前後比較 | 成功（34/34、973,936 scalar点と全成果物が一致） | `acceptance/stage5-summary.json` |
| PRD の指定検索・文字コード/改行・翻訳・検証道具 | 成功（検索22項目、形式36ファイル、日英5対、単体3テスト） | `final-acceptance-search.json`・`check_file_format.py`・`final-translation-check.json`・`final-212-translation-check.json`・`final-validation-tool-test.log` |

- ステージ 3 の受入完了: 全 intrinsic テストは 4,707 assertions、失敗 0（`stage3-intrinsic.xml`）。他の Runner・テスト・ビルドが無い状態で RND 有効の性能を新旧交互に 3 巡測り、変更後中央値 0.07030206907074898 秒/更新が基準最大 0.07331772067118436 秒/更新以内で合格（`acceptance/stage3-performance-summary.json`）。全 34 構成と旧 checkpoint の比較も一致済みで、ステージ 4 を開始する。

- ステージ 4 の実装: 固定指標の4群を `DiagnosticDemandTable` に載せ、解析・間隔・gate・要求を共有した。plasticity と attention は U1 の固有構造を保つ。skip・Munchausen・IQN・weight norm と6群の scalar を `ReadDiagnosticScalar` に寄せ、名前付きの位置とキー・値 pack の並びをコンパイル時に検査する。5つの probe RNG は Learner が Agent の基の seed から構築して保持する。Noisy 設定は Network に Noisy parameter があるときだけ Agent の構築時に解決して渡す。
- ステージ 4 の移行確認: 内部 Learner の引数変更の追随漏れ（派生4定義・テストの直接構築と購読呼び出し）をビルドで検出し、修正した。Debug 全ビルドは成功（`stage4-r5-build.log`）。直接構築テストは共通の Agent 基の seed 6000 を渡す。直接注入していた probe 抽選器は削除し、Noisy の opaque な検査用設定は構築時に渡す形へ移した。断定の期待値は変更していない。
- ステージ 4 の文書: 110・150・200 の日英の所有と購読・読み出しの説明を更新。日本語正本010（405行）と110（245行）の残りも全文読んだ。日英の構造・コードブロック・日本語残り・リンク検査は成功（`stage4-translation-check.json`）。110 の確認済み blob は 0915176d9bc219534c650a7365be9636d964be9e、150 は 4a0d5ce4f4673ad5fc1049a6a1630269340bb75f、200 は 960306aad1719d6bb62480bb407c34fc5e699fa3。glossary 追加なし。DQN関連テスト、34構成比較、Release全ビルドを実行中。

- ステージ 4 の並びの破壊検査: noisy の先頭2指標の交換は `MakeDiagnosticMetricKeys<NoisyMetric::Count>` の定数評価で C2131 により失敗した（`stage4-noisy-order-negative.log`）。noisy を戻し、固有構造を保つ plasticity weight norm の先頭2指標を交換した場合も同じ検査で失敗した（`stage4-plasticity-order-negative.log`）。両方を復元した後の DQN 本体 object のビルドは成功（`stage4-order-restored-build.log`、終了値0）。一時変更を実行体へリンクしていない。

- ステージ 4 のテスト完了: `[dqn]` は 142,325 assertions、失敗0、終了値0（`stage4-dqn.xml`、491.281秒）。ステージ0で追加した再購読の特性テストも含む。Release 全ビルドも成功（`stage4-release-build.log`）。全34構成の比較と性能の受入は継続中。

- ステージ 4 の文書の補足: PRD §6.6 が保持を指定した 200 の `caller-owned` の2行は原文を保った。これは ReplayBuffer から見た呼び出す側を意味し、今回の所有の移動と矛盾しない。200 の確認済み blob を 9611a186ecbe9a7836172c1431e08801e33c3ad2 へ更新した。

- ステージ 4 の前後比較: 全34構成で全scalar・設定ファイル・設定JSON・WARN/ERRORが一致し、終了値0（`acceptance/stage4-summary.json`）。残る性能の受入を、他のRunner・test・buildが0であることを確認して開始した（`stage4-performance-launcher.log`）。
- ステージ 5 の TDD 準備: 本体を変更する前に、RainbowのActor方策の7つの不正値を設定読み込みとDTO直接構築で拒否するテストを追加した。Debugのtest targetをビルドし、14例とも例外にならず失敗するREDを確認した（`stage5-policy-red.xml`）。ステージ4のRelease性能が済むまで本体の最小修正は適用しない。構築共通化の下書きは `.scratch/prd128/*stage5.*` に保存済みで、まだ本体へ適用していない。

- ステージ 5 の共通部品の細部（未適用の下書き）: `DqnComponentAssembler` は内部の dqn_based_agent.hpp/cpp に置き、具象 Agent の設定型を受け取らない。Head の初期化と射影は呼ぶたびに渡す。Learner のログは呼び出す側の Head の分位数と任意の Munchausen 接尾を受け、DTO 内の値が異なる場合も既存の文言を保つ。Actor の共通部分は複製と生成を担い、DefaultDQN 固有の skip の選択は Agent に残す。DefaultDQN の構築は `SelectHeadFactory`・`MakeNetworkInputSpec`・`RecordNetworkStructure`・`InitializeLearner` へ分ける。
- 自己点検の途中で、指標表の集成体の初期化を指示付きにすること、キー一覧の保持を借用ポインタから値へすること、値 pack の指標数もコンパイル時に照合することを下書きへ反映した。実行体へはまだ適用していない。noisy と weight norm の並びの検査そのものはステージ4で確認済み。

- ステージ 4 の受入完了: 性能は新旧交互3巡で、変更後中央値0.03963021008530632秒/更新 ≤ 基準最大0.04191145417280495秒/更新で合格（`acceptance/stage4-performance-summary.json`）。34構成・DQN関連142,325 assertions・並びの破壊検査・Debug/Releaseのビルドと合わせ、ステージ4の受入を完了した。ステージ5の最小修正としてRainbowの方策制限だけを設定読み込みとAgent構築から呼ぶ形で追加し、GREENを検証中。

- ステージ 5 の方策制限 GREEN: 最小修正で14 assertionsが成功（`stage5-policy-green.xml`、終了値0）。続いて通常のtau ruleとfull queryのtau ruleについて、num_taus=0と未知sample_modeを両構築経路で拒否するテストを追加し、REDを確認する。

- ステージ 5 のtau検証 RED: 通常問い合わせとfull queryのそれぞれの num_taus=0・未知sample_modeが、設定読み込みとDTO直接構築の8例で例外にならず失敗した（`stage5-tau-red.xml`）。最小修正として、方策制限の直後に `ValidateActionPolicy` を呼ぶ1行だけを追加した。GREENを検証中。

- ステージ 5 のtau検証 GREEN: 共通の検証を呼ぶ1行を追加した最小修正で、方策とtauの22 assertionsが成功（`stage5-tau-green.xml`）。次に、Rainbowが `target_policy.epsilon`・`plasticity_probe_srank`・`noisy_q_abs_diff` を返すこととepsilon-greedyの2つの表記を検査するテストを足した。6 assertionsが未知キーとの差で失敗するREDを確認（`stage5-routes-red.xml`）。既存の分岐へ対象の3経路だけを足した最小修正でGREENを検証中。

- ステージ 5 の値の振り分け GREEN: 最小修正で16 assertionsが成功（`stage5-routes-green.xml`）。その後、下書きの共通構築を本体へ適用した。最初のビルドでは値packの指標数を受けるラッパー側の型引数の追随漏れで停止し（`stage5-refactor-build.log`）、同じ `Count` 引数を両側へ定義して直した。Debug全ビルドを実行中。
- ステージ 5 の文書: 010・110・200の日英に、Rainbowを残す役割と制限を反映し、200の部品表・クラス図・構築の説明を共通部品と診断の構成へ追随した。日英の全5文書を構造・コードブロック・リンクの検査へ出した。

- ステージ 5 の Debug 全ビルドは成功（`stage5-refactor-r2-build.log`）。共通化後の `[prd128]` 特性テストは238 assertions、失敗0（`stage5-characterization.xml`）。最終の既定core suiteとCPU30構成の比較（3並列）を別の入口で開始した。CUDA4構成は両方の終了後に逐次実行し、suiteのGPUテストと重ならないようにする。前後比較は `stage5-cpu`・`stage5-gpu` の結果を最終的に34構成へまとめる。
- 日英5文書の検査は成功（`stage5-translation-check.json`）。確認済みblobは010=e9dc0f9af434537e68497fa7730b162da23fd1e5、110=c23257d699ad3b451bf14f9a67687be72af1a1db、150=4a0d5ce4f4673ad5fc1049a6a1630269340bb75f、200=92ff82582888b655469a126529b65eb7cc68f374、212=3e19e24825d44b926293b7aea7ac2878df4fc6e8。glossary追加なし。先の212の既存drift候補は保持する。

- 自己点検で、共通読み出しの英語コメント、繰り返すキー文字列、触ったテストの位置指定の集成体初期化、Actor複製の説明コメントを追随した。キーの文字列・判定・数値の期待値は変えていない。Head Factoryを共通層へ移したRainbowの不要なincludeも除いた。
- 最終検証は修正後の実行体でそろえるため、`stage5-cpu` と `final-suite` の2入口を、コマンドラインのlabelとworkspace内Python実体で特定して子processごと停止した。途中の入力・Run・ログを保持し、合格記録には採らない。再検証は新しい `stage5-cpu-r2`・`stage5-gpu-r2`・`final-suite-r2` のlabelで行う。
- 名前の動詞の照合: Head Factoryの `CreateHead`、Agentの `CreateActor`、旧DefaultDQNの `CreateActionPolicy` と同じく、Assemblerの `Create*` は対応する部品を生成する。Agentの `CreateLearner()` は従来どおり外側facadeを返し、Assemblerの `CreateLearner` は内側の具体Learnerを生成する。

## PRD121 の足す先（G2）

| PRD128 §6.7 の行 | 実装後の関数名と足す先 |
|---|---|
| 取り込みの成分とβの受け渡し | `IntrinsicRewardCoordinator::ScoreBonus` の返値と `DefaultDQNAgent::UpdateFromBatch` の加算1行。βの更新は `IntrinsicRewardCoordinator::AdvanceBeta` |
| 報酬スケーラの保存と読み込み | `DefaultDQNAgent::Save` / `LoadNetwork` の保存順。内的側をCoordinatorへ持たせる場合は `IntrinsicRewardCoordinator::Save` / `Load` |
| Evaluation Trace のボーナスの式 | `IntrinsicRewardCoordinator::CalculateBonus`。学習と `ScoreTrace` が共用する式1か所 |
| intrinsic.reward_scaler.scale の読み出し | `IntrinsicRewardCoordinator::GetScalar` |
| 標本の成分ごとの収益とstep報酬 | `ExperienceSamples::VisitFields` の名前・転送方針を伴う並び1か所 |
| 合成Head | `DqnComponentAssembler::CreateHeadFactory`。初期化と射影の設定を呼ぶたびに受け取る |
| 設定の読み取り | `DefaultDQNAgentConfig::ReadIntrinsic` / `ReadHeadAndCommon` / `ReadLearner`。語・数値の検証は `ValidateChoice` / `ValidateRange`、接頭辞のコピーと記録は `CopyAndRecordPrefix` |
| 診断の群 | `ReadDiagnosticScalar` に指標の解析と値の読み出しを渡す1組。固定指標の解析は `ParseDiagnosticMetric`、欠測は共通の処理でNaNへ変換する |
| §1.1 の外の処理 | 変更しない。`dqn::Actor`、ReplayBufferのPush、最適化、Qの合成はPRD121の対象として残す |

- 共通Learner Factoryの自己点検: 型ごとの3分岐で引数列を重ねず、局所のtemplate lambdaへ構築を集め、引数の列を1回だけ書く形にした。種類の選択と従来のログは各分岐に残す。課題72d640の「引数を足す場所」を1か所にするためで、構築へ渡す値と順は同じである。

- Actor の hint 判定も `DqnComponentAssembler::CreateActor` の1か所へ移した。共通のLearnerConfigからPERと初期priority modeを判定する。各Agentが選ぶhintの設定値は引き続き入力として受け、Rainbowが持たない設定は補わない。判定の位置はsourceのcloneの後に保つ。

- 自己点検の仕上げ後、Debug・Releaseの全ビルドが成功（`stage5-final2-debug-build.log`・`stage5-final2-release-build.log`）。最後の追加includeの配置変更後は、Debugのcore test/RunnerとReleaseのRunnerを対象に再ビルドする（`stage5-include-debug-build.log`・`stage5-include-release-build.log`）。数値・分岐・期待値の変更はない。

- 共通の診断packは固定配列を `torch::stack` のTensorListへ直接渡し、一時vectorの確保とTensor参照のコピーを除いた。libtorchのArrayRefがstd::arrayを受けることをローカルの正本で確認した。ここでコードの自己点検の修正を閉じ、以降は最終検証と独立レビューに進む。
- `docs/quality/impl.md` の全項目を適用範囲に従って通した。設定・bat・skillの変更は対象外。機械検査では36ファイルのUTF-8/BOM保持・LFが成功。指標番号・Agentのprobe乱数・Agent内のHead/Learner具体型生成の指定検索は0件。リテラルの重複候補の残りは、別の設定項目の列挙値、設定キー、既存テストから移したローカル入力、単独のテストの期待値である。追加した共通キーは定数へ寄せた。Runtimeの受入結果とレビューの判定は、この後に追記する。

- 最終版のDebug core test/RunnerとRelease Runnerのビルドは成功（`stage5-final-checked-debug-build.log`・`stage5-final-checked-release-build.log`）。`[prd128]` は238 assertions成功（`stage5-final-characterization.xml`）。この版でCPU30構成の比較と既定core suiteを `stage5-cpu-r2`・`final-suite-r2` の新しいlabelから開始した。CUDA4構成は両方の終了後に流す。以後は独立レビューの指摘が無ければコードを変更しない。

- 最終の既定core suiteは終了値0、失敗セット空で基準と一致した（`acceptance/final-suite-r2/suite.json`・`core.xml`）。CPU30構成の比較の残りを継続し、終了後にCUDA4構成を逐次実行する。
- 機械検査の最終結果: PRD §7 の指定検索22項目がすべて合格（`final-acceptance-search.json`）。比較道具の3テストも成功（`final-validation-tool-test.log`）。対象コードと設計文書・計画の `git diff --check 2f3e8d81` は終了値0。

- 最終core suiteの件数: 179,995 assertions、errors=0、failures=0、skipped=1。実時間503.1869611000002秒（JUnitのtest時間499.496秒）。skipの理由は基準側とも照合して記録する。

- skip 1件は `PRD070 reports every explicit CUDA key before NN initialization`。CUDAが無い環境向けの検査で、この機械はCUDAが使えるため `CUDA is available` としてskipされた。基準suiteでも同じ検査が同じ理由でskipされており、今回増えたskipではない。

- CPU30構成の最終比較は全項目一致、終了値0（`acceptance/stage5-cpu-r2-summary.json`）。既定suiteとCPU比較の終了後、CUDA4構成を `stage5-gpu-r2` で1つずつ開始した。コードと計画の独立レビューは並行して読み込みを始め、最終判定はこの追加の受入結果が揃ってから受ける。

- 独立レビュー: `anet-review-agent` の新しいコンテキスト `prd128_impl_review` にimpl-planとimplを依頼した。差分は `.scratch/prd128/diff-20261114-1610/`。自己点検と最終core/CPUの根拠を渡し、CUDA4構成の根拠を追送してから最終判定を受ける。コードはレビューへ出した版に固定する。

- 文書追随の最終補正: 212 §5.1の既存の「Atariの画像入力の費用は未測定」は、同じ文書の§5.2に測定結果があるため、日英とも次節の測定結果を参照する文へ直した。数値の新しい主張は足さず、今回追随する文書の正確性として1文を補正した。翻訳と元文書を同時に直し、差分と確認済みblobをレビューへ追送する。先に記録したdrift候補はこの補正で解消した。

- 最終のステージ5比較はCPU30・CUDA4の全34構成で成功した（`acceptance/stage5-summary.json`）。固定した `inputs/cases.json` の構成集合との完全一致を確認し、全973,936 scalar点・設定ファイル・設定JSON・WARN/ERRORが基準と一致した。成分の `stage5-cpu-r2-summary.json`・`stage5-gpu-r2-summary.json` も保持する。

- 実装レビュー初回は計画OK・実装差し戻し1件（`review-20261114-1622-impl.md`）。PRD §7受入7が指定する `Greedyy` の入力漏れで、拒否条件自体への指摘はなかった。既存の生成入力へ1値を追加し、ConfigDataとDTO直接構築の両方でD4の指定文言を確かめる。本体コードは変更しない。範囲外の検出と開発者への確認事項はなし。
- 差し戻し対応の検証: Debugのcore test targetを再ビルドし（`stage5-review-fix-build.log`）、`[prd128]` は240 assertions、失敗0・skip0で成功した（`stage5-review-fix-characterization.xml`、0.816秒）。方策の16 assertionsを成功内容付きでも実行し、`Greedyy` が両構築経路でD4の指定文言により拒否されることを確認した（`stage5-review-fix-policy.log`）。本体は受入済みの版のままで、既定suiteと34構成を再実行する変更はない。

## 決まらなかったこと（単独レビュー完了時）

ユーザー判断の未決はなし。U1 は群ごとの照合を上に記録済み。U2 は CPU の旧挙動を実測し、PRD へ反映・レビュー済み。U3 は 1 組で 6 群の有限点を確認済み。

## 完了報告（単独レビュー完了時）

- 実施内容: ステージ0〜5を完了した。ExperienceSamplesのフィールド走査、設定読み込み、内的報酬の所有と処理、診断の共通部分とprobe RNG、DefaultDQN/Rainbowの部品構築を整理し、日英の設計文書と再実行用の検証資材を追随した。PRD121の足す先はG2表へ記録済み。
- 変更ファイル: 主な変更に列挙した12コード/テスト、日英5対、PRDの実測補正、実装計画、検証資材3ファイル、対象5課題と一覧・トピック別まとめ。並行作業の変更を保持した。
- 検証: Debug・Releaseビルド、既定core 179,995 assertions（基準と同じskip1）、最終34構成973,936点と全成果物、旧RND checkpoint継続、3種類の性能の受入、指標順序の破壊検査が成功した。レビュー後に指定入力だけを補った関連240 assertionsも成功。既定suiteの件数はこの入力追加前の実行結果である。必要な検証の未実施なし。
- レビュー結果: 実装計画と実装を同じ独立レビューで2往復し、最終はともにOK（`review-20261114-1627-impl.md`）。初回の指定入力漏れ1件を修正した。再確認では両構築経路の指定文言・実行記録、既定suiteと34構成の結果を継承できること、記録の整合とLF/BOMを確認した。根拠が無く見送った指摘なし。
- 決まらなかったこと: なし。記録する課題の候補はありません。
- 判断してほしいこと: なし。対象5課題のAI判定を2026-11-14付で「残件無し」へ更新した。状態は「対応中」のままとし、解決の判断はコミットの準備で行う。
- 工程末の記録確認: `reports/tools/wiki_support.py maps` は終了値0、未解決のPRD参照なし。INDEXと318_rl_core・321_agent_core・324_dqnの課題区画を再生成した（`final-maps.log`）。36ファイルの形式検査はerrorsなし、対象の `git diff --check 2f3e8d81` も終了値0。コミットの準備・ステージ・commit・pushは行っていない。

## 追加の2モデルレビュー（2026-11-14）

- 開発者の追加指示で、同じ依頼をCodexとClaudeへ渡した。PRDと実装計画/実装を観点別に分けた。依頼は `request-20261114-1635-prd.md`・`request-20261114-1635-impl.md`、差分は `diff-20261114-1635/`。
- CodexはPRD・計画・実装をOKとした（`review-20261114-1638-prd.md`・`review-20261114-1638-impl.md`）。ClaudeはPRDのU2未確認表記1件、計画の検証コマンド不足1件、実装の定数参照とコメントの意図/所有者の2件を不足とした（`review-20261114-1636-prd-claude.md`・`review-20261114-1649-impl-claude.md`）。モデルごとの原文を保持し、指摘を省かず修正対象へ積んだ。
- 修正: PRD §2.2とD7の理由をU2の実測記録へそろえ、冒頭に改訂を記録。計画の検証に全モードの起動コマンドを追加した。コードの枠はAGENTS.mdの指定の `bash` を使い、PowerShellで実行することを明記した。snapshotとprepareは既存基準/入力を保持する条件を付け、反復/比較/継続/性能/suiteは未使用labelを日時から作る。
- 修正: QR/IQNの文言は既存のquantile mode定数、Munchausen/weight normの期待サイズは比較と同じCount/PackCountから組み立てる。直接Learnerを構築するテストの37か所は、ファイルごとの `static constexpr anet::seed_t kTestAgentSeed = 6000` へ寄せた。元の値と同じで、抽選列は変えない。
- 修正: 既知の診断の鍵と計算要求を分ける意図をCoordinatorの関数へ、報酬へ加算する意図をAgentの1行へ戻した。移したcredit・保存/復元の所有者のコメント4か所をCoordinatorへそろえた。コードの変更は定数の参照とコメントだけで、計算式・条件・実行順は変えていない。
- 変更範囲外の検出1件: readbackでMunchausen診断を載せる既存処理（`dqn_based_agent.cpp`、`munchausen_diagnostics.numel() != 5` と対応する期待値の文言）が数値を直書きしている。今回は変更せず、工程末に扱いを選ぶ候補とする。新規の課題はまだ起票しない。
- 修正後の検証: Debug・Release全ビルドは終了値0（`cc-fix-debug-build.log`・`cc-fix-release-build.log`）。計画に記載した関連テストのコマンドは15,233 assertions、失敗0・エラー0・skip0で成功（`cc-fix-related.xml`、40.062秒）。既存のscalar異常shape検査も含み、期待サイズの文言を維持している。PRDのリンク/状態行/重複番号/U2、seed37か所、所有者コメントの検査は `cc-fix-document-check.json` でpassed=true。計画の11コマンドはPowerShell構文検査に成功し、snapshot/prepareの既存資材保持の分岐も実行確認した。全モードのCLIは `--help` でも確認済み。36ファイルのLF/BOMと対象のdiff --checkも成功。
- 検証結果の版: 34構成・既定suite・旧checkpoint・性能の受入は追加レビューの修正前の実行体による結果である。今回の本体変更は同じ値を持つ定数からのメッセージ生成とコメントに限り、正常入力の数値処理・条件・順序・seed・保存形式は変わらない。修正後は全ビルドと関連15,233 assertionsを実施し、正常Run全34構成と既定suite・性能の再測定は繰り返していない。
- PRDの2モデル再レビューは両方OK（`review-20261114-1654-prd.md`・`review-20261114-1653-prd-claude.md`）。U2の実測・現在の説明・当時の対応表の区別を確認した。
- 修正後の実装計画/実装も両方OK（Codex `review-20261114-1700-impl.md`、Claude `review-20261114-1657-impl-claude.md`）。ただしClaudeは日英設計文書の全文を補読していないと報告したため、未読部分の確認を完了する追加依頼を出した。この時点では範囲外の検出1件が候補として残っていた。
- 全文確認の補完: `request-20261114-1701-impl.md` に日英10文書の全4,983行を同梱した。元ファイルの行数とSHA-256は同名の `sources.json` に保持。両モデルへ同じ依頼を渡し、Codexは既読の全文読解を継承、Claudeは同梱した全文で確認を完了した。最終判定は計画/実装とも両方OK（Codex `review-20261114-1702-impl.md`、Claude `review-20261114-1701-impl-claude.md`）。新たな不足はない。
- 全文確認で追加された範囲外の検出1件: `docs/design/200_dqn_agents.en.md` §9.1 の665・667行に、step diagnosticsの同じ説明が2文ある。日本語版は1文で、基準コミットから存在する翻訳の重複としてClaudeが報告した。今回は変更せず、先のreadbackの直書きと合わせて2件の候補として提示する。両方とも今回の判定には影響しない。
- 2モデルレビューの結果: Claudeの不足4項目（PRD1・計画1・実装2）を修正し、PRDは2往復、計画/実装は全文確認の補完を含む3往復で最終は両モデルともOK。修正後の全ビルド・関連15,233 assertions・形式/文書/コマンド検査が成功した。必要な読解の未実施はない。ClaudeのSDK最終表示は実装セッション8.4859932 USD、PRDセッション1.7086877 USD（2セッション計10.1946809 USD）。
- 候補の提示時点の判断待ち: 変更範囲外の2件をこの場で直すか、課題へ記録するか、今回は見送るかを工程末に質問した。この時点では2件とも未対応で、起票・ステージ・commit・pushは行っていない。

## 選択された2件の修正（2026-11-14）

- 開発者は「2件とも修正して再レビューする」を選択した。readback側のMunchausen診断サイズの比較/期待値の文言を既存の `MunchausenMetric::Count` へそろえ、200英語版§9.1の重複文1つを削除する。前のレビューでは範囲外だった2件を、今回の明示された変更範囲へ加える。
- readbackの変更は現在の値5と同じ定数を参照し、正常入力の処理とエラー文を維持する。翻訳の変更は同じ説明の重複だけを除き、日本語正本と翻訳元情報は変えない。関連する既存テスト、日英対応とリンク/翻訳元情報、Debug・Releaseビルドを検証し、同じ両モデルへ出し直す。
- 選択された2件を修正した。Debug・Release全ビルドが終了値0（`cc-selected-debug-build.log`・`cc-selected-release-build.log`）。`anet-core-test.exe '[dqn][munchausen],[dqn][scalar]' --reporter junit --out .scratch/prd128/cc-selected-related.xml` は終了値0、7,958 assertions、失敗/エラー/skip0、29.320秒。通常ログも `cc-selected-related.log` へ保存した。
- 200の日英の見出し53、コードフェンス28、表の行80が一致し、コードブロック一致・日本語残りなし・リンク切れなし（`cc-selected-translation-check.json`）。日本語正本のblobは前回と同じ `92ff82582888b655469a126529b65eb7cc68f374` で、英語版の翻訳元情報と一致する。残す1文の内容は対応する日本語1文と一致し、glossaryの追加・翻訳元情報の変更はない。36ファイルの形式検査と対象のdiff --checkも成功。
- 以前に範囲外だった2件は、今回の選択で修正対象となり対応済み。対応の充足を同じ両モデルへ確認依頼する。以前の「範囲外」として修正の確認から除外しない。新規の課題は起票していない。
- 選択された2件の最終レビューは、Codex・Claudeとも計画/実装がOK（`review-20261114-1745-impl.md`・`review-20261114-1742-impl-claude.md`）。比較とエラー文の同じCount参照、日本語版と同じ1文への英訳の整理、検証結果と対応記録の追随を確認した。新しい不足・範囲外の検出・開発者への確認事項はない。

## 完了報告（追加の2モデルレビューと選択2件の反映後）

- 実施内容: 追加レビューの不足4項目と、開発者が選択した範囲外2件を修正し、両モデルへ再レビューした。日英10文書の全文確認も双方で完了した。
- 変更ファイル: PRD128の元の変更に加え、検証コマンドを実装計画へ追加し、定数参照/テストseedとコメントを修正した。最後の2件では `dqn_based_agent.cpp` と `200_dqn_agents.en.md`、実装計画、対応課題の記録を追随した。
- 検証: 最後の修正後のDebug・Release全ビルド、Munchausen/scalar関連7,958 assertions、翻訳対応・形式・差分検査が成功した。以前の追加修正後の15,233 assertionsと、さらに前の34構成/既定suite/checkpoint/性能の結果は、それぞれの版の記録として保持する。最後の修正後に長時間検証を再実行したとは扱わない。
- レビュー結果: PRDは2モデルともOKを維持、実装計画/実装は今回の再レビューでも2モデルともOK。根拠が無く見送った指摘なし。Claudeが実行できなかった機械検査は、呼び出し元とCodex側の検査結果で確認している。
- 決まらなかったこと: なし。以前の範囲外2件は修正済みで、記録する課題の候補はありません。
- 判断してほしいこと: なし。対象5課題の残件無し・対応中の記録を維持する。起票・ステージ・commit・pushは行っていない。
- ClaudeのSDK最終表示: 実装セッション11.997343 USD、PRDセッション1.7086877 USD。2セッション計13.7060307 USD（約13.71 USD）。

## 完了後のかるいレビューでの直し（2026-11-14）

- `Learner::PrepareDiagnosticResources` の noisy の probe のエラー文から「a dedicated RNG」を除き、「requires a resolved sample forward config.」にした。D13 で Learner が乱数を必ず持つようになり、欠けうるのは forward の設定だけだからである（attention の同じ形の文は実装で除いていた）。この文言を断定するテスト・文書は無い（検索 0 件）。判定・条件・正常経路は変えていない。ビルドは開発者が行う。
