# RND の統計の性能改善と方式の整理 実装メモ

## 概要

正本は [PRD130](0130_rnd_estimate_statistics_perf_10rfd.md) の 2026-11-16 の確定事項 D1〜D18、実装契約、受入である。ステージ 1・2 の実装、機能・性能受入 1〜15、比較基準の保存、日英の文書の追随をすべて終えた。主・副・Taiga の最終性能比較も合格し、詳細を末尾に記録した。測定は終了しており、評価 Run を再開できることを開発者へ通知済み。評価 Run の PAUSE・再開操作は行っていない。

性能を変えるステージ 1 と、値を変えないステージ 2 を分ける。ステージ 1 の受入を通して実行体を保存するまで、ステージ 2 に入らない。計画作成時の基準コミットは `69bbc3b1e8bfc2d97039041519242f8fa3fa229b`。比較基準はコミットそのものではなく、実装着手時の作業ツリーから保存した実行体と設定である。

## 未決事項監査

ユーザー判断が必要なブロッカー: 0。

以下は実装着手前に行った監査の結果である。現在の実装と検証の到達点は末尾に記録する。

| 分類 | 決定点と根拠 |
| --- | --- |
| repo evidenceで解決済み | 現在の `intrinsic_rnd.cpp` は計測区間と試作スイッチを持つ。試作が効くのは whiten の `Estimate` と `Learn` で、`EstimateWithoutUpdate` はまだ CPU 正規化である。D1・D2・D14 に従い 3 入口を揃える |
| repo evidenceで解決済み | `RunningMeanStd::Update` は scalar の全要素集計と batch 方向の集計を分け、CPU・倍精度で合流する。D7 は後者の集計だけを替える。累積の不偏分散と `TargetFeatureVariance` の母分散への換算は維持する |
| repo evidenceで解決済み | `RunningStdSingleTensorNormalizer::NormalizeAndUpdateStats` に robust 更新と診断値の更新がある。D4 はこの更新処理を共通の口へ分ける。Q 入力の呼び出しを変える必要はない |
| repo evidenceで解決済み | `intrinsic_test.cpp` に warm-up、カテゴリ ID、device、保存継続、読むだけの採点、写しの独立性の既存テストがある。PRD §7 の条件に合うものを再利用し、混合辞書と前後比較を補う |
| 2ximplへ固定 | 基準実行体の保存、テスト、ステージ 1 の受入、次の基準の保存、ステージ 2 の整理、再受入の順とする。採取器は `testdata/prd130/` に置き、既存の PRD116・123・128 の資材は読み取りで再利用する |
| 2ximplへ固定 | RND の方式の型は PRD §6.2 の名前と既存の intrinsic_rnd グループ内に置く。派生する方式の状態はその型が所有する。新しい設定や外部向けの口は `ObservationNormalizer::UpdateStats` 以外に足さない |
| PRD範囲外 | Clone の高速化、可塑性メトリクスの修正、補助学習の並列処理化、Q 入力の正規化の呼び出しの変更、NN 内の前処理の変更、NGU の実装 |

`CONTEXT.md` に追加する用語はない。既存の方式の意味と、PRD の新しい型名に衝突は見つからなかった。ADR は D18 の決定どおり作らない。

## 主な変更

### 1. 実装前に比較の入力と実行体を固定する

1. 現在の staged・unstaged・untracked の一覧と、対象ファイルの差分を保存する。既存の並行変更を保持する。とくに `intrinsic_rnd.cpp` の試作と計測を、無関係な変更として消さない。
2. Debug と Release を MSVC 初期化経由でビルドし、Runner・core テスト・依存 DLL・設定・コミット ID・対象ソースのハッシュを `.scratch/prd130/acceptance/before/` へ保存する。基準 Run では試作環境変数を無効にし、その子プロセスの環境も記録する。既存の stage 0 と試作の結果は比較の参考であり、新しい受入の基準として流用しない。
3. `[scaler]`・`[prd116]` を回して現状の成否と丸めに敏感な断定を洗う。PRD127 の既存テストの断定を緩めない。
4. LunarLander の既定連鎖を凍結し、PRD §7 の (a) 無効、(b) whiten 累積、(c) whiten EMA、(d) variance_feedback、(e) collect_errors を生成する。Taiga の 8 値は実装着手時の `Atari.txt` と照合し、LunarLander のネットワーク設定を維持したまま明示する。
5. seed 1・決定的 backend・100k exp step で、無効と variance_feedback の前比較、whiten CUDA の 2 回の再現確認、累積・EMA・variance_feedback の checkpoint 採取を行う。既定の RND warm-up は 200k なので、100k の Run だけでは warm-up 明けを検証できない。受入の Run の条件は変えず、境界と warm-up 明けの倍率・更新は単体テストで補う。
6. GridMaze CPU・256 lane の基準時間と Tracy を保存する。性能測定は build・他の Runner・テストを同時に走らせずに行う。

### 2. ステージ 1: 性能の直し

- `scaler.hpp`・`scaler_impl.hpp/.cpp`: `void UpdateStats(const TensorDict&)` と、内部の単一 tensor 用の同じ役割の口を足す。辞書の鍵の取り扱いは既存の正規化に揃える。固定値の正規化は更新する統計を持たず、何も更新しない。統計つきは既存の robust 更新を共通化し、外れ値の診断を維持する。`NormalizeAndUpdateStats` は更新のあとに正規化と clip の割合の計測を行う。`UpdateStats` は正規化も clip の割合の更新も行わない。
- 統計つき正規化の平均と標準偏差の単精度の写しは、統計更新時に、その入力の device に作る。`Normalize` は同じ device の写しを読む。写しが無いときと device が異なるときは局所の値を作って使い、共有状態を書かない。Load・Reset で写しを捨て、保存の中身へ加えない。centering が無い構成でも既存の式を維持する。
- `Normalize` では clip を行い、clip の割合を数えない。更新を伴う正規化では今までどおり数え、診断値を保存する。固定値の正規化と統計つきの正規化の両方で、値と診断値の意味を保つ。
- `RunningMeanStd::Update` の batch 方向だけ、入力の精度で平均を計算し、平均との差の二乗の平均から母分散を計算する。既存の CPU・倍精度への転送、M2 と件数の合流、全要素の集計は変えない。空の入力と 1 件の入力の扱いも維持する。
- `intrinsic_rnd.cpp`: whiten の採点・読むだけの採点・補助学習を、観測の計算 device への転送、連続 byte の浮動小数点化、正規化の順にする。離散の byte・整数の ID は変換対象へ加えない。観測と target の出力の batch の集計は計算 device、誤差の検査・倍率・誤差の統計・累積・診断値の組み立ては CPU に残す。whiten の観測の統計更新は `UpdateStats` を使う。
- variance_feedback の正規化と逐次の更新は CPU のまま維持する。共通の target 出力の batch 集計による `target_feature_var` の丸め差だけを許す。
- 試作環境変数 `ANET_PRD130_DEVICE_NORM`、専用の通知、試作の CPU 正規化の選択経路を外す。既存の通常のログ・WARN・ERROR の文言・順・回数は維持する。
- 計測区間の名前を維持する。`observation_transfer` は計算 device への転送になり、`input_transfer` には転送を残さない。分けた統計更新にも計測を置く。同期の読み戻しの一括化、非同期転送、固定メモリは入れない。

### 3. ステージ 1 を受け入れ、次の比較基準を保存する

PRD §7 の 1〜10 をすべて確認してから、Debug・Release・設定を `.scratch/prd130/acceptance-cuda/stage1-accepted/package/` へ保存し、同階層の `acceptance.json` から同じ実行体で採った機能検証の Run と checkpoint へ案内する。ステージ 1 の途中の実行体を、ステージ 2 の比較に混ぜない。受入に届かないときは同じステージで原因を調べ、契約を変えなければ直せない場合は止めて報告する。

### 4. ステージ 2: 値を変えない統計方式の整理

- `intrinsic_rnd.hpp/.cpp` の既存のグループ内で `RndStatistics` の口と `WhitenStatistics`・`VarianceFeedbackStatistics` の実装を持ち、構築時に 1 つ選ぶ。正規化の device・正規化時の統計更新・warm-up の判定単位・補助学習の休止条件は方式の型が持つ。
- `RndErrorStatistics` を口とし、`CumulativeErrorStatistics` と `EmaErrorStatistics` に状態と計算を分ける。EMA の順序、補正、件数と保存の並びを変えない。
- whiten は、読むだけの採点、統計更新、補助学習を分ける。採点から統計更新へ誤差・target 出力・観測を渡す。採点と補助学習の間で正規化済み入力や forward の結果を使い回さない。
- variance_feedback は lane 順の正規化と統計更新を 1 つの流れに置き、誤差更新の件数を外の呼び出し順から逆算する前提を無くす。lane ごとの warm-up と正規化は維持し、whiten の 3 単位へ無理に載せない。
- 正規化のあとの target/predictor forward、誤差の計算、有限性の検査を共通化する。採点は勾配無し、補助学習は predictor の forward から勾配有効、読むだけの採点の非有限値は既存の WARN と NaN を保つ。共通化で ERROR と WARN を取り違えない。
- Save・Load・Clone・Diagnostics は選んだ型へ委ね、方式を確かめる分岐を構築時だけにする。書式識別子・統計の直列化順・optimizer options の置換通知・score_count・乱数と forward の回数を維持する。写しは深い複製とし、元と状態を共有しない。

### 5. 検証資材と文書

- `.scratch/prd130/stage0.py` を材料に、`core/anet-core/testdata/prd130/run_performance.py` を作る。配置が変わるため repo root の解決を直す。試作の群・環境変数と試作用の検査を除き、主・副・GridMaze、実行体を交互に測る比較、Taiga を扱う。PRD116 の採取の共通処理を借り、原本は変えない。
- `core/anet-core/testdata/prd130/run_validation.py` に保存・設定の凍結・100k Run・scalar checksum・checkpoint 継続比較の入口を置く。PRD123・128 の既存の採り方と除外規則を再利用し、過去の保存先へ書かない。除外した tag の一覧、比較した点数、設定ハッシュ、実行体ハッシュと終了値を結果に残す。
- 同ディレクトリに `README.md` を作り、基準の保存、各段階の入口、比較条件、Tracy と Nsight の採り方、結果の保存先と失敗時の再開位置を書く。未実行のコマンドを検証済みとして書かない。
- `docs/design/212_rnd_lifelong_novelty.jp.md` の §3.2・§3.6・§3.8・§5.1・§5.2 とテストの案内を追随させる。`200_dqn_agents.jp.md` は推定器・読むだけの採点の説明を点検し、変わる記述だけ直す。英語版は `anet-translate-docs` の手順による。設定・CONTEXT・ADR は変更しない。
- 実装のレビュー前に各課題の対応内容へ実装と確認結果を追記し、レビュー後に残件の判定を記録する。段取り整理だけの段階では実装済みとして判定しない。

## テスト

Public interface / surface は、`ObservationNormalizer` の Factory・UpdateStats・Normalize・NormalizeAndUpdateStats・GetTensor/GetScalar・Save/Load/Reset、`RunningMeanStd` の集計結果、`LifelongNoveltyEstimator` の採点・読むだけの採点・Learn・Diagnostics・Save/Load・Clone、および Runner の出荷設定と checkpoint 継続である。private 関数や写しのメンバを読むテストは足さない。

後続の実装は、次の順に 1 behavior ずつ RED → 最小実装 → GREEN とする。GREEN のあとにだけ整理し、関連テストを回す。すでに通る既存の保護テストは、RED の記録を作るために壊さない。

| 順 | 観測可能な behavior と TDD の単位 |
| --- | --- |
| 1 | 最初の 1 テストは Factory で混合辞書用の正規化を作り、`UpdateStats` → 統計の取得 → `Normalize` を通す。通常更新と同じ統計・正規化値になり、clip の割合は変えない。この経路を通す最小の実装を行う |
| 2 | robust 更新の warm-up・正常行・全行除外、固定値と素通しの更新を、1 ケースずつ追加する。既存の外れ値の割合と鍵の扱いを保つ |
| 3 | 更新後の統計、Load 直後の統計、更新したときと異なる device の入力の各 1 テストを順に追加し、写しを使っても正規化の値が正しいことを確認する。Reset 後も既存の初期値へ戻る。読み取り時に写しを書かないことはレビューで確認する |
| 4 | 統計更新しない正規化の値と、更新を伴う正規化の clip 診断をテストする。clip の割合を数えないことはコードと Tracy で確認する。計測だけの差に、失敗を捏造したテストを作らない |
| 5 | batch 集計を CPU・倍精度の独立した参照計算に照らす。複数 batch・空・1 件を 1 ケースずつ追加する。全要素集計の値が維持される既存テストを保護に使う |
| 6 | RND の混合辞書（連続 byte・離散値・浮動小数点）を公開 Factory から通し、参照計算との値の近さを確かめる。CPU と利用可能な CUDA、累積と EMA、warm-up の前・境界・後を順に追加する。既存のネットワーク・保存の読出し手段を借り、本体へテスト専用 API は足さない |
| 7 | 読むだけの採点が直後の 1 件の採点と一致する 3 方式、読み取りを挟む学習・診断・保存が変わらない既存テストを、断定を変えずに回す。ステージ 1 を受け入れる |
| 8 | ステージ 2 の前に、同じ混合辞書で採点・Learn・保存継続・Clone の特性を固定する。方式を 1 つずつ分け、分けるたびに GREEN を保つ。既存挙動の整理は新しい挙動の追加ではないため、特性テストと前後の実行体の一致を使う |

値の近さは PRD §7 の 5 のとおり、観測統計の平均・標準偏差が相対 1e-4 以内、α が絶対 1e-3 以内とする。観測統計の参照値が 0 の要素は一致を確認する。単に失敗した期待値を実装結果で更新せず、集計式の丸めが原因と確かめた断定だけに許容差を付ける。

## 検証

以下は PRD 全体の検証手順である。受入 1〜15 はすべて完了した。最後の性能比較 13 では本体を変えず、受入済みの固定 Release 実行体を使ったため build は行っていない。過去の build は AGENTS.md の「AI エージェントでのビルド注意事項」の MSVC 初期化経由で Debug・Release を行った。Runner 起動前にユーザーガイドの指定 3 節を読んだ。実施結果は末尾に記録する。

リポジトリルートの PowerShell から使うビルドコマンドは次のとおり。AGENTS.md の既存手順による。今回の実行では並列数を 4 に制限し、標準出力と標準エラーを専用のログへ保存する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release'
```

| PRD 受入 | 実施する確認と合格条件 |
| --- | --- |
| 1 | 主（128 lane・11 bat の 27 番を凍結）と副（64 lane・PRD116 条件）。同じ実行体で off/on/on/off を 3 巡、各群 6 本。同じ経験数の中央値の比が主 ≤1.15、副 ≤1.25。両群とも可塑性メトリクスの tag を測定用の設定から外し、実効の定義とログで計算が無いことを確認する |
| 2 | GridMaze CPU・256 lane の RND ありを before/after/after/before で 3 巡。変更後の中央値が変更前以下、または中央値の差が変更前の最小〜最大の幅の内。両実行体の Tracy で target 出力の `batch_moments` の 1 回あたりの時間が短い |
| 3 | 主の RND の Tracy を別採取し、統計内に捨てている正規化が無いことを確認する。採点・補助学習・転送・集計・診断の内訳と、旧区間との対応を記録する |
| 4 | before と stage1 の LunarLander 無効・variance_feedback を、seed 1・決定的 backend・100k で比較。全 scalar 点の checksum 一致。時間・資源だけ除外し、variance_feedback だけ `56_agent_in/14_target_feature_var` も除く |
| 5 | 上記の混合辞書の単体テストを CPU と CUDA で実行。統計相対 1e-4、α 絶対 1e-3 の範囲内 |
| 6 | before と stage1 それぞれで LunarLander 既定 RND・CUDA・seed 1・決定的 backend・100k を 2 回実行。before が一致する場合は stage1 の全 scalar も時間・資源以外が一致する。before から再現しないときだけ、その事実を記録して 11 の比較を CPU にする。変更後だけ再現しないときは原因を直し、直せなければ停止する |
| 7 | `[prd127]` の既存の読み取り不変性の断定を維持。累積・EMA・variance_feedback で、読み取りの倍率と直後の同じ 1 件の採点の倍率が一致する |
| 8 | before の累積・EMA・variance_feedback の 100k 終了 checkpoint を、stage1 が同設定で読み、続きの 100k を ERROR 無しで完走。継続値の同一性は要求しない |
| 9 | 追跡対象全体で試作スイッチを検索し、参照が本 PRD と本計画だけになる。採取器にも試作の切り替えを残さない |
| 10 | core 既定 suite と `[prd116]`・`[scaler]`・`[shipped_config]`、追加テストが通る。関連する `[prd123]`・`[prd127]` も確認する |
| 11 | stage1 と stage2 の LunarLander 5 構成（無効・累積・EMA・variance_feedback・collect_errors）。seed 1・決定的 backend・100k。全 scalar 点が時間・資源以外すべて一致。ここでは target_feature_var を除外しない |
| 12 | stage1 の累積・EMA・variance_feedback の 100k 終了 checkpoint を、stage1 と stage2 がそれぞれ読み、続きの 100k の全 scalar が 11 と同じ除外だけで一致 |
| 13 | 主の既定 RND、副の既定 RND、副の Taiga を stage1/stage2/stage2/stage1 で各 3 巡。群ごとの中央値の差が、stage1 の最小〜最大の幅の内 |
| 14 | estimator の方式分岐が構築時の 1 か所だけであることをコードレビューで確認し、`feedback_stats_` の旧メンバの参照が intrinsic_rnd の 2 ファイルに無いことを補助検索で確認する |
| 15 | 10 と同じテストを stage2 でも実行する |

速さの測定は評価・描画・画像出力・終了時 checkpoint を止め、Q warm-up 20,000、RND warm-up 12,800、torch thread 1 とし、最速群が約 20 秒の経験数を pilot で選んで固定する。主の off は RND の profile と intrinsic の上書き・価値分解のメトリクスも外す。主の on は価値分解を維持する。副の Taiga は速度の中立性の検査に使い、既定 RND の改善比には混ぜない。Nsight の再採取は転送や同期の原因が不明な場合に行い、Tracy 付きの時間を通常の性能比に混ぜない。

長時間の build・test・Run の標準出力と標準エラーは `.scratch/prd130/acceptance/` の label ごとのログへ保存し、画面には成否・件数・保存先だけを返す。既存 label を上書きしない。結果の raw・設定・実行体・checkpoint・比較 summary を同じ label で追えるようにする。

## 前提

- ステージ 1・2 の機能・性能受入 1〜15 は完了した。最後の性能比較 13 は、開発者が改めて評価 Run の PAUSE 完了を明示したあとで実行した。測定中は build・test・別の Run を重ねていない。
- PRD §8 は未決無し。D1〜D18 を再グリルせず、§6.2 の型名と置き場を使う。
- stage 0 の分析は済んでいる。試作の比は別の基準実行体との参考値であり、受入の代わりにならない。
- 現用の設定・メトリクス・保存形式・公開の新しさの口・通常の通知と計測区間名・乱数の消費順・forward の回数は PRD §6.1 の例外以外維持する。
- stage1 の丸め差と stage2 の完全一致を区別する。再現性の例外は PRD §7 の 6 だけに限定する。
- CUDA・Atari ROM・Tracy は実際の Run と採取で稼働を確認した。追加の Nsight 採取は不要と判断し、CLI の利用準備までで、再採取はしていない。利用していない検証を成功扱いしない。

## 経過と確認結果

### 性能比較前までの指示を受けた時点のステージ 2 の実装順序と停止位置

- 開発者の「性能比較前まで実行」により、受入 11・12・14・15 と文書の追随まで進め、13 の性能比較を行わず停止する。利用者の評価 Run は操作しない。ログは `acceptance/stage2-*`、通常 Run は `acceptance-cuda/stage2/`、基準側の checkpoint 継続は `acceptance-cuda/stage1-reference/` に分ける。
- 公開 Factory の混合辞書で、採点・Learn・Clone・保存継続の同一性を先に固定する。これは新しい挙動ではなく整理の保護テストなので、現在の実装で GREEN を確認し、意図して断定を壊して RED を作らない。誤差統計の累積/EMA の分離、方式の分離、共通 forward の順に整理し、その都度関連テストを通す。
- `RndErrorStatistics` を口にし、累積・EMA の計算と保存順を各型へ移す。`RndStatistics` とその実装は `intrinsic_rnd` 内の private な補助型とする。whiten は読む採点と統計更新を分け、feedback は観測更新前の件数を同じ採点の流れで保持し、誤差更新時に更新後の件数から逆算しない。
- 各方式は正規化の device、byte の変換と正規化、warm-up と学習休止、誤差からの倍率、保存・読み込み・深い複製と統計の診断を所有する。方式の選択は estimator の構築時だけとし、既存の target 出力の共通統計は estimator に残す。共通 forward は採点/読むだけ/学習の用途で勾配と非有限値の扱いを保つ。FULL の計測名は分割前の名前を維持する用途だけに使う。新しい device 転送・乱数・forward は足さない。
- 現用 Runner が保存済みステージ 1 と同じ時点で `stage1-reference` を保存し、ステージ 1 の累積・EMA・feedback の 100k checkpoint から基準の継続を開始した。整理後も同じ checkpoint を読み、時間・資源以外の全 scalar（target_feature_var を含む）の一致を確かめる。

### ステージ 1 の経過

- 2026-11-16: 「130 段取り整理だけ」の指示で計画を作成。PRD の確定事項と受入、scaler と intrinsic_rnd の経路、既存のテストと採取器、CONTEXT、関連文書を照合した。ユーザー判断が必要なブロッカーは 0。
- 計画保存時の変更の対象は本計画と、計画保存に伴う課題の工程記録である。既存の未コミット変更は保持する。
- build・テスト・Run・性能測定は未実施。段取り整理だけという依頼範囲による。上表は実装時に満たす確認であり、達成済みの結果ではない。
- 自己点検: 実装品質確認観点の全項目を適用範囲に従って点検。計画の前提、公開の確認面、テスト順、PRD 受入 1〜15 の対応、未実施の検証を確認した。実装だけに適用する項目は、今回は実装していないため対象外。
- 計画レビュー 1 往復目: ビルドコマンドの記載不足 1 件で差し戻し。AGENTS.md の MSVC 初期化経由の Debug・Release コマンドを検証の節へ追記した。コマンドの実行は行っていない。
- 計画レビュー 2 往復目: OK。追加したコマンド、PRD 受入 1〜15 の対応、ステージ間の比較基準を確認した。原文は `.scratch/prd130/review-20261116-0137.md` と `.scratch/prd130/review-20261116-0138.md`。
- 計画保存に伴い、課題 f9790d・0ddffa を調査中から対応中に変更し、備考へ段取り整理だけであることを記録した。`reports/tools/wiki_support.py maps` は成功し、課題一覧と 321_agent_core・324_dqn の課題の区画を更新した。実装の完了判定は書いていない。
- 計画の名前と所在、LF、適用する語の検査、対象の `git diff --check` を確認した。新しい起票候補はなし。段取り整理を完了し、実装開始の指示を待つ。
- 2026-11-16 続行: 開発者の「機能検証まで」の指示により、ステージ 1 の実装・機能検証を開始。評価 Run（着手時 PID 53272、AnetRLRunner_ab.exe）は保持し、性能測定と PAUSE 操作は行わない。ステージ 2 も開始しない。変更前の GridMaze の性能採取は、保存済み実行体で、ステージ 1 の性能測定時へ寄せる。
- 変更前の Debug・Release 全ビルドと `[scaler],[prd116]` が成功（2,161 assertions、失敗 0）。実行体と設定は `.scratch/prd130/acceptance/before/package/` に保存した。最初の API テストは `UpdateStats` 未実装のコンパイルエラーで RED を確認し、最小実装後に GREEN を確認した。
- 入力の shape が broadcast で通ると統計が進むことをテストで RED とし、既存の正規化の入力検査を統計更新と共有して更新前に拒否するようにした。その後の保護テストは 58 assertions 成功。統計の写しと clip の計数の分離は、それぞれ 100 assertions が成功した。写しの初回ビルドの Tensor 初期化の曖昧さは明示的な `torch::Tensor{}` に直し、初回の失敗ログも保持した。
- 機能検証の採取器を `testdata/prd130/run_validation.py` に作成。最初の CPU 指定の基準 Run は出荷の大きな NN で時間がかかったため、自分が起動した Run と採取器だけを停止した。途中の記録は `acceptance/before/` に保持した。比較は出荷の auto と同じ CUDA を明示し、`.scratch/prd130/acceptance-cuda/` の新しい出力先で保存済みの変更前の実行体から採り直している。基準と変更後の設定は同じ凍結ファイルを使う。
- batch 集計式の変更後の `[prd130],[scaler],[prd116]` は 2,219 assertions 成功。RND の配置変更前に、混合辞書と CPU・倍精度の参照計算を使うテストを追加し、CPU/CUDA・累積/EMA/collect_errors の 478 assertions が成功した。配置変更後にも同じ断定を使う。
- 設計文書は日本語版 212 と 200 を今回の実装に追随させ、PRD130 の検証資材へ案内を足した。英語版には着手前から翻訳元との差があり（212 は 95 行、200 は 2 行の差分）、全体の訳の確認はステージ 1 の後続作業に残す。英語版を今回更新したとは報告しない。
- ステージ 1 の Debug 全ビルドと `[prd130],[scaler],[prd116],[prd123],[prd127],[shipped_config]` が成功（8,926 assertions、失敗 0）。比較器の欠測を成功扱いするケースを RED とし、空の学習 scalar を拒否する修正後の Python 4 tests が GREEN。core 既定 suite、Release 全ビルド、100k Run の機能検証は進行中。
- ステージ 1 の Release 全ビルドも成功。独立レビューは計画 OK、実装は子プロセス環境の固定・記録の不足 1 件で差し戻し。親の `ANET_PRD` 系の一時スイッチを除外し、実際に渡した環境のハッシュとバックエンド関連の値を記録する処理を足した。環境の RED→GREEN を含む Python 5 tests が成功。進行中の変更前の採取は、起動時に旧版の採取器が試作スイッチを明示的に除いた実行である。変更後の採取は修正後の採取器で開始した。
- ステージ 1 のレビュー 2 往復目は計画・実装とも OK（原文 `.scratch/prd130/review-20261116-0229.md`）。その後、core 既定 suite は 195,067 assertions 中 24 失敗・1 skip。失敗は追加した RND の参照誤差の比較だけで、統計や α の比較、既存のテストには失敗が無かった。乱数 seed は 3498861754。
- 診断: 単独の関連テストは通るが、`[backend][env]` のあとに同じ参照テストを実行すると 24 失敗を短い経路で再現した（`reference-order-repro.xml`）。初期化テストが TF32 行列積を有効にするため、参照計算のテストの中だけ FP32 に揃え、終了時に以前の設定へ戻す guard を追加した。本体は変更していない。最小再現と、失敗時と同じ seed の core 既定 suite を再実行している。
- 精度の guard 追加後、上の最小再現 430 assertions は失敗 0。Release 全ビルドも成功。実行順序の検査は core 既定 suite の同 seed 再試験でも行う。
- RND 無効の変更前後の 100k Run は、時間・資源以外の 38,922 scalar 点の checksum が一致した（`acceptance-cuda/before-vs-stage1/off/comparison.json`）。
- 検証資材の自己点検で Run 名を `run_p130_` の形へ揃え、比較と checkpoint の参照は `inputs.json` の実行記録から行うようにした。初回の `p130_` の証拠は改名しない。記録したパスを参照するテストの RED→GREEN を含む Python 6 tests が成功。
- core 既定 suite の同 seed（3498861754）の再試験は 195,067 assertions、失敗 0、skip 1（`stage1-default-r2.xml`、679.197 秒）。skip は CUDA が利用できない環境だけで行う異常系の検査で、この環境では CUDA を利用できるためである。参照テストの実行順依存は解消した。本体と Runner の保存済み実行体は変更していない。
- ステージ 1 のレビュー 3 往復目は計画・実装とも OK（原文 `.scratch/prd130/review-20261116-0247.md`）。精度の guard、記録した Run パスの参照、core 既定 suite と Python 6 tests、無効群の 38,922 点の一致を追確認した。残る Run の結果も完了後に追確認へ渡す。
- variance_feedback の変更前後の 100k Run は、時間・資源と PRD D10 の target_feature_var を除く 43,150 scalar 点の checksum が一致した（`acceptance-cuda/before-vs-stage1/feedback/comparison.json`）。
- 変更前の 7 構成の 100k Run がすべて終了値 0、採取器の errors は空。CUDA・既定 RND・seed 1 の 2 回も、時間・資源以外の 43,447 scalar 点が一致した（`acceptance-cuda/before/cuda-repeat.json`）。変更前からの非再現による PRD §7 の 6 の例外には当たらない。変更後の反復と旧 checkpoint の継続を引き続き確認する。
- 保存済み stage1 の本体ソース（intrinsic_rnd.cpp、scaler_impl.cpp/.hpp、scaler.hpp）は現用ファイルと SHA256 が一致する。参照テストの精度の guard はテストだけの変更で、機能 Run の Runner は保存後に変更していない。今回の同時稼働下の Run の実時間は性能の受入には使わない。
- 変更後の 7 構成の 100k Run もすべて終了値 0、採取器の errors は空。CUDA・既定 RND・seed 1 の 2 回は、時間・資源以外の 43,465 scalar 点が一致した（`acceptance-cuda/stage1/cuda-repeat.json`）。PRD §7 の 6 を満たす。前後で値が変わりうる whiten を、変更前後の完全一致と取り違えていない。
- 変更前の累積・EMA・variance_feedback の 100k 終了 checkpoint を、変更後がそれぞれ同じ凍結設定で読み、続きの 100k を終了値 0 で完走した。入力 checkpoint のパス、設定ハッシュと変更前との一致、実行体ハッシュ、終了時 checkpoint を確認した。`acceptance-cuda/stage1/load-cumulative-ema-feedback-summary.json` の errors は空。17 本すべての Run ログに ERROR/FATAL の行は無い。継続値の同一性は受入条件に含まない。
- 最終確認: 保存済み stage1 の Debug・Release Runner は現用 Runner と SHA256 が一致する。稼働中の評価 Run（PID 53272、起動 2026-11-16 00:53:35）は操作せず、そのまま稼働している。性能測定・PAUSE 操作・ステージ 2 は行っていない。
- 機能検証の到達点: PRD §7 の 4〜10 は確認済み。1〜3 の性能受入と、その後のステージ 1 の受入確定・比較基準の保存、ステージ 2、英語版の追随は残る。今回の指示範囲である機能検証までで停止する。課題 f9790d の判定を残件有り（2026-11-16）とし、状態は対応中のままにした。課題 0ddffa の方式の整理は未実装。
- 自己点検: 実装品質確認観点の全項目を適用範囲に従って通し、機能の合格と未実施の性能受入を区別した。既存の読み取り不変性の断定は維持し、対象の差分の空白、文字コードと改行、試作識別子の参照先を確認した。新規の起票候補・追加の開発者判断は無い。wiki の派生区画の再生成は成功（INDEX・321_agent_core・324_dqn、unresolved_prds は空）。
- 最終レビュー: 計画・ステージ 1 の実装と機能検証はともに OK。比較・CUDA 反復・3 方式の checkpoint 継続と、残件の記録を追確認し、不足・修正要求・範囲外の検出・開発者への確認事項は無い。原文は `.scratch/prd130/review-20261116-0347.md`。性能受入の直前で停止し、評価 Run の PAUSE 完了の明示的な返事を受けてから後続へ進む。
- 性能受入の続行（2026-11-16）: 開発者の「PAUSEしました」の返事を受けた。PID 53272 は保持し、2 秒間の CPU 時間の増加が 0 であることを確認した。保存済み before/stage1 の Release Runner と変更前の設定を使い、主・副の off/on 3 巡、GridMaze の before/stage1 3 巡、別 Run の Tracy を順に採る。測定・標準出力・標準エラー・実効設定は `.scratch/prd130/performance-20261116-0612/` へ保存し、画面には進捗と成否・比・受入結果だけを返す。他の Runner・build・test を重ねない。
- 性能の採取器を `core/anet-core/testdata/prd130/run_performance.py` へ移した。stage0 の主の構成と Tracy の集計を移し、PRD116 の時計・設定の組み立て・保存を再利用した。試作の群と切り替え・通知の検査は持ち込まない。測定専用の設定コピーから可塑性メトリクスの 57 定義を除き、実効設定と Notifier のログでも不在を確認する。休止済み PID だけを共存させ、測定中の CPU 時間の増加が 1 秒を超えた採取は受入へ使わない。Tracy は別の port 18086 で測定用 Runner に接続し、休止中の利用者の Runner への接続を避ける。
- 採取器の TDD: 主の上限 1.15 を超えた比を不合格にするテストを未実装で RED→GREEN、経験数の不一致を拒否するテストを RED→GREEN、GridMaze が変更前の幅を超えて遅くなった場合を不合格にするテストを RED→GREEN とした。各ログは `acceptance/performance-*.log` に保存した。
- 性能採取器の追加レビューは計画・実装とも OK（原文 `.scratch/prd130/review-20261116-0612.md`）。性能の結果と README の追記は採取後の追確認へ残す。本体・機能テスト・保存済み Runner は変更していない。
- 主の 3 巡: RND なし中央値 19.852997 秒、RND あり 21.499999 秒、比 1.082960（上限 1.15）で合格。なしの幅は 19.553002〜20.113004 秒、ありは 21.362003〜21.624001 秒。`summary_main.json` と順序付き `measurements_main.json` に保存した。可塑性定義の無い実効設定・登録ログの検査も全本が通過している。
- 副の 3 巡: RND なし中央値 20.089498 秒、RND あり 22.048002 秒、比 1.097489（上限 1.25）で合格。なしの幅は 19.791002〜20.629002 秒、ありは 21.891005〜22.183999 秒。`summary_legacy.json` と順序付き `measurements_legacy.json` に保存した。
- GridMaze の初回 pilot は、採取器が Atari 専用の `greedy`・`greedy_single` の評価スケジュールを追加したため、未知の評価タグで構築時に停止した。Runner 本体の問題ではない。GridMaze の設定生成の回帰テストで RED を確認し、評価タグを Env ごとの既存定義へ限定したあと、Python 4 tests が GREEN。主・副の設定生成は変わらない。失敗した記録は最初の測定先に保持し、GridMaze だけ新しい `.scratch/prd130/performance-grid-r2/` で、同じ保存済み実行体から採り直す。
- GridMaze の再採取の 3 巡: 変更前中央値 20.613500 秒、変更後 20.414000 秒、差 -0.199501 秒で合格。変更前の幅は 20.375000〜20.895001 秒（幅 0.520001 秒）、変更後は 20.105000〜20.769999 秒。変更前後の ABBA 12 本はすべて正常終了し、`performance-grid-r2/summary_grid.json` と `measurements_grid.json` に保存した。target 出力の batch 集計時間は別 Run の Tracy で確認する。
- 主の最初の Tracy（`main_profile_on`）は Runner の終了値 0、8 秒の trace 保存まで終えたが、休止済み PID の CPU 時間が 61.85 秒中に 1.15625 秒増え、採取器の 1 秒の上限で停止した。この記録は受入へ使わない。評価 Run のログは 05:47 の末尾から進んでおらず、再度の 2 秒の CPU 時間の増加は 0 だった。判定の上限は変えず、未使用のケース名 `main_profile_on_r2` へ同じ `execute` を実行して主の Tracy を採り直し、そのあと GridMaze の前後を採る。初回の trace と実行記録はそのまま保持した。
- 主の再採取と GridMaze 前後の 3 本の Tracy は成功。主は完結した 76 intake で RND 全体 5.999638 ms/intake（採取 span の 5.594605%）、採点 2.167598 ms、補助学習 3.624247 ms。採点の観測転送 0.508344 ms、正規化 0.079168 ms、統計 0.424386 ms（うち観測統計 0.331477 ms）、target 出力統計 0.155937 ms、診断 0.135877 ms。補助学習の観測転送は 0.107720 ms、正規化 0.068648 ms、backward 1.273929 ms、optimizer step 1.267401 ms。いずれも CPU host 区間の inclusive 時間で、親子を足し合わせない。GPU kernel 時間とは呼ばない。
- 旧区間との対応: `observation_transfer` は観測の計算 device への転送に変わり、その後の `normalize` で変換と正規化を行う。旧 `input_transfer` の名前は保持し、採点 0.000849 ms、補助学習 0.000690 ms のほぼ素通しになっている。`statistics.observation_statistics` の中は旧 `NormalizeAndUpdateStats` から `UpdateStats` へ移り、Tracy の全該当パスに `::Normalize` は無い。観測の統計更新・target 出力の統計・診断の既存区間名は保っている。
- GridMaze の `target_feature_statistics > RunningMeanStd::Update.batch_moments` は、前後各 196 回の 1 回平均が 0.356280 ms から 0.021336 ms に短縮した。受入 2 の集計時間と受入 3 の不要な正規化の除去を確認した。転送や同期について未解明の原因は残らず、追加の Nsight 採取は行っていない。3 本の trace と包含関係の集計を、通常の 36 本の実時間へ混ぜていない。
- 採用した 36 本の本測定はすべて正常終了し、休止済み PID の CPU 時間の増加の最大は 0.875 秒で上限内。集約した根拠は `acceptance/performance-evidence.json`。2026-11-16 06:40 頃、全採取の終了と評価 Run を再開してよいことを開発者へ伝えた。Run 自体の PAUSE・再開操作は行っていない。
- 性能受入 1〜3 と既確認の機能受入 4〜10 が揃った。Debug・Release・core テスト・設定・本体ソースを `acceptance-cuda/stage1-accepted/package/` へ保存し、`acceptance.json` から機能検証 stage1 と性能結果へ案内した。機能検証で使った stage1 と新しい Debug・Release Runner のハッシュはともに一致する。最後の Python 再試験は性能 4 tests・機能採取器 6 tests とも成功。本体 C++ を変えていないため、既定 core suite と build は前の成功結果を使う。
- 工程の残り: ステージ 1 の性能課題 f9790d は事象の解消と受入を確認し、判定を残件無し（状態は対応中のまま）とする。ステージ 2 の課題 0ddffa は残件有りで、方式の整理・受入 11〜15 は未実施。英語版設計文書の追随も未完了。PRD 全体を完了とは扱わない。
- 性能結果の最終レビューは、実測・trace・保存基準には不足なし、計画と README の冒頭に残っていた機能検証時点の範囲の記述で差し戻し。現行の概要・検証・前提と README 冒頭を機能・性能受入 1〜10 の完了へ揃え、過去の経過行は保持した。監査表は着手前の結果と明示し、比較基準の保存先も実際の所在へ揃えた。原文は `.scratch/prd130/review-20261116-0647.md`。
- 修正後の再レビューは計画・実装とも OK。ステージ 1 の機能・性能受入 1〜10、比較基準と未完了範囲の記録を確認し、追加の不足・修正要求・範囲外の検出・起票候補・開発者への確認事項は無い。原文は `.scratch/prd130/review-20261116-0648.md`。実装品質確認観点の全項目を適用範囲に従って自己点検し、対象差分の空白・UTF-8/LF・試作識別子の参照先・不要な pyc の不在も確認した。今回の作業はここで停止する。ステージ 2 と英語版追随、コミットの準備は実施していない。

### ステージ 2 の経過

- 2026-11-16: 「性能比較前まで実行」の指示を受けて再開。ステージ 1 の受入済み実行体を保持し、同じ 100k checkpoint の基準側の続きの採取を開始した。性能比較は行わない。C++ の品質確認観点の指定 8 節と所有権ガイド、TDD・karpathy の手順、Runner ガイドの指定 3 節を確認した。
- 最初に、混合 byte・float・離散値の辞書の Clone・Save/Load とその後の採点・Learn・読むだけの採点・診断の同一性を、CPU/CUDA と累積/EMA/feedback/collect_errors で固定した。整理前の Debug 全ビルドと `[prd130],[prd127]` は 3,752 assertions、失敗 0、skip 0（`stage2-characterization.xml`）。
- 累積と EMA の誤差統計を分け、保存順・計算順と旧計測名を維持した。Debug 全ビルドと `[prd130],[prd116],[prd123],[prd127]` は 6,284 assertions、失敗 0、skip 0（`stage2-error-split.xml`）。
- checkpoint 継続の比較入口 `compare-load` を加えた。記録された `load_` の Run を読み、target_feature_var を除かず一致を求める回帰テストの RED→GREEN を確認し、採取器の Python 7 tests が成功した。
- 方式の責任を `WhitenStatistics` と `VarianceFeedbackStatistics` へ分けた。feedback は観測更新前の件数を保持して誤差更新へ渡し、更新後の件数から逆算しない。Debug 全ビルドと同じ関連テストは 6,284 assertions、失敗 0、skip 0（`stage2-mode-split.xml`）。
- target/predictor の forward と誤差計算を `ComputePredictionErrors` にまとめ、補助学習だけ predictor の勾配を有効にした。Debug・Release 全ビルドと `[prd130],[scaler],[prd116],[prd123],[prd127],[shipped_config]` は 9,886 assertions、失敗 0、skip 0（`stage2-related.xml`）。
- 先行レビューは計画 OK、実装は旧計測名の重複と静的文字列の借用保持の 2 件で差し戻し。計測名の定数を共有し、checkpoint の書式名を `std::string` の値として所有する修正後、コードの再レビューは計画・実装とも OK（原文 `.scratch/prd130/review-20261116-0739.md`）。最終ビルド・既定 suite・数値比較・文書の追確認は結果取得後に行う。
- 最終 Debug・Release 全ビルドが成功し、`acceptance-cuda/stage2/package/` に実行体・設定・本体ソースを固定した。core 既定 suite と 5 構成の 100k Run を開始。採取器の Python 7 tests の再試験も成功（`acceptance/stage2-python-tests.log`）。
- 設計文書 212 の方式の責任と共通 forward を更新し、200 と 212 の英語版を現行日本語へ追随させた。見出し階層、コードブロックの内容、表の行数が一致し、コード外の日本語と不在リンクは 0（`acceptance/stage2-translation-check.json`）。用語表への追加と新たな drift 候補はない。文書レビューは比較手順のケース指定の欠け 1 件で差し戻しとなり、直前と同じ 5 構成を明記した（原文 `.scratch/prd130/review-20261116-0743.md`）。結果取得後の再レビューで修正も追確認する。
- 基準側 `stage1-reference` の checkpoint 継続 3 構成が正常終了した。Debug Runner の SHA256 は元の `stage1` と一致し、同じ 100k checkpoint の続きを採っている。変更後の無効構成も 100k を正常終了し、時間・資源以外の 38,922 scalar 点が完全一致した（`acceptance-cuda/stage1-vs-stage2/off/comparison.json`）。
- 最終 core 既定 suite の初回は 196,024 assertions、assertion の失敗 0、例外 1、skip 1（seed 459223024、`stage2-default.xml`）。例外は RND ではなく `WorkspaceService persists MRU and last workspace` の `history.txt` の置換時の Access is denied だった。該当テスト単独は 4 assertions、失敗 0（`stage2-workspace-repro.xml`）。ファイルの属性は Archive で、原因は未特定。一時的な外部ファイル保持、テスト間の状態持越し、今回の変更の影響を候補とし、本体や断定を変えずに同じ seed の既定 suite を `stage2-default-r2.xml` へ再実行する。初回失敗と採取待ちの停止記録も保持した。checkpoint 継続は独立した検証として開始し、最終判定では既定 suite と継続比較の両方を要求する。
- 同じ seed の core 既定 suite 再試験は、196,027 assertions、例外 0・失敗 0・skip 1、終了値 0（`stage2-default-r2.xml`、766.129 秒）。skip は CUDA 非搭載時だけ行う `PRD070 reports every explicit CUDA key before NN initialization` で、この環境では CUDA が使えるためである。初回のファイル置換エラーは再発せず、RND 本体・テストの断定・既定 suite の範囲は変えていない。最初のエラーの原因の特定までは行っておらず、初回と再試験の記録を調査材料として保持する。
- 累積・EMA の通常 100k Run と累積の checkpoint 継続も正常終了し、全 scalar の checksum が一致した。通常 Run は 3/5 構成、checkpoint 継続は 1/3 構成まで確認した。残りの長時間 Run を継続する。
- 通常の残りの variance_feedback・collect_errors と、checkpoint 継続の EMA・variance_feedback がすべて正常終了し、時間・資源以外の全 scalar の checksum が一致した。通常 5 構成と継続 3 構成の両側で、実行体のハッシュ、入力設定のハッシュ、バックエンドに関係する環境、同じ元 checkpoint を照合した。集約した証拠は `.scratch/prd130/acceptance/stage2-evidence.json`。並行 Run の実時間を性能判定へ使っていない。

## ステージ 2 の機能検証終了時点の確認結果

| PRD 受入 | 確認 | 結果 |
| --- | --- | --- |
| 11 | 通常の 100k Run、RND 無効 | 38,922 点が完全一致 |
| 11 | 同、whiten 累積 | 43,465 点が完全一致 |
| 11 | 同、whiten EMA | 43,371 点が完全一致 |
| 11 | 同、variance_feedback | 43,507 点が完全一致 |
| 11 | 同、collect_errors | 43,533 点が完全一致 |
| 12 | 同じ stage1 checkpoint からの 100k 継続、累積 | 43,393 点が完全一致 |
| 12 | 同、EMA | 43,351 点が完全一致 |
| 12 | 同、variance_feedback | 43,525 点が完全一致 |
| 14 | 統計方式の選択 | estimator 内は構築時の型の選択だけ。`feedback_stats_` は 0 件。コードの独立レビューでも確認 |
| 15 | Debug・Release 全ビルド、既定 suite、関連テスト | ビルド成功。既定 196,027 assertions、関連 9,886 assertions が失敗 0。既定の skip 1 は CUDA 非搭載時だけの検査 |
| 検証資材 | Python の機能採取器 | 7 tests 成功。checkpoint の比較対象と target_feature_var の非除外を保護 |
| 文書 | 200・212 の日英版 | 現行の device 配置と方式の責任を同期。対象 2 文書の未訳・更新残り 0、用語表の追加 0、drift 候補なし |
| 13 | ステージ 2 の性能比較 | この時点では開発者の指定により未実施。後述の性能比較で完了 |

各比較は target_feature_var も含む全 scalar を対象に、時間・資源だけを除外した。計 343,067 点が一致し、採用した通常・継続の全 Run が終了値 0。基準側 `stage1-reference` の Debug Runner は、元の `stage1` と SHA256 が一致する。変更後は `acceptance-cuda/stage2/package/` の固定した実行体を使い、保存した本体ソース・Runner と現用ファイルも一致する。

最終の既定 suite の実行は `core/anet-core/bin/Debug/anet-core-test.exe --rng-seed 459223024 --reporter junit --out .scratch/prd130/acceptance/stage2-default-r2.xml`。標準出力・標準エラーは `acceptance/stage2-default-r2.log` に保存した。初回の履歴ファイルの置換例外と単独再試験は、上の経過のとおり別記録で保持し、成功した再試験と混ぜていない。

今回増えた公開 API はなく、ステージ 1 の `UpdateStats` 以外の契約変更もない。実装品質確認観点の全 83 項目を適用範囲に従って自己点検した。UTF-8・LF・既存 BOM・対象差分の空白、保存ソースと Runner の同一性、旧フィールドと試作の参照先、文字列の重複を確認した。重複候補 3 件はテストの設定キーで、許される直書きである。判定に使う記録は `acceptance/stage2-source-check.json`・`stage2-literal-candidates.log`・`stage2-translation-check.json` にある。

機能検証終了時点では、ステージ 2 の実装・機能検証と文書の追随までが到達点で、PRD 全体の受入完了とはしていなかった。課題 0ddffa は性能比較 13 を残すため残件有り、f9790d のステージ 1 の確認済み結果は維持した。この時点でも新規の起票候補と開発者判断待ちはなかった。

機能検証済みの範囲と性能未実施の印は `acceptance-cuda/stage2/functional-acceptance.json` に保存した。課題一覧と wiki の派生区画の再生成は成功し、変更対象と未解決 PRD の参照はともに 0（`acceptance/stage2-maps.log`）。課題の状態と監理の項目は変えていない。

最終レビューはステージ 2 の 4 往復目で計画・実装とも OK（原文 `.scratch/prd130/review-20261116-0841.md`）。前のコード 2 件と比較手順 1 件の指摘は解消した。通常 5 構成・同一 checkpoint 継続 3 構成の全 scalar 一致、ビルドと試験、日英の文書、範囲の記録に追加の不足・修正要求・範囲外の検出・開発者への確認事項は無い。初回ファイル置換例外の原因は未特定であることを維持し、同 seed の再試験の成功と区別している。指定された範囲をここで終え、性能比較 13 の前で停止する。評価 Run の操作は行っていない。

## ステージ 2 の性能比較の実行計画

- 2026-11-16、開発者の「PAUSEしました」により受入 13 を再開。既存の `run_performance.py` に `--stage 2` を足し、固定した stage1 と stage2 の Release Runner を選ぶ。主・副の既定 RND と副の Taiga の 3 条件を、それぞれ stage1/stage2/stage2/stage1 の 3 巡で測る。
- 既存の可塑性メトリクス除去・評価と描画の休止・時計・実効設定検査・休止済み PID の CPU 時間の検査を使う。warm-up と区間の選び方は上の検証欄のとおり。各条件の pilot で同じ経験区間を固定し、Tracy を通常時間へ混ぜない。
- 採取器の public な集計結果で、正しい順序・経験区間と、ステージ 1 の幅を超える中央値の差の拒否を先に RED→GREEN で保護する。次に生成する設定と実効設定検査で、両側が同じ主・副・Taiga 構成を選ぶことを保護する。C++ と受入済みの機能検証用実行体は変えない。
- 判定は中央値の差の絶対値が stage1 の最小〜最大の幅以内かを見る。初回の結果を保存し、失敗を隠して測り直したり、閾値や区間を後から緩めたりしない。独立した条件の失敗後も、採取条件の安全が保たれる限り残りの条件は測る。
- 詳細出力は `.scratch/prd130/acceptance/stage2-performance-*.log`、各測定先の `cases/`・`measurements_*.json`・`summary_*.json` へ保存する。画面へは進捗・成否・保存先を返す。既存の測定先は上書きせず、最終結果と自己点検・レビューを記録して停止する。
- 採取器の拡張は、中央値の幅の判定、両群の同じ設定、Taiga の実効値の検査、受入済み保存基準の選択の 4 つを順に RED→GREEN で確認した（`stage2-performance-*-red.log`・`*-green.log`）。幅の内の許可と順序違反の拒否も含む Python 9 tests が成功（`stage2-performance-tests.log`）。`--help` の実行も成功した。本体 C++ と保存済みの実行体は変えていない。
- 採取器の独立レビューは計画・実装とも OK（`.scratch/prd130/review-20261116-0911.md`）。測定先は `.scratch/prd130/performance-stage2-20261116-0910/`。`prepare --stage 2` が成功し、可塑性の定義 57 行を測定用コピーから除いた。休止中 PID 53272 の 2 秒の CPU 時間増加は 0、ほかの Runner と build・test は無い。Taiga・主・副の順で採取し、結果取得後に追確認を依頼する。
- Taiga の pilot 2 本は正常終了（8,192 経験が 16.507 秒・16.483 秒）し、本測定区間を 16,384 経験に固定した。本測定の最初の 5 本も正常終了したが、6 本目は Runner の終了値 0 のあと、休止中 PID の CPU 時間が 99.381 秒で 1.0625 秒増えたため、既存の 1 秒の上限で拒否した（`cases/taiga_r06_stage2/execution.json`）。最初の Taiga の一式を受入へ使わず、失敗と途中結果を保持する。主・副の独立した比較は続行する。
- 利用者の評価 Run は `run_20261116-005336_mz_tg_g99ne_split_s1`。09:00:59 の `Training paused` 以降、学習ログと `metrics.jsonl` の更新が無いことを確認した。CPU 時間だけでは GUI の活動と学習を分けられないが、検査の閾値は緩めない。主・副の採取後に、同じ実行体・設定・warm-up・経験区間・判定で、Taiga 全体を新しい測定先へ採り直す。最初の途中結果を新しい中央値へ混ぜない。
- 主の 12 本は正常終了し、中央値 stage1=19.905501 秒・stage2=20.014501 秒、差 0.109000 秒が stage1 の幅 0.581997 秒以内で合格。副の 12 本も正常終了し、中央値 20.068000 秒・19.957000 秒、差 -0.110999 秒の絶対値が幅 0.692996 秒以内で合格。これらは全プロセス区間の CPU 上限 1 秒も通過している。
- Windows の画面確認では評価 Run のウィンドウは既に最小化されており、表示状態を変えなかった。Taiga の最初の有効な pilot の 16,384 経験を引き継ぎ、同一の実行体・設定ハッシュを照合した新しい `.scratch/prd130/performance-stage2-taiga-r2-20261116-0952/` で全体を再採取したが、最初の Run が全生存期間 100.186 秒・休止 CPU 1.09375 秒で拒否された。最初の 2 つの Taiga の測定先は不採用のまま保持する。
- 原因の切り分けは、実測外の CPU 増加、学習・評価の再開、別 Run の競合を候補とし、採用しない別 Run `taiga_guard_diagnostic` で区間別の CPU と休止 Run のファイル状態を記録した（2 回目の測定先の `paused_cpu_diagnostic.json`）。全プロセスの CPU は約 1.1 秒、実測付近の 2 サンプル間は 0.375 秒だった。ただし開始サンプル後の公開時計は 32,960 で、測定始点 32,768 を越えているため、0.375 秒を実測全体の値とは断定できない。休止 Run のログ・メトリクスのサイズと更新時刻は全区間で同一。広い CPU 検査には初期化・warm-up も含まれるが、診断だけでは実測の包含を証明していない。診断 Run の時間は受入へ使わない。
- 採取器の CPU 上限 1 秒は維持し、実測開始の 8,192 経験前（24,576）から終了後までを検査する。CPU を読んだあとの公開時計が 32,768 未満、終了後の時計が測定終了以上でなければ拒否し、実測を含まない検査の採用を防ぐ。全プロセス区間では休止 Run のログ・メトリクスのサイズと更新時刻が変わらないことを追加要求する。全区間の CPU 増加も別欄で残す。新しい `--paused-run` は PID と 1 対 1 で指定する。性能の中央値・幅・経験区間・固定実行体は変更しない。
- CPU 検査区間と上限の保護テストは未実装を RED とし、実装後に GREEN。低 CPU でも休止 Run のファイルが進んだ場合と、CPU 読み取りが実測開始に間に合わない場合の拒否も確認し、Python 13 tests が成功（`stage2-performance-tests-r2.log`）。先に通過した主・副は、実測を含むさらに広い CPU 区間が 1 秒以内なので、元の記録と結果を維持する。Taiga は不採用データを再解釈して合格にせず、新しい検査で同じ 16,384 経験の全 12 本を採る。再開前に修正部分を独立レビューへ出す。
- 独立レビューで、公開時計のファイル出力には遅延があり、経験数の比較だけでは開始前の CPU 読み取りを保証できないと指摘された（`.scratch/prd130/review-20261116-1014.md`）。採取を始めずに修正した。Runner の `elapse_hour` は子プロセス起動後の `TrainRunner::Step` の初期化で 0 から始まる（`trainer.cpp`）。CPU 読み取り完了の単調時計を、子の起動前からの経過秒数として保存し、これに丸め余裕 0.01 秒を足しても、測定始点の Runner 経過秒数より短いことを要求する。Runner の原点が遅い分も含む保守的な条件で、出力が遅れても実測前の読取を保証する。終了側は子の終了後に読むので実測終了を含む。経験数の検査も併用する。古い公開時計を示す遅い CPU 読み取りの拒否を RED→GREEN で保護し、Python 14 tests が成功（`stage2-performance-clock-red.log`・`stage2-performance-clock-green.log`）。全品質観点を再点検し、修正後の再レビューを受ける。
- 修正後の再レビューは計画・実装とも OK（`.scratch/prd130/review-20261116-1016.md`）。全 12 本の新規採取を `.scratch/prd130/performance-stage2-taiga-r3-20261116-1016/` に開始する。最終時に、時間包含・CPU 上限・休止ファイルの不変・交互順序・性能結果を追確認する。
- 3 回目の採取は最初の 3 本が通過したが、4 本目で CPU 読取完了の起動前原点からの秒数 53.054071 が、Runner の測定始点の秒数 52.450000 より遅く、保守的な包含条件を満たさず停止した。CPU は選択区間 0.59375 秒、全プロセス 0.796875 秒で、停止ファイルも不変だったが、この一式も不採用とし保持する。起動時間の差により、24,576 経験の 1 回だけでは保守的な条件に間に合わないケースがある。
- 対応として同じ保守的な時間条件を維持し、約 1 秒間隔の CPU 履歴から条件を満たす最後の時刻だけで始点を選ぶ。Python の Windows API `GetProcessTimes` で同じ HANDLE の kernel＋user 時間を読み、終了・API 失敗・欠測・非有限値・逆行は拒否する。1 回の read は数個の OS 呼出しだけで、毎回 PowerShell を起動しない。全サンプルと選択した始点・終端・全期間 CPU を保存する。測定中の採取は低負荷の読取だけで、数値上限・計測経験数・固定実行体は不変。遅い最新値より前の安全な履歴を時刻だけで選ぶ RED→GREEN、欠測・非有限値・逆行の拒否の RED→GREEN を確認し、Python 16 tests が成功（`stage2-performance-sample-*.log`・`stage2-performance-counter-*.log`）。実際の生存 PID の読取と、自分で作った短い子の終了後の拒否も成功（`stage2-performance-native-cpu-check.log`）。再レビュー後に新しい全 12 本を採る。
- CPU 履歴版の再レビューは計画・実装とも OK（`.scratch/prd130/review-20261116-1030.md`）。4 回目は `.scratch/prd130/performance-stage2-taiga-r4-20261116-1030/` で、同じ 16,384 経験の全 12 本を新たに測る。

## ステージ 2 の性能結果と完了確認

2026-11-16、Taiga の新しい全 12 本が正常終了し、主・副と合わせた採用 36 本の原記録を照合した。実行体・設定のハッシュ、実効設定、終了値 0、学習の進行、同一の測定時計区間、交互 3 巡、中央値と幅を確認し、受入 13 の 3 条件がすべて合格した。

| 条件 | stage1 中央値（秒） | stage2 中央値（秒） | 差（秒） | stage1 の幅（秒） | 結果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 主の既定 RND | 19.905501 | 20.014501 | +0.109000 | 0.581997 | 合格 |
| 副の既定 RND | 20.068000 | 19.957000 | -0.110999 | 0.692996 | 合格 |
| 副の Taiga | 32.709499 | 32.590000 | -0.119499 | 1.121995 | 合格 |

判定は事前に固定した `abs(stage2 median - stage1 median) <= stage1(max - min)`。主・副の保存先は `.scratch/prd130/performance-stage2-20261116-0910/`、Taiga は `.scratch/prd130/performance-stage2-taiga-r4-20261116-1030/`。Taiga の pilot の 16,384 経験を維持し、不採用 3 組の途中値・診断 Run は混ぜていない。

主・副は全プロセス区間の CPU 上限を通過した元の記録で、最大は 0.84375 秒・0.875 秒。Taiga は時刻だけで選んだ始点の包含条件、同じプロセスの CPU、全区間の休止ログ・メトリクスの不変を全 12 本で確認し、CPU 最大は 0.500 秒。0.01 秒の丸め余裕を差し引いた包含条件の最小余裕は 0.001111 秒で、全件が正である。全履歴と選んだ始点も保存している。

照合は `.scratch/prd130/collect_stage2_performance.py` で実行し、結果は `acceptance/stage2-performance-evidence-check.log`・`acceptance/stage2-performance-evidence.json` へ保存した。`acceptance-cuda/stage2/acceptance.json` はステージ 1 の受入 1〜10、機能受入 11・12・14・15、今回の 13 を参照し、全 1〜15 を完了とする。過去のステージ 1 と機能検証だけの受入記録は当時の範囲のまま保持した。

測定終了を確認した直後に評価 Run を再開できることを開発者へ通知した。評価 Run の PAUSE・再開・終了の操作は行っていない。C++ の本体と受入済み Debug・Release 実行体は変更していない。初回の workspace 履歴ファイル置換例外の原因は未特定だが、同 seed の全 suite 再試験の成功と区別して記録している。未実施の必須検証・開発者判断待ち・新規起票候補はなし。

実装品質確認観点の全 83 項目を適用範囲に従って自己点検した。CPU 履歴版の 16 tests・実 API 検証、全 36 本の証拠照合、最終のソース・実行体同一性と UTF-8/LF・空白の検査、日英の文書対応とリンクの検査はすべて成功（`stage2-performance-tests-r4.log`・`stage2-performance-native-cpu-check.log`・`stage2-performance-evidence-check.log`・`stage2-final-source-check.log`・`stage2-final-translation-check.log`）。課題 0ddffa を残件無しとし、状態は対応中を維持した。派生一覧の再生成も成功し、変更は INDEX と 321_agent_core の該当区画、未解決 PRD 参照は 0（`stage2-final-maps.log`）。受入済みビルド以降に C++ は変えていないため、全ビルド・既定 suite の成功記録を再利用する。最終の結果と記録を独立レビューへ出す。

結果の追確認では、全 36 本の原記録・休止検査・受入結果に不足はなかったが、計画と README の冒頭の現行到達点に性能未完了の表記が残っていたため差し戻し（`.scratch/prd130/review-20261116-1054.md`）。概要・検証冒頭・前提・README の今回の範囲を完了へ揃え、過去の機能検証終了時点はその時点だと明記した。全体の同種表記も検索し、品質観点の該当項目と空白・文字コードを再点検した。コード・測定値は変えていない。

### 完了報告

性能採取器と結果のレビューは 6 往復目で計画・実装とも OK（原文 `.scratch/prd130/review-20261116-1055.md`）。全受入 1〜15、採用 36 本の時間包含と休止検査・測定順序・集計、機能の完全一致、ビルド・テスト、日英の文書と到達点を追確認した。根拠が無く見送った指摘、範囲外の検出、新規起票候補、開発者への確認事項はなし。初回 workspace 例外の原因未特定は再試験成功と区別して保持する。PRD130 の実装工程を完了する。

## 2 モデルレビューと指摘対応

- 2026-11-16、開発者の `cc-review` 指定により、同じ依頼で PRD130 の完成した実装全体を Codex と Claude に出した。Codex は計画・実装とも OK、Claude は計画 OK・実装差し戻しで、212 日英の性能未完了表記、クラス宣言の区切りと型の宣言位置、共通化で失われた意図コメントの 3 件を指摘した（原文 `.scratch/prd130/review-20261116-1214.md`・`review-20261116-1209-claude.md`）。
- 開発者の「指摘修正して再レビュー」により、3 件を修正する。実行する計算・状態・レイアウトを変えず、`WhitenStatistics` の関数群の空行、`RndNoveltyEstimator` の追加した型の前方宣言の位置、有限検査と warm-up の意図コメントを直す。212 の日英の性能比較の現行到達点と翻訳元情報も更新する。Debug・Release 全ビルドと関連テスト、非機能変更であることの固定ソースとの差分、文書の対応・リンク・文字コードを確認して、同じ Codex 担当と Claude の session の続きへ同じ依頼を出す。性能測定は固定した実行体の受入記録を保持する。
- 3 件の修正を終えた。`WhitenStatistics` の private 関数群の生成・上書き・固有処理の切れ目に空行を置き、追加した 3 つの型の前方宣言を `ComputePredictionErrors` の前へ移した。有限な誤差を更新前に検査して統計・重み・moment を保護する意図と、predictor が慣れるまで誤差統計を空に保つ意図を移動先へ戻した。212 日英 §5.1 は主・副・Taiga の性能受入完了へ揃え、同じ段落周辺の別途測定の表記も解消した。英語版の翻訳元 blob を現在の日本語版に合わせた。既存の残件の意味、数式、条件、データメンバ、ログや計測名は変えていない。
- 検証は MSVC 初期化経由の `cmake --build --preset x64-Debug` と `x64-Release` の全ビルドが成功（`acceptance/cc-r2-debug-build.log`・`cc-r2-release-build.log`）。修正後の Debug で `anet-core-test.exe "[prd130],[scaler],[prd116],[prd123],[prd127],[shipped_config]" --reporter junit --out .scratch/prd130/acceptance/cc-r2-related.xml` を実行し、通常ログは `cc-r2-related.log` へ保存した。9,886 assertions、失敗・例外・skip 0 で成功。
- `.scratch/prd130/check_cc_r2.py` は固定 package と修正前スナップショットを照合し、cpp のコード行はコメントと空行以外で同一、hpp は指定した 3 つの前方宣言の移動以外で同一、ほかの保存ソースは完全同一と確認した。19 ファイルの UTF-8/LF と既存 BOM、対象の空白も検査した（`acceptance/cc-r2-source-check.json`・`cc-r2-source-check-final.log`）。日英の構造・コード・表・未訳・リンク・翻訳元情報の検査も成功（`cc-r2-translation-check.json`）。計算に関する挙動を変えない修正なので、受入済みの 100k 比較・checkpoint 継続・全既定 suite・性能測定の記録はそのまま参照し、再実行していない。固定した実行体は変更しておらず、再ビルド後の現用実行体のハッシュは別欄で保存した。
- 実装品質確認観点の全 83 項目を今回の差分へ適用して自己点検した。今回の変更には新たな挙動・公開 API・設定・所有状態を足していないため、TDD の追加テストは対象外とし、既存の関連テストで確認した。翻訳用語の追加と新しい drift 候補はない。前回の共通依頼から変えた 7 ファイルだけを、新しい差分と検証記録で両モデルの続きへ出す。
- 同じ共通依頼 `.scratch/prd130/request-20261116-1241.md` で両モデルへ続けて依頼し、2 往復目は Codex・Claude とも計画・実装が OK。前回の 3 件は解消し、追加の修正要求・範囲外の検出・開発者への確認事項はない。原文は `.scratch/prd130/review-20261116-1245.md`・`review-20261116-1241-claude.md`。両者とも今回の検査スクリプトを再実行せず、読み取り比較と保存記録を使った。Claude はビルドログと関連 XML を直接読まず、依頼の検証記録を根拠にしていることも原文へ残している。Claude の SDK 表示費用は今回の結果で USD 8.851619（前回の表示は USD 4.8760974）。セッション継続での表示値として記録し、増分や合計へ読み替えていない。修正と両モデル再レビューをここで完了する。
