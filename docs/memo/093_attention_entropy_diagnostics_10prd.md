# TransformerEncoder の attention エントロピー診断 PRD

> 2026-10-16 グリル（初回）: 暫定の既定をすべて裁定に置き換えた（§8）。取り出し口は capture の一般化（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)）、設定キーは `final_feature_branch_key` へ改名、`attention_entropy_min` は cut、公開は `$update_result`・interval 503、層番号は 0 始まり。TraceCallback の統合は [999_nn_trace_capture_unification_10prd.md](999_nn_trace_capture_unification_10prd.md) へ先送り。
> 2026-10-10 起票（暫定）。
> 起点: TQL(Dong et al. 2026、ICML 2026)の論文解説 [reports/dong2026_tql_paper_2026-10-05.md](../../reports/dong2026_tql_paper_2026-10-05.md)。Transformer の価値関数を大きくすると、層の attention が少数のトークンへ偏り（エントロピーの崩壊）、成績が下がるという報告。
> 関連: [done/062_plasticity_metrics_10prd.md](done/062_plasticity_metrics_10prd.md)（可塑性メトリクス。測り方の型を借りる）、[done/012_sdpa_attention_10prd.md](done/012_sdpa_attention_10prd.md)（self-attention の SDPA 化）、[084_noisynet_10prd.md](084_noisynet_10prd.md)、[092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md)（公開の形の先例）、[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)、[ADR 0004](../adr/0004-sdpa-attention-via-aten.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)。

## 1. 目的と問題

TransformerEncoder の各層で attention がどれだけ少数のトークンへ偏っているかを、学習中のスカラーの指標として見られるようにする。これで、TQL が報告した attention の崩壊が anet の Run で起きているか、起きているなら成績の頭打ちや下降と時期が重なるかを、Run のたびに確かめられる。

anet では Transformer を含むネットが本線と常設の構成に入っているが、その内部を見る指標が無い。

- DropMerge の本線（`main_feature` の選択 `net.branch.@SuikaImpala2_ViT128`）は、IMPALA 型の CNN の後ろに TransformerEncoder を置いている。
- Atari の ViT 混成（`run.@vitva`、`run.@vitbtr`）は、成績によらず常設のプロファイルである。

可塑性メトリクス（`34_agent_plasticity`）は final feature branch の出力の健全性（dormant、srank など）を測るが、attention の分布は見ない。

## 2. 現状

- TransformerEncoder は `CustomTransformerEncoderLayer` を層とする module で、設定キーは `tf.d_model`、`tf.nhead`、`tf.num_layers`、`tf.dim_feedforward`、`tf.norm_first`、`tf.activation`、`tf.use_sdpa` と dropout 系である（[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) の `TransformerEncoderModule`）。層は `layer_0` から番号を持ち、SN 診断の名前も `layer_0.q` のように 0 始まりである。
- self-attention は `at::scaled_dot_product_attention` で計算していて（[ADR 0004](../adr/0004-sdpa-attention-via-aten.md)）、attention の重みを返さない。重みの保持器としての `MultiheadAttention` は残っているので、Q と K は同じ重みから計算し直せる。SN 有効時は `ForwardAttention` が実効重みを使う。
- DropMerge の本線の Transformer は `DMImpalaTrans128` で、d_model 128、nhead 8、4 層、dim_feedforward 512、pre-LN、GELU、SDPA である（[DropMerge.txt](../../apps/runner/config/DropMerge.txt)）。トークンは CNN 出力の空間位置で、[VALUE] のような専用トークンは無い。全トークンを Flatten して head へ渡す。
- Atari の ViT 混成は `AtariImpalaTrans64`（d_model 64、nhead 4、4 層）で、`run.@vitva`（`@AtariImpalaX2ViT64LN`）は 11×11 = 121 トークン、`run.@vitbtr`（`@AtariImpalaX2SN12ViT64Pool6Flat`）は Pool6 で 6×6 = 36 トークンである（[Atari.txt](../../apps/runner/config/Atari.txt)）。
- 構成の良し悪しを成績で判断するには、DropMerge では 1 腕あたり最低 50M、最終判断に 100M を使っている。
- 学習経路の外で測る仕組みは可塑性メトリクスにある（[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)）。ReplayBuffer の一様・非復元の sample を、部分 forward（`ForwardUpTo`、NoGrad・eval mode、Learner と同じ autocast）へ通す。購読された指標の interval でだけ測り、乱数を消費しない。probe の大きさは `learner.<対象>.probe.batch_size` の形で持つ（`plasticity` 4096、`policy_churn` 1024、`replay_fit` 1024、`noisy_probe` 128）。
- forward から出力のほかに持ち帰る口は `NetworkBranchCapture`（branch 名 1 つと出力 1 つ。body が branch ループの末尾で state から detach して書く）で、`Network::Forward` と実行器の `Forward` だけが受ける。`ForwardUpTo` は受けない。[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) 以降、`ForwardUpTo` は実行情報を受けず既定 Context（Noisy は μ-only）で動き、module は State のポインタと呼び出し情報を束ねた `NetworkForwardContext` を受け取る。
- `TraceCallback` は名前付き tensor を外へ出すもう 1 つの口だが、要求の仕組みが無く、Actor が `MakeAction` で毎回渡す。追加計算を伴う値をここへ乗せると、購読ゼロでも Actor の毎 forward に計算が乗る（ADR 0031 の A''）。
- `learner.plasticity.feature_key`（既定 `main_feature`）は、可塑性 probe の到達 branch、重みノルムの feature/readout 分割、Noisy の σ 統計、勾配ノイズの群分けの 4 診断が「feature 閉包」として共用している。用途ラベル `plasticity.` の下に共用のものが置かれた形である。
- 新しい診断（policy churn、replay_fit、Noisy、勾配ノイズ）は、購読の発火する更新の中で測り、その `BatchUpdateResult` に値を載せて `$update_result` で読む。interval は 503（素数。hard update C=200 と位相が固定されるのを避け、同じ `learn_step` で横に読む）。可塑性 probe だけが `$agent` の最新値で interval 500 である。
- TQL の測り方: 層 $\ell$ のクエリ $i$ の attention 分布 $A^\ell_{i\cdot}$ のエントロピー $H^\ell_i = -\sum_j A^\ell_{ij}\log A^\ell_{ij}$ を、クエリで平均して層の値 $H^\ell$ とする。一様なら最大の $\ln n$（$n$ はトークン数）、1 トークンへの集中で 0 になる。TQL の実験は head で attention を平均してからエントロピーを取る（付録 C.2）。TQL の目標エントロピーの初期値は $0.8\ln n$ である。
- attention entropy collapse という語を作った Zhai ら（ICML 2023、σReparam）は、学習中に head ごとの attention エントロピーを追跡し、低エントロピーの状態を崩壊と呼んで学習の不安定化との対応を示した。head ごとの定義が文献の標準で、TQL の head 平均が変則である。

## 3. ゴールと非ゴール

ゴール:

1. TransformerEncoder の層ごとの attention エントロピーを、学習の `learn_step` 軸のスカラーとして記録する。
2. 学習の forward の計算と数値（SDPA の経路）を変えない。指標を購読しない構成では計算も起きない。

非ゴール（§7 に、作る条件を書いた）:

- エントロピーの正則化（TQL の本体。層ごとの温度と補助損失）。
- 価値を読むトークン（行動ごとのクエリトークン）を置く構造。
- attention マップの画像化（GUI や Viewer での表示）、head 単位の値の公開。
- 学習の forward から取る値（可塑性メトリクスでいう actual チャネル）。
- target network の測定、DefaultDQN 以外の Agent。
- TraceCallback の capture への統合（999 PRD）。

## 4. 決定した設計

1. **測り方**: 購読された指標が発火する更新の中で、ReplayBuffer から一様・非復元で probe の sample を取る（専用 RNG。足りなければその更新は測らない）。既存の観測の正規化を通し、online network を部分 forward で `learner.final_feature_branch_key` の branch まで実行する（NoGrad・eval mode・Learner と同じ autocast・既定 Context）。このときだけ、閉包の中の各 TransformerEncoder 層で $\mathrm{softmax}(QK^\top/\sqrt{d_h})$ を明示的に計算する。SDPA の経路は触らない。IQN でも `taus` は要らない（final feature branch で止まるため）。
2. **値**: head ごと・クエリごとの attention 分布のエントロピーを sample・head・クエリで平均し、$\ln n$（$n$ は系列長）で割って 0〜1 にそろえる。1 が一様、0 が 1 トークンへの集中。$n < 2$ は NaN。head で attention を平均してからエントロピーを取る TQL の定義は採らない（平均分布のエントロピーはエントロピーの平均以上になり、head ごとに別のトークンへ崩壊した状態を隠す）。Zhai らの head ごとの定義に従う。エントロピーの計算は純関数の中で autocast を切り、Q・K の射影から softmax まで FP32 で行う（BF16 の logit では小さい確率が潰れる）。forward 自体は Learner の精度のまま。
3. **取り出し口**: `NetworkBranchCapture` を、呼び出し側が鍵の集合で要求する `NetworkForwardCapture` に一般化する（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)）。操作は `Wants`・`Add`・`Get`・`Unfulfilled`・`Reset`。書き手は 2 種で器は区別しない。body は branch ループの末尾で、要求鍵のうち branch 名に当たるものを detach して `Add`（今の動作そのまま）。module は forward 中に、自分が所有する鍵が `Wants` なら計算して `Add`。鍵は書き手の公開定数で、module の鍵は型名を含む（`TransformerEncoder/attention_entropy`）。branch 名には `/` を使えない（Network 構築時に reject。衝突が起きたことはなく、規約を構造にするための番人）。`Forward`・実行器の `Forward`・`ForwardUpTo` が同じ型を受け、内部は `NetworkForwardContext` にポインタ 1 本を足して module まで運ぶ。入口で `Reset`、出口で `Unfulfilled` が空でなければ鍵名と利用できる branch 名を添えて `ANET_SYSTEM_ERROR`。渡さない forward は module で null 判定 1 回。TransformerEncoder は層ごとに FP32 の `[B, H, n]`（sample・head・クエリの未正規化エントロピー）を forward 順に `Add` する。確率行列 `[B, H, n, n]` は外へ出さない。平均と $\ln n$ の正規化は `nn_util` の純関数で行う。
4. **層の番号**: closure の中で実行される TransformerEncoder の層を、実行順に 0 から数えた通し番号。checkpoint の `layer_0` と同じ 0 始まり（`ram_metric.[n]` の 1 始まりは 0 を「番号なし」に使うためで、層には当てはまらない）。
5. **公開**: 測定した更新の `BatchUpdateResult` に層ごとの値を載せ、`GetScalar("attention_entropy.[n]")` で返す。測らない更新は NaN。metrics の定義に新しいプロファイル `metrics.scalar.@attention` を置き、群は `48_agent_attention`、interval は 503。baseline には入れない。現行の 4 層構成に合わせて 4 行を置く。

   ```properties
   metrics.scalar.@attention : [48_agent_attention/01_entropy_layer0] ?= attention_entropy.[0] @learn $learn_step $update_result interval:503
   metrics.scalar.@attention : [48_agent_attention/02_entropy_layer1] ?= attention_entropy.[1] @learn $learn_step $update_result interval:503
   metrics.scalar.@attention : [48_agent_attention/03_entropy_layer2] ?= attention_entropy.[2] @learn $learn_step $update_result interval:503
   metrics.scalar.@attention : [48_agent_attention/04_entropy_layer3] ?= attention_entropy.[3] @learn $learn_step $update_result interval:503
   ```

   Run への追加は `@gradnoise` と同じく `metrics.scalar.$` のチェーンへ足す Run プロファイル `run.@attention` を DropMerge.txt と Atari.txt に置く。tag は Viewer の一覧で省略されない 43 文字以内に収める（本 PRD の tag は 37 文字）。
6. **設定**:

   ```properties
   DefaultDQNAgent.@baseline : learner.attention_entropy.probe.batch_size ?= 256
   DefaultDQNAgent.@baseline : learner.final_feature_branch_key ?= main_feature
   ImageClsAgent.final_feature_branch_key = main_feature
   ```

   `probe.batch_size` は 1 未満で読み込み時に `ANET_SYSTEM_ERROR`。可塑性の 4096 を流用しないのは、attention の確率行列が sample 数 × head 数 × $n^2$ になるからで、`@vitva`（121 トークン・4 head）では 4096 で 1 層あたり約 1GB、256 で約 60MB の一時領域になる（層ごとに捨てる）。256 × 4 × 121 ≈ 12 万本の分布の平均なので推定の揺れは問題にならない。
   `final_feature_branch_key` は `plasticity.feature_key` の改名で、4 診断と本 PRD が共用する「観測から計算された特徴が集約される feature 側の最後の branch」を指す。移行はクリーンブレーク: 読む側はキー名の置換だけで、旧キーの検出・別名・互換読みは置かない。互換性は移行側でケアする（`agent.txt`・`ImageCls.txt` の 2 行、PRD072 の golden 17 本を README の手順で採り直し、030 ガイド、CHANGELOG に改名と「過去 dump を再生するときの読み替え」）。実験記録の dump は記録なので触らない。
7. **fail-fast**: `attention_entropy.[n]` の n が整数でない・負なら購読の解釈時（学習開始前）。n が層数以上なら最初の測定で、鍵・層数・許される範囲を添えて止める。Transformer の無い閉包で購読すると、capture の未達規則で最初の測定の forward の出口で止まる。
8. **対象**: DefaultDQN（TD/QR/IQN、共有 Learner）。ImageCls は ReplayBuffer が無く probe の形が合わない。Rainbow は現行 env 設定に無く検証できない。

## 5. 機構ごとの根拠

| 機構 | 根拠（起きたこと・測った量） | 裁定 |
|---|---|---|
| probe での層ごとの測定 | DropMerge の本線と Atari の常設プロファイルが Transformer を含むのに、内部を見る指標が無い。Atari の ViT 混成が V/A 基準を下回った原因を切り分けられていない。成績で構成を判断するには DropMerge で 1 腕 50〜100M かかる | 作る |
| 測定のときだけ attention を明示的に計算する経路 | SDPA は重みを返さない（ADR 0004）。学習の forward を変えないため | 作る |
| 層ごとの値 | TQL の切り分けで、温度を全層で 1 つにすると 26M の成功率が 41%、層ごとだと 67% と差が出た。崩壊する層を特定するには層ごとの値が要る。費用は層数分のスカラー | 作る |
| head ごとのエントロピー | TQL の head 平均は head ごとに別のトークンへ崩壊した状態を隠す（凹性）。Zhai らの先例 | 作る |
| 鍵要求型 capture への一般化 | 汎用の `ForwardUpTo` に固有目的の引数を付けない。forward の持ち帰りの概念を 1 つに保つ | 作る（ADR 0055） |
| 専用の probe の大きさ | 可塑性の 4096 では確率行列が 1 層 1GB 級 | 作る |
| 新プロファイル `@attention`（baseline 外） | 費用を確かめる前に既定 ON にしない規約。Transformer の無いネットでは購読できない | 作る |
| 設定キーの改名 | 用途ラベル `plasticity.` の下に 5 診断が共用するキーがあるのは「設定階層はフレームワーク概念」の規約に反する | 作る（step 0） |
| 層の最小値 `attention_entropy_min` | 痛みの記載なし。4 層なら 4 本の線で読める。min はどの層かを隠す | 作らない。6 層構成で読みにくくなったら戻す |
| head 単位の値の公開（head 最小値、閾値以下 head 割合） | 層平均が下がったとき内訳が要るかはまだ分からない。capture の `[B,H,n]` を保てば nn_util と鍵の追加だけで足せる | 作らない。層平均が下がる Run が出て内訳が要る日に |
| 専用の `attention_entropy.feature_key` | 今の 2 構成は Transformer が feature 閉包の中。別の閉包を測りたい痛みは無い | 作らない。Transformer が feature 閉包の外に置かれた日に（段階 3 の行動クエリ、IQN fusion 後の Transformer、可塑性側が閉包を変えるとき） |
| 学習開始前のダミー forward で層数を数える | 黙って NaN が続く痛みは最初の測定の fail-fast で防げる。API が 1 つ増える | 作らない |
| TraceCallback の統合 | 1 関心 2 機構だが、pin された痛みは未計測 | 作らない。999 PRD にゲートを書いた |
| 学習の forward から取る値（actual） | まだ要る理由が無い | 作らない。probe で崩壊が見え、train mode の値との違いを確かめる必要が出たら |
| エントロピーの正則化 | anet ではまだ崩壊を観測していない | 作らない。§7 の条件を満たしたら別の PRD で |

## 6. 受入基準

1. 単体テスト（`nn_test.cpp`）
   - capture: `Wants`・`Add`・`Get`・`Unfulfilled`・`Reset`。未要求の鍵への `Add` は error、未達の要求は Network の出口で error、渡さない forward は不変（既存の `NetworkBranchCapture` テストの置き換え）。名前に `/` を含む branch は構築時に error。
   - エントロピー関数: Q・K の射影重みを 0 にした層で正規化値 1.0、1 トークンへ集中するよう組んだ重みで 0 に近い、手計算の参照と一致、SN 有効時は実効重みで一致。
   - 2 層の小さな Transformer で `ForwardUpTo` に capture を渡すと、`[B, H, n]` が層数分 forward 順に積まれる。
2. Learner テスト（`dqn_based_agent_test.cpp`）: `attention_entropy.[n]` の解釈（非整数・負は購読時に error）、測定した更新の `BatchUpdateResult` に値、測らない更新は NaN、n ≥ 層数は最初の測定で鍵・層数・範囲を添えて error。
3. 改名（step 0）の等価性: 改名前後で config dump の差分がキー名だけ、既存テスト全通過、PRD072 の golden を README の手順で採り直す。
4. smoke: DropMerge 本線に `run.@attention` を足して短く回し、`inspect_run.py tags` に `48_agent_attention/` の 4 本が出て、値が 0〜1。Atari の `run.@vitva` でも 1 本。
5. ON/OFF 等価性: 同じ seed・決定論の設定で、`@attention` の有無によらず学習経路の metrics checksum が一致する（ADR 0031 と同じ確かめ方）。
6. 性能: 測定に `ANET_PROFILE_SCOPE` を入れ、interval 503 で回したときの測定時間の合計が Run の実時間の 1% 以下を目標とする。092 と同じく目標であり、超過だけで不合格にしない。

成功の測り方: DropMerge の本線の 100M Run と Atari の `run.@vitva` の Run で、層ごとの値が学習中にどう動くかを読み、次のどちらかを実験記録に残せること。

- 下がり続ける層があり、成績の頭打ちや下降と時期が重なる → §7 の段階 2 へ進む。
- 崩壊していない → 正則化は作らず、ネットを大きくする腕を試すときにだけ見る。

## 7. 先送りした段階と、作る条件

1. 段階 2（TQL の正則化）: 層ごとの温度（学習する値）と目標エントロピーを TransformerEncoder のオプションに持たせ、補助損失を Learner の損失に足す。作る条件: 本 PRD の指標で、正規化した値が下がり続ける層があり、成績の頭打ちや下降と時期が重なること。作るときは、ネットの forward で出た値を Learner の損失へ足す口（今の anet には無い）と、学習の forward でも attention の確率を計算する経路が要る。損失側の定義（head ごとか head 平均か）はそのとき改めて選ぶ。
2. 段階 3（価値を読むトークン）: 行動ごとにクエリトークンを置き、各トークンが盤面のトークンを読んで Q を出す構造。作る条件: 行動クエリの構成を試すと決めたとき。その block は feature 閉包の外になるので、専用の到達キーと別の capture 鍵が要る。
3. ゲート付きの先送り（§5）: head 単位の値の公開、専用の到達キー、TraceCallback の統合（[999_nn_trace_capture_unification_10prd.md](999_nn_trace_capture_unification_10prd.md)）。

## 8. 決定事項（2026-10-16）

| # | 論点 | 裁定 |
|---|---|---|
| D1 | ゴール | 層ごとの attention エントロピーの正規化値を `learn_step` 軸で記録。学習の forward と数値は不変、未購読なら計算ゼロ |
| D2 | `attention_entropy_min` | cut。4 層なら 4 本で読め、min はどの層かを隠す |
| D3 | 取り出し口 | `NetworkBranchCapture` を鍵要求型の `NetworkForwardCapture` に一般化（ADR 0055）。`Forward`・実行器 `Forward`・`ForwardUpTo` が同じ型を受け、内部は Context のポインタ 1 本。入口 Reset・出口で未達 error。branch 名の `/` 禁止 |
| D4 | capture の中身 | TransformerEncoder が層ごとに FP32 `[B, H, n]` を forward 順に積む。鍵は `TransformerEncoder/attention_entropy`。畳みは nn_util の純関数 |
| D5 | 到達 branch と層番号 | `learner.final_feature_branch_key` の依存閉包。閉包内の TransformerEncoder 層を実行順に 0 から |
| D6 | 設定キーの改名 | `learner.plasticity.feature_key` → `learner.final_feature_branch_key`、ImageCls も同時。クリーンブレーク（移行側でケア、実装に番人を残さない）。step 0 の独立コミット |
| D7 | 値の定義 | head ごとのエントロピーを head・クエリ・sample で平均、ln n で正規化、n < 2 は NaN。Zhai らの定義。forward は Learner 精度、エントロピー計算は FP32 |
| D8 | 公開 | `$update_result`、interval 503、群 `48_agent_attention`、プロファイル `@attention`（baseline 外）、`attention_entropy.[n]` 0 始まり、tag は 43 文字以内 |
| D9 | probe の大きさ | `learner.attention_entropy.probe.batch_size ?= 256`、専用 RNG、足りなければ NaN |
| D10 | fail-fast | 非整数・負は購読時。n ≥ 層数は最初の測定。Transformer 無し閉包は capture の未達規則。batch_size < 1 は読み込み時 |
| D11 | 用語 | CONTEXT.md に「attention エントロピー」「final feature branch」「capture」。「weight norm 分割」の「feature key」を読み替え |
| D12 | ADR | 新 ADR 0055。ADR 0031 先頭に注記。head ごとの定義とキー改名は ADR にしない |
| D13 | 対象 Agent | DefaultDQN だけ |
| D14 | 可塑性 probe の行 | `$agent`・500 のまま据え置く（範囲外） |
| D15 | 実装順 | step 0 改名 → step 1 capture の一般化 → step 2 attention（smoke・ON/OFF 等価性・文書同期まで含めて完成）。コミットは分けてよいが、完成の単位は step 2 まで |

## 9. 複雑さ監査（最終簡素化パス、2026-10-16）

ゴールアンカーを物差しに、導入する機構を逆方向から問うた結果。

| 機構 | 判定 | 理由 |
|---|---|---|
| attention を明示計算する純関数 | keep | 値が出ない |
| 鍵要求型 capture（一般化） | keep | ForwardUpTo に固有目的の引数を付けない。概念 1 つ。編集量の少ない別型案より概念の少なさを優先 |
| 未達 error・入口 Reset | keep | 既存の未知 branch 名 error の置き換え。Transformer 無し閉包の fail-fast を無料で得る |
| branch 名の `/` 禁止 | keep | 衝突は起きていない（想像上）。3 行で規約を構造にする |
| Context のポインタ 1 本 | keep | module の署名を変えずに届ける唯一の経路 |
| 層ごとの鍵と nn_util の畳み | keep | どの層が崩れるか。純関数で単体テスト |
| 専用 `probe.batch_size` と RNG | keep | 4096 流用で 1 層 1GB |
| プロファイル `@attention`・Run プロファイル 2 本 | keep | 費用を確かめる前に既定 ON にしない |
| `BatchUpdateResult` への搭載 | keep | 新しい 4 診断と同じ読み方 |
| 最初の測定での範囲 fail-fast | keep（「学習開始前」から shrink） | 黙って NaN を防ぐ。dry-run は API が増える |
| キー改名（ImageCls 含む） | keep | 用途ラベル下の共用キーを解消 |
| `migrate.py` への改名規則 | shrink（置かない） | 演算子を整える道具で役割が違う。CHANGELOG の注記と golden の採り直しで足りる |
| 用語 3 つ・ADR 1 本・0031 注記 | keep | 取捨選択を後から辿れるように |
| `attention_entropy_min` | cut | 痛み無し |
| head 単位の統計 | defer-behind-gate | 層平均が下がる Run が出て内訳が要る日 |
| 専用の到達キー | defer-behind-gate | Transformer が feature 閉包の外に置かれた日 |
| 学習開始前の dry-run | cut | 最初の測定で足りる |
| TraceCallback の統合 | defer-behind-gate | 999 PRD。trace に要求フィルタが要る日、名前付き出力の 3 つ目が要る日 |
| 可塑性 probe 行の `$update_result` 化 | 範囲外 | 動くものを触らない |

最小解との差分: pin された要件（層ごとの正規化値を記録・学習経路不変・汎用 forward に固有引数を付けない・設定キーを用途ラベルの下に置かない・新 4 診断と同じ読み方・添字の fail-fast・専用 probe サイズ）だけから引き直した最小設計との差は、`migrate.py` の規則（置かない）だけで、他は keep の集合と一致した。

段階の独立性: step 0・1・2 は各々止めても悪化しない。step 1 だけで止まった状態（利用者が可塑性だけの一般化）は長く置かない。

成功の測りやすさ: 同 seed・決定論設定の ON/OFF で metrics checksum 一致、`inspect_run.py tags` の 4 本が 0〜1、`rg plasticity.feature_key` と `rg NetworkBranchCapture` がコード・設定・設計文書でゼロ（実験記録の dump と done PRD を除く）、プロファイラ比 ≤ 1% の報告、2 つ目の module 診断を足すときの `nn.hpp` 変更が鍵定数 1 行、100M DropMerge と Atari `@vitva` の判定が実験記録に日付付きで残る。

## 10. 文書同期（実装時に更新するもの）

- [130_neural_networks.jp.md](../design/130_neural_networks.jp.md): capture の節を `NetworkForwardCapture`（鍵要求、body と module の 2 書き手、入口 Reset・出口 error、`/` 禁止、鍵定数）に書き換え、`ForwardUpTo` が capture を受けることを足す。TraceCallback の節は据え置く。
- [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md): 可塑性の節の `feature_key` を `final_feature_branch_key` に読み替え、attention エントロピー probe の節を新設（測り方、値の定義、`BatchUpdateResult` の欄、fail-fast）。
- [030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md): `48_agent_attention` の読み方（0〜1、1 が一様、下がり続ける層と成績の時期、TQL の 0.8 に相当する目安、head 平均の TQL 値とは比べないこと）と、`final_feature_branch_key` への改名。PRD・ADR を参照させない。
- [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt): `@attention` の 4 行。
- [agent.txt](../../apps/runner/config/agent.txt): キー改名 1 行と `attention_entropy.probe.batch_size` 1 行。[ImageCls.txt](../../apps/runner/config/ImageCls.txt): キー改名 1 行。
- [DropMerge.txt](../../apps/runner/config/DropMerge.txt)・[Atari.txt](../../apps/runner/config/Atari.txt): `run.@attention`。
- [CONTEXT.md](../../CONTEXT.md): 用語 3 つと「weight norm 分割」の読み替え。
- [ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) 新規、[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) 先頭の注記。
- `core/anet-core/testdata/prd072/`: golden 17 本の採り直し（README の手順）。
- `core/anet-core/testdata/prd093/README.md`: smoke・ON/OFF 等価性・性能測定の手順と確認条件。
- `CHANGELOG.md`: 指標の追加と購読による有効化、キーの改名と過去 dump を再生するときの読み替え。
- 英訳（`.en.md`）は `anet-translate-docs` の通常運用に任せる。

## 11. 依存と順序

- step 0: キーの改名（独立コミット。config dump の差分がキー名だけであることで等価性を確認）。
- step 1: capture の一般化と `ForwardUpTo` の配線、`/` 禁止、既存テストの置き換え。
- step 2: エントロピーの純関数、TransformerEncoder の書き込み、Learner の probe と `BatchUpdateResult`、購読の解釈、metrics 行とプロファイル、テスト、smoke、ON/OFF 等価性、性能、文書同期。ここまでで完成。
- [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) とは独立。
- [999_nn_trace_capture_unification_10prd.md](999_nn_trace_capture_unification_10prd.md) は本 PRD の capture がコミットされた後、ゲートが開いたとき。
- 本 PRD の指標が段階 2 の作る条件を判断する材料になるので、正則化より先に行う。

## 12. 参照

- [reports/dong2026_tql_paper_2026-10-05.md](../../reports/dong2026_tql_paper_2026-10-05.md) の 4.2 節（測り方と誘導）、5.2 節（崩壊と成績の相関）、5.5 節（層ごとの温度の効果）、付録 C.2（head 平均）、キューの論点への答え。
- Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse", ICML 2023（head ごとの attention エントロピーの追跡）。<https://proceedings.mlr.press/v202/zhai23a.html>
- [CONTEXT.md](../../CONTEXT.md) の「部分 forward」「probe」「購読ヒント」「capture」「final feature branch」「attention エントロピー」。
