# TransformerEncoder の attention エントロピー診断 PRD

> 2026-10-16 追加グリル: 観測の解釈を層平均の集中度の推移に限定し、測定を更新前へ統一。capture は層ごとの正規化済み FP32 scalar Tensor へ縮小。PRD072 の固定入力・golden は保持し、長期 Run の分析を機能完了条件から分離した。0.8 は比較・閾値に使わない論文の背景説明として残す。§8・§9 は追加裁定を反映した現行の決定と監査結果。
> 2026-10-16 グリル（初回）: 暫定の既定をすべて裁定に置き換えた（§8）。取り出し口は capture の一般化（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)）、設定キーは `final_feature_branch_key` へ改名、`attention_entropy_min` は cut、公開は `$update_result`・interval 503、層番号は 0 始まり。TraceCallback の統合は [999_nn_trace_capture_unification_10prd.md](999_nn_trace_capture_unification_10prd.md) へ先送り。
> 2026-10-10 起票（暫定）。
> 起点: TQL(Dong et al. 2026、ICML 2026)の論文解説 [reports/dong2026_tql_paper_2026-10-05.md](../../reports/dong2026_tql_paper_2026-10-05.md)。Transformer の価値関数を大きくすると、層の attention が少数のトークンへ偏り（エントロピーの崩壊）、成績が下がるという報告。
> 関連: [done/062_plasticity_metrics_10prd.md](done/062_plasticity_metrics_10prd.md)（可塑性メトリクス。測り方の型を借りる）、[done/012_sdpa_attention_10prd.md](done/012_sdpa_attention_10prd.md)（self-attention の SDPA 化）、[084_noisynet_10prd.md](084_noisynet_10prd.md)、[092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md)（公開の形の先例）、[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)、[ADR 0004](../adr/0004-sdpa-attention-via-aten.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)。

## 1. 目的と問題

TransformerEncoder の各層で attention がどれだけ少数のトークンへ偏っているかを、層平均のスカラー指標として学習中に記録する。学習を変えずに「層平均で集中の進行が見えた／見えなかった」を読み、成績の頭打ちや下降と時期が重なるかを調べられるようにする。一部 head の集中が存在しないことや、集中が成績低下の原因であることまでは、この指標だけから断定しない。

本 PRD でいう「集中の進行」は、層平均の正規化済み attention エントロピー（1 が一様、0 が 1 トークンへの集中）が下がり続けることを指す。「attention collapse」という現象の判定を意味しない。

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
- 学習経路の外で測る仕組みは可塑性メトリクスにある（[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)）。ReplayBuffer の一様・非復元の sample を、部分 forward（`ForwardUpTo`、NoGrad・eval mode、Learner と同じ autocast）へ通す。購読された指標の interval でだけ測り、学習用の乱数を消費しない。probe の大きさは `learner.<対象>.probe.batch_size` の形で持つ（`plasticity` 4096、`policy_churn` 1024、`replay_fit` 1024、`noisy_probe` 128）。
- forward から出力のほかに持ち帰る口は `NetworkBranchCapture`（branch 名 1 つと出力 1 つ。body が branch ループの末尾で state から detach して書く）で、`Network::Forward` と実行器の `Forward` だけが受ける。`ForwardUpTo` は受けない。[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) 以降、`ForwardUpTo` は実行情報を受けず既定 Context（Noisy は μ-only）で動き、module は State のポインタと呼び出し情報を束ねた `NetworkForwardContext` を受け取る。
- `TraceCallback` は名前付き tensor を外へ出すもう 1 つの口だが、要求の仕組みが無く、Actor が `MakeAction` で毎回渡す。追加計算を伴う値をここへ乗せると、購読ゼロでも Actor の毎 forward に計算が乗る（ADR 0031 の A''）。
- `learner.plasticity.feature_key`（既定 `main_feature`）は、可塑性 probe の到達 branch、重みノルムの feature/readout 分割、Noisy の σ 統計、勾配ノイズの群分けの 4 診断が「feature 閉包」として共用している。用途ラベル `plasticity.` の下に共用のものが置かれた形である。
- 新しい診断（policy churn、replay_fit、Noisy、勾配ノイズ）は、購読の発火する更新の中で測り、その `BatchUpdateResult` に値を載せて `$update_result` で読む。interval は 503（素数。hard update C=200 と位相が固定されるのを避け、同じ `learn_step` で横に読む）。可塑性 probe だけが `$agent` の最新値で interval 500 である。
- TQL の測り方: 層 $\ell$ のクエリ $i$ の attention 分布 $A^\ell_{i\cdot}$ のエントロピー $H^\ell_i = -\sum_j A^\ell_{ij}\log A^\ell_{ij}$ を、クエリで平均して層の値 $H^\ell$ とする。一様なら最大の $\ln n$（$n$ はトークン数）、1 トークンへの集中で 0 になる。TQL の実験は head で attention を平均してからエントロピーを取る（付録 C.2）。TQL の目標エントロピーの初期値は $0.8\ln n$ である。これは論文の背景説明であり、集約方法の異なる本指標との比較、正常判定、警告閾値には使わない。
- attention entropy collapse という語を作った Zhai ら（ICML 2023、σReparam）は、学習中に head ごとの attention エントロピーを追跡し、低エントロピーの状態を崩壊と呼んで学習の不安定化との対応を示した。head ごとの定義が文献の標準で、TQL の head 平均が変則である。

## 3. ゴールと非ゴール

ゴール:

1. TransformerEncoder の層平均の attention エントロピーを、学習の `learn_step` 軸のスカラーとして記録し、集中度の推移と成績の時期を対応させて読めるようにする。
2. 学習の forward の計算と数値（SDPA の経路）を変えない。指標を購読しない構成では計算も起きない。

非ゴール（§7 に、作る条件を書いた）:

- エントロピーの正則化（TQL の本体。層ごとの温度と補助損失）。
- 価値を読むトークン（行動ごとのクエリトークン）を置く構造。
- attention マップの画像化（GUI や Viewer での表示）、head 単位の値の公開。
- 学習の forward から取る値（可塑性メトリクスでいう actual チャネル）。
- target network の測定、DefaultDQN 以外の Agent。
- TraceCallback の capture への統合（999 PRD）。

## 4. 決定した設計

1. **測り方**: 購読された指標が発火する `learn_step` のパラメータ更新前に、ReplayBuffer から一様・非復元で probe の sample を取る（専用 RNG。足りなければその更新は測らない）。既存の観測の正規化を通し、online network を部分 forward で `learner.final_feature_branch_key` の branch まで実行する（NoGrad・eval mode・Learner と同じ autocast・既定 Context）。このときだけ、閉包の中の各 TransformerEncoder 層で $\mathrm{softmax}(QK^\top/\sqrt{d_h})$ を明示的に計算する。SDPA の経路は触らない。既存の可塑性 probe・重みノルム・Noisy 診断・replay fit と重みの時点を揃えるが、入力は専用サンプル、実行モードは eval であり、学習 forward そのものの値ではない。IQN でも `taus` は要らない（final feature branch で止まるため）。probe は学習用 RNG、ReplayBuffer の優先度・sampled-once 統計、正規化器や network の buffer 状態を更新せず、終了時には元の実行モードを復元する。
2. **値**: head ごと・クエリごとの attention 分布のエントロピーを sample・head・クエリで平均し、$\ln n$（$n$ は系列長）で割って 0〜1 にそろえる。1 が一様、0 が 1 トークンへの集中。$n < 2$ は NaN。head で attention を平均してからエントロピーを取る TQL の定義は採らない（平均分布のエントロピーはエントロピーの平均以上になり、head ごとに別のトークンへ崩壊した状態を隠す）。Zhai らの head ごとの定義に従う。エントロピーの計算は `nn_util` の純関数の中で autocast を切り、Q・K の射影から softmax、エントロピー、平均・正規化まで FP32 で完結させる。各層の attention が実際に受け取る入力（pre-LN は正規化後、post-LN は正規化前）と、SN 有効時は実効重みを使う。診断計算は学習グラフを保持せず、forward 自体は Learner の精度のまま。公開するのは層平均なので、例えば 8 head 中 7 head が一様で 1 head だけが完全に集中しても値は 0.875 となる。一部 head の集中の有無を判定する値ではない。
3. **取り出し口**: `NetworkBranchCapture` を、呼び出し側が鍵の集合で要求する `NetworkForwardCapture` に一般化する（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)）。操作は `Wants`・`Add`・`Get`・`Unfulfilled`・`Reset`。書き手は 2 種で器は区別しない。body は branch ループの末尾で、要求鍵のうち branch 名に当たるものを detach して `Add`（今の動作そのまま）。module は forward 中に、自分が所有する鍵が `Wants` なら計算して `Add`。鍵は書き手の公開定数で、module の鍵は型名を含む（`TransformerEncoder/attention_entropy`）。branch 名には `/` を使えない（Network 構築時に reject。衝突が起きたことはなく、規約を構造にするための番人）。`Forward`・実行器の `Forward`・`ForwardUpTo` が同じ型を受け、内部は `NetworkForwardContext` にポインタ 1 本を足して module まで運ぶ。入口で `Reset`、出口で `Unfulfilled` が空でなければ鍵名と利用できる branch 名を添えて `ANET_SYSTEM_ERROR`。渡さない forward は module で null 判定 1 回。TransformerEncoder は `nn_util` の純関数が返す層ごとの正規化済み FP32 scalar Tensor を、`TransformerEncoder/attention_entropy` という共通の鍵へ forward 順に `Add` する。Tensor は GPU 実行時には GPU 上に保持し、確率行列や sample・head・query ごとの詳細 Tensor は capture へ搬送しない。Learner は層別 scalar をまとめて CPU へ転送して更新結果に載せ、平均・正規化を再度行わない。層ごとの CPU 同期を増やさない。
4. **層の番号**: closure の中で実行される TransformerEncoder の層を、実行順に 0 から数えた通し番号。checkpoint の `layer_0` と同じ 0 始まり（`ram_metric.[n]` の 1 始まりは 0 を「番号なし」に使うためで、層には当てはまらない）。
5. **公開**: 測定した更新の `BatchUpdateResult` に、その更新前の層ごとの値を載せ、`GetScalar("attention_entropy.[n]")` で返す。測らない更新は NaN。metrics の定義に新しいプロファイル `metrics.scalar.@attention` を置き、群は `48_agent_attention`、interval は 503。baseline には入れない。現行の 4 層構成に合わせて 4 行を置く。

   ```properties
   metrics.scalar.@attention : [48_agent_attention/01_entropy_layer0] ?= attention_entropy.[0] @learn $learn_step $update_result interval:503
   metrics.scalar.@attention : [48_agent_attention/02_entropy_layer1] ?= attention_entropy.[1] @learn $learn_step $update_result interval:503
   metrics.scalar.@attention : [48_agent_attention/03_entropy_layer2] ?= attention_entropy.[2] @learn $learn_step $update_result interval:503
   metrics.scalar.@attention : [48_agent_attention/04_entropy_layer3] ?= attention_entropy.[3] @learn $learn_step $update_result interval:503
   ```

   Run への追加は `@gradnoise` と同じく `metrics.scalar.$` のチェーンへ足す Run プロファイル `run.@attention` を DropMerge.txt と Atari.txt に置く。tag は群名と `/` を含めて現行規約の 40 文字以内に収める（本 PRD の tag は 36 文字）。
6. **設定**:

   ```properties
   DefaultDQNAgent.@baseline : learner.attention_entropy.probe.batch_size ?= 256
   DefaultDQNAgent.@baseline : learner.final_feature_branch_key ?= main_feature
   ImageClsAgent.final_feature_branch_key = main_feature
   ```

   `probe.batch_size` は 1 未満で読み込み時に `ANET_SYSTEM_ERROR`。可塑性の 4096 を流用しないのは、attention の確率行列が sample 数 × head 数 × $n^2$ になるからで、`@vitva`（121 トークン・4 head）では FP32 の確率行列だけで 4096 件時に 1 層あたり約 1GB、256 件時に約 60MB になる（層ごとに捨てる。処理全体のピークメモリではない）。256 は一時メモリの負荷を抑えるための変更可能な既定値であり、統計精度は保証しない。同じ sample 内の head・query を独立した観測として数えず、推定の揺れが実験の判断を妨げたときに件数比較や追加統計を検討する。
   `final_feature_branch_key` は `plasticity.feature_key` の改名で、4 診断と本 PRD が共用する「観測から計算された特徴が集約される feature 側の最後の branch」を指す。移行はクリーンブレーク: 現用のコード・設定・テスト・サンプル・ドキュメントを同じ変更で移行し、旧キーの検出・別名・互換読みは置かない。`agent.txt`・`ImageCls.txt` の設定と 030 ガイドを更新し、CHANGELOG に改名と「過去 dump を再生するときの読み替え」を記す。改名単独の前後で現用設定の config dump を比較し、差分がキー名だけであることを確認する。PRD072 の固定入力・golden 17 本・manifest は resolver の過去入力の検証資料として保持し、今回の改名では採り直さない。実験記録の dump などの履歴資料も移行対象から除く。
7. **fail-fast**: `attention_entropy.[n]` の n が整数でない・負なら購読の解釈時（学習開始前）。n が層数以上なら最初の測定で、鍵・層数・許される範囲を添えて止める。Transformer の無い閉包で購読すると、capture の未達規則で最初の測定の forward の出口で止まる。
8. **対象**: DefaultDQN（TD/QR/IQN、共有 Learner）。ImageCls は ReplayBuffer が無く probe の形が合わない。Rainbow は現行 env 設定に無く検証できない。

## 5. 機構ごとの根拠

| 機構 | 根拠（起きたこと・測った量） | 裁定 |
|---|---|---|
| probe での層ごとの測定 | DropMerge の本線と Atari の常設プロファイルが Transformer を含むのに、内部を見る指標が無い。Atari の ViT 混成が V/A 基準を下回った原因を切り分けられていない。成績で構成を判断するには DropMerge で 1 腕 50〜100M かかる | 作る |
| 測定のときだけ attention を明示的に計算する経路 | SDPA は重みを返さない（ADR 0004）。学習の forward を変えないため | 作る |
| 層ごとの値 | TQL の切り分けで、温度を全層で 1 つにすると 26M の成功率が 41%、層ごとだと 67% と差が出た。層平均で集中が進む層を特定するには層ごとの値が要る。費用は層数分のスカラー | 作る |
| head ごとのエントロピー | TQL の head 平均は head ごとに別のトークンへ崩壊した状態を隠す（凹性）。Zhai らの先例 | 作る |
| 鍵要求型 capture への一般化 | 汎用の `ForwardUpTo` に固有目的の引数を付けない。forward の持ち帰りの概念を 1 つに保つ | 作る（ADR 0055） |
| 層ごとの正規化済み scalar の搬送 | 今回の利用は層平均だけ。詳細 Tensor を持ち帰る現用の用途は無い | scalar に縮小 |
| 専用の probe の大きさ | 可塑性の 4096 では確率行列が 1 層 1GB 級 | 作る。256 件の統計精度は保証しない |
| 新プロファイル `@attention`（baseline 外） | 費用を確かめる前に既定 ON にしない規約。Transformer の無いネットでは購読できない | 作る |
| 設定キーの改名 | 用途ラベル `plasticity.` の下に 5 診断が共用するキーがあるのは「設定階層はフレームワーク概念」の規約に反する | 作る（step 0） |
| 層の最小値 `attention_entropy_min` | 痛みの記載なし。4 層なら 4 本の線で読める。min はどの層かを隠す | 作らない。6 層構成で読みにくくなったら戻す |
| head 単位の値の公開（head 最小値、閾値以下 head 割合） | 層平均が下がったとき内訳が要るかはまだ分からない。将来のための詳細 Tensor は保持せず、必要になったときに診断値と搬送契約を設計する | 作らない。層平均が下がる Run が出て内訳が要る日に |
| 専用の `attention_entropy.feature_key` | 今の 2 構成は Transformer が feature 閉包の中。別の閉包を測りたい痛みは無い | 作らない。Transformer が feature 閉包の外に置かれた日に（段階 3 の行動クエリ、IQN fusion 後の Transformer、可塑性側が閉包を変えるとき） |
| 学習開始前のダミー forward で層数を数える | 黙って NaN が続く痛みは最初の測定の fail-fast で防げる。API が 1 つ増える | 作らない |
| TraceCallback の統合 | 1 関心 2 機構だが、pin された痛みは未計測 | 作らない。999 PRD にゲートを書いた |
| 学習の forward から取る値（actual） | まだ要る理由が無い | 作らない。probe で層平均の集中の進行が見え、train mode の値との違いを確かめる必要が出たら |
| エントロピーの正則化 | anet ではまだ本指標による層平均の集中の進行を観測していない | 作らない。§7 の条件を満たしたら別の PRD で |

## 6. 受入基準

本節は診断機能の完了条件とする。長期 Run の分析、集中の発見、成績改善は機能の合否に含めない。

1. 単体テスト（`nn_test.cpp`）
   - capture: `Wants`・`Add`・`Get`・`Unfulfilled`・`Reset`。既存 branch 出力の値・detach を維持し、再利用時には前回の値が残らない。未要求の鍵への `Add` は error、未達の要求は Network の出口で error、渡さない forward は不変（既存の `NetworkBranchCapture` テストの置き換え）。名前に `/` を含む branch は構築時に error。
   - エントロピー関数: 一様な attention 分布で正規化値 1.0、1 トークンへの強い集中で 0 に近い、参照計算と数値誤差の許容範囲で一致、系列長 1 で NaN。pre/post-LN のそれぞれで attention の実際の入力を使い、SN 有効時は実効重みで一致する。
   - 2 層の小さな Transformer で `ForwardUpTo` に capture を渡すと、正規化済み FP32 scalar Tensor が層数分 forward 順に積まれる。診断値は学習グラフを保持しない。
2. Learner テスト（`dqn_based_agent_test.cpp`）: 購読 interval の更新前にだけ測定し、その更新の `BatchUpdateResult` に値が載る。未測定・サンプル不足の更新は NaN で、前回値を持ち越さない。`attention_entropy.[n]` の非整数・負は購読時に error、n ≥ 層数は最初の測定で鍵・層数・範囲を添えて error。Transformer の無い閉包は capture の未達規則で error。購読ゼロでは probe の抽出・追加 forward・診断計算をしない。
3. 改名（step 0）の等価性: 現用設定の改名前後の config dump の差分がキー名だけで、既存テストが通る。PRD072 の固定入力・golden・manifest は保持し、既存の resolver 比較をそのまま使う。固定資料と実験記録の dump などの履歴資料は、現用の旧キー参照が残っていないことを確認する検索の対象から除く。
4. smoke: DropMerge 本線と Atari の `run.@vitva` にそれぞれ `run.@attention` を足して短く回し、両方で `inspect_run.py tags` に `48_agent_attention/` の 4 本が出る。メトリクスの値が 0〜1 の範囲にあり、測定点を記録できることも確認する。
5. ON/OFF 等価性: 同じ seed・決定論の設定で、`@attention` の有無によらず学習由来の metrics checksum が一致する（ADR 0031 の非干渉性を検証）。比較対象を明記し、実時間などの性能値と追加診断値を除く。checkpoint ファイルの SHA は直列化の非決定性により一致しない場合があるため、等価性の判定には使わない。probe を eval mode・NoGrad で実行する前提で、学習用 RNG、ReplayBuffer の優先度・sampled-once 統計、正規化器や network の buffer 状態が probe 単独で変わらず、実行モードが復元されることも確認する。
6. 性能: 測定に `ANET_PROFILE_SCOPE` を入れ、interval 503 での測定費用を報告する。測定時間の合計が Run の実時間の 1% 以下であることは目標であり、超過だけで不合格にしない。

後続実験として、DropMerge 本線の 100M Run と Atari の `run.@vitva` の Run で層平均の推移を読み、成績の頭打ちや下降との時期の対応を実験記録に残す。この実施・分析は別の実験タスクとし、機能完了を待たせない。

- 層平均で集中の進行が見え、成績の頭打ちや下降と時期が重なる → §7 の段階 2 を検討する材料にする。因果関係の確定や改善の保証とは扱わない。
- 層平均で集中の進行が見えなかった → 今回の観測では正則化へ進む根拠が得られなかったと記録する。一部 head の集中が存在しないとは結論しない。

## 7. 先送りした段階と、作る条件

1. 段階 2（TQL の正則化）: 層ごとの温度（学習する値）と目標エントロピーを TransformerEncoder のオプションに持たせ、補助損失を Learner の損失に足す。検討を始める条件: 本 PRD の指標で層平均の集中の進行が見え、成績の頭打ちや下降と時期が重なること。この対応だけで因果関係や改善を断定せず、実装は別の PRD で決める。作るときは、ネットの forward で出た値を Learner の損失へ足す口（今の anet には無い）と、学習の forward でも attention の確率を計算する経路が要る。損失側の定義（head ごとか head 平均か）はそのとき改めて選ぶ。
2. 段階 3（価値を読むトークン）: 行動ごとにクエリトークンを置き、各トークンが盤面のトークンを読んで Q を出す構造。作る条件: 行動クエリの構成を試すと決めたとき。その block は feature 閉包の外になるので、専用の到達キーと別の capture 鍵が要る。
3. ゲート付きの先送り（§5）: head 単位の値の公開、専用の到達キー、TraceCallback の統合（[999_nn_trace_capture_unification_10prd.md](999_nn_trace_capture_unification_10prd.md)）。

## 8. 決定事項（2026-10-16 追加グリル反映）

| # | 論点 | 裁定 |
|---|---|---|
| D1 | ゴール | 層平均の attention エントロピーの正規化値を `learn_step` 軸で記録し、集中度の推移と成績の時期を読む。一部 head の集中の不存在や因果関係は断定しない。学習の forward と数値は不変、未購読なら計算ゼロ |
| D2 | `attention_entropy_min` | cut。4 層なら 4 本で読め、min はどの層かを隠す |
| D3 | 取り出し口 | `NetworkBranchCapture` を鍵要求型の `NetworkForwardCapture` に一般化（ADR 0055）。`Forward`・実行器 `Forward`・`ForwardUpTo` が同じ型を受け、内部は Context のポインタ 1 本。入口 Reset・出口で未達 error。branch 名の `/` 禁止 |
| D4 | capture の中身 | nn_util の純関数で平均・正規化まで完結し、TransformerEncoder が層ごとの正規化済み FP32 scalar Tensor を forward 順に積む。共通の鍵は `TransformerEncoder/attention_entropy`。CPU 転送は全層をまとめ、Learner では平均しない |
| D5 | 到達 branch と層番号 | `learner.final_feature_branch_key` の依存閉包。閉包内の TransformerEncoder 層を実行順に 0 から |
| D6 | 設定キーの改名 | `learner.plasticity.feature_key` → `learner.final_feature_branch_key`、ImageCls も同時。現用箇所をクリーンブレークで移行し、旧キーの互換処理は置かない。step 0 は独立した完成単位。PRD072 の固定資料は保持し、現用設定の前後比較で検証 |
| D7 | 値の定義 | head ごとのエントロピーを head・クエリ・sample で平均、ln n で正規化、n < 2 は NaN。Zhai らの定義。forward は Learner 精度、エントロピー計算は FP32 |
| D8 | 公開 | `$update_result`、interval 503、群 `48_agent_attention`、プロファイル `@attention`（baseline 外）、`attention_entropy.[n]` 0 始まり、tag は 40 文字以内 |
| D9 | probe の大きさ | `learner.attention_entropy.probe.batch_size ?= 256`、専用 RNG、足りなければ NaN。256 は変更可能な既定値であり統計精度は保証しない |
| D10 | fail-fast | 非整数・負は購読時。n ≥ 層数は最初の測定。Transformer 無し閉包は capture の未達規則。batch_size < 1 は読み込み時 |
| D11 | 用語 | 初回に CONTEXT.md へ反映済みの「attention エントロピー」「final feature branch」「capture」と「weight norm 分割」の読み替えを維持。今回の縮小で用語は追加しない |
| D12 | ADR | 初回に記録済みの ADR 0055 と ADR 0031 先頭の注記を維持。今回の縮小では ADR を追記・新設しない |
| D13 | 対象 Agent | DefaultDQN だけ |
| D14 | 可塑性 probe の行 | `$agent`・500 のまま据え置く（範囲外） |
| D15 | 実装順と完成単位 | step 0 の改名は独立。step 1 capture の一般化と step 2 attention は一つの機能完成単位とし、テスト・smoke・ON/OFF 等価性・性能報告・文書同期までで完成。長期実験は別タスク |
| D16 | 測定時点 | その learn_step のパラメータ更新前。専用サンプルの eval 部分 forward であり、学習 forward そのものの値ではない |
| D17 | 0.8 の説明 | 論文の背景説明として残す。集約方法が異なるため、本指標との比較・正常判定・警告閾値には使わない |

## 9. 複雑さ監査（最終簡素化パス、2026-10-16 追加グリル反映）

ゴールアンカー「層平均の attention 集中度を、学習を変えずに記録し、成績推移との対応を調べられるようにする」を物差しに、初回の裁定を追加グリルで見直した結果。

| 機構・要件 | 判定 | 理由 |
|---|---|---|
| attention を明示計算する純関数 | keep | SDPA は確率分布を返さず、削ると診断値が出ない |
| 鍵要求型 capture（一般化） | keep | ForwardUpTo に固有目的の引数を付けず、持ち帰りの概念を 1 つに保つ（ADR 0055） |
| 未達 error・入口 Reset | keep | 指定ミスや前回値の残留を防ぎ、Transformer 無し閉包も同じ規則で検出する |
| branch 名の `/` 禁止 | keep | 衝突の実例は無いが、branch と module の鍵の名前空間を分ける局所的な構文契約として維持 |
| Context のポインタ 1 本 | keep | module へ既存の Context 経由で要求を届ける |
| sample・head・query ごとの詳細 Tensor の搬送 | shrink | 今回必要なのは層平均だけ。nn_util で平均・正規化し、共通の鍵へ層ごとの scalar を積む |
| 専用 `probe.batch_size` と RNG | keep | 4096 件では確率行列が 1 層 1GB 級。学習用の抽選にも干渉させない |
| 256 件なら推定の揺れは問題ないという保証 | cut | 未検証。同じ sample 内の head・query は独立した観測ではない。変更可能な既定値としてのみ採用 |
| プロファイル `@attention`・Run プロファイル 2 本 | keep | 購読で有効化し、費用を確かめる前に既定 ON にしない |
| `BatchUpdateResult` への搭載・更新前の測定 | keep | 測定した更新に属する値とし、既存診断と重みの時点を揃える |
| 最初の測定での範囲 fail-fast | keep（初回に shrink） | 黙って NaN が続く指定ミスを防ぐ。層数確認だけの dry-run は増やさない |
| キー改名（ImageCls 含む） | keep | 既に複数診断が共用する境界を表す名前にする。単独でも価値があり、step 0 として独立 |
| PRD072 の固定入力・golden の再採取 | cut | resolver の契約は変わらない。過去入力の固定資料を保持し、現用設定の前後を別途比較する |
| `migrate.py` への改名規則 | cut | 演算子を整える道具で役割が違う。現用箇所の移行と CHANGELOG の注記で足りる |
| 用語 3 つ・ADR 0055・0031 注記 | keep | 初回に記録済みの判断を維持。今回の縮小で新しい用語・ADR は増やさない |
| `attention_entropy_min` | cut | 4 層は 4 本で読め、最小値はどの層かを隠す |
| head 単位の統計 | defer-behind-gate | 層平均が下がる Run が出て内訳が必要になったとき。詳細データは先回りして保持しない |
| 専用の到達キー | defer-behind-gate | Transformer が feature 閉包の外に置かれたとき |
| 学習開始前の dry-run | cut | 最初の測定での fail-fast で足りる |
| TraceCallback の統合 | defer-behind-gate | 999 PRD。trace に要求フィルタが要るとき、名前付き出力の 3 つ目が要るとき |
| 学習 forward から取る actual・正則化 | defer-behind-gate | probe の観測からそれぞれ §5・§7 の必要性が生じたとき |
| 長期 Run の実施・分析を機能完了条件に含める | cut | 診断機能の正しさと、研究上の有用性の評価を分ける。長期分析は後続実験 |
| 可塑性 probe 行の `$update_result` 化 | 範囲外 | 既存の公開方法を変える必要はない |

要件の根拠: 今回追加・明確化した層平均という解釈範囲、更新前という測定時点、scalar への集約は、現在の利用と公開値の意味に必要なもの。統計精度の保証や、将来の head 統計を見越した詳細データの保持は要件に含めない。

前提変更後の整理: 詳細 Tensor の搬送・Learner での平均・将来のための保持説明を削除する。「崩壊していない」という広い結論は、層平均で集中の進行が見えなかったという観測へ限定する。0.8 は論文の背景説明として残し、異なる集約方法の本指標の比較・正常判定・警告閾値には使わない。

最小解との差分: 必要な構成は、更新前の専用 probe、層ごとの正規化済み scalar を運ぶ共通 capture、購読駆動と更新結果への格納、指定ミスの検証、既定設定、共用キーの改名で満たす。初回案から詳細 Tensor の搬送を縮小し、固定 golden の再採取を削除し、長期実験を機能完了から分離する。これ以外の診断機構は追加しない。

段階の独立性: 完成単位は「step 0 のキー改名」と「step 1 の capture 一般化＋step 2 の attention 診断」の 2 つ。改名だけで止めても既存診断の設定名の整理が残る。capture 一般化の作業・レビューは分割できるが、それだけを attention 診断の機能完了とは扱わない。

成功の測りやすさ: §6 の数値・capture・購読のテスト、同 seed・決定論設定での学習由来の metrics checksum の ON/OFF 一致、両 smoke Run での 4 層の記録、改名単独の config dump 比較、測定費用の報告で判定する。現用コード・設定・テスト・設計文書から旧キーと旧 capture 型を除き、固定資料（PRD072 の入力・golden を含む）・実験記録の dump・ADR・過去 PRD などの履歴資料は検索対象から除く。1% は費用の目標であり、集中の発見・成績改善・長期 Run の分析は機能完了の条件にしない。

## 10. 文書同期（実装時に更新するもの）

- [130_neural_networks.jp.md](../design/130_neural_networks.jp.md): capture の節を `NetworkForwardCapture`（鍵要求、body と module の 2 書き手、入口 Reset・出口 error、`/` 禁止、鍵定数）に書き換え、`ForwardUpTo` が capture を受けることを足す。attention 診断は共通の鍵へ層順に正規化済み scalar を積み、平均・正規化は nn_util で完結することを記す。TraceCallback の節は据え置く。
- [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md): 可塑性の節の `feature_key` を `final_feature_branch_key` に読み替え、attention エントロピー probe の節を新設（更新前の測定、層平均の定義と解釈範囲、全層まとめての CPU 転送、`BatchUpdateResult` の欄、fail-fast、未測定時の NaN）。
- [030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md): `48_agent_attention` の読み方（0〜1、1 が一様、層平均の集中度の推移と成績の時期、一部 head の集中の不存在や因果関係は断定しないこと）と、`final_feature_branch_key` への改名。TQL の 0.8 は論文の背景説明としてのみ残し、本指標とは集約方法が異なるため比較・正常判定・警告閾値には使わないと明記する。PRD・ADR を参照させない。
- [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt): `@attention` の 4 行。
- [agent.txt](../../apps/runner/config/agent.txt): キー改名 1 行と `attention_entropy.probe.batch_size` 1 行。[ImageCls.txt](../../apps/runner/config/ImageCls.txt): キー改名 1 行。
- [DropMerge.txt](../../apps/runner/config/DropMerge.txt)・[Atari.txt](../../apps/runner/config/Atari.txt): `run.@attention`。
- [CONTEXT.md](../../CONTEXT.md): 初回に反映済みの用語 3 つと「weight norm 分割」の読み替えを維持する。
- [ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) と [ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) 先頭の注記: 初回に記録済みの判断を維持する。今回の縮小では追記・新設しない。
- `core/anet-core/testdata/prd072/`: 固定入力・golden 17 本・manifest を保持する。今回の改名では再採取しない。
- `core/anet-core/testdata/prd093/README.md`: 現用設定の改名前後比較、smoke、ON/OFF 等価性の比較対象、状態の非干渉性、性能測定の手順と確認条件。長期 Run の分析は機能完了条件と分ける。
- `CHANGELOG.md`: 指標の追加と購読による有効化、キーの改名と過去 dump を再生するときの読み替え。
- 英訳（`.en.md`）は `anet-translate-docs` の通常運用に任せる。

## 11. 依存と順序

- 完成単位 1（step 0）: キーの改名。現用箇所と文書を同時に移行し、現用設定の config dump の差分がキー名だけであることで等価性を確認する。PRD072 の固定資料は保持する。この段階は独立して完了できる。
- 完成単位 2（step 1＋step 2）: capture の一般化と attention 診断を一つの機能として完成させる。step 1 は `ForwardUpTo` の配線、`/` 禁止、既存 capture 利用側・テストの移行。step 2 は平均・正規化まで完結する純関数、TransformerEncoder の scalar 書き込み、更新前の probe と `BatchUpdateResult`、購読、metrics 設定、テスト、両 smoke、ON/OFF 等価性、性能報告、文書同期。作業・レビュー・コミットは分割できるが、capture 一般化だけを診断機能の完了とはしない。
- 後続実験: DropMerge 100M と Atari `@vitva` の長期 Run の実施・分析は別タスクで行う。機能完了を待たせず、その結果を §7 の検討材料にする。
- [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) とは独立。
- [999_nn_trace_capture_unification_10prd.md](999_nn_trace_capture_unification_10prd.md) は本 PRD の capture がコミットされた後、ゲートが開いたとき。
- 本 PRD の指標が段階 2 の作る条件を判断する材料になるので、正則化より先に行う。

## 12. 参照

- [reports/dong2026_tql_paper_2026-10-05.md](../../reports/dong2026_tql_paper_2026-10-05.md) の 4.2 節（測り方と誘導）、5.2 節（崩壊と成績の相関）、5.5 節（層ごとの温度の効果）、付録 C.2（head 平均）、キューの論点への答え。
- Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse", ICML 2023（head ごとの attention エントロピーの追跡）。<https://proceedings.mlr.press/v202/zhai23a.html>
- [CONTEXT.md](../../CONTEXT.md) の「部分 forward」「probe」「購読ヒント」「capture」「final feature branch」「attention エントロピー」。
