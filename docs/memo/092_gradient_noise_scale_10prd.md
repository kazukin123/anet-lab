# 勾配ノイズの大きさ（gradient noise scale）の計測 PRD

> 2026-10-15 起票（暫定）。同日に初回グリル（`grill-with-docs` の anet 版手順: ゴールアンカー → スコープスクリーン → 詳細 6 問 → 最終簡素化パス 6 項目）で確定し、同日の追加グリルで以下の現行仕様へ改訂した。
> 初回改訂の要点: 本線（DropMerge）の Learner が BF16 AMP ON だと config dump で確認し、AMP を非ゴールから外して BF16 を測る（FP16+GradScaler だけ対象外）。
> 測るのは NoisyNet の ε を更新と同じ 1 抽選に固定した条件付きのサンプルノイズだと明記し、GNS は機序の観測量で因果は分岐 Run で検証すると位置づけた。
> 初回は Adam 前処理版を主読みにして公開 key を 11 本に整理、interval は 503、`Optimize` を 2 引数化して測定を 1 箇所に置き、純粋計算部は nn_util へ。
> 2026-10-15 追加グリル: feature / readout の Adam raw 4 key を追加し、全 15 指標を一式で完成させる。購読の和集合で測定・EMA 更新し、公開は各 key の発火時だけとする。使用パラメータと Adam state を今回の勾配・群ごとに判定し、未成立を前回値で埋めない。
> 完成範囲は計測機能の検証・本線 smoke・性能報告まで。決定論環境で非干渉を検証し、費用 2% 以下は目標とする。因果を評価する分岐実験、BatchNorm の専用禁止機構、追加状態指標、将来 Agent 用 API は今回の範囲に含めない。
> 2026-10-15 追加グリル後のレビュー反映: 空群の raw を NaN に統一して値の未成立率（§7.3）と整合させ、`grad_noise_*` の購読間隔が混在するときの起動時 1 回の WARN を足し、外部参照の版を使用中の LibTorch 2.11 に揃え、§13 を 1 つの一覧にした。履歴（§9.2・§10.1）はそのまま。
> 2026-10-15 3 回目のグリル（2 回目の追加グリル）: 機構は増減しない。推定の分解能（§2.4）、B・RR の違う腕の比べ方と scale が NaN になりがちなときの読み方（§7.4）、`05_ratio_adam` の用途を明確にし、費用が目標を超えたときに $k$ を下げるかは報告を読んだ開発者が決めることにした（§5 の 10）。裁定は §9.1 の A16〜A20、監査は §10.3。
> 初回の判断理由は [ADR 0054](../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md)。追加グリル後の現行の決定は §9.1、監査は §10.2・§10.3 に記録し、初回の決定表と監査は §9.2・§10.1 に履歴として保持する。実装時の文書同期は §11。
> 起点: DropMerge のフィードバック実験で、「長期 Run の後半にバッチ（B）や RR を上げると停滞を抜けられる」という経験則の機序を、成績ではなく測った量で確かめたくなった。2026-10-14 の分岐（§3 の最後）では、B を上げて 1 回の更新のノイズを下げた腕が効き、取りこぼしを減らす腕や更新を増やす腕は小さかった。この「ノイズ」を直接測る指標が無い。
> 関連: [done/062_plasticity_metrics_10prd.md](done/062_plasticity_metrics_10prd.md)（actual / probe のチャネルと購読の型を借りる）、[093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md)（同じ時期に起票した診断。群の番号 48 を使う予定）、[091_checkpoint_load_optimizer_options_10prd.md](091_checkpoint_load_optimizer_options_10prd.md)（継続の Run で学習率を変えられるようにする）、[999_metric_targeted_hyperparam_control_10prd.md](999_metric_targeted_hyperparam_control_10prd.md)（測った値でハイパラを動かす構想）、[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)、[ADR 0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md)、[実験記録 2026-07-27](../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、[サーベイ 2026-10-15](../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md)（観測と対処の一般事例。§7 の考察を本 PRD に反映した）。
> 行番号は 2026-10-15 時点の作業ツリー。実装時は近傍のシンボル名で再検索する。

## 1. 目的と問題

学習中の各更新が、どれだけノイズに埋もれているかを、`learn_step` 軸のスカラーとして記録する。これで、次の問いに成績ではなく機序の量で答えられるようにする。

- 今のバッチ（DropMerge では 256）は、学習のどの時期からノイズに負けているか。
- 後半に B を上げると伸びるのは、そのノイズが原因か。
- 序盤に B を上げても得が無いのは、序盤はノイズが小さいからか。

anet には勾配ノルムはある（`Optimize` の `grad_norm_tensor`）が、勾配の「ばらつき」と「平均の向き」を分けて測る指標は無い。

この指標は機序の観測量であって、B の効き目の因果判定ではない。因果は、[091](091_checkpoint_load_optimizer_options_10prd.md) のあとに学習率だけを下げた腕を含む分岐 Run で検証し、本 PRD の値はその読み解きを支える（§7）。

## 2. 概念

### 2.1 1 回の更新の勾配は、真の勾配とノイズの和

学習の 1 回の更新は、replay から B 個の経験を取り、その平均の損失の勾配で重みを動かす。B 個は全データの一部なので、この勾配は「全データで平均したときの勾配」（真の勾配 $G$）の推定値で、取った B 個によってぶれる。

$$
g_B = G + \text{ノイズ},\qquad \mathbb{E}\,\lVert g_B\rVert^2 = \lVert G\rVert^2 + \frac{\operatorname{tr}\Sigma}{B}
$$

| 記号 | 意味 |
|---|---|
| $g_B$ | B 個の経験から計算した 1 回の更新の勾配 |
| $G$ | 全データで平均したときの勾配（向かうべき方向）。ここでは「今の抽選・重み付け・target・乱数で定義した損失の期待勾配」であり、環境の成績が最適になる方向という意味ではない |
| $\Sigma$ | 1 個の経験ごとの勾配のばらつき（共分散）。$\operatorname{tr}\Sigma$ はその大きさの合計 |
| $B$ | バッチの大きさ |

経験が同じ分布から独立に抽出されるなら、ノイズの大きさは $1/B$ で小さくなる。バッチを倍にすると、ノイズの 2 乗は半分になる。

### 2.2 noise scale は「信号とノイズが同じ大きさになるバッチの大きさ」

$$
B_{\text{noise}} = \frac{\operatorname{tr}\Sigma}{\lVert G\rVert^2}
$$

$B = B_{\text{noise}}$ のとき、信号（$\lVert G\rVert^2$）とノイズ（$\operatorname{tr}\Sigma / B$）が同じ大きさになる。読み方は次のとおり。

| B と noise scale の関係 | 意味 | B を上げると |
|---|---|---|
| $B \ll B_{\text{noise}}$ | 1 回の更新はほぼノイズ | よく効く。B を倍にすると、同じ進み方に要る更新回数がほぼ半分になる |
| $B \approx B_{\text{noise}}$ | 信号とノイズが半々 | 効くが、効き目は減っていく |
| $B \gg B_{\text{noise}}$ | バッチで向きは十分取れている | ほぼ効かない。計算とデータの使い方が無駄になる |

この表は、局所の SGD モデルと独立抽出の仮定の下での理論的な予測である（McCandlish ら 2018）。Adam では単純な noise scale が実際の有効バッチを数桁過小推定した例（Merrill ら 2025、OLMo 1B / 7B）があり、曲率・optimizer・抽出方法・比較予算にも依存する。この表を決定則にせず、§7 の読み方に従う。

### 2.3 学習が進むと大きくなる

学習が進むと、残っている改善は細かくなり、向かうべき方向 $\lVert G\rVert$ は小さくなる。一方、経験ごとのばらつき $\operatorname{tr}\Sigma$ はそれほど小さくならない。そのため $B_{\text{noise}}$ は学習とともに大きくなりやすい。序盤は小さなバッチで足り、後半は大きなバッチが要る。「B を後から上げる」の教科書的な根拠はこれである。

学習率を下げることも、ノイズの効きを下げるが、同じ操作ではない。固定した状態での 1 回の更新の分散は $\eta^2 \Sigma / B$ で、$\eta$ を半分にすると信号の歩幅も半分になる（noise scale 自体は変わらない）。B を倍にすると、期待勾配を保ったまま分散だけが半分になる。長時間の SGD を連続時間で近似したときのノイズ強度に $\eta / B$ が現れる（Smith ら 2017）のとは、時間の単位が違う。B と学習率のどちらで下げるかの比較は、[091](091_checkpoint_load_optimizer_options_10prd.md) で継続の Run の学習率が効くようになってからできる（§7 の分岐）。

### 2.4 測り方（2 つの大きさのバッチから推定する）

$G$ も $\Sigma$ も直接は分からない。同じデータから、小さいバッチ（$B_s$）と大きいバッチ（$B_b$）の勾配のノルムを測ると、次の式で推定できる（McCandlish ら 2018 の付録）。

$$
\widehat{\lVert G\rVert^2} = \frac{B_b\,\lVert g_{B_b}\rVert^2 - B_s\,\lVert g_{B_s}\rVert^2}{B_b - B_s},\qquad
\widehat{\operatorname{tr}\Sigma} = \frac{\lVert g_{B_s}\rVert^2 - \lVert g_{B_b}\rVert^2}{1/B_s - 1/B_b}
$$

本 PRD は、同じ batch を $k$ 個の小バッチに分け、$B_s = B/k$、$B_b = B$ とする。小バッチ $j$ の勾配を $g_j$ とすると $s = \frac{1}{k}\sum_j \lVert g_j\rVert^2$、$b = \lVert \frac{1}{k}\sum_j g_j\rVert^2$ で、上の式は

$$
\widehat{\operatorname{tr}\Sigma} = \frac{(s - b)\,B}{k - 1},\qquad
\widehat{\lVert G\rVert^2} = \frac{k\,b - s}{k - 1}
$$

になる。1 回ごとの推定はぶれが大きく、$\widehat{\lVert G\rVert^2}$ は大きい数の差なので負になることがある（B がノイズ優勢なほど起きやすい）。分子と分母をそれぞれ指数移動平均してから割る（同論文の推奨）。それでも分母が正でない間は比を定義しない（§5 の NaN 規則）。小バッチ数 $k$ を増やすと分散側の精度が上がる（Gray ら 2024 は小バッチを小さくするほど推定の標準誤差が下がると報告）が、backward が $k$ 回要る。

推定の精度の目安（小バッチ勾配をガウスとみなした近似）: 1 回の測定の信号推定 $\widehat{\lVert G\rVert^2}$ のばらつき（標準偏差）は、$\lVert G\rVert^2$ に対する比でおよそ

$$
\sqrt{\frac{2k}{k-1}}\cdot\frac{B_{\text{noise}}}{B}\cdot\frac{1}{\sqrt{r}},\qquad r = \frac{(\operatorname{tr}\Sigma)^2}{\operatorname{tr}(\Sigma^2)}
$$

である。$r$ は、経験ごとの勾配のばらつきが実質いくつの方向に広がっているかを表す実効ランクである。信号推定には、このほか平均の向きとノイズの内積の項のばらつきも加わる。知りたい領域（$B_{\text{noise}} > B$）ほど信号の推定が粗くなり、平滑化後の信号が正でない区間（scale が NaN）が出やすい。EMA（係数 0.05）は、ばらつきを減らす効果でいえばおよそ 39 回分を平均するので、このばらつきは約 1/6 になる（§5 の 9 の平滑化幅は時定数 $1/\alpha = 20$ 回で、別の量である）。係数 $\sqrt{2k/(k-1)}$ は $k=8$ で 1.51、$k=2$ で 2.0、$k\to\infty$ で 1.41 なので、$k$ は信号側の精度にほとんど効かない。$k$ が効くのは分散 $\widehat{\operatorname{tr}\Sigma}$ 側の精度で、その相対的なばらつきはおよそ $\sqrt{2/((k-1)\,r)}$ である。

### 2.5 強化学習・NoisyNet・Adam・BF16 での注意

- **目標が動く:** TD の目標は target network から作るので、損失そのものが学習とともに変わる。ここで測るのは「その時点の更新の損失」のノイズであり、教師あり学習の値とそのまま比べない。連続する learn step をまたいで平均すると θ の動きが混ざるので、1 回の測定は 1 更新の中で閉じ、平均は測定間の EMA で取る。
- **何がノイズに入るか:** 経験の取り方（PER）、IS 重み、IQN の τ の抽選（経験ごとに抽選）、Munchausen の項は、どれも 1 回の更新の勾配のばらつきに入る。これらを含めた「実際に当てている更新」のノイズを測るのが目的である。
- **NoisyNet の ε は固定される:** Learner の forward は ε を batch 全体で共有する（`nn_forward.[act_sample].noisy.share = batch`）。小バッチもその 1 抽選を共有するので、本 PRD が測るのは ε を固定した**条件付きのサンプルノイズ**である。ε の抽選によるばらつきは信号側（$\lVert G\rVert^2$）に紛れ込み、EMA を長くしても分離できない（$\operatorname{Cov}(g_B) = \mathbb{E}_Z[\Sigma_Z]/B + \operatorname{Cov}_Z(G_Z)$。Wen ら 2018 の共有摂動の床）。B で減るのは前者だけなので、条件付きの値は B で減る成分を調べる手掛かりになるが、総ノイズの絶対値や B を増やしたときの改善を保証する値ではない。
- **Adam:** anet の optimizer は AdamW で、座標ごとに勾配を $1/(\sqrt{\hat v}+\epsilon)$ でそろえてから動かす。前処理つき版は、この固定 $P$ に対する noise scale（$P G$ と $P\Sigma P$）であり、実際の歩幅（$m$ の履歴、$v$ の更新、clip、weight decay を含む）の SNR ではない。主読みにするのは、optimizer の幾何に合う尺度だからで、素の版は文献比較と $P$ 非依存の参照として残す。
- **BF16:** 本線は Learner の forward が BF16 autocast で、backward は autocast の外、パラメータは FP32 なので勾配は FP32 で得られる。BF16 の丸めは実際の更新のノイズの一部なので、測定に含める（FP32 の Run とは値が違いうる）。
- **grad clip:** 測るのは clip 前の勾配である。clip が頻発する区間（`37_agent_qtd/24_grad_clip_ratio`）では、実際の歩幅はさらに違う。
- **BatchNorm:** BatchNorm を使う構成では、全 batch の forward で共有したバッチ統計を通じて各サンプルの勾配が他のサンプルに依存する。損失を小バッチに分けても、この依存は消えない。このようにサンプル間を結合する構成でも計算は許すが、独立サンプルのノイズとしての解釈や $1/B$ の関係は保証しない。専用の検出・禁止機構は追加せず、適用と解釈は利用側が判断する。本線（IMPALA2-ViT128、LayerNorm）は該当しない。[PyTorch 2.11 の BatchNorm2d 仕様](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm2d.html)（使用中の LibTorch 2.11.0 と同じ版）

## 3. 現状

- 更新の本体は `IQNLearner::UpdateFromSamples`・`QRLearner::UpdateFromSamples`・`TDLearner::UpdateFromSamples`（[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp)）。forward・`MakeTarget`・要素損失・重み付き平均までが `anet::Autocast` ブロックの中にあり、3 つとも `loss = (element_loss * weights).mean()`（TD `:3796`、QR `:3930`、IQN `:4108`）を作ってから `Optimize(loss)` へ渡す。Munchausen の項は target 側（NoGrad）だけに足される。
- `Learner::Optimize`（`:2621-2746`）はスカラーの `loss` しか受け取らない。経路は 3 つ: BF16（`use_amp && use_amp_bf16`）は GradScaler 無しで `backward → ForeachGradNorm → clip → step`、FP16 は `GradScaler` の `scale → backward → unscale → norm → clip → Step`、FP32 は BF16 と同じ順。`grad_norm_tensor` は clip 前で、`$update_result` の `grad_norm` として公開済み（`37_agent_qtd/21`）。
- 「この step を測る」の判定は `UpdateFromSamples` の中ではなく、共通の `UpdateFromBatch`（`:3130-3177`）が加算前の `vars_.learn_step` で行う。要求を Optimize の中で読む先例は policy churn（`CapturePolicyChurnOnlineBefore/After`、`:2442-2454`）。
- 学習の外で測る仕組みは、可塑性・NoisyNet・policy churn・replay_fit の probe にある。購読された指標の interval でだけ測り（`ConfigureScalarMetricSubscriptions`（`:2088-2188`）が購読から要求を組み立て、key ごとの `IntervalGate` で発火を決める）、専用の乱数で replay から一様に取る。
- 指標の値の載せ方には 2 つの経路がある。学習の更新そのものから取る値（actual チャネル）は `$update_result`、probe の値は `$agent` である（[062](done/062_plasticity_metrics_10prd.md)）。既知 key で値が無いときは NaN、未知 key だけ nullopt。
- metrics 定義側の `$ema` は 1 つの key の値の列にかける EMA で、分子と分母を別々に平滑化してから割る仕組みは無い。比をライブで読むには Learner が EMA を持つしかない。
- feature / readout の分け方は `Network::ComputeParameterNormSplit`（[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) `:1987-2088`）の中にあり、`feature_key` の依存閉包（bind を逆向きに辿った branch の集合）に入る branch のパラメータが feature、閉包外の branch と head が readout。Noisy の σ 統計も同じ閉包で分ける。分けたパラメータ集合を外へ返す API は無い。
- optimizer は `FusedAdamW`（`torch::optim::AdamW` 派生、[nn_util.cpp](../../core/anet-core/src/nn_util.cpp) `:91-202`）。state は基底の `state()` を `param.unsafeGetTensorImpl()` で引き、`AdamWParamState{step, exp_avg, exp_avg_sq}`。偏り補正は ATen カーネルの中で行われ、anet 側には無い。勾配が未定義のパラメータには state が作られない。
- PER の抽選は i.i.d.（[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) `:1002-1065`、1 件ずつ rejection sampling）。層化ではないので、batch を並び順に分けても小バッチ間は交換可能である。IS 重みは batch 内の最大値で正規化される（測定 1 回の中では共通係数）。
- `torch::autograd::grad` と `retain_graph` の使用は本体・テストとも 0 件。
- 指標の定義は [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) にあり、群の番号は 46（replay_fit）・47（noisy）まで使っている。44・45・48・49 は空きで、48 は attention エントロピーの暫定 PRD が使う予定である。
- **DropMerge 本線の実効設定**（DropMerge-02 の FB 系列 10/02〜10/15 の config dump。NoisyNet 基準を含む全 Run で同じ）: `learner.use_amp = true`、`use_amp_bf16 = true`、`adam_eps = 1e-4`、B256 / RR1、`use_grad_clip = true`、`grad_clip_tau = 5`、`head_projection.noisy.enabled = true`（10/14 以降）。チェーンは `DefaultDQNAgent.$ = @baseline > @iqn > @fast > A1 > @bf16 > A2 > A3`（[DropMerge.txt](../../apps/runner/config/DropMerge.txt) `:21`）で、`@bf16` が [agent.txt](../../apps/runner/config/agent.txt) `:96-98` にある。backend は `@non-deterministic`（`DropMerge.txt:101`）。
- 勾配のばらつきや noise scale を測る指標は無い。
- 2026-10-14 の DropMerge の分岐（NoisyNet の基準の 100M の checkpoint から各 20M、学習中の平均点は 1 万局単位で読み、同じ条件の 2 本の差は 0.15%）:

  | 腕 | 一度も使われずに捨てられた割合 | 学習中の平均点（110–120M） |
  |---|---|---|
  | B256/RR1（対照） | 29% | 1,806 |
  | 新しい経験の初期優先度を max | 23% | +0.8% |
  | 初期優先度 0.03 | 35% | −0.6% |
  | B512/RR1（1 回のノイズ↓、更新は半分） | 29% | +2.1% |
  | B512/RR2（1 回のノイズ↓、更新は同じ） | 5.6% | +0.8% |

  取りこぼしを変えずにノイズだけを下げた B512/RR1 が一番効いた。B512/RR1 は経験あたりの更新の総量も半分なので、「1 回のノイズ」と「更新の総量（学習率に近い）」はまだ分けられていない。

## 4. ゴールと非ゴール

ゴール:

1. 実際の更新（PER・IS 重み・τ・Munchausen 込み。NoisyNet の ε は更新と同じ 1 抽選に固定）の noise scale を、Adam 前処理版を主読みとして、ネット全体と feature / readout ごとに `learn_step` 軸のスカラーとして記録する。
2. 学習の経路（重み、乱数、replay の抽選、`.grad`、optimizer の state）を変えない。決定的な同一環境では、測定の ON/OFF で学習状態と既存の数値指標が一致することを検証する。非決定論設定の本線 Run 間に bit 一致までは要求しない（§7）。
3. 購読しない構成では計算が起きない。
4. **本線の DropMerge Run（BF16 AMP ON・B256/RR1・NoisyNet）で読める。**
5. **全体と feature / readout を含む 15 指標を一式で完成・検証する。** 完成範囲は計測機能の正しさ、本線での動作、実時間・追加メモリの実測報告までとし、因果を評価する分岐キャンペーンは別の実験とする。

非ゴール（§8 に、作る条件を書いた）:

- 測った値で B や学習率を自動で変えること（[999_metric_targeted_hyperparam_control_10prd.md](999_metric_targeted_hyperparam_control_10prd.md) の範囲）。
- 曲率まで入れた臨界バッチ（ヘッセ行列を使う版の noise scale）。
- 層ごとの細かい内訳、target network 側の測定。
- ε を引き直して NoisyNet のばらつきまで含めた総ノイズの測定。
- FP16（GradScaler）を有効にした Run での測定。BF16 は対象である。
- DefaultDQN 以外の Agent への配線、および将来の複数損失・別 optimizer 用の拡張口。
- B・RR・学習率を変える分岐キャンペーンの実施、スコア改善やノイズ仮説の成立を本 PRD の完了条件とすること。

## 5. 決定した設計

1. **測る batch と場所:** 購読された指標が発火する `learn_step`（`UpdateFromBatch` が加算前の `vars_.learn_step` で要求を立てる）で、その更新自身の batch を使う。PER の抽選・IS 重み・τ・NoisyNet の ε は実際の更新と同じになる。replay から別に取る probe にしないのは、測りたいのが「実際に当てている更新」のノイズだからである。
   `Optimize` を `Optimize(loss, weighted_element_loss)` の 2 引数にする。3 つの Learner は今と同じく `weighted = element_loss * weights`（長さ B）と `loss = weighted.mean()` を autocast ブロックの中で作り、`weighted` を添えて渡すだけで、`loss` の計算は動かさない（学習経路の数値は変更前と同一）。測定は Optimize の先頭（`zero_grad` と `backward` の前）で行う。
   1 引数の `Optimize(loss)` を呼んでいる既存のテスト（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) の 2 箇所。policy churn の決定的な 1 update を作るテスト用 Learner と、grad clip のテスト）も、2 引数の形へ移す。第 2 引数には、平均が `loss` に一致する長さ B の要素損失を渡す。
2. **分け方と勾配:** `weighted` を並びの順に $k$ 個の小バッチ（既定 $k=8$、B256 なら 32 個ずつ）に分け、小バッチ $j$ の損失 $L_j$ = その区間の平均を作る（$k$ 本の平均は `loss` に一致する）。$g_j$ は `torch::autograd::grad(L_j, params, retain_graph=true)` で個別に計算する。`.grad` には書かず、optimizer の step もせず、乱数も使わない。$\lVert g_j\rVert^2$ を足し込み、$g_j$ 自体はパラメータと同サイズの作業バッファ 1 本に足し込む。そのあと通常の `zero_grad → backward → clip → step` がそのまま走る。PER が i.i.d. なので、シャッフルは要らない。
   対象は online の学習パラメータとし、未使用判定は今回取得した小バッチ勾配で行う。全小バッチで勾配が未定義のパラメータは除外する。一部の小バッチだけ未定義なら、その小バッチでの寄与を 0 として扱う。測定時点で残っている前回更新の `.grad` は判定に使わない。
3. **推定と EMA:** 1 回の測定で $s$ と $b$ を出し、§2.4 の式で $\widehat{\operatorname{tr}\Sigma}$ と $\widehat{\lVert G\rVert^2}$ を作る。成立した有限の信号・分散を `EmaFilter`（[045](done/045_ema_debias_warmup_10prd.md) の型、初期の偏りを補正）で別々に更新し、noise scale は移動平均の比 $\overline{\operatorname{tr}\Sigma}/\overline{\lVert G\rVert^2}$ とする。有限の負の信号推定も観測結果として取り込み、0 へ丸めたり除外したりしない。移動平均した信号が正でない間、scale と ratio は NaN とする。
   EMA は版・群ごとに Learner の State として持ち、購読全体の測定点を共有する（8 の購読規則）。同じ更新で複数の指標が発火しても、各 EMA 対の更新は 1 回だけとする。EMA 入力となる raw 信号・分散の推定が未成立の版・群（Adam state 不足・非有限値など）は、その回の EMA 対を更新せず、公開値を前回値で埋めない。scale が NaN になること自体は EMA 更新を止める理由ではなく、有限の負の信号は引き続き取り込む。購読間隔ごとの別 EMA は持たない。
4. **Adam の前処理つき（主読み）:** 同じ $g_j$ を要素ごとに $1/(\sqrt{\hat v}+\epsilon)$ 倍してから同じ量を出す。$\hat v$ は更新前の AdamW の 2 次モーメント（`exp_avg_sq` を `step` で偏り補正）、$\epsilon$ は `learner.adam_eps`。今回の損失から勾配が得られたパラメータだけを、state の成立判定の対象にする。使用パラメータの state が未作成、または偏り補正がまだ成立しない場合（最初の optimizer step の前など）、そのパラメータを含む群と all の Adam 版を NaN にする。例えば readout に不足があれば readout と all を NaN とし、成立している feature と素の勾配版は記録する。対象外の未使用パラメータに state が無いことは測定を妨げない。state を測定のために作成・更新しない。
5. **群:** ネット全体（all）、feature、readout の 3 つ。feature / readout の分け方は重みノルム・Noisy 統計と同じ（`learner.plasticity.feature_key` の依存閉包）。`Network` に閉包分類を返す口を足し、既存の `ComputeParameterNormSplit` と分類規則を共有する。重みノルムの対象フィルタまで流用せず、NoisyNet の σ も今回の勾配が得られれば所属 branch の群に入れる。未使用パラメータの除外は 2 に従う。1 回の測定で $\lVert g\rVert^2_{\text{all}} = \lVert g\rVert^2_{\text{feature}} + \lVert g\rVert^2_{\text{readout}}$ が成り立つ。群ごとに未成立回が異なる場合、EMA の履歴まで同一とは限らない。
   空群（閉包分類でその群に学習パラメータが 1 つも無い構成）は推定が定義されないので、その群の raw 信号・分散と scale は NaN とし、EMA も更新しない。all への寄与は 0 として扱い（上の恒等式は空群の項を 0 と読む）、Adam state 不足とは区別する。空群は構成で決まる静的な状態なので、Run を通してその群の key は NaN のままになる（§7.3 では計測予定点に数えない）。feature / readout の key（追加する raw を含む）を購読したときだけ `feature_key` を必須・検証する（可塑性と同じ規則）。
6. **部品の置き場:** 計算部（小バッチ勾配のノルム統計、信号・分散の EMA 対と比を返す推定器、Adam の前処理の読み出し）は [nn_util](../../core/anet-core/include/anet/nn_util.hpp)（`ForeachGradNorm` / `FusedAdamW` の隣）に置く。配線（購読から要求を組み立てる、Optimize 内の呼び出し、`BatchUpdateResult` の key、EMA 状態の所有）は `dqn::Learner` に置く。[062](done/062_plasticity_metrics_10prd.md) の「計算部は共有ヘルパ、配線は Agent ごと」と同じ分け方とする。
   API とテストの必須要件は今回の単一損失・AdamW の用途で完結させる。全 Agent 共通の診断フレームワークや、PPO / SAC の複数損失・別 optimizer を見越した拡張口は作らない。将来の利用側ができた時点で必要な設計を行う（§8）。
7. **公開する値（actual チャネル、`$update_result`、群 `49_agent_gradnoise`）:**

   | tag | key | 意味 |
   |---|---|---|
   | `01_scale_adam` | `grad_noise_scale_adam` | 主読み。Adam 前処理つきの noise scale（EMA の比、all） |
   | `02_scale_adam_feature` | `grad_noise_scale_adam_feature` | 同、feature 群 |
   | `03_scale_adam_readout` | `grad_noise_scale_adam_readout` | 同、readout 群 |
   | `05_ratio_adam` | `grad_noise_ratio_adam` | `scale_adam ÷ B`。各 Run の中で閾値 1 と比べてライブで読む用で、1 を超えると測った座標系ではノイズが信号を上回る。B 変更による成績改善は保証しない。B・RR の違う Run どうしの比較は §7.4 の方法（経験数でそろえた窓の raw の平均の比）で行う |
   | `11_scale` | `grad_noise_scale` | 素の勾配の noise scale（all。文献比較と、$\hat v$ が動く Adam 版に対する $P$ 非依存の参照） |
   | `21_signal_adam` | `grad_noise_signal_adam` | $\overline{\lVert P G\rVert^2}$（all、EMA）。scale が NaN の理由を読む用 |
   | `22_variance_adam` | `grad_noise_variance_adam` | $\overline{\operatorname{tr} P\Sigma P}$（all、EMA） |
   | `31_signal_adam_raw` | `grad_noise_signal_adam_raw` | その測定 1 回の $\widehat{\lVert P G\rVert^2}$（all）。後から自分で平均し直す用 |
   | `32_variance_adam_raw` | `grad_noise_variance_adam_raw` | 同、$\widehat{\operatorname{tr} P\Sigma P}$ |
   | `33_signal_raw` | `grad_noise_signal_raw` | 素の版の測定 1 回の $\widehat{\lVert G\rVert^2}$（all） |
   | `34_variance_raw` | `grad_noise_variance_raw` | 同、$\widehat{\operatorname{tr}\Sigma}$ |
   | `35_signal_adam_feature_raw` | `grad_noise_signal_adam_feature_raw` | Adam 版の測定 1 回の信号推定（feature） |
   | `36_variance_adam_feature_raw` | `grad_noise_variance_adam_feature_raw` | Adam 版の測定 1 回の分散推定（feature） |
   | `37_signal_adam_readout_raw` | `grad_noise_signal_adam_readout_raw` | Adam 版の測定 1 回の信号推定（readout） |
   | `38_variance_adam_readout_raw` | `grad_noise_variance_adam_readout_raw` | Adam 版の測定 1 回の分散推定（readout） |

   公開は計 15 指標。比の raw は平均し直せないので、raw は信号と分散で出す。内訳も scale だけでなく raw を残し、未成立の調査や平滑化のやり直しに使う。各 key は自身の購読が発火した更新だけで値を公開し、未購読・未発火・未成立は NaN、未知 key だけ nullopt とする。測定自体が無い更新では全 key が NaN となる。

8. **購読:** 新しいプロファイル `metrics.scalar.@gradnoise` を置く。baseline には入れない。既定の interval は 503（素数。hard update C=200 / 120 の腕で位相が固定されるのを避け、churn・noisy・replay_fit と同じ learn_step を測るので同じ更新を横に読める）。

   ```properties
   metrics.scalar.@gradnoise : [49_agent_gradnoise/01_scale_adam] ?= grad_noise_scale_adam @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/02_scale_adam_feature] ?= grad_noise_scale_adam_feature @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/03_scale_adam_readout] ?= grad_noise_scale_adam_readout @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/05_ratio_adam] ?= grad_noise_ratio_adam @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/11_scale] ?= grad_noise_scale @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/21_signal_adam] ?= grad_noise_signal_adam @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/22_variance_adam] ?= grad_noise_variance_adam @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/31_signal_adam_raw] ?= grad_noise_signal_adam_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/32_variance_adam_raw] ?= grad_noise_variance_adam_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/33_signal_raw] ?= grad_noise_signal_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/34_variance_raw] ?= grad_noise_variance_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/35_signal_adam_feature_raw] ?= grad_noise_signal_adam_feature_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/36_variance_adam_feature_raw] ?= grad_noise_variance_adam_feature_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/37_signal_adam_readout_raw] ?= grad_noise_signal_adam_readout_raw @learn $learn_step $update_result interval:503
   metrics.scalar.@gradnoise : [49_agent_gradnoise/38_variance_adam_readout_raw] ?= grad_noise_variance_adam_readout_raw @learn $learn_step $update_result interval:503
   ```

   各購読行の `IntervalGate`（初回発火と bucket-crossing）で要求を判定し、その和集合で測定する。同じ更新で複数行が発火しても、小バッチ勾配の取得は 1 組だけとする。購読されている版・群はその測定点を共有し、成立した EMA 対を各 1 回更新する。その回に記録する key は各購読行の発火条件で決める。
   例えば scale を 503 間隔、raw を 100 間隔で購読すると、両方の発火点を EMA に取り込むため、scale の追従速度も変わる。記録間隔と EMA の更新間隔は同じとは限らない。保存済み raw の再平滑化はできるが、内部 EMA を完全に再現するには、当該 EMA へ取り込んだ全測定点の raw 値が必要である。既定の全 15 行を同じ 503 間隔で購読すれば、この測定点はそろう。
   `grad_noise_*` の購読行の interval が揃っていないときは、購読ヒントの解釈時に interval の一覧を添えて 1 回だけ WARN する。契約違反ではなく、EMA の時間幅が意図と違う可能性を知らせる診断なので fail-fast にはしない（policy churn の位相 WARN と同じ型）。

   Run への追加は、既存の `@noisy` と同じく `metrics.scalar.$` のチェーンへ足す（DropMerge なら `run.@gradnoise : metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@iqn_search_p0 > metrics.scalar.@gradnoise > M1 > M2`）。

9. **設定:**

   ```properties
   DefaultDQNAgent.@baseline : learner.grad_noise.num_micro_batches ?= 8     # 小バッチ数 k。replay_batch_size を割り切ること（購読時だけ検証）
   DefaultDQNAgent.@baseline : learner.grad_noise.ema_alpha ?= 0.05          # 測定 1 回あたり。全購読が interval:503 の場合、約 1 万 learn step（B256/RR1 で約 2.6M 経験）の平滑化幅
   ```

   どちらも推定の期待値は変えない（k は主に分散側の精度と費用で、信号側の精度にはほとんど効かない（§2.4）。α は平滑化）ので、可塑性の δ / τ のように定数化はしない。検証（Fail-Fast 原則。`grad_noise_*` の購読があるときだけ）: `num_micro_batches < 2`、`replay_batch_size % num_micro_batches != 0`、`ema_alpha` が (0, 1] の外、FP16（`use_amp && !use_amp_bf16`）のいずれかなら、購読ヒントの解釈時（学習開始前）に `ANET_SYSTEM_ERROR`。メッセージには key、指定値、期待する範囲（B の値を含む）を入れる。購読が無ければ NoCare（意図された休止）。feature / readout の key を購読していて `feature_key` が無い・未知の branch なら、可塑性と同じく `GetBranchNames()` の一覧付きで止める。

10. **費用:** 測定 1 回で backward が $k$ 回増える（共有の計算グラフ全体を通る）。全購読が interval 503・$k=8$ の場合、backward の回数は約 1.6% 増えるが、ノルム集計・転送・同期を含む実時間の増加率をこの比率から保証しない。勾配の加算バッファ以外に、取得した勾配や autograd の作業領域もあるため、追加メモリをバッファ 1 本分と断定しない。測定には `ANET_PROFILE_SCOPE` を入れ、処理内訳の確認に使う。追加負担は本線の OFF/ON 反復 A/B で実測し、実時間の増加率 2% 以下は目標とする（§7）。超過だけでは不合格にせず、実時間・ばらつき・追加メモリの報告を必須にする。実装する側は既定の $k=8$ のまま実測を報告し、$k$ や interval を変えない。目標を超えたときに費用を下げるかどうかは、報告を読んだ開発者が決める。下げる場合の第一候補は $k$ である（信号側の精度はほとんど落ちない。§2.4）。

11. **完成の単位と検証順序:** all の素・Adam 版と feature / readout、15 指標のプロファイル、設定キー、文書同期（§11）を一式で完成させる。全体だけを完成扱いする P1/P2 の段階分けは設けない。単体の推定・成立条件・購読・非干渉を確認し、15 指標をそろえた本線 smoke と性能比較までで受け入れる。

## 6. 機構ごとの根拠

| 機構 | 根拠（起きたこと・測った量） | 裁定 |
|---|---|---|
| noise scale の記録 | B を上げた腕が効き、取りこぼしの腕は小さかった（§3）。その効き目がノイズによるものかを確かめる量が無い | 作る |
| 更新自身の batch を分けて測る | 測りたいのは実際の更新のノイズ（PER・IS 重み・τ・Munchausen を含む）。forward を増やさず、乱数も `.grad` も触らないので学習の経路を変えない。PER が i.i.d. なので並び順の分割で足りる | 作る |
| `Optimize` の 2 引数化 | 測定を 1 箇所に置ける。`loss` の計算を動かさないので学習経路の数値は変更前と同一 | 作る |
| 移動平均の比と共有する測定点 | 1 回ごとの推定はぶれが大きく、信号の推定が負になりうる（§2.4）。`$ema` では比が作れない。購読の和集合で各 EMA を 1 回更新し、間隔ごとの状態を増やさない | 作る |
| Adam の前処理つきの版（主読み） | 実際の optimizer は AdamW。Adam では素の版が有効 B を数桁外した例がある（§2.2） | 作る |
| 素の版（all だけ） | 文献比較と、$\hat v$ が動く Adam 版に対する $P$ 非依存の参照。同じ $g_j$ のノルムを 1 組足すだけ | 作る（縮小） |
| feature / readout の内訳と各群の raw | all だけでは群ごとの差を分離できず、終了後の Run から勾配を再計算できない。追加グリルで、全体だけへ縮小せず両群を最初から観測する方針を選んだ。どちらが支配的かは測定で確認する | 作る（15 指標の一部） |
| BF16 の Run で測る | 本線が BF16（§3）。BF16 経路は GradScaler が無いので同じ経路で測れる | 作る |
| 臨界バッチ（曲率を含む版） | ヘッセ行列とベクトルの積が要り、費用が大きい。単純な版で傾向が見えるかが先 | 作らない（§8） |
| B や学習率の自動調整 | まず値の動きを読んでから | 作らない（§8） |
| ε を引き直す版の総ノイズ | forward が増え RNG を消費する。今回は B で減る条件付きの成分を観測し、B 変更の効果は別の分岐実験で確かめる | 作らない（§8、gate 付き） |

## 7. 受入基準

### 7.1 計測機能の検証

1. **推定式・群分割・Adam 前処理:** 経験ごとの勾配の平均と共分散が分かっている合成問題（線形回帰）で、§2.4 の推定（EMA 後）が解析値の $\operatorname{tr}\Sigma / \lVert G\rVert^2$ と許容誤差内で一致すること。小バッチ勾配の平均が batch 全体の損失の勾配と一致すること。各測定の feature と readout の二乗量および raw 信号・分散の和が all と一致すること。Adam 版は $P=I$ で素の版と一致し、非一様な $P$ と既知の state では偏り補正・epsilon を含む手計算と一致すること。
2. **成立条件:** 今回未使用のパラメータを、過去の `.grad` や Adam state の有無にかかわらず除外すること。一部の小バッチだけで未使用ならその寄与が 0 になること。使用パラメータの state が不足する群と all の Adam 版は NaN、成立する群と素の版は記録されること。空群はその群の raw 信号・分散と scale が NaN で、all と他の群の値が影響を受けないこと。未成立回で EMA が進まず、公開値が過去値で埋まらないこと。有限の負の信号は EMA に取り込み、平滑化後の信号が非正なら scale・ratio を NaN にすること。未成立回を挟んだ再成立時も確認する。
3. **購読・公開:** 異なる interval の購読を組み合わせ、初回発火と bucket-crossing を含む発火点の和集合で測ること。同時発火しても勾配取得と各 EMA 更新は 1 回だけであること。購読されている版・群が測定点を共有し、各 key は自身の発火時だけ公開され、未発火時は NaN であること。未購読時は計測しないこと。raw の追加購読で EMA の更新間隔が変わり、全測定点の raw を取り込んだ再計算が内部 EMA と一致することを確認する。interval が混在する購読では起動時に 1 回だけ WARN が出て、揃っていれば出ないことを確認する。
4. **学習を変えない:** 決定的な同一環境・同じ seed で、`@gradnoise` を購読した Learner と購読しない Learner を並走させ、毎 step のパラメータ・`.grad`・buffer・optimizer state（step、exp_avg、exp_avg_sq）・Learner の乱数状態が `torch::equal` で一致すること（既存の Noisy 診断・Munchausen 非干渉テストの型）。TD / QR / IQN × FP32 を対象にし、CUDA では BF16 も検証する。CUDA 未検証なら BF16 の受入は未確認として残す。環境・backend・決定論設定を記録し、非決定論設定の本線 Run 間には bit 一致を要求しない。[PyTorch 2.11 の再現性の条件](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
5. **設定契約:** `num_micro_batches < 2`、B を割り切れない、`ema_alpha` 範囲外、FP16 で購読、feature / readout 購読時の `feature_key` 欠如・未知 branch のそれぞれで、学習開始前に `ANET_SYSTEM_ERROR` で止まり、メッセージに key・指定値・期待する範囲が入ること。購読が無ければ、同じ設定でも起動すること（NoCare）。
6. **本線 smoke:** DropMerge 本線（BF16・NoisyNet）に `metrics.scalar.@gradnoise` を足し、15 指標すべての定義・購読・source key の配線を実効設定と出力で確認する。Adam state が形成された後の測定点で、all・feature・readout の Adam raw 6 指標と素の raw 2 指標が有限であることを確認する。EMA 値と scale・ratio は成立条件に従って公開されることを確認し、scale が常に有限・正になることは要求しない。非有限値を保存しない記録経路なので、全 tag が必ず保存されることを条件にせず、未成立の key は定義・購読と単体検証の結果を照合する。
7. **既存指標の非干渉:** 決定論設定（LunarLander CPU）で、`@gradnoise` の有無によって既存の数値指標の metrics checksum が変わらないこと（ADR 0031 の ON/OFF 等価性と同じ道具 `core/anet-core/testdata/prd084/compare_metrics.py`）。OFF 側は変更前のバイナリとも一致すること。

### 7.2 実時間と追加メモリ

同じ Release ビルドと本線構成を使い、計測 OFF → ON を交互に各 3 回、計 6 回測定する。ON は全 15 指標・interval 503・$k=8$・EMA 係数 0.05 とする。初期 checkpoint、warmup の長さ、warmup 後に比較する学習量（同じ learn step 数）、出力などの条件をそろえる。

各回の wall-clock 実時間、OFF/ON それぞれの中央値とばらつき（最小・最大を含む）、中央値の増加率 `median(ON) / median(OFF) - 1`、追加メモリを報告する。メモリは同じ測定方法で CPU/GPU のピーク使用量と OFF/ON 差分を確認し、取得した勾配や autograd 作業領域も含む実測として扱う。

測定処理に `ANET_PROFILE_SCOPE` を入れ、内訳の調査に使う。ただし CPU の計測区間や NVTX range の合計を、GPU の非同期実行・同期を含む end-to-end の追加時間の代わりにしない。**実時間の増加率 2% 以下は目標であり、超過だけでは不合格にしない。** 目標超過も含めた実測報告を完了条件とする。実装する側は費用を隠すために interval や $k$ を変えず、既定値のまま報告する。費用を下げるかどうかは、報告を読んだ開発者が決める（§5 の 10）。

### 7.3 完成範囲と値の未成立率

本 PRD は、15 指標を一式で実装し、§7.1・§7.2 と文書同期（§11）を満たした時点で完成とする。B / RR / 学習率を変える分岐実験や、スコア改善・ノイズ仮説の支持を完成の条件にしない。

従来の「NaN 率」は **値の未成立率** として扱う。計測を有効にした正常完了 Run について、実際に行われた学習更新と購読 interval から、各 key 自身の発火規則（初回発火・bucket-crossing）に従う計測予定点を求め、記録された有限値を照合する。

- 分母は対象 key の計測予定点数、分子はそのうち有限値を記録できなかった点数とする。通常の未計測 step や、その key が発火しない他の購読の測定点は分母に含めない。空群の群に属する key は構成から静的に NaN と決まるので、計測予定点を 0 とし、率は未定義とする。
- 対象区間・購読設定・予定点数・有限値数を併記する。予定点が 0 なら率は未定義とする。
- 非有限値を保存しない記録仕様を前提に、Run の正常完了とログの完全性を確認して集計する。異常終了や記録欠損を、そのまま値の未成立とみなさない。
- 未成立の理由は信号非正だけではない。Adam state 不足、非有限の推定値などもあり（空群は予定点から除く）、raw・平滑化値と測定条件を併せて読む。記録されていない値の理由を一意に復元できるとは限らない。理由別の追加状態指標は設けない。

### 7.4 読み方と後続の分岐実験

`ratio_adam > 1` は、測定した座標系でノイズ推定が信号推定を上回ることを示す。B 変更による成績改善の保証ではない。Adam の実更新、曲率、共有 ε、学習量の変化もあるため、因果は分岐 Run で確かめる。

- 後半に `scale_adam` が B を大きく超え、B512 の腕も改善すれば、B256 でのサンプルノイズが改善余地に関係するという読みを支持する。ただし単独の因果証明とはしない。
- `scale_adam` が B を大きく下回っても B512 の腕が改善する場合、条件付きの noise scale だけでは説明しきれない。更新の総量（学習率）、共有 ε の床、Adam の非線形性などを候補に、学習率だけを下げた腕とも比べる。
- 序盤に小さい値が得られても、それだけで B を上げる効果が無いとは決めない。序盤に B を上げたときの NEET は、別の問い（序盤の過剰適合）として残る。
- 内部 EMA の再現には、その EMA が取り込んだ全測定点の raw が必要である。保存した raw の間隔が疎い場合、その再平滑化は内部 EMA とは別の系列になる（§5 の 8）。
- scale が NaN になりがちな区間は、$B_{\text{noise}} \gg B$ か、勾配のばらつきの実効ランクが低いことを示しうる（§2.4）。raw の信号・分散を長い窓で平滑し直して読む。

[091](091_checkpoint_load_optimizer_options_10prd.md) 後の分岐キャンペーン（サーベイ §7.5 の A〜E: 基準 / 学習率だけ半分 / B512 RR1 / B512 RR2 / B512 RR2 で学習率半分）は別の実験記録で計画する。

B・RR の違う腕どうしは、経験数でそろえた終盤窓に入る raw の信号・分散（Adam 版は `31`/`32`、素の版は `33`/`34`）の平均の比で比べ、窓内の測定点数を併記する。記録の軸は `learn_step` なので（記録側と Learner 側が同じ軸で interval を数える）、B512/RR1 の腕は同じ経験数のあいだの測定点が半分になり、内部 EMA の時定数は経験数に換算すると 2 倍になる。そのため窓末の `01_scale_adam`・`05_ratio_adam` を腕どうしで直接比べず、各腕の中の時間変化を読むのに使う。

腕ごとの比、値の未成立率、成績との照合（支持 / 反証 / 判定不能）を記録し、次の B / RR の提案に使う。判定不能の原因が NaN の多さなら、§8 の先送り項目を検討する前に raw を長い窓で平滑し直す。この活用は計測機能の受入後に行う。本 PRD の効果（§1 の問いに測った量で答えられること）は、この記録が残ったことで実現したとみなし、別の数値閾値は設けない。

## 8. 先送りした段階と、作る条件

1. **ε を引き直す版の総ノイズ測定:** 小バッチごとに NoisyNet の ε を引き直して、ε のばらつきまで含めた noise scale を測る（forward が増え、専用の乱数が要る）。作る条件: 分岐 Run の結果が条件付きの値の読みと食い違い、かつ `47_agent_noisy/09_q_abs_diff` が大きい（ε の効きが無視できない）とき。
2. **臨界バッチ（曲率を含む版）:** ヘッセ行列とベクトルの積で $\operatorname{tr}(H\Sigma)/(G^\top H G)$ を測る。作る条件: 単純な版と実際の B の効き目が食い違ったとき。
3. **自動調整:** noise scale に合わせて B や学習率を動かす。作る条件: 本 PRD の値で「後半に noise scale が B を超える」ことが Run をまたいで再現し、手動で B を上げた分岐と効き目が合ったとき。[999_metric_targeted_hyperparam_control_10prd.md](999_metric_targeted_hyperparam_control_10prd.md) の入力の 1 つにする。
4. **層ごとの内訳:** feature / readout で差が見え、どの block かを特定する必要が出たとき。
5. **PPO / SAC への対応:** その Learner と観測したい損失・optimizer が具体化したときに、計算部の再利用範囲と必要な API を検討する。今回の共通部は単一損失・AdamW の用途で完結させ、複数損失の key、別 optimizer の拡張口、ロールアウト相関への対応を先に設計・実装しない。

## 9. 決定事項（2026-10-15）

### 9.1 追加グリルで確定した現行の裁定

初回グリルの記録は §9.2 に保持する。追加グリル（A1〜A15）と 3 回目のグリル（2 回目の追加グリル、A16〜A20）で変更・具体化した点はこの表と §5・§7 を現行の裁定とし、初回の 11 指標・段階分け・state 不足時の一括 NaN を現行要件として使わない。

| # | 論点 | 裁定 |
|---|---|---|
| A1 | ゴール・完成範囲 | BF16 DropMerge 本線で学習経路を変えずに条件付きの勾配ノイズスケールを観測する。完成は計測機能の検証まで。B / RR / 学習率の因果を評価する分岐実験は別に行う |
| A2 | 公開指標 | 既存 11 指標に Adam feature / readout の raw 信号・分散（tag 35〜38）を加えた 15 指標。素の 3 指標と全体の平滑化済み信号・分散 2 指標も維持する |
| A3 | 既定値・有効化 | $k=8$、interval 503、EMA 係数 0.05 を維持。`metrics.scalar.@gradnoise` は baseline 外、未購読時は計測ゼロ |
| A4 | 測定点と EMA | 購読の発火点の和集合で測り、同一更新で勾配取得・各 EMA 更新は 1 回。購読されている版・群は測定点を共有し、間隔別 EMA は持たない。異なる interval の購読追加は平滑化の時間幅も変える。interval が揃っていなければ起動時に 1 回 WARN（診断） |
| A5 | 公開と再現 | 各 key は自身の発火時だけ公開し、未発火・未成立は NaN。内部 EMA の完全な再現には、その EMA が取り込んだ全測定点の raw が必要 |
| A6 | 使用パラメータ | 今回の小バッチ勾配から判定し、前回の `.grad` を見ない。全小バッチで未使用なら除外、一部だけ未使用ならその寄与は 0。NoisyNet の σ も使用されれば所属群に含める |
| A7 | Adam state | 使用パラメータの state を群ごとに判定。不足する群と all は NaN、成立する群と素の版は記録する。計測のために state を作成・変更しない |
| A8 | 未成立・負の信号・空群 | 未成立の群の EMA は更新せず前回値で埋めない。有限の負の信号は EMA に取り込み、平滑化後の信号が非正なら scale・ratio は NaN。空群はその群の raw・scale とも NaN（推定が定義されない）で EMA も更新せず、all への寄与は 0 |
| A9 | 解釈の限界 | `ratio > 1` は B 変更による改善の保証ではない。BatchNorm 等のサンプル間結合では独立サンプルのノイズとしての解釈を保証せず、専用の検出・禁止機構は追加しない |
| A10 | 共通計算部 | `Optimize(loss, weighted_element_loss)` と nn_util 配置を維持。必須 API・テストは今回の単一損失・AdamW で完結し、将来 Agent 対応は先送り |
| A11 | 検証 | 合成問題で式・群・Adam 前処理・成立条件・異なる interval と同時発火を検証。決定的な同一環境で TD / QR / IQN の ON/OFF 非干渉を確認し、CUDA BF16 も対象とする。本線の非決定論 Run 間に bit 一致は要求しない |
| A12 | smoke | 15 指標の購読・配線と Adam state 形成後の raw を確認。scale が常に有限・正になることや、全 tag が有限値として保存されることは要求しない |
| A13 | 値の未成立率 | 正常完了 Run の各 key の計測予定点と記録済み有限値を照合する。通常の未計測 step は分母から除き、NaN を信号非正だけに帰属させない。追加状態指標は設けない |
| A14 | 性能 | 同じ Release ビルド・本線構成で OFF/ON を交互に各 3 回、warmup 後の同じ学習量で比較。実時間の中央値・ばらつき・追加メモリを報告。2% 以下は目標で、超過だけでは不合格にしない |
| A15 | 実装の完成単位 | 全体と feature / readout、15 指標を一式で完成させる。推定・成立条件・購読・非干渉の検証から本線 smoke・性能比較・文書同期へ進む。all のみを先行完成扱いしない |
| A16 | 推定の分解能 | 機構は足さず文書化する（§2.4）。1 回の信号推定の相対的なばらつきは $B_{\text{noise}}/B$ に比例し、知りたい領域ほど粗く NaN が出やすい。$k$ は信号側の精度にほとんど効かず分散側だけに効く。実装する側は $k=8$ のまま実測を報告し（A3・§7.2）、費用が目標を超えたときに下げるかどうかは開発者が報告を読んで決める。下げる場合の第一候補は $k$（§5 の 10） |
| A17 | 腕どうしの比べ方 | 記録の軸は `learn_step` のまま。B・RR の違う腕は、経験数でそろえた窓に入る raw の信号・分散の平均の比で比べ、窓内の測定点数を併記する（§7.4）。EMA の幅を経験数基準にする設定は作らない |
| A18 | 0→100M の最初の読み | §1 の「どの時期からノイズに負けるか」に答える Run の計画は、PRD に書かず実験記録で行う |
| A19 | `05_ratio_adam` | 残す。用途は各 Run の中で閾値 1 と比べてライブで読むこととし、Run どうしの比較は A17 による |
| A20 | 成功の判定 | 新しい指標は足さない。受入（§7.1〜7.3）に加え、実験記録に支持 / 反証 / 判定不能と値の未成立率が残ったことで実現とみなす。判定不能が NaN の多さによるなら、§8 を検討する前に raw を長い窓で平滑し直す |

### 9.2 初回グリルの決定履歴（2026-10-15、保持）

以下は初回時点の裁定をそのまま残した記録であり、追加グリルと 3 回目のグリルでの変更は §9.1 による。

| # | 論点 | 裁定 |
|---|---|---|
| D1 | ゴール | 実際の更新（PER・IS 重み・τ・Munchausen 込み。ε は更新と同じ 1 抽選に固定）の noise scale を `learn_step` 軸で記録。学習経路は不変、未購読は計算ゼロ、BF16 AMP ON の本線で読める。noise scale は機序の観測、因果は 091 後の分岐 Run |
| D2 | 非ゴール | 自動調整、曲率入り臨界バッチ、層ごとの内訳、target net 側、DefaultDQN 以外、ε を引き直す版、FP16+GradScaler の Run |
| D3 | 測る batch | 更新自身の batch を並び順に $k$ 分割（PER が i.i.d. なのでシャッフル不要） |
| D4 | 測る場所 | `Optimize(loss, weighted_element_loss)` の 2 引数化。`loss` の計算は autocast 内のまま不変。測定は Optimize 先頭で `autograd::grad(L_j, params, retain_graph=true)` を $k$ 回。`.grad` 不変、乱数不使用。要求フラグは `UpdateFromBatch` が立てる |
| D5 | 部品の置き場 | 純粋計算部は nn_util、配線は `dqn::Learner`。全 Agent 共通の診断フレームワークは作らない |
| D6 | 推定 | $s$・$b$ から §2.4 の式。`EmaFilter` で信号・分散を別々に平滑化して比。EMA(信号) ≤ 0 なら scale / ratio は NaN |
| D7 | Adam 前処理版 | 主読み。$P = \operatorname{diag}(1/(\sqrt{\hat v}+\epsilon))$（偏り補正後の `exp_avg_sq`、`learner.adam_eps`）。固定 $P$ の診断であり実更新の SNR ではない。state の無い param があればその測定は NaN |
| D8 | 群 | all / feature / readout（`feature_key` の依存閉包。重みノルム・Noisy 統計と同じ）。σ は所属 branch の群、勾配未定義の param は数えない。閉包分類を返す口を `Network` に足す。feature / readout 購読時だけ `feature_key` 必須 |
| D9 | 設定キー | `learner.grad_noise.num_micro_batches ?= 8`（≥ 2、B を割り切る。購読時だけ検証）、`learner.grad_noise.ema_alpha ?= 0.05`（(0, 1]、測定 1 回あたり） |
| D10 | 公開 key | §5 の 7 の 11 本（`49_agent_gradnoise`、`$update_result`）。未測定の更新は全 key NaN |
| D11 | 購読 | 新プロファイル `metrics.scalar.@gradnoise`、既定 `interval:503`。baseline に入れない |
| D12 | AMP | BF16 / FP32 は同じ経路（BF16 の丸めはノイズに含まれる）。FP16 で購読したら購読ヒント解釈時に `ANET_SYSTEM_ERROR` |
| D13 | 用語 | [CONTEXT.md](../../CONTEXT.md) に「勾配ノイズスケール」（_Avoid_: ノイズスケール、臨界バッチ、勾配ノイズ） |
| D14 | ADR | [ADR 0054](../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md) |
| D15 | 実装順 | P1 all の素・Adam 版＋プロファイル＋非干渉テスト → P2 feature / readout → P3 文書同期と smoke |

## 10. 複雑さ監査（最終簡素化パス）

### 10.1 初回グリルの監査履歴（2026-10-15、保持）

初回のゴールアンカーを物差しに、導入する機構を逆方向から問うた結果をそのまま保持する。以下の key 数・先送り範囲・最小解の評価は初回時点の記録であり、現行の判断は §10.2 と §10.3 による（`ratio_adam` の現行の位置づけは §10.3）。

| 機構 | 判定 | 理由 |
|---|---|---|
| Optimize 内の小バッチ分割測定 | keep | 10/14 の分岐（B512/RR1 だけ +2.1%）。既存指標に無い |
| Learner の EMA 対と比 | keep | 1 回の推定は負になりうる。`$ema` では比が作れない |
| Adam 前処理版（主読み） | keep | optimizer が AdamW。Adam で素の版が有効 B を数桁外す例（Merrill ら 2025） |
| 素の版 | shrink（all のみ、key 11 / 33 / 34） | 文献比較と $P$ 非依存の参照。費用はノルム 1 組 |
| feature / readout 分割＋閉包分類の口 | keep | all が readout に支配されるリスク。Run は再計算できず、追加費用は小さい |
| 設定キー k・α | keep | 推定の期待値を変えないので、δ / τ を定数にした可塑性の理由は当たらない |
| プロファイル `@gradnoise`（baseline 外） | keep | 費用を確かめる前に既定 ON にしない規約 |
| fail-fast（k、B % k、α、FP16） | keep | フレームワーク所有の契約 |
| raw の信号・分散 4 key | keep | 外で平均し直す用。比の raw は平均し直せないので、暫定の `scale_raw` を置き換えた |
| `ratio_adam` | keep | B の違う Run を重ねたとき閾値 1 がそろう。状態を持たず GetScalar で派生 |
| 部品を nn_util に置く | keep | 置き場だけの決定で追加費用なし。PPO / SAC でも使える |
| ε を引き直す版の総ノイズ測定 | defer-behind-gate | §8 の 1 |
| 曲率入り臨界バッチ・自動調整・層ごと内訳 | defer | §8 の 2〜4 |
| 実更新幅 $\lVert\Delta\theta\rVert$ の記録（サーベイ §7.4） | cut | 単独では SNR にならない。更新の効果は churn 指標が既に測る |
| AMP を非ゴールのまま置く（暫定 PRD） | cut | 本線が BF16。このままでは本 PRD が答えたい Run を測れない |

最小解との差分: pin された要件（BF16 本線で条件付きの noise scale を測る・学習を変えない・未購読は計算ゼロ・ライブで読める・feature / readout・Adam 版が主読み）だけから引き直した最小設計は、上の keep の集合と一致した（差分なし）。

### 10.2 追加グリルの複雑さ監査（2026-10-15）

BF16 本線・非干渉・15 指標の一式完成というゴールアンカーに照らし、残す量と先送りする機構を再確認した。15 指標は数学的な最小集合ではなく、実行中の読みやすさと終了後の再集計を両立させるために選んだ完成範囲である。

| 対象・削減案 | 判定 | 理由・境界 |
|---|---|---|
| feature / readout を省き all だけにする | cut（削減案を不採用） | 群ごとの違いを初回から読みたいという合意を優先。15 指標を一式で完成させる |
| Adam feature / readout の raw 4 key | keep | 各群の信号・分散を後から再集計するために追加。既存 4 key と合わせ raw は 8 key |
| 全体の平滑化済み信号・分散 2 key | keep | ライブでの解釈と実装が使用した EMA 値の確認に使う。raw からの再計算だけへ縮小しない |
| 素の勾配版 3 key | keep | Adam 前処理に依存しない全体の参照を残す。feature / readout の素の版へは広げない |
| 共有測定点を使う版・群ごとの EMA | keep | 購読の和集合で測り各 EMA を 1 回更新する。購読追加で時間幅が変わる性質と、再現に必要な raw の範囲を明記する |
| 購読間隔の混在に対する起動時 1 回の WARN | keep（レビュー反映で追加） | 契約違反ではなく、EMA の時間幅が意図と違う可能性の診断。fail-fast にしない |
| 購読間隔ごとの別 EMA | cut | 状態と系列の増加を避ける。独立した時間幅が必要なら記録した raw を利用側で再平滑化する |
| 使用パラメータ・群ごとの state 成立判定 | keep | 今回の損失と無関係なパラメータや別群の未成立で、成立する観測まで失わないため。stale `.grad` を根拠にしない |
| BatchNorm の専用検出・禁止 | cut | 独立サンプルとしての解釈の制約を文書化し、構成と解釈は利用側が判断する |
| 未成立理由の追加状態指標 | cut | 正常完了 Run の予定点と有限値、raw・平滑化値・測定条件で読む。理由が一意に復元できない場合も含め、追加の公開契約を増やさない |
| nn_util の共通部 | keep（用途を限定） | 単一損失・AdamW の現用途で完結させる。PPO / SAC、複数損失、別 optimizer の拡張口は具体的な利用側ができるまで defer |
| all を先行完成させる P1/P2 区分 | cut | 15 指標の配線・検証・文書同期がそろった時点を完成とする。実装作業の順番と完成単位を混同しない |
| 2% 以下の性能合格線 | shrink（目標に変更） | 費用は反復 A/B で実測して報告し、超過だけで不合格にしない。プロファイラ区間の合計で end-to-end の増加率を代用しない |
| 因果キャンペーンを計測機能の受入に含める | defer（別実験） | まず計測の正しさ・非干渉・本線動作・費用を検証する。B / RR / 学習率の効果は後続の実験で判断する |

初回監査からの差分は、raw 4 key の追加、共有 EMA と群ごとの成立条件の明文化、将来 API の必須要件からの除外、性能目標と完成範囲の整理である。レビュー反映で、空群の raw を NaN に統一し、購読間隔の混在に 1 回の WARN を足した。k・interval・EMA 係数の既定値と、Optimize / nn_util の責任分担は維持した。

### 10.3 3 回目のグリル（2 回目の追加グリル）の複雑さ監査（2026-10-15）

同じゴールアンカーで、追加グリル後の設計を逆方向から問い直した。機構の増減は無く、変更は読み方と用途の文書化だけである。

| 対象・削減案 | 判定 | 理由・境界 |
|---|---|---|
| 既存の全機構（§10.2 の keep） | keep（変更なし） | 今回足したのは文書だけで、削ると戻る困りごとは §10.2 のとおり |
| 推定の精度の目安と NaN の読み方（A16） | keep（文書のみ） | 推定式の性質を読み手に渡す。合成問題で分解能（値の未成立率・誤差）を測る受入の追加は不採用 |
| 経験数でそろえた窓の raw 平均の比（A17） | keep（文書のみ） | 計画済みの分岐 A〜E の読み方。EMA の幅を経験数基準にする設定は、規則が増えるので不採用 |
| 0→100M の最初の読みを PRD に書く | cut | 実験記録の範囲（A18） |
| `05_ratio_adam` | keep（用途を縮小） | A17 で Run 間比較の役目を外し、Run 内で閾値 1 と比べる用途に絞った |
| 最小解（12 key）との差: `21`/`22`・`05` | keep | `21` は scale が NaN のとき、信号が非正（`21` が有限で 0 以下）か未成立（`21` も NaN）かをライブで見分ける唯一の手段。状態は既存の EMA で、追加費用はほぼ無い |
| 一式完成の単位 | keep | 途中で止めても `loss` は不変で、未購読なら計算ゼロなので実害が無い |
| 成功の数値閾値 | cut | §7.4 の判定と値の未成立率の記録で足りる（A20） |

## 11. 文書同期（実装時に更新するもの）

- [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md): §9.8「勾配ノイズスケール」を新設（2 引数 `Optimize`、単一損失・AdamW の共通部、今回の勾配による対象判定、群ごとの state・空群・負の信号、共有測定点と EMA、各 key の公開、BF16 / FP16、サンプル間結合の解釈限界）。§10 の確認事項に第 2 引数の平均と `loss` の一致、§7 の受入観点、PRD092 固有の検証手順への参照を足す。
- [030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md): 4.x に全 15 指標の用途、`01_scale_adam` を主読みとする理由、`05_ratio_adam > 1` が改善を保証しないこと、値の未成立率と原因の限界、購読間隔による EMA 時間幅の変化、内部 EMA の再現に必要な全 raw 測定点、精度の目安と scale が NaN になりがちなときの読み方（raw を長い窓で平滑し直す）、B・RR の違う Run は経験数でそろえた区間の raw の平均の比で比べること、BatchNorm 等の注意を書く。利用者向けの現行説明とし、PRD・ADR を参照させない。
- [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt): `@gradnoise` の 15 行（§5 の 8）。tag 35〜38 と対応する source key を含め、全行 interval 503 とする。
- [agent.txt](../../apps/runner/config/agent.txt): `learner.grad_noise.*` の 2 キー（§5 の 9）。
- [DropMerge.txt](../../apps/runner/config/DropMerge.txt): `run.@gradnoise` のプロファイル。
- `core/anet-core/testdata/prd092/README.md`: 合成問題・成立条件・購読・非干渉・本線 smoke・OFF/ON 各 3 回の性能測定・値の未成立率集計の手順と確認条件をまとめる。決定論設定と BF16 検証環境、記録経路で非有限値が省かれる点も記載する。共通の `AGENTS.md` 検証節には PRD 固有手順を足さない。
- `CHANGELOG.md`: 15 指標の追加と購読による有効化を記載する。
- 英訳（`.en.md`）は `anet-translate-docs` の通常運用に任せる。

## 12. 依存と順序

- [091_checkpoint_load_optimizer_options_10prd.md](091_checkpoint_load_optimizer_options_10prd.md) の後に着手する（ユーザーの順序。091 は実装・レビュー済み）。091 の学習率を変えた継続は、後続の分岐実験で「B か学習率か」を比較するために使う。
- 計算部と `Optimize` の配線、全体・feature / readout の 15 指標と購読をそろえる。次に合成問題・成立条件・異なる間隔と同時発火・決定論環境での非干渉を検証し、本線 smoke、実時間・追加メモリの比較、文書同期までを一式で完了させる（§7・§11）。作業は順に進めても、全体だけを先行完成とはしない。
- attention エントロピー（[093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md)）とは独立に作れる。
- 計測を足すだけで、checkpoint の形は変えない（[930](930_serialize_10prd.md) には関わらない。EMA の状態は保存せず、引き継ぎでは未初期化から始める）。
- 因果を評価するキャンペーンは §7.4 と [サーベイ §7.5](../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md) の A〜E を踏まえ、別の実験記録で設計する。本 PRD の計測機能の受入には含めない。

## 13. 参照

- McCandlish ら 2018「An Empirical Model of Large-Batch Training」（arXiv:1812.06162）: noise scale の定義（2 章）、2 つの大きさのバッチからの推定と移動平均の扱い（付録）。画像分類のほか、Atari や Dota 2 などの強化学習でも測っている。
- Smith ら 2017「Don't Decay the Learning Rate, Increase the Batch Size」（arXiv:1711.00489）: 学習率を下げることと B を上げることが、更新のノイズの面で似た効果を持つこと（連続時間近似）。
- Gray ら 2024「Normalization Layer Per-Example Gradients are Sufficient to Predict Gradient Noise Scale in Transformers」（NeurIPS 2024）: 小バッチを小さくするほど推定の標準誤差が下がること、層ごとの noise scale の違い。
- Wen ら 2018「Flipout」（ICLR 2018）: batch 内で共有された重みの摂動が、B を増やしても消えない分散の床を作ること。
- Merrill ら 2025「Critical Batch Size Revisited」（NeurIPS 2025）: Adam の LLM で noise scale が臨界バッチを数桁過小推定した例と、分岐学習による検証。
- [PyTorch 2.11 BatchNorm2d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm2d.html): batch 統計によるサンプル間の結合。損失を分割しても独立サンプルのノイズとは限らないことの根拠。版は使用中の LibTorch 2.11.0 に合わせる。
- [PyTorch 2.11 Reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html): 同じ seed だけでは実行環境・backend・非決定論演算をまたぐ完全な再現を保証できないこと。非干渉は決定的な同一環境で検証する。
- [サーベイ 2026-10-15](../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md): 上記の整理と、本 PRD への考察（§7）。
- [実験記録 2026-07-27](../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md): 長期 Run での B512/RR2 と B256/RR1 の比較（成熟した checkpoint から B512/RR2 が +13%）。
- [CONTEXT.md](../../CONTEXT.md) の「購読ヒント」「部分 forward」「ノイズ probe」「weight norm 分割（feature/readout）」「勾配ノイズスケール」。
