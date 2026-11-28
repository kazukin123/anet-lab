# 実装サーベイ: IQN / FQF / QR-DQN における τ(quantile fraction)生成

- 調査日: 2026-08-12
- 調査手段: WebSearch / WebFetch(GitHub raw ソース閲覧含む)。ローカルコードは対象外。
- コードの参照はすべて各リポジトリの default branch(master / main)を 2026-08-12 に取得したもの。commit hash は WebFetch では取得できないため branch 参照で記載(citation not confirmed for exact commit)。
- 凡例: 【公式】= アルゴリズム提案元またはフレームワーク運営組織による実装/記事。【第三者】= 個人等による再現実装/解説。

---

## 1. Google Dopamine(IQN 論文著者所属 Google の公式フレームワーク)【公式】

運営: Google(Dopamine は IQN 論文(Dabney et al. 2018)と同系のチームが公開する研究フレームワーク)。

### 1.1 JAX 版 `ImplicitQuantileNetwork`

τ は毎 forward で一様乱数。工夫(層化・ソート等)は無し。

```python
state_net_tiled = jnp.tile(x, [num_quantiles, 1])
quantiles_shape = [num_quantiles, 1]
quantiles = jax.random.uniform(rng, shape=quantiles_shape)
quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
```
[google/dopamine master (2026/08 参照), dopamine/jax/networks.py, class ImplicitQuantileNetwork]

### 1.2 TensorFlow(legacy)版

こちらも `tf.random.uniform` の一様乱数。サンプリングはエージェントではなくネットワーク側(`legacy_networks.ImplicitQuantileNetwork.call`)にある。TF 版エージェント(`dopamine/tf/agents/implicit_quantile/implicit_quantile_agent.py`)は `num_tau_samples`(online loss 用)/`num_tau_prime_samples`(target 用)/`num_quantile_samples`(Q 値計算用)の 3 種のサンプル数を持つ。

```python
quantiles_shape = [num_quantiles * batch_size, 1]
quantiles = tf.random.uniform(
    quantiles_shape, minval=0, maxval=1, dtype=tf.float32
)
```
[google/dopamine master (2026/08 参照), dopamine/discrete_domains/legacy_networks.py, ImplicitQuantileNetwork.call]

---

## 2. DeepMind DQN Zoo / rlax / Acme【公式】

運営: Google DeepMind。DQN Zoo は「DeepMind で開発された DQN 系エージェントのリファレンス実装集」を自称する公式リポジトリ。

### 2.1 DQN Zoo IQN

τ 生成はヘルパ関数で一様乱数。actor(行動選択)と learner(loss の 3 箇所: online / policy 選択 / target)すべて同じ `_sample_tau` を使う。層化等の工夫は無し。

```python
def _sample_tau(
    rng_key: parts.PRNGKey,
    shape: Tuple[int, ...],
) -> jnp.ndarray:
  """Samples tau values uniformly between 0 and 1."""
  return jax.random.uniform(rng_key, shape=shape)
```
[google-deepmind/dqn_zoo master (2026/08 参照), dqn_zoo/iqn/agent.py]

### 2.2 DQN Zoo QR-DQN

固定 τ = 等間隔中点 (i+0.5)/N。

```python
quantiles = (jnp.arange(0, num_quantiles) + 0.5) / float(num_quantiles)
```
[google-deepmind/dqn_zoo master (2026/08 参照), dqn_zoo/qrdqn/run_atari.py]

### 2.3 rlax(loss ライブラリ)

rlax 自体は τ をサンプリングせず、引数として受け取る設計(`quantile_regression_loss(dist_src, tau_src, ...)`、`quantile_q_learning(dist_q_tm1, tau_q_tm1, ...)`)。docstring は次のとおり。

```python
tau_src: source distribution probability thresholds.
```
[google-deepmind/rlax master (2026/08 参照), rlax/_src/value_learning.py, quantile_regression_loss docstring]

### 2.4 Acme

Acme の JAX DQN には QR-DQN loss(`QrDqn`)があり、固定中点 τ を作って rlax に渡す。IQN loss はこのファイルには無い。

```python
quantiles = (
    (jnp.arange(self.num_atoms, dtype=jnp.float32) + 0.5) / self.num_atoms)
```
[google-deepmind/acme master (2026/08 参照), acme/agents/jax/dqn/losses.py, class QrDqn]

---

## 3. pfrl(旧 ChainerRL、Preferred Networks)【公式フレームワーク】

運営: Preferred Networks(ChainerRL の PyTorch 後継)。τ は `torch.rand` の一様乱数を 3 箇所(K: 行動選択用 / N: online / N': target)で独立に引く。

```python
taus = torch.rand(
    batch_size,
    self.quantile_thresholds_N,
    device=self.device,
    dtype=torch.float,
)
```
[pfnet/pfrl master (2026/08 参照), pfrl/agents/iqn.py, _compute_y_and_taus]

特徴的な点: `act_deterministically=True` のときだけ乱数をやめ、等間隔 `linspace` に切り替える(評価時の決定化)。

```python
taus_tilde = torch.linspace(
    start=0, end=1, steps=self.quantile_thresholds_K,
    device=self.device, dtype=torch.float,
).repeat(len(batch_obs), 1)
```
[pfnet/pfrl master (2026/08 参照), pfrl/agents/iqn.py, _evaluate_model_and_update_recurrent_states]

なお ChainerRL 時代の IQN 導入 issue(chainer/chainerrl #282、muupan 氏が 2018-06-24 起票)は論文リンクのみで、τ サンプリング方式の議論は無い。

---

## 4. Tianshou(清華大学 thu-ml)【公式フレームワーク】

運営: THU-ML(清華大学)。IQN の τ 生成はネットワーク側 `ImplicitQuantileNetwork.forward` で一様乱数。

```python
taus = torch.rand(batch_size, sample_size, dtype=logits.dtype, device=logits.device)
```
[thu-ml/tianshou master (2026/08 参照), tianshou/utils/net/discrete.py, ImplicitQuantileNetwork.forward]

FQF の `FractionProposalNetwork` は log-softmax→cumsum で τ を生成し、中点 τ̂ を detach する。docstring に第三者実装からの移植であることが明記されている。

```python
taus_1_N = torch.cumsum(dist.probs, dim=1)
taus = F.pad(taus_1_N, (1, 0))
tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0

# docstring: "Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/network.py"
```
[thu-ml/tianshou master (2026/08 参照), tianshou/utils/net/discrete.py, FractionProposalNetwork.forward]

FQF の fraction loss は W1 勾配の符号場から組み立て、エントロピー正則化(`ent_coef`)を引いて fraction 専用 optimizer で更新する(現 master ではファイルが `tianshou/algorithm/modelfree/fqf.py` に移動。0.4.x 系では `tianshou/policy/modelfree/fqf.py`)。

```python
fraction_loss = (gradient_of_taus * taus[:, 1:-1]).sum(1).mean()
...
entropy_loss = out.fractions.entropies.mean()
fraction_entropy_loss = fraction_loss - self.ent_coef * entropy_loss
self.fraction_optim.step(fraction_entropy_loss, retain_graph=True)
```
[thu-ml/tianshou master (2026/08 参照), tianshou/algorithm/modelfree/fqf.py]

FQF proposal network に関する「既知の issue」: 検索では Tianshou リポジトリの τ / fraction proposal に関する具体的な bug issue は特定できなかった(見つからなかった)。

---

## 5. d3rlpy(Takuma Seno 氏、オフライン RL ライブラリ)【公式フレームワーク(個人発・現在は広く利用)】

作者: Takuma Seno(妹尾卓磨)氏。学習時は一様乱数、推論時は等間隔 linspace に切り替える明示的な train/eval 分岐を持つ。

```python
def _make_taus(
    batch_size: int, n_quantiles: int, training: bool, device: torch.device
) -> torch.Tensor:
    if training:
        taus = torch.rand(batch_size, n_quantiles, device=device)
    else:
        taus = torch.linspace(start=0, end=1, steps=n_quantiles, ...)
```
[takuseno/d3rlpy master (2026/08 参照), d3rlpy/models/torch/q_functions/iqn_q_function.py, _make_taus(一部省略)]

---

## 6. Stable-Baselines3 Contrib【公式フレームワーク】

運営: Stable-Baselines3 チーム(Antonin Raffin 氏ら)。IQN/FQF は無く QR-DQN のみ。τ は `quantile_huber_loss` 内で固定中点として生成。

```python
if cum_prob is None:
    n_quantiles = current_quantiles.shape[-1]
    # Cumulative probabilities to calculate quantiles.
    cum_prob = (th.arange(n_quantiles, device=current_quantiles.device, dtype=th.float) + 0.5) / n_quantiles
```
[Stable-Baselines-Team/stable-baselines3-contrib master (2026/08 参照), sb3_contrib/common/utils.py, quantile_huber_loss]

---

## 7. CleanRL【公式フレームワーク】

運営: CleanRL プロジェクト(Costa Huang 氏ら)。アルゴリズム一覧に IQN / QR-DQN / FQF は無い。分布型は C51 のみ。

```text
✅ [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf)
（一覧に IQN / QR-DQN / FQF は記載なし）
```
[CleanRL docs (2026/08 参照), docs.cleanrl.dev/rl-algorithms/overview/]

---

## 8. DI-engine(OpenDILab / SenseTime 系)【公式フレームワーク】

運営: OpenDILab。IQN 用 `QuantileHead` は一様乱数だが、行動選択側の τ に beta function(リスク歪曲)を通せる点が特徴。FQF 用 head は cumsum + 中点 detach(toshikwa 系と同型)。

```python
q_quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1).to(x)
logit_quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1).to(x)
# logit_quantiles には self.beta_function(logit_quantiles) を適用
```
[opendilab/DI-engine main (2026/08 参照), ding/model/common/head.py, QuantileHead.forward]

```python
q_quantiles = torch.cumsum(q_quantiles, dim=1)
tau_0 = torch.zeros((batch_size, 1)).to(x)
q_quantiles = torch.cat((tau_0, q_quantiles), dim=1)
q_quantiles_hats = (q_quantiles[:, 1:] + q_quantiles[:, :-1]).detach() / 2.
```
[opendilab/DI-engine main (2026/08 参照), ding/model/common/head.py, FQFHead.forward]

---

## 9. 有名第三者リファレンス実装

### 9.1 toshikwa(ku2482)/ fqf-iqn-qrdqn.pytorch【第三者】

作者: Toshiki Watanabe 氏(Qiita @ku2482)。Tianshou の FQF が明示的に依拠する再現実装。IQN の τ は一様乱数(online N 個 / target N' 個を独立に)。

```python
taus = torch.rand(
    self.batch_size, self.N, dtype=state_embeddings.dtype,
    device=state_embeddings.device)
```
[toshikwa/fqf-iqn-qrdqn.pytorch master (2026/08 参照), fqf_iqn_qrdqn/agent/iqn_agent.py, calculate_loss]

FQF は log_softmax→cumsum→中点 detach。

```python
log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
taus_1_N = torch.cumsum(probs, dim=1)
taus = torch.cat((tau_0, taus_1_N), dim=1)
tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
```
[toshikwa/fqf-iqn-qrdqn.pytorch master (2026/08 参照), fqf_iqn_qrdqn/network.py, FractionProposalNetwork.forward]

### 9.2 BY571 / IQN-and-Extensions【第三者】

作者: Sebastian Dittert 氏。PER / Noisy / N-step / Dueling / Munchausen 等の拡張付き IQN。τ は一様乱数のみで層化等の工夫は無し。

```python
taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
```
[BY571/IQN-and-Extensions master (2026/08 参照), model.py, calc_cos]

### 9.3 Kaixhin / Rainbow【第三者】

作者: Kai Arulkumaran 氏。分布型要素は C51(カテゴリカル、固定 atom)であり、τ(quantile fraction)は存在しない。README の参照文献は Bellemare et al. 2017(C51)で、QR-DQN/IQN への言及は無い。

### 9.4 τ 生成に「変わった工夫」をしている実装の有無

今回サーベイした範囲(Dopamine / DQN Zoo / rlax / Acme / pfrl / Tianshou / d3rlpy / SB3-contrib / DI-engine / toshikwa / BY571)では、学習時の τ に stratified sampling・ソート・低差異列(quasi-random)等を使う実装は**見つからなかった**。乱数一様サンプリング以外のバリエーションとして確認できたのは次の 3 種のみ:

1. 評価/決定的行動時の等間隔 linspace 切替(pfrl、d3rlpy)
2. リスク歪曲関数(beta function)を行動選択 τ に適用(DI-engine)
3. τ 自体を学習する FQF 系(fraction proposal network、Tianshou / DI-engine / toshikwa)

---

## 10. 実務者向け解説記事・ブログ

### 10.1 Microsoft Research Blog(FQF 提案元)【公式】

著者: Li Zhao(Principal Researcher, Microsoft Research Asia)、2019-12-18。IQN のランダムな quantile fraction が最適とは限らないことを FQF の動機として明言。

```text
"the sampled quantile fractions aren't necessarily the best quantile fractions."
"This motivated us to find the learning targets—that is, the quantile fractions—that would result in the least approximation error."

(訳)サンプリングされた quantile fraction が最良の fraction とは限らない。これが、近似誤差を最小にする学習ターゲット(=quantile fraction)を見つけようという動機になった。
```
[Microsoft Research Blog (2019/12), FQF 紹介記事本文]

### 10.2 Qiita @ku2482(Toshiki Watanabe 氏)IQN 解説【第三者】

2020-02-03 投稿(2020-03-04 更新)。τ の一様サンプリングと N/K/N' の効果を解説し、自身の再現実装(9.1)へリンク。

```text
一様分布からのサンプル τ ～ U([0,1]) を用いると，Z_τ(x,a) ～ Z(x,a) となることがわかります
```
[Qiita @ku2482 IQN 解説 (2020/02), 本文]

記事では実験結果として N=64 が最良、N'=8 で十分、K=32 を使用と整理している(数値は記事の論文まとめ部分。原典は IQN 論文)。

### 10.3 Qiita @keisuke-nakata IQN 論文解説【第三者】

2019-03-05 投稿(2020-06-07 更新)。所属記載なし。τ ～ U([0,1]) の定式化、リスク歪曲(CPW / Wang / Power / CVaR)、サンプル数の効果に言及。

```text
N=N′=8で十分であるとわかった
（また）N=N′=1の場合であっても DQN よりも3倍性能が良い
```
[Qiita @keisuke-nakata IQN 解説 (2019/03), 本文(サンプル数に関する論文まとめ)]

### 10.4 horomary 氏「どこから見てもメンダコ」FQF 解説【第三者】

2021-04-23。IQN の「ランダムサンプリングされた分位 τ」の限界(固定分位を予測できない・多数サンプルが必要)を指摘し、FQF を「状態 s に応じていい感じの分位 τ セットを提案するネットワーク」の追加として説明。stratified sampling への言及は無し。

```text
（IQN は）ランダムサンプリングされた分位τを与えて（分位点を予測する）
（FQF は）状態sに応じていい感じの分位τセットを提案するネットワーク（を追加する）
```
[horomary はてなブログ FQF 解説 (2021/04), 本文(要約的抜粋)]

### 10.5 Opher Lieber 氏「Fast Sample Efficient Q-Learning With Recurrent IQN」【第三者】

2019-09-22。R2D2 系 + IQN の実験記事。τ サンプル数(32、代替検討では 8)への言及はあるが、τ 生成方式そのもの(uniform 以外)の選択・影響の議論は無い。

### 10.6 見つからなかったもの

- τ の stratified 化・低差異列化を明示的に論じる実務者ブログ: 信頼できる著者のものは**見つからなかった**(検索でヒットした shadecoder.com の記事は運営実態不明の SEO 型サイトのため除外)。
- τ サンプリング方式の比較実験を行った個人ブログ: 見つからなかった。

---

## 11. GitHub issue / discussion での τ サンプリング議論

- τ サンプリングの偏り・stratified 化を主題にした issue / discussion は、Dopamine / Tianshou / pfrl / dqn_zoo / toshikwa / BY571 の各リポジトリを対象に検索した範囲では**見つからなかった**。
- 近いものとして確認できたのは以下のみ:
  - google/dopamine #37「Reproducing the scores reported by the IQN paper」— スコア再現性の議論だが、原因は環境バージョン(v4 vs v0 sticky actions)であり τ 生成の議論ではない。
  - chainer/chainerrl #282 — IQN 実装依頼 issue(muupan 氏、2018-06-24)。本文は論文リンクのみで τ の議論なし。

---

## まとめ(τ 生成方式の一覧)

| 実装 | 区分 | アルゴリズム | 学習時 τ | 推論/評価時 τ |
|---|---|---|---|---|
| Dopamine (JAX/TF) | 公式 | IQN | `random.uniform` | 同左(乱数) |
| DQN Zoo | 公式 | IQN / QR-DQN | uniform / 固定中点 (i+0.5)/N | 同左 |
| rlax | 公式 | QR 系 loss | τ は引数(サンプリングしない) | — |
| Acme | 公式 | QR-DQN | 固定中点 (i+0.5)/N | 同左 |
| pfrl | 公式 | IQN | `torch.rand` ×3 (K/N/N') | `act_deterministically` で linspace |
| Tianshou | 公式 | IQN / FQF | `torch.rand` / FPN(cumsum+中点detach) | 同左 |
| d3rlpy | 公式 | IQN | `torch.rand` | linspace(train/eval 分岐) |
| SB3-contrib | 公式 | QR-DQN | 固定中点 (i+0.5)/N | 同左 |
| CleanRL | 公式 | (C51 のみ、τ 無し) | — | — |
| DI-engine | 公式 | IQN / FQF | `uniform_` / FPN | 行動選択 τ に beta function 適用可 |
| toshikwa/fqf-iqn-qrdqn | 第三者 | IQN/FQF/QR-DQN | `torch.rand` / FPN | 同左 |
| BY571/IQN-and-Extensions | 第三者 | IQN+拡張 | `torch.rand` | 同左 |
| Kaixhin/Rainbow | 第三者 | C51(τ 無し) | — | — |

stratified / quasi-random 等の τ 生成の工夫は、公式・第三者いずれの主要実装にも確認できなかった。乱数一様以外の実在バリエーションは「評価時 linspace」「リスク歪曲」「FQF(τ を学習)」の 3 系統。

---

## ソース一覧

- [google/dopamine jax networks, 2026/08] Google. "dopamine/jax/networks.py (master)." GitHub. https://github.com/google/dopamine/blob/master/dopamine/jax/networks.py
- [google/dopamine legacy networks, 2026/08] Google. "dopamine/discrete_domains/legacy_networks.py (master)." GitHub. https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/legacy_networks.py
- [google/dopamine tf iqn agent, 2026/08] Google. "dopamine/tf/agents/implicit_quantile/implicit_quantile_agent.py (master)." GitHub. https://github.com/google/dopamine/blob/master/dopamine/tf/agents/implicit_quantile/implicit_quantile_agent.py
- [dqn_zoo iqn, 2026/08] Google DeepMind. "dqn_zoo/iqn/agent.py (master)." GitHub. https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/iqn/agent.py
- [dqn_zoo qrdqn, 2026/08] Google DeepMind. "dqn_zoo/qrdqn/run_atari.py (master)." GitHub. https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/qrdqn/run_atari.py
- [rlax value_learning, 2026/08] Google DeepMind. "rlax/_src/value_learning.py (master)." GitHub. https://github.com/google-deepmind/rlax/blob/master/rlax/_src/value_learning.py
- [acme dqn losses, 2026/08] Google DeepMind. "acme/agents/jax/dqn/losses.py (master)." GitHub. https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/dqn/losses.py
- [pfrl iqn, 2026/08] Preferred Networks. "pfrl/agents/iqn.py (master)." GitHub. https://github.com/pfnet/pfrl/blob/master/pfrl/agents/iqn.py
- [chainerrl issue 282, 2018/06] muupan. "Implicit Quantile Networks for Distributional Reinforcement Learning · Issue #282." GitHub chainer/chainerrl. https://github.com/chainer/chainerrl/issues/282
- [tianshou discrete nets, 2026/08] THU-ML. "tianshou/utils/net/discrete.py (master)." GitHub. https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/discrete.py
- [tianshou fqf, 2026/08] THU-ML. "tianshou/algorithm/modelfree/fqf.py (master)." GitHub. https://github.com/thu-ml/tianshou/blob/master/tianshou/algorithm/modelfree/fqf.py
- [d3rlpy iqn, 2026/08] Takuma Seno. "d3rlpy/models/torch/q_functions/iqn_q_function.py (master)." GitHub. https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/models/torch/q_functions/iqn_q_function.py
- [sb3-contrib utils, 2026/08] Stable-Baselines3 Team. "sb3_contrib/common/utils.py (master)." GitHub. https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/utils.py
- [cleanrl overview, 2026/08] CleanRL. "Overview — CleanRL docs." https://docs.cleanrl.dev/rl-algorithms/overview/
- [di-engine head, 2026/08] OpenDILab. "ding/model/common/head.py (main)." GitHub. https://github.com/opendilab/DI-engine/blob/main/ding/model/common/head.py
- [toshikwa network, 2026/08] Toshiki Watanabe. "fqf_iqn_qrdqn/network.py (master)." GitHub toshikwa/fqf-iqn-qrdqn.pytorch. https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/network.py
- [toshikwa iqn agent, 2026/08] Toshiki Watanabe. "fqf_iqn_qrdqn/agent/iqn_agent.py (master)." GitHub toshikwa/fqf-iqn-qrdqn.pytorch. https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/agent/iqn_agent.py
- [BY571 model, 2026/08] Sebastian Dittert. "model.py (master)." GitHub BY571/IQN-and-Extensions. https://github.com/BY571/IQN-and-Extensions/blob/master/model.py
- [Kaixhin Rainbow, 2026/08] Kai Arulkumaran. "Rainbow (README)." GitHub. https://github.com/Kaixhin/Rainbow
- [MSR FQF blog, 2019/12] Li Zhao. "Finding the best learning targets automatically: Fully Parameterized Quantile Function for distributional RL." Microsoft Research Blog. https://www.microsoft.com/en-us/research/blog/finding-the-best-learning-targets-automatically-fully-parameterized-quantile-function-for-distributional-rl/
- [Qiita ku2482 IQN, 2020/02] Toshiki Watanabe (@ku2482). "[論文解説] IQN: Implicit Quantile Networks for Distributional Reinforcement Learning." Qiita. https://qiita.com/ku2482/items/2df60889d87e4018d44d
- [Qiita ku2482 FQF, 2020/02] Toshiki Watanabe (@ku2482). "[論文解説] FQF: Fully Parameterized Quantile Function for Distributional Reinforcement Learning." Qiita. https://qiita.com/ku2482/items/044f0f75805b5978b902
- [Qiita keisuke-nakata IQN, 2019/03] @keisuke-nakata. "【論文】Implicit Quantile Networks for Distributional Reinforcement Learning (IQN; 2018)." Qiita. https://qiita.com/keisuke-nakata/items/1f6d4387fc6f01a5bcac
- [horomary FQF, 2021/04] horomary. "深層分布強化学習 ③FQF: Fully Parameterized Quantile Function for Distributional RL." どこから見てもメンダコ(はてなブログ). https://horomary.hatenablog.com/entry/2021/04/23/214611
- [Lieber Recurrent IQN, 2019/09] Opher Lieber. "Fast Sample Efficient Q-Learning With Recurrent IQN." 個人ブログ. https://opherlieber.github.io/rl/2019/09/22/recurrent_iqn
- [dopamine issue 37, 2018] (起票者不明). "Reproducing the scores reported by the IQN paper · Issue #37." GitHub google/dopamine. https://github.com/google/dopamine/issues/37
