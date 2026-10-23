# Bootstrapped DQN 型の複数の Head（1 本の Run の中に別々の価値関数を持つ）暫定 PRD

> 2026-10-18 起票。詳細設計は別途グリルで行う前提で、暫定の既定を置き、決めることを §7 に残した。
> 2026-10-23 改訂: §2 の KFM の見立てを直した。当初の「壁は挑む割合が跳ねる正のフィードバックに火が付くか」は、2026-10-18〜20 の既定の 10M × 16 seed で覆り、挑むことは珍しくなく、壁は挑んでいる間に撃破 ÷ 命中が上がらないことにある（[999_intrinsic_reward_novelty](999_intrinsic_reward_novelty_10prd.md) の §2.2・§2.3、[2026-10-15_kfm-boss-route.md](../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md) の探索ブロック 04）。
> 起点: Atari kung_fu_master（以下 KFM）で、ボス行きに入った Run（seed 3）と入らなかった Run の違いが、設定ではなく seed、つまりネットワークがどの谷に落ちたか（方策そのものの違い）だったこと（§2）。lane ごとの ε や UQE の τ の違い（行動のゆらぎ）では、ボスに挑み続ける方策は生まれなかった。
> 関連: [999_intrinsic_reward_novelty_10prd.md](999_intrinsic_reward_novelty_10prd.md)（段階 3 の方策の家族。目的を変えた方策を 1 つのネットワークで持つ）、[999_seed_population_early_stop_10prd.md](999_seed_population_early_stop_10prd.md)（独立の seed を多数回す側）、[084_noisynet_10prd.md](084_noisynet_10prd.md)（lane ごと・episode ごとに保持する仕組みの前例）、[CONTEXT.md](../../CONTEXT.md) の「保持方式」「共有範囲」。

## 1. 目的と問題

1 つのネットワークの中に、Body（特徴抽出）を共有し、Q の Head を K 個持たせる（Bootstrapped DQN、Osband ら 2016）。Head はそれぞれ別の価値関数として学び、Actor の各 lane はエピソードの初めに Head を 1 つ選んで、そのエピソードの間はその Head で行動する。Head ごとに違う谷（KFM ならボス行きか稼ぎか）に落ちうるので、1 本の Run の中で、独立の seed を K 本回すのに近い多様性を、Body の計算を共有した安い形で得る。

### ゴール

- DQN 系 Agent（IQN を含む）で、Q の Head を K 個持てる。既定は 1 個（今と同じ）。
- Actor の lane ごとに、エピソードの初めに Head を選び、エピソードの間は保持できる。
- Learner が Head ごとに損失と target を計算し、Body には全 Head の勾配を平均して流す。
- Head ごとの振る舞い（エピソードの点数・長さ・RAM メトリクス）と、Head どうしの意見の割れを指標で見られる。

### 非ゴール

- 独立のネットワークを複数持つこと（PBT や独立 seed は [999_seed_population_early_stop](999_seed_population_early_stop_10prd.md) の側）。
- 目的（報酬の重みや割引率）を Head ごとに変えること（[999_intrinsic_reward_novelty](999_intrinsic_reward_novelty_10prd.md) の段階 3 の側）。
- ゲームの仕様を変えること（2026-10-16 のユーザーの裁定）。

## 2. 背景

- KFM の実時間優先（既定）で、ボス行きに入ったのは 5 本中 1 本（seed 3）。seed 3 と seed 2 の設定の差は `run.seed` だけだった。
- seed 3 は 3-4M にボス命中 10%・撃破 ÷ 命中 0.03 で、2026-10-17 に挑んだが撤退した UQE の腕と同じ水準だった。そこから 4-5M に命中 51% へ跳ね、撃破 ÷ 命中は 4-5M の 0.24 から 5-6M に 0.40、6-7M に 0.71、11-16M に 0.9 まで上がった。
- 挑むことは珍しくない。seed 3 と同じ既定の設定を 10M で 16 seed 回すと、0.5M 窓の命中率のピークが 10% 以上の Run が 7 本あったが、挑んでいる間に撃破 ÷ 命中が上がった Run は無く、定着した Run も無かった。s08 は挑み始めの撃破 ÷ 命中が 0.22 と seed 3 の 0.24 とほぼ同じだったが、0.14 → 0.09 と下がり、7.5M 以降は挑まなかった。壁は挑み始めることではなく、挑んでいる間に勝ち方を覚えることにある（[999_intrinsic_reward_novelty](999_intrinsic_reward_novelty_10prd.md) の §2.2・§2.3、[2026-10-15_kfm-boss-route.md](../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md) の探索ブロック 04）。当初は「壁は稼ぎが固まる前に、挑む割合が跳ねる正のフィードバックに火が付くか」と書いたが、この 16 seed で覆った。
- 行動のゆらぎでは足りなかった。2026-10-17 の UQE（低い ε）の Run では、4〜12M のボス命中が lane の 4 区分で 7.1% / 7.6% / 10.8% / 10.2% とほぼ一様で、lane ごとの τ の違いがあっても全 lane が 1 つの戦略として挑み、1 つの戦略として撤退した。
- 方策の多様性には 3 つの段がある（① 行動のゆらぎ、② 目的を変えた方策の家族、③ 別々のパラメータ）。seed 3 との違いは ③ で、本 PRD は ③ を 1 本の Run の中で安く作る。
- Bootstrapped DQN（Osband ら 2016）は、Head ごとに別の Q を持ち、エピソードの間 1 つの Head で行動することで、時間的に一貫した探索（deep exploration）を作る。Atari では、遷移を Head ごとに間引く bootstrap の印を付けなくても（全 Head が全遷移を学んでも）、Head の初期値の違いだけで多様性が出たと報告している（本 PRD の既定はこれに従う。原文での照合はグリルの前に行う）。Osband ら 2018 は、Head ごとに固定の乱数の「事前関数」を足すと多様性が保たれることを示した。

## 3. 現状（2026-10-18 に確認したこと）

- 実時間優先の Atari のネットワークは、Body の `main_feature`（IMPALA ×2 ＋ LayerNorm）と、価値と優位の 2 本の流れ（`value_stream` / `adv_stream`、各 `AtariHeadFC512 > ReLU`）と、射影（`head_projection`）からなる（[Atari.txt](../../apps/runner/config/Atari.txt) の `run.@va_base`）。Head は 1 組だけ。
- Actor は 1 回の forward で全 lane の Q（IQN では τ ごとの分位点の平均）を出し、その argmax と ε で行動を決める。
- NoisyNet（PRD 084）は、ノイズを lane ごとに、episode の始まりで引き直して保持する仕組み（CONTEXT の「保持方式」の `episode`、「共有範囲」の `sample`）を持つ。Head の選択も「lane ごとに episode の間保持する値」で、同じ形をしている。
- Learner の損失・target・PER の優先度は 1 つの Q を前提にしている。target network は hard 更新（実時間優先では 125 回ごと）。
- replay は遷移ごとに Head の印を持たない。

## 4. 提案の骨子

### 段階 1: 複数の Head と、lane ごとの Head の保持

1. **Head**: 価値と優位の流れと射影を K 組持つ（既定 K = 8）。Body（`main_feature` と IQN の τ の埋め込みの合成）は共有する。target network も同じ形で全 Head を持つ。
2. **Actor**: lane ごとに、エピソードの始まりで Head を一様に選び、エピソードの間保持する。1 回の forward で全 Head を計算し、lane ごとに選んだ Head の Q で行動を決める。ε（spatial のラダー）はそのまま掛ける。
3. **Learner**: Head ごとに、その Head の Q と、その Head の target で損失を計算する。既定では全 Head が全遷移を学ぶ（bootstrap の印なし）。Body には全 Head の勾配の平均を流す（Osband ら 2016 と同じく 1/K で正規化）。
4. **評価**: 評価の Actor は、全 Head の Q の平均で行動する（既定）。Head ごとの評価は指標の側で行う。
5. **指標**:
   - Head どうしの意見の割れ（probe のバッチで、Head ごとの argmax が一致しない割合）。
   - Head ごとの学習中のエピソードの点数と長さ。
   - 学習側の episode の trace に、その lane が使った Head の番号を載せる。KFM では RAM メトリクス（ボス命中・撃破・面クリア）と並べて、どの Head がボス行きに入ったかを見る。

段階 1 だけで、1 本の Run の中に別々の谷へ落ちる Head がいくつできるか、そのうちボス行きの Head があるかを確かめられる。

### 段階 2: Head ごとの多様性を保つ仕組み

1. **事前関数**: Head ごとに、固定の乱数の小さなネットワークの出力（係数つき）を Q に足す（Osband ら 2018）。学習が進んでも Head が似てこないようにする。
2. **bootstrap の印**: 遷移ごとに K ビットの印を付け、Head ごとに一部の遷移だけで学ぶ（確率 p、既定は段階 1 と同じ p = 1）。

段階 2 は、段階 1 で Head どうしの意見の割れがすぐに消える（Head が同じ谷に集まる）と分かったときに入れる。

## 5. User Stories

1. 研究者として、1 本の Run の中で、別々の戦略に落ちる価値関数を複数育て、どの Head がどの戦略を取ったかを見たい。
2. 研究者として、KFM で、どれか 1 つの Head でもボス行きに火が付けば、その経験が Body を通して他の Head や評価の方策にも効くかを確かめたい。
3. 研究者として、Head の数を 1 にしたときに今と同じ学習系列になることを確かめたい。

## 6. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| Head ごとに目的（報酬の重み、割引率）を変える | 対象外 | 999_intrinsic_reward_novelty の段階 3 |
| 成績の良い Head を他の Head へコピーする（Head の単位の PBT） | 先送り | 点数で選ぶと、覚えている途中のボス行きを捨てる（seed 3 の eval2 は 10-20M に稼ぎの seed 1 より低かった）。選ぶ基準が決まってから起こす |
| Head の不確かさ（意見の割れ）を行動の選び方に使う（UCB など） | 先送り | まず Head を 1 つ選んで通す形で多様性が出るかを見る |
| Head ごとに別の PER の優先度を持つ | 不採用 | replay の優先度は遷移ごとに 1 つ。§7 の 4 のとおり Head の値をまとめる |

## 7. 決めること（詳細設計で詰める）

暫定の既定を置いた。グリルで確定する。

| # | 決めること | 暫定の既定 |
| --- | --- | --- |
| 1 | 設定キー | `DefaultDQNAgent.net.num_q_heads`（1 以上の整数、既定 1）。1 未満は構築時に fail-fast |
| 2 | Head の作り方 | NN 設定の流れ（`value_stream` / `adv_stream`）と射影を K 回複製する。設定の書き方（1 つの定義から K 組作るか）を NN 設定の仕組みに合わせて決める |
| 3 | lane ごとの Head の保持 | NoisyNet の lane ごと・episode ごとの保持と同じ形で持つ。仕組みを共有するか、Head の選択専用にするか |
| 4 | PER の優先度 | Head ごとの \|TD\| の平均。最大にするかはグリルで決める |
| 5 | Munchausen と UQE | Munchausen の項は Head ごとの Q で計算する。UQE は選んだ Head の分布で選ぶ |
| 6 | 評価の行動 | 全 Head の Q の平均（既定）。多数決にするか、学習中に最も点の高い Head を使うか |
| 7 | 計算の費用 | 流れの全結合（特徴 → 512）が K 倍になる。K = 8 で learn の速さがどれだけ落ちるかを測り、目安を決める |
| 8 | 指標の tag | 群番号 58（`58_agent_qheads/...`）。例 `58_agent_qheads/01_argmax_disagree`。40 文字以内 |
| 9 | 用語 | 本 PRD の Head は「価値と優位の流れ＋射影」の 1 組、Body はそれより前の共有部分（`main_feature` と τ の埋め込みの合成）を指す。NN 設定の `head_projection`（射影だけ）と範囲が違うので、CONTEXT に Head / Body と「Head の選択」を足すときに書き分ける |
| 10 | ADR | 「Head は lane ごとに episode の間保持し、評価は全 Head の平均で行う」を ADR にするか |

## 8. 成功の測り方と受入の目安

成功の測り方:

- KFM（実時間優先、既定の設定）で、Head どうしの意見の割れが学習の後半まで残る。Head ごとのボス命中を見ると、少なくとも 1 つの Head が他より高い割合でボスに挑み続ける。
- 多数 seed の短い Run（10〜12M）で、「10M までにどれかの Head、または評価の方策がボス行きに火が付く割合」を、Head 1 個の基準（5 本中 1 本）と比べる。
- 害の確認: Breakout など 1 ゲームで、評価の点数が基準の seed 幅に収まる。

受入の目安（詳細はグリルで確定する）:

1. `num_q_heads = 1` で、本機能の前後の同 seed・決定的モードの短い Run の metrics checksum が一致する。
2. 単体テスト: Head ごとの損失が、その Head の Q と target だけで計算され、Body の勾配が全 Head の平均になる。
3. 単体テスト: lane ごとの Head の選択が、episode の始まりでだけ変わり、episode の間は保持される。
4. smoke Run（K = 8）で、新しい tag と、trace の Head の番号が出る。
5. learn の速さの低下を、ラウンドロビンの A/B で測って記録する。

## 9. 依存と順序

依存は無い。段階 2 は段階 1 の結果を見てから。KFM での評価は、多数 seed の短い Run（[999_seed_population_early_stop](999_seed_population_early_stop_10prd.md) の段階 1、または 13 bat の 10M × 16 seed）の基準が揃ってから組む。

## 10. 参照

- Osband ら 2016「Deep Exploration via Bootstrapped DQN」（NeurIPS 2016、arXiv 1602.04621）
- Osband ら 2018「Randomized Prior Functions for Deep Reinforcement Learning」（NeurIPS 2018、arXiv 1806.03335）
