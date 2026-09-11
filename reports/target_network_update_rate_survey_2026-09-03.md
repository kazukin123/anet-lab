# Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法

Date: 2026-09-03
Scope: 深層強化学習における target network の更新速度について、(1) 主要アルゴリズムが使う値とその選定根拠、(2) 遅れと安定性・バイアスの理論解析、(3) 自動・適応調整手法の存在、(4) 調整に使われる観測指標、(5) 遅れを勾配ステップ軸と環境ステップ軸のどちらで測るかの議論、の 5 点を一次情報から確定させる。

調査は 4 つの角度（実設定値と選定根拠 / 理論解析 / 自動・適応調整 / 調整指標と replay ratio 連動）へ分けて並行実施し、本書へ統合した。確認した出典は延べ 135 件、うち本書が直接引用したものを §10 に挙げる。

## Table of Contents

1. [単位の不一致という前提](#1-単位の不一致という前提)
2. [主要アルゴリズムの設定値と選定根拠](#2-主要アルゴリズムの設定値と選定根拠)
3. [更新速度と収束・バイアスの理論](#3-更新速度と収束バイアスの理論)
4. [自動・適応調整の手法](#4-自動適応調整の手法)
5. [調整に使われる観測指標](#5-調整に使われる観測指標)
6. [replay ratio との連動](#6-replay-ratio-との連動)
7. [批判と留保](#7-批判と留保)
8. [総合評価](#8-総合評価)
9. [調査の限界](#9-調査の限界)
10. [出典リスト](#10-出典リスト)

---

## 1. 単位の不一致という前提

**C や τ の値を文献間で比較する前に、単位を確認する必要がある。同じ「10,000」が論文と公式実装で 4 倍違う例がある。**

Nature DQN の付録表は、C の単位を parameter update 数と明示している。

```
target network update frequency | 10000 | The frequency (measured in the number of parameter updates) with which the target network is updated (this corresponds to the parameter C from Algorithm 1).

target network 更新頻度 | 10000 | target network が更新される頻度（parameter update の回数で測る）。これは Algorithm 1 のパラメータ C に対応する。
```

[Mnih DQN (2015/02), Extended Data Table 1]

一方、公式 Lua 実装では同じ 10,000 が agent step 単位のカウンタに対する剰余で判定されており、`update_freq = 4` と組み合わせると 2,500 parameter updates に相当する（本調査で公式実装 `google-deepmind/dqn` の `NeuralQLearner.lua` を確認）。Double DQN 論文が「τ = 10,000 steps」と書いているのは、論文の表ではなく実装側と整合する。

文献全体では、frames（Rainbow 32K）、agent steps（Dopamine 8000）、learner gradient steps（Ape-X 2,500 training batches、R2D2 2,500 learner steps、BTR 500 gradient steps）が混在している。BTR は付録に frames / steps / transitions の定義節を設けており、ハイパーパラメータ表で両軸を併記している。

```
Replace Target Network Frequency (C) | 500 Gradient Steps (32K Environment Steps)

target network 置換頻度 (C) | 500 gradient step（32K 環境ステップ）
```

[Clark BTR (2025/05), Appendix D.2 ハイパーパラメータ表]

## 2. 主要アルゴリズムの設定値と選定根拠

**「実験で選ばれた」と明記しているのは少数で、大半は先行研究からの継承と明記されている。**

| アルゴリズム | 方式 | 値 | 単位 | 選定根拠 |
|---|---|---|---|---|
| Nature DQN | hard | C = 10,000 | parameter updates（論文）/ agent steps（実装） | 実験（5 ゲームでの informal search、系統的グリッドサーチは非実施と明記） |
| Double DQN | hard | 10,000 | agent steps | 継承 |
| Double DQN (tuned) | hard | 30,000 | frames / steps 表記が論文内で不整合 | 実験（過大評価をさらに減らす目的） |
| Dueling / PER / IQN / QR-DQN | hard | 論文に数値なし | — | 継承 |
| Rainbow | hard | 32K | frames（= 8,000 agent steps = 2,000 gradient steps） | 継承＋限定的な手動調整 |
| **DDPG** | soft | **τ = 0.001** | 学習更新ごと | **記載なし（定性的正当化のみ、スイープの記載なし）** |
| TD3 | soft | τ = 5·10⁻³ | gradient steps | 実験（束として評価） |
| SAC | soft | τ = 0.005 | gradient steps | **実験（τ 感度スイープを図と本文で明示）** |
| Ape-X | hard | 2,500 | training batches | 継承 |
| R2D2 | hard | 2,500 | learner steps | 継承 |
| R2D2+ (DMLab) | hard | 400 | learner steps | 実験（2,500 → 400 へ短縮） |
| Munchausen DQN | hard | C = 8,000 | environment steps | 継承（Dopamine の値） |
| BBF / SR-SPR | soft (EMA) | τ = 0.005、period = 1 | gradient steps | 存在のアブレーションあり、τ 値のスイープ記載なし |
| **BTR** | hard | **C = 500** | gradient steps | **実験（{250, 500, 1000} を試行、EMA も試して棄却）** |

**τ = 0.001 の出所である DDPG は、その値の選定経緯を書いていない。** 論文は soft update という方式の正当化は与えるが、0.001 という値についてはスイープもアブレーションも記載がなく、本調査では根拠を特定できなかった（本調査では「出典未確認」として記録した）。

実装側にはさらに差分がある。CleanRL は 1,000（論文の 1/10、差分を docs で明記）、Stable-Baselines3 の既定は 10,000 だが単位が environment steps、OpenAI Spinning Up は `polyak = 0.995` という補数規約で τ 換算 0.005（DDPG 論文の 0.001 と 5 倍差）。

## 3. 更新速度と収束・バイアスの理論

**target network 付き TD の半勾配は「TD(0) 項 + online を target へ引き寄せる正則化項」に分解でき、その正則化の強さは更新周期でしか制御できない。**

```
Therefore, TNs are inflexible in the sense that the weight of the regularizer cannot be controlled independently from the agent's effective horizon, and can only be controlled through the target update period T.

したがって TN は、正則化項の重みをエージェントの実効ホライズンから独立に制御できず、target 更新周期 T を通してしか制御できないという意味で柔軟性を欠く。
```

[Piché FR (2023/09), Section 3.1]

**この直後の一文について、2 つの調査経路が矛盾する引用を返した。** 一方は "the update period T can be **effective** at controlling the strength of the regularization"、他方は "can be **ineffective** at controlling..." である。論文全体の主張（TN の柔軟性の欠如を指摘し明示的正則化 FR を提案する）とは "ineffective" のほうが整合するが、**本調査では原文を確定できなかった。この一文は引用しない。**

同論文は target 更新周期の掃引としては本調査で確認した中で最も広く、Atari 6 ゲーム × 5 seeds・約 30,000 GPU 時間で T ∈ {500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000} を試し、**TN 側の最良は T = 5,000 だったと報告している**（本調査で確認）。既定値として広く使われる 1,000 / 8,000 は、この掃引では最良ではない。

**更新周期の両側（速すぎ／遅すぎ）を定量化した解析が存在する。** Fellows らは周期 k を含む条件関数と誤差再帰を与え、k を大きくすると縮小係数は下がるが残差分散の下限が上がることを示している。同論文は、周期 k が TD（k=1）と fitted policy evaluation（k→∞）を補間するものだと解釈している（Fellows らの条件関数 `C(α_l,k)` と誤差再帰）。

**定数スケジュールが理論的に準最適であることを証明した研究がある。**

```
We prove that constant target update schedules are suboptimal, incurring a logarithmic overhead in sample complexity that is entirely avoidable with adaptive schedules. Our analysis shows that the optimal target update frequency increases geometrically over the course of the learning process.

我々は、定数の target 更新スケジュールが最適でなく、サンプル複雑度において対数的なオーバーヘッドを招くこと、そしてそれが適応的スケジュールによって完全に回避可能であることを証明する。我々の解析は、最適な target 更新頻度が学習過程を通じて幾何級数的に増加することを示す。
```

[Weissmann TUF (2026/02), Abstract]

その直観は、学習の段階によってベルマン作用素の縮小性の働きが変わる、というものである。

```
(i) In early Bellman iterations, when current value estimates are far from the optimal fixed point Q*, the contractivity of T* moves estimates in the correct direction, even for large approximation errors of T*.
(ii) As learning progresses and the current estimates Q approach the optimal Q*, the margin for approximation error diminishes. Continued convergence requires increasingly accurate approximations of T* to avoid domination of approximation errors.

(i) 初期のベルマン反復では、現在の価値推定が最適な不動点 Q* から遠いため、T* の縮小性は、T* の近似誤差が大きくても推定を正しい方向へ動かす。
(ii) 学習が進み現在の推定 Q が最適な Q* に近づくにつれ、近似誤差に許される余裕は減少する。収束を継続するには、近似誤差の支配を避けるために T* のますます正確な近似が必要となる。
```

[Weissmann TUF (2026/02), Section 1]

**soft と hard を等価とする結果は見つからなかった。** 両者を同一枠組みで扱った収束解析は、それぞれ別の条件（「十分大きい周期 m すべて」対「十分小さいゲイン β すべて」）を証明しており、大きい m の極限で projected Q-value iteration に帰着する（Lee による linear Q-learning の JSR 解析）。

## 4. 自動・適応調整の手法

**存在する。ただし小さく孤立した系統で、調整信号に合意はない。**

本調査で確認できた 7 件の調整信号は次のとおりである。

| 手法 | 調整信号 |
|---|---|
| t-soft update | main / target パラメータ差の二乗平均 |
| CAT-soft update | t-soft のハイパーパラメータ自体を適応化 |
| Sun ら | モーメント推定 |
| Kim | 相互情報量と報酬 |
| TDU-DQN | エピソード報酬 |
| Badran & Rezghi | エージェントの直近挙動 |
| Park ら | 勾配追従（周期そのものを廃する） |

この系統で唯一二桁の被引用があるのは t-soft update（Neural Networks 2021、被引用 71）である。評価環境は PyBullet 4 タスク・Mountain Car・CartPole どまりで、**Atari 規模で優位を示した例は本調査では見つけられなかった。**

**イベント駆動型の提案がある。** Weissmann らの Appendix D が、全状態行動対にわたる平均 TD 誤差が閾値 ε_n = n⁻² を下回った時点で同期する ATQL を提案している。ただし著者ら自身が今後の展望として位置づけており、表形式 GridWorld の予備実験のみである。

**メタ勾配 RL と PBT の本流には不在である。** メタ勾配の原論文は target network を「リターンに表現しうる他の設計選択肢」として名指ししながら適応対象に含めておらず、STAC は包含基準を「損失関数を微分可能にパラメータ化していること」と明示している。PBT 系（PBT 原論文、PB2、PB2-Mix、FIRE PBT、SEARL）ではいずれも探索空間に含まれていない。AutoRL サーベイ（JAIR 2022）には target network 関連語が一度も現れない（本調査で当該サーベイの全文抽出により確認）。

転機は 2024 年で、ARLBench と HPO-RL-Bench が初めて探索空間に導入している。

## 5. 調整に使われる観測指標

**target 更新速度そのものを直接評価する標準指標は確立されていない。** 実際に測られているのは、過大評価バイアス、TD 誤差、target-online 乖離、action gap、policy churn、可塑性指標である。

**target と online の乖離を直接測る指標は存在する。** ただしパラメータ空間の L2 距離ではなく、方策空間・価値空間・関数空間で定式化されている。

```
In agents that use a target network (inducing π) that is an older copy of the online network (inducing π′), it is easy to measure W̄(π,π′) by comparing their arg max actions

target network（π を誘導する）が online network（π′ を誘導する）の古いコピーであるようなエージェントでは、両者の arg max 行動を比較することで W̄(π,π′) を容易に測ることができる
```

[Schaul PolicyChurn (2022/06), Section 1.2 Quantifying the phenomenon]

Piché らの関数空間版では、乖離が明示的な損失項 (κ/2)‖Q_w − Q_w̄‖²_D として書き下されている。

**これらの指標が「スコアの予測子」として妥当かの検証は存在し、結論は否定寄りである。** Lyle らは可塑性の喪失が単一の測定量に帰着できないとし、Nauman らは過大評価が良い予測子になるのは可塑性が緩和された条件下のみだとする。可塑性喪失のサーベイは、dormant neuron が駆動因か症状かは不明であり、effective rank と性能の関連は単純ではないとしている。

**action gap と policy churn は結び付けられているが、action gap と target 更新速度を直接結ぶ研究は本調査では見つからなかった。** Schaul らは小さい action gap を churn の要因として仮説化し、Advantage Learning による churn 減少を実測している。

## 6. replay ratio との連動

**業界の慣行は「gradient step 軸で固定」である。一次情報が複数ある。**

data-efficient Rainbow は replay period を 4 → 1 に変えた（replay ratio 0.25 → 1）が、target 更新周期を両変種で共通の 2,000 updates に据え置き、単位を脚注で明示している。

```
The target network update period depends on the number of updates (not frames). This means that this update is more frequent in the data-efficient variant, in terms of frames.

target network の更新周期は（フレーム数ではなく）更新回数に依存する。これは、フレーム換算ではこの更新が data-efficient 版においてより頻繁になることを意味する。
```

[vanHasselt DER (2019/06), Table 2 の脚注]

Scaled-QL はバッチサイズを 16 倍にしても gradient step 単位の周期を据え置いている。

```
Since we utilize a larger batch size, that is 16 times larger than the standard batch size of 32 on Atari, we scale up the learning rate from 5e-05 to 0.0002, but keep the target network update period fixed to the same value of 1 target update per 2000 gradient steps as with single-task Atari.

Atari の標準バッチサイズ 32 の 16 倍という大きなバッチサイズを用いるため、学習率を 5e-05 から 0.0002 へ引き上げるが、target network の更新周期は single-task Atari と同じ 2000 gradient step あたり 1 回の target 更新に固定したままとする。
```

[Kumar ScaledQL (2022/11), Appendix B.3]

**一方、replay ratio に応じてスケールする例もある。** ReDo は replay ratio に応じて target 更新周期を変えており、値は (0.25, 8000)、(0.5, 4000)、(1, 2000)、(2, 1000) である。ただし表の単位が明記されておらず、gradient step 軸固定と読めるかは本調査では断定を避けた。

**BBF は τ をアニールしていない。** 公式設定ファイルは `target_update_tau = 0.005` 固定・`target_update_period = 1`（毎 gradient step EMA）であり、コードに τ annealing 機構は存在するが `max_target_update_tau = None` で無効化されている。BBF がアニールしているのは n-step（10 → 3）と割引率（0.97 → 0.997）のみである（公式リポジトリの `BBF.gin` / `SR_SPR.gin` で確認）。

**連動則を定式化した論文は本調査では見つからなかった。** BTR は「target 更新頻度はバッチサイズと replay ratio と密接に絡み合う」と述べ、Hussing らは「速すぎれば発散、遅すぎれば学習遅延」と述べつつ未解決課題と明記している。

## 7. 批判と留保

**target network は万能ではなく、発散を招きうるという反例が複数ある。**

Piché らの Theorem 3.1 は、target network 付き TD(0) が、素の TD(0) が収束する条件下で発散しうることを示している。理由は、暗黙の正則化項に対応する行列が負の固有値を持つ場合に、それが正則化として振る舞わないことにある。Chen らは target network 単独では証明可能に不十分だとし、van Hasselt らは Tsitsiklis–Van Roy の反例上で target network が発散を遅らせるだけで防がないと報告している。

**「固定でよい」側の実測も強い。** Fu らは τ を 9 点スイープし、1 桁変えてもデータ効率の劣化は 19%（バッチサイズは 52%）に留まるとして既定固定を選択している。

**Rainbow については、既定値が「先行研究からの惰性」だと明言する実測がある。**

```
Also, to our surprise, in the case of no-resetting (which is indicated in black and corresponds to the standard Rainbow agent), notice that performance is basically flat as a function of K. Even taken to the extreme with K = 1, we observe no performance degradation in Rainbow. This means that freezing the target network for 8000 steps, while harmless, provides no meaningful improvement to the performance of Rainbow. It is not clear to us, then, why the choice of K = 8000 was made in the original Rainbow paper. One possible explanation is that this choice is just a legacy from the previous papers. As we show later, in DQN it is important that a large value of K is chosen, otherwise there will be extreme performance degradation, so it is possible that an assumption about the need for a large K in Rainbow was made, and the validity of this assumption was not tested thoroughly.

また驚いたことに、リセットなしの場合（黒で示され標準 Rainbow エージェントに対応する）、性能は K の関数として基本的に平坦であることに注目されたい。K = 1 という極端な設定でさえ、Rainbow に性能劣化は観測されない。これは、target network を 8000 ステップ凍結することが、無害ではあるものの Rainbow の性能に意味のある改善を与えていないことを意味する。では、なぜ元の Rainbow 論文で K = 8000 という選択がなされたのかは我々には明らかでない。ひとつの説明は、この選択が単に先行論文からの惰性であるというものである。後に示すように、DQN では大きい K を選ぶことが重要であり、そうしないと極端な性能劣化が生じる。したがって Rainbow でも大きい K が必要だという仮定がなされ、その仮定の妥当性が十分に検証されなかった可能性がある。
```

[Asadi ResetOptimizer (2023/06), Section 4.1]

**同じ著者らが Rainbow で hard / Polyak 双方をチューニングし、周期の調整が性能にほとんど影響しないことを追試している。** その際 Polyak 側では τ = 0.005 が最良だったと報告している（Asadi らの追試）。

一方で Hernandez-Garcia と Sutton は、この感度がアルゴリズムのブートストラップ依存度によって大きく変わることを統計的に示している。

```
the performance of Sarsa and Q-learning was more robust to the effect of the target network update frequency than the performance of Tree Backup, Q(σ), and Retrace in this particular task.

このタスクにおいて、Sarsa と Q-learning の性能は、Tree Backup、Q(σ)、Retrace の性能に比べて target network 更新頻度の影響に対してより頑健であった。
```

[HernandezGarcia DQNTarget (2019/01), Abstract]

**感度の有無はアルゴリズム依存であり、一般化された結論は存在しない。** 「target network からより多くの情報を引き出すアルゴリズムほどこのハイパーパラメータに敏感」という仮説が同論文の結論で述べられている。

**重み平均の汎化効果に関する文献は、ほぼ教師あり・自己教師あり学習のものである。** RL の target network と明示的に橋渡ししている一次情報は 1 件のみで、「Bellman 更新を統べるネットワークの安定化器」を「より広い極小 → より良い汎化」と並置して列挙している（Busbridge ら）。**EMA target が汎化上の利得を持つという RL 固有の定量的検証は、本調査では見つけられなかった。**

## 8. 総合評価

一次情報から確定した点は次の 5 つである。

**第一に、単位が文献間で統一されていない。** frames / agent steps / parameter updates / gradient steps が混在し、原典 DQN では論文の表と公式実装が 4 倍食い違う。値を比較するには毎回単位を確認する必要がある。

**第二に、選定根拠が明示されている値は少数である。** 明確に実験で選ばれたのは SAC の τ、BTR の C、Double DQN tuned、R2D2+ に限られる。DDPG の τ = 0.001 は、方式の正当化はあるが値の選定経緯が論文に見当たらない。

**第三に、定数スケジュールは理論的に準最適だと証明されている。** ただしこの結果は 2026 年 2 月のプレプリント 1 件であり、表形式 Q 学習が対象で、深層 RL での検証はない。

**第四に、適応調整の手法は存在するが主流に入っていない。** 7 件の系統はいずれも小規模環境での評価に留まり、調整信号にも合意がない。メタ勾配・PBT・AutoRL の主要文献では対象外である。

**第五に、replay ratio との連動については gradient step 軸固定が慣行であり、連動則を定式化した論文は見つからなかった。** 唯一の明示的なスケーリング例は ReDo だが、単位が明記されていない。

**したがって、target 更新速度を観測量ベースで自動調整することは、本調査の範囲では確立された手法が存在しない領域にあたる。** 理論側は「定数は準最適・頻度は上げていくべき」と示唆し、実測側は「感度は条件依存で、平坦な場合もある」と示している。両者は矛盾しないが、どちらも具体的な観測量と目標値の対応を与えていない。

## 9. 調査の限界

- **SR-SPR 本文 PDF が取得できなかった。** OpenReview のボット対策により、curl でも同様。公式ポスター・BBF 本文の記述・リポジトリの設定ファイルで代替し、逐語検証できなかった箇所は「出典未確認」と明示している。**なお本調査の完了後、SR-SPR 本文 PDF（ICLR 2023 版）が別途入手できたことを確認している。**
- **R2D2・SR-SPR は Wayback Machine 経由の PDF を使用**している。
- **被引用数の出典が調査角度によって異なる。** 理論解析の角度は Semantic Scholar API が全件レート制限（429）のため OpenAlex を使っており、OpenAlex はプレプリントと出版版のレコードを分割するため ML 系会議論文を大きく過小計上する。実設定値の角度では Semantic Scholar と OpenAlex で最大 3.1 倍の乖離を確認している（TD3: 2,372 対 7,376）。**本書の被引用数は桁感の目安に留まり、判断材料には使っていない。**
- **理論解析の角度で 9 件は HTML レンダリング経由のみで読んでおり、定理番号は未検証である。** 検証中に 1 件の誤り（Achiam らの線形化が Eq. 8 ではなく Eq. 10）が発見・訂正されており、HTML 由来の番号に同種の誤りが残る可能性がある。
- **適応調整の角度で 3 件（Sun ら、Kim、Jia ら）は出版社の認証・アクセス制限により本文を取得できなかった。** 抄録のみ確認しており、更新則の具体式は未検証である。
- **ReDo Table 2 の単位、Piché の被引用数、一部の著者所属が未確認**である。
- **数式引用の一部は PDF 組版からの書き起こしであり、文字単位で原文と一致しない**（該当箇所に注記した）。
- **Piché FR の §3.1 末尾の一文について、2 経路の調査が "effective" と "ineffective" という反対の引用を返した。** 原文を確定できなかったため本書では引用していない（§3）。
- **「event-triggered target network update」という語で target network 同期を扱う論文は本調査では見つからなかった。** この語で出てくる文献群（Euler–Lagrange 同期制御、integral RL、multi-agent tracking 等）は制御入力の発火や通信削減が対象で、target network とは別主題である。条件トリガ型として実在するのは Weissmann らの "accuracy-triggered" が唯一の直接的一致である。
- **検索エンジンの誤答を 2 件検出し、伝播させていない。** (a) ある論文の τ アニールという検索結果は、実際には expectile パラメータであって Polyak 係数ではなかった。(b) MDPI 論文の著者所属について検索結果と原文 PDF が食い違い、原文を採用した。
- **被引用数のレコード分裂**が複数確認されている（arXiv 版と会議版が別レコードになる）。本書では被引用数を判断材料に使っていない。

## 10. 出典リスト

本節は本書の本文で直接引用した出典を挙げる。本調査で確認した出典は延べ 135 件あり、本文が引用していないものはここに含めない。

[Mnih DQN, 2015/02] Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-level control through deep reinforcement learning." Nature 518(7540):529–533. 所属: Google DeepMind。https://www.nature.com/articles/nature14236

[vanHasselt DER, 2019/06] van Hasselt, H., Hessel, M., Aslanides, J. "When to use parametric models in reinforcement learning?" NeurIPS 2019 / arXiv:1906.05243. https://ar5iv.labs.arxiv.org/html/1906.05243

[Schaul PolicyChurn, 2022/06] Schaul, T., Barreto, A., Quan, J., Ostrovski, G. "The Phenomenon of Policy Churn." NeurIPS 2022 / arXiv:2206.00730. https://ar5iv.labs.arxiv.org/html/2206.00730

[Kumar ScaledQL, 2022/11] Kumar, A., Agarwal, R., Geng, X., Tucker, G., Levine, S. "Offline Q-Learning on Diverse Multi-Task Data Both Scales And Generalizes." ICLR 2023 / arXiv:2211.15144. https://ar5iv.labs.arxiv.org/html/2211.15144

[Fellows TargetNets, 2023/02] Fellows, M., Smith, M. J. A., Whiteson, S. "Why Target Networks Stabilise Temporal Difference Methods." ICML 2023 / arXiv:2302.12537. https://arxiv.org/abs/2302.12537

[Piché FR, 2023/09] Piché, A., Thomas, V., Pardinas, R., et al. "Bridging the Gap Between Target Networks and Functional Regularization." TMLR (09/2023) / arXiv:2106.02613. 所属: ServiceNow Research, Mila ほか。https://arxiv.org/pdf/2106.02613

[Clark BTR, 2025/05] Clark, T., Towers, M., Evers, C., Hare, J. "Beyond The Rainbow: High Performance Deep Reinforcement Learning On A Desktop PC." ICML 2025 / arXiv:2411.03820v2. 所属: University of Southampton。https://arxiv.org/html/2411.03820v2

[HernandezGarcia DQNTarget, 2019/01] Hernandez-Garcia, J. F., Sutton, R. S. "Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target." arXiv:1901.07510. 所属: University of Alberta, Department of Computing Science。https://ar5iv.labs.arxiv.org/html/1901.07510

[Asadi ResetOptimizer, 2023/06] Asadi, K., Fakoor, R., Sabach, S. "Resetting the Optimizer in Deep RL: An Empirical Study." NeurIPS 2023 / arXiv:2306.17833. https://arxiv.org/abs/2306.17833

[Weissmann TUF, 2026/02] Weissmann, S., Aach, T., Wille, B., Kassing, S., Döring, L. "The Role of Target Update Frequencies in Q-Learning." arXiv:2602.03911. 所属: University of Mannheim / University of Wuppertal。https://arxiv.org/html/2602.03911
