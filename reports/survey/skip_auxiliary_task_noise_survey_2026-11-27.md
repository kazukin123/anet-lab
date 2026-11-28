# Survey: 行動を続ける長さ（スキップ）の学習を、補助タスクやパラメータのノイズと組む事例

日付: 2026-11-27
範囲: 行動を続ける長さの価値を、行動の選択に使わず補助タスクとしてだけ学ぶ事例（多地平の価値の補助タスクなど近いものを含む）と、スキップ長の選択を ε-greedy ではなく NoisyNet などのパラメータのノイズで探索させる事例。TempoRL・FiGAR・UTE・Lazy-MDP・Tallec らの論文解説（reports/ の既存の報告）にあることは調べ直していない。

## 対象の問いと結論

対象の問い:

- TempoRL のような「行動を続ける長さの価値」を、行動の選択には使わず、表現の学習のための補助タスクとしてだけ学ぶ事例はあるか（多地平の価値の補助タスクなど、近いものを含む）。
- スキップ長（行動の繰り返しの長さ）の選択に、ε-greedy ではなく NoisyNet などのパラメータのノイズで探索させる事例はあるか。

結論:

- 問い 1: 長さの価値を補助タスクとしてだけ学ぶ事例は見つからなかった。TempoRL を引用する 45 本の抄録にも無かった（§1.1）。近いものは、割引率の違う 10 個の価値を補助として学び行動は 1 つの γ で選ぶ Multi-Rainbow（Fedus ら）で、n-step とはよく組み合わさり、PER とは 10 ゲーム中 4 ゲームで悪く組み合わさった（§1.2）。理論の側では、仕組みを足さなければ多数の割引率の予測は単一の課題より役立たないという解析がある（Lyle ら、§1.3）。長さの価値を全部の長さで同時に学ぶ PerDQN は、その価値で行動を選ぶ（§1.5）。
- 問い 2: パラメータのノイズでスキップ長を選ぶ事例は見つからなかった。近いものは、NoisyNet の Rainbow の上に、長さを zeta 分布から引く εz-greedy を重ねた例（長さは学ばない、§2.2）、NoisyNet で行動を選び繰り返すかを状態の新しさで決める SNAP（連続制御、§2.3）、繰り返すかを Q の Boltzmann 分布と entropy で決める TAAC（連続制御、§2.4）である。

調べた範囲と日付: 2026-11-27 に、論文の本文（arXiv・ar5iv・学会の版）と抄録、Semantic Scholar の TempoRL の被引用の一覧（45 本、抄録つき）を読んだ。検索語は §5 に挙げる。

見つからなかったこと: (1) スキップ長（行動を続ける長さ）の価値を、行動に使わず補助タスクとしてだけ学ぶ事例、(2) スキップ長の選択に NoisyNet などのパラメータのノイズを使う事例、(3) スキップの価値の Head にだけノイズを入れる・外す比較。

## 目次

1. [長さの価値を補助タスクとしてだけ学ぶ事例](#1-長さの価値を補助タスクとしてだけ学ぶ事例)
2. [スキップ長の選択をパラメータのノイズで探索する事例](#2-スキップ長の選択をパラメータのノイズで探索する事例)
3. [批判と留保](#3-批判と留保)
4. [総合評価](#4-総合評価)
5. [調査の限界](#5-調査の限界)
6. [出典の一覧](#6-出典の一覧)

## 1. 長さの価値を補助タスクとしてだけ学ぶ事例

### 1.1 探した範囲と結果

TempoRL を引用する論文（Semantic Scholar で 45 本）の題と抄録を読み、長さの価値を行動に使わず補助としてだけ学ぶものを探したが、無かった。繰り返しを扱うものは、どれも学んだ価値か別の規則で行動か長さを選んでいた（TAAC、UTE、PerDQN は学んだ価値、SNAP は状態の新しさ）[S2 TempoRL-citations (2026/11)]。既存の報告の TempoRL・FiGAR・UTE・Lazy-MDP の論文解説にも記述は無い。

### 1.2 多地平の価値の補助タスク（Fedus ら）

Fedus らは、割引率の違う複数の Q を共有の表現の上の軽い Head で同時に学ぶことを補助タスクにした。

```
often improves over a strong value-based RL agent, Rainbow

強い価値ベースの RL の Agent である Rainbow を、しばしば上回る
```

[Fedus MultiHorizon (2019/02), Abstract]

- 形: Head は γ ごとの線形の層で、Atari の設定は γ が 10 個、最大は 0.99 [Fedus MultiHorizon (2019/02), Section 6.1, Appendix C, Table 6]。Multi-Rainbow は固定の γ_action で行動を選ぶ [Fedus MultiHorizon (2019/02), Section 9]。
- 条件: Dopamine の Rainbow（sticky actions 0.25）、各 3 seed [Fedus MultiHorizon (2019/02), Section 9, Figure 10, Figure 11]。
- 結果: 双曲の割引で行動する Hyper-Rainbow と Multi-Rainbow に有意な差は無く（19 ゲーム）、著者は改善が多地平の補助タスクから来るとした [Fedus MultiHorizon (2019/02), Section 9, Figure 10]。C51 に足した Multi-C51 は 10 ゲーム中 9 ゲームで C51 を上回った [Fedus MultiHorizon (2019/02), Section 9, Figure 12]。
- 他の要素との組み合わせ: n-step とはよく組み合わさり、PER とは 10 ゲーム中 4 ゲーム（Pong、Venture、Gravitar、Zaxxon）で成績を下げた。著者は、全 Head の TD 誤差を平均して優先度にしたためと説明し、最大の γ で優先度を付ける案を予備の結果として付録 E に置く [Fedus MultiHorizon (2019/02), Section 9, Figure 12, Appendix E]。
- 表現を直接調べた実験は無い。

### 1.3 補助タスクとしての多数の割引率の理論（Lyle ら）

Lyle らは、TD の学習の動きから補助タスクが誘う表現を解析し、複数の割引率の価値の予測は、平均の割引率の単一の予測と同じ表現に向かうとした（§4.3、表 1）。

```
doesn't provide additional utility over the single task setting

単一の課題の設定に比べて、追加の役には立たない
```

[Lyle AuxDynamics (2021/02), Section 4.3]

これは追加の仕組み（課題ごとの初期化の部分空間など）が無い場合の結論で、著者はその仕組みを試していない [Lyle AuxDynamics (2021/02), Section 4.3]。Atari（Double DQN）の実験は、ランダムな累積量（random cumulants）とアンサンブルを補助にしたもので、Montezuma's Revenge での強さを挙げるが、本文に数値は無い。多数の割引率は Atari で試していない [Lyle AuxDynamics (2021/02), Section 5.2, Figure 4]。

### 1.4 行動の列に条件付けた予測を補助にする（Zheng ら）

Zheng らは、ランダムに選んだ行動の列に条件付けた多段の予測（ランダムな GVF）を補助タスクにすると、価値の予測・pixel control・CURL といった手作りの補助と同じくらいの表現が得られると報告した。RL の目的の勾配を表現に返さなくても、補助だけで学んだ表現が end-to-end の actor-critic を上回ったとする（Atari と DeepMind Lab）。抄録だけを読んだ [Zheng RandomGVF (2021/02), Abstract]。行動の繰り返しに限った予測ではない。

### 1.5 長さの価値を全部同時に学ぶ（PerDQN）

Sabbioni らは、どの長さの遷移からも全部の長さ（persistence）の行動価値を同時に更新する作用素を導き、DQN に載せた（PerDQN）[Sabbioni AllPersistence (2022/11), Section 5, Section 7]。長さの価値は行動と長さの選択に使い、探索は行動と長さの組に対する ε-greedy である [Sabbioni AllPersistence (2022/11), Section 5, Algorithm 2]。Atari の 6 ゲーム（各 5 Run）のうち 5 ゲームで DQN と TempoRL より学習曲線が速く、Seaquest では撃つ行動を続けるために DQN が上回った [Sabbioni AllPersistence (2022/11), Section 7, Figure 5]。補助タスクやノイズには触れていない。

### 1.6 地平を入力に取る価値（Γ-nets）

Sherstan らは、割引率を入力に取って任意の時間尺度の価値を 1 つの網で予測する Γ-nets を提案し、考察で補助タスクとして役立つかを調べることを次の課題に挙げた。補助タスクの実験は無く、Fedus らを引くだけである [Sherstan GammaNets (2019/11), Section 5]。

## 2. スキップ長の選択をパラメータのノイズで探索する事例

### 2.1 探した範囲と結果

TempoRL の被引用 45 本の抄録に、NoisyNet・パラメータのノイズで長さを選ぶものは無かった [S2 TempoRL-citations (2026/11)]。不確かさで長さを選ぶ UTE（アンサンブル）と、FiGAR（ε-greedy と出力分布からの抽選）は既存の報告にある。

### 2.2 NoisyNet の上の時間的に延ばした ε（εz-greedy）

Dabney らの εz-greedy は、探索の行動を、zeta 分布 z(n) ∝ n^(−μ)（既定は μ = 2）から引いた長さだけ続ける。長さは学ばない [Dabney EzGreedy (2020/06), Section 4.2]。Rainbow の実験では、εz-greedy を Rainbow の NoisyNet の方策の上に重ねた。行動は NoisyNet で、探索を始める時期と続ける長さは ε と zeta 分布で決まる形である [Dabney EzGreedy (2020/06), Appendix B.3]。Atari-57 の人間で正規化した得点の中央値は、R2D2（30B フレーム、3 seed）で 16.44 → 17.77、Rainbow（200M フレーム、5 seed）で 2.03 → 2.30 だった [Dabney EzGreedy (2020/06), Section 5.2, Table 1]。NoisyNet とパラメータのノイズは、Thompson sampling に関わる探索の方法として挙げるだけで、直接は比べていない [Dabney EzGreedy (2020/06), Section 3]。

### 2.3 NoisyNet で行動を選び、繰り返すかを新しさで決める（SNAP）

Hu らの SNAP は、直前の行動を繰り返す確率を、状態の擬似回数から決める（α / max(1, √Ñ(s))）。繰り返しのための価値や方策は学ばず、長さは毎 step の「繰り返すか選び直すか」で決まる [Hu SNAP (2024/09), Section 5.1, Section 5.2, Eq. 8]。ε-greedy・entropy 正則化・NoisyNet の 3 つの探索に重ね、NoisyNet の場合も標本効率が上がったとするが、数値は無い。画像入力の DeepMind Control（DrQ-v2、5 seed）で、Atari は無い [Hu SNAP (2024/09), Section 6.3, Figure 7]。

### 2.4 繰り返すかを Q の Boltzmann 分布で決める（TAAC）

Yu らの TAAC は、直前の行動を繰り返すか新しい行動にするかの 2 択を、Q から作る Boltzmann 形の分布 β* ∝ exp(Q/α) で選び、温度は entropy の項で自動調整する。ε は使わない [Yu TAAC (2021/04), Section 4.3, Eq. 10, Eq. 13]。著者は前の行動に沿った持続する探索が生じるとした（14 の連続制御の課題、3 seed）[Yu TAAC (2021/04), Section 5.1, Section 5.6.1]。

### 2.5 NoisyNet のノイズを引き直す単位

NoisyNet の原典は、重みの 1 回の変化が複数の step にわたって一貫した、状態に依存した方策の変化を生むことを利点に挙げる [Fortunato NoisyNet (2017/06), Section 1]。DQN では行動ごとに 1 回の最適化をするので、ノイズは行動の前ごとに引き直し、target の網は別のノイズを使う [Fortunato NoisyNet (2017/06), Section 3.1]。行動の繰り返しや時間的に延ばした行動には触れていない。

## 3. 批判と留保

- Fedus らの論文は arXiv の版で、本調査で読んだ頁から掲載先を確かめられなかった（RFD0104 §2.3 は ICLR 2020 としている）。図 11 の全ゲームの比較は、改善の数を本文で数えていない。
- Lyle らの多数の割引率の結論は、線形・アンサンブルの解析の前提の下のもので、Fedus らの Atari の結果と同じ条件ではない。
- SNAP と TAAC は連続制御で、離散の行動とスキップ長の Q を持つ DQN 系の形ではない。
- εz-greedy の長さは学ばず、TempoRL のスキップ長の価値は持たない。

## 4. 総合評価

事実として言えるのは、長さの価値を補助タスクとしてだけ学ぶ事例も、パラメータのノイズで長さを選ぶ事例も見つからず、近いものは、行動の選択に使わない多地平の価値の補助（Fedus ら）と、NoisyNet で行動を選びつつ長さは別の仕組みで決める形（εz-greedy、SNAP）だったことである。

以下はこの報告の読み取り（推論）である。Fedus らの補助は Head を γ ごとに分けて行動に使わない点で、TempoRL の skip Head を補助としてだけ学ぶ形に構造が近いが、予測先は長さではなく割引率である。PER と悪く組み合わさった理由（全 Head の TD 誤差を平均した優先度）は、補助の Head を持つ構成で PER を使うときの点検の目安になる。文献には確かめた例が無い。

## 5. 調査の限界

- 使った検索語（一般の語だけ）: 「action repetition value auxiliary task representation learning deep RL "auxiliary" "action repeat" predict duration」「temporally extended actions repetition length exploration NoisyNet parameter noise skip policy」「"NoisyNet" OR "noisy networks" "action repetition" OR "frame skip" OR "TempoRL" learned repetition DQN exploration」「TempoRL "learning when to act" follow-up skip policy extension 2023 2024 2025 action repetition value」「Gamma-nets generalizing value estimation over timescale auxiliary task Sherstan」「Fedus Gelada Bengio Bellemare Larochelle "Hyperbolic Discounting and Learning over Multiple Horizons" openreview」。あわせて、Semantic Scholar の API で TempoRL（arXiv 2106.05262）の被引用の一覧を取った。
- 被引用の一覧のうち数本（Timing as an Action、V-Former など）は抄録が無く、題だけで判断した。Zheng らは抄録だけを読んだ。
- εz-greedy の OpenReview の頁は自動アクセスの確認の画面で開けず、掲載先を確かめていない。
- 被引用数は Semantic Scholar の API で 2026-11-27 に取得した。

## 6. 出典の一覧

[Fedus MultiHorizon, 2019/02] William Fedus, Carles Gelada, Yoshua Bengio, Marc G. Bellemare, Hugo Larochelle（Google Brain・Mila）. "Hyperbolic Discounting and Learning over Multiple Horizons." arXiv:1902.06865. https://ar5iv.labs.arxiv.org/html/1902.06865 — 被引用 130。
[Lyle AuxDynamics, 2021/02] Clare Lyle, Mark Rowland, Georg Ostrovski, Will Dabney. "On the Effect of Auxiliary Tasks on Representation Dynamics." AISTATS 2021（PMLR 130）. https://proceedings.mlr.press/v130/lyle21a.html（本文は https://ar5iv.labs.arxiv.org/html/2102.13089）— 被引用 84。
[Zheng RandomGVF, 2021/02] Zeyu Zheng, Vivek Veeriah, Risto Vuorio, Richard Lewis, Satinder Singh. "Learning State Representations from Random Deep Action-conditional Predictions." NeurIPS 2021. https://arxiv.org/abs/2102.04897 — 被引用 6。
[Sabbioni AllPersistence, 2022/11] Luca Sabbioni, Luca Al Daire, Lorenzo Bisi, Alberto Maria Metelli, Marcello Restelli（Politecnico di Milano・ML cube）. "Simultaneously Updating All Persistence Values in Reinforcement Learning." AAAI 2023. https://ar5iv.labs.arxiv.org/html/2211.11620 — 被引用 6。
[Sherstan GammaNets, 2019/11] Craig Sherstan, Shibhansh Dohare, James MacGlashan, Johannes Günther, Patrick M. Pilarski（University of Alberta・Cogitai）. "Gamma-Nets: Generalizing Value Estimation over Timescale." AAAI 2020（Semantic Scholar の記載）. https://ar5iv.labs.arxiv.org/html/1911.07794 — 被引用 15。
[Dabney EzGreedy, 2020/06] Will Dabney, Georg Ostrovski, André Barreto（DeepMind）. "Temporally-Extended ε-Greedy Exploration." arXiv:2006.01782. https://arxiv.org/abs/2006.01782（本文は https://ar5iv.labs.arxiv.org/html/2006.01782）— 被引用 40。
[Hu SNAP, 2024/09] Jianshu Hu, Paul Weng, Yutong Ban（上海交通大学 UM-SJTU Joint Institute・Duke Kunshan University）. "State-Novelty Guided Action Persistence in Deep Reinforcement Learning." arXiv:2409.05433（Semantic Scholar は Machine Learning 誌）. https://arxiv.org/html/2409.05433 — 被引用 4。
[Yu TAAC, 2021/04] Haonan Yu, Wei Xu, Haichao Zhang. "TAAC: Temporally Abstract Actor-Critic for Continuous Control." NeurIPS 2021. https://arxiv.org/abs/2104.06521（本文は https://ar5iv.labs.arxiv.org/html/2104.06521）— 被引用 32。
[Fortunato NoisyNet, 2017/06] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot ほか（DeepMind）. "Noisy Networks for Exploration." ICLR 2018（Semantic Scholar の記載）. https://ar5iv.labs.arxiv.org/html/1706.10295 — 被引用 1058。
[S2 TempoRL-citations, 2026/11] Semantic Scholar. "Citations of: TempoRL: Learning When to Act（arXiv 2106.05262）." Semantic Scholar Graph API. https://api.semanticscholar.org/graph/v1/paper/arXiv:2106.05262/citations?fields=title,year,venue,abstract,externalIds&limit=100（2026-11-27 取得、45 本）。
