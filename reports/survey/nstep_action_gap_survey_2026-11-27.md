# Survey: DQN 系の n-step と行動間の価値の差（action gap）・まれな報酬の伝わる速さ

日付: 2026-11-27
範囲: DQN 系の制御で n-step の n を大きくしたときの action gap と報酬の伝播（理論と実験）、および action gap が小さいときの対処と n-step の関係。既存の報告（reward_scale_discount_survey_2026-09-23.md の §5.5・§5.6、bellemare2016_increasing_action_gap_paper_2026-10-03.md、tallec2019_time_discretization_paper_2026-10-03.md ほか）にあることは調べ直していない。

## 対象の問いと結論

対象の問い:

- DQN 系の制御で、n-step の n を大きくしたとき、行動間の価値の差（action gap。最善の行動と次善の行動の Q の差）や、まれな報酬を手前の状態へ伝える速さがどう変わるかを、扱った文献はあるか。理論（割引率・時間の刻みと action gap の関係、n-step の偏りと分散）と、実験の両方で。
- action gap が小さいと Greedy の方策が不安定になる（同じ所で止まる、行動が入れ替わる）ことへの対処（Advantage Learning、時間の刻みに頑健な手法など）は、n-step とどう関係づけられているか。

結論:

- 問い 1: n を変えて action gap を測った DQN 系の制御の文献は見つからなかった。報酬の伝播については、Rainbow が多段の目標の利点を「新しい報酬を手前の状態へ速く伝える」こととし、外すと 57 ゲーム中 53 ゲームで下がった（§3.1）。Mountain Car の DQN では n = 20 が序盤と全体で最もよかった（§3.2）。理論は、補正なしの n-step が分散を下げる代わりに固定点が偏ること（Rowland ら）と、n とともに分散が増えること（Daley ら）を扱うが、action gap との関係は扱っていない（§2）。
- 問い 2: action gap と Greedy の不安定さを結ぶ文献はある（Farahmand の損失の上界、Schaul らの policy churn）。対処の側では、Munchausen の論文が M-DQN を n-step なしで、M-IQN を 3-step で使い、1-step の M-IQN も Rainbow を上回るとしたのが n-step に触れた唯一の記述だった（§4.2）。Advantage Learning・DAU と n-step を組み合わせて action gap を比べた実験は見つからなかった。

調べた範囲と日付: 2026-11-27 に、論文の本文（arXiv・ar5iv・学会の版）と抄録、Semantic Scholar の書誌を読んだ。検索語は §7 に挙げる。

見つからなかったこと: (1) DQN 系の制御で n ごとに action gap を測った実験、(2) 時間の刻み（frame skip）と n-step と action gap を結ぶ理論、(3) Advantage Learning・DAU・Munchausen と n-step の組み合わせで action gap や Greedy の止まりを比べた実験。

## 目次

1. [理論: 割引率・時間の刻みと action gap](#1-理論-割引率時間の刻みと-action-gap)
2. [理論: n-step の偏りと分散](#2-理論-n-step-の偏りと分散)
3. [実験: DQN 系で n を大きくしたとき](#3-実験-dqn-系で-n-を大きくしたとき)
4. [対処と n-step の関係](#4-対処と-n-step-の関係)
5. [批判と留保](#5-批判と留保)
6. [総合評価](#6-総合評価)
7. [調査の限界](#7-調査の限界)
8. [出典の一覧](#8-出典の一覧)

## 1. 理論: 割引率・時間の刻みと action gap

### 1.1 既存の報告にあること

時間の刻み δt を小さくすると Q が V に潰れること（Tallec ら。tallec2019_time_discretization_paper_2026-10-03.md）、長い地平で action gap が崩れうること（Lehnert ら。reward_scale_discount_survey_2026-09-23.md の §4.2）、action gap を広げる作用素（Bellemare ら。bellemare2016_increasing_action_gap_paper_2026-10-03.md）は既存の報告にある。Tallec らの解説は、多段の目標値との組み合わせを論文が扱っていないと書いている（tallec2019_time_discretization_paper_2026-10-03.md、論文が扱っていない点）。ここでは調べ直していない。

### 1.2 action gap と Greedy の方策の損失

Farahmand は、推定した Q から作る Greedy の方策の損失が、action gap の規則性 ζ ≥ 0 によって O(‖Q̂ − Q*‖^(1+ζ)) で抑えられると示した。ζ > 0 なら、Q の誤差の大きさから予想するより損失が小さい。抄録だけを読んだ（本文は読んでいない）[Farahmand ActionGap (2011/12), Abstract]。

### 1.3 小さい action gap と方策の入れ替わり（policy churn）

Schaul らは、DQN 系の Agent で 1 回の更新ごとに Greedy の行動が変わる状態の割合（policy churn）を測り、その要因の 1 つとして action gap の小ささを仮説にした。

```
small approximation error can suffice for sub-optimal actions to overtake optimal ones

小さな近似誤差でも、次善の行動が最善の行動を追い越すのに足りうる
```

[Schaul PolicyChurn (2022/06), Section 3.2]

Advantage Learning（AL）は churn を大きく減らし、action gap が大きいことと対応した（図 3 右は、ゲーム・seed・学習の時期ごとの点を重ねた図。詳しい図 17 は 5 ゲーム）[Schaul PolicyChurn (2022/06), Section 3.2, Figure 3, Figure 17]。ε = 0 の評価でも AL のほうが成績が高かった [Schaul PolicyChurn (2022/06), Figure 18]。

### 1.4 行動の繰り返しと地平

Kalyanakrishnan らは、同じ行動を d step 続ける行動の繰り返しの損失を、課題ごとの「慣性の代価」で上から抑え、その損失は実効の地平が短くなることによる学習の得で相殺されうると論じた。抄録は n-step と action gap に触れていない [Kalyanakrishnan FrameSkip (2021/02), Abstract]。

## 2. 理論: n-step の偏りと分散

Kearns と Singh の誤差上界（大きい k は収束が速いが漸近誤差が大きい）は既存の報告の §5.6 にある。

### 2.1 補正なしの n-step の固定点と縮小の速さ

Rowland らは、重点重みを掛けない補正なしの n-step（Rainbow・R2D2 が使う形）を、分散が小さい代わりに固定点が偏る更新として整理した。

```
incurring some amount of fixed-point bias for reduced variance

分散を減らす代わりに、ある程度の固定点の偏りを負う
```

[Rowland AdaptiveTradeoffs (2019/10), Section 2]

固定点は目標の方策の価値ではなく、1 step を目標の方策 π、続く n−1 step を行動の方策 μ で進むことを繰り返す非定常な方策の価値になる [Rowland AdaptiveTradeoffs (2019/10), Section 2, Proposition B.1]。20 状態の鎖の MDP（行動の方策は一様、目標の方策は最適）では、環境の step が少ないうちは n の大きい補正なしの更新が最もよく、step が増えると固定点の偏りが効いて、偏りの小さい手法に追い越された [Rowland AdaptiveTradeoffs (2019/10), Section 4, Figure 3]。Atari-57 の DQN 系の比較では、n-step の対照は n = 5 が最もよかった（C-trace は n = 10、3 seed）[Rowland AdaptiveTradeoffs (2019/10), Section 4.2, Appendix E.2]。

### 2.2 補正なしの多段の更新が効く理由の別の説明

Schaul らは、補正なしの多段の更新が意外によく効くことを churn で説明した。Greedy の方策が速く変わるので、補正のある方法では数回の更新の遅れでも軌跡の打ち切りが大きくなり、補正を省く損より得が大きくなりうる、という議論で、実験ではない [Schaul PolicyChurn (2022/06), Section 4.3]。Kozuno らは、軌跡を切らない Peng's Q(λ) が、行動の方策が Greedy の方策をゆっくり追うなら最適な方策に収束することを示した [Kozuno PengQ (2021/02), Abstract]。

### 2.3 n とともに増える分散

Daley らは、n-step の収益の分散が n とともに増え、TD 誤差に相関が無ければ n に線形、完全に相関すれば 2 乗で増える形を示した [Daley AveragingNstep (2024/02), Section 3.1, Proposition 3.3]。同じ縮小率の複合の収益（λ 収益など）は、n-step より分散が大きくならない [Daley AveragingNstep (2024/02), Section 3.3, Theorem 3.7]。MinAtar の 5 ゲーム（32 試行）の多段の DQN では、n = 5 の比較で 5 ゲーム中 3 ゲーム、n = 3 で 1 ゲームで複合の収益が n-step を上回った [Daley AveragingNstep (2024/02), Section 5]。action gap には触れていない。

## 3. 実験: DQN 系で n を大きくしたとき

### 3.1 Rainbow

Rainbow は、多段の目標の利点を報酬を速く伝えることとした。

```
propagate newly observed rewards faster to earlier visited states

新しく観測した報酬を、先に訪れた状態へより速く伝える
```

[Hessel Rainbow (2017/10), Introduction]

n は 3 で、調整では n = 1・3・5 を比べ、3 と 5 は序盤がよく、最後は 3 が最もよかった [Hessel Rainbow (2017/10), Hyper-parameter tuning, Table 1]。多段の目標を外すと序盤も最終の成績も下がり、57 ゲーム中 53 ゲームで完全な Agent を下回った（約 200M フレーム。seed の数は論文に無い）[Hessel Rainbow (2017/10), Ablation studies, Figure 3, Figure 4]。action gap は測っていない。

### 3.2 Mountain Car の DQN

Hernandez-Garcia と Sutton は、DQN に似た構成（隠れ層 1 層、γ = 1、各 100 回の独立な Run）で n ∈ {1, 3, 5, 10, 20} を比べた。

```
Increasing the value of n improved the performance of the algorithm.

n の値を大きくすると、アルゴリズムの成績が上がった。
```

[HernandezGarcia MultiStepDQN (2019/01), Section 4.2]

n = 20 が序盤と全体の成績で最もよく、最終の成績は n ≥ 3 で同じくらいで、成績を損なうほど大きい n は見つからなかった [HernandezGarcia MultiStepDQN (2019/01), Section 4.2, Table 3, Table 5]。著者は効果を n-step の偏りと分散の釣り合いで説明し、報酬の伝わる速さは測っていない。action gap にも触れていない。

### 3.3 疎な報酬の迷路

Episodic Backward Update の論文の 10×10 の MNIST の迷路（ゴール +1000、壁 −1、壁の密度ごとに 50 の迷路、100,000 step のあと）では、最短路に対する経路の長さの平均が、壁 20% で 1-step DQN 14.40・n-step DQN（n はエピソードの長さ）3.26、壁 50% で 22.36・11.32 だった（表 1）[Lee EBU (2018/05), Section 5.1, Table 1]。著者は、一様に引く 1-step の更新が疎で遅れた報酬で非効率だとする [Lee EBU (2018/05), Section 1, Section 5.1]。Atari の比較（10M フレーム）は n-step を含まない [Lee EBU (2018/05), Section 5.2]。

### 3.4 探索のボーナスと多段の更新

Ostrovski らは、Monte Carlo 収益を混ぜた更新（MMC）を、一過性のボーナスを速く伝える多段の方法と位置づけた。

```
a simple multi-step method, helping propagate reward information faster

報酬の情報を速く伝えるのを助ける、単純な多段の方法
```

[Ostrovski PixelCNN (2017/03), Section 6]

MMC は Bank Heist と Time Pilot では改善の大半を説明したが、Ms. Pac-Man と Breakout では成績を下げた（図 11）[Ostrovski PixelCNN (2017/03), Section 6, Figure 11]。著者は、Monte Carlo 収益は方策に依存して分散を増やし、off-policy の学習で最適な方策への収束を妨げうるとする [Ostrovski PixelCNN (2017/03), Section 6]。Retrace(λ) を使う Reactor では、方策が外れた軌跡を慎重に切る性質がボーナスの効きを抑えたと推測している [Ostrovski PixelCNN (2017/03), Section 4.2]。最難関の疎な報酬のゲームで MMC とボーナスの組み合わせだけが進んだことは既存の報告にある。

### 3.5 action gap を n ごとに測った報告

見つからなかった。Schaul らの 2 つの Agent は、1-step の Double DQN の churn が更新ごとに約 9%、5-step の R2D2 が約 6% だったが、割引率（0.99 と 0.997）ほかの設定も違い、n を変えた比較ではない [Schaul PolicyChurn (2022/06), Table 1, Table 2]。

## 4. 対処と n-step の関係

### 4.1 Advantage Learning と DAU

Bellemare らの Atari の実験は 1-step の DQN だけで、n-step との組み合わせは無い（bellemare2016_increasing_action_gap_paper_2026-10-03.md、SMDP の PRD の検討に効く点）。Tallec らの DAU も多段の目標値を扱っていない（tallec2019_time_discretization_paper_2026-10-03.md、論文が扱っていない点）。

### 4.2 Munchausen

Munchausen の論文は、極限で action gap が (1+α)/(1−α) 倍になることを示し（定理 2）、温度 τ → 0 で Advantage Learning に帰着するとした [Vieillard Munchausen (2020/07), Section 3, Theorem 2]。Asterix で測った action gap は、M-DQN と AL がどちらも Adam の DQN より大きかった [Vieillard Munchausen (2020/07), Section 4, Figure 2]。n-step との関係は次のとおりである。

```
without using common deep RL tricks like n-steps returns, PER or distRL

n-step の収益・PER・分布型といった、よく使う深層 RL の工夫を使わずに
```

[Vieillard Munchausen (2020/07), Section 4]

M-IQN は 3-step の収益の IQN と組み合わせ（§2）、付録 B.3 では 1-step の M-IQN も僅差で Rainbow を上回り、3-step の M-IQN は 1-step の M-IQN をはっきり上回った [Vieillard Munchausen (2020/07), Section 2, Appendix B.3]。脚注 5 は 1-step の収益を使うと書いており、本文の記述と食い違う [Vieillard Munchausen (2020/07), Section 4, footnote 5]。n ごとの action gap は報告していない。BTR の Munchausen の有無の action gap（0.282 と 0.055）は既存の報告（btr_hyperparams_survey_2026-08-26.md の §9.2）にあり、n の比較は無い。

### 4.3 policy churn と Advantage Learning

§1.3 のとおり、AL で churn が減ることは示されたが、比べた Agent の n は固定で、n との組み合わせは調べていない [Schaul PolicyChurn (2022/06), Section 3.2, Table 1]。

## 5. 批判と留保

- Farahmand の結果は抄録だけで確かめた。定理の前提（規則性 ζ の定義）は読んでいない。
- Rowland らの鎖の MDP の結果は表形式の小さな問題で、Atari の DQN の n = 5 は C-trace の論文の対照としての調整の結果である。
- Hernandez-Garcia らは γ = 1 の Mountain Car で、ε 0.1 の固定の方策である。EBU の迷路の n-step は n がエピソードの長さで、Atari の n とは別物である。
- Munchausen の論文は M-IQN の n について本文と脚注が食い違う。

## 6. 総合評価

事実として言えるのは、n を大きくすると報酬が速く伝わり序盤の学習が速くなる報告が DQN 系で複数あること（§3.1・§3.2・§3.3）、補正なしの n-step は固定点が行動の方策の分だけ偏ること（§2.1）、action gap の小ささが Greedy の入れ替わりの要因として扱われていること（§1.3）である。n と action gap を直接結ぶ文献は見つからなかった。

以下はこの報告の読み取り（推論）である。Rowland らの固定点の形から、行動の方策が Greedy に近ければ、補正なしの n-step の固定点の action gap は 1-step と大きく変わらない見込みで、n が変えるのは主に途中の推定の誤差（伝わる速さと分散）と、行動の方策の分の偏りになる。Schaul らの仮説の形で言えば、n が Greedy の止まりに効くとすれば、action gap そのものより、action gap に対する近似誤差の大きさを通してである。どちらも文献で確かめられていない。

## 7. 調査の限界

- 使った検索語（一般の語・論文の題・著者名だけ）: 「n-step returns action gap deep Q-learning」「Hernandez-Garcia Sutton "Understanding Multi-Step Deep Reinforcement Learning" DQN target」「"action gap" "multi-step" OR "n-step" bootstrapping reinforcement learning paper」「advantage learning gap-increasing operator combined with multi-step returns n-step DQN」「n-step return horizon in physical time frame skip time discretization Q-learning action gap vanishes small time step」「"action gap" Atari measured "n-step" OR "multi-step" ablation Rainbow IQN agent action gap increase」。被引用数は Semantic Scholar の API に arXiv の番号を渡して取った。
- Farahmand（2011）は抄録、Kozuno ら（2021）と Kalyanakrishnan ら（2021）は抄録だけを読んだ。被引用数は Semantic Scholar の API で 2026-11-27 に取得し、Farahmand の論文は取得の上限に当たって取れなかった。
- 見つけたが範囲外にしたもの: オフラインの多段の目標を扱う Horizon-Scalable Q-Learning（arXiv 2611.10987、2026/11）。action gap と時間の刻みに触れていない。
- Hernandez-Garcia らの論文の掲載先は、arXiv の頁に記載が無く確かめていない。

## 8. 出典の一覧

[Farahmand ActionGap, 2011/12] Amir-massoud Farahmand. "Action-Gap Phenomenon in Reinforcement Learning." NeurIPS 2011（NIPS 24）. https://proceedings.neurips.cc/paper_files/paper/2011/hash/013d407166ec4fa56eb1e1f8cbe183b9-Abstract.html — 被引用数は未取得。
[Schaul PolicyChurn, 2022/06] Tom Schaul, André Barreto, John Quan, Georg Ostrovski（DeepMind）. "The Phenomenon of Policy Churn." NeurIPS 2022. https://arxiv.org/abs/2206.00730（本文は https://ar5iv.labs.arxiv.org/html/2206.00730）— 被引用 41。
[Kalyanakrishnan FrameSkip, 2021/02] Shivaram Kalyanakrishnan ほか. "An Analysis of Frame-skipping in Reinforcement Learning." arXiv:2102.03718. https://arxiv.org/abs/2102.03718 — 被引用 29。
[Rowland AdaptiveTradeoffs, 2019/10] Mark Rowland, Will Dabney, Rémi Munos（所属は確かめていない）. "Adaptive Trade-Offs in Off-Policy Learning." AISTATS 2020. https://arxiv.org/abs/1910.07478（本文は https://ar5iv.labs.arxiv.org/html/1910.07478）— 被引用 26。
[Kozuno PengQ, 2021/02] Tadashi Kozuno, Yunhao Tang, Mark Rowland, Rémi Munos, Steven Kapturowski, Will Dabney, Michal Valko, David Abel. "Revisiting Peng's Q(λ) for Modern Reinforcement Learning." arXiv:2103.00107（Semantic Scholar は ICML 2021）. https://arxiv.org/abs/2103.00107 — 被引用 29。
[Daley AveragingNstep, 2024/02] Brett Daley, Martha White, Marlos C. Machado（University of Alberta・Amii）. "Averaging n-step Returns Reduces Variance in Reinforcement Learning." ICML 2024. https://arxiv.org/abs/2402.03903（本文は https://arxiv.org/html/2402.03903v4）— 被引用 12。
[Hessel Rainbow, 2017/10] Matteo Hessel ほか. "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI 2018（arXiv の頁は投稿中と記載、Semantic Scholar は AAAI）. https://arxiv.org/abs/1710.02298（本文は https://ar5iv.labs.arxiv.org/html/1710.02298）— 被引用 2783。
[HernandezGarcia MultiStepDQN, 2019/01] J. Fernando Hernandez-Garcia, Richard S. Sutton（University of Alberta）. "Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target." arXiv:1901.07510. https://arxiv.org/abs/1901.07510（本文は https://ar5iv.labs.arxiv.org/html/1901.07510）— 被引用 82。
[Lee EBU, 2018/05] Su Young Lee, Sungik Choi, Sae-Young Chung（所属は確かめていない）. "Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update." NeurIPS 2019. https://arxiv.org/abs/1805.12375（本文は https://ar5iv.labs.arxiv.org/html/1805.12375）— 被引用 88。
[Ostrovski PixelCNN, 2017/03] Georg Ostrovski, Marc G. Bellemare, Aäron van den Oord, Rémi Munos. "Count-Based Exploration with Neural Density Models." ICML 2017（Semantic Scholar の記載）. https://arxiv.org/abs/1703.01310（本文は https://ar5iv.labs.arxiv.org/html/1703.01310）— 被引用 711。
[Vieillard Munchausen, 2020/07] Nino Vieillard, Olivier Pietquin, Matthieu Geist（Google Research, Brain Team）. "Munchausen Reinforcement Learning." NeurIPS 2020. https://arxiv.org/abs/2007.14430（本文は https://ar5iv.labs.arxiv.org/html/2007.14430）— 被引用 123。
