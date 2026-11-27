# Survey: DQN 系の n-step と行動間の価値の差（action gap）・まれな報酬の伝わる速さ

## 元の材料

[Survey: DQN 系の n-step と行動間の価値の差（action gap）・まれな報酬の伝わる速さ](../../../../reports/nstep_action_gap_survey_2026-11-27.md) blob:ba56347f85a3d9754537da9ab4cfd4ae7e024c03

## 要点

- DQN 系の制御で n ごとに action gap を測った実験、時間の刻み（frame skip）と n-step と action gap を結ぶ理論、Advantage Learning・DAU・Munchausen と n-step を組み合わせて action gap や Greedy の止まりを比べた実験は、2026-11-27 の調査では見つからなかった。
- Rainbow は多段の目標の利点を、新しく観測した報酬を先に訪れた状態へ速く伝えることとし（n は 3）、外すと 57 ゲーム中 53 ゲームで完全な Agent を下回った [Hessel Rainbow (2017/10), Ablation studies, Figure 4]。
- DQN に似た構成の Mountain Car（γ = 1、各 100 回の独立な Run）では、n = 20 が序盤と全体の成績で最もよく、成績を損なうほど大きい n は見つからなかった [HernandezGarcia MultiStepDQN (2019/01), Section 4.2]。
- 補正なしの n-step は分散が小さい代わりに固定点が偏り、固定点は 1 step を目標の方策 π、続く n−1 step を行動の方策 μ で進むことを繰り返す非定常な方策の価値になる [Rowland AdaptiveTradeoffs (2019/10), Section 2, Proposition B.1]。
- n-step の収益の分散は n とともに増え、同じ縮小率の複合の収益は n-step より分散が大きくならない [Daley AveragingNstep (2024/02), Section 3.1, Section 3.3]。
- Schaul らは、action gap が小さいと小さな近似誤差で次善の行動が最善を追い越すという仮説を立て、Advantage Learning で policy churn が大きく減り action gap が大きくなることを示した [Schaul PolicyChurn (2022/06), Section 3.2]。
- Munchausen の論文は M-DQN を n-step なしで、M-IQN を 3-step で使い、1-step の M-IQN も僅差で Rainbow を上回った [Vieillard Munchausen (2020/07), Section 4, Appendix B.3]。

## 内容

### 1. 理論: 割引率・時間の刻みと action gap

時間の刻みで Q が V に潰れること、長い地平で action gap が崩れうること、action gap を広げる作用素は既存の報告にあり、調べ直していない。Greedy の方策の損失は action gap の規則性 ζ ≥ 0 によって O(‖Q̂ − Q*‖^(1+ζ)) で抑えられる（抄録で確かめた）[Farahmand ActionGap (2011/12), Abstract]。AL は ε = 0 の評価でも成績が高かった [Schaul PolicyChurn (2022/06), Figure 18]。同じ行動を d step 続ける繰り返しの損失は「慣性の代価」で抑えられ、実効の地平が短くなる得で相殺されうる（抄録で確かめた。n-step と action gap には触れていない）[Kalyanakrishnan FrameSkip (2021/02), Abstract]。

### 2. 理論: n-step の偏りと分散

20 状態の鎖の MDP では、環境の step が少ないうちは n の大きい補正なしの更新が最もよく、step が増えると偏りの小さい手法に追い越された。Atari-57 の DQN 系の比較では n-step の対照は n = 5 が最もよかった [Rowland AdaptiveTradeoffs (2019/10), Section 4, Appendix E.2]。Schaul らは、補正なしの多段の更新が効くことを、方策が速く変わるため補正のある方法では打ち切りが大きくなる、と churn で説明した（議論で、実験ではない）[Schaul PolicyChurn (2022/06), Section 4.3]。Peng's Q(λ) は、行動の方策が Greedy をゆっくり追うなら最適な方策に収束する（抄録で確かめた）[Kozuno PengQ (2021/02), Abstract]。MinAtar の 5 ゲーム（32 試行）の多段の DQN では、n = 5 の比較で 5 ゲーム中 3 ゲーム、n = 3 で 1 ゲームで複合の収益が上回った [Daley AveragingNstep (2024/02), Section 5]。

### 3. 実験: DQN 系で n を大きくしたとき

Rainbow の調整では n = 1・3・5 を比べ、最後は 3 が最もよかった [Hessel Rainbow (2017/10), Hyper-parameter tuning]。Mountain Car の DQN の最終の成績は n ≥ 3 で同じくらいだった [HernandezGarcia MultiStepDQN (2019/01), Section 4.2]。10×10 の MNIST の迷路では、経路の長さの平均が壁 20% で 1-step DQN 14.40・n-step DQN 3.26、壁 50% で 22.36・11.32 だった [Lee EBU (2018/05), Section 5.1, Table 1]。Monte Carlo 収益を混ぜた更新（MMC）は、Bank Heist と Time Pilot では改善の大半を説明し、Ms. Pac-Man と Breakout では成績を下げた。Retrace(λ) を使う Reactor では、軌跡を慎重に切る性質がボーナスの効きを抑えたと著者は推測した [Ostrovski PixelCNN (2017/03), Section 4.2, Section 6]。1 回の更新で Greedy の行動が変わる割合は、1-step の Double DQN で約 9%、5-step の R2D2 で約 6% だったが、n を変えた比較ではない [Schaul PolicyChurn (2022/06), Table 1]。

### 4. 対処と n-step の関係

Bellemare らの AL の Atari の実験は 1-step だけで、Tallec らの DAU も多段の目標値を扱っていない（既存の報告）。Munchausen は極限で action gap を (1+α)/(1−α) 倍にし、温度 τ → 0 で AL に帰着する。Asterix で測った action gap は M-DQN と AL がどちらも Adam の DQN より大きかった。脚注 5 は 1-step の収益を使うと書き、本文と食い違う [Vieillard Munchausen (2020/07), Section 3, Section 4, Figure 2]。

### 5. 批判と留保・調査の限界

Farahmand、Kozuno ら、Kalyanakrishnan らは抄録だけを読んだ。Rowland らの鎖の MDP は小さな表形式の問題で、Hernandez-Garcia らは ε 0.1 の固定の方策、EBU の迷路の n はエピソードの長さである。

## 主要な概念

- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [訪問回数と擬似カウントによる探索](../../concepts/methods/pseudo_counts.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
