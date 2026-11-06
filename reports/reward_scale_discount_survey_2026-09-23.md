# Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)

Date: 2026-09-23
Scope: 深層強化学習(主に Atari)の報酬スケール処理と割引率について、次の 4 点を論文と公式実装の一次情報から確定させる。(1) 大規模エージェントが報酬クリップの代わりに使っているもの(可逆な価値の再スケーリング h(x)、Pop-Art、TD 誤差の正規化、symlog、分類型の価値表現)。(2) 報酬クリップの有無を統制して比べた実験。特に DQN 系で 200M frames 前後の予算のもの。(3) 割引率 γ の選び方と、0.997 を超える γ と h(x) を組み合わせたときの効き方。(4) 学習中に γ と n-step を変えるスケジュールの効果。

調査は 5 つの角度に分けて並行して行い、本書へ統合した。角度は、大規模エージェントの報酬処理、クリップ有無の統制比較、γ と h(x)、γ・n-step のスケジュール、公式実装での確認である。確認した出典は延べ約 160 件(角度間の重複を含む)で、本書が直接引用したものを §10 に挙げる。

## Table of Contents

1. [前提: 報酬クリップとは何か](#1-前提-報酬クリップとは何か)
   - [1.1 DQN が報酬をクリップした理由と原典が認めた副作用](#11-dqn-が報酬をクリップした理由と原典が認めた副作用)
   - [1.2 クリップの実装は符号化と切り詰めの 2 通りがある](#12-クリップの実装は符号化と切り詰めの-2-通りがある)
   - [1.3 後続論文が指摘した副作用](#13-後続論文が指摘した副作用)
2. [大規模エージェントは報酬クリップの代わりに何を使っているか](#2-大規模エージェントは報酬クリップの代わりに何を使っているか)
   - [2.1 エージェント別の一覧](#21-エージェント別の一覧)
   - [2.2 可逆な価値の再スケーリング h(x)](#22-可逆な価値の再スケーリング-hx)
   - [2.3 Pop-Art](#23-pop-art)
   - [2.4 TD 誤差の正規化](#24-td-誤差の正規化)
   - [2.5 symlog と分類型の価値表現](#25-symlog-と分類型の価値表現)
   - [2.6 報酬クリップを残している系統](#26-報酬クリップを残している系統)
3. [報酬クリップの有無を統制した比較](#3-報酬クリップの有無を統制した比較)
   - [3.1 比較研究の一覧](#31-比較研究の一覧)
   - [3.2 正規化も変換もしない生の報酬では集計値が崩れやすい](#32-正規化も変換もしない生の報酬では集計値が崩れやすい)
   - [3.3 代替手段を使うと集計ではクリップとほぼ同等かやや劣る](#33-代替手段を使うと集計ではクリップとほぼ同等かやや劣る)
   - [3.4 ゲーム別の結果は割れる](#34-ゲーム別の結果は割れる)
   - [3.5 DQN 系で約 200M frames のままクリップだけを外した A/B は見つからなかった](#35-dqn-系で約-200m-frames-のままクリップだけを外した-ab-は見つからなかった)
4. [割引率 γ の設定と高い γ での h(x) の効き方](#4-割引率-γ-の設定と高い-γ-での-hx-の効き方)
   - [4.1 Atari で使われる γ とその選定根拠](#41-atari-で使われる-γ-とその選定根拠)
   - [4.2 高い γ で何が問題になるか](#42-高い-γ-で何が問題になるか)
   - [4.3 h(x) と 0.997 超の γ を組み合わせた実験](#43-hx-と-0997-超の-γ-を組み合わせた実験)
   - [4.4 報酬クリップと γ の相互作用](#44-報酬クリップと-γ-の相互作用)
   - [4.5 低い γ と価値の写像](#45-低い-γ-と価値の写像)
5. [学習中に γ と n-step を変えるスケジュール](#5-学習中に-γ-と-n-step-を変えるスケジュール)
   - [5.1 γ を上げていく DQN の実験](#51-γ-を上げていく-dqn-の実験)
   - [5.2 BBF の γ と n のアニール](#52-bbf-の-γ-と-n-のアニール)
   - [5.3 メタ勾配による γ の適応](#53-メタ勾配による-γ-の適応)
   - [5.4 γ の異なる方策群から選ぶメタコントローラ](#54-γ-の異なる方策群から選ぶメタコントローラ)
   - [5.5 n-step と λ を変える研究](#55-n-step-と-λ-を変える研究)
   - [5.6 スケジュールの理論的な説明](#56-スケジュールの理論的な説明)
6. [公式実装での確認](#6-公式実装での確認)
   - [6.1 実装別の一覧](#61-実装別の一覧)
   - [6.2 h(x) の実装と ε の既定値](#62-hx-の実装と-ε-の既定値)
   - [6.3 報酬クリップの既定値](#63-報酬クリップの既定値)
   - [6.4 BBF と BTR のスケジュール実装](#64-bbf-と-btr-のスケジュール実装)
7. [批判と留保](#7-批判と留保)
8. [総合評価](#8-総合評価)
9. [調査の限界](#9-調査の限界)
10. [出典リスト](#10-出典リスト)

表記の約束:

- 「本調査の計算」「本調査の読み取り」「本調査の整理」と書いた箇所は、本調査による算術・図の読み取り・分類であり、出典の主張ではない。
- 【目視】は論文の図から読み取った概算値、【画素換算】は対数軸の図の棒の端を目盛りの画素位置から換算した概算値で、どちらも原文に数値はない。
- 引用中のギリシャ文字・根号・上付きなど、PDF のテキスト抽出で崩れた記号は arXiv HTML 版・ar5iv・PDF の表示で確認して復元した。
- 出典ラベルの年月は初出(arXiv 初版または掲載)の年月で、読んだ版は §10 に書いた。コードの年月は参照したコミットの年月である。
- 「frames」は ALE の生フレーム、「agent steps」は行動反復後のステップ、「gradient steps」は勾配更新の回数を指す。Atari の標準の行動反復 4 では 1 agent step = 4 frames である。

---

## 1. 前提: 報酬クリップとは何か

### 1.1 DQN が報酬をクリップした理由と原典が認めた副作用

**DQN は学習中だけ報酬を [−1, 1] にクリップした。理由はゲーム間のスコア規模の差で、同じ学習率を全ゲームで使うためである。原典は同じ段落で、報酬の大小を区別できなくなる副作用も認めている。**

```
While we evaluated our agents on unmodified games, we made one change to the reward structure of the games during training only. As the scale of scores varies greatly from game to game, we clipped all positive rewards at 1 and all negative rewards at −1, leaving 0 rewards unchanged. Clipping the rewards in this manner limits the scale of the error derivatives and makes it easier to use the same learning rate across multiple games. At the same time, it could affect the performance of our agent since it cannot differentiate between rewards of different magnitude.

エージェントの評価は無改変のゲームで行ったが、学習中に限り、ゲームの報酬構造に 1 点だけ変更を加えた。スコアの規模はゲームごとに大きく異なるため、正の報酬はすべて 1 に、負の報酬はすべて −1 にクリップし、0 の報酬はそのままにした。このように報酬をクリップすると誤差微分の規模が制限され、複数のゲームで同じ学習率を使いやすくなる。同時に、大きさの異なる報酬を区別できないため、エージェントの性能に影響する可能性がある。
```

[Mnih DQN (2015/02), Methods "Training details"]

DQN はこれとは別に、更新式の TD 誤差項も [−1, 1] にクリップしている。原典はこれを、区間外の誤差に絶対値損失を使うことと同じだと説明している [Mnih DQN (2015/02), Methods "Training algorithm for deep Q-networks"]。γ は全ゲーム共通の 0.99 で、ハイパーパラメータ全体は 5 ゲームでの非公式な探索で選んだと書かれている(§4.1)。

### 1.2 クリップの実装は符号化と切り詰めの 2 通りがある

**「報酬クリップ」と呼ばれる処理には、符号で {+1, 0, −1} に置き換える実装と、[−1, 1] に切り詰める実装の 2 通りがある。** Stable-Baselines3 とそこから移植した CleanRL は前者、Dopamine の Runner は後者である。

```python
    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))
```

[cleanrl (2026/04), cleanrl_utils/atari_wrappers.py:L223-L230]

```python
      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)
```

[dopamine (2026/03), dopamine/discrete_domains/run_experiment.py:L408-L410]

両者の結果が異なるのは、絶対値が 1 未満の非ゼロ報酬があるときだけである(本調査の整理)。この定義の違いは第三者の GitHub issue で議論されており、TorchRL のメンテナは符号化への修正に同意している [TorchRL issue1777 (2024/01)]。DQN Zoo は行動反復 4 回分の報酬を合算してから ±1 にクリップする [dqn_zoo (2023/12), dqn_zoo/processors.py:L476-L487]。

Machado らの ALE 再検討論文は、ALE 向けのエージェントのほとんどが何らかの報酬正規化を行っていると述べ、符号だけを使う方法と、最初の非ゼロ報酬の大きさで割る方法を例に挙げている。この論文にクリップの有無を比べる実験はない [Machado RevisitingALE (2017/09), Section 2.1]。

### 1.3 後続論文が指摘した副作用

**後続論文は、クリップが目的関数そのものを変えると指摘している。** 最適方策の集合が変わる例として、Pohlen らは Bowling を挙げた。

```
Mnih et al. [13] showed that clipping rewards to the canonical interval [-1, 1] is one way to achieve stability. However, this clipping operation may change the set of optimal policies. For example, the agent no longer differentiates between striking a single pin or all ten pins in BOWLING.

Mnih ら [13] は、報酬を標準区間 [-1, 1] にクリップすることが安定化の一方法であることを示した。しかし、このクリップ操作は最適方策の集合を変えうる。たとえば、エージェントは BOWLING でピンを 1 本倒すことと 10 本すべて倒すことを区別しなくなる。
```

[Pohlen ApeXDQfD (2018/05), Section 1]

van Hasselt らは、クリップによって報酬の総和ではなく頻度を最適化することになると書いている。

```
First, such clipping introduces domain knowledge. Most games have sparse non-zero rewards outside of [−1,1]. Clipping then results in optimizing the frequency of rewards, rather than their sum. This is a good heuristic in Atari, but it does not generalize to other domains. More importantly, the clipping changes the objective, sometimes resulting in qualitatively different policies of behavior.

第一に、このようなクリップはドメイン知識を持ち込む。多くのゲームでは、[−1,1] の外側の非ゼロ報酬がまばらに出る。するとクリップによって、報酬の総和ではなく報酬の頻度を最適化することになる。これは Atari では良いヒューリスティックだが、他のドメインには一般化しない。より重要なのは、クリップが目的関数を変え、ときに質的に異なる行動方策をもたらすことである。
```

[vanHasselt PopArt (2016/02), Section 1]

**同時に、クリップが Atari では結果的にうまく働くという指摘もある。** Schaul らは、クリップが一部のゲームを解けなくする一方で、多くの Atari ゲームの設計とたまたま整合すると書いている。

```
Reward clipping, i.e. capping rewards to lie in [−1, 1], is commonly used in Atari since at least DQN (Mnih et al., 2015). It breaks the original problem semantics, as the agent becomes blind to large reward events, making some games impossible to solve (e.g., BOWLING or SKIING) or imposing a performance ceiling. However, the heuristic of accumulating many rewards independently of their magnitude happens to be well-aligned with the design of many Atari games, so overall results tend to be good.

報酬クリップ、すなわち報酬を [−1, 1] に収めることは、少なくとも DQN(Mnih et al., 2015)以来 Atari で一般に使われている。これは元の問題の意味を壊す。エージェントは大きな報酬イベントが見えなくなり、一部のゲーム(例: BOWLING や SKIING)が解けなくなるか、性能に天井ができる。しかし、報酬の大きさに関係なく多くの報酬を積み上げるというヒューリスティックは、たまたま多くの Atari ゲームの設計とよく整合しているため、全体としての結果は良くなりやすい。
```

[Schaul ReturnScaling (2021/05), Section 4.3]

Toromanoff らは、クリップ付きで学習した Rainbow-IQN が Bowling でストライクやスペアを避けるようになった、と観察として報告している。クリップなしとの比較はしていない [Toromanoff SABER (2019/08), Section 6]。DQfD の著者は、人間が DQN を上回るゲームの多くはクリップが原因だったと書き、Private Eye を例に挙げている。そのうえで符号付き対数変換 sign(r)·log(1+|r|) を使ったが、クリップとの A/B は示していない [Hester DQfD (2017/04), Section "Experimental Setup"]。

## 2. 大規模エージェントは報酬クリップの代わりに何を使っているか

### 2.1 エージェント別の一覧

**クリップを外した大規模エージェントの多くは、報酬ではなく価値のスケールを可逆関数 h(x) で縮める。そこに分類型の表現や TD 誤差の正規化を重ねる。** 下表は論文本文と公式実装から確認した値である。「公式実装」とある値は論文ではなくコードから読んだ(§6)。

| エージェント(年) | 報酬処理 | 定数 | γ | n-step など | 予算(原文の単位) | 出典 |
|---|---|---|---|---|---|---|
| DQN (2015) | クリップ [−1, 1]、TD 誤差もクリップ | — | 0.99 | 1 | 50 million frames | [Mnih DQN (2015/02), Methods] |
| Pop-Art Double DQN (2016) | Pop-Art(報酬と TD 誤差のクリップを外す) | step size はおおまかに 10⁻⁴ | 0.99 | 1 | 200M frames | [vanHasselt PopArt (2016/02), Section 4] |
| Rainbow (2017) | クリップ + C51 の support | 51 atoms、[−10, 10] | 0.99 | 3 | 200M frames | [Hessel Rainbow (2017/10), Table 1, 3, 4] |
| Ape-X DQN (2018) | クリップ(actor 側の前処理) | — | 本文に数値なし | 3 | 22800M frames | [Horgan ApeX (2018/03), Appendix C, Table 1] |
| IMPALA (2018) | Atari はクリップ。DMLab は tanh を使う非対称クリップ | — | 0.99 | V-trace、unroll 20 | ゲームあたり 200M frames | [Espeholt IMPALA (2018/02), Table G.1, Figure D.1] |
| Ape-X DQfD (2018) | h(x)、報酬は未加工 | ε = 10⁻² | 0.999 | 1-step と 10-step を保存 | 学習時間(最大 140 h)で表示 | [Pohlen ApeXDQfD (2018/05), Section 3.2, 3.4, Appendix C] |
| PopArt-IMPALA (2018) | Pop-Art(タスクごとの統計)。クリップあり/なしの両方で実験 | β = 3×10⁻⁴、σ を [0.0001, 1e6] に制限 | 0.99 | unroll 20 | 57 × 200M = 1.14×10¹⁰ frames | [Hessel PopArtIMPALA (2018/09), "Implementation notes", Table 5, 6] |
| R2D2 (2018) | h(x)、クリップなし | ε = 10⁻³ | 0.997 | 5 | 本文に総量の記載なし | [Kapturowski R2D2 (2018/09), Section 2.3, Table 2] |
| NGU (2020) | 変換 Retrace(h) | 付録は ε = 10⁻²、表は 0.001(不一致) | 0.99〜0.997(N = 32) | Retrace λ = 0.95 | 35 billion frames | [Badia NGU (2020/02), Section 3, 4.2, Appendix E, Table 6] |
| Agent57 (2020) | 変換 Retrace(h)、外的・内的価値を別ネットワークに分離 | 表は 0.001 | 0.99〜0.9999(N = 32) | Retrace λ = 0.95 | Skiing の人間超えは 78 billion frames 後 | [Badia Agent57 (2020/03), Section 3.1, 4, 4.1, Appendix G] |
| MuZero (2019) | h(x) + 分類型(601 support) | ε = 0.001、support −300〜300 | 0.997 | 10 | 20.0B frames | [Schrittwieser MuZero (2019/11), Appendix C, F, G, Table 1] |
| MuZero Reanalyze (2019) | MuZero と同じ記述の範囲 | 同上 | 0.997 | 5 | ゲームあたり 200M frames | [Schrittwieser MuZero (2019/11), Appendix H, Table 1] |
| Muesli (2021) | 非線形変換空間での分類型損失 | 本文に数値なし | 0.995(大規模設定は 0.997) | Retrace λ = 0.95 | 200M frames | [Hessel Muesli (2021/04), Section 4.5, Table 4, 5, 6] |
| MEME (2022) | h(x) + TD 誤差の正規化 | 0.001(表の式の形は異なる。§2.2)、σ の下限 0.01 | 0.97〜0.9997(N = 16) | Soft Watkins Q(λ)、λ = 0.95 | 予算 1B frames、全 57 ゲームの人間超えは 390M frames | [Kapturowski MEME (2022/09), Section 4, 5, Appendix A Table 2] |
| EfficientZero (2021) | クリップ + 分類型。公式実装は h(ε = 0.001)と support −300〜300 | 601 出力 | 0.997⁴(表記どおり) | TD steps 5(古いデータほど短縮) | Atari 100k | [Ye EfficientZero (2021/11), Appendix A.1 Table 6]、[EfficientZero code (2022/08)] |
| DreamerV3 (2023) | symlog / symexp twohot + リターンのパーセンタイル正規化 | 255 bins(v1 の説明)、公式実装は clip_reward: False | 0.997(ホライズン 333) | λ-return、λ = 0.95 | 200M frames(v2) | [Hafner DreamerV3 (2023/01), "Robust predictions", Table 4]、[dreamerv3 code (2026/05)] |
| Stop Regressing (2024) | 分類型(HL-Gauss)。報酬処理の記載なし | 51 locations、[−10, 10] | 0.99 | 1 | 200M frames、60 ゲーム、5 seeds | [Farebrother StopRegressing (2024/03), Section 3.1, Appendix B.1] |
| BBF (2023) | 論文に記載なし。公式実装は Dopamine の既定で [−1, 1] にクリップ + C51 | 51 atoms、[−10, 10] | 0.97 → 0.997(スケジュール) | 10 → 3(スケジュール) | Atari 100k | [Schwarzer BBF (2023/05), Section 4]、[bigger_better_faster (2023/04)] |
| BTR (2024) | クリップ + IQN + Munchausen | Munchausen τ = 0.03、α = 0.9、l₀ = −1 | 0.997 | 3 | 200M frames | [Clark BTR (2024/11), Section 3.2, Table D4, D6] |
| PQN (2024) | 論文に記載なし。公式実装は reward_clip: True | — | 0.99 | Q(λ)、λ = 0.65 | 公式設定は 5e7 steps(コメントは 200M frames 相当) | [Gallici PQN (2024/07), Table 5]、[purejaxql (2025/11)] |
| Return-based scaling (2021) | TD 誤差を σ で割る。R2D2 ベースで未クリップ報酬 | σ² = V[R] + V[γ]E[G²] | 0.997(1 ヘッド) | multi-step | 10⁹ frames | [Schaul ReturnScaling (2021/05), Section 3, 4.1, 4.2] |

本調査の整理として、h(x) を使う系統(Ape-X DQfD、R2D2、NGU、Agent57、MEME、MuZero、Muesli)はいずれも DeepMind 所属の論文である。一方、単一 GPU やサンプル効率を重視する系統(Rainbow、BBF、BTR、EfficientZero、PQN)はクリップを残している(§2.6)。

### 2.2 可逆な価値の再スケーリング h(x)

**Pohlen らは、報酬ではなく行動価値のスケールを縮める作用素 T_h を定義した。** h は可逆な圧縮関数で、ブートストラップ値を h⁻¹ で実空間に戻して報酬を足し、再び h で縮めた値をターゲットにする。

```
Instead of reducing the magnitude of the rewards, we propose to focus on the action-value function instead. We use a function h : ℝ → ℝ that reduces the scale of the action-value function. Our new operator T_h is defined as
(T_h Q)(x,a) := E_{x′∼P(·|x,a)} [ h( R(x,a) + γ max_{a′∈A} h⁻¹(Q(x′,a′)) ) ], ∀(x,a) ∈ X × A.

報酬の大きさを下げる代わりに、行動価値関数の方に注目することを提案する。行動価値関数のスケールを縮める関数 h : ℝ → ℝ を用いる。新しい作用素 T_h を次で定義する:
(T_h Q)(x,a) := E_{x′∼P(·|x,a)} [ h( R(x,a) + γ max_{a′∈A} h⁻¹(Q(x′,a′)) ) ]、∀(x,a) ∈ X × A。
```

[Pohlen ApeXDQfD (2018/05), Section 3.2]

**Pohlen らが使った h は sign(z)(√(|z|+1) − 1) + εz で、ε = 10⁻² である。** εz の項は h⁻¹ をリプシッツ連続にするために入れてある。

```
In our algorithm, we use h : z ↦ sign(z)(√(|z|+1) − 1) + εz with ε = 10⁻² where the additive regularization term εz ensures that h⁻¹ is Lipschitz continuous (see Proposition A.1). We chose this function because it has the desired effect of reducing the scale of the targets while being Liptschitz continuous and admitting a closed form inverse.

我々のアルゴリズムでは h : z ↦ sign(z)(√(|z|+1) − 1) + εz(ε = 10⁻²)を用いる。加算的な正則化項 εz は h⁻¹ がリプシッツ連続であることを保証する(命題 A.1 を参照)。この関数を選んだのは、ターゲットのスケールを縮めるという望ましい効果を持ちつつ、リプシッツ連続で、閉じた形の逆関数を持つからである。
```

[Pohlen ApeXDQfD (2018/05), Section 3.2]

逆関数は h⁻¹(x) = sign(x)(((√(1+4ε(|x|+1+ε)) − 1)/(2ε))² − 1) で、リプシッツ定数は L_h = 1/2 + ε、L_{h⁻¹} = 1/ε である [Pohlen ApeXDQfD (2018/05), Appendix A, Proposition A.2]。

**R2D2 は同じ h を ε = 10⁻³ で採用し、報酬をクリップせずに n-step ターゲットへ適用した。** 以後の係数も 0.001 で、NGU・Agent57・MEME はハイパーパラメータ表に、MuZero は本文に書いている(§2.1 の表)。

```
Following the modified Ape-X version in Pohlen et al. (2018), we do not clip rewards, but instead use an invertible value function rescaling of the form h(x) = sign(x)(√(|x| + 1) − 1) + εx which results in the following n-step targets for the Q-value function:
ŷ_t = h( Σ_{k=0}^{n−1} r_{t+k} γ^k + γ^n h⁻¹( Q(s_{t+n}, a*; θ⁻) ) ), a* = arg max_a Q(s_{t+n}, a; θ).

Pohlen et al. (2018) の変更版 Ape-X にならい、報酬をクリップせず、代わりに h(x) = sign(x)(√(|x| + 1) − 1) + εx という形の可逆な価値関数の再スケーリングを用いる。これにより、Q 値関数の n-step ターゲットは次のようになる:
ŷ_t = h( Σ_{k=0}^{n−1} r_{t+k} γ^k + γ^n h⁻¹( Q(s_{t+n}, a*; θ⁻) ) )、a* = arg max_a Q(s_{t+n}, a; θ)。
```

[Kapturowski R2D2 (2018/09), Section 2.3]

R2D2 の表は ε = 10⁻³ を明記している [Kapturowski R2D2 (2018/09), Table 2]。公式ライブラリ rlax、Acme の R2D2、SEED RL の R2D2 の既定値も 1e-3 である(§6.2)。

**h の理論保証は限定的である。** T_h^k Q が h∘Q* に収束するのは、h が線形の場合と、MDP が決定的(遷移と報酬が点測度)で h が狭義単調の場合だけだと示されている [Pohlen ApeXDQfD (2018/05), Section 3.2, Proposition 3.1]。確率的 MDP では、γ < 1/(L_h L_{h⁻¹}) のときに縮小写像になるが、固定点は h∘Q* とは限らない。しかも実際に使う γ はこの条件を超えていると、原典自身が書いている。

```
The following proposition shows that transformed Bellman operator is still a contraction for small γ if we assume a stochastic MDP and a more generic choice of h. However, the fixed point might not be h∘Q*.
[...]
While Proposition A.2 shows that the transformed operator is a contraction, the discount factor γ we use in practice is higher than 1/(L_h L_{h⁻¹}). We leave a deeper investigation of the contraction properties of T_h in stochastic MDPs for future work.

次の命題は、確率的 MDP とより一般的な h を仮定しても、γ が小さければ transformed Bellman operator が依然として縮小写像であることを示す。ただし、固定点は h∘Q* でないかもしれない。
[...]
命題 A.2 は変換された作用素が縮小写像であることを示すが、実際に使う割引率 γ は 1/(L_h L_{h⁻¹}) より高い。確率的 MDP における T_h の縮小性のより深い検討は今後の課題とする。
```

[Pohlen ApeXDQfD (2018/05), Appendix A]

本調査の計算では、1/(L_h L_{h⁻¹}) = ε/(1/2 + ε) であり、ε = 10⁻² で約 0.0196、ε = 10⁻³ で約 0.0020 になる。原典に数値はない。Muesli の著者はこの変換を、実用上とても役立つが確率的なリターンに対してはバイアスがある、と評している。

```
The non-linear transformation by Pohlen et al. (2018) is practically very helpful, although biased for stochastic returns.

Pohlen et al. (2018) の非線形変換は実用上とても役立つが、確率的なリターンに対してはバイアスがある。
```

[Hessel Muesli (2021/04), Appendix E.3]

**MuZero は h で変換した値を、さらに 601 個の support 上のカテゴリ分布で表す。** Pohlen を引用して h(ε = 0.001)でターゲットを縮め、−300〜300 の各整数に support を置いた。推論時は softmax 分布の期待値をとってから変換を逆にかける [Schrittwieser MuZero (2019/11), Appendix F]。

**論文によって h の式の印字が異なる。** 以下は本調査による式の突き合わせである。

- MuZero の本文は sign(x)(√(|x|+1) − 1 + εx) と εx を括弧の内側に置いており、x < 0 で Pohlen の形と符号が変わる [Schrittwieser MuZero (2019/11), Appendix F]。EfficientZero の公式実装は Pohlen の形、EfficientZero V2 の公式実装の Atari 分岐は MuZero の印字の形である [EfficientZero code (2022/08), core/config.py]、[EfficientZeroV2 code (2024/08), ez/utils/format.py]。
- MEME の表は sgn(x)(√(x²+1) − 1) + 0.001x と印字している [Kapturowski MEME (2022/09), Appendix A Table 2]。
- NGU と Agent57 の付録の h⁻¹ は、Proposition A.2 にある 2 乗が無い形で印字されている [Badia NGU (2020/02), Appendix E]、[Badia Agent57 (2020/03), Appendix A]。
- NGU の ε は、付録 E が 10⁻²、Table 6 が 0.001 で一致しない [Badia NGU (2020/02), Appendix E, Table 6]。

どれが誤植でどれが意図的な変更かを述べた記述は、読んだ範囲には無い。

### 2.3 Pop-Art

**Pop-Art は、ターゲットの平均と分散を逐次推定して正規化する処理(ART)と、正規化を変えるたびに最終線形層を補正して正規化前の出力を正確に保つ処理(POP)を組み合わせた手法である。** Atari で報酬をクリップしていた先行研究が主な動機だと原典は書いている。

```
The two properties that we want to simultaneously achieve are
(ART) to update scale Σ and shift μ such that Σ⁻¹(Y − μ) is appropriately normalized, and
(POP) to preserve the outputs of the unnormalized function when we change the scale and shift.

同時に達成したい性質は次の 2 つである。
(ART) Σ⁻¹(Y − μ) が適切に正規化されるように、scale Σ と shift μ を更新すること。
(POP) scale と shift を変えたときに、正規化前の関数の出力を保つこと。
```

[vanHasselt PopArt (2016/02), Section 2]

補正は W_new = Σ_new⁻¹ΣW、b_new = Σ_new⁻¹(Σb + μ − μ_new) で、これにより全入力で正規化前の出力が一致する。正規化を先に更新できるので、大きすぎる更新をその直前で避けられる、と原典は述べている [vanHasselt PopArt (2016/02), Section 2.1, Proposition 1]。Atari での結果は §3 で扱う。

**PopArt-IMPALA は統計をタスクごとのベクトルにし、1 つのエージェントで 57 ゲームを同時に学習した。** 統計はロールアウトごとに減衰率 β = 3×10⁻⁴ で更新し、σ を [0.0001, 1e6] に制限する。β は調整を必要としなかったと書かれている [Hessel PopArtIMPALA (2018/09), "Implementation notes"]。

### 2.4 TD 誤差の正規化

**Schaul らは、ターゲットではなく TD 誤差を、報酬とリターンの統計から求めたスケール σ で割る方法(return-based scaling)を提案した。** σ² = V[R] + V[γ]E[G²] で近似し、統計はそれまでに見た全データから推定する [Schaul ReturnScaling (2021/05), Section 3, 3.1]。

```
We propose to replace raw TD-errors δ_t by a scaled version δ̄_t := δ_t / σ, (1) where σ ∈ ℝ⁺ is an adaptive scale factor.

生の TD 誤差 δ_t を、スケールした版 δ̄_t := δ_t / σ (1) に置き換えることを提案する。ここで σ ∈ ℝ⁺ は適応的なスケール係数である。
```

[Schaul ReturnScaling (2021/05), Section 3]

**MEME は h と併用する形で、TD 誤差の正規化を取り入れた。** 割引率と内的報酬のスケールが異なる Q 関数の族を学習するので、スケールの大きい Q 値が学習を支配しうる、というのが理由である。

```
As we learn a family of Q-functions which vary over a wide range of discount factors and intrinsic reward scales, we expect that the Q-functions will vary considerably in scale. This may cause the larger-scale Q-values to dominate learning and destabilize learning of smaller Q-values. This is a particular concern in environments with very small extrinsic reward scales. To counteract this effect we introduce a normalization scheme on the TD-errors similar to that used in Schaul et al. (2021).

広い範囲の割引率と内部報酬スケールにわたって変化する Q 関数の族を学習するので、Q 関数はスケールが大きく変わると予想される。これにより、スケールの大きい Q 値が学習を支配し、スケールの小さい Q 値の学習を不安定にするかもしれない。これは外部報酬のスケールが非常に小さい環境で特に懸念される。この影響に対抗するため、Schaul et al. (2021) で使われたものと同様の TD 誤差の正規化方式を導入する。
```

[Kapturowski MEME (2022/09), Section 4 "B1"]

MEME の σ は online ネットワークの TD 誤差の標準偏差の移動推定とバッチ内の標準偏差の大きい方で、下限 0.01 を設け、損失と優先度の両方に適用する [Kapturowski MEME (2022/09), Section 4]。アブレーションでの効果は他の改善ほど顕著ではなく、信頼領域など他の正則化的な改善と重なる部分が大きいという仮説を著者は述べている [Kapturowski MEME (2022/09), Section 5]。

### 2.5 symlog と分類型の価値表現

**DreamerV3 は、報酬と価値の予測に symlog/symexp と指数間隔の bin 上の twohot 損失を使い、Atari でも報酬をクリップしない。** 原典は、二乗損失の発散、Huber 損失の停滞、統計による正規化の非定常性という板挟みへの単純な解として symlog を提示している。

```
Predicting large targets using a squared loss can lead to divergence whereas absolute and Huber losses stagnate learning. On the other hand, normalizing targets based on running statistics introduces non-stationarity into the optimization. We suggest the symlog squared error as a simple solution to this dilemma.

大きなターゲットを二乗損失で予測すると発散しうる。一方、絶対値損失や Huber 損失は学習を停滞させる。他方、実行統計に基づいてターゲットを正規化することは、最適化に非定常性を持ち込む。この板挟みへの単純な解として、symlog 二乗誤差を提案する。
```

[Hafner DreamerV3 (2023/01), "Robust predictions"(arXiv v2)]

注: 原文の本文中にある文献番号の上付き数字は省いた。

**DreamerV3 の原典は、R2D2 の変換を「非対称な変換」と呼び、領域全体の平均では効果が低かったと書いている。** どういう意味で非対称なのかの説明は、読んだ範囲(v1・v2)には無い。

```
For critic learning, an alternative asymmetric transformation has previously been proposed, which we found less effective on average across domains. Unlike alternatives, symlog transformations avoid truncating large targets, introducing non-stationary from normalization, or adjusting network weights when new extreme values are detected.

critic の学習については、別の非対称な変換が以前に提案されているが、我々の実験では領域全体の平均で効果が低かった。他の方法と異なり、symlog 変換は、大きなターゲットを切り詰めること、正規化による非定常性の導入、新しい極端な値を検出したときにネットワークの重みを調整することを避けられる。
```

[Hafner DreamerV3 (2023/01), "Robust predictions"(arXiv v2。文献番号 35 = R2D2、7 = DQN、5 = PPO、36 = PopArt-IMPALA を省いた)]

arXiv v1 の Figure D.2 は、symlog が R2D2 の「より複雑な非対称の平方根変換」をわずかに上回ったと書いている [Hafner DreamerV3 (2023/01), arXiv v1 Appendix D, Figure D.2]。actor 側では、リターンの 5〜95 パーセンタイルの幅の指数移動平均で割り、分母は max(1, S) とする。標準偏差による正規化は、まばらな報酬で標準偏差が 0 に近いと失敗しうると原典は述べている [Hafner DreamerV3 (2023/01), "Actor learning"]。

**Stop Regressing は、DQN の MSE 回帰を HL-Gauss の交差エントロピーに置き換え、Atari の 200M frames で大きく上回ったと報告した。** ただし全文に "clip" が 1 件も出てこず、報酬処理は明記されていない(本調査の全文検索)。

```
Following the setup of Mnih et al. (2015), we train DQN for 200M frames with the aforementioned losses. We report aggregated human-normalized IQM performance and optimality gap across 60 Atari games in Figure 4. Observe that HL-Gauss substantially outperforms the Two-Hot and MSE losses. Interestingly, HL-Gauss also improves upon categorical distributional RL

Mnih et al. (2015) の設定にならい、前述の損失で DQN を 200M フレーム学習する。60 の Atari ゲームにわたる人間正規化 IQM と optimality gap の集約を図 4 に報告する。HL-Gauss は Two-Hot と MSE の損失を大きく上回る。興味深いことに、HL-Gauss はカテゴリ分布 RL も改善する
```

[Farebrother StopRegressing (2024/03), Section 4.1]

同論文は、two-hot 交差エントロピー損失が MuZero・DreamerV3・Muesli などで分析なしに使われてきたとまとめている [Farebrother StopRegressing (2024/03), Section 6]。設定は 51 locations、[v_min, v_max] = [−10, 10]、σ/ς = 0.75、γ = 0.99、n = 1 である [Farebrother StopRegressing (2024/03), Appendix B.1]。

### 2.6 報酬クリップを残している系統

**Rainbow は、ベースラインとの公平な比較のためにクリップを踏襲したと明記し、Pop-Art ならクリップを外しても同程度の性能を保てると書いている。** Rainbow 自身はクリップの有無を比べていない。

```
To evaluate Rainbow fairly against the baselines, we have followed the common domain modifications of clipping rewards, fixed action-repetition, and frame-stacking, but these might be removed by other learning algorithm improvements. Pop-Art normalization (van Hasselt et al. 2016) allows reward clipping to be removed, while preserving a similar level of performance.

Rainbow をベースラインと公平に比較するため、報酬のクリップ、固定の行動反復、フレームスタックという一般的なドメイン改変を踏襲したが、これらは他の学習アルゴリズムの改良によって取り除けるかもしれない。Pop-Art 正規化(van Hasselt et al. 2016)を使えば、同程度の性能を保ったまま報酬クリップを取り除ける。
```

[Hessel Rainbow (2017/10), Discussion]

2023 年以降の単一 GPU・サンプル効率重視のエージェントも、クリップを残している。

- BTR の表は Reward Clipping [−1, 1]、γ 0.997、n-step 3 である [Clark BTR (2024/11), Table D4, D6]。
- EfficientZero の表は Reward clipping True で、公式実装はクリップに加えて h と 601 出力の分類型を使う [Ye EfficientZero (2021/11), Appendix A.1 Table 6]、[EfficientZero code (2022/08)]。
- BBF と PQN は論文に報酬処理の記述がない。公式実装はクリップしている(§6.3)。

## 3. 報酬クリップの有無を統制した比較

### 3.1 比較研究の一覧

**同じエージェント・同じ予算でクリップの有無を比べた一次資料は、DQN 系、actor-critic 系、PPO で見つかった。** 下表は本調査による整理で、各行の根拠は §3.2〜§3.4 の引用にある。

| 研究 | エージェント | 予算 | ゲーム / seed | 比べた条件 | 集計の結果 |
|---|---|---|---|---|---|
| van Hasselt 2016 | Double DQN | 200M frames | 57 / 記載なし | クリップあり、クリップなし(スコア未報告)、クリップなし + Pop-Art | Pop-Art が 32/57 ゲームで同等以上、中央値差 +0.4%、平均差 +34% |
| Hessel 2018(PopArt-IMPALA) | IMPALA(マルチタスク) | 合計 1.14×10¹⁰ frames | 57(1 エージェント)/ PBT | IMPALA と PopArt-IMPALA をクリップあり/なしで | IMPALA は 59.7% → 0.3%、PopArt-IMPALA は 110.7% → 107.0%(中央値) |
| Hessel 2019(Inductive Biases) | A2C | 200M frames | 57 / 8 | クリップ、PopArt、生の報酬 | クリップ > PopArt > 生の報酬。クリップが有意な差を保った |
| Schaul 2021(1 ヘッド) | R2D2 系(Adam) | 約 10⁹ frames | 57 / 1 | 生の報酬、return-based scaling、Pop-Art、クリップ、h 変換 | 集計で最良はクリップ。生の報酬も崩れない |
| Kapturowski 2018(R2D2) | R2D2 | 2.0×10⁶ learner updates | 5 / 3 | h 変換(クリップなし)と、クリップ(h なし) | 効果はまちまち |
| Pohlen 2018 | Ape-X DQfD | 140 h | 6 / 記載なし | h 変換と標準の作用素(クリップ条件の記載なし) | 標準の作用素は安定だが有意に劣る |
| Sullivan 2023 | PPO | 25M steps(節の題は Atari 100M) | 57 / 3 | クリップの有無 × DreamerV3 の工夫 | 工夫入りのクリップなしは、クリップ付きをわずかに下回る |
| de la Cruz 2018 | A3C | 200M frames | 6 / 4 trials | クリップと、クリップなし + h 変換 | h 変換が 5/6 ゲームで上回る |

### 3.2 正規化も変換もしない生の報酬では集計値が崩れやすい

**Double DQN でクリップを外すと、勾配ノルムの中央値がゲーム間で 6 桁以上にわたり、単一の step size を選べなくなる。** 生の報酬の条件は走らせているが、スコアは報告されていない。

```
Without clipping the rewards, Pop-Art produces a much narrower band within which the gradients fall. Across games, 95% of median norms range over less than two orders of magnitude (roughly between 1 and 20), compared to almost four orders of magnitude for clipped Double DQN, and more than six orders of magnitude for unclipped Double DQN without Pop-Art. The wide range for the latter shows why it is impossible to find a suitable step size with neither clipping nor Pop-Art: the updates are either far too small on some games or far too large on others.

報酬をクリップしなくても、Pop-Art では勾配の収まる帯がずっと狭くなる。ゲーム全体で、ノルムの中央値の 95% は 2 桁未満(おおよそ 1 から 20)の範囲に収まる。これに対し、クリップした Double DQN はほぼ 4 桁、Pop-Art なしのクリップしない Double DQN は 6 桁を超える。後者の範囲の広さは、クリップも Pop-Art も使わない場合に適切な step size を見つけることが不可能な理由を示している。更新は、あるゲームでは小さすぎ、他のゲームでは大きすぎる。
```

[vanHasselt PopArt (2016/02), Section 4]

**マルチタスクの IMPALA は、クリップを外すと人間正規化スコアの中央値が 59.7% から 0.3% に落ちた。** PopArt-IMPALA は両条件でほぼ同じだった。

```
The IMPALA agent (blue line) performs much worse. The baseline barely reaches 60% with reward clipping, and the median performance is close to 0% in the unclipped setup.

IMPALA エージェント(青の線)ははるかに悪い。ベースラインは報酬クリップありでかろうじて 60% に達し、クリップなしの設定では中央値の性能は 0% に近い。
```

[Hessel PopArtIMPALA (2018/09), "Atari-57 results"]

Table 1 の値は、IMPALA がクリップあり 59.7%(random starts)/ 28.5%(human starts)、クリップなし 0.3% / 1.0% である [Hessel PopArtIMPALA (2018/09), Table 1]。

**A2C では、生の報酬の条件が 3 条件の最下位だった。RMSProp と勾配ノルムのクリップを併用しても、この結果は変わらなかった。**

```
Again, the naive solution performed very poorly, compared to using either the domain heuristic or the learned solution. Note that the naive solution is using RMSProp as an optimizer, in combination with gradient clipping by norm (Pascanu et al., 2012); together these techniques should provide at least some robustness to scaling issues, but in our experiments PopArt provided an additional large increase in performance.

ここでも素朴な解は、ドメインのヒューリスティックや学習による解に比べて非常に成績が悪かった。素朴な解は最適化手法として RMSProp を使い、ノルムによる勾配クリップ(Pascanu et al., 2012)と組み合わせている点に注意されたい。これらを組み合わせれば少なくともある程度はスケールの問題に頑健になるはずだが、我々の実験では PopArt がさらに大きな性能向上をもたらした。
```

[Hessel InductiveBiases (2019/07), Section 3.2]

図 2c の終端の値は【目視】で、クリップ約 0.93、PopArt 約 0.69、生の報酬約 0.28 である。PPO でも同じ傾向で、25M steps 時点の人間正規化スコアの中央値は【目視】で、クリップなし約 0.20、クリップあり約 0.67 だった [Sullivan PPOTricks (2023/10), Figure 1(b)]。

**例外は R2D2 系の単一ヘッドで、生の報酬でも集計値は崩れなかった。** 著者は、主に働いている仕組みを適応的最適化手法の Adam だとしている。

```
Perhaps surprisingly, in the 1-head case, return-based scaling does not lead to a meaningful performance difference. The main reason for this is likely that as a high-performing agent on Atari, the unscaled baseline must have been designed and tuned to handle the scale differences sufficiently well. In fact, the main mechanism at work here is the adaptive optimiser Adam (Kingma & Ba, 2014): we look at this in more depth in Appendix B.

意外かもしれないが、1 ヘッドのケースでは return-based scaling は意味のある性能差を生まない。主な理由はおそらく、Atari で高性能なエージェントである unscaled のベースラインが、スケールの違いに十分対処できるよう設計・調整されていたはずだという点にある。実際、ここで働いている主な仕組みは適応的最適化手法の Adam(Kingma & Ba, 2014)であり、これは付録 B で詳しく見る。
```

[Schaul ReturnScaling (2021/05), Section 4.2]

この設定は R2D2 の JAX 実装、γ = 0.997、Adam、勾配クリップなし、Huber 損失なしで、予算は約 10⁹ frames、各ゲーム 1 seed である [Schaul ReturnScaling (2021/05), Appendix A.2, Table 1]。

### 3.3 代替手段を使うと集計ではクリップとほぼ同等かやや劣る

**Double DQN に Pop-Art を足してクリップを外すと、集計ではクリップ版とほぼ同じか、わずかに上回った。**

```
On the whole, the results show that with Pop-Art we can successfully remove the clipping heuristic that has been present in all prior DQN variants, while retaining overall performance levels. Double DQN with Pop-Art performs slightly better than Double DQN with clipped rewards: on 32 out of 57 games performance is at least as good as clipped Double DQN and the median (+0.4%) and mean (+34%) differences are positive.

全体として、結果は、それまでのすべての DQN 変種にあったクリップのヒューリスティックを、全体の性能水準を保ったまま Pop-Art で取り除けることを示している。Pop-Art 付きの Double DQN は、クリップした報酬の Double DQN よりわずかに良い。57 ゲーム中 32 ゲームで性能はクリップした Double DQN と同等以上で、中央値(+0.4%)と平均(+34%)の差はいずれも正である。
```

[vanHasselt PopArt (2016/02), Section 4]

この比較では、クリップ版の Double DQN のスコアを Wang et al. 2016 から転載しており、同じ論文の中で走らせたものではない [vanHasselt PopArt (2016/02), Appendix Table 1 caption]。また TD 誤差のクリップも同時に外している。

PopArt-IMPALA の中央値は、クリップありで 110.7%、クリップなしで 107.0%(random starts)だった。本文は後者を 101% と書いており、表と一致しない理由は書かれていない [Hessel PopArtIMPALA (2018/09), Table 1, "Atari-57 results"]。

**A2C では、クリップが PopArt に対して有意な優位を保った。** 著者は、調べたドメイン固有のヒューリスティックのうち有意差が残ったのはクリップだけだったと書き、クリップの利点は報酬スケール以外にもある可能性を挙げている。

```
In this case, the domain heuristic (reward clipping) retained a significant edge over the adaptive solution.
[...]
This suggests that reward clipping might not be helping exclusively with reward scales; the inductive bias of optimizing for a weighted frequency of rewards is a very good heuristic in many Atari games, and the qualitative behaviour resulting from optimizing the proxy objective might result in a better learning dynamics.

この場合は、ドメインのヒューリスティック(報酬クリップ)が適応的な解に対して有意な優位を保った。
[...]
このことは、報酬クリップが報酬スケールだけに効いているのではない可能性を示している。重み付けされた報酬の頻度を最適化するという帰納バイアスは多くの Atari ゲームで非常によいヒューリスティックであり、代理目的を最適化した結果として生じる定性的な振る舞いが、より良い学習ダイナミクスにつながっているのかもしれない。
```

[Hessel InductiveBiases (2019/07), Section 3.2]

**R2D2 系の単一ヘッドでも、集計で最良だったのはクリップだった。**

```
Single-head Atari performance across various scaling methods. The best aggregate performance is attained by reward clipping (cyan), a method that does not preserve optimality (and is highly detrimental in some games) yet seems to be a surprisingly beneficial heuristic.

さまざまなスケーリング手法での単一ヘッドの Atari 性能。集計で最も良い性能を達成するのは報酬クリップ(シアン)である。これは最適性を保たない(そして一部のゲームでは大きく有害な)手法だが、驚くほど有益なヒューリスティックであるように見える。
```

[Schaul ReturnScaling (2021/05), Figure 12 caption]

学習終了時点の値は【目視】で、人間正規化スコアの平均がクリップ約 21.6 対生の報酬約 10.0、中央値が約 3.05 対約 2.7 である(1 = 人間)。h 変換の条件(non-linear V)の中央値も約 3.05 だった [Schaul ReturnScaling (2021/05), Figure 12]。本文は、クリップが他の方法に比べて平均スコアを大きく押し上げると書いている [Schaul ReturnScaling (2021/05), Section 4.3]。

**PPO でも、DreamerV3 の工夫を入れたクリップなしはクリップ付きをわずかに下回った。**

```
This suggests that the tricks make PPO significantly more robust to varying reward scales, though they slightly underperform a simple reward clipping baseline.

これは、工夫によって PPO がさまざまな報酬スケールに対して大幅に頑健になることを示唆する。ただし、単純な報酬クリップのベースラインはわずかに下回る。
```

[Sullivan PPOTricks (2023/10), Section 4.2]

**h 変換とクリップを直接比べた R2D2 と A3C の結果は、ゲームによって割れた。**

- R2D2 のアブレーション(5 ゲーム、3 seeds)の「Clipped」は、h 変換を外して報酬クリップに置き換えた条件で、クリップと h の有無が同時に変わる。著者はこれを含む LSTM 以外の設計選択の効果を「まちまち」と書いている [Kapturowski R2D2 (2018/09), Section 5, Figure 7]。
- 同じ図の【目視】では、h 変換版が MsPacman(約 41,000 対 約 26,000)と SeaQuest(約 1,000,000 対 約 140,000)で上回り、Breakout と Gravitar はほぼ同じだった。
- A3C では、クリップなし + h 変換の A3C-TB が 6 ゲーム中 5 ゲームで上回った。Pong だけは、h 変換が報酬値を縮めて報酬の伝播を遅らせたため悪化した、と著者は説明している [delaCruz A3CTB (2018/12), Section 6]。

### 3.4 ゲーム別の結果は割れる

**クリップを外すと、ゲームごとの勝ち負けが大きく入れ替わる。** van Hasselt らは、改善した例として Gopher と Centipede を、悪化した例として Video Pinball と Star Gunner を挙げている。

```
The main eye-catching result is that the distribution in performance drastically changed. On some games (e.g., Gopher, Centipede) we observe dramatic improvements, while on other games (e.g., Video Pinball, Star Gunner) we see a substantial decrease. For instance, in Ms. Pac-Man the clipped Double DQN agent does not care more about ghosts than pellets, but Double DQN with Pop-Art learns to actively hunt ghosts, resulting in higher scores.

最も目を引く結果は、性能の分布が劇的に変わったことである。いくつかのゲーム(例: Gopher、Centipede)では劇的な改善が見られ、別のゲーム(例: Video Pinball、Star Gunner)では大幅な低下が見られる。たとえば Ms. Pac-Man では、クリップした Double DQN エージェントは幽霊を餌より重視しないが、Pop-Art 付きの Double DQN は幽霊を積極的に狩ることを学び、より高いスコアに至る。
```

[vanHasselt PopArt (2016/02), Section 4]

同論文は、クリップを外すと問題の性質が変わって成績が下がる例として Time Pilot を挙げ、理由の 1 つを γ = 0.99 の近視眼性に求めている。

```
Some games fare worse with unclipped rewards because it changes the nature of the problem. For instance, in Time Pilot the Pop-Art agent learns to quickly shoot a mothership to advance to a next level of the game, obtaining many points in the process. The clipped agent instead shoots at anything that moves, ignoring the mothership. However, in the long run in this game more points are scored with the safer and more homogeneous strategy of the clipped agent. One reason for the disconnect between the seemingly qualitatively good behavior combined with lower scores is that the agents are fairly myopic: both use a discount factor of γ = 0.99, and therefore only optimize rewards that happen within a dozen or so seconds into the future.

クリップしない報酬では問題の性質が変わるため、成績が悪くなるゲームもある。たとえば Time Pilot では、Pop-Art のエージェントは母艦をすばやく撃って次のレベルへ進むことを学び、その過程で多くの点を得る。一方、クリップしたエージェントは動くものなら何でも撃ち、母艦を無視する。しかしこのゲームでは、長い目で見ると、クリップしたエージェントのより安全で均質な戦略のほうが多くの点を取る。定性的には良さそうな振る舞いとスコアの低さが食い違う理由の 1 つは、エージェントがかなり近視眼的なことである。どちらも割引率 γ = 0.99 を使うので、せいぜい十数秒先までに起こる報酬しか最適化しない。
```

[vanHasselt PopArt (2016/02), Section 4]

注: 原文の "ignoring the mothership." の直後にある脚注記号は省いた。

Double DQN の生スコア表の抜粋(評価は 30 分 = 108,000 frames のプレイ)を示す。クリップ版は Wang et al. 2016 からの転載値である。

| ゲーム | DDQN(クリップ) | DDQN + Pop-Art(クリップなし) | 大きいほう |
|---|---|---|---|
| Ms. Pacman | 2,711 | 4,964 | Pop-Art |
| Gopher | 14,841 | 56,218 | Pop-Art |
| Centipede | 5,409 | 49,066 | Pop-Art |
| Frostbite | 1,683 | 3,470 | Pop-Art |
| Bowling | 68 | 102 | Pop-Art |
| Video Pinball | 309,942 | 56,287 | クリップ |
| Star Gunner | 60,142 | 589 | クリップ |
| Time Pilot | 8,339 | 4,870 | クリップ |
| Wizard of Wor | 7,492 | 483 | クリップ |
| Battle Zone | 31,700 | 8,220 | クリップ |
| Q*Bert | 15,089 | 5,237 | クリップ |
| Skiing | −9,022 | −13,585 | クリップ |

[vanHasselt PopArt (2016/02), Appendix Table 1](小数点以下を四捨五入して抜粋)

R2D2 系の単一ヘッド(1 seed、約 10⁹ frames)で、クリップと生の報酬を最終付近で比べた【目視】の値を示す。

| ゲーム | クリップ | 生の報酬 | 大きいほう |
|---|---|---|---|
| Bowling | 約 30 | 約 200 | 生の報酬 |
| Centipede | 約 1.5〜2 万 | 終盤 約 6〜11 万 | 生の報酬 |
| Ms. Pac-Man | 約 7,900 | 約 12,400 | 生の報酬 |
| Bank Heist | 約 1,500 | 約 6,000〜8,000 | 生の報酬 |
| Alien | 約 16,000 | ほぼ 0 | クリップ |
| Kung-Fu Master | 約 165,000 | ほぼ 0 | クリップ |
| Video Pinball | 約 40〜90 万(変動大) | 下端付近 | クリップ |
| Skiing | 約 −9,000 | 約 −30,000 | クリップ |

[Schaul ReturnScaling (2021/05), Figure 15](線の色は凡例と画素の色で判別した)

PopArt-IMPALA のクリップありとクリップなしを比べた【画素換算】の値(人間正規化 %)は、Gopher が約 610 対約 2,100、Centipede が約 32 対約 170、Kangaroo が約 130 対約 460 でクリップなしが上だった。Demon Attack は約 4,100 対約 540、Video Pinball は約 510 対約 75、Time Pilot は約 420 対約 61、Frostbite は約 53 対約 5 でクリップが上だった。Bowling はどちらも約 30 だった [Hessel PopArtIMPALA (2018/09), Figure 8]。

本調査による突き合わせは次のとおりである。

- Video Pinball・Star Gunner・Time Pilot・Q*Bert は、Double DQN の表でも PopArt-IMPALA の図でもクリップ側が高い。
- Centipede・Gopher・Ms. Pac-Man は、両方でクリップなし側が高い。
- Frostbite は両者で向きが逆である。
- Skiing は Schaul らが「クリップすると解けなくなる例」に挙げているが(§1.3)、Double DQN の表でも R2D2 系の図でも、クリップ側のスコアのほうが高い。

A2C の著者も、集計ではクリップが上だった一方、クリップしたエージェントが準最適な方策にはまっていた Centipede などでは PopArt が有意に良かったと書いている [Hessel InductiveBiases (2019/07), Section 3.2]。

### 3.5 DQN 系で約 200M frames のままクリップだけを外した A/B は見つからなかった

**「Atari・DQN 系・約 200M frames・クリップだけを外す」の条件をすべて満たす比較は、本調査の範囲では見つからなかった。** 近い候補と、条件の違いは次のとおりである(本調査の整理)。

- van Hasselt 2016 は、DQN 系・200M frames・57 ゲームの 3 点がそろう唯一の例である。クリップを外しただけの条件も走らせているが、報告は勾配ノルムの分布だけでスコアはない。スコアを報告した比較は Pop-Art 併用で、クリップ版のスコアは他の論文からの転載である(§3.2、§3.3)。
- Schaul 2021 は、クリップだけを変えた DQN 系の比較として最も近い。ただし予算は約 10⁹ frames で 5 倍、seed は各ゲーム 1 本、基準は h 変換なしで Adam を使う R2D2 である(§3.2)。
- Hessel 2019 は、クリップだけを変えた比較として予算(200M frames)とゲーム数(57)が一致し、seed も 8 本ある。ただしエージェントは A2C である(§3.2)。
- Stop Regressing は DQN・200M frames・60 ゲームだが、クリップの比較を含まない(§2.5)。Lyle らの可塑性喪失の研究も、全文に "clip" が出てこない [Lyle PlasticityCauses (2024/02)]。
- 2025〜2026 年の研究で、Atari の報酬クリップの有無を統制して比べたものは、検索した範囲では見つからなかった。

第三者の情報としては、LightZero(MuZero 実装)のメンテナが GitHub issue で、MsPacman でクリップなしは有意には良くないと報告している [LightZero issue233 (2024/07)]。添付画像の凡例では各ラン 1 本(seed0)で、学習長もランごとに異なる。査読を経ておらず、条件も統制されていない。

## 4. 割引率 γ の設定と高い γ での h(x) の効き方

### 4.1 Atari で使われる γ とその選定根拠

**0.99 は DQN の値で、γ を他の値と体系的に比べて選んだことを示す一次情報は見つからなかった。** DQN は全ハイパーパラメータを 5 ゲームでの非公式な探索で選んだと書いている。

```
The values of all the hyperparameters and optimization parameters were selected by performing an informal search on the games Pong, Breakout, Seaquest, Space Invaders and Beam Rider. We did not perform a systematic grid search owing to the high computational cost.

すべてのハイパーパラメータと最適化パラメータの値は、Pong、Breakout、Seaquest、Space Invaders、Beam Rider で非公式な探索を行って選んだ。計算コストが高いため、体系的なグリッドサーチは行わなかった。
```

[Mnih DQN (2015/02), Methods "Training details"]

Rainbow は DQN と同じ値だと明記している [Hessel Rainbow (2017/10), Table 4]。DQN 以前の ALE の SARSA(λ) ベースラインは、パラメータ探索の結果として γ = 0.999 を使っていた [Bellemare ALE (2012/07), Section 3.1.2, Appendix C]。Pohlen らは 0.99 を「経験的に全ゲームで安定学習できた最大値だった」と位置づけているが、この段落に引用文献はない。

```
The majority of approaches use a discount factor of γ = 0.99. Empirically, this used to be the highest discount factor that allows stable learning on all games. However, the TC loss allows us to use a much higher discount factor of γ = 0.999 giving the algorithm an effective planning horizon of 1000 instead of 100 steps.

大半の手法は割引率 γ = 0.99 を使う。経験的には、これがすべてのゲームで安定に学習できる最も高い割引率だった。しかし TC 損失により、はるかに高い γ = 0.999 を使えるようになり、アルゴリズムの実効的な計画ホライズンは 100 ステップではなく 1000 ステップになる。
```

[Pohlen ApeXDQfD (2018/05), Appendix C]

**0.997 は R2D2 が理由を書かずに導入し、以後は「先行研究と同じ」として引き継がれた。**

```
Finally, compared to Ape-X, we used the slightly higher discount of γ = 0.997, and disabled the loss-of-life-as-episode-end heuristic that has been used in Atari agents in some of the work since (Mnih et al., 2015).

最後に、Ape-X と比べてわずかに高い割引率 γ = 0.997 を用い、(Mnih et al., 2015) 以降の一部の研究の Atari エージェントで使われてきた「ライフ喪失をエピソード終了とする」ヒューリスティックを無効にした。
```

[Kapturowski R2D2 (2018/09), Section 2.3]

以後の論文は、次のように先行研究に合わせたと書いている。

- MuZero は R2D2 と同じ割引率(0.997)と価値変換を使うと書いている [Schrittwieser MuZero (2019/11), Appendix C]。MuZero Unplugged と Muesli の大規模設定も先行研究に従ったと書いている [Schrittwieser MuZeroUnplugged (2021/04), Appendix F]、[Hessel Muesli (2021/04), Table 6]。
- BBF はスケジュールの終点を MuZero と EfficientZero に合わせた(§5.2)。ただし EfficientZero の表の値は 0.997⁴ と表記されている [Ye EfficientZero (2021/11), Table 6]。
- BTR は MuZero Reanalyse に従うと書き、次の定性的な理由を添えている。

```
For many years, RL algorithms have used a discount rate of 0.99, however, when reaching high performance, lower discount rates alter the optimal policy, causing even optimally performing agents to not collect the maximum cumulative rewards. To prevent this, we follow MuZero Reanalyse (Schrittwieser et al., 2021) using γ = 0.997.

長年、RL アルゴリズムは割引率 0.99 を使ってきた。しかし高い性能に達すると、低い割引率は最適方策を変えてしまい、最適に振る舞うエージェントでさえ累積報酬の最大値を集められなくなる。これを防ぐため、MuZero Reanalyse(Schrittwieser et al., 2021)にならい γ = 0.997 を使う。
```

[Clark BTR (2024/11), Section 3.2]

**0.997 を超える γ は、γ の族として使われている。** NGU は 0.99〜0.997、Agent57 は 0.99〜0.9999、MEME は 0.97〜0.9997 である。

| 手法 | 族の数 | γ の範囲 | 選定根拠の記述 | 出典 |
|---|---|---|---|---|
| NGU | 32 | 0.99〜0.997(1−γ の対数空間で等間隔) | 探索的な方策は内的報酬が密なので低い γ でよい、という定性的な理由 | [Badia NGU (2020/02), Section 3, Appendix A] |
| Agent57 | 32 | 0.99〜0.9999 | 「より高い値を許すため」 | [Badia Agent57 (2020/03), Section 4, Appendix G.1] |
| MEME | 16 | 0.97〜0.9997 | ハイパーパラメータ全体を 8 ゲームで調整したという一般的な記述のみ | [Kapturowski MEME (2022/09), Section 5, Table 2] |

NGU と Agent57 のハイパーパラメータ表には "Discount r^e 0.997" の行もあり、本文の γ 族の説明とどう対応するかは書かれていない [Badia NGU (2020/02), Table 6]、[Badia Agent57 (2020/03), Appendix G.3 Table 3]。

PPO 系の RND は、Montezuma's Revenge のアブレーションで外的報酬側 γ_E = 0.999、内的報酬側 γ_I = 0.99 を選んだ。γ_E を 0.999 に上げると大きく改善し、γ_I を 0.999 に上げると悪化した。外的報酬は [−1, 1] にクリップしている [Burda RND (2018/10), Section 3.3, Appendix A.3 Table 2]。

### 4.2 高い γ で何が問題になるか

**Pohlen らは、h を使っても γ が 1 に近づくと不安定になりうると明記している。** 理由として、報酬のない状態の間の価値の差が小さくなり、隣接する目標値が似ているためにネットワークが次状態へ望まない汎化をすることを挙げている。

```
While the transformed Bellman operator provides an atemporal reduction of the target's scale and variance, instability can still occur as the discount factor γ approaches 1. Increasing the discount factor decreases the temporal difference in value between non-rewarding states. In particular, unwanted generalization of the neural network f_θ to the next state x′ (due to the similarity of temporally adjacent target values) can result in catastrophic TD backups.

変換ベルマン作用素は目標のスケールと分散を時間に依存しない形で減らすが、割引率 γ が 1 に近づくと、それでも不安定が起こりうる。割引率を上げると、報酬のない状態の間の価値の時間差が小さくなる。特に、(時間的に隣接する目標値が似ているために)ニューラルネットワーク f_θ が次状態 x′ へ望まない汎化をすると、破滅的な TD 更新が生じうる。
```

[Pohlen ApeXDQfD (2018/05), Section 3.3]

**他の原典は、推定・表現・分散の面から高い γ の問題を挙げている。** ただし多くは Atari の深層 RL ではなく、表形式・線形・MuJoCo などで検証された主張である。

- Jiang らは、推定したモデルで計画する表形式の設定で、計画ホライズンが方策クラスの複雑さを制御するパラメータだと示した。データが少ないと短いホライズンのほうが良くなりうる [Jiang PlanningHorizon (2015/05), Abstract, Section 4]。

```
We show formally that the planning horizon is a complexity control parameter for the class of policies to be learned.

計画ホライズンが、学習される方策クラスの複雑さを制御するパラメータであることを形式的に示す。
```

[Jiang PlanningHorizon (2015/05), Abstract]

- Lehnert らは、計画ホライズンが長いと action gap(最良行動と次善行動の価値の差)が崩壊しうると示した [Lehnert LongHorizon (2018/02), Section 3.2]。
- Agent57 の著者は、Skiing で高い割引率を使うとリターンの分散が高くなり、学習に多くのデータが必要になると書いている。

```
To be able to achieve such performance on Skiing, Agent57 uses a high discount (as we show in Sec. 4.4). This naturally leads to high variance in the returns, which leads to needing more data in order to learn to play the game.

Skiing でこのような性能を達成するために、Agent57 は高い割引を用いる(Sec. 4.4 で示す)。これは当然リターンの高い分散を招き、ゲームのプレイを学ぶのにより多くのデータを必要とすることにつながる。
```

[Badia Agent57 (2020/03), Section 4.1]

- Amit らは、TD 学習で γ を下げることが、報酬と学習率を所定の倍率にする条件のもとで、価値推定の二乗を罰する正則化と等価になると示した。実験は MuJoCo で Atari は扱っていない [Amit DiscountRegularizer (2020/07), Section 3.1]。Petrik と Scherrer は、近似動的計画法(Tetris)で人為的に低い割引率のほうが解の質が上がることを示した [Petrik LowerDiscount (2008/12), Abstract]。
- François-Lavet らは、DQN で γ を 0.99 より上げると過大評価が深刻になって成績が大きく落ちたと報告している(§5.1)。Tang らは、MuJoCo の方策勾配で γ ≥ 0.999 は実践ではめったにうまくいかないと書いている [Tang TaylorDiscount (2021/06), Section 1]。

### 4.3 h(x) と 0.997 超の γ を組み合わせた実験

**γ = 0.999 の Ape-X DQfD では、h に加えて補助損失(TC loss)が安定な学習に必要だった。** TC loss は、ネットワークが未見の状態へ早まって汎化するのを防ぐ補助損失である [Pohlen ApeXDQfD (2018/05), Section 1, 3.3]。これを外すと、学習途中で性能が崩壊した。

```
In our setup, the TC loss is paramount to learning stably. We see that without the TC loss the algorithm learns faster at the beginning of the training process. However, at some point during training, the performance collapses and often the process dies with floating point exceptions.

我々の設定では、TC loss は安定した学習に最も重要である。TC loss がないと、アルゴリズムは訓練の初期にはより速く学習する。しかし訓練中のある時点で性能が崩壊し、多くの場合、プロセスは浮動小数点例外で停止する。
```

[Pohlen ApeXDQfD (2018/05), Section 4.3 "TC loss"]

同じアブレーション(6 ゲーム)では、TC loss を残して h を外すと、安定ではあるが性能は有意に悪化した。h を PopArt に置き換える版と、TC loss を constrained TD 更新に置き換える版も、どちらもうまくいかなかった [Pohlen ApeXDQfD (2018/05), Section 4.3, 4.4]。アブレーション節自体に γ の値はないが、全ハイパーパラメータの表の割引率は 0.999 で、γ を変えたという記述はない(本調査の読み取り)。なおこの設定は ε = 10⁻² で、専門家デモのデータを含む。

**γ = 0.9999 では、h 付きの損失を使う R2D2 でも、全ゲームに固定すると非常に不安定だった。** Agent57 はこれを γ の族とメタコントローラで回避した。

```
Using the meta-controller allows to include very high discount values in the set {γ_j}_{j=0}^{N}. Specifically, running R2D2 with a high discount factor, γ = 0.9999 surpasses the human baseline in the game of Skiing. However, using that hyperparameter across the full set of games, renders the algorithm very unstable and damages its end performance. All the scores in the challenging set for a fixed high discount (γ = 0.9999) variant of R2D2 are reported in App. H.1. When using a meta-controller, the algorithm does not need to make this compromise: it can adapt it in a per-task manner.

メタコントローラを使うと、集合 {γ_j}_{j=0}^{N} に非常に高い割引率を含めることができる。具体的には、高い割引率 γ = 0.9999 で R2D2 を走らせると、Skiing で人間のベースラインを超える。しかし、そのハイパーパラメータを全ゲームで使うと、アルゴリズムは非常に不安定になり、最終性能を損なう。高い割引率(γ = 0.9999)に固定した R2D2 の変種の challenging set での全スコアは App. H.1 に示す。メタコントローラを使えば、アルゴリズムはこの妥協をする必要がなく、タスクごとに適応させることができる。
```

[Badia Agent57 (2020/03), Section 4.4]

ここでの R2D2 は、変換 Retrace 損失(h 付き)を使う変種 R2D2 (Retrace) である [Badia Agent57 (2020/03), Section 3.2]。

**γ = 0.997 の R2D2 は h(ε = 10⁻³)を TC loss なしで使っている。0.99 とのアブレーションの結論は「まちまち」だった。** R2D2 の本文に "temporal consistency" や "TC loss" は出てこない(本調査の全文検索)。

```
Ablation results with standard deviations shown by shading (3 seeds). 'Clipped' refers to the agent variant using clipped rewards (instead of value function rescaling), 'discount' refers to the use of a discount value of 0.99 (instead of 0.997).

標準偏差を網掛けで示したアブレーション結果(3 seed)。「Clipped」は(価値関数の再スケーリングの代わりに)クリップした報酬を使うエージェントの変種を、「discount」は(0.997 の代わりに)割引率 0.99 を使うことを指す。
```

[Kapturowski R2D2 (2018/09), Figure 7 caption]

R2D2 の著者は、人間を超えられなかった 5 ゲームのうち 3 つ(Skiing、Solaris、Private Eye)について、より高い割引率と速いターゲット更新で人間を超えうることを「逸話的に」観察したと書いている [Kapturowski R2D2 (2018/09), Section 4.1]。

**0.997 を超える γ を使う MEME は、h に加えて TD 誤差の正規化(§2.4)を入れ、γ は 16 本の族として使っている。** TD 誤差の正規化の効果は他の改善ほど顕著ではなかったと著者は書いている [Kapturowski MEME (2022/09), Section 5]。

本調査の整理として、h だけで 0.997 を超える γ が安定したと報告した原典は見つからなかった。確認できたのは次の 3 点である。

- γ = 0.999 では TC loss が必要だった(Pohlen)。
- γ = 0.9999 の固定は不安定で、族とメタコントローラで回避した(Agent57)。
- γ = 0.997 は h で使われているが、安定性を h に帰する記述はなく、0.99 との比較は「まちまち」だった(R2D2)。

h を使わずに高い γ を使った例として、BTR(クリップ + 0.997)と RND(クリップ + γ_E = 0.999)がある。どちらも h との比較はしていない。

### 4.4 報酬クリップと γ の相互作用

**報酬クリップと γ から価値の上限を導いた原典はあるが、γ を変えてその影響を調べた比較実験は見つからなかった。** Deadly Triad の論文は、クリップ報酬と γ = 0.99 から |Q| の上限 100 を導き、それを超える値を soft divergence と定義した。

```
Because the rewards are clipped to [−1, 1], and because the discount factor γ is 0.99, the maximum absolute true value in each game is bounded by 1 + γ + γ² + . . . = 1/(1−γ) = 100 (and realistically attainable values are typically much smaller). Therefore, values for which |q| > 100 are unrealistic. We call this phenomenon soft divergence.

報酬は [−1, 1] にクリップされ、割引率 γ は 0.99 なので、各ゲームの真の価値の絶対値の最大は 1 + γ + γ² + . . . = 1/(1−γ) = 100 で抑えられる(現実に到達可能な値は通常これよりずっと小さい)。したがって |q| > 100 となる値は非現実的である。この現象を soft divergence と呼ぶ。
```

[vanHasselt DeadlyTriad (2018/12), Section 4]

本調査の計算では、同じ上限は γ = 0.997 で約 333、γ = 0.999 で 1000 になる。DreamerV3 の表も γ = 0.997 の割引ホライズン 1/(1−γ) を 333 と書いている [Hafner DreamerV3 (2023/01), Table 4]。

**Schaul らは、割引率ごとに損失のスケールが約 100 倍異なり、クリップの有無による差はさらに大きいと実測した。**

```
Note how the unscaled losses vary significantly across discounts (typically about 100x), and the difference in scales between clipped and unclipped rewards is even bigger.

スケーリングしない損失が割引率によって大きく(典型的には約 100 倍)異なり、クリップした報酬とクリップしない報酬の間のスケールの差はさらに大きいことに注目されたい。
```

[Schaul ReturnScaling (2021/05), Figure 5 caption]

同じ論文は、リターンを実効ホライズン 1/(1−γ) で割る正規化は保守的すぎることがあるとも書いている。最大誤差はホライズンとともに増えるが、典型的な誤差はしばしばずっと小さいからである [Schaul ReturnScaling (2021/05), Section 2.1]。Hessel らは、実効ホライズンを変えるとリターンの大きさが変わり、学習率に影響すると述べている。ただし行動反復の文脈で、実験では示していない [Hessel InductiveBiases (2019/07), Section 4]。LogDQN の公式実装は、ターゲットを [0, 1/(1−γⁿ)] に切っている [logrl (2019/10), log_dqn_experiments/log_dqn/log_dqn_agent.py]。

### 4.5 低い γ と価値の写像

**逆に、価値の写像で低い γ を使えるようにする研究もある。** van Seijen らは、低い γ の不振の原因は action gap が小さいことではなく、action gap の大きさが状態間で大きくばらつくことだという仮説を立て、対数写像で均した。

```
We optimized hyper-parameters using a subset of 6 games. In particular, we performed a scan over the discount factor γ between γ = 0.84 and γ = 0.99. For DQN, γ = 0.99 was optimal; for LogDQN, the best value in this range was γ = 0.96. We tried lower γ values as well, such as γ = 0.1 and γ = 0.5, but this did not improve the overall performance over these 6 games.

ハイパーパラメータは 6 ゲームの部分集合で最適化した。特に、割引率 γ を γ = 0.84 から γ = 0.99 の間で走査した。DQN では γ = 0.99 が最適で、LogDQN ではこの範囲の最良値は γ = 0.96 だった。γ = 0.1 や γ = 0.5 のようなさらに低い γ も試したが、これら 6 ゲームでの全体の性能は改善しなかった。
```

[vanSeijen LogMapping (2019/06), Section 5]

報酬を 100 倍しても、価値を 100 押し上げても性能は大きく変わらず、「action gap が小さいから」「相対的な action gap が小さいから」という 2 つの仮説は否定された [vanSeijen LogMapping (2019/06), Section 3.2]。後続の Fatemi と Tavakoli は、対数写像は毎ステップ報酬がある Skiing のような密な報酬の場面で不利になると述べている [Fatemi OrchestratedMapping (2022/03), Section 4.3]。

## 5. 学習中に γ と n-step を変えるスケジュール

### 5.1 γ を上げていく DQN の実験

**François-Lavet らは、DQN のハイパーパラメータを保ったまま、250,000 steps の epoch ごとに γ_{k+1} = 1 − 0.98(1 − γ_k) で γ を上げた。** 1 − γ を epoch ごとに 0.98 倍する幾何的な更新である。γ の初期値は本文になく、図の軸では 0.95 から始まり、20M steps で 0.99 に達する(本調査の読み取り)。

```
It can be observed that by simply using an increasing discount factor, learning is faster for four out of the five tested games and similar for the remaining game. We conclude that by starting with a low discount factor, we obtain faster policy improvement thanks to less instability.

割引率を増やすだけで、テストした 5 ゲームのうち 4 ゲームで学習が速くなり、残り 1 ゲームは同等であることがわかる。低い割引率から始めることで、不安定性が減るおかげで方策改善が速くなると結論する。
```

[FrancoisLavet HowToDiscount (2015/12), Section 4]

評価は 5 ゲーム × 各 5 seed で、報告するスコアは各シミュレーションの評価平均の最高値である。20M steps 時点の改善率は Q-bert +63%、Seaquest +49%、Enduro +17%、Beam rider +11%、Breakout −1% だった [FrancoisLavet HowToDiscount (2015/12), Section 4, Figure 2]。付録の表では、Seaquest の改善は 5 seed のうち 2 seed(11,000 台)に引っ張られている(本調査の読み取り)[FrancoisLavet HowToDiscount (2015/12), Appendix Table 1]。

**0.99 を超えて γ を上げ続けると、過大評価が深刻になって成績が大きく落ちた。**

```
It is shown that increasing γ without additional care degrades severely the score obtained beyond γ ≈ 0.99. By looking at the average V value, it can be seen that overestimation is particularly severe which causes the poor policies.

追加の対策なしに γ を増やすと、γ ≈ 0.99 を超えたところで得られるスコアが大きく劣化することが示される。平均 V 値を見ると、過大評価が特にひどく、それが劣った方策を生んでいることがわかる。
```

[FrancoisLavet HowToDiscount (2015/12), Section 4.1]

γ の増加に学習率の減衰を組み合わせた 50M steps の実験では、6 ゲームすべてで固定条件を上回った [FrancoisLavet HowToDiscount (2015/12), Section 4.2, Appendix Table 2]。一方、同じ表で γ だけを変えて学習率を固定した条件は、Seaquest・Breakout・Space Invaders の 3 ゲームで固定条件を下回っている(本調査の読み取り。この実験の seed 数は書かれていない)。著者は、低い γ は探索を減らし局所最適に陥る危険があるとも書いている [FrancoisLavet HowToDiscount (2015/12), Section 4.3]。

### 5.2 BBF の γ と n のアニール

**BBF は、ネットワークのリセットのたびに、その後の最初の 10K gradient steps で n-step を 10 から 3 へ指数的に下げる。** リセットは 40k gradient steps ごとなので、アニールの区間は replay ratio によらず訓練の 25% だと著者は書いている。

```
One of the surprising components of BBF is the use of an update horizon (n-step) that decreases exponentially from 10 to 3 over the first 10K gradient steps following each network reset. Given that we follow the schedule of D'Oro et al. 2023 and reset every 40k gradient steps, the annealing phase is always 25% of training, regardless of the replay ratio.

BBF の意外な構成要素の一つは、各ネットワークリセットの後、最初の 10K gradient steps で 10 から 3 へ指数的に減少する update horizon(n-step)を使うことである。D'Oro et al. 2023 のスケジュールに従い 40k gradient steps ごとにリセットするので、アニール区間は replay ratio によらず常に訓練の 25% となる。
```

[Schwarzer BBF (2023/05), Section 4 "Receding update horizon"]

**γ も同じ指数スケジュールで 0.97 から 0.997 へ上げる。** 根拠として挙げられているのは François-Lavet らの知見と、遅延報酬へ徐々に重みを与えるという一文である。

```
Motivated by findings that increasing the discount factor γ during learning improves performance (François-Lavet et al., 2015), we increase γ from γ1 to γ2, following the same exponential schedule as for the update horizon. Note that increasing γ has the effect of progressively giving more weights to delayed rewards. We choose γ1 = 0.97, slightly lower than the typical discount used for Atari, and γ2 = 0.997 as it is used by MuZero (Schrittwieser et al., 2021) and EfficientZero (Ye et al., 2021).

学習中に割引率 γ を増やすと性能が向上するという知見(François-Lavet et al., 2015)に動機づけられ、update horizon と同じ指数スケジュールに従って γ を γ1 から γ2 へ増やす。γ を増やすことには、遅延報酬に徐々に大きな重みを与える効果がある点に注意されたい。γ1 = 0.97(Atari で一般的な割引率よりやや低い)と、MuZero(Schrittwieser et al., 2021)と EfficientZero(Ye et al., 2021)で使われている γ2 = 0.997 を選ぶ。
```

[Schwarzer BBF (2023/05), Section 4 "Increasing discount factor"]

**公式のスコア CSV から計算すると、アニールの効果の大部分は n のスケジュールによるもので、γ のスケジュールの効果は小さい。** 下表は、リポジトリの bigger_better_faster/scores/ にあるゲーム別・run 別の CSV から、26 ゲーム(Atari 100k)の IQM を本調査で計算したものである。95% CI はゲーム層別のブートストラップ 2000 回による近似で、rliable とは実装が異なる。求めた IQM は論文 Figure 5 の中央線の位置と一致した(本調査の照合)。

| 条件(CSV の名前) | RR=2 の IQM [95% CI] | 完全な BBF 比 | RR=8 の IQM [95% CI] | 完全な BBF 比 |
|---|---|---|---|---|
| BBF | 0.940 [0.873, 1.012] | — | 1.023〜1.033(run 数の揃え方による) | — |
| BBF+γ=0.99(γ を固定) | 0.873 [0.819, 0.936] | −7.2% | 0.982 [0.933, 1.038] | −4.0% |
| BBF-Annealing(γ と n を固定) | 0.649 [0.606, 0.698] | −30.9% | 0.852 [0.781, 0.926] | −16.7% |
| BBF+n=10(n を固定) | 0.541 [0.517, 0.564] | −42.5% | 0.640 [0.605, 0.678] | −37.4% |

[BBF scores (2023/06), bigger_better_faster/scores/](本調査の計算)

両 RR とも、γ を 0.99 に固定した条件の CI は完全な BBF の CI と重なる。n = 10 に固定した条件や、アニールを丸ごと外した条件の CI は重ならない(本調査の計算)。論文は γ についても固定値より優れると書いているが [Schwarzer BBF (2023/05), Section 4]、効果量は n より明確に小さい。「BBF − Annealing」で固定した n と γ の値は、論文とコードのどちらにも明記がない。

公式コードでは、スケジュールの経過量を gradient steps で数え、重みリセットの処理の中で 0 に戻す(§6.4)。公開設定ではリセットは 3 回実行され、4 回目は「回復時間が足りない」として見送られる。このため訓練全体に占めるアニール区間は約 20% になり、論文の「25%」は 1 リセット周期の中での割合と読める(本調査の計算)。BBF は Atari 100k(100k agent steps)の設定であり、公式実装は報酬を [−1, 1] にクリップする(§6.3)。

**BTR は、γ を 0.97 から 0.997 へアニールしても有意差がなかったと報告している。** 試したのは BattleZone の 1 ゲームだけである。

```
Only testing on a single environment (BattleZone), we also tried:
• Annealing the discount rate from 0.97 to 0.997 throughout training, but found no significant difference.

単一の環境(BattleZone)でのみテストして、次も試した:
• 訓練を通じて割引率を 0.97 から 0.997 へアニールしたが、有意な差は見られなかった。
```

[Clark BTR (2024/11), Appendix H "Other Things We Tried"]

BTR の公式コードにはこのアニールのオプションが残っており、既定では無効である(§6.4)。

### 5.3 メタ勾配による γ の適応

**メタ勾配で γ を学習中に適応させる手法は、γ を方策と価値の入力に与えるかどうかで結果が大きく変わった。** Xu らは IMPALA(200M frames、57 ゲーム)で γ を適応させ、human starts の中央値を 211.9%(γ = 0.995)から 267.9% に上げた。ただし γ を入力に与えないと 183% に落ちた。

```
The human-normalised median score was only 183%, well below the IMPALA baseline with γ=0.995 (211.9%), and much worse than the full meta-gradient algorithm that includes the discount factor embedding (267.9%).

人間正規化スコアの中央値はわずか 183% で、γ=0.995 の IMPALA ベースライン(211.9%)を大きく下回り、割引率の埋め込みを含む完全なメタ勾配アルゴリズム(267.9%)よりずっと悪かった。
```

[Xu MetaGradient (2018/05), Section 3.2]

{γ, λ} を適応させると 292.9% だった [Xu MetaGradient (2018/05), Table 1]。固定 γ の IMPALA は 0.995 で 211.9%、0.998 で 208.5%、0.999 で 114.9% で、0.999 が最も悪かった [Xu MetaGradient (2018/05), Appendix C, Table 3]。

**STAC では、γ だけを自己調整しても改善しなかった。** 中央値は 243% から 240% になり、主ヘッドの γ は外側の損失の値 0.995 に収束した。著者は、適応が起きていることは、それが役立つことを意味しないと明記している。

```
We emphasize that these observations imply that adaptivity happens in self-tuning agents. It does not imply that this adaptivity is directly helpful.

これらの観察は自己調整エージェントで適応が起きていることを意味すると強調しておく。この適応が直接役に立つことを意味するわけではない。
```

[Zahavy STAC (2020/02), Section 4.3]

γ だけの自己調整の値は [Zahavy STAC (2020/02), Figure 2(b), Section 9.5] による(57 ゲーム、3 seeds、200M frames)。A2C で割引率をメタ勾配で学習した Hessel らは、学習した γ が調整済みの固定値をわずかに上回ったと書いている [Hessel InductiveBiases (2019/07), Section 3.2]。学習されたホライズン (1−γ)⁻¹ は、Robotank などでは約 1000 以上に伸び、Surround などでは時間とともに縮んだ [Hessel InductiveBiases (2019/07), Appendix B]。単調に上がっていくわけではない。

### 5.4 γ の異なる方策群から選ぶメタコントローラ

**Agent57 は、γ の異なる方策の族のうち、どれで行動するかをエピソードごとにバンディットで選ぶ。** 各価値関数は固有の γ_j を持つので、学習ターゲットの γ を連続的に変える仕組みではない(本調査の読み取り)。

```
At the beginning of each episode, say, the k-th episode, the meta-controller chooses an arm J_k setting which policy will be executed.

各エピソード(例えば k 番目のエピソード)の開始時に、メタコントローラは実行する方策を決めるアーム J_k を選ぶ。
```

[Badia Agent57 (2020/03), Section 3.2]

メタコントローラを R2D2 に載せると、10 ゲームの challenging set で最終性能(CHNS)が 20% 近く上がった [Badia Agent57 (2020/03), Section 4.4]。Gravitar、Crazy Climber、Beam Rider、Jamesbond では、初めは低い割引率の探索的な方策が選ばれ、訓練が進むと高い割引率の方策へ移った。Skiing では高い割引率がすぐに選ばれた [Badia Agent57 (2020/03), Section 4.4]。MEME はこの仕組みを N = 16 で引き継ぎ、バンディットのハイパーパラメータはあまり重要でないようだったと書いている [Kapturowski MEME (2022/09), Appendix E]。メタコントローラそのものを外すアブレーションは、MEME の本文と付録には見当たらない(本調査の確認範囲)。

### 5.5 n-step と λ を変える研究

**データの古さに応じて n を下げる EfficientZero の dynamic horizon は、外すと 8 ゲーム中 7 ゲームで性能が下がった。** 古い軌跡ほど n を短くし、その先は現在の方策で探索し直した根の価値でブートストラップする。

```
More specifically, we propose to use rewards of a dynamic horizon l from the old trajectory, where l < k and l should be smaller if the trajectory is older. This reduces the policy divergence by fewer rollout steps.

より具体的には、古い軌跡から dynamic horizon l の報酬を使うことを提案する。ここで l < k であり、軌跡が古いほど l を小さくすべきである。これによりロールアウトのステップ数が減り、方策の乖離が小さくなる。
```

[Ye EfficientZero (2021/11), Section 4.3]

Breakout だけは外した方が高かった(427.0 対 388.8)[Ye EfficientZero (2021/11), Appendix A.2, Table 9]。これは訓練時間に対するスケジュールではなく、サンプルごとにデータの古さで n を決める方式である。

**n や λ を動的に選ぶ他の手法は、手で選んだ最良の固定値を上回らなかった。** Daley と Amato の動的 λ 選択(21 個の λ-リターンの中央値)は、3-step のベースライン以上にはなったが、最良の固定 λ は上回らなかった [Daley ReconcilingLambda (2018/10), Section 5]。

**n とリセットや replay 容量の相互作用を示した研究がある。** どちらも n を固定した比較である。

- Fedus らは、replay 容量を増やして性能が上がるのは n-step を含むエージェントだけで、Rainbow から n-step を外すと改善が消えたと報告した [Fedus ReplayFundamentals (2020/07), Section 3.2, 4.1]。
- Nikishin らは、SPR でリセットの効果は n が大きいほど大きかった(n = 20 で最大 40%、n = 3 では改善なし)と報告した [Nikishin PrimacyBias (2022/05), Section 5.4]。

### 5.6 スケジュールの理論的な説明

**n を下げていく理由として原典が挙げる理論は、Kearns と Singh の誤差上界である。** 大きい k(n)は収束が速いが漸近誤差が大きく、時間とともに最適な k は小さくなる。対象は方策評価の phased TD で、深層 RL の制御を直接扱うものではない。

```
For moderate values of t, values of k that are too small suffer from their overemphasis on a still-inaccurate value function approximation, while values of k that are too large suffer from their refusal to bootstrap. Of course, as t increases, the optimal value of k decreases, since small values of k have time to reach their superior asymptotes.

中程度の t では、小さすぎる k はまだ不正確な価値関数近似を重視しすぎることに苦しみ、大きすぎる k はブートストラップを拒むことに苦しむ。もちろん t が増えるにつれ、小さな k が優れた漸近値に到達する時間を持つので、最適な k の値は減少する。
```

[Kearns BiasVarianceTD (2000/06), Section 3]

BBF は、自分たちの指数アニールがこの論文の最適な減少スケジュールによく似ていると書いている [Schwarzer BBF (2023/05), Section 4]。

**γ を上げていく理由として原典が挙げるのは、低い割引率は収束が速く不安定性が小さい、という説明である。** Xu らは先行研究を引いて、まず近視眼的なホライズンで最適化してから割引率を上げる実践を紹介している。

```
It is known that many algorithms converge faster with lower discounts (Bertsekas and Tsitsiklis 1996), but of course too low a discount can lead to highly sub-optimal policies that are too myopic. In practice it can be better to first optimise for a myopic horizon, e.g., with γ=0 at first, and then to repeatedly increase the discount only after learning is somewhat successful (Prokhorov and Wunsch 1997).

多くのアルゴリズムは低い割引率で速く収束することが知られている(Bertsekas and Tsitsiklis 1996)が、もちろん割引率が低すぎると近視眼的すぎる、大きく準最適な方策につながりうる。実際には、まず近視眼的なホライズン(例えば最初は γ=0)で最適化し、学習がある程度うまくいった後にはじめて割引率を繰り返し上げる方がよい場合がある(Prokhorov and Wunsch 1997)。
```

[Xu MetaGradient (2018/05), Introduction]

François-Lavet らは、ニューラル fitted な価値学習で高い割引率を狙うと誤差の伝播と不安定性を招く、と説明している [FrancoisLavet HowToDiscount (2015/12), Section 3]。Agent57 は、学習初期には低 γ の方策が、後半には高 γ の方策が進歩しやすいと「予想するのは自然である」と書いており、仮説の扱いである [Badia Agent57 (2020/03), Section 3.2]。BBF の γ スケジュールには、François-Lavet らの引用と上の一文以上の理論的な説明はない。

## 6. 公式実装での確認

### 6.1 実装別の一覧

**公開コードの既定で報酬をクリップしていないのは、R2D2 系の実装(Acme、SEED RL)と DreamerV3 である。** DQN 系のフレームワークと、サンプル効率系の論文著者実装はクリップしている。下表は 2026-09-23 に取得したコードを静的に読んだ結果で、実行はしていない。

| 実装(区分) | 報酬クリップ | 変換 | 定数 | γ | n-step | スケジュール | 出典 |
|---|---|---|---|---|---|---|---|
| rlax(公式ライブラリ) | —(関数群) | h / h⁻¹、signed_logp1、2-hot、PopArt | h の ε 既定 1e-3 | 引数 | 引数 | — | [rlax (2026/09)] |
| Acme R2D2 JAX(公式) | なし(clip_rewards=False) | h(SIGNED_HYPERBOLIC_PAIR) | ε 1e-3 | 0.997 | 5 | — | [acme (2026/09)] |
| SEED RL R2D2(公式) | なし | h | ε 1e-3 | 0.997 | 5 | — | [seed_rl (2022/11)] |
| DQN Zoo 全 7 エージェント(公式) | ±1(4 フレーム合算後) | — | C51・Rainbow は ±10、51 atoms | 0.99 | Rainbow のみ 3 | — | [dqn_zoo (2023/12)] |
| Dopamine DQN / C51 / Rainbow / IQN / QR-DQN(公式) | [−1, 1](Runner の既定) | なし | — | 0.99 | DQN・C51 は 1、他は 3 | — | [dopamine (2026/03)] |
| Munchausen(munchausen_rl、論文著者) | Dopamine の Runner に委ねる | — | — | 0.99 | 1 | — | [munchausen_rl (2026/09)] |
| BBF(論文著者) | [−1, 1](Dopamine の既定) | —(C51) | 51 atoms、±10 | 0.97 → 0.997 | 10 → 3 | リセット後 10,000 gradient steps で対数線形 | [bigger_better_faster (2023/04)] |
| EfficientZero(論文著者) | sign | h + 2-hot | ε 0.001、support 601 | 0.997 を 4 乗 | 5(古いデータで短縮) | — | [EfficientZero code (2022/08)] |
| EfficientZero V2(論文著者) | sign | h + 2-hot(symlog も選択可) | ε 0.001 | 0.997 を 4 乗 | 5 | — | [EfficientZeroV2 code (2024/08)] |
| DreamerV3(論文著者) | なし(clip_reward: False) | symexp twohot、リターンのパーセンタイル正規化 | 255 bins | 1 − 1/333 | λ-return(λ 0.95) | — | [dreamerv3 code (2026/05)] |
| CleanRL DQN / C51 / Rainbow(第三者) | sign(ClipRewardEnv) | — | — | 0.99 | Rainbow は 3 | — | [cleanrl (2026/04)] |
| PQN(purejaxql、論文著者) | ±1(envpool reward_clip) | — | — | 0.99 | Q(λ)、λ 0.65 | — | [purejaxql (2025/11)] |
| RLlib R2D2(第三者、ray-2.4.0) | Atari + deepmind 前処理では sign | h(use_h_function=True) | ε 1e-3 | 0.997 | 1 | — | [ray r2d2 (2023/04)] |
| BTR(論文著者) | ±1 | —(IQN) | — | 0.997 | 3 | 任意(既定は無効): γ 0.97 → 0.997 を前半で線形 | [BTR code (2025/05)] |
| Agent57 / MEME | 公式の公開コードは確認できない | — | — | — | — | — | [GitHub search (2026/09)] |

### 6.2 h(x) の実装と ε の既定値

**rlax の h は signed_hyperbolic として実装されており、ε の既定値は 1e-3 である。** 逆関数 signed_parabolic も同じ既定値を持つ。

```python
def signed_hyperbolic(x: Array, eps: float = 1e-3) -> Array:
  """Signed hyperbolic transform, inverse of signed_parabolic."""
  chex.assert_type(x, float)
  return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x
```

[rlax (2026/09), rlax/_src/transforms.py:L60-L63]

rlax の transformed_n_step_q_learning と transformed_retrace は、tx_pair の既定値が IDENTITY_PAIR である。呼び出し側がペアを渡さない限り変換はかからない [rlax (2026/09), rlax/_src/nonlinear_bellman.py:L221-L285]。

**Acme の R2D2 設定は、報酬クリップを無効にし、h のペアを既定で使う。** 既定値は R2D2 論文の値(クリップなし、γ 0.997、ε 10⁻³、n 5)と一致する。

```python
@dataclasses.dataclass
class R2D2Config:
  """Configuration options for R2D2 agent."""
  discount: float = 0.997
  [...]
  bootstrap_n: int = 5
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
```

[acme (2026/09), acme/agents/jax/r2d2/config.py:L22-L38](途中の行を省略)

SEED RL の R2D2 も、フラグの既定値を ε 1e-3、n 5、γ 0.997 としている。SEED 論文の表は、報酬クリップなしと書いている [seed_rl (2022/11), agents/r2d2/learner.py:L82-L87]、[Espeholt SEED (2019/10), Table 8, 9]。

### 6.3 報酬クリップの既定値

**Dopamine の Runner は既定で報酬を [−1, 1] にクリップする。** このオプションは 2020 年 11 月のコミット de73a04 で追加され、それ以前の Runner は無条件にクリップしていた [dopamine de73a04 (2020/11)]。Dopamine のオンライン Atari エージェントには、h 変換や PopArt の選択肢は見当たらない [dopamine (2026/03)]。

```python
      clip_rewards: bool, whether to clip rewards in [-1, 1].
```

[dopamine (2026/03), dopamine/discrete_domains/run_experiment.py:L214]

BBF のランナーは Dopamine の Runner を継承し、clip_rewards を渡していない。BBF の gin もこの値を上書きしないので、公式設定のまま実行すると報酬はクリップされる [bigger_better_faster (2023/04), bbf/eval_run_experiment.py:L173, L197-L198, L434-L436]。Munchausen の公式実装も、クリップを Dopamine の Runner に委ねている [munchausen_rl (2026/09), munchausen_rl/train.py:L170-L171]。

**DreamerV3 の公式設定は、Atari と Atari 100k のどちらでも clip_reward を False にしている。**

```yaml
    atari: {size: [96, 96], repeat: 4, sticky: True, gray: True, actions: all, lives: unused, noops: 30, autostart: False, pooling: 2, aggregate: max, resize: pillow, clip_reward: False}
```

[dreamerv3 code (2026/05), dreamerv3/configs.yaml:L30]

LightZero(第三者の MuZero 系ライブラリ)は、学習用の収集環境だけを符号でクリップし、評価環境はクリップしない。h(ε 0.001)も併用する [LightZero (2026/09)]。

### 6.4 BBF と BTR のスケジュール実装

**BBF の公式コードは、n と γ を対数空間で線形に補間するスケジューラで変える。** γ は 1 − γ を補間する(reverse=True)。

```python
  if reverse:
    initial_value = 1 - initial_value
    final_value = 1 - final_value

  start = onp.log(initial_value)
  end = onp.log(final_value)
  [...]
  def scheduler(step):
    steps_left = decay_period + warmup_steps - step
    bonus_frac = steps_left / decay_period
    bonus = onp.clip(bonus_frac, 0.0, 1.0)
    new_value = bonus * (start - end) + end

    new_value = onp.exp(new_value)
    if reverse:
      new_value = 1 - new_value
    return new_value
```

[bigger_better_faster (2023/04), bigger_better_faster/bbf/agents/spr_agent.py:L331-L350](途中の行を省略)

経過量 cycle_grad_steps は勾配更新ごとに増え、重みリセットの処理 reset_weights の中で 0 に戻る。リセットの間隔 reset_every = 20_000 は training_steps 単位で、公開設定では約 40,000 gradient steps にあたる [bigger_better_faster (2023/04), bbf/agents/spr_agent.py:L1531, L1629-L1630, L1776-L1779; bbf/configs/BBF.gin:L22-L31]。リプレイバッファは生の報酬を保存し、n-step リターンはサンプル時に現在の n と γ で計算し直す。このためスケジュールの変更は過去のデータにも即時に反映される [bigger_better_faster (2023/04), bbf/replay_memory/subsequence_replay_buffer.py:L601-L607]。本調査の計算では、1 − γ(x) = 0.03 × 0.1^(x/10000)、n(x) = round(10 × 0.3^(x/10000))(x はリセット後の gradient steps、0 ≤ x ≤ 10,000)になる。

**BTR の公式コードにある γ のアニールは、既定では無効で、有効にすると総勾配ステップの前半で γ を 0.97 から 0.997 へ線形に上げる。**

```python
        if discount_anneal:
            self.discount_anneal = True
            self.gamma = 0.97
            self.final_gamma = 0.997
            self.annealing_period = self.total_grad_steps // 2  # first half of training
            self.gamma_inc = (self.final_gamma - self.gamma) / self.annealing_period
        else:
            self.gamma = discount
            self.discount_anneal = False
```

[BTR code (2025/05), Agent.py:L210-L218]

引数 --discount_anneal の既定値は 0 である [BTR code (2025/05), main.py:L163]。

## 7. 批判と留保

**h(x) の理論保証は、実際に使われる条件をカバーしていない。** 確率的 MDP で縮小写像になる条件(本調査の計算で γ < 約 0.02)は実用の γ を大きく下回り、固定点も h∘Q* とは限らない(§2.2)。Muesli の著者は、この変換は確率的なリターンに対してバイアスがあると書いている。論文によって式の印字が異なり(εx の位置、√(x²+1)、h⁻¹ の 2 乗の欠落、ε の不一致)、どれが実装どおりかは本文から確認できない(§2.2)。

**Pop-Art には否定的な結果と批判がある。** Pohlen らの枠組みでは、Pop-Art は h より大きく劣った。

```
One possible limiting factor that makes PopArt a bad choice for our framework is that training batches contain highly rewarding states from the very beginning of training. SGD updates performed before the moving statistics have adequately adapted the moments of the target distribution might result in catastrophic changes to the network's weights.

我々の枠組みで PopArt が悪い選択になる制約要因として考えられるのは、学習の最初から学習バッチに高報酬の状態が含まれることである。移動統計がターゲット分布のモーメントに十分に適応する前に行われた SGD 更新が、ネットワークの重みに壊滅的な変化をもたらすかもしれない。
```

[Pohlen ApeXDQfD (2018/05), Section 4.4]

Schaul らは、Pop-Art はハイパーパラメータが複数あり、最終層の重みへの書き込みが必要で実装への介入が大きいと書いている。また、短いホライズンでスケールを追う性質が不安定さにつながりうるとも述べている [Schaul ReturnScaling (2021/05), Section 4.3]。

**クリップの利点は、報酬スケールをそろえることだけではない可能性がある。** A2C の著者は、報酬の頻度を最適化するという帰納バイアスが多くの Atari ゲームで良いヒューリスティックになっている可能性を挙げた(§3.3)。報酬分布を変えることは、更新の大きさだけでなく探索やリスク選好にも影響すると述べ、クリップの利点を取り戻す一般的な解は今後の課題としている [Hessel InductiveBiases (2019/07), Section 4]。

**比較研究の多くは統計的な厚みが薄い。**

- seed 数: Schaul らの 1 ヘッドの比較は各ゲーム 1 seed である。van Hasselt らと Pohlen らは seed 数を書いていない。
- ゲーム数: R2D2 のアブレーションは 5 ゲーム、Pohlen らは 6 ゲームである。
- 転載値: van Hasselt らのクリップ版のスコアは他の論文からの転載である。
- 図からの読み取り: 本書の【目視】【画素換算】の値は図からの概算である。

**クリップの不要を主張する側にも、データの薄いものがある。**

- DNA 論文(PPO)は、クリップが報酬正規化より有利だとは分からなかったと書くが、比較のデータを示していない [Aitchison DNA (2022/06), Appendix F]。
- LightZero の第三者報告は 1 seed で学習長も不揃いである(§3.5)。

**γ のスケジュールの効果は、報告によって大きさが違う。**

- BBF のアニールの効果の大部分は n のスケジュールによるもので、γ だけを固定した条件は CI が重なる(§5.2)。
- BTR は 1 ゲームで有意差なしと報告した。
- メタ勾配による γ の適応は、γ を入力に与えないと逆効果だった。
- STAC では γ だけの自己調整は効かなかった(§5.3)。
- François-Lavet らの結果は 5〜6 ゲームで、改善の一部は少数の seed に引っ張られている(§5.1)。

## 8. 総合評価

論点の問いに対して、一次情報から確定できたことを次にまとめる。

**第一に、大規模エージェントの多くは、報酬クリップの代わりに可逆な価値の再スケーリング h(x) を使っている。** 報酬ではなくターゲットの価値のスケールを縮める方式で、Pohlen らが導入し(ε = 10⁻²)、R2D2 が ε = 10⁻³ で採用した。NGU、Agent57、MEME、MuZero、Muesli がこれを引き継いでいる。

- MuZero と Muesli は、h で変換した値を分類型で表す。
- MEME は TD 誤差の正規化を重ねている。
- 公式ライブラリの ε の既定値は 1e-3 である。

これとは別系統として、Pop-Art(Double DQN、PopArt-IMPALA)、DreamerV3 の symlog / symexp twohot、TD 誤差を割る return-based scaling がある。一方、Rainbow、BBF、BTR、EfficientZero、PQN と、Dopamine・DQN Zoo・CleanRL の既定はクリップを残している。

**第二に、DQN 系で 200M frames 前後の予算のまま、クリップだけを外した A/B は見つからなかった。** 最も近いのは次の 3 件である。

- van Hasselt ら(Double DQN、200M frames、57 ゲーム)は、スコアを報告したのが Pop-Art 併用の比較だけである。
- Schaul ら(R2D2 系)は、クリップだけを変えているが約 10⁹ frames・1 seed である。
- Hessel ら(A2C、200M frames、57 ゲーム、8 seeds)は、エージェントが DQN 系ではない。

これらを並べると、次の 3 つの傾向が見える。

- 正規化も変換もしない生の報酬では、IMPALA・A2C・PPO で集計値が大きく崩れた。R2D2 系(Adam)だけは崩れなかった。
- Pop-Art、h、symlog でスケールを扱っても、集計ではクリップとほぼ同等かやや劣った。A2C・R2D2 系・PPO の 3 研究で、クリップが最良かそれに近いと報告されている。
- ゲーム別の勝ち負けは大きく割れる。Gopher・Centipede・Ms. Pac-Man はクリップなしが上、Video Pinball・Star Gunner・Time Pilot・Q*Bert はクリップが上だった。R2D2 系の生の報酬では、Kung-Fu Master と Alien がほぼ 0 に落ちた【目視】。

**第三に、h だけで 0.997 を超える γ を安定させたという報告は見つからなかった。**

- γ = 0.999 の Ape-X DQfD では、h に加えて TC loss が必要だった。外すと学習途中で崩壊し、浮動小数点例外でプロセスが止まることが多かった。
- γ = 0.9999 は、h 付きの R2D2 でも全ゲームに固定すると非常に不安定で、Agent57 は γ の族とメタコントローラで回避した。
- 0.997 は R2D2 以来 h と組み合わせて使われているが、0.99 との比較は「まちまち」だった。
- 理論面では、h の縮小性の保証は実用の γ に届かない。高い γ の問題として、次状態への過汎化、action gap の崩壊、リターンの分散の増大が挙げられている。
- クリップ報酬のもとで |Q| の上限 1/(1−γ) は、0.997 で約 333、0.999 で 1000 になる(本調査の計算)。ただしこの上限が安定性に効くことを、γ を変えて示した実験は見つからなかった。

**第四に、学習中の γ・n-step のスケジュールには効果の報告があるが、効いているのは主に n のスケジュールで、γ のスケジュール単独の効果は弱い。**

- BBF の公式 CSV から計算すると、アニールを丸ごと外すと IQM は 31%(RR=2)/ 17%(RR=8)下がる。n = 10 固定では 42% / 37% 下がる。一方、γ を 0.99 に固定しても 7% / 4% しか下がらず、CI は完全版と重なった。このアニールはリセットと連動しており、Atari 100k の設定で、報酬はクリップされている。
- BTR は γ のアニールで有意差なし(1 ゲーム)と報告した。François-Lavet らは γ の増加で 5 ゲーム中 4 ゲームの学習が速くなったと報告したが、0.99 を超えると過大評価で崩壊した。
- メタ勾配による γ の適応は、γ を入力に与える前提で効いた。メタコントローラは γ を連続的に変えるのではなく、γ の異なる方策から選ぶ仕組みである。
- n を下げていくことには Kearns と Singh の理論的な裏付けがある。γ を上げていくことの説明は「低い γ は収束が速く安定」という定性的なものにとどまる。

## 9. 調査の限界

- **本文の取得経路に制約があった。**
  - R2D2 は OpenReview がブラウザ検証を要求したため、Wayback Machine が保存した採択版 PDF で読んだ。
  - DreamerV3 の Nature 版と MEME の ICLR 採択版は読んでいない。
  - Hessel らの "On Inductive Biases" の採否は、OpenReview を取得できず確認していない。
- **図からの読み取り値は概算である。** 【目視】【画素換算】の値(R2D2、Schaul らの図 12・15、PopArt-IMPALA の図 8、A2C・PPO の図)は、原文に数値がない。PopArt-IMPALA の図 8 は対数軸で、棒の判別は凡例の順序と塗りの色による。
- **seed 数と単位が不明なものがある。**
  - seed 数: van Hasselt ら、Pohlen ら、Xu らの Atari 実験、François-Lavet らの 50M steps 実験は書かれていない。
  - Sullivan らの "Steps" と frames の換算は本文にない。
  - R2D2 の Atari の総フレーム数は本文になく、第三者の記載は MuZero が 37.5B、NGU が 35 B で一致しない。
  - Ape-X DQN の γ の数値は本文に見当たらない。
- **BBF の数値は本調査の計算である。**
  - IQM と CI は公式 CSV から計算した。CI はゲーム層別ブートストラップの近似で、rliable とは実装が異なる。
  - 論文の Figure 5 は「15 runs」と書くが、CSV の run 数は条件により 14〜60 と異なる。
  - 29 の検証ゲームでの低下率は、論文 Figure 8 と数ポイントずれる。
  - RR=8 用の設定ファイルはリポジトリになく、RR=8 でのリセット回数は確認していない。
- **原典内・原典間で記述が食い違う。**
  - PopArt-IMPALA のクリップなしの中央値は、本文 101%、表 107.0%。
  - A2C の図 2 の指標は、本文が中央値、キャプションが "mean episode return"。
  - NGU の ε は、付録 10⁻²、表 0.001。
  - Munchausen の M-IQN の n-step は、本文脚注が 1-step、付録が 3-step。
  - DreamerV2 の γ は、本文 0.999、表 0.995。
  - Agent57 は、本文の γ 族と表の "Discount r^e 0.997"、窓幅 160 と 90。
  - François-Lavet らの Figure 4 の Breakout と Q-bert の値は、Table 2 と入れ替わっているように見える。
  - Stop Regressing の Figure 15 は、キャプションと本文で優劣の記述が逆。
- **公式コードは静的に読んだだけで、実行による確認はしていない。**
  - 「クリップしない」「記述なし」は、読んだファイルと grep の範囲での判断である。
  - BBF・PQN・Munchausen の報酬クリップは、論文ではなく現行の公式実装から読み取った。論文の実験時点のコードと同じかは確認していない。
  - Agent57 と MEME の公式の公開コードは、GitHub の組織内検索と論文の範囲で見つからなかった。
  - Dopamine の分類型損失の本体は非公開で、Stop Regressing の公式コードは見つからなかった。
- **被引用数は桁感の目安である。** Semantic Scholar の API から 2026-09-23 に取得した。arXiv 版と会議版でレコードが分かれている論文があり(STAC、Daley と Amato、DreamerV3 など)、本書では判断材料に使っていない。
- **網羅性には限りがある。**
  - 2025〜2026 年の研究で、Atari の報酬クリップの有無、γ の値そのもの、γ・n のスケジュールを主題にしたものは、数回の検索の範囲では BTR の付録の 1 行しか見つからなかった。
  - Reddit では比較の投稿を見つけられなかった。
  - PBT による割引率の調整は対象外とした。

## 10. 出典リスト

本節は本書の本文で直接引用・参照した出典を挙げる。被引用数は Semantic Scholar から 2026-09-23 に取得した値である。

論文

[Mnih DQN, 2015/02] Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-level control through deep reinforcement learning." Nature 518, 529–533. 所属: Google DeepMind. 被引用数: 33,248. https://www.nature.com/articles/nature14236

[Bellemare ALE, 2012/07] Bellemare, M. G., Naddaf, Y., Veness, J., Bowling, M. "The Arcade Learning Environment: An Evaluation Platform for General Agents." Journal of Artificial Intelligence Research 47 (2013). 所属: University of Alberta. 被引用数: 3,389. https://arxiv.org/abs/1207.4708

[vanHasselt PopArt, 2016/02] van Hasselt, H., Guez, A., Hessel, M., Mnih, V., Silver, D. "Learning values across many orders of magnitude." NIPS 2016(arXiv:1602.07714 v2 を参照). 所属: Google DeepMind. 被引用数: 211. https://arxiv.org/abs/1602.07714

[Hester DQfD, 2017/04] Hester, T., Vecerik, M., Pietquin, O., et al. "Deep Q-learning from Demonstrations." AAAI 2018(arXiv:1704.03732). 所属: Google DeepMind. 被引用数: 160(arXiv 版レコード). https://arxiv.org/abs/1704.03732

[Machado RevisitingALE, 2017/09] Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., Bowling, M. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." Journal of Artificial Intelligence Research 61 (2018). 所属: University of Alberta; Google Brain; Franklin & Marshall College; DeepMind; Microsoft Research. 被引用数: 630. https://arxiv.org/abs/1709.06009

[Hessel Rainbow, 2017/10] Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., Silver, D. "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI 2018. 所属: DeepMind. 被引用数: 2,715. https://arxiv.org/abs/1710.02298

[Espeholt IMPALA, 2018/02] Espeholt, L., Soyer, H., Munos, R., et al. "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures." ICML 2018(arXiv:1802.01561 v3 を参照). 所属: DeepMind. 被引用数: 1,937. https://arxiv.org/abs/1802.01561

[Horgan ApeX, 2018/03] Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel, M., van Hasselt, H., Silver, D. "Distributed Prioritized Experience Replay." ICLR 2018. 所属: DeepMind. 被引用数: 849. https://arxiv.org/abs/1803.00933

[Pohlen ApeXDQfD, 2018/05] Pohlen, T., Piot, B., Hester, T., Azar, M. G., Horgan, D., Budden, D., Barth-Maron, G., van Hasselt, H., Quan, J., Večerík, M., Hessel, M., Munos, R., Pietquin, O. "Observe and Look Further: Achieving Consistent Performance on Atari." arXiv:1805.11593(プレプリント). 所属: DeepMind; Google Brain. 被引用数: 128. https://arxiv.org/abs/1805.11593

[Hessel PopArtIMPALA, 2018/09] Hessel, M., Soyer, H., Espeholt, L., Czarnecki, W., Schmitt, S., van Hasselt, H. "Multi-task Deep Reinforcement Learning with PopArt." AAAI 2019. 所属: DeepMind(AAAI 公式ページのメタデータ). 被引用数: 370. https://arxiv.org/abs/1809.04474

[Kapturowski R2D2, 2018/09] Kapturowski, S., Ostrovski, G., Quan, J., Munos, R., Dabney, W. "Recurrent Experience Replay in Distributed Reinforcement Learning." ICLR 2019(Wayback Machine が保存した OpenReview の採択版 PDF を参照). 所属: DeepMind. 被引用数: 585. https://openreview.net/forum?id=r1lyTjAqYX

[Burda RND, 2018/10] Burda, Y., Edwards, H., Storkey, A., Klimov, O. "Exploration by Random Network Distillation." ICLR 2019. 所属: OpenAI; University of Edinburgh. 被引用数: 1,753. https://arxiv.org/abs/1810.12894

[delaCruz A3CTB, 2018/12] de la Cruz Jr., G. V., Du, Y., Taylor, M. E. "Pre-training with Non-expert Human Demonstration for Deep Reinforcement Learning." arXiv:1812.08904. 所属: Washington State University. 被引用数: 27. https://arxiv.org/abs/1812.08904

[vanHasselt DeadlyTriad, 2018/12] van Hasselt, H., Doron, Y., Strub, F., Hessel, M., Sonnerat, N., Modayil, J. "Deep Reinforcement Learning and the Deadly Triad." arXiv:1812.02648. 所属: DeepMind; University of Lille. 被引用数: 293. https://arxiv.org/abs/1812.02648

[Hessel InductiveBiases, 2019/07] Hessel, M., van Hasselt, H., Modayil, J., Silver, D. "On Inductive Biases in Deep Reinforcement Learning." arXiv:1907.02908(プレプリント). 所属: DeepMind. 被引用数: 49. https://arxiv.org/abs/1907.02908

[Toromanoff SABER, 2019/08] Toromanoff, M., Wirbel, E., Moutarde, F. "Is Deep Reinforcement Learning Really Superhuman on Atari? Leveling the playing field." arXiv:1908.04683. 所属: MINES ParisTech; Valeo. 被引用数: 24. https://arxiv.org/abs/1908.04683

[Espeholt SEED, 2019/10] Espeholt, L., Marinier, R., Stanczyk, P., Wang, K., Michalski, M. "SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference." ICLR 2020(arXiv:1910.06591 v2 を参照). 所属: Brain Team, Google Research. 被引用数: 未取得. https://arxiv.org/abs/1910.06591

[Schrittwieser MuZero, 2019/11] Schrittwieser, J., Antonoglou, I., Hubert, T., et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature 588 (2020)(arXiv:1911.08265 v2 を参照). 所属: DeepMind. 被引用数: 2,727. https://arxiv.org/abs/1911.08265

[Badia NGU, 2020/02] Badia, A. P., Sprechmann, P., Vitvitskyi, A., et al. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020. 所属: DeepMind. 被引用数: 370. https://arxiv.org/abs/2002.06038

[Badia Agent57, 2020/03] Badia, A. P., Piot, B., Kapturowski, S., Sprechmann, P., Vitvitskyi, A., Guo, D., Blundell, C. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020. 所属: DeepMind. 被引用数: 599. https://arxiv.org/abs/2003.13350

[Hessel Muesli, 2021/04] Hessel, M., Danihelka, I., Viola, F., et al. "Muesli: Combining Improvements in Policy Optimization." ICML 2021(arXiv:2104.06159 v2 を参照). 所属: DeepMind; University College London. 被引用数: 70. https://arxiv.org/abs/2104.06159

[Schrittwieser MuZeroUnplugged, 2021/04] Schrittwieser, J., Hubert, T., Mandhane, A., Barekatain, M., Antonoglou, I., Silver, D. "Online and Offline Reinforcement Learning by Planning with a Learned Model." NeurIPS 2021. 所属: DeepMind. 被引用数: 149. https://arxiv.org/abs/2104.06294

[Schaul ReturnScaling, 2021/05] Schaul, T., Ostrovski, G., Kemaev, I., Borsa, D. "Return-based Scaling: Yet Another Normalisation Trick for Deep RL." arXiv:2105.05347(プレプリント). 所属: DeepMind. 被引用数: 29. https://arxiv.org/abs/2105.05347

[Ye EfficientZero, 2021/11] Ye, W., Liu, S., Kurutach, T., Abbeel, P., Gao, Y. "Mastering Atari Games with Limited Data." NeurIPS 2021. 所属: Tsinghua University; UC Berkeley; Shanghai Qi Zhi Institute. 被引用数: 340. https://arxiv.org/abs/2111.00210

[Aitchison DNA, 2022/06] Aitchison, M., Sweetser, P. "DNA: Proximal Policy Optimization with a Dual Network Architecture." arXiv:2206.10027. 所属: The Australian National University. 被引用数: 7. https://arxiv.org/abs/2206.10027

[Kapturowski MEME, 2022/09] Kapturowski, S., Campos, V., Jiang, R., Rakićević, N., van Hasselt, H., Blundell, C., Badia, A. P. "Human-level Atari 200x faster." ICLR 2023(arXiv:2209.07550 v1 を参照). 所属: DeepMind. 被引用数: 44. https://arxiv.org/abs/2209.07550

[Hafner DreamerV3, 2023/01] Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. "Mastering Diverse Domains through World Models." arXiv:2301.04104(v1 と v2 を参照。Nature 版は 2025/04). 所属: DeepMind; University of Toronto. 被引用数: 1,415(arXiv レコード). https://arxiv.org/abs/2301.04104

[Schwarzer BBF, 2023/05] Schwarzer, M., Obando-Ceron, J., Courville, A., Bellemare, M. G., Agarwal, R., Castro, P. S. "Bigger, Better, Faster: Human-level Atari with human-level efficiency." ICML 2023(arXiv:2305.19452 v3 を参照). 所属: Google DeepMind; Mila; Université de Montréal. 被引用数: 176. https://arxiv.org/abs/2305.19452

[Sullivan PPOTricks, 2023/10] Sullivan, R., Kumar, A., Huang, S., Dickerson, J. P., Suarez, J. "Reward Scale Robustness for Proximal Policy Optimization via DreamerV3 Tricks." NeurIPS 2023. 所属: University of Maryland; Massachusetts Institute of Technology; Drexel University. 被引用数: 10. https://arxiv.org/abs/2310.17805

[Lyle PlasticityCauses, 2024/02] Lyle, C., Zheng, Z., Khetarpal, K., Dabney, W., van Hasselt, H., Pascanu, R., Martens, J. "Disentangling the Causes of Plasticity Loss in Neural Networks." arXiv:2402.18762(Semantic Scholar の venue 表記は CoLLAs 2024). 所属: Google DeepMind. 被引用数: 94. https://arxiv.org/abs/2402.18762

[Farebrother StopRegressing, 2024/03] Farebrother, J., Orbay, J., Vuong, Q., Taïga, A. A., Chebotar, Y., Xiao, T., Irpan, A., Levine, S., Castro, P. S., Faust, A., Kumar, A., Agarwal, R. "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL." ICML 2024. 所属: Google DeepMind; Mila. 被引用数: 162. https://arxiv.org/abs/2403.03950

[Gallici PQN, 2024/07] Gallici, M., Fellows, M., Ellis, B., Pou, B., Masmitja, I., Foerster, J. N., Martin, M. "Simplifying Deep Temporal Difference Learning." ICLR 2025. 所属: Universitat Politècnica de Catalunya; University of Oxford; Barcelona Supercomputing Center; Institut de Ciències del Mar. 被引用数: 89. https://arxiv.org/abs/2407.04811

[Clark BTR, 2024/11] Clark, T., Towers, M., Evers, C., Hare, J. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." ICML 2025(arXiv:2411.03820 v2 を参照). 所属: University of Southampton. 被引用数: 11. https://arxiv.org/abs/2411.03820

γ と価値の写像

[Petrik LowerDiscount, 2008/12] Petrik, M., Scherrer, B. "Biasing Approximate Dynamic Programming with a Lower Discount Factor." NIPS 2008. 所属: University of Massachusetts Amherst; LORIA. 被引用数: 68. https://proceedings.neurips.cc/paper/2008/file/08c5433a60135c32e34f46a71175850c-Paper.pdf

[Jiang PlanningHorizon, 2015/05] Jiang, N., Kulesza, A., Singh, S., Lewis, R. "The Dependence of Effective Planning Horizon on Model Accuracy." AAMAS 2015. 所属: University of Michigan. 被引用数: 162. https://dl.acm.org/doi/10.5555/2772879.2773300

[Lehnert LongHorizon, 2018/02] Lehnert, L., Laroche, R., van Seijen, H. "On Value Function Representation of Long Horizon Problems." AAAI 2018. 所属: Microsoft Maluuba; Brown University. 被引用数: 26. https://ojs.aaai.org/index.php/AAAI/article/view/11646

[vanSeijen LogMapping, 2019/06] van Seijen, H., Fatemi, M., Tavakoli, A. "Using a Logarithmic Mapping to Enable Lower Discount Factors in Reinforcement Learning." NeurIPS 2019. 所属: Microsoft Research Montréal; Imperial College London. 被引用数: 35. https://arxiv.org/abs/1906.00572

[Amit DiscountRegularizer, 2020/07] Amit, R., Meir, R., Ciosek, K. "Discount Factor as a Regularizer in Reinforcement Learning." ICML 2020. 所属: Technion; Microsoft Research Cambridge. 被引用数: 91. https://arxiv.org/abs/2007.02040

[Tang TaylorDiscount, 2021/06] Tang, Y., Rowland, M., Munos, R., Valko, M. "Taylor Expansions of Discount Factors." ICML 2021. 所属: Columbia University; DeepMind. 被引用数: 9. https://arxiv.org/abs/2106.06170

[Fatemi OrchestratedMapping, 2022/03] Fatemi, M., Tavakoli, A. "Orchestrated Value Mapping for Reinforcement Learning." ICLR 2022. 所属: Microsoft Research Montréal; Max Planck Institute for Intelligent Systems. 被引用数: 8. https://arxiv.org/abs/2203.07171

スケジュール

[Kearns BiasVarianceTD, 2000/06] Kearns, M., Singh, S. "Bias-Variance Error Bounds for Temporal Difference Updates." COLT 2000. 所属: AT&T Labs. 被引用数: 108. https://www.learningtheory.org/colt2000/papers/KearnsSingh.pdf

[FrancoisLavet HowToDiscount, 2015/12] François-Lavet, V., Fonteneau, R., Ernst, D. "How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies." NIPS 2015 Deep Reinforcement Learning Workshop. 所属: University of Liège. 被引用数: 122. https://arxiv.org/abs/1512.02011

[Xu MetaGradient, 2018/05] Xu, Z., van Hasselt, H., Silver, D. "Meta-Gradient Reinforcement Learning." NeurIPS 2018. 所属: DeepMind. 被引用数: 383. https://arxiv.org/abs/1805.09801

[Daley ReconcilingLambda, 2018/10] Daley, B., Amato, C. "Reconciling λ-Returns with Experience Replay." NeurIPS 2019. 所属: Northeastern University. 被引用数: 42(NeurIPS 版レコード). https://arxiv.org/abs/1810.09967

[Zahavy STAC, 2020/02] Zahavy, T., Xu, Z., Veeriah, V., Hessel, M., Oh, J., van Hasselt, H., Silver, D., Singh, S. "A Self-Tuning Actor-Critic Algorithm." NeurIPS 2020. 所属: DeepMind. 被引用数: 106(NeurIPS 版レコード). https://arxiv.org/abs/2002.12928

[Fedus ReplayFundamentals, 2020/07] Fedus, W., Ramachandran, P., Agarwal, R., Bengio, Y., Larochelle, H., Rowland, M., Dabney, W. "Revisiting Fundamentals of Experience Replay." ICML 2020. 所属: Google Brain; Mila; CIFAR; DeepMind. 被引用数: 335. https://arxiv.org/abs/2007.06700

[Nikishin PrimacyBias, 2022/05] Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., Courville, A. "The Primacy Bias in Deep Reinforcement Learning." ICML 2022. 所属: Mila, Université de Montréal. 被引用数: 328. https://arxiv.org/abs/2205.07802

公式・準公式実装と第三者資料(年月は参照したコミットの年月)

[rlax, 2026/09] Google DeepMind. "rlax." GitHub, commit 036c859. https://github.com/google-deepmind/rlax

[acme, 2026/09] Google DeepMind. "acme." GitHub, commit 89080fe. https://github.com/google-deepmind/acme

[seed_rl, 2022/11] Google Research. "seed_rl." GitHub, commit 0e1e0ac. https://github.com/google-research/seed_rl

[dqn_zoo, 2023/12] Google DeepMind. "dqn_zoo." GitHub, commit 45061f4. https://github.com/google-deepmind/dqn_zoo

[dopamine, 2026/03] Google. "dopamine." GitHub, commit 5873f54. https://github.com/google/dopamine

[dopamine de73a04, 2020/11] Google. "dopamine: Add option to remove reward clipping." GitHub, commit de73a04. https://github.com/google/dopamine/commit/de73a04e2cea575ce54b6f8df8c51431b4c05353

[munchausen_rl, 2026/09] Google Research(Vieillard et al.). "google-research/munchausen_rl." GitHub, google-research commit 758b894. https://github.com/google-research/google-research/tree/master/munchausen_rl

[bigger_better_faster, 2023/04] Google Research(Schwarzer et al.). "google-research/bigger_better_faster." GitHub(ロジックの最終変更は 2023-04-26 commit b7a527e、参照は google-research commit 758b894). https://github.com/google-research/google-research/tree/master/bigger_better_faster

[BBF scores, 2023/06] Google Research(Schwarzer et al.). "bigger_better_faster/scores." GitHub, commit 82124ce. https://github.com/google-research/google-research/tree/master/bigger_better_faster/scores

[EfficientZero code, 2022/08] Ye, W., et al. "EfficientZero." GitHub, commit 468bb03. https://github.com/YeWR/EfficientZero

[EfficientZeroV2 code, 2024/08] Wang, S., et al. "EfficientZeroV2." GitHub, commit 12b9e77. https://github.com/Shengjiewang-Jason/EfficientZeroV2

[dreamerv3 code, 2026/05] Hafner, D. "dreamerv3." GitHub, commit e3f0224. https://github.com/danijar/dreamerv3

[cleanrl, 2026/04] Huang, S., et al. "cleanrl." GitHub, commit fe8d8a0. https://github.com/vwxyzjn/cleanrl

[purejaxql, 2025/11] Gallici, M., et al. "purejaxql." GitHub, commit 47af6d7. https://github.com/mttga/purejaxql

[BTR code, 2025/05] Clark, T., et al. "BTR." GitHub, commit 093ca77. https://github.com/VIPTankz/BTR

[ray r2d2, 2023/04] Ray contributors. "ray (tag ray-2.4.0) rllib/algorithms/r2d2." GitHub. https://github.com/ray-project/ray/tree/ray-2.4.0/rllib/algorithms/r2d2

[LightZero, 2026/09] OpenDILab. "LightZero." GitHub, commit 13bc1cb. https://github.com/opendilab/LightZero

[logrl, 2019/10] Microsoft. "logrl"(LogDQN の公式実装). GitHub. https://github.com/microsoft/logrl

[GitHub search, 2026/09] GitHub. "Search repositories API(org:google-deepmind agent57 / meme、org:deepmind agent57 の検索、2026-09-23 取得)." https://api.github.com/search/repositories?q=org:google-deepmind+agent57

[LightZero issue233, 2024/07] puyuan1996(LightZero のメンテナとして返信). "Bad performance on long run on MsPacman and SpaceInvaders" へのコメント. opendilab/LightZero issue #233. https://github.com/opendilab/LightZero/issues/233#issuecomment-2208138236

[TorchRL issue1777, 2024/01] skandermoalla. "[BUG] Incorrect reward "clipping" in Atari baseline." pytorch/rl issue #1777. https://github.com/pytorch/rl/issues/1777
