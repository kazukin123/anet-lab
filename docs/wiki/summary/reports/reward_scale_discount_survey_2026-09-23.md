# Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)

## 元の材料

[Survey: 報酬のスケール処理と割引率(報酬クリップの代替、h(x)・Pop-Art、γ の設定とスケジュール)](../../../../reports/reward_scale_discount_survey_2026-09-23.md) blob:f3ccd16e2f96c0c64b5f39ac812a79a27bca98cc

## 要点

- Machado らの ALE 再検討論文は、ALE 向けのエージェントのほとんどが何らかの報酬正規化を行っていると述べ、符号だけを使う方法と、最初の非ゼロ報酬の大きさで割る方法を例に挙げている。この論文にクリップの有無を比べる実験はない [Machado RevisitingALE (2017/09), Section 2.1]。
- R2D2 のアブレーション(5 ゲーム、3 seeds)の「Clipped」は、h 変換を外して報酬クリップに置き換えた条件で、クリップと h の有無が同時に変わる。著者はこれを含む LSTM 以外の設計選択の効果を「まちまち」と書いている [Kapturowski R2D2 (2018/09), Section 5, Figure 7]。
- {γ, λ} を適応させると 292.9% だった [Xu MetaGradient (2018/05), Table 1]。固定 γ の IMPALA は 0.995 で 211.9%、0.998 で 208.5%、0.999 で 114.9% で、0.999 が最も悪かった [Xu MetaGradient (2018/05), Appendix C, Table 3]。
- R2D2 のクリップ比較では h 変換も同時に変わり、効果を片方に帰属できない。
- PopArt-IMPALA の中央値が本文と表で異なるため、出典の位置を添えて比較する。

## 内容

### 1. 前提: 報酬クリップとは何か

DQN は学習中だけ報酬を [−1, 1] にクリップした。理由はゲーム間のスコア規模の差で、同じ学習率を全ゲームで使うためである。原典は同じ段落で、報酬の大小を区別できなくなる副作用も認めている。

DQN はこれとは別に、更新式の TD 誤差項も [−1, 1] にクリップしている。原典はこれを、区間外の誤差に絶対値損失を使うことと同じだと説明している [Mnih DQN (2015/02), Methods "Training algorithm for deep Q-networks"]。γ は全ゲーム共通の 0.99 で、ハイパーパラメータ全体は 5 ゲームでの非公式な探索で選んだと書かれている(§4.1)。

「報酬クリップ」と呼ばれる処理には、符号で {+1, 0, −1} に置き換える実装と、[−1, 1] に切り詰める実装の 2 通りがある。 Stable-Baselines3 とそこから移植した CleanRL は前者、Dopamine の Runner は後者である。

両者の結果が異なるのは、絶対値が 1 未満の非ゼロ報酬があるときだけである(本調査の整理)。この定義の違いは第三者の GitHub issue で議論されており、TorchRL のメンテナは符号化への修正に同意している [TorchRL issue1777 (2024/01)]。DQN Zoo は行動反復 4 回分の報酬を合算してから ±1 にクリップする [dqn_zoo (2023/12), dqn_zoo/processors.py:L476-L487]。

後続論文は、クリップが目的関数そのものを変えると指摘している。 最適方策の集合が変わる例として、Pohlen らは Bowling を挙げた。

Toromanoff らは、クリップ付きで学習した Rainbow-IQN が Bowling でストライクやスペアを避けるようになった、と観察として報告している。クリップなしとの比較はしていない [Toromanoff SABER (2019/08), Section 6]。DQfD の著者は、人間が DQN を上回るゲームの多くはクリップが原因だったと書き、Private Eye を例に挙げている。そのうえで符号付き対数変換 sign(r)·log(1+|r|) を使ったが、クリップとの A/B は示していない [Hester DQfD (2017/04), Section "Experimental Setup"]。

### 2. 大規模エージェントは報酬クリップの代わりに何を使っているか

本調査の整理として、h(x) を使う系統(Ape-X DQfD、R2D2、NGU、Agent57、MEME、MuZero、Muesli)はいずれも DeepMind 所属の論文である。一方、単一 GPU やサンプル効率を重視する系統(Rainbow、BBF、BTR、EfficientZero、PQN)はクリップを残している(§2.6)。

Pohlen らは、報酬ではなく行動価値のスケールを縮める作用素 T_h を定義した。 h は可逆な圧縮関数で、ブートストラップ値を h⁻¹ で実空間に戻して報酬を足し、再び h で縮めた値をターゲットにする。

Pohlen らが使った h は sign(z)(√(|z|+1) − 1) + εz で、ε = 10⁻² である。 εz の項は h⁻¹ をリプシッツ連続にするために入れてある。

R2D2 は同じ h を ε = 10⁻³ で採用し、報酬をクリップせずに n-step ターゲットへ適用した。 以後の係数も 0.001 で、NGU・Agent57・MEME はハイパーパラメータ表に、MuZero は本文に書いている(§2.1 の表)。

h の理論保証は限定的である。 T_h^k Q が h∘Q* に収束するのは、h が線形の場合と、MDP が決定的(遷移と報酬が点測度)で h が狭義単調の場合だけだと示されている [Pohlen ApeXDQfD (2018/05), Section 3.2, Proposition 3.1]。確率的 MDP では、γ < 1/(L_h L_{h⁻¹}) のときに縮小写像になるが、固定点は h∘Q* とは限らない。しかも実際に使う γ はこの条件を超えていると、原典自身が書いている。

### 3. 報酬クリップの有無を統制した比較

クリップを外すと、ゲームごとの勝ち負けが大きく入れ替わる。 van Hasselt らは、改善した例として Gopher と Centipede を、悪化した例として Video Pinball と Star Gunner を挙げている。

同論文は、クリップを外すと問題の性質が変わって成績が下がる例として Time Pilot を挙げ、理由の 1 つを γ = 0.99 の近視眼性に求めている。

注: 原文の "ignoring the mothership." の直後にある脚注記号は省いた。

Double DQN の生スコア表の抜粋(評価は 30 分 = 108,000 frames のプレイ)を示す。クリップ版は Wang et al. 2016 からの転載値である。

PopArt-IMPALA のクリップありとクリップなしを比べた【画素換算】の値(人間正規化 %)は、Gopher が約 610 対約 2,100、Centipede が約 32 対約 170、Kangaroo が約 130 対約 460 でクリップなしが上だった。Demon Attack は約 4,100 対約 540、Video Pinball は約 510 対約 75、Time Pilot は約 420 対約 61、Frostbite は約 53 対約 5 でクリップが上だった。Bowling はどちらも約 30 だった [Hessel PopArtIMPALA (2018/09), Figure 8]。

A2C の著者も、集計ではクリップが上だった一方、クリップしたエージェントが準最適な方策にはまっていた Centipede などでは PopArt が有意に良かったと書いている [Hessel InductiveBiases (2019/07), Section 3.2]。

### 4. 割引率 γ の設定と高い γ での h(x) の効き方

Rainbow は DQN と同じ値だと明記している [Hessel Rainbow (2017/10), Table 4]。DQN 以前の ALE の SARSA(λ) ベースラインは、パラメータ探索の結果として γ = 0.999 を使っていた [Bellemare ALE (2012/07), Section 3.1.2, Appendix C]。Pohlen らは 0.99 を「経験的に全ゲームで安定学習できた最大値だった」と位置づけているが、この段落に引用文献はない。

BBF はスケジュールの終点を MuZero と EfficientZero に合わせた(§5.2)。ただし EfficientZero の表の値は 0.997⁴ と表記されている [Ye EfficientZero (2021/11), Table 6]。

0.997 を超える γ は、γ の族として使われている。 NGU は 0.99〜0.997、Agent57 は 0.99〜0.9999、MEME は 0.97〜0.9997 である。

PPO 系の RND は、Montezuma's Revenge のアブレーションで外的報酬側 γ_E = 0.999、内的報酬側 γ_I = 0.99 を選んだ。γ_E を 0.999 に上げると大きく改善し、γ_I を 0.999 に上げると悪化した。外的報酬は [−1, 1] にクリップしている [Burda RND (2018/10), Section 3.3, Appendix A.3 Table 2]。

他の原典は、推定・表現・分散の面から高い γ の問題を挙げている。 ただし多くは Atari の深層 RL ではなく、表形式・線形・MuJoCo などで検証された主張である。

### 5. 学習中に γ と n-step を変えるスケジュール

François-Lavet らは、DQN のハイパーパラメータを保ったまま、250,000 steps の epoch ごとに γ_{k+1} = 1 − 0.98(1 − γ_k) で γ を上げた。 1 − γ を epoch ごとに 0.98 倍する幾何的な更新である。γ の初期値は本文になく、図の軸では 0.95 から始まり、20M steps で 0.99 に達する(本調査の読み取り)。

γ も同じ指数スケジュールで 0.97 から 0.997 へ上げる。 根拠として挙げられているのは François-Lavet らの知見と、遅延報酬へ徐々に重みを与えるという一文である。

BTR は、γ を 0.97 から 0.997 へアニールしても有意差がなかったと報告している。 試したのは BattleZone の 1 ゲームだけである。

メタ勾配で γ を学習中に適応させる手法は、γ を方策と価値の入力に与えるかどうかで結果が大きく変わった。 Xu らは IMPALA(200M frames、57 ゲーム)で γ を適応させ、human starts の中央値を 211.9%(γ = 0.995)から 267.9% に上げた。ただし γ を入力に与えないと 183% に落ちた。

STAC では、γ だけを自己調整しても改善しなかった。 中央値は 243% から 240% になり、主ヘッドの γ は外側の損失の値 0.995 に収束した。著者は、適応が起きていることは、それが役立つことを意味しないと明記している。

### 6. 公式実装での確認

rlax の h は signed_hyperbolic として実装されており、ε の既定値は 1e-3 である。 逆関数 signed_parabolic も同じ既定値を持つ。

rlax の transformed_n_step_q_learning と transformed_retrace は、tx_pair の既定値が IDENTITY_PAIR である。呼び出し側がペアを渡さない限り変換はかからない [rlax (2026/09), rlax/_src/nonlinear_bellman.py:L221-L285]。

Acme の R2D2 設定は、報酬クリップを無効にし、h のペアを既定で使う。 既定値は R2D2 論文の値(クリップなし、γ 0.997、ε 10⁻³、n 5)と一致する。

SEED RL の R2D2 も、フラグの既定値を ε 1e-3、n 5、γ 0.997 としている。SEED 論文の表は、報酬クリップなしと書いている [seed_rl (2022/11), agents/r2d2/learner.py:L82-L87]、[Espeholt SEED (2019/10), Table 8, 9]。

Dopamine の Runner は既定で報酬を [−1, 1] にクリップする。 このオプションは 2020 年 11 月のコミット de73a04 で追加され、それ以前の Runner は無条件にクリップしていた [dopamine de73a04 (2020/11)]。Dopamine のオンライン Atari エージェントには、h 変換や PopArt の選択肢は見当たらない [dopamine (2026/03)]。

### 7. 批判と留保

h(x) の理論保証は、実際に使われる条件をカバーしていない。 確率的 MDP で縮小写像になる条件(本調査の計算で γ < 約 0.02)は実用の γ を大きく下回り、固定点も h∘Q* とは限らない(§2.2)。Muesli の著者は、この変換は確率的なリターンに対してバイアスがあると書いている。論文によって式の印字が異なり(εx の位置、√(x²+1)、h⁻¹ の 2 乗の欠落、ε の不一致)、どれが実装どおりかは本文から確認できない(§2.2)。

seed 数: Schaul らの 1 ヘッドの比較は各ゲーム 1 seed である。van Hasselt らと Pohlen らは seed 数を書いていない。

ゲーム数: R2D2 のアブレーションは 5 ゲーム、Pohlen らは 6 ゲームである。

転載値: van Hasselt らのクリップ版のスコアは他の論文からの転載である。

LightZero の第三者報告は 1 seed で学習長も不揃いである(§3.5)。

### 8. 総合評価

第一に、大規模エージェントの多くは、報酬クリップの代わりに可逆な価値の再スケーリング h(x) を使っている。 報酬ではなくターゲットの価値のスケールを縮める方式で、Pohlen らが導入し(ε = 10⁻²)、R2D2 が ε = 10⁻³ で採用した。NGU、Agent57、MEME、MuZero、Muesli がこれを引き継いでいる。

van Hasselt ら(Double DQN、200M frames、57 ゲーム)は、スコアを報告したのが Pop-Art 併用の比較だけである。

Schaul ら(R2D2 系)は、クリップだけを変えているが約 10⁹ frames・1 seed である。

Pop-Art、h、symlog でスケールを扱っても、集計ではクリップとほぼ同等かやや劣った。A2C・R2D2 系・PPO の 3 研究で、クリップが最良かそれに近いと報告されている。

ゲーム別の勝ち負けは大きく割れる。Gopher・Centipede・Ms. Pac-Man はクリップなしが上、Video Pinball・Star Gunner・Time Pilot・Q*Bert はクリップが上だった。R2D2 系の生の報酬では、Kung-Fu Master と Alien がほぼ 0 に落ちた【目視】。

### 9. 調査の限界

Hessel らの "On Inductive Biases" の採否は、OpenReview を取得できず確認していない。

図からの読み取り値は概算である。 【目視】【画素換算】の値(R2D2、Schaul らの図 12・15、PopArt-IMPALA の図 8、A2C・PPO の図)は、原文に数値がない。PopArt-IMPALA の図 8 は対数軸で、棒の判別は凡例の順序と塗りの色による。

R2D2 の Atari の総フレーム数は本文になく、第三者の記載は MuZero が 37.5B、NGU が 35 B で一致しない。

IQM と CI は公式 CSV から計算した。CI はゲーム層別ブートストラップの近似で、rliable とは実装が異なる。

RR=8 用の設定ファイルはリポジトリになく、RR=8 でのリセット回数は確認していない。

## 主要な概念

- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
