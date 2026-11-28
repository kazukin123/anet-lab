---
title: Making Deep Q-learning Methods Robust to Time Discretization — 詳細解説
page_kind: サマリ
material: "[[reports/paper/tallec2019_time_discretization_paper_2026-10-03.md|tallec2019_time_discretization_paper_2026-10-03]]"
material_kind: 論文解説
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# Making Deep Q-learning Methods Robust to Time Discretization — 詳細解説

## 元の材料

[Making Deep Q-learning Methods Robust to Time Discretization — 詳細解説](../../../../../reports/paper/tallec2019_time_discretization_paper_2026-10-03.md) blob:4ffd7e7a60df27288f2f2a89aac1c5fbf2757a67

## 要点

- 本論文はそのうち時間離散化への敏感さを扱う。著者の例では、観測と行動を毎秒 10 回から 50 回に増やす場面である。反応が速くなるのだから性能は上がりこそすれ下がらないはずだが、状態行動価値を推定する DQN [Mnih et al. (2015)] や DDPG [Lillicrap et al. (2015)] は時間刻みの変更にまったく頑健でない [p.1, Section 1]。直観的な理由は、1 つの行動を保つ時間が短いほど、その行動がリターンにもたらす差も小さくなることにある。最適行動価値 $Q^{\ast}(s,a)$（行動 $a$ をとってから最適に振る舞ったときの価値）でみると、行動どうしの差は時間刻みとともに縮み、割引率を適切に調整してもこの縮みは残る。時間刻みが無限小の極限では Q 関数は行動を区別できなくなる（Theorem 2）。この議論は連続時間強化学習 [Doya (2000)]、[Baird (1994)] の枠組みで厳密にできる [p.1, Section 1]。
- 連続時間を選ぶのは理論が明快になるからで、多数の小さな行動の積み重ねで価値が決まる設定にも問題意識は当てはまり、適切なスケーリングや再パラメータ化がなければ学習が失敗しうると著者は論じる [p.1, Section 1]。目標は、時間刻みの変更にできるだけ不変で、時間刻みを 0 に近づけた極限が存在するアルゴリズムであり、この要請からアーキテクチャ・探索方策・学習率のスケーリングが決まる。時間刻みのオーダーが分かれば、多くのハイパーパラメータの刻みへの依存関係・オーダーについて指針が得られる [p.1–2, Section 1]。
- 時間の数え方には、ステップ数 $k$ と物理時間 $t = k\,\delta t$ の 2 つがある。一般に 1 ステップあたりの割引係数が $\gamma$ のエージェントが見通す範囲（時間ホライズン）はおよそ $1/(1-\gamma)$ ステップで、物理時間ではその $\delta t$ 倍になる [p.3, Section 3.1]。(5) のスケーリングのもとでは
- Q 学習系の手法は小さな時間刻みで学習に失敗することが理論と実験の両面から示され、連続時間の解析にもとづいて時間刻みの変更への頑健性を改善した実用的なオフポリシー深層 RL アルゴリズムが作られ、その頑健性が実験で確かめられた、と著者はまとめる [p.9, Section 7]。
- 理論は逐次的なサンプルを前提としており、実験で使った固定サイズのリプレイバッファは厳密には不変性を崩す [p.8, Section 6]。

## 内容

### 1. はじめに（Introduction）

[Baird (1994)] を土台に、準連続時間では Q 関数が V 関数へ縮退し、標準的な Q 学習がうまく振る舞わないことを形式的に示す。

論文の「collapse」には 2 つの意味がある。アルゴリズムについては学習の破綻を指し、Q 関数については行動に依存しない V 関数に一致してしまうことを指す。本解説では前者を「破綻」、後者を「縮退」と書き分ける。

### 2. 関連研究（Related Work）

出発点は [Baird (1994)] の Advantage Updating である。Baird は小さな時間刻みで Q 学習が破綻することを指摘し、価値の部分とアドバンテージの部分を $\delta t$ に応じて別々にスケーリングする方法を提案したが、検証は 2 次・線形の問題に限られていた [p.2, Section 2]。本論文はこれを 5 つの方向に広げる。アドバンテージの正規化を学習させずにパラメータ化 (27) で満たすこと、深層ネットワークで多様な環境に初めて適用すること（Deep Advantage Updating）、Q 学習の破綻と、適切にスケーリングした Advantage Updating が縮退しないことの一般的な証明、$\delta t$ に不変な探索、そして頑健性の厳しい実験的検証である [p.2, Section 2]。

対象はオフポリシー手法に限られる。A3C [Mnih et al. (2016)]、PPO [Schulman et al. (2017)]、TRPO [Schulman et al. (2015)] などのオンポリシー手法が特定の設定で不変になりうるかは範囲外とされる [p.2, Section 2]。Dueling architecture [Wang et al. (2015)] も価値とアドバンテージを分けるが、学習は両者を足した Q に対して通常どおり行うのでスケーリングは変わらず、小さな $\delta t$ で不変にはならない [p.2, Section 2]。

### 3. 準連続時間の強化学習（Near Continuous-Time Reinforcement Learning）

多くの RL 環境は、連続時間の系を一定の時間刻みで区切ったものである。Mujoco の環境 [Lillicrap et al. (2015)] や OpenAI Gym の classic control [Brockman et al. (2016)] では刻みはシミュレータが決め、実機ではカメラのフレームレートやアクチュエータの応答速度が刻みの下限を決める [p.2, Section 3]。刻みは主に計算資源の制約なので、細かくしても性能は落ちないはずであり（最適化は難しくなりうる）、アルゴリズムは $\delta t\to0$ でも機能し続けるべきだと著者は論じる。刻みに不変なアルゴリズムは、時間ホライズンに関わるハイパーパラメータに良い既定値を与えることにもつながる [p.2–3, Section 3]。刻みの小さい離散化環境を、本論文は準連続（near-continuous）環境と呼ぶ [p.3, Section 3]。

$\mathcal{M}_{\delta t}$ では、状態 $s$ で行動 $a$ を選ぶと、$a$ を時間 $\delta t$ だけ保って (1) を解いた先の状態に移る [p.3, Section 3.1]。頑健であるための必要条件は $\delta t\to0$ でも機能し続けることなので、関係する量が意味のある極限をもつように定義していく。本文の定理は半形式的な主張で、厳密な主張と仮定は補足資料（arXiv 版では付録 A）にある [p.3, Section 3.1]。

### 4. 連続時間極限をもつ強化学習（Reinforcement Learning with a Continuous-Time Limit）

を導入する [p.4, Section 4.1]。差が $O(\delta t)$ なので割っても有限に残る。価値とアドバンテージとでは $\delta t$ に対する大きさのオーダーが違う、という点を明示的に分けた形である。

各エポックで、探索方策（OU ノイズ $\nu_{\delta t}$ を使う (31) の方式）により nb_steps 回行動し、遷移 $(s,a,r,d,s')$ をバッファ $\mathcal{D}$ に貯める。$d$ はエピソード終了の信号である。

### 5. 実験（Experiments）

実験の目的は、DAU が個別の調整なしに広い範囲の時間刻みで働き、標準的な深層 Q 学習はそうでないことを示すことである。比較相手は連続行動では DDPG、離散行動では DQN で、オンポリシー手法は扱わない [p.6, Section 5]。

設定 1（本文の図。キャプションの表記は unscaled）: 割引率だけを $\gamma^{\delta t}$ とスケーリングし、報酬の倍率と学習率は基準の刻み $\delta t_0 = 10^{-2}$ に合わせた値に固定する。どの $\delta t$ でも受け取る報酬の値は $r\,\delta t_0$ である。

### 6. 考察（Discussion）

提案手法は理論上は刻みに不変で、単純な移動タスクなどでは実際に頑健さを改善したが、改善の余地が残る環境もある。著者は、この理論と実践のずれの原因候補を 3 つ挙げる [p.7–8, Section 6]。その前提として、Alg. 1 は時間刻み $\delta t$ を知っている必要があるが、フレームレート、アクチュエータや観測の周波数、物理シミュレータの刻みなどから通常は分かる、と述べている [p.8, Section 6]。

Theorem 5 は 1 本の軌道から順にサンプルする場合の主張だが、実験では相関を避けるためにリプレイバッファからサンプルしている [p.8, Section 6]。著者はこの考察では、1 本の探索軌道で埋めていく固定サイズの循環バッファを念頭に置く。一方、実験設定 B.1 は 256 環境を並列に動かすとしている [p.16, Appendix B.1]。バッファは全刻みで同じ大きさ（容量 $10^6$ [p.16, Appendix B.1]）だったため、バッファ内のサンプルが物理時間でどれだけ古いかは刻みによって変わる。厳密にはこれで不変性が崩れている。物理時間で一定の記憶量になるよう、大きさを $1/\delta t$ に比例させる案（メモリは多く要る）が挙げられている。

### A. 証明（Proofs）

を考える。$B(0) = V^\pi(s)$、$T\to\infty$ で $B(T)\to V^{\pi'}(s)$ であり、微分すると $\dot B(T) = \gamma^T A^\pi(s_T,\pi'(s_T))\ge0$ (81)–(82) なので $B$ は増加する [p.12, Appendix A]。付録冒頭で強化学習に固有とされた議論はこれである。

証明は、刻み $\delta t^2$ の軌道を $\lfloor 1/\delta t\rfloor$ ステップ（物理時間でほぼ $\delta t$）ずつまとめて見るものである。その間の変化を平均ダイナミクス $f$ による部分と、期待値 0 で $\mathbb{E}\lVert\xi\rVert^2\le\sigma^2\delta t^3$ (108) を満たす誤差 $\xi$ とに分け、$f$ のオイラー近似から $\varepsilon$ 以上ずれる確率が $O(\sqrt{\delta t})$ であること (124) から確率収束（ずれが一定以上になる確率が 0 に向かうこと）を示す。最後に Lemma 1 と同じ型の議論で連続な方程式への収束を結論する [p.13–14, Appendix A]。

### B. 実装の詳細（Implementation details）

Algorithm 2 [p.14, Algorithm 2] は Algorithm 1 に方策ネットワーク $\pi_\phi$ を加えたものである。予測値に含まれる $\max_{a'}\bar A_\psi(s,a')$ を $\bar A_\psi(s,\pi_\phi(s))$ に置き換え、$\pi_\phi$ は $\partial_a\bar A_\psi(s,\pi_\phi(s))\,\partial_\phi\pi_\phi(s)$ のバッチ平均を使って学習率 $\alpha^\pi\delta t$ で、アドバンテージを増やす方向へ更新する。通常の最小化 optimizer に渡すなら、この actor 勾配には負符号が必要である。これは本文の最大化目的と勾配式からの整理で、原典の擬似コードは optimizer の更新方向を明記していない。

### C. 追加結果（Additional results）

付録 C は、追加結果を示すという短い導入文と図からなり、図別の追加考察はない [p.16, Appendix C]。

Figure 6 [p.18, Figure 6]: Figure 2 と同じ 5 つの環境で、DAU と設定 2 のベースラインの学習曲線を比べたもの。キャプションと青系の凡例は DDPG と表記するが、Cartpole の手法名については Section 5 と同じ不一致がある。

### 論文が述べている限界

Half-Cheetah では DAU も刻みに不変とは言えず、Cartpole はノイズが大きく結果を解釈しにくい [p.7, Section 5]。改善の余地が残る環境があることは著者も認めている [p.7–8, Section 6]。

アルゴリズムを使うには時間刻み $\delta t$ を知っている必要がある [p.8, Section 6]。

### 論文内の表記の食い違い

Theorem 9 の (98) は主項と剰余で $\delta t$ のべきが異なり、(99) の指数 $(\beta-1)/3$ も、(94) と $K_{\delta t}$ の定義から再計算した $(\beta-1)/2$ と一致しない。両者は付録 Theorem 9 の解説で区別した [p.12–13, (94)–(100)]。

### キューの論点への答え

reports/INDEX.md のキューに書かれた問いに、論文の記述の範囲で答える。論文が扱っていない点はそう明記する。

## 主要な概念

- [行動の時間尺度](../../../concepts/methods/temporal_abstraction.md)
- [報酬尺度と割引率](../../../concepts/methods/reward_scale_discount.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
