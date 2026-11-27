---
title: "Survey: 外的と内的の価値を分ける位置と、内的の n-step の和をエピソードの境でどう扱うか"
page_kind: サマリ
material: "[[reports/value_decomposition_head_nstep_survey_2026-11-27.md|value_decomposition_head_nstep_survey_2026-11-27]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
updated: 2026-11-27
---
# Survey: 外的と内的の価値を分ける位置と、内的の n-step の和をエピソードの境でどう扱うか

## 元の材料

[Survey: 外的と内的の価値を分ける位置と、内的の n-step の和をエピソードの境でどう扱うか](../../../../reports/value_decomposition_head_nstep_survey_2026-11-27.md) blob:d662c588da2f0a3d3dcb04d39c5509aa433f73db

## 要点

- Head だけを分ける形と Network ごとに分ける形を同じ条件で比べた事例は見つからなかった。Agent57 は 2 つの Network と合わせた報酬の 1 つの Network（NGU）を比べ、MEME は torso を共有したまま Head 2 つと 1 つ、合わせた損失と別々の損失を比べた（§2.5）。
- 分ける位置は 3 段ある。Agent57 は畳み込みの torso・全結合 512・LSTM 512・Dueling の Head まで全部を 2 組持つ [Badia Agent57 (2020/03), 付録 F 図 9・10]。MEME は torso と LSTM（隠れ 1024）を共有し、Head ごとに隠れ層 1024 を 2 つ持つ [Kapturowski MEME (2022/09), 付録 B.1〜B.4]。Burda らの公開実装と CleanRL は価値側の層まで共有し、最後の 1 出力の全結合だけを分ける。NGU は分けない（§2）。
- 理由は、Agent57 が報酬ごとの尺度と分散・optimizer の状態を分けること、MEME が Network ごとは計算が重く低い層の特徴を共有できないことである。Burda らは Head を分ける理由を書くが、分岐の位置の理由は論文にも実装にも書かれていない（§2.1〜§2.3）。
- 読んだ価値ベースの公開実装は、どれも n-step の和をエピソードの終わりで止め、次のエピソードの報酬を足さない。Unity ML-Agents の SAC は RND と curiosity の信号だけ done で bootstrap を切らないが、目標値は 1 step で、終端の step の観測から bootstrap する（§3.2）。
- deep_rl_zoo の Agent57 は 2 つの Network に同じ合わせた報酬と同じ終端の mask を渡す。Taïga らの Rainbow＋RND は内的を足した報酬を Dopamine の n-step 3 に渡し、和は最初の終端で止まる（§3.2）。
- 価値ベースで内的の n-step の和をつなぐ・切るを比べた実験は見つからず、比べた例は Burda らの PPO だけだった（§3.4）。
- Replay の系列がエピソードの境をまたぐ実装はある。Acme の系列の adder は次のエピソードへ続ける CONTINUE を持ち、deep_rl_zoo の系列の切り出しは既定でまたぐ（IMPALA が使う）。R2D2 系の設定と Agent57・MEME の原典はまたがない（§3.3）。

## 内容

### 1. 問いと結論

問いは 2 つ。外的と内的の価値を分ける位置（Head だけか Network ごとか）と、その比較・理由。価値ベースで内的の割引率や終端の扱いを分けたとき、n-step の和と Replay の系列をエピソードの境でどう扱うか。2026-11-27 に論文 5 本と公開実装 8 つを読み、web 検索を 6 回行った。

### 2. 外的と内的の価値を分ける位置と理由

Agent57: 2 つの Network は同じ構造で重みが重ならず、入力はどちらも観測と前の行動・前の外的報酬・前の内的報酬・方策の番号 j である。別々の transformed Retrace の損失で学び、target 方策は合わせた Q の greedy で共通にする。単純な勾配降下のもとでは合わせた報酬で 1 つの Q を学ぶのと目的が等価になる（付録 B）。理由に、尺度と分散、optimizer の状態、β ごとに違う値を表すための柔軟な表現を挙げる。random coin で効果が出るのはコインでエピソードが終わる設定だから、とも書く。結果は 3 seed（Agent57 は 6 seed）の平均で、分割を外すと 10 本の challenging set で 20% を超えて下がった [Badia Agent57 (2020/03), §3.1、§4、§4.2、付録 B・H.3]。

MEME: Head は行動の数 × 方策の組の数（16）を出し、方策の Head は別の MLP である。行動を決めるのは合わせた Q なので損失も合わせ、分解は厳密な意味を持たなくなると書く。Head 1 つでも同程度だった（付録 H.4。設定は書かない）。系列はエピソードの境をまたがず、割引率は方策の組ごとに 1 つ（最大 0.9997、最小 0.97）で合わせた報酬に掛ける [Kapturowski MEME (2022/09), §4 C2、付録 D・H.4、表 2]。

Burda ら: 収益が報酬について線形なので価値を別々に学んで足せ、Head を分けると割引率と終端の扱いを変えられ、非定常な内的報酬への追加の教師信号にもなると書く。両方をエピソード的にすると Head 1 つのほうがわずかに良かった（図 4、5 seed）。表 5 の既定は γ_E 0.999・γ_I 0.99 である [Burda RND (2018/10), §2.3、§3.2〜3.3、付録 A]。公開実装は畳み込み・全結合・GRU を方策と共有し、価値側だけの残差の全結合（fc2val）の上に 1 出力の全結合を 2 つ載せる [RND 公開実装 (2018/11)]。

ほかの公開実装: CleanRL は 2 つの価値だけが共有する全結合 448 の上に 1 出力の全結合を 2 つ載せる。ML-Agents は報酬の信号ごとに共有の Body の上の線形の Head と割引率を持つ。deep_rl_zoo の Agent57 は NGU と同じ Network を 2 つ作る（§2.4）。

同じ条件の比較: MEME の ablation は 8 本・3 seed・1B フレームで、torso を共有する形を Network ごとに戻す比較は無い（§2.5）。

### 3. 価値ベースでの割引率・終端・n-step の扱い

on-policy の参照として、RND の公開実装は `use_news` が偽（既定は 0）のとき内的の GAE を終端で切らず、CleanRL も内的の次の非終端を常に 1.0 にする（既定は外的 0.999・内的 0.99）。BYOL-Explore は V-MPO で、内的をエピソード的に扱う（§3.1）。

価値ベース: deep_rl_zoo の Agent57 の割引率は方策ごとに 1 つで 2 つの Network に共通である。Taïga らの設定は n-step 3、γ 0.99、RND の報酬の倍率 0.00005。Dopamine は終端の印なしにエピソードが終わる窓を抽選しない。DI-engine の NGU は Q が 1 つで、`ignore_done` が目標値の全体に掛かる。ML-Agents の終端を無視する印の説明は、終わりの無い報酬の正の偏りを抑えるためと書く。Acme の n-step の adder は葉ごとに n-step の収益を作り（成分ごとの割引率を表せる）、窓を次のエピソードへ延ばさない。Temporally-Extended ε-Greedy の R2D2＋RND は合わせ方も終端の扱いも書いていない（§3.2）。

系列: Acme の系列の adder は 4 つの扱いから選び、既定は ZERO_PAD、Acme の R2D2 は TRUNCATE。deep_rl_zoo の系列の切り出しは、またがないときは短いエピソードの系列を前の系列の末尾で埋める（§3.3）。

比較の実験: 内的のエピソード的・非エピソード的を比べたのは Burda らの PPO（非エピソード的のほうが探索した部屋が多い）だけで、BYOL-Explore は再実装した RND の部屋がエピソード的な設定で少なかったと書く（§3.4）。

### 4. 批判と懸念

MEME は Agent57 の 2 つの Network を計算と標本効率の点で批判し、NGU は表現の共有が活用側に干渉するという仮説を述べる。2 つを同じ条件で比べた結果は無い。deep_rl_zoo の Agent57 は、原典の「別々の報酬で別々の損失」とは違う。

### 5. 総合評価

分ける位置の 3 段、理由、Head の数と分割の有無の比較、n-step をエピソードの終わりで閉じる実装、PPO だけにある比較、をまとめる。

### 6. 調査の限界

Agent57 の付録 H.1 の seed、MEME の付録 H.4 の設定と図の値、RLeXplore の 2 Head の位置、Acme で CONTINUE を使う agent、DI-engine の `ignore_done` での系列の作り方、Taïga らの当時の Dopamine の版は確かめていない。被引用数は NGU 以外、2026-11-18 の値を写した。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
