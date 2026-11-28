# Survey: 新しさのボーナスで、長い移動の先や、ふだん通らない状態へ届くか

Date: 2026-11-27
Scope: 報酬の無い長い移動の先（横スクロールのゲームの先の場面など）へ、新しさのボーナスで行き着いた事例と、手前に稼げる報酬があるときの新しさの働き、Kung-Fu Master を名指しした探索の記述（Q11 の外の部分）。収束した価値ベースの方策が、ふだん通らない状態で粗くなる問題の扱いと、その測り方（Q9 の外の部分）。既にある報告（内的報酬の全体像 2026-11-18、エピソード内の新しさを価値ベースに載せる論点 2026-11-18、LLM の事前知識 2026-10-18）に書いてあることは調べ直さない。

## 対象の問い

- Q11 の外の部分: 報酬の無い長い移動（数百 step）の先にある場面（横スクロールのゲームのボスなど）へ、偶然に頼らず行き着かせることを、一生の新しさ（RND など）やエピソード内の新しさ（NGU の k 近傍、E3B など）のボーナスで達成した事例はあるか。稼げる報酬が手前にあるとき（密な報酬で局所的に稼ぐ方策に固まる）に、ボーナスが進む向きを作れるかを扱った研究はあるか。Kung-Fu Master を名指しで扱った探索の研究はあるか。
- Q9 の外の部分: 学習が収束したあとの方策が、ふだん通らない状態（開始位置を変えずに、途中から誘導された状態）からは目標に着けない、という問題を、DQN 系（価値ベース、Replay）で扱った研究はあるか。対処（探索役の方策と活用役の方策を分ける、状態の被覆を保つ、分布の外の状態での価値の粗さを測る）と、その測り方（訪問の被覆、ランダムな開始からの成功率など）は。環境の側で開始位置をばらす方法（exploring starts）を使わない前提のものを先にする。

## 結論

- **Q11（長い移動の先）**: 外的報酬を使わない好奇心だけで、横スクロールの Super Mario Bros の 11 の面を見つけてボスを倒すまで探索した報告（PPO、並列の環境 2048）と、固定の開始から 350 step 先の目標に ICM が 66% の Run で届いた報告（VizDoom、A3C）がある（§2）。どちらも手前に稼げる報酬が無い条件で、ICM の Mario は決まったキー入力の列が要る穴で止まった。エピソード内の新しさ（NGU の k 近傍・E3B）で同じことをした事例は見つからなかった。
- **Q11（手前に稼げる報酬があるとき）**: 報酬の密な Seaquest で、報酬だけの方策が「浮上しない」局所解に固まるのを新しさで抜けた報告（NS-ES、進化戦略）と、エピソード内の位置の新しさで中央値がほぼ 2 倍になった報告（DeepCS、A2C）がある（§3）。どちらも DQN 系ではない。EIPO は、内的報酬が気をそらして外的だけより悪くなるゲームがあることを 61 ゲームで示した。
- **Q11（Kung-Fu Master）**: 探索の分類の原典は KFM を「ε-greedy が得点の抜け道を見つける」易しいゲームの例に挙げる（§4.1）。Rainbow 上の比較では KFM の RND は 10M〜200M フレームのどの時点でも ε-greedy を下回り、PPO 上の EIPO の表では RND が外的だけを上回った（§4.2・§4.3）。KFM のボスや面の進みを探索の比較で測った研究は見つからなかった。
- **Q9（扱った研究と対処）**: 開始位置を変えずに途中から誘導した状態からの到達を DQN 系で測った研究は見つからなかった。近いものとして、DQN の Q は学習の極限の分布に従う状態・行動でだけ正確という解析（§5.1）、探索を減らすと Replay の経験が好む軌跡に集まり別の技能を忘れる報告（DDPG、§5.2）、広い分布で学ぶほうが収益が高い単体試験（§5.3）、他の方策のデータだけで学ぶと崩れ、自分の遷移を 10〜20% 混ぜると大きく減る tandem の実験（Double DQN、§5.4）、DQN の中で被覆の方策を分ける β-DQN（§5.5）がある。
- **Q9（測り方）**: 人の play から取った 100 の開始点からの評価（human starts）が「学習で見ていないかもしれない状態への汎化」を測る目的で使われたが、開始点を変える方式で、偏りの無い開始点を選びにくいという批判がある（§6.1）。訪問の被覆は β-DQN では訪問の分布図で定性的に示されただけだった。

## 調べた範囲と日付

- 日付: 2026-11-27。
- 読み方: arXiv・JMLR の PDF を取得し、手元でテキストに直して本文・付録を読んだ。表の値は PDF のページの画像で照合した（Taïga らの表 2〜5、EIPO の表 7）。Mega-Reward の表 1 はテキストの列がずれていたので、ar5iv の HTML と突き合わせて一致した列だけを使った。Burda らの図 8 は図を目で読んだだけで値は読み取っていない。
- web 検索の語: 「"Kung-Fu Master" exploration intrinsic reward Atari boss agent」「deep Q-learning policy fails from states off the trajectory fixed start state coverage gridworld "out-of-distribution states" exploration exploitation decoupled」「de Bruin Kober Tuyls Babuska "Experience Selection in Deep Reinforcement Learning for Control" JMLR」。ほかは論文の arXiv の番号で直接取得した。
- 被引用数は Semantic Scholar の API の値（取得日 2026-11-27）。de Bruin らの値は API の回数制限で取れなかった。

## 見つからなかったこと

- エピソード内の新しさ（NGU の k 近傍、E3B など）のボーナスで、報酬の密な横スクロールのゲームの先の場面（ボスなど）へ行き着かせた事例。
- KFM のボスへの到達・撃破・面の進みを、探索の手法どうしで比べた研究（KFM を含む比較は点だけを報告する）。
- 開始位置を変えずに、途中から誘導した状態（方策が通らない状態）からの成功率を、DQN 系の方策で測った研究。
- 探索役と活用役を分けたとき、活用役の「通らない状態からの到達」が上がるかを測った研究。

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. Q11: 報酬の無い長い移動の先へ届いた事例](#2-q11-報酬の無い長い移動の先へ届いた事例)
3. [3. Q11: 手前に稼げる報酬があるときの新しさ](#3-q11-手前に稼げる報酬があるときの新しさ)
4. [4. Q11: Kung-Fu Master を名指しした記述](#4-q11-kung-fu-master-を名指しした記述)
5. [5. Q9: 収束した方策が通らない状態で粗くなることの扱い](#5-q9-収束した方策が通らない状態で粗くなることの扱い)
6. [6. Q9: 測り方](#6-q9-測り方)
7. [7. 批判と懸念](#7-批判と懸念)
8. [8. 総合評価](#8-総合評価)
9. [9. 調査の限界](#9-調査の限界)
10. [10. 出典一覧](#10-出典一覧)

## 1. 対象と読み方

本文は「主張（筆者の要約）→ 出典の位置」の順で書く。原文の逐語は著作権への配慮から §4.1 の 1 か所に絞り、ほかは要約と位置（節・表・図）で示す。「この報告での読み取り」と書いた文は筆者の推論で、原典の主張ではない。

## 2. Q11: 報酬の無い長い移動の先へ届いた事例

### 2.1 横スクロール（Super Mario Bros）の好奇心だけの学習

Burda らは、外的報酬もエピソードの終わりの信号も使わず、前向きモデルの予測誤差（好奇心）だけで PPO を学ばせた。Super Mario Bros では、並列の環境を 128 から 2048 に増やすと探索が大きく広がり、11 の面を見つけ、隠し部屋を見つけ、ボスを倒すまでになった。[Burda Large-Scale (2018/08), §3.1 B・図 3(a)] 図 3(a) と図 9 の凡例は 1024 の環境と書き、本文・付録 A の 2048 と食い違う。[Burda Large-Scale (2018/08), 図 3(a)・付録 A・図 9]

原典は、終わりの信号を外すと死は単なる遷移の 1 つになるが、実際の Agent は死を避けたと書き、その理由を、死ぬと何度も見た開始へ戻るからと説明する。[Burda Large-Scale (2018/08), §2.3] 好奇心が外的報酬とよく揃う理由は、ゲームの設計者がプレイヤーを導く順序を作っているからかもしれないと推測し、揃わずランダムより悪くなるゲーム（Atlantis、IceHockey）もあると書く。[Burda Large-Scale (2018/08), §3.1 A]

### 2.2 固定の開始から遠い目標（ICM の VizDoom と Mario）

Pathak らの ICM（A3C）は、VizDoom の迷路で開始を固定し、最適な方策でも目標まで 270 step（sparse）と 350 step（very sparse）の部屋から始めた。報酬は目標に着いたときだけで、エピソードの上限は 2100 step である。very sparse では A3C と画素を予測する版は一度も届かず、ICM は 66% の Run で満点だった。[Pathak ICM (2017/05), §4.1]

報酬なしの Mario では、ICM は面 1 の 30% 超まで進み、敵を倒す・避ける行動を報酬なしで覚えた。原典は、敵にやられると見られる範囲が狭くなり好奇心が尽きるので、新しい所へ進むために避け方を覚えたと解釈する。[Pathak ICM (2017/05), §4.2「Mario: Learning to play with no rewards」] 30% 超で止まった理由は、38% の位置の穴を越えるのに 15〜20 のキー入力の決まった列が要り、越えられずに死ぬと、その先に探索できる世界があるという手がかりが得られないためと書く。[Pathak ICM (2017/05), §6]

## 3. Q11: 手前に稼げる報酬があるときの新しさ

### 3.1 Seaquest の局所解と新しさ（NS-ES）

Conti らは進化戦略（ES）に新しさの探索を足した。行動の特徴づけは各 step の RAM の列で、過去の方策の集まりとの距離を新しさにする。Seaquest では、報酬だけの ES は中央値 960 で頭打ちになり、底へ潜って魚を撃ち続け、息継ぎに浮上しない局所解に固まった。浮上には一時的に報酬を手放すことが要る。新しさだけの NS-ES は 5 本すべてで浮上を覚えて 1044.5、新しさと報酬を混ぜた NSR-ES は 2329.7 だった。[Conti NS-ES (2017/12), §4.2・図 3] 報酬を先に追い、停滞してから新しさの重みを上げる NSRA-ES は、重みを 50 世代ごとに 0.05 ずつ変える設定では 200 世代で同じ局所解から抜けず、10 世代ごとに変えると NSR-ES と並んだ。[Conti NS-ES (2017/12), §4.2] 局所解が無いゲームでは新しさの圧力が費用になり、NSR-ES は ES より悪かった。[Conti NS-ES (2017/12), §4.2]

### 3.2 エピソード内の位置の新しさ（DeepCS）

Stanton と Clune の DeepCS は、RAM から取った自機の位置で画面を格子に分け、そのゲームでまだ行っていない区画に入ると内的報酬 1 を出す。格子はゲームの初めに空にする（エピソード内の新しさ）。A2C で、1 Run は 640M フレーム、各 25 本である。[Stanton DeepCS (2018/06), §2] 原典は、学習の全期間で数える新しさは、すぐには報酬にならない状態への関心を早く失い、あとで要る状態へ戻らなくなりうると論じ、エピソード内の新しさはそれを避けると仮説を置く。[Stanton DeepCS (2018/06), §1]

報酬が密で 1 部屋だけの Seaquest では、格子は 700 フレームでほぼ埋まり、残るボーナスは届きにくい少数の区画だけになる。それでも最後の中央値は A2C の 1791 に対して 3443 だった。[Stanton DeepCS (2018/06), §3・図 2・図 4] 著者は、全体を動き回ることで、得点だけを追うと見つからない避け方や浮上を見つけた可能性を挙げつつ、確かめていないと書く。[Stanton DeepCS (2018/06), §3] 格子を入力から外しても内的報酬があれば Seaquest の成績はほぼ同じで、内的報酬を外すと下がった。[Stanton DeepCS (2018/06), §3・図 5] 原典は、DeepCS の探索で集めた状態を Replay に入れ、得点だけを最大にする DQN を off-policy で学ばせる案を挙げるが、実験はしていない。[Stanton DeepCS (2018/06), §2]

### 3.3 内的報酬が気をそらす（EIPO）

Chen らは、探索の易しいゲームでは Agent が内的報酬に気をそらされ、外的報酬だけで学ぶより悪くなると述べ、外的報酬の最適性を制約にして内的報酬の重みを自動で調整する EIPO を出した。PPO の上の RND を使い、61 ゲームで比べた。[Chen EIPO (2022/11), Abstract・§1] 61 ゲーム全体で P(RND > EO) は 0.49（信頼区間 0.46〜0.52）で、RND の上乗せは揃わない。[Chen EIPO (2022/11), §4.2・図 3(a)] 2 次元の迷路の例では、廊下にランダムに置いた点が新しさの源になり、RND はそれに引かれて外的だけに届かなかった。[Chen EIPO (2022/11), §4.1・図 2]

## 4. Q11: Kung-Fu Master を名指しした記述

### 4.1 探索の難しさの分類

Bellemare らは Atari のゲームを、局所的な探索（ε-greedy など）で高い得点に届く易しいゲームと、届かない難しいゲームに分け、易しい側を、ε-greedy が得点の抜け道（score exploit）を見つけるかで 2 つに分けた。KFM はその抜け道の側の例として BOXING と並べて名指しされる。[Bellemare Count-Based (2016/06), 付録 C.2・表 1] 抜け道の定義は次のとおりである。

```
maximizes the score without achieving the game's stated objective

ゲームの述べられた目的を果たさずに、得点を最大にする
```
[Bellemare Count-Based (2016/06), 付録 C.2]

原典は、この区別を本文では使っておらず、分類は粗いものと断る。KFM で何が抜け道かは書いていない。[Bellemare Count-Based (2016/06), 付録 C.2] Taïga らも同じ分類を付録 B の表 1 に載せる。[Taïga Bonus-ALE (2021/09), 付録 B・表 1]

### 4.2 Rainbow の上のボーナスの比較（Taïga ら）の KFM の値

Taïga らは Dopamine の Rainbow に RND・CTS・PixelCNN・ICM を足し、NoisyNets と ε-greedy を基準に、sticky actions 0.25 で比べた（曲線は 5 本の平均）。[Taïga Bonus-ALE (2021/09), §3.1〜§3.2] 付録の表（学習のフレーム数ごとの生の得点）の KFM は次のとおりで、RND はどの時点でも ε-greedy を下回った。[Taïga Bonus-ALE (2021/09), 表 2〜5]

| フレーム | ε-greedy | RND | CTS | PixelCNN | ICM | NoisyNets |
|---|---|---|---|---|---|---|
| 10M | 19824.1 | 17676.9 | 22668.2 | 21673.9 | 19199.3 | 19150.5 |
| 50M | 26929.2 | 22972.3 | 25573.2 | 26135.5 | 25883.8 | 30161.7 |
| 100M | 26459.1 | 23429.2 | 25450.0 | 27472.8 | 26899.8 | 34510.8 |
| 200M | 28741.8 | 24087.6 | 26439.4 | 27234.6 | 28168.3 | 39382.1 |

200M では、4 つのボーナスがどれも ε-greedy より低く、最も高いのは NoisyNets の 39382.1 だった。表の値が何本の平均か、評価の方策が何かは表の題に書かれていない。原典の結論は 60 ゲーム全体の話で、KFM を論じた文は無い。[Taïga Bonus-ALE (2021/09), 表 2〜5・§3.3]

### 4.3 PPO の上の RND（EIPO）の KFM の値

EIPO の付録は、RND が外的報酬だけの PPO（EO）を上回るゲームの組を挙げ、KFM はその組に入る。[Chen EIPO (2022/11), 付録 A.5] 表 7 の KFM の平均点は、EO 47266.5、RND 66954.2、Ext-norm RND 48944.5、Decay-RND 47403.2、Decouple-RND 45666.8、EIPO 48329.4 である。[Chen EIPO (2022/11), 付録 A.6・表 7] 多くのゲームの学習は 49152000 フレーム、各手法・各ゲームで 5 seed 以上、比較の統計は最後の 100 エピソードの外的収益の中央値である。[Chen EIPO (2022/11), 付録 A.4.1〜A.4.2] ボスや面の進みは報告していない。

### 4.4 報酬なしの内的報酬だけで学んだ KFM

Burda らの 48 ゲームの図 8 には KungFuMaster の枠があり、IDF と RF の好奇心の Agent の曲線は、ランダムの Agent の線より上にある（図を目で読んだ。値は読み取っていない）。[Burda Large-Scale (2018/08), 付録 B.1・図 8] Mega-Reward の表 1（外的報酬を使わない PPO の上の比較、80M step、最後の 50 エピソードの平均、画面の点は隠す）の KFM は、好奇心（Cur）3042、RND 652.1、Mega-Reward 352.4 だった。本文は Mega-Reward が 6 つの基準を大きく上回ると書くが、KFM の行では Cur と RND のほうが高い。[Song Mega-Reward (2019/05), §「Intrinsically-motivated play」・表 1] どちらもボスや面の進みは報告していない。

## 5. Q9: 収束した方策が通らない状態で粗くなることの扱い

### 5.1 DQN の Q は学習の分布の上でだけ正確（Ramaswamy・Hüllermeier）

Ramaswamy と Hüllermeier は、DQN の学習を力学系として解析し、学習した DQN が最適な Q を正確に近似するのは、学習の極限の状態・行動の分布に従う組だけだとした。その分布で出会わなかった組（探索と活用の釣り合いの悪さや、初期の重みで起きうる）では Q が粗くなりえて、そこで最適な行動は期待できないと書く。シミュレータで学ぶならその分布を経験から推定でき、学習が足りない場面を見つけられるかもしれないとも書く。理論の論文で、実験は無い。[Ramaswamy Asymptotic (2020/08), §4.1]

### 5.2 Replay のデータが好む軌跡に縮む（de Bruin ら）

de Bruin らは DDPG で振り子の振り上げを学ばせ、FIFO の Replay で探索を減らしていくと、バッファの経験が好む向きの軌跡のまわりに集まり、反対向きに振り上げる技能が失われた（破滅的な忘却）と報告する。好む向きが変わると、しばらく学んでいない領域の critic の値で更新することになり、学習が崩れた。すべての経験を残して TD 誤差で引く形（FULL DB[PER]）はこれをおおむね避けた。[de Bruin Experience-Selection (2018/08), §6.2・図 3・図 4] TD 誤差で引くと状態空間の全体で近似の精度がそろうので、学習と同じ開始の分布で試すなら望ましくないが、広い状態へ汎化させたいなら重要度の補正を掛けないほうがよいかもしれないと書く。[de Bruin Experience-Selection (2018/08), §3.3.2]

### 5.3 広い分布で学ぶほうがよい（Fu ら）

Fu らは、正解の Q を計算できる表形式の 8 領域（格子の世界などを含む）で、2 層の ReLU の NN の Q 学習を単体試験した。重みの分布の比較では、状態・行動の一様な重み、Replay の重み（学習中のすべての方策の平均）、TD 誤差の重みが、どの NN の大きさでも最も高い収益になり、方策自身の分布（on-policy）は常によいわけではなかった。分布のエントロピーが高いほど収益が高い傾向で、原典は状態の被覆を最大にする探索の動機になると書く。この試験では全状態が観測されていて、標本の不足は切り離している。[Fu Diagnosing (2019/02), §3.2〜§3.3・§7.2・図 8・図 9]

### 5.4 他の方策のデータだけで学ぶ難しさ（tandem の実験）

Ostrovski らは、Double DQN の能動の Agent と、同じ形の受け身の Agent を同じ batch の列で学ばせ、受け身の Agent が能動の Agent より大きく劣ることを 57 ゲームで確かめた（tandem 効果）。既定は Atari の 4 ゲーム、5 seed 以上、能動の Agent の 200 回 × 1M step である。[Ostrovski Tandem (2021/10), §2.1〜§2.2] 能動の Agent の ε を上げる（軌跡に沿った argmax でない行動の被覆が増える）と受け身の相対の成績が上がり、sticky actions は効かず、Replay を大きくすると劣化が遅れるだけだった。[Ostrovski Tandem (2021/10), §3.2・図 3・図 4] 能動の Agent を止めてその方策でデータを作り続けると、同じ重みから始めた受け身の Agent でも、易しいゲーム以外で成績が崩れた。[Ostrovski Tandem (2021/10), §3.2・図 5] 受け身の Agent 自身の遷移を 10〜20% 混ぜると tandem 効果は大きく減り、半々で消えた。[Ostrovski Tandem (2021/10), §3.2・図 7] 原典は、関数近似が固定のデータの分布の隙間で過度に外挿することが中心の要因で、相互作用（自分の誤りから学ぶこと）がその歯止めになると結論する。[Ostrovski Tandem (2021/10), §1・§5]

### 5.5 被覆を作る方策を DQN の中で分ける（β-DQN）

Zhang らの β-DQN は、Replay から状態ごとの行動の頻度（β）を学び、まれな行動を選ぶ被覆の方策、過大評価を確かめる方策、純粋な活用の方策を組にして、エピソードの中で切り替え、どれを使うかをスライディング窓のバンディットで決める。[Zhang β-DQN (2025/01), §4.1〜§4.3] MiniGrid の成功率の平均は DQN 0.49 に対して 0.83、MinAtar の収益の平均は 36.44 に対して 52.44 で、各 10 seed、5M step である。[Zhang β-DQN (2025/01), §5.2・表 1] 状態の被覆は CliffWalk の訪問の分布図で定性的に示しただけで、MiniGrid と MinAtar では成功率と収益で報告する。[Zhang β-DQN (2025/01), §5.1・図 3]

## 6. Q9: 測り方

### 6.1 人の play から取った開始（human starts）

Nair らは、エミュレータが決定的なので、人のプロの play から取った 100 の開始点から各 Agent を動かす評価を入れ、学習で見ていないかもしれない状態への汎化を測るためと書く。[Nair Gorila (2015/07), §5.2] Machado らは、この方式は大きく違う場面で評価できる一方、開始のあとは決定的なままで、行動列を覚える Brute 型でも点が出て、意味があり偏りの無い開始点を選ぶのが難しく、開始点ごとの点も比べられないと書く。[Machado Revisiting-ALE (2017/09), §5.3] 開始点を変える方式なので、問いの「開始位置を変えない」前提には当たらない。

### 6.2 開始の状態を学習用と試験用に分ける（Zhang ら）

Zhang らは、初期状態の標本を学習用と試験用に分け、学習用だけで学んだ A3C の Agent を試験用で測る手順を迷路で使った。ここでの初期状態は迷路の構成（level）で、同じ迷路の別の開始位置ではない。学習での成績が同じく最適でも試験での成績は大きく違い、評価で環境に乱れを入れる方法（ランダムな開始、sticky actions など）は過学習を防ぐとも見つけるとも限らないと報告する。[Zhang Overfitting (2018/04), §1・§3]

### 6.3 価値の不確かさを表す仕組み（randomized prior）

Osband らは、深層学習の不確かさの推定の多くは逐次の意思決定に向かず、bootstrap は観測したデータから来ない不確かさを表す仕組みを持たないとして、学習しない乱数の prior の NN を各アンサンブルの成員に足す方法を出した。要旨だけを読み、本文の実験は読んでいない。[Osband Randomized-Prior (2018/06), 要旨]

## 7. 批判と懸念

- **基盤の違い**: §2〜§3 の到達の事例は PPO・A3C・A2C・進化戦略の上で、DQN 系での同じ結果は見つからなかった。Rainbow の上では、KFM を含め多くのゲームでボーナスが ε-greedy を上回らない（§4.2）。
- **外的報酬が無い条件**: Mario の到達（§2.1）は外的報酬も終端の信号も使わない条件で、手前に稼げる報酬がある条件とは違う。
- **RAM の位置への依存**: NS-ES と DeepCS は RAM から取った状態を新しさに使い、画素から学ぶ方法ではない。[Conti NS-ES (2017/12), §4.2；Stanton DeepCS (2018/06), §2]
- **点だけの比較**: KFM を含む比較（§4.2〜§4.4）は点だけを報告し、ボスや面の進みは測っていない。
- **受け身の学習の崩れ**: 探索役のデータで活用役を学ばせる構成は、tandem の結果（§5.4）からは、活用役自身のデータが少ないと崩れうる。原典 DeRL 系の批判（方策のずれによる不安定）は既にある報告の §4.3 にある。

## 8. 総合評価

1. **長い移動の先へ届いた事例**: ある。ただし手前に稼げる報酬が無い条件で、PPO・A3C の上である（§2）。決まった行動の列を要する箇所（ICM の Mario の穴）は好奇心でも越えなかった。
2. **手前に稼げる報酬があるとき**: 新しさが局所解を抜けた事例は Seaquest にあり（§3.1・§3.2）、内的報酬が気をそらす事例も 61 ゲームの比較にある（§3.3）。どちらが起きるかはゲームと重みの調整に依る。
3. **Kung-Fu Master**: 分類の原典は KFM を得点の抜け道のあるゲームに入れる（§4.1）。点の比較では、Rainbow で RND が ε-greedy を下回り、PPO（EIPO）では上回った（§4.2・§4.3）。この報告での読み取り（推論）: 学習器と正規化で RND の向きが変わるので、KFM で RND が効くかは公表値からは一般化できない。
4. **通らない状態で粗くなる問題（DQN 系）**: 理論（§5.1）、Replay の構成（§5.2・§5.3）、他の方策のデータからの学習（§5.4）の 3 つの方向に材料がある。直接の対処として DQN の中で被覆の方策を分ける β-DQN（§5.5）があるが、通らない状態からの到達は測っていない。
5. **測り方**: human starts（§6.1）と初期状態の分割（§6.2）は開始点を変える。途中から誘導した状態からの成功率を測る手順は見つからなかった。この報告での読み取り（推論）: 開始位置を変えない前提では、学習の分布（訪問の被覆）を推定する案（§5.1）と、受け身の Agent との方策の食い違い（§5.4 の原典は Replay から一様に引いた状態の 75% 超で greedy の方策が食い違ったと書く。[Ostrovski Tandem (2021/10), §2.2]）が、粗さを測る候補になる。

## 9. 調査の限界

- **読めなかったもの・浅く読んだもの**: Osband らは要旨だけ。Nair らは ar5iv の HTML の評価の節だけ。β-DQN は HTML の前半（約 10 万字）だけで、付録の後半は読んでいない。Large-Scale Study の付録の図 8 は値を読み取っていない。
- **表の読み取り**: Mega-Reward の表 1 は、テキストに直すと Sto・Div の列が行をまたいでずれていたので、KFM の行は Emp・Cur・RND・Dir・Meg の列だけを、ar5iv の HTML と一致した値で使った（本文では Cur・RND・Meg だけを書いた）。Taïga ら・EIPO の表はページの画像で照合した。
- **原典の中の食い違い**: Large-Scale Study の Mario の環境の数（本文・付録は 2048、図 3(a)・図 9 の凡例は 1024）。β-DQN の step 数（本文 5M、表 2 は 2M とする記述がある、と取得の要約が示したが原文で確かめていない）。Large-Scale Study の好奇心がランダムを上回る割合（本文は IDF 75%・RF 70%、図 8 の題は IDF 約 71%・RF 約 67%）。
- **版**: Ramaswamy・Hüllermeier は arXiv v2（2021-04）の題で読み、Semantic Scholar には v1 の題（Theory of Deep Q-Learning: A Dynamical Systems Perspective）で載る。Machado らは arXiv v2（2017-12）を読み、雑誌の版と照合していない。Fu らは arXiv v1 を読み、会議の版と照合していない。
- **wiki の置き場の判断**: KFM の値（§4）は題材の kung_fu_master に置く。Mario・VizDoom・NS-ES・DeepCS（§2〜§3）は内的報酬の比較の intrinsic_novelty に置く（NS-ES は内的報酬ではないが、新しさを扱ういちばん近いページとした）。Replay の構成の 3 本（§5.2〜§5.4）は、Replay のデータの持ち方を扱う replay_ratio_lane_window に置く（tandem は探索役と活用役の分離にも関わるが、主張は Replay のデータの出どころにある）。β-DQN（§5.5）は行動の頻度を数える点で pseudo_counts に置く。Ramaswamy ら（§5.1）と human starts（§6.1）は評価の条件の evaluation に置く。Zhang ら（§6.2）と Osband ら（§6.3）は、問いへの寄与が小さいので wiki の項目にしない。

## 10. 出典一覧

[Bellemare Count-Based, 2016/06] Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos（Google DeepMind）. "Unifying Count-Based Exploration and Intrinsic Motivation." NIPS 2016. 被引用 1806. https://arxiv.org/pdf/1606.01868

[Burda Large-Scale, 2018/08] Burda, Edwards, Pathak, Storkey, Darrell, Efros（OpenAI、UC Berkeley、Univ. of Edinburgh）. "Large-Scale Study of Curiosity-Driven Learning." arXiv v1（Preprint）. 被引用 812. https://arxiv.org/pdf/1808.04355

[Chen EIPO, 2022/11] Chen, Hong, Pajarinen, Agrawal（MIT Improbable AI Lab、Aalto Univ.）. "Redeeming Intrinsic Rewards via Constrained Optimization." NeurIPS 2022. 被引用 51. https://arxiv.org/pdf/2211.07627

[Conti NS-ES, 2017/12] Conti, Madhavan, Such, Lehman, Stanley, Clune（Uber AI Labs）. "Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents." NIPS 2018（arXiv v3、2018-10）. 被引用 387. https://arxiv.org/pdf/1712.06560

[de Bruin Experience-Selection, 2018/08] de Bruin, Kober, Tuyls, Babuška（Delft Univ. of Technology、DeepMind、Univ. of Liverpool）. "Experience Selection in Deep Reinforcement Learning for Control." Journal of Machine Learning Research 19(9). 被引用は取れなかった. https://jmlr.org/papers/volume19/17-131/17-131.pdf

[Fu Diagnosing, 2019/02] Fu, Kumar, Soh, Levine（UC Berkeley）. "Diagnosing Bottlenecks in Deep Q-learning Algorithms." arXiv v1（Semantic Scholar は ICML 2019 とする）. 被引用 159. https://arxiv.org/pdf/1902.10250

[Machado Revisiting-ALE, 2017/09] Machado, Bellemare, Talvitie, Veness, Hausknecht, Bowling（Univ. of Alberta、Google Brain、Franklin & Marshall College、DeepMind、Microsoft）. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." arXiv v2（Semantic Scholar は JAIR とする）. 被引用 657. https://arxiv.org/pdf/1709.06009

[Nair Gorila, 2015/07] Nair, Srinivasan, Blackwell ほか（Google DeepMind）. "Massively Parallel Methods for Deep Reinforcement Learning." arXiv. 被引用 546. https://ar5iv.labs.arxiv.org/html/1507.04296

[Osband Randomized-Prior, 2018/06] Osband, Aslanides, Cassirer. "Randomized Prior Functions for Deep Reinforcement Learning." arXiv（Semantic Scholar は NeurIPS 2018 とする）. 被引用 454. https://arxiv.org/abs/1806.03335

[Ostrovski Tandem, 2021/10] Ostrovski, Castro, Dabney（DeepMind、Google Research Brain Team）. "The Difficulty of Passive Learning in Deep Reinforcement Learning." NeurIPS 2021. 被引用 76. https://arxiv.org/pdf/2110.14020

[Pathak ICM, 2017/05] Pathak, Agrawal, Efros, Darrell（UC Berkeley）. "Curiosity-driven Exploration by Self-supervised Prediction." ICML 2017. 被引用 3229. https://arxiv.org/pdf/1705.05363

[Ramaswamy Asymptotic, 2020/08] Ramaswamy, Hüllermeier. "Deep Q-Learning: Theoretical Insights from an Asymptotic Analysis." arXiv v2（2021-04）. 被引用 4. https://arxiv.org/pdf/2008.10870

[Song Mega-Reward, 2019/05] Song, Wang, Lukasiewicz, Xu, Zhang, Wojcicki, Xu（Univ. of Oxford、Beihang Univ.、Hebei Univ. of Technology、Lighthouse）. "Mega-Reward: Achieving Human-Level Play without Extrinsic Rewards." AAAI 2020（arXiv v4、2019-11）. 被引用 19. https://arxiv.org/pdf/1905.04640 （表 1 の照合に https://ar5iv.labs.arxiv.org/html/1905.04640 ）

[Stanton DeepCS, 2018/06] Stanton, Clune（Univ. of Wyoming、Uber AI Labs）. "Deep Curiosity Search: Intra-Life Exploration Can Improve Performance on Challenging Deep Reinforcement Learning Problems." NIPS 2018 Deep RL Workshop（arXiv v3）. 被引用 42. https://arxiv.org/pdf/1806.00553

[Taïga Bonus-ALE, 2021/09] Taïga, Fedus, Machado, Courville, Bellemare（MILA、Google Brain）. "On Bonus-Based Exploration Methods in the Arcade Learning Environment." ICLR 2020（arXiv 2021-09）. 被引用 75. https://arxiv.org/pdf/2109.11052

[Zhang β-DQN, 2025/01] Zhang, Bai, Xiao, Gao, Xu, Müller（Univ. of Alberta・Amii、Shanghai Jiao Tong Univ.、CUHK-Shenzhen、Huawei、CASIA）. "β-DQN: Improving Deep Q-Learning By Evolving the Behavior." AAMAS 2025. 被引用 9. https://arxiv.org/html/2501.00913v1

[Zhang Overfitting, 2018/04] Zhang, Vinyals, Munos, Bengio（Google）. "A Study on Overfitting in Deep Reinforcement Learning." arXiv. 被引用 451. https://arxiv.org/pdf/1804.06893
