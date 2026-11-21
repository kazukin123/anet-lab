# Survey: エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点

Date: 2026-11-18
Scope: NGU・E3B などのエピソード内の新しさのボーナスを、Replay を使う off-policy の価値ベースの学習に載せるときの論点を、NGU 系（NGU・Agent57・MEME）の実装、NGU 以外の手法の off-policy での扱い、Replay に入れた内的報酬の古さと外的・内的の分け方の 3 つの観点で整理する。手法の全体像は [内的報酬の手法の全体像](intrinsic_reward_landscape_survey_2026-11-18.md) にあり、本冊はその続きとして実装の論点に絞る。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。著作権への配慮から原文の逐語は引かず、要約と位置（節・付録・表・図・式、実装はファイルと行）で示す。本文は arXiv・ar5iv の HTML 版や PDF、公開リポジトリから取得した要約を経ており、表の数値は PDF で 1 語ずつ照合していない（§8）。被引用数は Semantic Scholar の値で、取得日は 2026-11-18。調査は観点ごとに 3 本のサブエージェントで行い、本冊で統合した。

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. NGU 系の実装](#2-ngu-系の実装)
   - [2.1 エピソード内メモリ](#21-エピソード内メモリ)
   - [2.2 埋め込みの学習](#22-埋め込みの学習)
   - [2.3 内的報酬の保存](#23-内的報酬の保存)
   - [2.4 方策の組と価値の分け方](#24-方策の組と価値の分け方)
   - [2.5 再帰と系列](#25-再帰と系列)
   - [2.6 公開の再実装](#26-公開の再実装)
3. [3. NGU 以外のエピソード内ボーナスと off-policy](#3-ngu-以外のエピソード内ボーナスと-off-policy)
   - [3.1 原典の基盤アルゴリズム](#31-原典の基盤アルゴリズム)
   - [3.2 E3B の行列の費用と埋め込み](#32-e3b-の行列の費用と埋め込み)
   - [3.3 Replay からの計算し直し](#33-replay-からの計算し直し)
   - [3.4 一生とエピソード内の組み合わせ方](#34-一生とエピソード内の組み合わせ方)
   - [3.5 Atari の難探索での結果](#35-atari-の難探索での結果)
4. [4. Replay に入れた内的報酬の古さ](#4-replay-に入れた内的報酬の古さ)
   - [4.1 問題を名指しした原典と対処](#41-問題を名指しした原典と対処)
   - [4.2 外的と内的の価値の分け方](#42-外的と内的の価値の分け方)
   - [4.3 探索方策と活用方策の分離](#43-探索方策と活用方策の分離)
   - [4.4 正規化の尺度](#44-正規化の尺度)
   - [4.5 多段の収益とエピソードの区切り](#45-多段の収益とエピソードの区切り)
5. [5. 対処の形の一覧](#5-対処の形の一覧)
6. [6. 批判と懸念](#6-批判と懸念)
7. [7. 総合評価](#7-総合評価)
8. [8. 調査の限界](#8-調査の限界)
9. [9. 出典一覧](#9-出典一覧)

## 1. 対象と読み方

エピソード内の新しさのボーナスは、同じエピソードの中で見たかどうかで報酬を出し、エピソードの始めに記憶を空にする。PPO・IMPALA のような on-policy の手法では、rollout の中でその場で計算してすぐ学習に使える。Replay を使う off-policy の価値ベースでは、行動した時点の記憶で計算した報酬が Replay に残り、学習はあとで別の時点に行う。本冊は、この時間差から生じる論点（記憶の持ち方、報酬の保存と古さ、価値の分け方、多段の収益）を、原典がどう扱ったかで整理する。

## 2. NGU 系の実装

NGU・Agent57・MEME は R2D2 系の分散した価値ベースの学習で、エピソード内の新しさを使う代表例である。

### 2.1 エピソード内メモリ

- エピソードの始めにメモリを完全に空にする（MEME も同じ）。容量は 30,000 件で、満杯なら古いものから上書きする ring buffer、k 近傍の k は 10 である。Agent57 も同じ値を使う。MEME は容量と k を書いていない。[Badia NGU (2020/02), §2、付録 A.2、付録 F.2 表6；Badia Agent57 (2020/03), 付録 G 表3；Kapturowski MEME (2022/09), 付録 G.2・表2]
- 報酬は擬似カウントの逆数の平方根で近似し、kernel は k 近傍への距離の二乗をその移動平均で割った逆カーネルである。類似度の合計が上限を超えたら報酬を 0 にする。[Badia NGU (2020/02), §2 式2・式3、Algorithm 1]
- 計算費用は、メモリを学習の開始時に確保するので空間は一定、時間はメモリ全体への距離計算で O(メモリの大きさ × フレーム数) と書く。MEME は通信を減らすため近傍検索を actor 側で行うと書くが、定量的な費用は書かない。[Badia NGU (2020/02), 付録 A.2；Kapturowski MEME (2022/09), 付録 D]
- メモリを actor ごとに持つか lane ごとに持つかは、NGU・Agent57 の本文に明記がない。[Badia NGU (2020/02), §2・付録 A；Badia Agent57 (2020/03), 付録 E・G]

### 2.2 埋め込みの学習

- NGU の埋め込みは、2 つの埋め込みから行動を当てる逆モデルとして最尤で学習し、learner が Replay から抜き出した系列の最後の 5 フレームを使う。RND の predictor も同じ 5 フレームで学ぶ。[Badia NGU (2020/02), §2、付録 A、付録 H.1]
- MEME の行動予測の Network と RND も、learner が Replay の系列で学習する。埋め込みは 32 次元で、学習の頻度は書いていない。[Kapturowski MEME (2022/09), 付録 D・表2]
- 埋め込みが学習で変わると、メモリに入れた過去の埋め込みと新しい埋め込みの距離が変わる。この古さに触れた記述は、NGU・Agent57・MEME の読めた範囲にはなかった。[Badia NGU (2020/02)；Badia Agent57 (2020/03)；Kapturowski MEME (2022/09)]

### 2.3 内的報酬の保存

- NGU は actor が内的報酬を計算し、観測・行動・合成した報酬・内的報酬などを Replay に入れる。learner は保存された値をそのまま使い、内的報酬を Network の入力にもする。learner で計算し直すとは書いていない。[Badia NGU (2020/02), 付録 A]
- Agent57 も actor が内的報酬を計算し、外的報酬と別々に保存する。MEME は NGU の内的報酬と外的報酬を Replay の別の欄に持つ。[Badia Agent57 (2020/03), §2・付録 E；Kapturowski MEME (2022/09), 付録 D]
- 保存した内的報酬が古くなる問題には、3 本とも読めた範囲で触れていない（§4.1）。

### 2.4 方策の組と価値の分け方

- **NGU**: 32 個の (β, γ) の組を 1 つの Network で表し、組の番号の one-hot を入力に足す（UVFA）。β は 0〜0.3、γ は活用側 0.997 から探索側 0.99 で、内的報酬は密で値の幅が小さいので探索側の γ を低くすると説明する。actor ごとに固定の組を割り当て、損失は変換つきの Retrace（λ=0.95）。評価は β=0 の組で greedy に行う。[Badia NGU (2020/02), §3、付録 A 式4、付録 E]
- **Agent57**: 行動価値を外的と内的の 2 つの Network（同じ構造、別々の損失と optimizer の状態）に分け、actor ごとの sliding-window UCB のバンディットがエピソードの始めに組を選ぶ。評価は推定平均が最大の組で行う。[Badia Agent57 (2020/03), §3.1・§3.2、付録 B・D・E]
- **MEME**: 胴体を共有して外的・内的で別の Head を持ち、損失は合成した値にかける（Head 1 つでも同程度）。組は 16 個で、Q の Head が全組の値を出し、全組の損失を足す。損失は Soft Watkins Q(λ)、評価はバンディットの greedy な組と EMA のパラメータで行う。[Kapturowski MEME (2022/09), §4・式3〜5、付録 B.4・D・E・H.4]

### 2.5 再帰と系列

- NGU は、状態と行動から予測しにくく変わる報酬が問題を部分観測にしうるので、内的報酬を入力に入れ、エピソード内の履歴をまとめる再帰の状態を持つ、と説明する。系列は長さ 80 を 40 ずつ重ね、エピソードの境をまたがない。[Badia NGU (2020/02), §3、付録 A]
- Agent57 は系列 160 で LSTM の状態を保存し、MEME は系列 160・burn-in 0 で再帰の状態を推論サーバー側に持つ。[Badia Agent57 (2020/03), 付録 E・表3；Kapturowski MEME (2022/09), 表2・付録 D]

### 2.6 公開の再実装

- **RLeXplore（rllte の実装）**: NGU のエピソード内の項（PseudoCounts）のメモリを並列環境ごとのリストで持ち、終端（terminated か truncated）で空にする。off-policy では 1 step ごとに計算して Replay のバッチで部品を更新するが、エピソード内の手法は並列化しない環境だけに対応する。[Yuan RLeXplore (2024/05), 付録 A・C.1 表9・C.2；rllte 実装 (2025/07), pseudo_counts.py]
- **DI-engine の NGU（査読なしの OSS）**: 集めたデータは内的報酬を足さずに Replay に入れ、learner が sample するたびに RND とエピソード内の報酬を計算し直して、組の番号から引いた β で足す。エピソード内のメモリは、抜き出した系列の中でそれより前の step の埋め込みだけで、埋め込みはその時点の Network で計算する。[DI-engine NGU (2025/12), serial_entry_ngu.py、ngu_reward_model.py]

## 3. NGU 以外のエピソード内ボーナスと off-policy

### 3.1 原典の基盤アルゴリズム

- E3B は IMPALA（Habitat は DD-PPO）、Henaff et al. 2023 は IMPALA・DD-PPO・PPO（Montezuma's Revenge）、NovelD は PPO、RIDE は IMPALA、Episodic Curiosity は PPO を基盤にした。価値ベースや off-policy の実験は、これらの原典にない。[Henaff E3B (2022/10), §5.1・付録 C；Henaff Global-Episodic (2023/06), 付録 E；Zhang NovelD (2021/12), §4；Raileanu RIDE (2020/02), 実験設定；Savinov EC (2018/10), 実験節]
- NovelD（と前身の BeBold）の価値ベースの例は、外部報酬のない廊下の玩具環境で、深層 Q 学習とクリップを外した内的報酬を組み合わせた純粋な探索だけである。[Zhang NovelD (2021/12), §4、Tab.1；Zhang BeBold (2020/12), §5.2]
- RE3 は Rainbow（Atari）・RAD・Dreamer・A2C と組んだが、k 近傍はエピソード内ではなく Replay 全体の表現の中で取る。[Seo RE3 (2021/02), §3 式3、§4、Algorithm 1、付録 D]
- RLeXplore の off-policy の実験（SAC、Ant-UMaze）は Disagreement・RND・ICM だけで、エピソード内の手法（NGU・PseudoCounts・RIDE・E3B）は、Replay からランダムな遷移でなくエピソード全体を引く必要があるとして外し、対応は今後の課題とした。[Yuan RLeXplore (2024/05), 付録 G]
- SOFE は Stable-Baselines3 の DQN（Replay 100 万遷移）で DeepSea を扱った。E3B は IMPALA で最適化した。[Creus SOFE (2023/10), §5.1〜5.3、付録 A.4.1]
- E3B・NovelD・RIDE を DQN・Rainbow・SAC に載せた論文は、今回の検索では見つからなかった。

### 3.2 E3B の行列の費用と埋め込み

- E3B はエピソード内の埋め込みの外積の和に λI を足した行列 C を持ち、その逆行列をエピソードの開始で初期化する。埋め込みの次元はおおむね 256〜1024 で、毎 step の逆行列の計算は次元の 3 乗かかるので、rank-1 更新で 2 乗に下げる。実測は rank-1 更新が 767 FPS、毎回の逆行列が 256 FPS だった。[Henaff E3B (2022/10), §4.1 式1、Algorithm 1、付録 D.2]
- IMPALA の actor は 256 スレッドだが、actor ごとに C をどう持つか、その費用の記述は見つからなかった。[Henaff E3B (2022/10), 付録 Table 2]
- 埋め込みは方策と別に持ち、逆モデルと一緒に学習する。原典に埋め込みの古さの議論はない。Henaff et al. 2023 は、学習の初めにエピソード内のボーナスの大きさがいったん下がり、その時期が逆モデルの損失が落ち着く時期と重なると観察し、Montezuma's Revenge では逆モデルの埋め込みより固定の乱数の埋め込みのほうが良かったと書く。[Henaff E3B (2022/10), §4.2、Algorithm 1；Henaff Global-Episodic (2023/06), 付録 E.5・G.2]
- RE3 は、固定の乱数エンコーダなら 2 つの状態の距離が学習中に変わらずボーナスが安定するとし、低次元の表現を Replay に保存して観測を通し直さずに済ませる。[Seo RE3 (2021/02), §3]

### 3.3 Replay からの計算し直し

- エピソード内の回数や新しさは、そのエピソードのそれまでの文脈が要る。RLeXplore はこのため off-policy ではエピソード全体を引く必要があるとした。[Yuan RLeXplore (2024/05), 付録 C.1・G]
- SOFE は、回数・E3B の C などの十分統計量を状態に足すと、報酬が拡張した状態の決定的な関数になり、定常でマルコフになると述べる（統計量の更新が現在と次の状態だけで決まることが条件）。E3B の C は CNN で埋め込んで観測の特徴に連結する。Replay への保存や付け直しを明示的に論じた記述は、読めた範囲では見つからなかった。[Creus SOFE (2023/10), §4、§5.3、付録 A.3.1]
- RE3 はエピソードの文脈を使わず、更新時に引いたミニバッチの各サンプルについて Replay 全体の中で k 近傍を探す設計で、この問題を避ける。[Seo RE3 (2021/02), Algorithm 1]
- E3B・Henaff et al. 2023・RIDE・NovelD は、ボーナスを rollout 中にその場で計算し、Replay から計算し直すことを論じていない。[Henaff E3B (2022/10), Algorithm 1；Henaff Global-Episodic (2023/06)；Raileanu RIDE (2020/02)；Zhang NovelD (2021/12)]

### 3.4 一生とエピソード内の組み合わせ方

- Henaff et al. 2023 は、E3B の楕円ボーナスに RND・NovelD・AGAC の一生のボーナスを掛け算か重み付きの和でつないだ。MiniHack 16 タスクでは、和はどの指標でも E3B 単独を意味のある形で上回らず、E3B × RND と E3B × NovelD の積は中央値と IQM を有意に上げた。[Henaff Global-Episodic (2023/06), 式8、§4.2、Fig.6、付録 F]
- 積が効く理由として、一生のボーナスは学習とともに大きさが大きく下がり、エピソード内のボーナスは毎エピソード初期化されて下がらないので、和ではエピソード内の項が支配的になるが、積なら一生の項の効き目が大きさによらず残る、と説明する。[Henaff Global-Episodic (2023/06), §4.2、付録 G.2]
- NGU の k 近傍のエピソード内ボーナスを NovelD と組み合わせると E3B を使う版より大きく劣り、ユークリッド距離が大きさの大きい次元に支配されやすいためと解釈する。NGU 型の倍率のクリップとの直接の比較はない。[Henaff Global-Episodic (2023/06), 付録 G.3、Fig.9]
- NovelD は一生の新しさの差の正の部分に、エピソード内の初訪問の指示関数を掛ける。RND にこの制限を足すと改善し、NovelD から外すと成績が落ちた。[Zhang BeBold (2020/12), 式2・式3；Zhang NovelD (2021/12), Fig.7]
- RIDE は埋め込みの差の大きさをエピソード内の訪問回数で割り、回数の項を外すと最も難しいタスクを解けなかった。[Raileanu RIDE (2020/02), §3、付録 A.5]
- 価値ベースでの組み合わせ方の比較は見つからなかった。

### 3.5 Atari の難探索での結果

- エピソード内のボーナスを価値ベースで使い、Montezuma's Revenge・Pitfall の結果を報告した例は、NGU 系（R2D2 系）のほかに見つからなかった。
- 近い報告として、RE3 と Rainbow（Replay 全体の k 近傍）は Montezuma's Revenge を含む Atari 6 本で Rainbow 単独より学習が速かった。[Seo RE3 (2021/02), §4.3、Fig.13・15、付録 D]
- PPO の上では、NovelD が Montezuma's Revenge で約 20 億フレームに約 1 万点（RND の報告値は約 6,700）、探索した部屋は 23（RND は 17）。Henaff et al. 2023 では E3B が振るわず E3B × RND が RND にほぼ並び、RLeXplore では RND だけがまともな結果（160 点）を出した。[Zhang NovelD (2021/12), §4.4、Fig.11；Henaff Global-Episodic (2023/06), §3.5；Yuan RLeXplore (2024/05), 付録 D.5・F]

## 4. Replay に入れた内的報酬の古さ

### 4.1 問題を名指しした原典と対処

- DEEP は、新しさのボーナスを足す方式を off-policy で使うと、報酬を使う時点でなく集めた時点で計算するので、Replay が過去のいくつもの回数推定から計算したボーナスの混合になり、今のボーナスでなくその混合に対する収益を最大化して状態空間を覆うのが遅れる、と述べる。対処は、探索用の Q を更新するときにその時点のボーナスで報酬を計算し直し、探索方策を task 方策から切り離して速く追従させることである。task 方策は Replay の task の報酬だけで学ぶ。[Whitney DEEP (2021/01), §1、§3、§4.2〜4.3、Fig.3]
- DeRL は、内的報酬は探索が進むと減るように作られているので報酬の整形が非定常になり、学習が不安定になると述べ、探索方策（外的＋内的、on-policy の A2C）と活用方策（外的報酬だけのデータで学ぶ。DeDQN では Replay）を分ける。[Schäfer DeRL (2021/07), §1、§4、§5.1]
- NGU は、混ぜた報酬が予測できない形で変わると問題が部分観測になりうるとし、内的報酬を入力に入れ再帰の状態を持つことで対処する。保存値の古さを論じた記述は見つからなかった。Agent57 も同様である。[Badia NGU (2020/02), §3、付録 A；Badia Agent57 (2020/03), 付録 E]
- APT は Replay から引いた batch の中で k 近傍の報酬を計算し、内的報酬の非定常性は残る課題として挙げる。SOFE はボーナスの十分統計量を状態に足して定常にする（§3.3）。[Liu APT (2021/03), §3.1、結論；Creus SOFE (2023/10), Abstract、§2]
- 「Replay を短くして古さを抑える」対処を名指しで論じた原典は見つからなかった。

### 4.2 外的と内的の価値の分け方

- RND は、収益は報酬について線形なので外的と内的の価値を別々に学んで足せ、割引率の違う流れの合成にも使えると述べる。両方がエピソード的な設定では 2 Head は 1 Head より良くならなかったが、性質の違う流れを合わせるために以後は 2 Head にした。外的の割引率を 0.999 にすると大きく改善し、内的を 0.999 に上げると悪化した。[Burda RND (2018/10), §2.3、§3.2〜3.3、Fig.4・5]
- Agent57 は、2 Network に分けることは単純な勾配降下のもとでは混ぜた報酬の 1 本の Q と等価だが、各 Network が自分の報酬の尺度と分散に合わせられる点が違うとする。内的報酬の重みを上げると NGU の活用方策が解けなくなる格子の課題で、分けると重みへの頑健性が大きく上がり、分割を外すと難しい 10 ゲームの平均が 20% 超下がった。[Badia Agent57 (2020/03), §3.1、§4.2、付録 B、Fig.4・5]
- NGU は、Breakout と Beam Rider で R2D2 に届かない理由を、内的信号で学んだ表現が活用側の学習に少し干渉しているためと推測し、表現を共有しなければ改善するという仮説を述べる。[Badia NGU (2020/02), §4.2]
- RLeXplore は PPO で、価値を 2 つ持ち advantage を足す構成が報酬を単純に足す構成より良かったと報告する（Procgen-Maze）。[Yuan RLeXplore (2024/05), §5.2]

### 4.3 探索方策と活用方策の分離

- DeRL は、内的報酬の尺度と減衰率を振ってもほかの手法より頑健で、活用方策が最大 40% 少ないエピソードで高い収益に達したと報告する。一方 DeDQN は 2 つの方策のずれによる分布のずれで不安定になり、KL の制約で緩和した。[Schäfer DeRL (2021/07), §5.5〜5.6、§6〜7、Fig.4・5・7・8]
- DEEP は、ボーナスを足す方策はボーナスが 0 に近づくまで偏るが、task の報酬だけで学ぶ task 方策は偏らないとし、報酬が密な環境で性能を落とさず、疎な環境でデータ効率が数倍になったと報告する（SAC と組んだ実験）。[Whitney DEEP (2021/01), Abstract、§4.2〜4.3]
- NGU は β=0 の活用方策を同じ UVFA で同時に学ぶ。Montezuma's Revenge では探索的な方策のほうが点が高く、活用方策が探索側の知識を取り込むには非常に長い信用割当が要ると書く。[Badia NGU (2020/02), §3、§4.2]
- SOFE は、DeRL は活用方策への非定常性の影響を和らげるが探索方策の最適化は難しいまま残ると評し、DeepSea の難しい設定で SOFE-DQN が DeRL-DQN を上回った。[Creus SOFE (2023/10), §2、§5.2]

### 4.4 正規化の尺度

- RND は内的報酬を内的収益の標準偏差の走る推定で割り、BYOL-Explore は生の報酬を標準偏差の EMA で割る。どちらも非定常性への対処と位置づける。[Burda RND (2018/10), §2.4；Guo BYOL-Explore (2022/06), §3.3、付録 A.3]
- NGU の一生の新しさの係数は誤差を走る平均と標準偏差で正規化して作り、この係数を含む内的報酬を actor の時点で計算して保存する。[Badia NGU (2020/02), §2、付録 A]
- RLeXplore は、ほぼすべての内的報酬が何らかの正規化なしでは探索できなかったと報告する。[Yuan RLeXplore (2024/05), §5.1]
- 保存した時点と今とで正規化の尺度が違うことを直接論じた原典は見つからなかった。

### 4.5 多段の収益とエピソードの区切り

- RND は、内的の収益は game over で打ち切らないのが探索に自然で、外的を非エピソード的にすると序盤の報酬を取ってわざと終わる戦略に悪用されうるので、外的はエピソード的、内的は非エピソード的にして 2 Head で合成した。ただし学習は PPO の GAE である。[Burda RND (2018/10), §2.3、§3.1〜3.2]
- BYOL-Explore は内的をエピソード的に扱い、非エピソード的のほうが危険を避けすぎず探索しやすいと述べる。[Guo BYOL-Explore (2022/06), §4.2]
- NGU・Agent57 は Replay の系列がエピソードの境をまたがず、多段の学習は Retrace で行う。NGU はメモリを消さずに続けるエピソード数（1 か 3）をハイパーパラメータの範囲に入れている。[Badia NGU (2020/02), 付録 A・F.5 表9；Badia Agent57 (2020/03), 付録 E]
- Ostrovski らは、一過性のボーナスを速く伝えるために Monte Carlo 収益を混ぜた多段の更新を使い、Replay で終わっていないエピソードの遷移は Monte Carlo 収益を 0 として扱った。[Ostrovski PixelCNN (2017/03), §6、脚注3]
- Taïga らは Rainbow の多段の収益（n=3）を混ぜた 1 本の報酬で使い、内的報酬だけを別に扱う記述はない。[Taïga Bonus-ALE-workshop (2019/08), §3、付録 B.1]
- DQN 系で多段の収益に内的報酬を含めるとき、done で切るかどうかを比べた実験は見つからなかった。

## 5. 対処の形の一覧

| 論点 | 原典の対処 | 原典が書く効果・限界 | 出典 |
|---|---|---|---|
| メモリの持ち方 | エピソードの始めに空にする ring buffer と k 近傍（NGU 系）。並列環境ごとのリスト（RLeXplore） | 計算は O(メモリ × フレーム)。並列環境の off-policy は未対応（RLeXplore） | [Badia NGU (2020/02), §2・付録 A.2；Yuan RLeXplore (2024/05), 付録 C.1] |
| 行動時に計算して保存 | NGU・Agent57・MEME | 古さには触れない | [Badia NGU (2020/02), 付録 A] |
| sample 時に計算し直す | DI-engine の NGU、DEEP（探索用 Q だけ）、APT、RE3 | 速く覆える（DEEP）。非定常性は残る（APT） | [DI-engine NGU (2025/12)；Whitney DEEP (2021/01), §4.3；Liu APT (2021/03), §3.1] |
| 内的を Replay に入れない | DeRL（探索は on-policy） | 尺度・減衰率に頑健。方策のずれで不安定 | [Schäfer DeRL (2021/07), §4〜§7] |
| 統計量を状態に足して定常化 | SOFE | DeepSea で DeRL より良い | [Creus SOFE (2023/10), §4、§5.2] |
| 外的・内的の価値を分ける | Agent57（2 Network）、MEME（Head 分け）、RND・RLeXplore（PPO の 2 Head） | 重みへの頑健性、分割を外すと 20% 超低下（Agent57）。Head 1 つでも同程度（MEME） | [Badia Agent57 (2020/03), §4.2；Kapturowski MEME (2022/09), 付録 H.4] |
| 一生とエピソード内の組み合わせ | 掛け算（Henaff 2023）、倍率（NGU）、初訪問の指示関数（NovelD） | 掛け算は和より良い（MiniHack） | [Henaff Global-Episodic (2023/06), §4.2] |
| 埋め込みの古さ | 固定の乱数エンコーダ（RE3、Henaff 2023 の Montezuma） | 距離が変わらず安定 | [Seo RE3 (2021/02), §3；Henaff Global-Episodic (2023/06), 付録 E.5] |

## 6. 批判と懸念

- **古さを論じない代表例**: 価値ベースでエピソード内のボーナスを使う代表の NGU・Agent57・MEME は、保存した内的報酬と埋め込みの古さに触れていない（§2.2・§2.3）。古さを名指しした DEEP は査読未確認で、主な実験は連続制御である。[Whitney DEEP (2021/01)]
- **エピソード全体の文脈が要る**: エピソード内のボーナスは Replay のランダムな遷移から計算し直せず、統一実装でも off-policy の並列環境には対応していない。[Yuan RLeXplore (2024/05), 付録 C.1・G]
- **方策の分離の副作用**: 探索と活用の方策を分けると分布のずれで不安定になりうる。共有した表現は活用側に干渉しうる。[Schäfer DeRL (2021/07), §5.6；Badia NGU (2020/02), §4.2]
- **エピソード内のボーナスが効かない課題**: 文脈が 1 つの Montezuma's Revenge では、エピソード内の E3B は振るわなかった。[Henaff Global-Episodic (2023/06), §3.5]
- **Atari での価値ベースの証拠の薄さ**: NGU 系以外に、エピソード内のボーナスを価値ベースで難探索の Atari に使った報告は見つからなかった（§3.5）。

## 7. 総合評価

本冊の出典の範囲で、論点ごとに原典が示す選択肢をまとめる。

1. **メモリ**: NGU 系はエピソードの始めに空にする ring buffer（30,000 件、k=10）を使い、費用はメモリ全体への距離計算である。E3B の行列は rank-1 更新で次元の 2 乗の費用になる。並列の actor・lane での持ち方を明記した原典はない（§2.1・§3.2）。
2. **報酬の保存**: 代表例（NGU 系）は行動時に計算して保存し、古さに触れない。sample 時に計算し直す例（DI-engine の NGU、DEEP）と、内的を Replay に入れない例（DeRL）、状態に統計量を足す例（SOFE）がある（§2.3・§4.1・§3.3）。
3. **価値の分け方**: Agent57 は 2 Network で重みへの頑健性を報告し、MEME は Head を分けても 1 つでも同程度とした。RND と RLeXplore は on-policy で 2 Head が良いと報告した（§4.2）。
4. **組み合わせ方**: 一生とエピソード内のボーナスは、MiniHack では掛け算が和より良く、NGU の k 近傍のエピソード内ボーナスは E3B より劣った（§3.4）。
5. **多段の収益と区切り**: NGU 系は系列をエピソードの境で切り、RND は on-policy で内的を非エピソード的に扱った。DQN 系で比べた実験は見つからなかった（§4.5）。

## 8. 調査の限界

- **照合**: 本文は要約ツールを通して読み、数値は PDF で 1 語ずつ照合していない。
- **原典の中の食い違い**: NGU の kernel の ε（§2 は 10⁻³、表 6 は 10⁻⁴）、NGU の優先度の指数（本文 1.0、表 0.9）、Agent57 のバンディットの窓（表 3 は 90、本文は別の値）、NGU・Agent57 の表の割引率の書き方（内的 0.99・外的 0.997）と本文の組ごとの γ。
- **読めなかったもの**: R2D2 の原典（OpenReview の確認画面）、NGU・MEME の末尾の一部、SOFE と DeepCS の本文（要約・アブストラクトだけ）、RLeXplore の Table 14 の基盤アルゴリズム。
- **確かめられなかった事実**: エピソード内メモリを actor ごとに持つかの明記、NGU 表の「埋め込みの target を 1 エピソードに 1 回更新」の意味、MEME が組ごとに報酬を組み直す実装、RIDE の式の平方根の有無、NovelD の Q 学習の玩具実験でエピソード内の制限を使ったか、SOFE-DQN の DeepSea のボーナスの種類、E3B・NovelD・RIDE を DQN・Rainbow・SAC に載せた査読論文。
- **実装の版**: rllte は main の b5d55c82（2025-07-22）、DI-engine は d0b21d06（2025-12-07）を読んだ。RLeXplore の論文が指すリポジトリと rllte の対応は未確認。
- **引用**: 著作権への配慮から原文の逐語は引かず、要約と位置で示した。

## 9. 出典一覧

被引用数は Semantic Scholar の値（取得日 2026-11-18）。

- [Badia NGU, 2020/02] Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi ほか. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020（査読あり）. https://arxiv.org/abs/2002.06038 — 所属 DeepMind。被引用 380。
- [Badia Agent57, 2020/03] Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020（査読あり）. https://arxiv.org/abs/2003.13350 — 所属 DeepMind。被引用 613。
- [Kapturowski MEME, 2022/09] Steven Kapturowski, Víctor Campos, Ray Jiang, Nemanja Rakićević, Hado van Hasselt, Charles Blundell, Adrià Puigdomènech Badia. "Human-level Atari 200x faster." ICLR 2023（査読あり）. https://arxiv.org/abs/2209.07550 — 所属 DeepMind。被引用 45。
- [Yuan RLeXplore, 2024/05] Mingqi Yuan, Roger Creus Castanyer, Bo Li, Xin Jin, Wenjun Zeng, Glen Berseth. "RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning." TMLR（査読あり）. https://arxiv.org/abs/2405.19548 — 所属 The Hong Kong Polytechnic University、Mila・Université de Montréal、Eastern Institute of Technology, Ningbo。被引用 19。
- [rllte 実装, 2025/07] RLE-Foundation. rllte（rllte/xplore/reward/pseudo_counts.py・ngu.py、main b5d55c82）. https://github.com/RLE-Foundation/rllte — 公開実装。
- [DI-engine NGU, 2025/12] OpenDILab. DI-engine（ding/entry/serial_entry_ngu.py、ding/reward_model/ngu_reward_model.py、ding/policy/ngu.py、d0b21d06）. https://github.com/opendilab/DI-engine — 公開実装（査読なし）。
- [Henaff E3B, 2022/10] Mikael Henaff, Roberta Raileanu, Minqi Jiang, Tim Rocktäschel. "Exploration via Elliptical Episodic Bonuses." NeurIPS 2022（査読あり）. https://arxiv.org/abs/2210.05805 — 所属 Meta AI Research、University College London。被引用 76。
- [Henaff Global-Episodic, 2023/06] Mikael Henaff, Minqi Jiang, Roberta Raileanu. "A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs." ICML 2023（査読あり）. https://arxiv.org/abs/2306.03236 — 所属 Meta AI Research、UCL。被引用 24。
- [Zhang NovelD, 2021/12] Tianjun Zhang, Huazhe Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph E. Gonzalez, Yuandong Tian. "NovelD: A Simple yet Effective Exploration Criterion." NeurIPS 2021（査読あり）. https://papers.neurips.cc/paper_files/paper/2021/file/d428d070622e0f4363fceae11f4a3576-Paper.pdf — 所属 UC Berkeley、UC San Diego、清華大学、Facebook AI Research ほか。被引用 109。
- [Zhang BeBold, 2020/12] 同上の著者. "BeBold: Exploration Beyond the Boundary of Explored Regions." arXiv（査読なし）. https://arxiv.org/abs/2012.08621 — 被引用 47。
- [Raileanu RIDE, 2020/02] Roberta Raileanu, Tim Rocktäschel. "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments." ICLR 2020（査読あり）. https://arxiv.org/abs/2002.12292 — 所属 Facebook AI Research、NYU、UCL。被引用 225。
- [Seo RE3, 2021/02] Younggyo Seo, Lili Chen, Jinwoo Shin, Honglak Lee, Pieter Abbeel, Kimin Lee. "State Entropy Maximization with Random Encoders for Efficient Exploration." ICML 2021（査読あり）. https://arxiv.org/abs/2102.09430 — 所属 KAIST、UC Berkeley、University of Michigan、LG AI Research。被引用 172。
- [Savinov EC, 2018/10] Nikolay Savinov, Anton Raichuk, Raphaël Marinier, Damien Vincent, Marc Pollefeys, Timothy Lillicrap, Sylvain Gelly. "Episodic Curiosity through Reachability." ICLR 2019（査読あり）. https://arxiv.org/abs/1810.02274 — 所属 Google Brain、DeepMind、ETH Zürich。被引用 303。
- [Creus SOFE, 2023/10] Roger Creus Castanyer, Joshua Romoff, Glen Berseth. "Improving Intrinsic Exploration by Creating Stationary Objectives." ICLR 2024（査読あり）. https://arxiv.org/abs/2310.18144 — 所属 Mila・Université de Montréal、Ubisoft La Forge。被引用 8。
- [Whitney DEEP, 2021/01] William F. Whitney, Michael Bloesch, Jost Tobias Springenberg, Abbas Abdolmaleki, Kyunghyun Cho, Martin Riedmiller. "Decoupled Exploration and Exploitation Policies for Sample-Efficient Reinforcement Learning." arXiv（査読未確認）. https://arxiv.org/abs/2101.09458 — 所属 New York University、DeepMind。被引用 18。
- [Schäfer DeRL, 2021/07] Lukas Schäfer, Filippos Christianos, Josiah P. Hanna, Stefano V. Albrecht. "Decoupled Reinforcement Learning to Stabilise Intrinsically-Motivated Exploration." AAMAS 2022（査読あり）. https://arxiv.org/abs/2107.08966 — 所属 University of Edinburgh、University of Wisconsin–Madison。被引用 30。
- [Burda RND, 2018/10] Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. "Exploration by Random Network Distillation." ICLR 2019（査読あり）. https://arxiv.org/abs/1810.12894 — 所属 OpenAI、University of Edinburgh。被引用 1,838。
- [Guo BYOL-Explore, 2022/06] Zhaohan Daniel Guo ほか. "BYOL-Explore: Exploration by Bootstrapped Prediction." NeurIPS 2022（査読あり）. https://arxiv.org/abs/2206.08332 — 所属 DeepMind。被引用 109。
- [Liu APT, 2021/03] Hao Liu, Pieter Abbeel. "Behavior From the Void: Unsupervised Active Pre-Training." NeurIPS 2021（査読あり）. https://arxiv.org/abs/2103.04551 — 所属 UC Berkeley。被引用 267。
- [Ostrovski PixelCNN, 2017/03] Georg Ostrovski, Marc G. Bellemare, Aäron van den Oord, Rémi Munos. "Count-Based Exploration with Neural Density Models." ICML 2017（査読あり）. https://arxiv.org/abs/1703.01310 — 所属 DeepMind。被引用 708。
- [Taïga Bonus-ALE-workshop, 2019/08] Adrien Ali Taïga, William Fedus, Marlos C. Machado, Aaron Courville, Marc G. Bellemare. "Benchmarking Bonus-Based Exploration Methods on the Arcade Learning Environment." arXiv v3（査読未確認。ICLR 2020 版は別の arXiv 2109.11052）. https://arxiv.org/abs/1908.02388 — 所属 MILA・Université de Montréal、Google Brain。被引用 45。
