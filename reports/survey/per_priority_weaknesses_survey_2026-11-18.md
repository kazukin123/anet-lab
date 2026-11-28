# Survey: PER の優先度の弱点と代替

Date: 2026-11-18
Scope: Prioritized Experience Replay（PER）の TD 誤差による優先度の弱点（確率的な報酬での張り付き、優先度の古さ、損失の形との等価性）、α・β の選び方、分布型 RL（C51・QR-DQN・IQN）での優先度の定義と揺らぎ、TD 誤差以外の優先の仕方（遡る系・収益ベース・gain×need）と Atari での効果量、低い replay ratio での PER の効き、優先度を計器として出す例を整理する。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。著作権への配慮から原文の逐語は引かず、要約と位置（節・付録・表・図・式、実装はファイルと関数）で示す。本文の一部は arXiv・ar5iv・PMC の HTML 版から要約ツールを経て読み、数値は PDF で 1 語ずつ照合していない（§10）。被引用数は Semantic Scholar の値で、取得日は 2026-11-18。調査は観点ごとに 3 本のサブエージェントで行い、本冊で統合した。

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. TD 誤差の優先度の弱点](#2-td-誤差の優先度の弱点)
   - [2.1 確率的な報酬と張り付き](#21-確率的な報酬と張り付き)
   - [2.2 優先度の古さと初期優先度](#22-優先度の古さと初期優先度)
   - [2.3 損失の形との等価性](#23-損失の形との等価性)
3. [3. α と β の選び方](#3-α-と-β-の選び方)
4. [4. 分布型 RL での優先度](#4-分布型-rl-での優先度)
   - [4.1 各構成が優先度に使う量](#41-各構成が優先度に使う量)
   - [4.2 τ のサンプリングと揺らぎ](#42-τ-のサンプリングと揺らぎ)
   - [4.3 PER の寄与](#43-per-の寄与)
5. [5. TD 誤差以外の優先の仕方](#5-td-誤差以外の優先の仕方)
   - [5.1 遡る系](#51-遡る系)
   - [5.2 収益ベース](#52-収益ベース)
   - [5.3 gain と need](#53-gain-と-need)
   - [5.4 珍しい大報酬の遷移](#54-珍しい大報酬の遷移)
6. [6. 低い replay ratio と大きな Replay での PER](#6-低い-replay-ratio-と大きな-replay-での-per)
7. [7. 優先度を計器として出す例](#7-優先度を計器として出す例)
8. [8. 批判と懸念](#8-批判と懸念)
9. [9. 総合評価](#9-総合評価)
10. [10. 調査の限界](#10-調査の限界)
11. [11. 出典一覧](#11-出典一覧)

## 1. 対象と読み方

PER は TD 誤差の大きい遷移を多く引き、偏りを重要度重み（IS、指数 β）で補正する。優先度の強さは α で決める。本冊は、TD 誤差の大きさが「学ぶ値打ち」を表さない場面、優先度が古くなる仕組み、抽選が実は損失の形を変えていること、分布型 RL での定義の違い、そして TD 誤差に代わる優先の仕方を、原典の記述で並べる。

## 2. TD 誤差の優先度の弱点

### 2.1 確率的な報酬と張り付き

- 原論文自身が、貪欲な TD 誤差の優先は確率的な報酬のノイズに弱く、bootstrap がそれを強め、関数近似の誤差もノイズ源になり、最初に誤差の大きかった遷移ばかり引かれて多様性が失われると書く。TD 誤差は報酬にノイズがあると学習の進みの推定として質が悪くなるとも書く。[Schaul PER (2016/02), §3.2・§3.3]
- 原論文の付録は、|δ| は報酬・遷移の確率性や部分観測、関数近似の容量の限界を無視するので、学びようのない遷移があると問題になると述べ、|δ| の変化分を代案に挙げたが、予備実験では |δ| を上回らず、調べた環境がほぼ決定的だったためかもしれないとする。[Schaul PER (2016/02), 付録 A]
- ReLo は、固有のノイズで TD 誤差が何度引かれても高いままの遷移に PER が高い優先度を与え続けると指摘し、7×7 の格子で一様乱数の報酬の状態を置くと、PER が序盤その状態への遷移を引き続けて本来のゴールを学べなかったことを示した。報酬にノイズを足した DMC でも、SAC+PER の TD 誤差が最も高いままだった。[Sujit ReLo (2023/11), §3、§4.1〜4.3、Figure 2・4a]
- UPER は、PER は乱れの大きい遷移を学びにならなくても繰り返し引きやすいと述べ、平均が同じでノイズだけ違う腕のバンディットで、推定が真の平均に近くても TD 誤差にサンプルのノイズが残ることを示し、ノイズのある格子で PER がノイズのある状態を過剰に引いたと報告する。[Carrasco-Davis UPER (2025/06), §1、§4.1〜4.2、Figure 1]
- Rainbow は、新しい遷移を最大優先度で入れると最近の遷移に偏り、確率的な遷移も学ぶことが少ないのに優遇されうると書き、分布型の KL 損失を優先度にするとノイズのある環境に頑健かもしれないと仮説を述べる。[Hessel Rainbow (2017/10), Background、The Integrated Agent]

### 2.2 優先度の古さと初期優先度

- 原論文は、計算量を抑えるため TD 誤差を引いた遷移についてしか更新しないので、初回の誤差が小さい遷移は長く引かれず、スライド窓の Replay では実質的に一度も引かれないと書く。新しい遷移はそれまでの最大優先度で入れ、全経験が最低 1 回は引かれるようにする。しばらく引かれていない遷移の優先度を上げる staleness bonus を案として挙げるが、評価結果は見当たらない。[Schaul PER (2016/02), §3.2・§3.3、Algorithm 1、付録 A]
- Ape-X は、最大優先度で入れて引かれるまで更新しない方式は多数の actor では最新データに目が向きすぎるとして、actor が行動時に初期優先度を計算する。[Horgan Ape-X (2018/03), §3、付録 F]
- LaBER は、PER で更新されるのは mini-batch の優先度だけで、ほかの遷移の優先度は任意に古くなりうるので勾配推定の分散が制御されないと述べ、大きい batch を引いて新しい優先度で絞る方式（PER-LaBER）が PER より良いことが多く、新しい優先度が決め手だと結論する。[Lahire LaBER (2022/07), §3.1〜3.2、§4、Figure 1]
- Pan らは、全優先度を毎 step 更新する版に比べ引いた分だけ更新する版が大きく劣ることを、教師ありと Mountain Car の DQN で示した。[Pan PER-limits (2022/06), §3.3、Figure 1(a)(c)]
- Dopamine（JAX）と BTR の公式実装は、新しい遷移に記録された最大優先度（BTR は max_prio^α）を入れる。[Dopamine 実装 (2026/11), dqn_agent.py `_store_transition`；BTR 実装 (2026/11), PER.py]

### 2.3 損失の形との等価性

- LAP は、優先度で抽選した損失と一様に抽選した別の損失の期待勾配が一致する条件を示し、損失 |δ|^τ に PER を組むと一様抽選で |δ|^(τ+α−αβ) に比例する損失を使うのと等価だとした。MSE の最小化は目標の平均、L1 は中央値を与えるので、τ+α−αβ が 2 でなければ PER の目的には偏りがあり、MSE と組んで β<1 なら冪が 2 を超えて外れ値に寄る。[Fujimoto LAP (2020/10), §4 定理1、§5.1 定理3]
- LAP（抽選する版）は優先度を max(|δ|^α, 1) で下限を 1 に切り、Huber 損失と組んで IS を使わない。PAL（抽選しない版）は期待勾配が同じ損失を一様抽選で使う。[Fujimoto LAP (2020/10), §5.2 式9、式10]
- 実験では、MuJoCo（TD3）で LAP と PAL に差がなく、改善は期待勾配の変化から来ると解釈した。Atari（Double DQN、10 ゲーム）では LAP が 9 ゲームで PER を上回り平均 +53% で、PAL は 6 ゲームで PER に劣り、Atari では優先付けそのものの役割が大きいとした。[Fujimoto LAP (2020/10), §6、Table 1・2]
- Pan らは、誤差に比例した抽選と MSE の組が一様抽選と 3 乗損失の組と等価で、高い冪の目的は目標のノイズに敏感だと示した。[Pan PER-limits (2022/06), §3.1]
- LaBER は、理想の抽選はサンプルごとの勾配ノルムに比例し、TD 誤差がその良い近似になるのは損失が L2 で出力の勾配ノルムがほぼ一定のときに限ると述べる。batch 内の最大重みで割る PER 流の正規化は、他の正規化より一貫して劣った。[Lahire LaBER (2022/07), §3.1、§3.3、§4]

## 3. α と β の選び方

| 構成 | α | β | 根拠として原典・実装が書くこと | 出典 |
|---|---|---|---|---|
| PER 原論文 proportional | 0.6 | 0.4→1 を線形 | Atari の部分集合の粗いグリッド探索。攻めと頑健さの釣り合い | [Schaul PER (2016/02), §4、付録 B.2.2] |
| PER 原論文 rank-based | 0.7 | 0.5→1 | 同上 | 同上 |
| Rainbow | 0.5（優先度は KL 損失） | 0.4→1 を線形 | 推奨値。KL 優先度では α の選択に非常に頑健 | [Hessel Rainbow (2017/10), Table 1] |
| Dopamine（JAX）Rainbow | 明示なし（sqrt(loss) で 0.5 相当） | 0.5 固定 | 5 ゲームで固定のほうが良かった（Pong を除く）とコードのコメント | [Dopamine 実装 (2026/11), rainbow_agent.py] |
| Ape-X（Atari） | 0.6 | 0.4 | 根拠の記述なし。連続制御では 0.3〜0.7 でほぼ同じ、0.7 超で不安定 | [Horgan Ape-X (2018/03), §4.1、付録 E] |
| Rainbow-IQN | 0.2 | 記述を確認できず | Rainbow の値では IQN の損失分布が広く一部の遷移が引かれすぎた。5 ゲームで {0.1〜0.25} を比べ 0.2 | [Toromanoff Rainbow-IQN (2019/08), §4] |
| BTR（論文） | 0.2 | 表に記載なし | Toromanoff の推奨に従う。α を持たない Dopamine の Replay はやや劣った | [Clark BTR (2024/11), §3.2、付録] |
| BTR（公式実装） | 0.2 | IS の指数に α を使う。β の焼きなましは既定で無効 | もとは誤りだったが性能が良かったとコードのコメント | [BTR 実装 (2026/11), PER.py・Agent.py] |
| LAP | 0.4（MuJoCo）・0.6（Atari） | IS なし | 下限 1 と Huber で偏りを抑える。HalfCheetah・Asterix で少数比較 | [Fujimoto LAP (2020/10), 付録 D.1・D.2] |

- 原論文は β を学習の終盤に 1 へ上げる理由を、不偏であることが最も重要になるのは収束の近くだからとし、α と β は相互作用し、IS は実効的な step size を下げるとも書く。完全な IS は序盤の学習を遅くしたが、早すぎる収束の危険は小さかった。[Schaul PER (2016/02), §3.4、付録 B.2.1、Figure 12]

## 4. 分布型 RL での優先度

### 4.1 各構成が優先度に使う量

| 構成 | 優先度に使う量 | 初期優先度と更新 | 出典 |
|---|---|---|---|
| Rainbow（C51） | n ステップの分布型 KL 損失の ω 乗。平均の TD 誤差でも計算できると述べたうえで KL を使った | 最大優先度で入れる | [Hessel Rainbow (2017/10), The Integrated Agent] |
| Dopamine（JAX）Rainbow・QR-DQN | sqrt(損失 + 1e-10)。QR-DQN は分位点 Huber 損失を目標側で平均・分位点で和 | 記録された最大優先度で入れ、学習 step ごとに新しい損失で上書き | [Dopamine 実装 (2026/11), rainbow_agent.py・quantile_agent.py・dqn_agent.py] |
| Dopamine（JAX）IQN | PER を使わない（引数は受けるが未使用、親は一様サンプラー） | — | [Dopamine 実装 (2026/11), implicit_quantile_agent.py] |
| IQN・QR-DQN 原典 | PER を組み込んでいない | — | [Dabney IQN (2018/06), §2.3・§5] |
| Rainbow-IQN | 学習損失の α 乗（α=0.2） | 挿入前に計算（Ape-X 方式） | [Toromanoff Rainbow-IQN (2019/08), §4、付録 A.1] |
| BTR | τ の組ごとの \|TD\| を片方の τ 軸で和・もう片方で平均した値（分位点 Huber 損失そのものではない）に 1e-6 を足して α 乗 | max_prio^α で入れ、毎 step 上書き | [BTR 実装 (2026/11), Agent.py・PER.py] |
| UPER | QR のアンサンブルから推定した不確かさの比（情報利得） | 未確認 | [Carrasco-Davis UPER (2025/06), §5] |

### 4.2 τ のサンプリングと揺らぎ

- IQN で τ を毎回サンプルするために同じ遷移でも引くたびに優先度が揺らぐ点を直接論じた原典・公式の記述は、見つからなかった。τ を固定する、平均を取るといった対処の記述もない。
- 関連する事実として、BTR の実装は τ を forward のたびに引き直し、その学習 step の τ から作った値で優先度を毎回上書きする（既定 taus=8）。Toromanoff らは IQN 系で損失の分布が広がり引かれすぎる遷移が出ると報告するが、原因を τ のサンプリングに帰してはいない。[BTR 実装 (2026/11), networks.py `calc_cos`、Agent.py；Toromanoff Rainbow-IQN (2019/08), §4]

### 4.3 PER の寄与

- Rainbow の ablation では、PER と multi-step が最も重要な 2 要素で、どちらを外しても中央値の性能と序盤の性能が大きく下がり、57 ゲーム中 53 で完全な Rainbow がどちらの ablation より良かった。[Hessel Rainbow (2017/10), Ablation studies、図2〜4]
- Revisiting Rainbow は、小規模環境（古典制御・MinAtar）では構成要素の効き方が環境ごとに一様でないと報告し、MinAtar では QR-DQN・M-DQN・M-IQN に PER を足したときが最も効いた。[Obando-Ceron Revisiting (2020/11), §4.2、付録 F 表6・7]
- BTR の本文の ablation に「PER なし」は含まれず、付録で BattleZone だけの試しとして PER を残したと書く（数値なし）。[Clark BTR (2024/11), §5.1、付録 H]
- UPER は Atari-57 で、TD 誤差で優先した QR-DQN・QR のアンサンブルより、情報利得で優先した版の中央値が高かった（各 3 seed）。[Carrasco-Davis UPER (2025/06), §5、図2]

## 5. TD 誤差以外の優先の仕方

### 5.1 遡る系

- **PSER**: 新しい遷移の優先度は PER と同じで、同じエピソードの手前の遷移へ優先度を減衰係数 ρ の累乗で流す（既定は大きいほうを残す MAX 版、窓幅 W）。流した遷移が一度引かれると優先度がほぼ ε に戻る「priority collapse」を指摘し、旧優先度の η 倍を下限に残す。PER はゴールの遷移 1 本しか優先しないので手前の遷移が偶然選ばれるのを待つ必要がある、と PER の弱点を挙げる。[Brittain PSER (2019/05), §3.1〜3.2 式3〜10、図2]
- PSER の Atari（Dopamine 上の Nature DQN 構成、200M フレーム）では、人間正規化の中央値 109%（PER 88%）、平均 832%（PER 607%）で、60 ゲーム中 40 で PER を上回った。図からの読み取りでは Venture・Solaris で大きく負、Montezuma's Revenge・Pitfall・Private Eye はほぼ 0% だった。[Brittain PSER (2019/05), 表1、図5、§5.2]
- **EBU**: 優先度を付けず、Replay からエピソード 1 本を一様に選び、全遷移の target を後ろから再帰的に作る（拡散係数 β で過大評価を抑える）。49 ゲーム・10M フレームで平均 253.55%・中央値 51.55%（PER は 156.57%・40.86%）、適応版の 20M フレームで 347.99%・92.50% だった。Montezuma's Revenge は DQN 比でほぼ 0 だった（図からの読み取り）。[Lee EBU (2019/12), §3.2〜3.3、Algorithm 2、表2、図4]
- **TER**: 状態をハッシュした頂点と遷移の辺のグラフを作り、終端から逆向きに幅優先で更新する。Minigrid・Sokoban で報酬が疎なほど差が大きく、Atari では終端のゴールが無いため基準と同程度だった。[Hong TER (2022/03), §4〜§5、付録 M]

### 5.2 収益ベース

- **SIL**: エピソード終了後に計算した割引収益が価値推定を上回る組だけに損失をかけ、上回った分を Replay の優先度にする。難探索 7 ゲーム中 6 で最良（Montezuma's Revenge 2,500）、49 ゲームの中央値 138.7%（A2C 96.1%）だったが、Venture では一度も正の報酬を得なかった。序盤に活用しすぎて準最適に留まるゲームがあり、SIL の更新を後半に下げると改善した。SIL と同じ優先度の PER を ACER に付けた版は A2C より悪く、著者は IS の項が原因と推測する。[Oh SIL (2018/07), §3、§5.3〜5.5、表1・2]
- **EMDQN**: エピソード記憶を学習の監督に使い、DQN の 1/5 の相互作用で高い性能に達したと要旨で主張する（本文は未読）。[Lin EMDQN (2018/05), 要旨]

### 5.3 gain と need

- Mattar & Daw は、backup の期待価値（EVB）を、方策が変わることによる期待収益の増加（gain）と、その状態をこの先訪れる見込み（need、successor representation から計算）の積に分解した。Prioritized Sweeping との違いとして、gain は方策を変える更新だけを数え、need は訪れそうな状態へ backup を偏らせると述べる。表形式・モデルベースで、gain を実行前に知る前提は非現実的だと自ら認める。[Mattar & Daw (2018/10), Methods §4.2〜4.5 式4〜6・13、Discussion]
- Li らは EVB を方策改善と評価改善の価値に分け、Q 学習では EVB の絶対値が「学習率×|TD 誤差|」以下になることを示した。上界を優先度にした版は soft DQN の 9 ゲームで一様比 +18.0%、PER 比 +8.47% だった（学習量の記載が本文と付録で食い違う）。need は扱っていない。[Li VER (2021/02), §2.2、定理3.1・4.1、§6]
- Yuan & Mattar は need を successor representation で学び、深層版では抽選を TD 誤差のままにして更新量に need を掛けた。全バッファの need を再計算する版は大きなバッファで非常に遅く、深層版は序盤の学習が遅くなった。[Yuan & Mattar (2021/11), §3.2〜§5]

### 5.4 珍しい大報酬の遷移

- 深層 RL の Atari で、1 本の希少な高報酬遷移の優先度を学習中に時系列で測った例は見つからなかった。
- PSER は、Blind Cliffwalk ではゴールの遷移が初めて引かれるまでどの方式も一様と同じで、その後の付け方で差が開き、流した優先度は一度引かれると ε 近くに落ちると書く。[Brittain PSER (2019/05), §3.1〜3.2、図2]
- Mattar & Daw の迷路のシミュレーションでは、学習が進むと gain と再生の回数が減り、予想より大きい報酬では逆向きの再生が増えた。[Mattar & Daw (2018/10), §2.5、図5・6]
- Li らは、報酬がゴールにだけある直線の格子で、必要な replay の回数が EVB の順で約 N、|TD 誤差| の順で約 4N、一様で約 4N² になることを示した。[Li VER (2021/02), §2.4、図1]

## 6. 低い replay ratio と大きな Replay での PER

- DQN の基準設定（Replay 10^6、4 遷移ごとに 32 件）では各遷移が平均 8 回引かれる。原論文は、一部の遷移が一度も引かれずに窓から落ち、一様抽選は古い方策の遷移に偏ると述べ、最大優先度での加点が両方に効くとする。[Schaul PER (2016/02), §1、§4、§5]
- Replay の容量を 1M→10M に増やす実験では、PER の有無は大きな容量での利得にほとんど影響せず、PER を外した Rainbow でも +17.3% の利得が残った。著者は直感に反する結果としている。[Fedus Revisiting-ER (2020/07), §4.1、Figure 6、付録 B]
- Pan らは、Replay が状態空間のごく一部しか覆わないと優先抽選の分布の推定が不正確になると述べる。[Pan PER-limits (2022/06), §3.2〜3.3]
- BTR の replay ratio は環境 step あたり 1/64（batch 256）、Replay は 2^20 で、実時間を抑えるため低く保ったと書くが、PER の効き方とは結び付けていない。[Clark BTR (2024/11), §3.1〜3.2]
- 1 遷移あたりの抽選回数を変えて PER の効き方を直接測った研究は、今回読んだ範囲には見つからなかった。

## 7. 優先度を計器として出す例

- 原論文は、Replay 内の全遷移の最後に見た |δ| を並べた曲線を学習時刻ごとに描き、実効的な抽選確率を |δ| の関数として一様の基準線と並べた。[Schaul PER (2016/02), §5、付録 Figure 10・11]
- LaBER は、代理の抽選分布・一様分布と最適分布の全変動距離を毎 step 計算してヒストグラムにした。[Lahire LaBER (2022/07), §4、Figure 2]
- Pan らは、Replay から引いた状態の分布をヒートマップにし、理想の抽選分布までの距離の推移を描いた。[Pan PER-limits (2022/06), §4、Figure 2・3]
- ReLo は、学びようのない状態を含む遷移が引かれた回数を数え、TD 誤差を学習中ずっと記録した。UPER は格子の状態ごとの抽選回数をヒートマップにした。[Sujit ReLo (2023/11), §4.1・§4.3、Figure 2・4；Carrasco-Davis UPER (2025/06), §4.2、Figure 1e]
- Toromanoff らは、損失（＝優先度）の分布の広がりを見て α を調整する必要に気付いた。[Toromanoff Rainbow-IQN (2019/08), §4]

## 8. 批判と懸念

- **ノイズへの張り付き**: TD 誤差の優先は確率的な報酬・遷移に張り付き、学べない遷移を引き続ける（§2.1）。
- **古い優先度**: 引いた遷移しか更新しないので優先度が古くなり、勾配推定の分散が制御されない（§2.2）。
- **目的の偏り**: MSE と組んだ PER は 2 乗より高い冪の損失を最小化することになり、外れ値に寄る（§2.3）。
- **分布型での未整理**: 分布型の優先度は構成ごとに定義がばらばらで、IQN の τ による揺らぎを論じた原典は見つからなかった。Dopamine の IQN は PER を使っていない（§4）。
- **遡る系の限界**: PSER は難探索ゲームでほぼ効かず、Venture・Solaris で PER より大きく劣った。PSER は査読未確認である（§5.1）。
- **収益ベースの過活用**: SIL は序盤の活用しすぎで準最適に留まる例がある（§5.2）。
- **gain×need の前提**: gain を実行前に知る前提は非現実的で、need の再計算は大きなバッファで重い（§5.3）。

## 9. 総合評価

キューの論点メモ（以前のキューの 13）に、本冊の出典の範囲で答える。

1. **TD 誤差が貴重さを表さない場面と張り付き**: 確率的な報酬・遷移、部分観測、関数近似の限界で TD 誤差が下がらない遷移に、PER は優先度を与え続ける（原論文の付録、ReLo、UPER）。
2. **優先度の古さ**: 引いた遷移しか更新しないことが原因で、新しい優先度で絞る LaBER や全更新の版が良い結果を出した。
3. **損失の形との等価性**: 優先抽選は損失の冪を変えることと等価で、MSE との組には偏りがある。LAP は下限 1・Huber・IS なしで Atari の 10 ゲーム中 9 で PER を上回った。
4. **α・β**: 原論文 0.6/0.4→1、Rainbow 0.5/0.4→1、Dopamine 0.5 固定、IQN 系は損失の分布が広いので 0.2（Toromanoff、BTR）。BTR の実装は IS の指数に α を使う。
5. **分布型の優先度と τ の揺らぎ**: Rainbow は KL、Dopamine の QR-DQN は分位点 Huber 損失の平方根、BTR は τ の組の |TD| の和と平均で、Dopamine の IQN は PER を使わない。τ による揺らぎを論じた原典は見つからなかった。
6. **遡り・収益・gain×need の効果量**: PSER は 200M フレームで中央値 109% 対 88%（ただし難探索ではほぼ効かない）、EBU は 10M フレームで中央値 51.55% 対 PER 40.86%、SIL は Montezuma's Revenge 2,500 で難探索 7 中 6 で最良、gain×need の深層版は効果が限定的だった。
7. **低い replay ratio での PER と計器**: 容量を増やしても PER の有無は利得に影響しなかった（Fedus）。抽選回数を変えて PER の効き方を測った研究は見つからなかった。計器としては、|δ| の分布、抽選分布の距離、状態ごとの抽選回数のヒートマップ、学べない遷移の抽選回数の例がある。

## 10. 調査の限界

- **照合**: Mattar & Daw（PMC）、Li・Yuan & Mattar・TER（ar5iv）は要約ツールを通して読み、数値と位置を PDF と照合していない。Mattar & Daw は本文の末尾を読んでいない。PSER と EBU の難探索ゲームの値は図からの読み取りである。
- **原典の中の食い違い**: Li らの Atari の学習量（本文 40M フレーム、付録 50,000）。PER 原論文の探索に使ったゲーム数（§4 は 8、付録の列挙は 7）。
- **確かめられなかった事実**: IQN の τ による優先度の揺らぎを論じた記述、希少な大報酬の遷移の優先度の時系列、抽選回数を変えた PER の効き方の測定、Ape-X の初期優先度の効果の比較、原論文の staleness bonus の評価、Toromanoff らの β と査読の有無、Dopamine の各エージェントの priority_exponent の上書き、Dopamine の TF 版の IQN（404）、UPER の QR-PER の TD 誤差の作り方、Revisiting Rainbow のリポジトリでの優先度の定義、EMDQN の本文、Rainbow の ablation の中央値の数値。
- **書誌**: TER と EMDQN の被引用数は取れていない（Semantic Scholar の回数制限）。Yuan & Mattar の所属と掲載先は未確認。
- **実装の版**: Dopamine は commit 1cba1f9、BTR は 093ca77 を読んだ（取得日 2026-11-18）。
- **引用**: 著作権への配慮から原文の逐語は引かず、要約と位置で示した。

## 11. 出典一覧

被引用数は Semantic Scholar の値（取得日 2026-11-18）。

- [Schaul PER, 2016/02] Tom Schaul, John Quan, Ioannis Antonoglou, David Silver. "Prioritized Experience Replay." ICLR 2016（査読あり）. https://arxiv.org/abs/1511.05952 — 所属 Google DeepMind。被引用 4,817。
- [Fujimoto LAP, 2020/10] Scott Fujimoto, David Meger, Doina Precup. "An Equivalence between Loss Functions and Non-Uniform Sampling in Experience Replay." NeurIPS 2020（査読あり）. https://arxiv.org/abs/2007.06049 — 所属 Mila、McGill University。被引用 89。
- [Lahire LaBER, 2022/07] Thibault Lahire, Matthieu Geist, Emmanuel Rachelson. "Large Batch Experience Replay." ICML 2022（査読あり）. https://proceedings.mlr.press/v162/lahire22a.html — 所属 ISAE-SUPAERO・Université de Toulouse、Google Research Brain Team。被引用 24。
- [Pan PER-limits, 2022/06] Yangchen Pan, Jincheng Mei, Amir-massoud Farahmand, Martha White, Hengshuai Yao, Mohsen Rohani, Jun Luo. "Understanding and Mitigating the Limitations of Prioritized Experience Replay." UAI 2022（査読あり）. https://arxiv.org/abs/2007.09569 — 所属 University of Alberta、University of Toronto・Vector Institute、Huawei Noah's Ark Lab ほか。被引用 29。
- [Sujit ReLo, 2023/11] Shivakanth Sujit, Somjit Nath, Pedro H.M. Braga, Samira Ebrahimi Kahou. "Prioritizing Samples in Reinforcement Learning with Reducible Loss." NeurIPS 2023（査読あり）. https://arxiv.org/abs/2208.10483 — 所属 Mila、ÉTS Montréal ほか。被引用 39。
- [Carrasco-Davis UPER, 2025/06] Rodrigo Carrasco-Davis, Sebastian Lee, Claudia Clopath, Will Dabney. "Uncertainty Prioritized Experience Replay." Reinforcement Learning Journal 2025（査読あり）. https://arxiv.org/abs/2506.09270 — 所属 Gatsby Unit UCL、Imperial College London、Google DeepMind ほか。被引用 7。
- [Hessel Rainbow, 2017/10] Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver. "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI 2018（査読あり）. https://arxiv.org/abs/1710.02298 — 所属 DeepMind。被引用 2,766。
- [Horgan Ape-X, 2018/03] Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, David Silver. "Distributed Prioritized Experience Replay." ICLR 2018（査読あり）. https://arxiv.org/abs/1803.00933 — 所属 DeepMind。被引用 866。
- [Dabney IQN, 2018/06] Will Dabney, Georg Ostrovski, David Silver, Rémi Munos. "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML 2018（査読あり）. https://arxiv.org/abs/1806.06923 — 所属 未確認。被引用 779。
- [Toromanoff Rainbow-IQN, 2019/08] Marin Toromanoff, Emilie Wirbel, Fabien Moutarde. "Is Deep Reinforcement Learning Really Superhuman on Atari? Leveling the playing field." arXiv（査読未確認。Semantic Scholar は NeurIPS と表示するが本会議か Workshop かは未確認）. https://arxiv.org/abs/1908.04683 — 所属 MINES ParisTech、Valeo。被引用 22。
- [Clark BTR, 2024/11] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." ICML 2025（査読あり。arXiv v2 の表記）. https://arxiv.org/abs/2411.03820 — 所属 University of Southampton。被引用 12。
- [BTR 実装, 2026/11] VIPTankz/BTR（commit 093ca77。PER.py・Agent.py・networks.py）. https://github.com/VIPTankz/BTR — 公式実装。
- [Dopamine 実装, 2026/11] google/dopamine（commit 1cba1f9。dopamine/jax/agents/rainbow/rainbow_agent.py、quantile/quantile_agent.py、implicit_quantile/implicit_quantile_agent.py、dqn/dqn_agent.py、replay_memory/samplers.py）. https://github.com/google/dopamine — 公式実装。
- [Obando-Ceron Revisiting, 2020/11] Johan S. Obando-Ceron, Pablo Samuel Castro. "Revisiting Rainbow: Promoting more Insightful and Inclusive Deep Reinforcement Learning Research." ICML 2021（査読あり）. https://arxiv.org/abs/2011.14826 — 所属 Google Research Brain Team。被引用 138。
- [Fedus Revisiting-ER, 2020/07] William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, Will Dabney. "Revisiting Fundamentals of Experience Replay." ICML 2020（Semantic Scholar の表示）. https://arxiv.org/abs/2007.06700 — 所属 Google Brain、MILA、DeepMind ほか。被引用 353。
- [Brittain PSER, 2019/05] Marc Brittain, Josh Bertram, Xuxi Yang, Peng Wei. "Prioritized Sequence Experience Replay." arXiv（査読未確認）. https://arxiv.org/abs/1905.12726 — 所属 Iowa State University。被引用 65。
- [Lee EBU, 2019/12] Su Young Lee, Sungik Choi, Sae-Young Chung. "Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update." NeurIPS 2019（査読あり）. https://arxiv.org/abs/1805.12375 — 所属 KAIST。被引用 87。
- [Hong TER, 2022/03] Zhang-Wei Hong, Tao Chen, Yen-Chen Lin, Joni Pajarinen, Pulkit Agrawal. "Topological Experience Replay." arXiv（掲載先未確認）. https://arxiv.org/abs/2203.15845 — 所属 MIT、Aalto University。被引用 未確認。
- [Oh SIL, 2018/07] Junhyuk Oh, Yijie Guo, Satinder Singh, Honglak Lee. "Self-Imitation Learning." ICML 2018（査読あり）. https://arxiv.org/abs/1806.05635 — 所属 University of Michigan、Google Brain。被引用 341。
- [Lin EMDQN, 2018/05] Zichuan Lin, Tianqi Zhao, Guangwen Yang, Lintao Zhang. "Episodic Memory Deep Q-Networks." IJCAI 2018（arXiv の comments 欄）. https://arxiv.org/abs/1805.07603 — 所属・被引用 未確認。要旨だけ読んだ。
- [Mattar & Daw, 2018/10] Marcelo G. Mattar, Nathaniel D. Daw. "Prioritized memory access explains planning and hippocampal replay." Nature Neuroscience 21(11):1609–1617（査読あり）. https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620 — 所属 Princeton University。被引用 385。
- [Li VER, 2021/02] Ang A. Li, Zongqing Lu, Chenglin Miao. "Revisiting Prioritized Experience Replay: A Value Perspective." arXiv（査読未確認）. https://arxiv.org/abs/2102.03261 — 所属 Peking University。被引用 14。
- [Yuan & Mattar, 2021/11] Yizhi Yuan, Marcelo G. Mattar. "Improving Experience Replay with Successor Representation." arXiv（査読未確認）. https://arxiv.org/abs/2111.14331 — 所属 未確認。被引用 1。
