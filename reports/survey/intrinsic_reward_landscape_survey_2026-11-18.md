# Survey: 内的報酬（探索のボーナス）の手法の全体像と今の主役

Date: 2026-11-18
Scope: 深層強化学習の探索のボーナス（内的報酬）を、予測誤差系・回数と密度の系・エピソード内の新しさと方策の組・戻る系の 4 系統と、比較研究・2024 年以降の動向に分けて整理する。各手法が採点に使う入力と計算費用、RND と ICM の使い分け、ベンチマークごとの基準、Go-Explore が挙げた detachment・derailment への各手法の対処を原典で確かめる。LLM を使う内的報酬そのものは既存の報告へのリンクに留める。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。原文の逐語は著作権への配慮から 1 か所に絞り、ほかは要約と位置（節・表・図・式）で示す。本文は arXiv・ar5iv の HTML 版や PDF から取得した要約を経ており、表の数値は PDF の表で照合していない（§11）。被引用数は Semantic Scholar の値で、取得日は 2026-11-18。調査は観点ごとに 5 本のサブエージェントで行い、本冊で統合した。

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. 予測誤差系](#2-予測誤差系)
   - [2.1 採点の入力と学習するもの](#21-採点の入力と学習するもの)
   - [2.2 RND と ICM の優劣が分かれる環境の性質](#22-rnd-と-icm-の優劣が分かれる環境の性質)
   - [2.3 難探索の Atari での比較](#23-難探索の-atari-での比較)
   - [2.4 BYOL-Explore と SND が改めた点](#24-byol-explore-と-snd-が改めた点)
3. [3. 回数と密度の系](#3-回数と密度の系)
   - [3.1 擬似カウントの定義とボーナスの形](#31-擬似カウントの定義とボーナスの形)
   - [3.2 Montezuma's Revenge での結果](#32-montezumas-revenge-での結果)
   - [3.3 後の論文が引き継いだと書くもの](#33-後の論文が引き継いだと書くもの)
   - [3.4 原典が書く苦手な場面](#34-原典が書く苦手な場面)
4. [4. エピソード内の新しさと方策の組](#4-エピソード内の新しさと方策の組)
   - [4.1 NGU が RND と逆モデルを選んだ理由](#41-ngu-が-rnd-と逆モデルを選んだ理由)
   - [4.2 Agent57 と MEME が NGU から改めた点](#42-agent57-と-meme-が-ngu-から改めた点)
   - [4.3 エピソード内のボーナスの手法](#43-エピソード内のボーナスの手法)
   - [4.4 一生とエピソード内のボーナスの使い分け](#44-一生とエピソード内のボーナスの使い分け)
5. [5. 戻る系と detachment と derailment](#5-戻る系と-detachment-と-derailment)
   - [5.1 detachment と derailment の定義](#51-detachment-と-derailment-の定義)
   - [5.2 Go-Explore の版と戻る操作の前提](#52-go-explore-の版と戻る操作の前提)
   - [5.3 ほかの手法の対処と残る問題](#53-ほかの手法の対処と残る問題)
   - [5.4 LLM が答えになる場面と前提](#54-llm-が答えになる場面と前提)
   - [5.5 Go-Explore の何が後に残ったか](#55-go-explore-の何が後に残ったか)
6. [6. 比較研究と 2024 年以降の動向](#6-比較研究と-2024-年以降の動向)
   - [6.1 Taïga らの Rainbow 上の比較](#61-taïga-らの-rainbow-上の比較)
   - [6.2 RLeXplore の比較と実装の細部](#62-rlexplore-の比較と実装の細部)
   - [6.3 DQN 系に内的報酬を足した例](#63-dqn-系に内的報酬を足した例)
   - [6.4 2024 年以降の動向](#64-2024-年以降の動向)
7. [7. 手法ごとの採点の入力と計算費用](#7-手法ごとの採点の入力と計算費用)
8. [8. ベンチマークごとの基準と主役](#8-ベンチマークごとの基準と主役)
9. [9. 批判と懸念](#9-批判と懸念)
10. [10. 総合評価](#10-総合評価)
11. [11. 調査の限界](#11-調査の限界)
12. [12. 出典一覧](#12-出典一覧)

## 1. 対象と読み方

扱う系統は次の 4 つである。

| 系統 | 何を新しさとみなすか | 本冊で扱う手法 |
|---|---|---|
| 予測誤差系 | モデルの予測が外れる度合い | ICM、Large-Scale Study の forward dynamics、RND、Disagreement、BYOL-Explore、SND、AMA |
| 回数と密度の系 | 状態をどれだけ見たか（擬似カウント） | CTS、PixelCNN、#Exploration、φ-EB |
| エピソード内の新しさと方策の組 | 同じエピソードの中で見たかどうか、探索の強さを変えた方策 | NGU、Agent57、MEME、NovelD、E3B、RE3、RIDE、Henaff et al. 2023 |
| 戻る系 | 有望な状態を覚えて戻り、そこから探索する | Go-Explore（2019 版・Nature 版・policy-based）、IGE、DTSIL、MEGA、LEXA ほか |

「一生の新しさ」は学習の全期間で減っていく新しさ、「エピソード内の新しさ」はエピソードごとに数え直す新しさを指す（原典の語は lifelong／global と episodic）。

## 2. 予測誤差系

### 2.1 採点の入力と学習するもの

ICM は s_t・a_t・s_{t+1} の 3 つで採点する。特徴 φ を逆モデル（2 観測から行動を当てる）で学び、順モデルが予測した φ(s_{t+1}) と実際の φ(s_{t+1}) の二乗誤差を内的報酬にする。encoder・逆モデル・順モデルを A3C の方策と同時に学ぶ。計算費用の数値は書かれていない。[Pathak ICM (2017/05), §2.2 式(6)(7)、§3]

Large-Scale Study は、順モデルの予測誤差を、画素・固定のランダム特徴（RF）・VAE・逆モデルの特徴（IDF）の 4 種の特徴空間で比べた。入力は x_t・a_t・x_{t+1} である。[Burda Large-Scale (2018/08), §2、§2.1、Table 1]

RND は次の観測 s_{t+1} だけで採点し、行動を使わない。固定したランダムの target と、学習する predictor の出力の二乗誤差が内的報酬である。観測は次元ごとの走る平均と標準偏差で白色化して [−5, 5] で切り（target と predictor の入力だけ）、内的報酬は内的 return の標準偏差の推定で割る。費用は「上乗せが小さい」「1 回の forward で済む」と書かれるが、実測はない。[Burda RND (2018/10), Abstract、§1、§2.2、§2.4、Appendix A.3・A.5]

Disagreement は φ(x_t) と a_t で採点し、次の観測を要しない。初期値と bootstrap の部分データを変えた 5 個の順モデルの予測の分散を内的報酬にする。[Pathak Disagreement (2019/06), §2.1 式(1)、§2.3、§3]

BYOL-Explore は観測の履歴と過去・未来の行動を使い、未来の観測の潜在埋め込みを open-loop で K 段先まで予測する（Atari は K=8）。予測先は online encoder の EMA でできた target encoder の出力である。Atari の学習に TPUv2 を 4 個、actor に CPU 400 個、推論に TPUv2 を 4 個使った。[Guo BYOL-Explore (2022/06), §3.2〜§3.4、Appendix C]

SND は状態 s_t の 1 フレームで採点し、行動を使わない。RND の target を固定のランダムでなく自己教師ありの損失（対照型、ST-DIM、VICReg の 3 型）で学び続ける。[Pecháč SND (2023/02), 式(2)(3)、§3.3、§4.1]

AMA は s_t・a_t から s_{t+1} の平均と分散を 2 つの Head で予測し、「二乗予測誤差 − η × 予測共分散の trace」を内的報酬にする。RTX 2080Ti での実時間を報告している（MiniGrid 約 40 分、Space Invaders 約 12 時間）。[Mavor-Parker AMA (2021/02), §3 式(3)(4)、Appendix C.10]

### 2.2 RND と ICM の優劣が分かれる環境の性質

**遷移のばらつき（確率的な遷移、noisy TV）。** RND の原典は予測誤差の源を、学習データの量・確率性・モデルの誤指定・学習の力学の 4 つに分ける。順モデルは確率性のために、白色雑音の TV やコイン投げのような確率的な源へ引き寄せられる。RND は target を決定的でモデル族の中に置けるので、確率性と誤指定の 2 つを除けると説明する。[Burda RND (2018/10), §2.2.1]

RND の原典は sticky actions（確率 0.25）の Atari で、ランダム特徴の順モデル（ICM そのものではない）と比べた。Montezuma's Revenge では順モデルの版が 2 部屋の間を往復し、sticky actions で部屋の境を越えるかを予測できないことが減らない誤差を生む、と説明する。[Burda RND (2018/10), §3.6、Figure 7]

Large-Scale Study は、確率的な遷移では完全なモデルでも報酬の期待値が遷移のエントロピーになり、エージェントがエントロピーの高い遷移を探しに行くと述べる。チャンネルを変える行動を持つ TV を迷路に置くと、IDF・RF とも学習が大きく遅れた。ゲームの実験ではこの効果を観測しておらず、そのために迷路を作ったと書いている。[Burda Large-Scale (2018/08), §5、Figure 6]

Disagreement は、モデルが確率的な遷移の平均を学ぶとアンサンブルの分散が下がる一方、予測誤差は残る、と説明する。Noisy MNIST、TV 付きの迷路、sticky actions の Atari で、予測誤差の基準を上回った。[Pathak Disagreement (2019/06), §2.2、§4.2]

ただし AMA の論文は、行動で起動する noisy TV では RND も弱いと報告する。MiniGrid と、CIFAR-10 の画像を映す行動を足した Space Invaders・Mario で RND の成績が落ち、罠が新奇な状態を多く含むと RND の回避が崩れると述べる。MiniGrid ではアンサンブルの不一致は頑健だった。[Mavor-Parker AMA (2021/02), §1、§4.2、§4.3]

**自分と無関係な画面の動き。** ICM の原典は、観測を変えるものを「制御できるもの」「制御できないが影響するもの」「制御できず影響もしないもの（風に揺れる葉）」に分け、好ましい特徴は前の 2 つを表し最後に左右されないものだとする。逆モデルで特徴を学ぶのはこのためである。VizDoom で画面の 40% を固定位置の白色雑音に置き換えると、ICM は満点を取り、画素空間で予測する版は大きく劣った。[Pathak ICM (2017/05), §2.1、§4.1、Figure 6]

RND の原典に、自分と無関係な画面の動きを扱う実験や議論は見当たらなかった（読んだ範囲）。SND の原典は、Pitfall で動く NPC のために SND のエージェントが雑音を新奇さとして追ったと報告する。[Pecháč SND (2023/02), §4.3]

**操作で変わらない大事な情報。** Large-Scale Study は、IDF の特徴は十分でない場合があり、エージェントがすぐには影響できない環境の大事な側面を表さないことがある、と書く。表では IDF が「安定: No、小さい: Yes、十分: Maybe」、RF は固定なので安定だが十分でない場合がある。[Burda Large-Scale (2018/08), §2.1、Table 1] この性質で RND と ICM を直接比べた実験は見つからなかった。

### 2.3 難探索の Atari での比較

RND の原典は PPO・sticky actions で、RND・PPO・ランダム特徴の順モデル（Dynamics）の最終平均点を比べた。Montezuma's Revenge は RND 8,152、PPO 2,497、Dynamics 400、Private Eye は 8,666、105、33 である。Venture・Gravitar・Pitfall! では Dynamics と同程度で、Pitfall! では両方とも正の報酬を見つけていない。本文は CNN 方策の比較とし、Figure 7 の説明文は RNN 方策と書いていて食い違う。[Burda RND (2018/10), §3.6、Table 1、Figure 7]

Taïga らは Dopamine の Rainbow・sticky actions（0.25）・5 seed で、各ボーナスを同条件で比べた。Montezuma's Revenge の 200M フレームでは RND 5,199.2、ICM 2,795.9、ε-greedy 780.0 で、基準を大きく超えられなかったのは ICM だけだった。[Taïga Bonus-ALE (2021/09), §3.2、Figure 1、Table 5] 数値は arXiv の拡張版から取り、ICLR 2020 版の表と一致するかは未確認である。

BYOL-Explore の Table 1（学習中の最高点）では、Montezuma's Revenge が BYOL-Explore 13,518、RND 4,554、ICM 300、Pitfall が 25,176、0、0 である。難探索 10 ゲームの人間正規化スコアの平均は BYOL-Explore 661、RND 209、ICM 91 だった。基準の RND・ICM が同じアーキテクチャと計算資源で実装されたかの明記は見つからなかった。[Guo BYOL-Explore (2022/06), §4.2、Table 1、Table 3、Appendix B]

SND は同じ PPO の構成で RND を学習し直して比べた（ICM は比較に入っていない）。Montezuma's Revenge の最高点は RND 6,689、SND-V 14,973 で、Mann-Whitney U 検定で有意差が出たのは一部の組だけだった。[Pecháč SND (2023/02), §4.3、Table 5、Table 6]

ICM の原典は VizDoom と Super Mario Bros だけを扱い、Atari の実験はない。[Pathak ICM (2017/05), §3、§4]

### 2.4 BYOL-Explore と SND が改めた点

BYOL-Explore は、RND の固定のランダム射影の代わりに、学習で変わる target（EMA encoder）を潜在空間で予測する。予測先を固定にした ablation は大きく劣り、学習される target の予測が重要だと結論する。1 つの予測損失を、表現の学習・world model・内的報酬に共用する。ICM との違いは多段の open-loop 予測で、1 段だけにすると学習がかなり遅れた環境がある。原典自身は、かなり決定的な力学を前提にしており、確率性の高い環境と手続き生成の環境は今後の課題だと書く。[Guo BYOL-Explore (2022/06), §1、§2、§4.2、§4.3、§5]

SND は、RND の問題として、target の良い初期化が要ること、内的報酬の分散が小さいこと、predictor が追いつくと信号が消えること、状態空間の小さな変化をほぼ捉えないことを挙げ、target を自己教師ありで学び続ける形に替えた。[Pecháč SND (2023/02), §1、§3.2、§3.3、§4.4.1]

## 3. 回数と密度の系

### 3.1 擬似カウントの定義とボーナスの形

CTS の擬似カウントは、密度モデルがある状態に与える確率と、その状態をもう 1 回学習した直後の確率から、「1 回見ると回数が 1 増える」と置いた連立式を解いて得る。モデルが学習で確率を下げない（learning-positive）とき非負になり、経験分布をモデルにすると実際の回数と一致する。予測利得 PG との間に N̂ ≈ (e^{PG} − 1)^{-1} の関係があり、N̂^{-1/2} に比例するボーナスは情報利得のボーナスと少なくとも同じだけ探索的になる、と述べる。[Bellemare CTS (2016/06), §3.1 式(1)(2)、§4 定理1]

DQN の実験のボーナスは β(N̂ + 0.01)^{-1/2}（β = 0.05）で、Double DQN の目標に Monte Carlo 収益を混ぜた変種（MMC）で学ぶ。A3C では N̂^{-1/2} が PG とボーナスなしを上回り、N̂^{-1} の不振は内的報酬の減衰が急すぎるためと推測している。[Bellemare CTS (2016/06), §6.1 式(4)、付録 C.5、図10]

PixelCNN の擬似カウントは同じ定義を引き継ぎ、理論が求める条件（各状態を 1 回だけ学習する、PG が 1/n で減る、learning-positive）と、ミニバッチ・固定の学習率・軽い計算という実際の要件の衝突を整理した。学習率を定数 0.001 にし、PG に 0.1·n^{-1/2} を掛け、負の PG を 0 で切る。ボーナスは N̂^{-1/2} で、外的報酬と足したあと [−1, 1] に切る。[Ostrovski PixelCNN (2017/03), §2.1、§3.2〜§3.3、§4]

#Exploration は状態をハッシュ（学習しない SimHash、手作りの BASS、自己符号化器の符号）で離散化して数える。状態行動の回数も試したが、状態の回数に対して有意な改善はなかった。[Tang #Exploration (2016/11), §2.2〜§2.3、補足 5.4] φ-EB は価値近似と共通の特徴の空間に密度を置き、特徴ごとの KT estimator の積で擬似カウントを作る。[Martin φ-EB (2017/06), §3.1〜§3.3]

### 3.2 Montezuma's Revenge での結果

CTS のボーナスありの DQN は 50M フレームで 15 部屋を探索し（ボーナスなしは 2 部屋）、平均点は 50M で 2,461、100M で 3,439、1 本の run は 6,600 点に達した。MMC なしではボーナスがあっても価値の近似が保てず、性能がすぐ落ちた。[Bellemare CTS (2016/06), §6.1、図3、付録 C.3]

PixelCNN の論文の 150M フレーム中の最大点（3 seed 平均）は DQN-CTS 3,705.5、DQN-PixelCNN 2,514.3、DQN 0 である。57 ゲームでは PixelCNN が CTS を最大点で 52 ゲーム上回った。疎な報酬の最難関ゲームでは、ボーナスと MMC を組み合わせたときにだけ学習が進んだ。[Ostrovski PixelCNN (2017/03), §4.1、図5・図6、§6、付録 D 表2]

φ-EB の評価点は 2,745.4（100M フレーム・5 試行、最大 14 部屋）。[Martin φ-EB (2017/06), §5.2、表1]

### 3.3 後の論文が引き継いだと書くもの

ここでは、後の論文自身が擬似カウントとの関係を書いた箇所だけを挙げる。

- PixelCNN の論文は、#Exploration のハッシュの回数を、擬似カウントの枠組みでの「ハッシュに基づく密度モデル」に当たると位置付ける。[Ostrovski PixelCNN (2017/03), §1]
- #Exploration は擬似カウントを最も近い先行手法とし、近似の回数で未訪問の状態へ汎化する点を共通点、密度モデルを作らずハッシュで汎化する点を違いとする。[Tang #Exploration (2016/11), §3]
- φ-EB は、密度モデルから擬似カウントを導く方法をそのまま採り、密度を置く空間を特徴の空間に替えた。[Martin φ-EB (2017/06), §1、§3.2]
- RND は先行研究のボーナスの形として 1/n と 1/√n を挙げ、擬似カウントは訪問済みの状態に似ていれば未訪問でも正の回数になると説明する。一方で、回数・擬似カウント・情報利得に基づく手法は多数の並列環境へ広げにくいと評価する。[Burda RND (2018/10), §1、§2.1]
- NGU のエピソード内の報酬は 1/√n の形で、n をエピソード内メモリの k 近傍の kernel の和で近似し、本文はこれを擬似カウントと呼ぶ。ただし着想の出典は状態行動の回数のボーナス（Strehl & Littman 2008）と書き、CTS からの継承とは書いていない。[Badia NGU (2020/02), §2 式(2)(3)]
- E3B は、楕円ボーナスは回数に基づくエピソード内ボーナスの一般化だと述べ、特徴を one-hot にすると 1/N_e(s) になるとする。この見方の出典は PC-PG で、CTS ではない。[Henaff E3B (2022/10), §4.1]
- Taïga らは、Rainbow に固定した条件で、Montezuma's Revenge の新しいボーナスが古い擬似カウント法より上でなかったと報告する。文献上の大きな成果は使ったデータ量が違う（RND は 1.97B フレーム、擬似カウント法は 100M フレーム）とも指摘する。[Taïga Bonus-ALE (2021/09), §1、§3.2]

### 3.4 原典が書く苦手な場面

- CTS の擬似カウントは実際の回数の一部にとどまり、比も実際の比と食い違う。密度モデルが状態空間に入れる距離の性質（汎化の出どころ）は扱っていない。DQN の学習が CTS より遅く、速さが合わない。連続空間でこれが正しい回数の概念かは不明だとする。[Bellemare CTS (2016/06), §3.2、§5、§8]
- 深層モデルで learning-positive を保証するのは難しく、最適化が行き過ぎると PG が負になる。探索が易しいゲームではボーナスが報酬の地形を歪めて害になりうる。ボーナスは一時的に現れて消えるので素早く拾える更新則に頼り、Retrace を使う Reactor は疎な報酬のゲームで活かしきれなかった。[Ostrovski PixelCNN (2017/03), §3.3、§4.1、§4.2、§6]
- 画素空間の類似度では意味のある類似度を得にくく、ハッシュの細かさが性能を決める。Montezuma's Revenge で敵の位置をハッシュに入れると、遠くの敵の動きを眺め続けて点が 1,672 に落ちた（除くと 5,661。RAM 入力・手設計ハッシュ）。[Tang #Exploration (2016/11), §2.3、補足 5.1・5.3]
- Freeway では方策に無関係な特徴が多数立ち、β が大きいと立ち止まって交通の配置を見るだけでボーナスを得続ける。[Martin φ-EB (2017/06), §5.2]
- 状態が毎回違う環境ではエピソード内の回数が常に 1 になり、ボーナスが意味を失う。MiniHack では観測に時刻が含まれるため、元の NovelD は完全に失敗した。[Henaff E3B (2022/10), §3、§5.1.2]
- 擬似カウントの原典 4 本に、確率的な環境での弱点を論じた節は見つからなかった（§11）。

## 4. エピソード内の新しさと方策の組

### 4.1 NGU が RND と逆モデルを選んだ理由

NGU は、エピソード内の報酬に一生の新しさの倍率を掛ける（倍率は RND の誤差を正規化して [1, L] に収め、L = 5 に固定）。[Badia NGU (2020/02), §2 式(1)]

エピソード内の埋め込みを逆モデルで学ぶ理由として、原典は、行動と無関係な変化を多く含む環境では、それを埋め込みに入れると意味のない探索に報酬が出ることを挙げる。2 つの連続観測から行動を当てる Siamese network で埋め込みを学び、制御できる状態だけを残す。§1 でも、状態空間のうち制御できない部分を避けるためと書く。[Badia NGU (2020/02), §1、§2 Embedding network の段落] 逆モデルの説明で ICM を名指ししているかは確認できなかった。

一生の新しさに RND を使う理由は、うまく働いたこと、実装が簡単なこと、並列化しやすいことの 3 点である。

```
worked well, is simple to implement and easy to parallelize

よく働き、実装が簡単で、並列化しやすい
```
[Badia NGU (2020/02), §2 Integrating life-long curiosity]

同じ段落は、一生の新しさは勾配法で学ぶのでゆっくり変わり、多くのエピソードを通じて慣れた状態を徐々に下げる役であり、この調整は時間とともに消える、と説明する。[Badia NGU (2020/02), §2]

NGU は難探索の Atari 6 本で評価し、Pitfall! で特権情報なしに正の点を得た最初の手法だと書く。表 1 の Montezuma's Revenge は 10.4k ± 1.6k で、最良の基準手法（11.6k）を下回る。[Badia NGU (2020/02), §4.2、表1] 原典は、密な報酬のゲームで R2D2 に劣ること、高いサンプル複雑度を残る問題として挙げる。[Badia NGU (2020/02), §4.2、§5]

### 4.2 Agent57 と MEME が NGU から改めた点

Agent57 は、行動価値を外的と内的の 2 つの Network に分けた（Q = Q_e + β·Q_i）。それぞれの報酬のスケールと分散に合わせるためで、NGU は不安定になり適切な近似を学べないことがあると書く。方策の組のどれを使うかは、actor ごとに独立した非定常の多腕バンディット（簡略化した sliding-window UCB と ε-greedy）が、割引なしの外的なエピソード収益で選ぶ。BPTT の窓を 2 倍にし、割引率の範囲を 0.99〜0.9999 にした。Atari 57 本すべてで標準の人間基準を上回った。[Badia Agent57 (2020/03), Abstract、§1〜§4]

MEME は Agent57 を出発点に、近似信頼領域、やわらかいトレースの打ち切り（Soft Watkins Q(λ)）、TD 誤差の標準偏差による価値の正規化、全方策を一度に学ぶ cross-mixture training、NFNet 系の構造、方策の蒸留を加えた。外的・内的の Head は分けたまま torso を共有し、分解はもはや厳密な意味を持たないと書く。Head 1 つで学んでも同程度の成績だった。方策の数を 16 に減らし、バンディットを中央の 1 つ（割引版の UCB）にした。内的報酬は NGU のものをほぼそのまま使う。390M フレームで全 57 本の人間基準を上回り、学習予算は 1B フレームである。[Kapturowski MEME (2022/09), §4、§6、付録 D・E・G・H.4]

### 4.3 エピソード内のボーナスの手法

- **NovelD**: RND の新しさの「次の状態と今の状態の差」を正の部分だけ取り、次の状態がそのエピソードで初めてのときだけ与える。行動は使わない。MiniGrid の静的な手続き生成タスクを 120M step ですべて解き、従来の最良（AMIGo）は半分しか解けなかったと書く。[Zhang NovelD (2021/12), Abstract、§1、§3.1]
- **E3B**: 逆モデルで学んだ埋め込みに対し、同じエピソードで見た埋め込みに当てはめた楕円からの距離（φᵀC⁻¹φ）をボーナスにする。Sherman-Morrison の rank-1 更新で 2 乗時間にし、他手法より 1.5〜2 倍遅かった。MiniHack の 16 タスクで最先端を更新し、タスク固有の帰納バイアスを要さないと書く。[Henaff E3B (2022/10), Abstract、§4.1、§4.2、付録 C.1.5・D.2]
- **RE3**: 学習しないランダムなエンコーダの表現で、replay buffer 内の k 近傍との距離の対数をボーナスにする。エンコーダの勾配が要らない。DMC・MiniGrid・Atari（Rainbow と組む）で評価した。[Seo RE3 (2021/02), §1、§3、§4]
- **RIDE**: 連続する状態の埋め込みの差を、エピソード内の訪問回数の平方根で割る。埋め込みは順モデルと逆モデルで学ぶ。MiniGrid の最も難しいタスクを解けたのは RIDE だけと書く。[Raileanu RIDE (2020/02), §3、§4、図3]

E3B は、RIDE・AGAC・NovelD がエピソード内の回数の項に頼っており、外すと 3 手法とも学習に失敗すると報告する。[Henaff E3B (2022/10), §3、図1]

### 4.4 一生とエピソード内のボーナスの使い分け

Henaff et al. 2023 は、文脈（エピソードごとに変わる環境）どうしで共有する構造が少ないときはエピソード内のボーナスが、多いときは一生のボーナスが効くと結論し、文脈をまたぐ価値関数の分散で説明する。時間の長さが短いと、毎エピソードで空間を覆うことは不可能かもしれないとも書く。[Henaff Global-Episodic (2023/06), Abstract、§3.3、§6]

文脈が 1 つの Montezuma's Revenge では、一生の RND が良く、エピソード内の E3B は悪い。2 つを掛け算で組み合わせると性能が頑健になり、MiniHack で新しい最先端になり、Montezuma's Revenge では RND にほぼ並んだ。足し算の組み合わせはどの指標でも E3B より意味のある改善がなかった。NGU の k 近傍のエピソード内ボーナスも試し、うまく働かなかったと書く。[Henaff Global-Episodic (2023/06), §3.4、§3.5、§4.2、付録 F]

## 5. 戻る系と detachment と derailment

### 5.1 detachment と derailment の定義

Go-Explore の 2019 年版は、detachment を、内的報酬の高い前線から離れてしまうことと定義する。内的報酬は使えば減る資源なので、前線の手前は探索済みで報酬が残らず、戻り方も覚えていない、という説明である（2 つの迷路の間から始まる例、図1）。derailment は、有望な状態へ戻る途中の行動が探索の揺らぎ（ε-greedy、行動ノイズなど）で乱され、長く精密な行動列ほど戻れなくなることを指す。[Ecoffet Go-Explore (2019/01), §1、図1]

対象は内的動機の手法全般で、surprise・新奇性・予測誤差を追う手法にも同じ議論が当てはまり、両者が伸び悩みの主因だという仮説を置く。[Ecoffet Go-Explore (2019/01), §1] Nature 版も同じ定義を置き、ロボット課題で回数に基づく内的動機の対照群が報酬を見つけられなかった主因を derailment とする。[Ecoffet First-return (2021/02), Supplementary の detachment・derailment の節]

### 5.2 Go-Explore の版と戻る操作の前提

- **2019 年版**: 有望な状態を archive に保存し（detachment への対処）、探索せずにまず戻ってから探索する（derailment への対処）。戻り方はシミュレータの状態の復元で、学習時は決定的でよくテスト時に確率性があればよいという立場をとる。学習時から確率的な環境での適用は今後の課題で、目標条件の方策で戻る案を挙げる。見つけた軌跡から模倣学習でノイズに頑健な方策を作る段（robustification）を持つ。[Ecoffet Go-Explore (2019/01), §1、§2.1.3、§2.2]
- **Nature 版（復元で戻る版）**: robustification は計算が重く、対象を 11 ゲームに絞った。[Ecoffet First-return (2021/02), Methods]
- **Nature 版（policy-based）**: 状態を復元せず、PPO で学んだ目標条件の方策で archive の cell へ戻る。sticky actions で評価し、Montezuma's Revenge は 12B フレームで 97,728、Pitfall は 10B フレームで 20,093。cell にはドメイン知識を使った。確率的な環境では、復元の版は偶然うまくいった経路を保存しうるが、policy-based の版は実際に戻る必要があるので信頼できる遠回りを学びやすい、と高速道路の例で説明する。確率的な報酬の扱いは今後の課題である。[Ecoffet First-return (2021/02), Methods、Supplementary]

### 5.3 ほかの手法の対処と残る問題

次の表は、各原典が自分で書いた対処だけを載せる。detachment・derailment の語を原典が使っていない手法は、その旨を書き、筆者が対応付けない。

| 手法 | detachment について原典が書くこと | derailment について原典が書くこと | 残る問題として原典が書くこと | 出典 |
|---|---|---|---|---|
| NGU | 語は使わない。新しさが薄れた状態には再訪の誘因がなくなるので、エピソードごとにリセットする記憶でなじみの状態を定期的に再訪させ、内的報酬が時間で消えないようにする | 語は使わない。β=0 の方策で探索を切れる | 密な報酬のゲームで R2D2 に劣る。Montezuma's Revenge では極めて長い信用割当が要る | [Badia NGU (2020/02), §1、§3、§4.2] |
| Agent57 | 語は使わない | 語は使わない | データ効率と計算量 | [Badia Agent57 (2020/03), §5] |
| DTSIL | 語は使わない。記憶した多様な軌跡から 1 本を選ぶ | 語は使わない。軌跡条件の方策で記憶をなぞってから探索する。任意の状態へのリセットを要さず、sticky actions で評価した | 状態表現（Atari の結果は RAM の位置を使う） | [Guo DTSIL (2020/12), §2.2〜§2.3、§3、§4.2、§5.1] |
| MEGA | 語は使わない。密度の低い達成済みの目標を選ぶ | 語は使わない。目標条件の方策で向かい、着いてから探索を増やす（Go-Explore 流と自称） | 意味のある目標空間が与えられている前提、特徴の再パラメータ化に敏感 | [Pitis MEGA (2020/07), §1、§3.2、§6] |
| LEXA | 語は使わない。過去の新しさの再訪ではなく、world model が想像した驚きのある状態を目指す | 原典に記述なし | 多くの課題が未解決、実機での検証が要る | [Mendonca LEXA (2021/10), §1、§2.2、§5] |
| Pîslar et al. | 語は使わない | 語は使わない。エピソード冒頭の状態は訪問済みなので活用から始める利点を示唆し、切り替えでどちらの基本方策だけでも届かない状態に届くとする | 状態分布のずれ、2 モード・7 ゲームに限る | [Pîslar When-explore (2021/08), §2.2、§4] |
| εz-greedy | 語は使わない。ε で選んだ行動を zeta 分布の長さだけ繰り返す | 原典に記述なし | 障害物や力学で長い探索軌跡が無駄になりうる | [Dabney εz-greedy (2020/06), Abstract、§4、§6] |

### 5.4 LLM が答えになる場面と前提

LLM を使う内的報酬そのものは既存の報告で扱った（[LLMの事前知識でRLの探索と報酬を補助する手法](llm_prior_knowledge_rl_survey_2026-10-18.md)、[ELLM](llm_rl_goals_and_reflection_2026-11-07.md#2-ellm有用そうな目標を探索の報酬にする)、[Intelligent Go-Explore](llm_rl_exploration_information_2026-11-07.md#4-intelligent-go-explore戻る場所もllmで選ぶ)）。ここでは detachment・derailment との関係と前提だけを足す。

- **IGE（戻る状態の選択）**: Go-Explore の手作りの判断のうち、archive のどの状態へ戻るか、どう行動するか、新しい状態を archive に入れるかを基盤モデルに置き換える。IGE 自身が Go-Explore の対処した問題として detachment・derailment を挙げるが、語の説明は原典とやや異なる（以前訪れた状態への到達法を忘れること、探索の前にまず戻ることに失敗すること）。戻る操作はシミュレータの状態の復元で、戻る方策を学ぶか過去の軌跡を prompt に入れれば緩められると書く。実験は決定的な環境だけで、確率的な環境は案を示すにとどまる。GPT-4 の API 費用は 1 課題あたり平均 1〜2 USD。[Lu IGE (2024/05), §2、§3.1〜§3.3、Appendix E.1・F]
- **ELLM（目標の提案）**: 原典は detachment・derailment の語を使わず、Go-Explore も引用しない。前提は観測を文に直す captioner があることで、cache なしで 5M step 回すと数千ドルかかり、定期的な問い合わせは費用的に厳しいと書く。[Du ELLM (2023/02), §3.5、§5、Appendix O]
- **Motif（LLM の選好から作る報酬）**: 原典は detachment・derailment の語を使わない。NetHack のメッセージ 2 つのどちらが進展らしいかを LLM に選ばせ、報酬モデルへ蒸留する。選好付けは offline で 1 回、RL 中は小さな報酬モデルを使う。最終の内的報酬は、報酬モデルの値をエピソード内のメッセージの出現回数の β 乗で割るので、エピソード内では繰り返すと減る（キューの論点メモの「訪問で減らない報酬」とは異なる）。新しさの手法が新しさを状態の性質として定義するのに対し、Motif は将来の発見につながりそうな状態に報酬を与える、と対比する。[Klissarov Motif (2023/10), §3 式2、§4.2、Appendix B]

### 5.5 Go-Explore の何が後に残ったか

Nature 版は、覚えて、戻って、そこから探索する原則を強力で、学習一般の基本的な特徴かもしれないと結論する。[Ecoffet First-return (2021/02), Conclusion] 後の論文自身の記述は次のとおりである。

- IGE は archive への保存と有望な状態からの探索を引き継ぎ、手作りの判断だけを基盤モデルに替える。[Lu IGE (2024/05), Abstract]
- DTSIL は Ecoffet らの拡張と自称し、リセットの前提を外す。[Guo DTSIL (2020/12), §3]
- MEGA は目標に着いてから探索を増やす部分を Go-Explore 流として採る。[Pitis MEGA (2020/07), §3.2]
- Cell-Free Latent Go-Explore は、状態空間を cell に分けるドメイン知識を、学習した潜在表現に置き換える。[Gallouédec Latent-GE (2022/08), Abstract]
- NGU・Agent57・εz-greedy・LEXA は Go-Explore に軽く触れるだけで、Agent57 はドメイン知識を使う研究の例、εz-greedy は決定的な forward model に頼る例、LEXA は「本当に新しい状態は探さない」側に置く。[Badia Agent57 (2020/03), §1；Dabney εz-greedy (2020/06), §3；Mendonca LEXA (2021/10), §1、§4]

## 6. 比較研究と 2024 年以降の動向

### 6.1 Taïga らの Rainbow 上の比較

Taïga らは Dopamine の Rainbow に CTS・PixelCNN・RND・ICM を載せ、NoisyNets と ε-greedy を基準に、sticky actions 0.25・200M フレーム・5 seed で比べた。ハイパーパラメータは Montezuma's Revenge で調整し、一部は易しいゲームで調整し直した。[Taïga Bonus-ALE (2021/09), §2、§3.1、§3.2、§3.4、付録 A]

結果は次のとおりである。

- Montezuma's Revenge では RND・CTS・PixelCNN が平均 5,000 点近くで基準を上回り、ICM は大差では超えない。[§3.2、Figure 1]
- 他の難探索 6 本（Gravitar、Private Eye、Venture、Solaris、Pitfall!、Freeway）では ε-greedy が他の手法と同等で、Gravitar では上回った。Montezuma's Revenge の成績は他の難探索ゲームの成績と相関しない。[§3.3、Fig.3(a)]
- 60 本全体では NoisyNets を除いてどの手法も ε-greedy を有意に上回らない。[§3.3、Fig.4〜6]
- 1B フレームに延ばしても難探索ゲームでデータを増やした効果はなく、Montezuma's Revenge で長い学習の効果が出たのは RND だけだった。[§3.5]
- 結論で、今のボーナス型の手法は ε-greedy を超えられず、これまでの進歩はボーナスの改良より交絡要因によるものかもしれないと書き、Rainbow の prioritized replay がボーナスを余分にしている可能性を推測として挙げる。[§4]

### 6.2 RLeXplore の比較と実装の細部

RLeXplore は ICM・RND・Disagreement・NGU・PseudoCounts・RIDE・RE3・E3B の 8 種を PPO の上に統一して実装し、SuperMarioBros・MiniGrid・Procgen Maze・ALE 5 本・Ant-UMaze で比べた。[Yuan RLeXplore (2024/05), §4.2、付録 B]

実装の細部が結果を左右すると書く。

- 観測の正規化は RMS が Min-Max よりおおむね良く、乱数の Network を使う手法では重要。[§5.1 Q1]
- 報酬の正規化はほぼすべての手法で決定的に必要。[§5.1 Q2]
- 更新に使うデータの割合は、全手法で一般に良い設定がはっきりしない。[§5.1 Q3]
- 直交初期化は大半の手法に有益で、初期化に最も敏感なのは RND。[§5.1 Q4]
- LSTM の方策で性能が大きく下がる手法がある。[§5.1 Q5]
- 価値 Head を外的・内的の 2 本に分ける構成は、報酬を単に足す方法を上回る。[§5.2 Q6]
- 複数の内的報酬を組み合わせると単独より質の高い振る舞いが出て、SuperMarioBros では単独で E3B と RIDE が最良、組み合わせでは E3B+RIDE が最高。[§5.3 Q7、Fig.8]

ALE 5 本（25M step、sticky 0.25、3 seed）では 1 つの手法がすべてで勝つことはなく、Montezuma's Revenge と Venture は RND、Gravitar は外的報酬だけ、Private Eye は PseudoCounts、Seaquest は E3B が最高だった。Gravitar では内的報酬で性能が下がる。[Yuan RLeXplore (2024/05), 付録 D.5、Table 14]

### 6.3 DQN 系に内的報酬を足した例

- Taïga らは Rainbow に 4 種のボーナスを足した（§6.1）。
- Agent57 と MEME は R2D2 系の価値ベースの agent に NGU の内的報酬（一生に RND、エピソード内に k 近傍）を載せる。[Kapturowski MEME (2022/09), §3、付録 G.1・G.2] MEME の付録の表では、難探索 3 本の得点が MEME（1B フレーム）と Agent57（90B フレーム）で Montezuma's Revenge 12,437 と 8,666、Pitfall! 46,735 と 13,655、Private Eye 100,799 と 79,348 である。[Kapturowski MEME (2022/09), 付録 M、Table 5]
- BTR は明示的な探索の部品を足しておらず、そのため Montezuma's Revenge など難探索のタスクで苦戦すると書く。NGU の仕組みが役立つかもしれないと将来の課題に挙げる。探索は NoisyNets と ε-greedy の併用で、NoisyNets だけでは十分に探索できなかった。付録の得点は Pitfall が 0、Private Eye が 100。[Clark BTR (2024/11), §3.2、§7、付録 A・F]
- RE3 は Atari で Rainbow と組んだ。[Seo RE3 (2021/02), §4]
- 2024〜2026 年に Rainbow・BTR へ内的報酬を足した論文は、今回の検索では見つからなかった。

### 6.4 2024 年以降の動向

- **一生とエピソード内のボーナスの組み合わせ**: エピソード内のボーナスは共有構造が少ないときに最も効き、2 つを組み合わせると頑健になる（§4.4）。[Henaff Global-Episodic (2023/06), Abstract] MiniGrid では、エピソード内の内的報酬だけで従来の最良に並ぶか上回り、一生の内的報酬だけでは探索がほとんど進まないという報告もある（要旨だけ読んだ）。[Wang Revisiting (2023), 要旨]
- **複数の内的報酬の融合**: RLeXplore の組み合わせの結果（§6.2）と、同じグループの HIRE（査読未確認、要旨だけ読んだ）。[Yuan HIRE (2025/01), 要旨]
- **実装の細部の標準化**: RLeXplore の体系的な調査（§6.2）。
- **内的報酬が効かないベンチマークの報告**: Craftax-1B では ICM・E3B・RND のどれも PPO を改善せず、E3B は報酬を大きく下げ、最も強かったのは PPO-RNN だった。理由は仮説として、報酬が密なことと内的報酬が気をそらす可能性を挙げる。[Matthews Craftax (2024/02), §4.3、Fig.3〜5]
- **LLM・基盤モデルによる内的報酬**: 報酬モデルのサーベイが LLM・VLM の利用を挙げて Motif を扱い、LLM で強化した RL のサーベイが「報酬の設計者としての LLM」の節で内的報酬と探索報酬の例を挙げる。[Yu Reward-Models-Survey (2025/06), §4.2.3；Cao LLM-RL-Survey (2024/10), §V-A]
- 2024〜2026 年の、探索・内的報酬に特化したサーベイ論文は今回の検索では見つからなかった。

## 7. 手法ごとの採点の入力と計算費用

| 手法 | 採点に使う入力 | 学習するもの・メモリ | 費用について原典が書くこと | 出典 |
|---|---|---|---|---|
| ICM | s_t、a_t、s_{t+1} | encoder、逆モデル、順モデル | 数値なし | [Pathak ICM (2017/05), §2.2、§3] |
| RND | s_{t+1} のみ | predictor（target は固定） | 上乗せは小さい、1 回の forward。実測なし | [Burda RND (2018/10), Abstract、§1、§2.2] |
| Disagreement | φ(x_t)、a_t | 順モデル 5 個 | 教師あり学習なので効率的、程度 | [Pathak Disagreement (2019/06), §2.1、§2.3] |
| BYOL-Explore | 観測履歴、過去と未来の行動 | encoder、2 種の RNN、predictor（target は EMA） | Atari: 学習 TPUv2×4、actor CPU 400、推論 TPUv2×4 | [Guo BYOL-Explore (2022/06), §3、Appendix C] |
| SND | s_t のみ | predictor と、自己教師ありで学ぶ target | 実時間なし | [Pecháč SND (2023/02), §3.3、§4.1] |
| AMA | s_t、a_t | 平均と分散の 2 Head の予測 Network | RTX 2080Ti の実時間（Space Invaders 約 12 時間など） | [Mavor-Parker AMA (2021/02), §3、Appendix C.10] |
| CTS | 42×42・3 bit の 1 フレーム | CTS 密度モデル | A3C で上乗せはごく小さい、メモリ 32 GB 未満 | [Bellemare CTS (2016/06), 付録 C.1・C.4] |
| PixelCNN | 同上 | 小型 PixelCNN（2 段・16 枚） | 1 step に評価 2 回と更新 1 回で重くなりうるので小型化。Reactor では 25% の step だけで計算 | [Ostrovski PixelCNN (2017/03), §1、§3.1、§4.2] |
| #Exploration | 最新の 1 フレーム | SimHash・BASS は学習なし、AE は符号器を定期的に学習 | 速く柔軟、BASS は上乗せ小 | [Tang #Exploration (2016/11), §1、§2、§4.2] |
| φ-EB | 価値近似の特徴 φ(s) | 特徴ごとの KT estimator | 1 step に最大 M 個の更新、状態行動空間の大きさによらない | [Martin φ-EB (2017/06), §3.3] |
| NGU | 観測の埋め込み（エピソード内）と観測（RND） | 逆モデルの埋め込み、RND、エピソード内メモリ（容量 30,000、k=10）、32 個の (β, γ) の組 | 高いサンプル複雑度。GPU 1 基、actor 256 個 | [Badia NGU (2020/02), §2、§5、付録 A・F.2] |
| NovelD | o_t と o_{t+1} の新しさ、エピソード内の回数 | RND と回数表 | ハイパーパラメータがほぼない | [Zhang NovelD (2021/12), §1、§3] |
| E3B | 逆モデルで学んだ埋め込み | 逆モデル、エピソード内の C⁻¹ | rank-1 更新で 2 乗時間、他手法より 1.5〜2 倍遅い | [Henaff E3B (2022/10), §4.1、付録 C.1.5] |
| RE3 | 積み重ねた観測 | 学習しないエンコーダ、replay buffer の k 近傍 | エンコーダの勾配が要らない | [Seo RE3 (2021/02), §1、§3] |
| RIDE | 連続する状態の埋め込みの差、エピソード内の回数 | 順モデルと逆モデル | 記述なし | [Raileanu RIDE (2020/02), §3〜§6] |
| Go-Explore | archive の cell と、戻る先の状態 | 復元版は robustification の方策、policy-based 版は目標条件の方策 | robustification は計算が重い | [Ecoffet First-return (2021/02), Methods] |
| IGE | 状態の文・画像表現と archive | 学習なし（基盤モデルの判断） | GPT-4 の API で 1 課題 1〜2 USD | [Lu IGE (2024/05), Appendix E.1] |

## 8. ベンチマークごとの基準と主役

各原典の記述どおりに並べる。筆者は順位を作らない。

| ベンチマーク | 原典が基準・最良とするもの | 出典 | 査読 |
|---|---|---|---|
| Atari の Montezuma's Revenge（Rainbow、200M） | RND・CTS・PixelCNN が約 5,000 点で基準を上回り、ICM は大差では超えない | [Taïga Bonus-ALE (2021/09), §3.2] | あり |
| Atari の難探索 6 本と 60 本全体（Rainbow） | ε-greedy が同等、NoisyNets 以外は ε-greedy を有意に上回らない | [Taïga Bonus-ALE (2021/09), §3.3] | あり |
| Atari の Montezuma's Revenge と Pitfall | Go-Explore が最も強力かもしれない（ただしタスクに特化した設計に依存） | [Yang Exploration-Survey (2022/01), §IV-C2] | 未確認 |
| Atari 57 本 | Agent57 が全 57 本で人間基準を上回り、MEME は 1B フレームの予算で上回る | [Badia Agent57 (2020/03), Abstract；Kapturowski MEME (2022/09), §6] | あり |
| Atari の難探索（BYOL-Explore の比較） | BYOL-Explore が RND・ICM を上回る（基準の実装条件は未明記） | [Guo BYOL-Explore (2022/06), Table 1・3] | あり |
| ALE 5 本（PPO、25M） | ゲームごとに違う（RND、外的のみ、PseudoCounts、E3B） | [Yuan RLeXplore (2024/05), Table 14] | あり |
| MiniGrid | NovelD が静的な手続き生成タスクを全部解く（2021）。エピソード内の報酬だけで従来の最良に並ぶ報告（2023） | [Zhang NovelD (2021/12), Abstract；Wang Revisiting (2023), 要旨] | あり |
| MiniHack | E3B が 16 タスクで最先端（2022）、その後 E3B と RND・NovelD の掛け算が新しい最先端（2023） | [Henaff E3B (2022/10), Abstract；Henaff Global-Episodic (2023/06), §4.2] | あり |
| Habitat | E3B が報酬なしの探索で既存手法を大きく上回る。組み合わせは E3B に届かない | [Henaff E3B (2022/10), §1；Henaff Global-Episodic (2023/06), §3.5] | あり |
| NetHack | Motif が既存手法を大きく上回ると主張 | [Klissarov Motif (2023/10), Abstract] | あり |
| Procgen Maze | どの内的報酬も外的報酬だけを上回る | [Yuan RLeXplore (2024/05), §5.2] | あり |
| Craftax-1B | 探索手法は PPO を改善せず、PPO-RNN が最も強い | [Matthews Craftax (2024/02), §4.3] | あり |

## 9. 批判と懸念

- **ε-greedy との差**: Rainbow の上では、今のボーナス型の手法は ε-greedy を超えられず、Montezuma's Revenge の成績は他の難探索ゲームの成績と相関しない。[Taïga Bonus-ALE (2021/09), §3.3、§4]
- **noisy TV への頑健さの限界**: RND は確率性を除けると原典が説明する一方、行動で起動する noisy TV では成績が落ちる。[Burda RND (2018/10), §2.2.1；Mavor-Parker AMA (2021/02), §4.2、§4.3]
- **新しさが尽きたあとの再訪**: 新しさのボーナスは新しさが消えると再訪を促さず、学習の速さと報酬が消える速さを合わせる必要がある。[Badia NGU (2020/02), §1] SND も predictor が追いつくと信号が消えると書く。[Pecháč SND (2023/02), §3.2]
- **漸近的な非一貫性**: NovelD は、RIDE のように表現の差を報酬にする手法は十分に探索しても内的報酬が 0 に近づかないと批判する。[Zhang NovelD (2021/12), §3.2]
- **エピソード内の回数への依存**: エピソード内のボーナスの多くは回数の項に頼り、状態が繰り返さない環境では意味を失う。[Henaff E3B (2022/10), §3] 回数のボーナスは複雑な環境で失敗するか、タスク固有の特徴抽出に強く依存する。[Henaff Global-Episodic (2023/06), §4.2]
- **戻る系の前提**: Go-Explore の成功はタスクに特化した設計に依存する。[Yang Exploration-Survey (2022/01), §IV-C2] 復元で戻る版は決定的でリセットできる学習環境を前提にし、IGE の実験も決定的な環境だけである。[Ecoffet Go-Explore (2019/01), §2.1.3；Lu IGE (2024/05), Appendix F]
- **比較条件の不揃い**: BYOL-Explore の比較で基準の RND・ICM が同じ条件で実装されたかは明記がない。文献上の Montezuma's Revenge の大きな成果は使ったデータ量が違う。[Guo BYOL-Explore (2022/06), Appendix B；Taïga Bonus-ALE (2021/09), §1]
- **密な報酬の環境での害**: Craftax では内的報酬が気をそらす可能性がある。PixelCNN の論文も、探索が易しいゲームでボーナスが報酬の地形を歪めて害になりうると書く。[Matthews Craftax (2024/02), §4.3；Ostrovski PixelCNN (2017/03), §4.1]
- **LLM の報酬のずれ**: Motif は外的報酬と組むとずれた行動になりうる例を報告する。ELLM は prompt に敏感で、常識を言葉で表せない領域では効きにくいと書く。[Klissarov Motif (2023/10), §4.2、§4.3；Du ELLM (2023/02), §5]

## 10. 総合評価

キューの論点メモの問いに、本冊の出典の範囲で答える。

1. **NGU が RND と逆モデルを使った理由**: 逆モデルの埋め込みは、行動と無関係な変化に報酬が出るのを避け、制御できる状態だけを残すため。RND は、うまく働き、実装が簡単で、並列化しやすいため（§4.1）。
2. **RND と ICM の使い分け**: 確率的な遷移では、RND の原典は予測先が決定的な RND が確率性を除けると説明し、ランダム特徴の順モデルが sticky actions で部屋の間を往復した例を示す（§2.2）。行動で起動する noisy TV では RND も弱い（§2.2）。自分と無関係な画面の動きには、ICM の原典が逆モデルの特徴で白色雑音に左右されない結果を示す（§2.2）。操作で変わらない大事な情報は、逆モデルの特徴が表さないことがあると Large-Scale Study が書く（§2.2）。Montezuma's Revenge の同条件の比較（Rainbow、200M）は RND 5,199、ICM 2,796 で、ICM だけが基準を大きく超えなかった（§2.3）。
3. **CTS と Go-Explore の何が残ったか**: 擬似カウントの考え方は、#Exploration・φ-EB が密度モデルやハッシュに置き換えて引き継ぎ、NGU と E3B はエピソード内の回数の形（1/√n、1/N_e）を使うが、出典としては CTS でなく状態行動の回数のボーナスや PC-PG を挙げる（§3.3）。Go-Explore の「覚えて、戻って、探索する」は IGE・DTSIL・MEGA・Cell-Free Latent Go-Explore が自ら引き継ぐと書く（§5.5）。
4. **今の基準・主役**: Atari の難探索では、Rainbow 上の比較で ε-greedy と同等という報告と、NGU 系（Agent57、MEME）や Go-Explore が高い得点を出す報告が並ぶ。手続き生成・部分観測では、MiniHack が E3B と一生のボーナスとの掛け算、MiniGrid がエピソード内のボーナス、NetHack が Motif、Craftax では探索手法が PPO を改善しない（§8）。
5. **採点の入力と計算費用**: §7 の表のとおり。RND・SND・CTS などは観測だけで、ICM・Disagreement・AMA は行動も使い、BYOL-Explore は行動列を使う。計算費用を実時間で報告したのは AMA と E3B などに限られ、RND と ICM の追加費用の実測は今回読んだ範囲になかった。
6. **detachment・derailment への対処**: Go-Explore は archive と「探索せずに戻る」で対処する。ほかの手法の多くは語を使わず、NGU はエピソード内の新しさで再訪させ、DTSIL・MEGA は記憶した軌跡や目標へ方策で向かってから探索する（§5.3）。原典が残る問題として書くのは、長い信用割当、状態表現、目標空間の前提、データ効率と計算量である。
7. **LLM が答えになる場面**: IGE は戻る状態の選択を基盤モデルに任せるが、状態の復元と決定的な環境を前提にする。ELLM は captioner と問い合わせの費用、Motif は文のメッセージと offline の選好付けを前提にし、Motif の報酬はエピソード内の回数で割るので繰り返すと減る（§5.4）。

## 11. 調査の限界

- **逐語の照合**: 本文は arXiv・ar5iv の HTML 版などから要約ツールで抜き出しており、表の数値と §4.1 の引用は PDF と 1 語ずつ照合していない。特に Large-Scale Study の Table 2 の ± 値、SND の Table 4 の指標の単位、#Exploration の表 1 の Montezuma's Revenge の列は崩れた抽出から読んだ。
- **版の違い**: Taïga らの数値は arXiv の拡張版（2021/09）から取り、ICLR 2020 版と一致するかは未確認。Go-Explore の Nature 版は誌面でなく arXiv の preprint を読み、節番号は ar5iv の番号である。AMA は学位論文の版とある arXiv v3 を読んだ。
- **原典どうし・原典内の食い違い**: RND §3.6 の方策（本文 CNN、図の説明 RNN）、NGU の kernel の ε（本文 10⁻³、付録 0.0001）、Agent57 のバンディットの窓（§4 と付録 G.3）、MEME の方策の数（§3 は 32、付録は 16）とバンディットの名前（付録 D と E）、Agent57 の学習量（約 78〜80B と 90B）、RLeXplore の sticky の確率（本文 0.25%、表 0.25）、SND の target の全結合層の数（§4.3 と §5）。
- **読めなかったもの**: OpenReview と DBLP は bot 確認で開けなかった。Go-Explore 2019 年版の後半、MEGA・Motif・Henaff 2023・NGU・Agent57 の付録の一部、NovelD の 6 ページ以降、Yang らのサーベイの後半、Wang 2023・HIRE・ADOPS の本文は読んでいない。
- **確認できなかった事実**: NGU が逆モデルの説明で ICM を名指ししているか。「操作で変わらない大事な情報」で RND と ICM を直接比べた実験。擬似カウントの原典が確率的な環境での弱点を論じているか。2024〜2026 年に Rainbow・BTR へ内的報酬を足した論文と、探索に特化したサーベイ。
- **書誌**: 被引用数は Taïga（拡張版）・AMA・NovelD（NeurIPS 版）・Cell-Free Latent Go-Explore・Yang らのサーベイで取れていない（Semantic Scholar の 429 など）。Google Scholar の値は取っていない。一部の会議の開催年と所属は Semantic Scholar の会議欄に頼った。
- **引用の数**: 原文の逐語は著作権への配慮から §4.1 の 1 か所に絞った。ほかの主張の根拠は位置で示したので、原文はその位置で確かめてほしい。

## 12. 出典一覧

被引用数は Semantic Scholar の値（取得日 2026-11-18）。査読の有無は会議・雑誌の記録による。

- [Pathak ICM, 2017/05] Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell. "Curiosity-driven Exploration by Self-supervised Prediction." ICML 2017（査読あり）. https://arxiv.org/abs/1705.05363 — 所属 UC Berkeley。被引用 3,210。
- [Burda Large-Scale, 2018/08] Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, Alexei A. Efros. "Large-Scale Study of Curiosity-Driven Learning." ICLR 2019（Semantic Scholar の表記）. https://arxiv.org/abs/1808.04355 — 所属 OpenAI、UC Berkeley、University of Edinburgh。被引用 807。
- [Burda RND, 2018/10] Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. "Exploration by Random Network Distillation." ICLR 2019（査読あり）. https://arxiv.org/abs/1810.12894 — 所属 OpenAI、University of Edinburgh。被引用 1,838。
- [Pathak Disagreement, 2019/06] Deepak Pathak, Dhiraj Gandhi, Abhinav Gupta. "Self-Supervised Exploration via Disagreement." ICML 2019（Semantic Scholar の表記）. https://arxiv.org/abs/1906.04161 — 所属 UC Berkeley、CMU、Facebook AI Research。被引用 493。
- [Guo BYOL-Explore, 2022/06] Zhaohan Daniel Guo, Shantanu Thakoor, Miruna Pîslar, Bernardo Avila Pires ほか. "BYOL-Explore: Exploration by Bootstrapped Prediction." NeurIPS 2022（査読あり）. https://arxiv.org/abs/2206.08332 — 所属 DeepMind。被引用 109。
- [Pecháč SND, 2023/02] Matej Pecháč, Michal Chovanec, Igor Farkaš. "Self-supervised network distillation: An effective approach to exploration in sparse reward environments." Neurocomputing, 2024（査読あり）. https://arxiv.org/abs/2302.11563 — 所属 Comenius University Bratislava ほか。被引用 12。
- [Mavor-Parker AMA, 2021/02] Augustine N. Mavor-Parker, Kimberly A. Young, Caswell Barry, Lewis D. Griffin. "How to Stay Curious while Avoiding Noisy TVs using Aleatoric Uncertainty Estimation." ICML 2022（査読あり）. https://proceedings.mlr.press/v162/mavor-parker22a.html — 所属 University College London ほか。被引用 未確認。
- [Bellemare CTS, 2016/06] Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Rémi Munos. "Unifying Count-Based Exploration and Intrinsic Motivation." NIPS 2016（査読あり）. https://arxiv.org/abs/1606.01868 — 所属 Google DeepMind。被引用 1,799。
- [Ostrovski PixelCNN, 2017/03] Georg Ostrovski, Marc G. Bellemare, Aäron van den Oord, Rémi Munos. "Count-Based Exploration with Neural Density Models." ICML 2017（査読あり）. https://arxiv.org/abs/1703.01310 — 所属 DeepMind。被引用 708。
- [Tang #Exploration, 2016/11] Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel. "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning." NIPS 2017（査読あり）. https://arxiv.org/abs/1611.04717 — 所属 UC Berkeley、Ghent University、OpenAI。被引用 890。
- [Martin φ-EB, 2017/06] Jarryd Martin, Suraj Narayanan S., Tom Everitt, Marcus Hutter. "Count-Based Exploration in Feature Space for Reinforcement Learning." IJCAI 2017（Semantic Scholar・OpenAlex の表記）. https://arxiv.org/abs/1706.08090 — 所属 Australian National University。被引用 134。
- [Badia NGU, 2020/02] Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi ほか. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020（査読あり）. https://arxiv.org/abs/2002.06038 — 所属 DeepMind。被引用 380。
- [Badia Agent57, 2020/03] Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020（査読あり）. https://arxiv.org/abs/2003.13350 — 所属 DeepMind。被引用 613。
- [Kapturowski MEME, 2022/09] Steven Kapturowski, Víctor Campos, Ray Jiang, Nemanja Rakićević, Hado van Hasselt, Charles Blundell, Adrià Puigdomènech Badia. "Human-level Atari 200x faster." ICLR 2023（査読あり）. https://arxiv.org/abs/2209.07550 — 所属 DeepMind。被引用 45。
- [Zhang NovelD, 2021/12] Tianjun Zhang, Huazhe Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph E. Gonzalez, Yuandong Tian. "NovelD: A Simple yet Effective Exploration Criterion." NeurIPS 2021（査読あり）. https://papers.neurips.cc/paper_files/paper/2021/file/d428d070622e0f4363fceae11f4a3576-Paper.pdf — 所属 UC Berkeley、UC San Diego、Tsinghua、Facebook AI Research ほか。被引用 未確認。
- [Henaff E3B, 2022/10] Mikael Henaff, Roberta Raileanu, Minqi Jiang, Tim Rocktäschel. "Exploration via Elliptical Episodic Bonuses." NeurIPS 2022（査読あり）. https://arxiv.org/abs/2210.05805 — 所属 Meta AI Research、University College London。被引用 76。
- [Seo RE3, 2021/02] Younggyo Seo, Lili Chen, Jinwoo Shin, Honglak Lee, Pieter Abbeel, Kimin Lee. "State Entropy Maximization with Random Encoders for Efficient Exploration." ICML 2021（査読あり）. https://arxiv.org/abs/2102.09430 — 所属 KAIST、UC Berkeley、University of Michigan、LG AI Research。被引用 172。
- [Raileanu RIDE, 2020/02] Roberta Raileanu, Tim Rocktäschel. "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments." ICLR 2020（査読あり）. https://arxiv.org/abs/2002.12292 — 所属 Facebook AI Research、NYU、UCL。被引用 225。
- [Henaff Global-Episodic, 2023/06] Mikael Henaff, Minqi Jiang, Roberta Raileanu. "A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs." ICML 2023（査読あり）. https://arxiv.org/abs/2306.03236 — 所属 Meta AI Research、UCL。被引用 24。
- [Ecoffet Go-Explore, 2019/01] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. "Go-Explore: a New Approach for Hard-Exploration Problems." arXiv preprint（査読なし）. https://arxiv.org/abs/1901.10995 — 所属 Uber AI Labs。被引用 469。
- [Ecoffet First-return, 2021/02] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. "First return, then explore." Nature 590, 580–586（査読あり）. https://arxiv.org/abs/2004.12919 — 所属 Uber AI、OpenAI。被引用 504。
- [Lu IGE, 2024/05] Cong Lu, Shengran Hu, Jeff Clune. "Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models." ICLR 2025（査読あり）. https://arxiv.org/abs/2405.15143 — 所属 University of British Columbia、Vector Institute。被引用 35。
- [Guo DTSIL, 2020/12] Yijie Guo, Jongwook Choi, Marcin Moczulski, Shengyu Feng, Samy Bengio, Mohammad Norouzi, Honglak Lee. "Memory Based Trajectory-conditioned Policies for Learning from Sparse Rewards." NeurIPS 2020（査読あり）. https://papers.nips.cc/paper/2020/hash/2df45244f09369e16ea3f9117ca45157-Abstract.html — 所属 University of Michigan、Google Brain。被引用 36。
- [Pitis MEGA, 2020/07] Silviu Pitis, Harris Chan, Stephen Zhao, Bradly Stadie, Jimmy Ba. "Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning." ICML 2020（査読あり）. https://arxiv.org/abs/2007.02832 — 所属 University of Toronto、Vector Institute。被引用 158。
- [Mendonca LEXA, 2021/10] Russell Mendonca, Oleh Rybkin, Kostas Daniilidis, Danijar Hafner, Deepak Pathak. "Discovering and Achieving Goals via World Models." NeurIPS 2021（査読あり）. https://arxiv.org/abs/2110.09514 — 所属 Carnegie Mellon University、University of Pennsylvania、University of Toronto。被引用 179。
- [Pîslar When-explore, 2021/08] Miruna Pîslar, David Szepesvari, Georg Ostrovski, Diana Borsa, Tom Schaul. "When should agents explore?" ICLR 2022（Semantic Scholar の会議欄）. https://arxiv.org/abs/2108.11811 — 所属 DeepMind。被引用 29。
- [Dabney εz-greedy, 2020/06] Will Dabney, Georg Ostrovski, André Barreto. "Temporally-Extended ε-Greedy Exploration." ICLR 2021（査読あり）. https://arxiv.org/abs/2006.01782 — 所属 DeepMind。被引用 40。
- [Du ELLM, 2023/02] Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, Jacob Andreas. "Guiding Pretraining in Reinforcement Learning with Large Language Models." ICML 2023（査読あり）. https://arxiv.org/abs/2302.06692 — 所属 UC Berkeley、University of Washington、MIT、Inria。被引用 299。
- [Klissarov Motif, 2023/10] Martin Klissarov, Pierluca D'Oro, Shagun Sodhani, Roberta Raileanu, Pierre-Luc Bacon, Pascal Vincent, Amy Zhang, Mikael Henaff. "Motif: Intrinsic Motivation from Artificial Intelligence Feedback." ICLR（Semantic Scholar の会議欄。開催年は未確認）. https://arxiv.org/abs/2310.00166 — 所属 Mila、FAIR at Meta、UT Austin ほか。被引用 123。
- [Gallouédec Latent-GE, 2022/08] Quentin Gallouédec, Emmanuel Dellandréa. "Cell-Free Latent Go-Explore." ICML 2023（arXiv の comments 欄）. https://arxiv.org/abs/2208.14928 — 所属・被引用 未確認。要旨だけ読んだ。
- [Taïga Bonus-ALE, 2021/09] Adrien Ali Taïga, William Fedus, Marlos C. Machado, Aaron Courville, Marc G. Bellemare. "On Bonus-Based Exploration Methods in the Arcade Learning Environment." ICLR 2020（査読あり。読んだのは arXiv の拡張版）. https://arxiv.org/abs/2109.11052 — 所属 MILA・Université de Montréal、Google Brain。被引用 75。
- [Yuan RLeXplore, 2024/05] Mingqi Yuan, Roger Creus Castanyer, Bo Li, Xin Jin, Glen Berseth, Wenjun Zeng. "RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning." TMLR（査読あり）. https://arxiv.org/abs/2405.19548 — 所属 The Hong Kong Polytechnic University、Mila・Université de Montréal、Eastern Institute of Technology, Ningbo。被引用 19。
- [Clark BTR, 2024/11] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." ICML（Semantic Scholar の会議欄。開催年は未確認）. https://arxiv.org/abs/2411.03820 — 所属 University of Southampton。被引用 12。
- [Matthews Craftax, 2024/02] Michael Matthews, Michael Beukman, Benjamin Ellis, Mikayel Samvelyan, Matthew Jackson, Samuel Coward, Jakob Foerster. "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning." ICML 2024（Semantic Scholar の会議欄）. https://arxiv.org/abs/2402.16801 — 所属 University of Oxford、UCL。被引用 112。
- [Wang Revisiting, 2023] Kaixin Wang, Kuangqi Zhou, Bingyi Kang, Jiashi Feng, Shuicheng Yan. "Revisiting Intrinsic Reward for Exploration in Procedurally Generated Environments." ICLR 2023（査読あり）. https://iclr.cc/virtual/2023/poster/10945 — 所属 未確認。被引用 18。要旨だけ読んだ。
- [Yuan HIRE, 2025/01] Mingqi Yuan, Bo Li, Xin Jin, Wenjun Zeng. "Deep Reinforcement Learning with Hybrid Intrinsic Reward Model." arXiv（査読未確認）. https://arxiv.org/abs/2501.12627 — 所属 未確認。被引用 2。要旨だけ読んだ。
- [Yang Exploration-Survey, 2022/01] Tianpei Yang, Hongyao Tang, Chenjia Bai, Jinyi Liu, Jianye Hao, Zhaopeng Meng, Peng Liu, Zhen Wang. "Exploration in Deep Reinforcement Learning: A Comprehensive Survey." arXiv v3（査読未確認）. https://arxiv.org/abs/2109.06668 — 所属 天津大学、ハルビン工業大学、西北工業大学。被引用 未確認。前半だけ読んだ。
- [Yu Reward-Models-Survey, 2025/06] Rui Yu, Shenghua Wan, Yucen Wang, Chen-Xiao Gao, Le Gan, Zongzhang Zhang, De-Chuan Zhan. "Reward Models in Deep Reinforcement Learning: A Survey." IJCAI 2025（査読あり）. https://arxiv.org/abs/2506.15421 — 所属 南京大学。被引用 41。
- [Cao LLM-RL-Survey, 2024/10] Yuji Cao, Huan Zhao, Yuheng Cheng ほか. "Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods." IEEE TNNLS（Semantic Scholar の雑誌欄）. https://arxiv.org/abs/2404.00282 — 所属 香港中文大学（深圳）ほか。被引用 283。
