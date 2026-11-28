# Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り

Date: 2026-11-24
Scope: Self-Imitation Learning（SIL）の「実際に得た割引収益で価値を下から支える」考え方を、Q 学習・価値ベースの学習器（DQN 系・IQN）へ広げた手法を集め、Atari・難探索ゲームでの実績、損失の形、記憶の持ち方（別の記憶か Replay 全体か、何を入れるか、抽選とバッチへの混ぜ方、重み）、分布型での扱いを整理する。あわせて、sticky actions など確率的な環境で、運よく得た収益を下限や目標に使ったときの偏りの指摘と、その対策の報告を整理する。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。著作権への配慮から原文の逐語は引かず、要約と位置（節・付録・表・図・式、実装はファイルと関数）で示す。式は記号で書き直している。出典のラベルの年月は、最初に公開された版（arXiv の初版など）の年月である。被引用数は Semantic Scholar の値で、取得日は 2026-11-24。調査は観点ごとに 3 本のサブエージェント（手法と実績、確率的な環境での偏り、公開実装の設定の値）で行い、本冊で統合した。

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. 価値ベースへ広げた手法](#2-価値ベースへ広げた手法)
   - [2.1 起点: SIL の下限ソフト Q 学習](#21-起点-sil-の下限ソフト-q-学習)
   - [2.2 Optimality Tightening](#22-optimality-tightening)
   - [2.3 EMDQN](#23-emdqn)
   - [2.4 一般化した下限 Q 学習](#24-一般化した下限-q-学習)
   - [2.5 SAIL](#25-sail)
   - [2.6 GEM](#26-gem)
   - [2.7 DB-ADP](#27-db-adp)
   - [2.8 DQN 系・分散アクター系に SIL を足した報告](#28-dqn-系分散アクター系に-sil-を足した報告)
3. [3. 難探索ゲームの数値](#3-難探索ゲームの数値)
4. [4. 記憶の持ち方と設定の値](#4-記憶の持ち方と設定の値)
   - [4.1 記憶と抽選](#41-記憶と抽選)
   - [4.2 損失と重み](#42-損失と重み)
   - [4.3 論文と実装の食い違い](#43-論文と実装の食い違い)
5. [5. 分布型での扱い](#5-分布型での扱い)
6. [6. 確率的な環境での偏り](#6-確率的な環境での偏り)
   - [6.1 理論と設計の上での指摘](#61-理論と設計の上での指摘)
   - [6.2 悪化を数値で示した報告](#62-悪化を数値で示した報告)
   - [6.3 sticky actions の有無で比べた報告](#63-sticky-actions-の有無で比べた報告)
7. [7. 偏りへの対策と効果](#7-偏りへの対策と効果)
8. [8. 批判と懸念](#8-批判と懸念)
9. [9. 総合評価](#9-総合評価)
10. [10. 調査の限界](#10-調査の限界)
11. [11. 出典一覧](#11-出典一覧)

## 1. 対象と読み方

- 「下限」は、価値の推定が実際の収益より低いときだけ引き上げる片側の掛け方を指す。「両側」は、記憶の値へ上下どちらからも寄せる掛け方を指す。
- 「確率的な環境」は出典の語に合わせ、遷移の確率性、報酬の確率性、行動の繰り返しの確率（sticky actions）を含む。sticky actions は、各時刻に確率 ς で直前の行動が実行される設定で、標準は ς = 0.25 である [Machado ALE (2017/09), §5.2]。
- 区分: 査読つきの論文、査読なしのプレプリント、著者の公式実装、著者側のブログ、第三者の記事を、出典一覧で分ける。
- 手法ごとに評価の条件（フレーム数、sticky actions の有無、評価の取り方）が違うので、行をまたいだ数値の比較はできない。

## 2. 価値ベースへ広げた手法

### 2.1 起点: SIL の下限ソフト Q 学習

SIL 本体は actor-critic で、Q 学習の学習器では実験していない。後続の手法が出発点として引くので、比較の基準として置く。

- 損失は、収益 R が Q を上回るときだけ二乗誤差を掛ける片側の形 ½((R − Q)₊)² である。根拠は、決定的な環境では行動方策の収益が最適ソフト Q の下限になることで、この関係が成り立つのは「環境が決定的なとき」と条件つきで書かれている [Oh SIL (2018/06), §4.2 式 (11)〜(13) とその直後]。
- SIL の損失（方策の項 −log π(a|s)(R − V)₊ と価値の項 ½((R − V)₊)²）は、エントロピーの係数を 0 に近づけた場合の下限ソフト Q 学習の一形態とみなせる、と著者は述べる [Oh SIL (2018/06), §3 式 (1)〜(3)、§4.3]。
- 記憶は過去のエピソードの (s, a, R) を入れる replay buffer で、R はエピソードの終わりのあとに計算する割引収益。抽選は優先度つきで、優先度は (R − V)₊ [Oh SIL (2018/06), §3、Algorithm 1]。
- Atari の設定: 記憶 10⁵、SIL のバッチ 512、A2C の 1 回の更新ごとに SIL を M = 4 回、SIL 損失の重み 1、価値の項の重み 0.01、優先度の指数 0.6、重要度補正 0.1（難探索の実験）・0.4（全体の評価）[Oh SIL (2018/06), 付録 A Table 3]。
- 上位何本のエピソードだけを残すという記述は論文に無い。公式実装は、生の報酬が 1 回でも正のエピソードだけを記憶に入れる（§4.3）。

### 2.2 Optimality Tightening

DQN の 1 step の目標への二乗誤差に、下限と上限の制約を二次の罰則で足す手法で、下限と上限で両側から挟む形である。

- 損失は (Q − y)² + λ(L^max − Q)₊² + λ(Q − U^min)₊²。下限 L は k 歩先までの実報酬にその先の最大の Q を足した値（k = 1〜K の最大）、上限 U は過去の Q と報酬から作る。境界は目標網で計算し、勾配を流さない [He OT (2016/11), §4 式 (3)・(4)]。
- エピソードの終わりのあとに計算した割引収益 R も下限の計算に入れた [He OT (2016/11), §5]。
- 記憶は DQN の replay をそのまま使い、各 tuple に R を書き足す。抽選は無作為で、その前後の K 歩分の遷移も使う。K = 4、λ = 4 [He OT (2016/11), §4 Algorithm 1、§5]。
- 49 ゲーム・10M フレーム・30 no-op で、人間正規化の平均・中央値は 345.70%・105.74%（DQN 200M フレームは 241.06%・93.52%）[He OT (2016/11), §5.1 Table 1]。
- 数値は arXiv の初版（審査中の版）から取った。採録版は読めていない（§10）。

### 2.3 EMDQN

エピソード記憶に (s, a) ごとの最大の割引収益 H を持ち、Q を TD の目標と H の両方へ二乗誤差で寄せる。H への項は片側ではなく両側で、Q が過大なら記憶が Q を引き下げる、と著者は説明する [Lin EMDQN (2018/05), §3 式 (2)・(4)〜(6)、§4.3]。

- 損失は Σ(Q − S)² + λ(Q − H)²、S は 1 step の DQN の目標、λ = 0.1。λ は 5 ゲームで {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} から選び、大きいと性能が落ちた [Lin EMDQN (2018/05), §3、§4]。
- 記憶は replay とは別の表で、状態を乱数行列で 4 次元に射影した鍵を使い、行動ごとに 500 万状態まで持つ。満杯なら最も長く更新されていないものを置き換える。記憶に無い (s, a) では記憶の項を省く [Lin EMDQN (2018/05), §3、§3.1、§4]。
- 57 ゲーム・40M フレーム・30 no-op で、人間正規化の平均・中央値は 528.4%・92.8%（DQN 40M フレームは 151.2%・52.7%）[Lin EMDQN (2018/05), §4.1 Table 1]。平均が大きいのは同じ状態を繰り返し見るゲームでよく効くため、と著者は述べる [同, §4.1]。

### 2.4 一般化した下限 Q 学習

n 歩の実報酬に n 歩先の学習済みの Q を足した値も最適 Q の下限になる、と示し、SIL の収益（n → ∞）を n 歩の下限に一般化した [Tang LBQ (2020/06), §3.1 定理 1 式 (5)・(6)]。

- 決定的な actor-critic 向けの Q の損失は ½([L̂ − Q]₊)² の片側の形で、ヒンジを除けば n-step Q 学習と同じ形 [Tang LBQ (2020/06), §3.2 式 (8)]。
- TD3 に足す形は L_TD3 + η·L_sil で、実験は TD3 を 1 step、下限を m = 5 歩、η = 0.1 に固定 [Tang LBQ (2020/06), §5.1]。
- Bellman・下限・n-step の演算子の凸結合を考え、下限を含むと不動点の偏りは正の向きに限られ、上は最適 Q で抑えられると示す [Tang LBQ (2020/06), §4 式 (9)、定理 2]。
- 優先度つきの抽選（α 0.6、β 0.1）と重要度補正の両方があるのが最もよく、優先度なしは下がった [Tang LBQ (2020/06), §5.2、付録 D.2 Table 2]。
- 実験は状態入力の連続制御 8 課題だけで、Atari の実験は無い [Tang LBQ (2020/06), §5]。

### 2.5 SAIL

報酬を書き換えて通常の TD の損失に入れる形で、DQN と IQN に適用した [Ferret SAIL (2020/12), §3]。

- 書き換えた報酬は r + α(max(G, Q⁻(s,a)) − max_b Q⁻(s,b))。これは Advantage Learning の書き換えに α(G − Q⁻(s,a))₊ を足したものに分解でき、自己模倣の部分は収益が Q を上回るときだけ効く片側の項になる [Ferret SAIL (2020/12), §3 の分解の式]。
- 収益をそのまま使い ReLU を掛ける素直な移植版（r + α(G − max_b Q⁻)₊）は劣り、著者はその理由に、off-policy で学習が進むと古い軌跡の収益が悲観側へずれること（古い収益の問題）を挙げる [Ferret SAIL (2020/12), §3 式 (1)、図 1]。
- α = 0.9 を全実験で使い、加算する項を [−1, 1] に切る [Ferret SAIL (2020/12), §4.1]。
- 記憶は別に持たず、replay の各遷移に収益 G の欄を足し、エピソードの終わりに後ろ向きに割引和を計算して書き込む [Ferret SAIL (2020/12), Algorithm 1]。優先度は使わない。確率的な環境の影響を強めると考えたため、と著者は述べる [同, §3]。
- 評価の条件は sticky actions 0.25、命の喪失でエピソードを切らない、Dopamine の参照実装とハイパーパラメータ、59 ゲーム、全ゲームの実験は 3 seed [Ferret SAIL (2020/12), §4.1]。難探索ゲームの数値は §3。
- 計算の速さは同じ機材で IQN 100 fps、SAIL-IQN 95 fps、RND-IQN 45 fps [Ferret SAIL (2020/12), §4.4]。

### 2.6 GEM

記憶上の軌跡を逆向きにたどり、各歩で「その先の実収益」と「学習済みの Q」の大きいほうを取った目標へ回帰する [Hu GEM (2021/03), §3.3 式 (4)・(5)]。

- 軌跡に沿って最大を取ると、Q が不偏でも推定の雑音から過大になる、と述べ、2 組の推定器で最良の長さを選ぶ側と値を出す側を分ける（twin back-propagation）[Hu GEM (2021/03), §3.4 式 (6)〜(8)]。各推定器は clipped double Q で作り [同, 式 (9)]、損失は過大な誤差を重く罰する非対称な二乗 (δ)₊² + α(−δ)₊² [同, §3.5 式 (10)]。α の値は論文に無い。
- 収束の保証は遷移が決定的なときだけで、近決定的な場合は性能の下界を示す [Hu GEM (2021/03), §4 定理 2・定義 4.1・定理 3]。
- Atari は 6 ゲーム・最大 10M step・3 seed の曲線だけで、難探索ゲームは入っていない。ゲーム名は図の見出しから読んだ [Hu GEM (2021/03), §5.2 図 3]。

### 2.7 DB-ADP

SIL を名乗る手法ではないが、経験から作った表形式の MDP で動的計画法をした値を下限として、Q の目標を下から切り上げる [Ren DB-ADP (2021/09), §4.1 式 (6)・(7)]。

- 目標は y = max(y_Boots, y_DP)。決定的な下限で切ると、目標の分散が元より大きくならない [Ren DB-ADP (2021/09), §4.2 命題 4]。
- y_DP が過大にならないよう、表形式の BCQ で控えめに推定し、見たことのない (s, a) を刈り込む [Ren DB-ADP (2021/09), §4.1、付録 B.5]。
- Atari は 12 ゲーム・10M step で、難探索ゲームは入っていない。30 no-op [Ren DB-ADP (2021/09), §5.2 図 4、付録 B.1・B.4]。

### 2.8 DQN 系・分散アクター系に SIL を足した報告

- SAIL の論文が、素直な移植版を strSIL-DQN・strSIL-IQN として全ゲームで比べている。難探索ゲームでの SAIL の相対改善は、DQN 系で平均 +72.2%・中央値 +45.2%、IQN 系で +16.0%・+22.3% [Ferret SAIL (2020/12), §4.6 図 11]。strSIL のゲームごとの最終の点の表は見つからなかった。
- Ape-X・R2D2 など分散アクター系に SIL を足した報告は見つからなかった（検索語は §10）。
- 本文を読めなかったものに、エピソード記憶のグラフで価値を逆向きに伝える ERLAM [Zhu ERLAM (2020/04), 要旨] と、経験の割引収益を Bellman の目標の下限に使う Value Target Lower Bounding（一次資料の所在を特定できず、第三者の索引の要旨だけ）がある。

## 3. 難探索ゲームの数値

各論文の表から写した。条件が論文ごとに違うので、行をまたいで比べない。

| 手法・論文 | 条件 | Montezuma | Private Eye | Venture | Gravitar | Pitfall | Solaris | 位置 |
|---|---|---|---|---|---|---|---|---|
| A2C+SIL（参考） | 49 ゲーム、200M フレーム、30 no-op | 1100.0 | 661.2 | 0.0 | 1874.2 | — | — | [Oh SIL (2018/06), 付録 Table 5] |
| A2C+SIL（難探索の実験） | 同上、重要度補正の値が別 | 2500 | 8684 | 0 | 2722 | — | — | [Oh SIL (2018/06), §5.3 Table 1。表の注記はプロットから読んだ値] |
| Optimality Tightening | 10M フレーム、30 no-op | 23.33 | 342.37 | 286.67 | 346.67 | — | — | [He OT (2016/11), 付録 Table S1] |
| DQN／AL-DQN／SAIL-DQN | sticky 0.25、Dopamine | 0／0／0 | −114／153／10068 | 35／371／989 | 299／540／1292 | −84／−100／−27 | 1437／2216／855 | [Ferret SAIL (2020/12), §4.2 Table 1] |
| IQN／AL-IQN／SAIL-IQN | 同上 | 0／0／833 | 357／49／4956 | 74／19／1525 | 766／1213／1375 | −19／−280／−42 | 2165／1461／992 | [Ferret SAIL (2020/12), §4.3 Table 2] |

- Montezuma's Revenge で、SAIL-IQN は最終 833（期間平均 513）、RND-IQN は 161（45）、IQN は 0。SAIL-IQN に RND と n = 3 を足すと最終 5180 で、Taïga らの RND-Rainbow と並ぶ、と著者は述べる [Ferret SAIL (2020/12), §4.4 図 8]。
- SAIL の論文は総学習量を数で書いていない。図 2 の横軸は 0〜100M フレーム、図 10 の横軸は 0〜200M フレーム [Ferret SAIL (2020/12), §4.1、図 2・図 10]。
- EMDQN、GEM、DB-ADP の論文には、難探索ゲームの個別の値が無い（§2.3・§2.6・§2.7）。

## 4. 記憶の持ち方と設定の値

### 4.1 記憶と抽選

値は公式実装の既定か起動スクリプトの値。論文の値が違うものは §4.3 に書く。

| 項目 | SIL（A2C、Atari） | Tang（TD3） | SAIL-DQN・SAIL-IQN | Optimality Tightening | EMDQN | GEM（Atari） |
|---|---|---|---|---|---|---|
| 記憶の大きさ | 10⁵（SIL 専用） | 10⁶（TD3 用とは別の SIL 専用） | 10⁶（通常の replay と共用） | 10⁶（共用） | 行動ごとに 500 万（別の表） | 10⁶（共用） |
| 入れるもの | 生の報酬が正のエピソードの遷移だけ | 全 n 歩の組 | 全遷移と収益 | 全遷移と収益 | 鍵（4 次元の射影）と最良の割引収益 | 全遷移と、長さごとの目標 |
| 収益の計算 | 終わりに割引和、報酬は符号、γ 0.99 | n 個の報酬の割引和＋γⁿ·min Q_targ | 終わりに割引和（報酬は Dopamine の Runner で [−1, 1]） | 終わりに割引和、報酬は切る、γ 0.99 | 終わりに割引和、報酬は切る、γ 0.99、既存と max | 終わりに長さごとの値（収集時の Q で継ぎ足す） |
| 時間上限の打ち切り | 継ぎ足さない（最後を終端として扱う） | 継ぎ足す | 継ぎ足さない（終端として保存）。未完了のエピソードの遷移は抽選から外す | 平均の Q で継ぎ足す | 未確認 | 継ぎ足さない |
| 抽選 | 優先度に比例（優先度は R − V を [0, 1] に切った値、α 0.6） | 優先度に比例（下回り量の和、α 0.6、β 0.1） | 一様 | 一様（前後の遷移も） | 一様 | 未確認 |
| 1 回の件数と頻度 | 512 件、80 遷移ごとに 4 回 | 100 件、エピソードの終わりに長さの回数 | 32 件、4 step に 1 回 | 32 件、4 step に 1 回 | 32 件、4 step に 1 回 | 32 件、500 step に 25 回 |
| 通常のバッチとの関係 | 別のバッチで別の最適化の段（最適化器は共有） | 別のバッチを同じ更新の別の項に | 同じバッチの目標に加算 | 同じバッチの目標を書き換え | 同じバッチに 2 つ目の目標 | 同じバッチで別の Head |

出典: [SIL 実装, self_imitation.py の SelfImitation・PrioritizedReplayBuffer、a2c_sil.py、run_atari_sil.py]、[nstep-sil 実装, td3_nstep_sil.py の td3、prioritized_buffer.py、nstep_wrapper.py]、[sail_rl 実装, common/replay_buffer.py、agents/sail_dqn.py、configs/atari_dqn.gin・atari_iqn.gin]、[OT 実装, run_OT.py]、[emdqn 実装, train.py の update_ec、lru_knn.py]、[GEM 実装, gem_atari の episodic_memory_ddq.py、atari_ddq.py、common.py]。

### 4.2 損失と重み

| 項目 | SIL（A2C） | Tang（TD3） | SAIL-DQN | SAIL-IQN | Optimality Tightening | EMDQN | GEM（Atari） |
|---|---|---|---|---|---|---|---|
| 損失の形 | 方策: adv·nlogp（adv は [0, 1]）。価値: 差を [−1, 0] に切った勾配 | relu(目標 − Q)² | Huber（目標に加算） | 分位点 Huber κ = 1（目標に加算） | Huber、和 | Huber(TD) + 0.1·Huber(Q − H) | 通常の Head: Huber。記憶の Head: 非対称な Huber |
| 重み | 全体 1.0、価値 0.01、エントロピー 0.01 | 0.1 | α 0.9、加算は [0, 0.9] | α 0.9、加算は [−0.9, 0.9] | 目標の混合 0.8、余白 0.1 | λ 0.1 | 記憶の Head ×10、alpha 1.0 |
| 勾配の clip | 大域ノルム 0.5 | 無し | 無し | 無し | Huber で代用 | 10 | 無し |
| sticky actions | 無し | — | 0.25 | 0.25 | 0 | 無し | 既定は無し |

出典は §4.1 と同じ。

### 4.3 論文と実装の食い違い

- **SIL**: 論文の Algorithm 1 は全エピソードを記憶に入れるが、公式実装は生の報酬が 1 回でも正のエピソードだけを入れる [SIL 実装, self_imitation.py の SelfImitation.update_buffer]。論文の (·)₊ は下側だけを切るが、実装は advantage を [0, 1]、価値の差を [−1, 0] に切る [同, SelfImitation.build_loss_op]。重要度補正の β の既定は、クラス 1.0、learn 0.0、起動スクリプト 0.1 の 3 か所で違う [同, SelfImitation.__init__、a2c_sil.py の learn、run_atari_sil.py の main]。
- **Tang**: 論文が推す n = 5 に対し、コードの既定は n = 1（README の例は n = 5、重み 0.1）。コードは β = 0 を assert で止める [nstep-sil 実装, README.md、td3_nstep_sil.py の __main__、prioritized_buffer.py の sample]。
- **SAIL**: 論文は加算の項に ReLU を掛けず [−1, 1] に切る。IQN 版のコードは同じ形だが、DQN 版のコードは relu を掛けてから [0, 1] に切るので負の加算が出ない [sail_rl 実装, agents/sail_dqn.py の _build_target_q_op、agents/sail_iqn.py の _build_target_quantile_values_op]。論文は IQN を n-step なしで使うと書くが、IQN の gin には n = 3 の指定がある。実行時にどちらの値になるかは未確認 [Ferret SAIL (2020/12), §4.3; sail_rl 実装, configs/atari_iqn.gin]。
- **Optimality Tightening**: 論文は λ = 4 の二次の罰則だが、コードは罰則を使わず、重み 0.8 で目標の値を混ぜる形 [OT 実装, run_OT.py]。
- **EMDQN**: 論文は 2 つの二乗の項の勾配をそれぞれ [−1, 1] に切ると書き、コードは Huber 損失と勾配の clip 10 [emdqn 実装, build_graph.py の build_train、train.py]。
- **GEM**: 論文の Table 2（τ 0.6、バッチ 100）と MuJoCo のコード（tau 0.005、バッチ 128）で値が違う。非対称な損失の α は論文に値が無く、コードは MuJoCo 0.5・Atari 1.0 [Hu GEM (2021/03), 付録 C Table 2; GEM 実装, gem_mujoco/run/train.py、gem_atari/common.py]。

## 5. 分布型での扱い

- 分布型（C51・QR-DQN・IQN）と組み合わせた報告は、調べた範囲では SAIL-IQN だけだった（検索語は §10）。
- SAIL-IQN の論文は、IQN にも同じ報酬の書き換えを掛けるとだけ書き、分位点への掛け方は書かない [Ferret SAIL (2020/12), §4.3]。
- 公式実装では、Q(s, a) と max の Q を、目標網の分位点を 32 個平均した値から取る。加算の項は遷移ごとの 1 つのスカラーで、報酬に足してから、64 個の目標分位点のすべてに同じ値を足す（分布全体の平行移動）。分位点 Huber 損失（κ = 1）は IQN のまま [sail_rl 実装, agents/sail_iqn.py の _build_target_quantile_values_op、configs/atari_iqn.gin]。
- SIL 本体、Optimality Tightening、EMDQN、Tang、GEM、DB-ADP は、論文・実装とも分布型の記述が無い。

## 6. 確率的な環境での偏り

### 6.1 理論と設計の上での指摘

- **SIL**: 下限の式は期待値で書かれているが、実装の損失は 1 本の軌跡の収益を使う。Q* ≥ R > Q が成り立つ根拠は「環境が決定的なとき」と条件つき。確率的な環境での扱いは本文と結論に見つからなかった [Oh SIL (2018/06), §4.2]。
- **一般化した下限 Q 学習**: 1 本の軌跡の収益を下限の式に代入すると、double-sample の問題で偏りが入り、遷移や方策の確率性で収益の分散が大きいときに特にそうなる [Tang LBQ (2020/06), §2.3]。実験の連続制御は方策も遷移も決定的なので、1 本の収益が正確だと明記する [同, §5.1 と脚注 1]。
- **SAIL の著者による SIL の評**: SIL は遷移の確率性と報酬の分散に弱い可能性があり、楽観的な更新が、たまに高い報酬をもたらす行動の価値を過大にしうる [Ferret SAIL (2020/12), §2]。
- **SIL を実演に広げた手法**: 方策や遷移の確率性で収益の分散が大きいと、価値が期待ではなく最高の収益を推定してしまう、と下限の過大の原因に挙げる [Pshikhachev SILfD (2022/03), §3.3]。
- **MFEC**: 表の値は max で更新されて下がらないので、確率的な環境では強いリスク志向になる、と書く [Blundell MFEC (2016/06), §2、脚注 1]。
- **EMDQN**: 狙いを近決定的な環境に置き、確率的な環境の扱いは今後の課題とする [Lin EMDQN (2018/05), §2.2、§5〜6]。
- **GEM**: 収束の定理は遷移が決定的なことを条件とし、この前提は記憶系の手法に共通だと述べる。近決定的な MDP では方策の価値に下界を示す [Hu GEM (2021/03), §4.2 定理 2・定義 4.1・定理 3]。
- **過去の上位の rollout を真似る自己模倣**: 確率的な環境では、過去の上位の rollout を真似るだけでは最適方策を取り戻しにくい、と 2 本腕のバンディットの例で述べる（実験は見つからなかった）[Gangwani SIDP (2018/05), §2.2 末]。
- **The Brute**: 履歴の木の上で下限を保つ手法で、それが正しい下限になるのは根の下の部分木が完全に決定的なときだけだと書く [Machado ALE (2017/09), 付録 A.2 式 (1)]。
- **Go-Explore**: 決定的にした探索で見つけた近道は、確率的な本番では確実に再現できないことがあり、その近道だけが頑健化に回ると頑健化できない、と述べる [Ecoffet Go-Explore (2019/01), §4; Ecoffet FRTE (2020/04), 補足 "Policy-based Go-Explore and Stochastic Environments"]。
- **Episodic control の源流**: 最良の系列に従う方式は、少ない経験では有利だが、単一の標本に頼る不利がいずれ効いてくる [Lengyel & Dayan (2007/12), §4・§5]。

### 6.2 悪化を数値で示した報告

- **The Brute（sticky actions の有無）**: 50M フレーム学習、学習中の直近 100 エピソードの平均で、決定的（ς = 0）から sticky（ς = 0.25）にすると、Asterix 6,909 → 308、Beam Rider 1,132 → 428、Seaquest 621 → 81、Space Invaders 1,432 → 148、Freeway 1.1 → 0.0。同じ条件の DQN と Sarsa(λ)+Blob-PROST はほぼ変わらなかった [Machado ALE (2017/09), §5.2.1 表 2]。開始時のランダム no-op だけでは、開始状態のあとが決定的なままなので、このような手法はなお好成績になる、と短所に挙げる [同, §5.3]。
- **MFEC・NEC（確率的にした古典制御）**: CartPole・MountainCar・LunarLander に、報酬のガウス雑音、確率 0.2 の報酬の符号の反転、確率 0.5 で次の観測が今のまま、の 3 通りを入れた 9 課題で、MFEC は速く学ぶが過度に楽観的な制御のため課題を解けず、難しい問題では MFEC・NEC は学習できなかった。DQN は遅いが学んだ [Le MBEC (2021/11), §4.2 図 3、付録 B.3 図 7]。数値は図だけで、読み取っていない。
- **記憶した行動系列をそのまま繰り返す方式**: DTSIL の比較用の DTRA は、遷移の確率性のため危険物を避けられず、Montezuma's Revenge と Pitfall で他の手法より大きく劣った [Guo DTSIL NeurIPS (2020/12), §4.2 図 4]。数値は図だけ。
- **SIL 本体**: 確率性の有無や強さを変えて SIL の悪化量を示した報告は見つからなかった（検索語は §10）。

### 6.3 sticky actions の有無で比べた報告

| 手法 | 条件と結果 | 位置 |
|---|---|---|
| SAIL（DQN） | 全ゲームで AL-DQN に対する相対改善は、sticky ありで平均 +101.4%・中央値 +6.7%、sticky なしで平均 +231.6%・中央値 +12.7%（図 9 の説明文は +222.6%・+11.7% で本文と食い違う）。決定的に近い設定で伸びは大きいが、sticky ありと大きくは離れない、と著者は述べる | [Ferret SAIL (2020/12), §4.2・§4.5、図 3・図 9] |
| SAIL（繰り返しの確率を上げる） | 難探索ゲームで、確率を 0.5・0.75 に上げると SAIL-DQN の人間正規化の中央値は下がったが、標準（0.25）の DQN よりは上。手法固有の影響と問題が難しくなった分の切り分けは今後の課題とする。数値は図だけ | [Ferret SAIL (2020/12), §4.5 図 10] |
| The Brute | §6.2 のとおり、5 ゲームすべてで大きく下がった | [Machado ALE (2017/09), §5.2.1 表 2] |
| 1 本の人のデモから始める方式（Backward） | 方策の最高点 74,500 に対し、sticky 0.25 で評価すると平均 10,000、確率 0.01 のランダム行動で平均 8,400。0〜30 回のランダム no-op ではあまり下がらなかった。最高点と平均で指標が違う | [Salimans Backward (2018/12), 要旨、§4.2] |
| Go-Explore（著者のブログ、査読なし） | sticky なしで頑健化した網を sticky で評価すると点が下がり、頑健化に sticky を入れて学習し直すと回復した（Montezuma 知識あり 281,264・知識なし 33,836、Pitfall 知識あり 20,527）。12/3 の追記の値の並び順は確かめられなかった | [Uber Go-Explore blog (2018/11), 本文と 12/3・12/12 の追記] |
| Go-Explore（Nature 版） | sticky actions を外すと問題が大きく簡単になると述べ、no-op だけで評価した研究を最先端の比較から外した | [Ecoffet FRTE (2020/04), 補足 "Policy-based Go-Explore vs. DTSIL"、方法 "State of the art on Atari"] |
| GEM | sticky actions を入れた 3 ゲームで、DQN 系の比較手法に対してなお競争力があると述べる。同じゲームで sticky の有無を直接比べた記述は無い | [Hu GEM (2021/03), 付録 D 図 5] |

- DTSIL の評価条件は版で食い違う。初版は 0〜30 回のランダム no-op [Guo DTSIL v1 (2019/07), 付録 表 3]、NeurIPS 版は sticky actions による確率性があると書く [Guo DTSIL NeurIPS (2020/12), §4.2]。Go-Explore の Nature 版の補足は、DTSIL（初版を引用）は sticky actions なしで試したと書く [Ecoffet FRTE (2020/04), 補足]。
- sticky actions なしで評価した手法: SIL 本体（30 no-op）[Oh SIL (2018/06), 付録 Table 5]、Optimality Tightening [He OT (2016/11), §5]、EMDQN [Lin EMDQN (2018/05), §4.1]。
- Episodic control 系（MFEC・NEC・EMDQN・GEM・MBEC）を sticky actions の有無で比べた報告は見つからなかった。

## 7. 偏りへの対策と効果

- **n 歩の下限**: 有限の n でブートストラップする下限は、1 本の収益の double-sample の偏りを一部和らげると期待する、と書く [Tang LBQ (2020/06), §3.1]。効果の測定は決定的な連続制御だけで、TD3 に 5 歩の下限を足すと、1 step・5 step の TD3 や収益の SIL より概ねよく、n = 5 が最良だった [同, §5.1 図 2、§5.2]。確率的な環境での効果の数値は無い。同じ方向の提案はほかにもある（効果は測っていない）[Pshikhachev SILfD (2022/03), §3.3]。
- **下限の重み**: 演算子の凸結合で下限の成分の重みを変えると、収縮の速さと正の偏りの釣り合いが変わる。実験は重み 0.1 に固定 [Tang LBQ (2020/06), §4.2 定理 2、§5.1]。
- **記憶の項の重み**: EMDQN は、確率的な環境では記憶の項を正則化とみなせ、λ を動的に調整できると述べる（確率的な環境での実験は無い）[Lin EMDQN (2018/05), §3・§5]。
- **二重推定**: GEM の twin back-propagation は、期待値で真の最大を過大にしないと示す。過小側の偏りを入れうるが、軌跡に沿って伝わる過大よりはずっとよいと述べる [Hu GEM (2021/03), §3.4 式 (6)〜(8)、§4.1 定理 1]。twin back-propagation や非対称な損失を外すと過大推定が大きくなり、とくに Humanoid で性能が大きく落ちた。単なるアンサンブルの寄与は小さかった [同, §5.3 図 4]。ここでの過大は推定の雑音によるもので、確率的な環境に限った測定ではない。
- **拡散係数（後ろ向き更新）**: EBU は、確率的な有限 MDP で拡散係数 β が十分小さければ収束すると示し、確率的な遷移を入れた MNIST の迷路で他の手法を上回った [Lee EBU (2018/05), §4.2 定理 2、§5.1 図 2(c)]。
- **記憶の書き方を平均に寄せる**: MFEC は、max を外して収益をそのまま書くと楽観は弱まるが、予備実験では性能が下がったと書く（Atari を決定的と明記した文脈）[Blundell MFEC (2016/06), 脚注 1、§4.1]。MBEC は近傍の記憶へ距離に応じて書き込み、平均への収束を示し、確率的な古典制御 9 課題で MFEC・NEC・DQN を上回った（数値は図だけ）[Le MBEC (2021/11), §3.2、§4.2、付録 A.2]。
- **古い収益**: SAIL は、学習が進むと古い軌跡の収益が悲観側へずれるので、観測した収益と今の Q の大きいほうを使う [Ferret SAIL (2020/12), §1・§3 図 1]。これは悲観側のずれへの対処で、確率性による楽観側の偏りを抑える目的とは書かれていない [同, §3]。
- **優先度を使わない**: SAIL は、優先度つきの抽選は確率性の影響を強めると考えて使わない [Ferret SAIL (2020/12), §3]。効果を比べた数値は見つからなかった。
- **方策で戻る形**: 状態の復元ではなく目標つきの方策で戻ると、確実な方策が無い近道は通らず、確実な遠回りを学びやすい、と述べる。到達の成功率を追って上書きを控える仕組みも挙げるが、Montezuma's Revenge と Pitfall では要らなかったので実装・検証していない [Ecoffet FRTE (2020/04), 補足 "Policy-based Go-Explore and Stochastic Environments"]。
- **複数のデモを使う**: 1 本のデモからの頑健化は成功が 40% だったが、10 本を同時に使うと 5 回すべて成功した [Ecoffet Go-Explore (2019/01), §3.1.1、付録 A.7]。
- **記憶の減衰、下限を掛ける頻度・時期**: 記憶の古いものを外す処理は MFEC・NEC・MBEC にあるが、どれも容量の管理として書かれている [Blundell MFEC (2016/06), §2; Pritzel NEC (2017/03), §3; Le MBEC (2021/11), §3.2]。確率性への対策として記憶を減衰させたり、下限を掛ける頻度・時期を変えたりして効果を測った報告は見つからなかった。SIL 本体は、序盤の活用しすぎで準最適に留まったゲームで SIL の更新を後半に減らすと解消したと述べるが、確率性への対策としてではない [Oh SIL (2018/06), §5.4]。

## 8. 批判と懸念

- **決定性への依存**: 第三者の批判として、決定的な環境で見つけた解は確率的な版で再現できない形で決定性を使いうるので、「決定的に解いてから乱す」方針に懐疑的だと述べる記事がある [Irpan (2018/11), 第三者のブログ]。
- **古い収益**: off-policy で過去の方策の収益を使うと、学習が進むにつれ悲観側にずれ、自己模倣の寄与が消えていく [Ferret SAIL (2020/12), §3 図 1]。
- **過大評価との相互作用**: 元から過大評価する基盤（DDPG）では下限の損失の利得が小さく、過大評価が強いと下限の前提が崩れるためだろう、と推測する [Tang LBQ (2020/06), §5.2、付録 D.2]。記憶上で最大を取ると過大評価が軌跡に沿って伝わる [Hu GEM (2021/03), §3.4]。DB-ADP は偏りを減らす厳密な保証を持たず、DP 側の誤差が大きいと下限を通じて過大評価を広げうる [Ren DB-ADP (2021/09), §4.2]。
- **序盤の活用しすぎ**: A2C+SIL は一部のゲーム（James Bond・Star Gunner）で準最適に留まった [Oh SIL (2018/06), §5.4]。EMDQN も BattleZone で後半に性能が少し下がり、λ を後半で下げれば補えるかもしれないと推測する [Lin EMDQN (2018/05), §4.2]。
- **難探索での空振り**: SAIL-DQN は Montezuma's Revenge で 0 のまま [Ferret SAIL (2020/12), Table 1]、SIL 本体は Venture で 0 [Oh SIL (2018/06), 付録 Table 5]。SAIL は Pitfall と Solaris で基準より上がっていない [Ferret SAIL (2020/12), Table 1・2]。
- **決定性を前提にしてよいとする側の主張**: 近決定的な状況は多いので、それを使う仕組みは学習を速くする [Blundell MFEC (2016/06), §2; Lin EMDQN (2018/05), §2.2]。下限による正の偏りは最適 Q で上から抑えられ安全で、実験では有益だった [Tang LBQ (2020/06), §4.2]。運のよい遷移の系列を確実に繰り返せない問題は、自分たちの実験では大きな障害にならなかった [Ecoffet FRTE (2020/04), 本文 "Policy-based Go-Explore"]。

## 9. 総合評価

- 価値ベースの学習器で、sticky actions のある Atari の難探索ゲームまで数値を出したのは、調べた範囲では SAIL だけだった。Montezuma's Revenge は SAIL-DQN が 0、SAIL-IQN が 833 で、RND と n = 3 を足すと 5180 だった（§3）。
- 価値ベースへの広げ方は 3 通りに分かれる。片側の下限を別の損失として足す形（Tang、SIL 本体）、目標そのものを書き換える形（SAIL、DB-ADP、GEM）、記憶の値へ両側から寄せる形（EMDQN、Optimality Tightening は下限と上限の両側）（§2）。
- 記憶の持ち方は、SIL 本体と Tang が別の優先度つきの記憶、SAIL と Optimality Tightening が replay の遷移に収益を書き足す形、EMDQN と DB-ADP が別の表である。上位何本のエピソードだけを残す方式は見つからなかった。SIL の公式実装は、正の報酬を含むエピソードだけを入れる（§4）。
- 分布型では、SAIL-IQN が、目標網の分位点を平均した Q から作ったスカラーを、すべての目標分位点に同じだけ足す形だけが見つかった（§5）。
- 確率的な環境での偏りは、SIL 本体・Tang・GEM の理論がどれも決定的な環境を前提にしていることと、1 本の収益の代入で偏りが入ることが指摘されている。悪化を数値で示した報告は、記憶して下限を最大化する The Brute（sticky で大きく低下）と、max で書く記憶（確率的な古典制御で解けない）だった。SAIL は sticky ありでも改善を保ったが、改善の幅は sticky なしより小さく、繰り返しの確率を上げると中央値が下がった（§6）。
- 対策として効果の記述があるのは、n 歩の下限（決定的な課題でだけ測定）、二重推定（GEM）、拡散係数（EBU）、平均に寄せる記憶の書き方（MBEC）、古い収益への max（SAIL）だった。記憶の減衰や下限を掛ける時期を、確率性への対策として測った報告は見つからなかった（§7）。

## 10. 調査の限界

- **読めなかったもの**: OpenReview はブラウザ確認のため取得できず、Optimality Tightening の採録版と査読、ERLAM の本文は読めていない。Optimality Tightening の数値は arXiv の初版（審査中の版）から取った。Value Target Lower Bounding は一次資料の所在を特定できなかった。EVA（Hansen ら 2018）と Fujita ら（IROS 2020）は他の論文の紹介だけで、本文は読んでいない。DTSIL の NeurIPS 版の補足資料も読めていない。
- **図から読んだもの**: GEM と DB-ADP のゲーム名と横軸の範囲は、図に埋め込まれた文字から読んだ。MBEC・DTSIL・SAIL（図 10）の曲線の値は読み取っていない。
- **版の違い**: SAIL は arXiv の初版と AAMAS 版の Table 1・2 の数値の一致を確かめた。初版では sticky なしの相対改善の値が本文と図 9 の説明文で違う。Go-Explore の Nature 版は掲載前の preprint を読んだ。Machado らは arXiv 版を読み、JAIR 版は読んでいない。
- **実装の読み方**: ソースは Web の取得で読んだので、行番号は一部が目安である。SAIL の実験に使った Dopamine の版は確かめていない（`dopamine-rl>=3.0.1` の指定だけ）。SAIL の train.py は学習の段だけを回す Runner を使う（master で確認）。GEM の Atari のコードと論文の式の対応、記憶の基底クラスと抽選の中身は確かめていない。EMDQN の時間上限の扱いは確かめていない。
- **観点の間の食い違い**: 手法と実績の観点では SAIL の公式実装が見つからなかったが、公開実装の観点では著者の公開ページから google-research の `sail_rl` を見つけた。本冊は後者に従う。
- **見つからなかったもの（検索語）**:
  - Ape-X・R2D2 など分散アクター系に SIL を足した報告（「self-imitation learning DQN Ape-X R2D2 lower bound return Atari hard exploration value-based」「"lower bound Q-learning" Atari DQN returns episodic」）。
  - SIL・下限 Q 学習を分布型と組み合わせた報告（「distributional reinforcement learning self-imitation lower bound return quantile IQN C51 QR-DQN」）。SAIL-IQN を除く。
  - SIL 本体を確率性の強さを変えて評価した報告（「self-imitation learning stochastic environments overestimation lucky returns bias」「"self-imitation" "sticky actions" Atari lower bound return stochastic dynamics ablation」）。
  - 記憶の減衰や下限の時期を確率性への対策として評価した報告（「episodic memory reinforcement learning max return overestimation stochastic environments」「episodic control stochastic environment optimistic bias」）。
- **被引用数**: 2026-11-24 の Semantic Scholar の値で、日により変わる。Lengyel & Dayan 2007 は回数制限で取得できなかった。

## 11. 出典一覧

### 査読つきの論文

- [Oh SIL, 2018/06] Junhyuk Oh, Yijie Guo, Satinder Singh, Honglak Lee. "Self-Imitation Learning." ICML 2018. https://arxiv.org/abs/1806.05635 — University of Michigan, Google Brain。被引用 342（2026-11-24）。
- [He OT, 2016/11] Frank S. He, Yang Liu, Alexander G. Schwing, Jian Peng. "Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening." ICLR 2017. https://arxiv.org/abs/1611.01606 — University of Illinois at Urbana-Champaign（He は Zhejiang University も）。被引用 90（2026-11-24）。読んだ版は arXiv v1。
- [Lin EMDQN, 2018/05] Zichuan Lin, Tianqi Zhao, Guangwen Yang, Lintao Zhang. "Episodic Memory Deep Q-Networks." IJCAI 2018. https://arxiv.org/abs/1805.07603 — Tsinghua University, Microsoft Research。被引用 105（2026-11-24）。
- [Tang LBQ, 2020/06] Yunhao Tang. "Self-Imitation Learning via Generalized Lower Bound Q-learning." NeurIPS 2020. https://arxiv.org/abs/2006.07442 — Columbia University。被引用 30（2026-11-24）。読んだ版は arXiv v3 と NeurIPS の本文。
- [Ferret SAIL, 2020/12] Johan Ferret, Olivier Pietquin, Matthieu Geist. "Self-Imitation Advantage Learning." AAMAS 2021. https://arxiv.org/abs/2012.11989 — Google Research Brain team（Ferret は Inria Lille も）。被引用 26（2026-11-24）。
- [Hu GEM, 2021/03] Hao Hu, Jianing Ye, Guangxiang Zhu, Zhizhou Ren, Chongjie Zhang. "Generalizable Episodic Memory for Deep Reinforcement Learning." ICML 2021. https://arxiv.org/abs/2103.06469 — Tsinghua University（IIIS）, Peking University, University of Illinois at Urbana-Champaign。被引用 50（2026-11-24）。
- [Ren DB-ADP, 2021/09] Zhizhou Ren, Guangxiang Zhu, Hao Hu, Beining Han, Jianglun Chen, Chongjie Zhang. "On the Estimation Bias in Double Q-Learning." NeurIPS 2021. https://arxiv.org/abs/2109.14419 — University of Illinois at Urbana-Champaign, Tsinghua University（IIIS）。被引用 35（2026-11-24）。
- [Zhu ERLAM, 2020/04] Guangxiang Zhu, Zichuan Lin, Guangwen Yang, Chongjie Zhang. "Episodic Reinforcement Learning with Associative Memory." ICLR 2020. https://openreview.net/forum?id=HkxjqxBYDB — 所属は未確認。被引用 61（2026-11-24）。要旨だけ読んだ。
- [Pritzel NEC, 2017/03] Alexander Pritzel ほか. "Neural Episodic Control." ICML 2017. https://arxiv.org/abs/1703.01988 — DeepMind。被引用 479（2026-11-24）。
- [Le MBEC, 2021/11] Hung Le, Thommen Karimpanal George, Majid Abdolshah, Truyen Tran, Svetha Venkatesh. "Model-Based Episodic Memory Induces Dynamic Hybrid Controls." NeurIPS 2021. https://arxiv.org/abs/2111.02104 — Deakin University。被引用 27（2026-11-24）。
- [Gangwani SIDP, 2018/05] Tanmay Gangwani, Qiang Liu, Jian Peng. "Learning Self-Imitating Diverse Policies." ICLR 2019. https://arxiv.org/abs/1805.10309 — UIUC, UT Austin。被引用 77（2026-11-24）。
- [Machado ALE, 2017/09] Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, Michael Bowling. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." Journal of Artificial Intelligence Research 61. https://arxiv.org/abs/1709.06009 — University of Alberta ほか。被引用 656（2026-11-24）。読んだ版は arXiv v2。
- [Ecoffet FRTE, 2020/04] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. "First return, then explore." Nature 590. https://arxiv.org/abs/2004.12919 — Uber AI, OpenAI。被引用 509（2026-11-24）。読んだ版は掲載前の preprint。
- [Guo DTSIL NeurIPS, 2020/12] Yijie Guo, Jongwook Choi, Marcin Moczulski, Shengyu Feng, Samy Bengio, Mohammad Norouzi, Honglak Lee. "Memory Based Trajectory-conditioned Policies for Learning from Sparse Rewards." NeurIPS 2020. https://papers.nips.cc/paper/2020/hash/2df45244f09369e16ea3f9117ca45157-Abstract.html — University of Michigan, Google Brain。被引用は NeurIPS 版の記録が未確認。
- [Lee EBU, 2018/05] Su Young Lee, Sungik Choi, Sae-Young Chung. "Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update." NeurIPS 2019. https://arxiv.org/abs/1805.12375 — KAIST。被引用 87（2026-11-24）。
- [Lengyel & Dayan, 2007/12] Máté Lengyel, Peter Dayan. "Hippocampal Contributions to Control: The Third Way." NIPS 2007. https://proceedings.neurips.cc/paper/2007/hash/1f4477bad7af3616c1f933a02bfabe4e-Abstract.html — University of Cambridge, Gatsby Computational Neuroscience Unit（UCL）。被引用は未取得。

### 査読なしのプレプリント

- [Blundell MFEC, 2016/06] Charles Blundell, Benigno Uria, Alexander Pritzel ほか. "Model-Free Episodic Control." arXiv. https://arxiv.org/abs/1606.04460 — Google DeepMind。被引用 352（2026-11-24）。
- [Pshikhachev SILfD, 2022/03] Georgiy Pshikhachev, Dmitry Ivanov, Vladimir Egorov, Aleksei Shpilman. "Self-Imitation Learning from Demonstrations." arXiv. https://arxiv.org/abs/2203.10905 — JetBrains Research。被引用 10（2026-11-24）。
- [Salimans Backward, 2018/12] Tim Salimans, Richard Chen. "Learning Montezuma's Revenge from a Single Demonstration." arXiv. https://arxiv.org/abs/1812.03381 — OpenAI ほか。被引用 174（2026-11-24）。
- [Ecoffet Go-Explore, 2019/01] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. "Go-Explore: a New Approach for Hard-Exploration Problems." arXiv. https://arxiv.org/abs/1901.10995 — Uber AI Labs。被引用 469（2026-11-24）。
- [Guo DTSIL v1, 2019/07] Yijie Guo, Jongwook Choi, Marcin Moczulski, Samy Bengio, Mohammad Norouzi, Honglak Lee. "Efficient Exploration with Self-Imitation Learning via Trajectory-Conditioned Policy." arXiv. https://arxiv.org/abs/1907.10247v1 — University of Michigan, Google Brain。被引用 10（2026-11-24、この記録の値）。

### 著者の公式実装

- [SIL 実装] junhyukoh/self-imitation-learning. https://github.com/junhyukoh/self-imitation-learning — コミット 13eb8a79e9585f92761e0a4670bd76c2e0a7bf05（2020-04-18）。
- [nstep-sil 実装] robintyh1/nstep-sil. https://github.com/robintyh1/nstep-sil — コミット 57b8092eec1477eaaf2ba4c8856c7303d694e980（2021-02-13）。
- [sail_rl 実装] google-research/google-research の `sail_rl`. https://github.com/google-research/google-research/tree/master/sail_rl — `sail_rl` に触れた最新のコミット 5b09c22d73a9d35eb6c5d2a99b95677a45053466（2026-02-12）。URL の出どころは著者の公開ページ https://ferretj.github.io 。
- [OT 実装] ShibiHe/Q-Optimality-Tightening. https://github.com/ShibiHe/Q-Optimality-Tightening — コミット 1cce8cf94e3fe026a8d18f7e2f2ed8e709392f08（2017-04-26）。
- [emdqn 実装] LinZichuan/emdqn. https://github.com/LinZichuan/emdqn — コミット 707dc88eb920e8696379183a15bf358c90590519（2018-09-05）。
- [GEM 実装] MouseHu/GEM. https://github.com/MouseHu/GEM — コミット a9f0c68d073b18d65dd29537536a3055cd99b221（2021-07-01）。
- [Dopamine 実装] google/dopamine. https://github.com/google/dopamine — master（2026-11-24 取得、コミットは固定していない）。`dopamine/discrete_domains/run_experiment.py`。

### 著者側のブログ・第三者・文書

- [Uber Go-Explore blog, 2018/11] Joel Lehman, Jeff Clune, Kenneth O. Stanley. "Montezuma's Revenge Solved by Go-Explore, a New Algorithm for Hard-Exploration Problems (Sets Records on Pitfall, Too)." Uber のブログ. https://www.uber.com/en-GB/blog/go-explore/ — Uber AI Labs。著者側の記事（査読なし）。2018-11-26 投稿、12-03 と 12-12 に追記。
- [Irpan, 2018/11] Alex Irpan. "Quick Opinions on Go-Explore." https://alexirpan.com/2018/11/27/go-explore.html — 第三者のブログ。
- [Zhao VTLB, 2021/09] Le Zhao, Wei Xu. "Faster Reinforcement Learning with Value Target Lower Bounding." 一次資料は未特定。第三者の索引の要旨だけ。所属・被引用は未確認。
