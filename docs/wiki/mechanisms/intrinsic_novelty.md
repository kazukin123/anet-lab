# 内的報酬による探索の比較

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

NGU 型のボーナスと E3B の比較結果は、学習器・環境・合成方法に依存する。MiniHack での結果を Atari の優劣へ持ち込めず、anet の BTR を基にした構成での効果はまだ測っていない。KFM の命中にボーナスが出るかも、逆動力学の特徴がバーの差を保つかに依存する。GridMazeのCPU・256 lane・torch thread 1・Releaseでは、既定RNDの時間はなしの1.339倍だった。Montezumaのサンプル効率優先＋RNDの30M（1本、RNDなしの対照なし）では、学習側で部屋7に入るゲームが現れた24〜25Mにbonusの平均が前後の1Mより高く、同じ24〜26MにlossとQの最大の平均が上がった。LunarLanderの3 seed×2Mでは、既定RNDを混ぜても評価は基準と同じ帯だったが、bonusは採点5〜10万件で消えた。predictorを採点50,000件ごとに初期化するとbonusは周期信号として続いたが、終盤の評価は同じ帯のままで、1 seedが途中で崩れ、Qの最大の平均と勾配ノルムは全seedで基準より大きかった。MontezumaのRND（β 0.01・Munchausen無効）では、2500点のあと外的報酬の残らない局面でQが18行動ともほぼ同じ値になり、Greedyが動かなかった。Taigaらの式のRNDは、Munchausenの有無に依らずMontezumaの鍵より先へ進まなかった。既定（`whiten`）のRNDでは、誤差の統計の二乗偏差の88〜99%を最初の200k exp stepの誤差が占め、KFMでは内的報酬が報酬の数%（Munchausenの項の約1/100）にとどまり、Montezumaでは誤差の標準偏差の膨らみが9〜17倍から3〜4倍へ薄まった。文献では、外的と内的の価値を別々に学ぶ形（Agent57・Burdaら）が内的報酬の重みへの頑健さと探索した部屋の数を改善した。

## 分かっていること

- 文献の比較条件: NGU 原論文は R2D2 を基にした再帰型の学習器を使う。RLeXplore の共通学習器上のモジュール比較とは構成と予算が違い、公開スコアから同じ条件での順位は付けられない（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.8）。
- 文献の比較条件: Henaff ら（2023）の NGU 型ボーナスとの比較は主に MiniHack で、Montezuma's Revenge では E3B 単独と RND・併用でも結果が異なる。E3B を NGU の無条件な上位互換とする根拠にはならない（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.8）。
- 未検証の仮説: KFM のバーの縮みを逆動力学の特徴が保存すれば、報酬 0 の命中にも新しさの手応えを与えられる。ただし自機の姿勢だけで行動を予測できる可能性があり、制御可能な変化が必ず特徴に残る保証は無い（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.5）。
- 調査の範囲: anet の BTR を基にした構成での NGU／E3B の効果は未測定であり、今回の調査では直接比較に使える BTR＋NGU 全体の実績も確認できていない（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§3「現状」・§2.8「NGU・E3B・BTR の関係と、比較結果の適用範囲」）。
- 切り出し: RND による一生の新しさは lane 別の記憶を持たず、facade（`DefaultDQNAgent::UpdateFromBatch`）で次観測を採点して報酬へ畳めば Replay・Actor の契約を変えずに腕にできる。KFM では対照腕で、判断に効くのは GridMaze の到達率と費用・統計の確認（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§1.1・§4 D1・D2・§7）。
- 切り出し: 一生の新しさの口は「倍率 α を返す採点・補助学習・診断・保存」の 4 操作で、RND の本体は DQN から見えない非公開の実装に閉じる暫定の設計（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§5.1・§5.7）。
- 比較対象の実装: Taiga ら（ICLR 2020）の Rainbow＋RND は、行動した観測 x_t を agent step ごとに 1 件で採点・学習し、誤差の z を分散に相当する値で割って 0.00005 倍し、Runner が [−1, 1] に clip した外的報酬に足す。合計は clip せず、負の値もそのまま入る。観測の統計は正規化後の値で更新する。Burda ら・anet が採点するのは次観測 x_{t+1}（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1・§8.1 の 6）。
- RND の尺度の条件: target と predictor の出力の尺度が揃っていれば、初期値の種類（he／xavier と直交）で新しさの判別は大きく変わらず、target だけ小さいと predictor が誤差を下げられず判別が落ちる。PyTorch の写しで Montezuma の画面を 6 万 step 学習した判断材料で、受入の基準ではない（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1）。
- 設定の落とし穴: anet の層カタログ `Conv2d_S`・`Conv2d_M` は葉の階層が読み込みと合わず、2026-11-06〜07 の Atari の RND の Run は既定の 128@3/1 ×2（1 Network 約 1 億パラメータ）で動いていた。写しの比較では判別は落ちず、実証された害は速さと checkpoint の大きさ（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1）。
- GridMaze では RND の新しさが最初の数千 step で尽きる: 観測が正規化した (x, y) の 2 値で盤面は固定（`seed_mode = process`）なので入力は 25 通り以下。ε = 1 から始まる 256 lane が全部を踏み、predictor が覚え切って一生の誤差平均を下回り続け、既定（`clipped`・`whiten`）の bonus は 0 になる（2026-11-07 の `run_20261107-212611`: 1,551 点中 9 点以外 0、predictor の損失 3e-9）。手で通っていないセルへ誘導したときの失敗は未踏ではなく off-path の Q の粗さ [PRD116 §6.4・§7]

- GridMazeでのRNDの費用: CPU・256 lane・torch thread 1・Release、同じ165,632経験のABBA3巡では、RNDなし/ありの実時間中央値が23.261秒/31.148秒で1.339倍。別RunのTracyではRND区間のforwardが56.47%、backwardとoptimizerが35.10%を占めた。観測は(x, y)の2値であり、Atariの画像入力やGPUへ数字を移さない（出典: [GridMazeのRND性能測定](../../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md)、§通常比較の結果・§Tracyによる内訳・§確認できた範囲と残件）。
- Montezumaのサンプル効率優先＋RND（β 0.1、RNDのNNは公開実装の形・出力512、30M、seed 1、RNDなしの対照なし）で、bonusの平均は24〜25Mに0.0144で、1M前（0.0062）と1M後（0.0033）より高く、26〜27Mに0.0013だった。学習側で部屋7に入ったゲームは24〜26Mに206本（22〜24Mは0本）、500点のゲームは24〜26Mに87本・26〜28Mに106本だった。同じ24〜26Mにlossは2.4〜3.0（20〜24Mは0.05〜0.11）、Qの最大の平均は14.6〜23.7（同3.7〜4.5）だった（出典: [Montezuma's RevengeのRND（サンプル効率優先・30M）](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ3・4・探索ブロック02・03。Run: run_20261108-003056_mz_se_rnd_k32_s1）。
- LunarLanderのIQN＋Munchausen＋TempoRL（分離×合算）に既定のRND（β 0.1、`clipped`・`whiten`、warm-up 12,800件）を混ぜた3 seed×2Mでは、UQE fullの1.5〜2Mが210.7／225.5／181.1で、RNDなしの基準9本の188.6〜225.7と同じ帯だった。bonusは採点5〜10万件で消え、shareが0.01を超えた最後の点は93,696／40,960／16,128 exp stepだった。predictorの誤差がwhitenの一生平均を常に下回り、bonusが0になる。bonusが続く状態での害は測れていない（出典: [RNDの害の確認とpredictorの周期リセット](../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、現時点のまとめ1・2・bonusの推移。Run: run_20261108-061056_ll_rnd_s1 ほか同記録の3本）。
- 同じLunarLanderで、predictorとwhitenの誤差統計を採点50,000件ごとに初期化してwarm-upをやり直す暫定の機構を足した3 seed×2Mでは、bonusは2Mの間続いた（shareのRun全体平均0.033〜0.038）。周期の中はwarm-up直後の約0.1から周期末の0.013〜0.020へ下がる鋸歯で、周期ごとの平均は後半ほど大きかった。UQE fullの1.5〜2Mは183.7／196.8／213.5で同じ帯だった。seed 2は0.78〜0.9Mに崩れて−1,000未満の局が8つ出て、同じ区間にQの最大の平均が1.35から3.5へ上がった。1.5〜2Mでは全seedでQの最大の平均（1.85〜2.17）と勾配ノルム（0.76〜0.81）が基準9本（1.06〜1.57、0.29〜0.48）より大きかった（出典: [RNDの害の確認とpredictorの周期リセット](../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、現時点のまとめ3〜6・seed 2の崩れ・学習側の指標と局の長さ。Run: run_20261108-125542_ll_rnd_rs50k_s1 ほか同記録の3本）。
- MontezumaのRND（β 0.01、Munchausen無効、Double DQN有効、seed 1）で2500点を取ったあと、Greedyは部屋1の開始の足場でUPを押し続けて27,000 stepの打ち切りまで動かず、Qは18行動とも0.055〜0.059（差0.004で、τの標準偏差0.009〜0.010より小さい）、V_sは0.056〜0.063だった。29.2MのGreedyは16ゲームすべてが2500点・打ち切りで、学習側は16M以降、2M窓ごとの打ち切りが終わったゲームの4〜11%で、残りはgame overで終わった。13M以降の1 stepの内的報酬は2〜4e-4（正の値13〜19%）（出典: [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§2.1、[Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ4・探索ブロック06。Run: run_20261110-003002_mz_tg_nmdd_b001_s1）。
- 同じTaigaら寄せの構成で、β 0.1・Munchausen有効のRunの1 stepの内的報酬は4M窓の平均で1.1e-03〜9.6e-03、Munchausenの項（`36_agent_munchausen/04_bonus_mean`）は−0.11〜−0.32だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック06。Run: run_20261108-232006_mz_tg_rnd_s1）。
- 効かなかった条件: Taigaらの式のRND（β 5e-05、`unclipped`・`variance_feedback`、行動した観測を採点）をTaigaら寄せの構成に載せた2本は、鍵より先へ進まなかった。Munchausen有効では2〜3Mに全laneが死なずに打ち切りまで続く形になり（3M台に終わったゲーム49本はすべて打ち切り）、Qの最大の平均は0.11から0.80へ上がった。Munchausen無効・Double DQN有効では、learn step 6万前後でprobeのdead率0.94・`04_srank` 53になり、Qの最大の平均は0.005〜0.012で平らだった。どちらも内的報酬の平均は−3e-06〜−1.5e-05だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ5・探索ブロック08。Run: run_20261109-174656_mz_tg_rnd_taiga_s1、run_20261109-221347_mz_tg_taiga_nmdd_s1）。
- 文献の効いた条件: Agent57はQを外的と内的の別々のNN（同じ形、どちらも方策の番号を入力）の和にし、内的報酬の重みの広い範囲で学習が安定した。外すと成績が20%以上落ちた（出典: [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§2.2）。
- 文献の効いた条件: Burdaら（RND）は内的の収益をgame overで切らず、価値のHeadを2つにした。内的を切らないと探索した部屋の数が増え、外的の割引率0.999・内的0.99で成績が大きく上がり、内的を0.999にすると下がった。両方を切る設定では、Headを2つにしても1つより良くならなかった（出典: [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§2.2）。
- 文献の比較: Taigaら（ICLR 2020）のRainbow＋RNDはMontezumaの5 seedすべてが90Mフレームまでに4,400点を超え、NoisyNetsの5 seedは200Mフレームまで2500点にとどまった（図7を目で読んだ値）（出典: [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§2.2）。
- 既定のRND（`whiten`・`clipped`、warm-up 12,800件）の誤差の平均・標準偏差はRunの最初からの累積で、warm-up中の誤差も入る。KFMの実時間優先＋RND（β 0.1、10M、7 seed）では、最初の200k exp stepの誤差が二乗偏差の総量の97〜99%を占め、10Mの標準偏差は200k以降の誤差だけで逆算した値（0.026〜0.034）の7〜19倍だった。3M以降の1 stepの内的報酬はlane平均1e-4〜8e-4、報酬の絶対値に占める割合0.014〜0.057、正のlane 4〜15%で、Munchausenの項の約1/100だった（出典: [RNDの内的報酬の尺度の暫定 PRD](../../memo/123_rnd_intrinsic_reward_scale_10prd.md)、§2.1。Run: run_20261110-123404_kfm_10m_rnd_b01_s04 ほか同PRDの7本）。
- 同じ既定のRNDのMontezuma（Taigaら寄せの構成、seed 1、`whiten`の5本）では、最初の200kが二乗偏差の総量の88〜97%を占め、標準偏差の膨らみ（実際の値÷200k以降の誤差だけの値）は0.5〜2Mで9〜17倍、4〜8Mで4〜7倍、14〜29Mで3〜4倍だった。膨らみは5本で同じ形で、内的報酬はRunの中で3〜5倍に強まっていた（出典: [RNDの内的報酬の尺度の暫定 PRD](../../memo/123_rnd_intrinsic_reward_scale_10prd.md)、§2.2。Run: run_20261108-232006_mz_tg_rnd_s1、run_20261110-003002_mz_tg_nmdd_b001_s1 ほか同PRDの5本）。

## 分かっていないこと

- 既定のRNDの序盤の大きな誤差は、predictorが未学習なためか、観測の正規化が未熟なためか。内的報酬のtagはQの学習が始まる200kより前に記録が無く、分けていない（出典: [RNDの内的報酬の尺度の暫定 PRD](../../memo/123_rnd_intrinsic_reward_scale_10prd.md)、§2.3）？
- Atari B=64のRNDなし・既定RND・Taigaの3群では、実時間中央値とTracyの内訳はどれほどか（出典: [GridMazeのRND性能測定](../../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md)、§確認できた範囲と残件）？

- 同じ anet の学習器・経験予算で、個々の訪問ベクトルとの局所的な近さを測る NGU と、まだ十分に経験していない特徴方向を評価する E3B のどちらが、KFM・GridMaze の外的成績を改善するか（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.8「NGU・E3B・BTR の関係と、比較結果の適用範囲」）？
- 探索による経験の増加が、点数を目的とする方策の未訪問状態からの到達率へ転移するか（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.7）？
- RND の target と predictor の形は、公開実装の形（predictor を深く、出力 512）と NGU の同形・出力 128 のどちらが長い学習で判別を保つか。写しの 6 万 step では同形のほうが誤差の比が大きい（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1・§8 の 2）？
- bonusが続く状態の一生の新しさを混ぜたとき、LunarLanderの成績は落ちるか。predictorの周期初期化で続くbonusは状態の新しさではないので、その代わりにならない（出典: [RNDの害の確認とpredictorの周期リセット](../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、解釈の限界と残る問い）？
- 周期初期化のseed 2の崩れは、直前のbonusの山とQの膨張から起きたのか。3 seed中1本の観察で、訪れた状態の分布も測っていない（出典: [RNDの害の確認とpredictorの周期リセット](../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、現時点のまとめ5・解釈の限界と残る問い）？

- MontezumaのRNDで、Greedyの2500点の有無を分けたのは、β（0.1と0.01）とMunchausen（有効と無効）のどちらか。2本は両方を同時に変えている（出典: [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、§1・§2.1）？

## 撤回・更新

- 2026-11-10: β 0.01のMontezumaのRunで「学習側は15M以降のゲームが5〜8千stepでgame overにより終わった」→「16M以降、2M窓ごとの打ち切りは終わったゲームの4〜11%で、残りはgame overで終わった」。ゲームの長さは集計し直しておらず、打ち切りの割合に替えた（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ4）。
- 2026-10-27: PRD の 2026-10-21 改訂記録にある「命中のたびにボーナスが付く」という説明 → 特徴がバーの差を保つかに依存する未検証の仮説へ限定した（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、冒頭の 2026-10-21 改訂記録・§2.5「新しさのボーナスが KFM に効くと見込む理由」）。

## 関連

- [KFM](../subjects/kung_fu_master.md)。
- [Montezuma's Revenge](../subjects/montezuma_revenge.md)。
- [LunarLander](../subjects/lunarlander.md)。
- [評価と比較の条件](evaluation.md)。
- [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)（機構・設計案と未決の正本）。
- [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)（924 の第一の腕。一生の新しさの口と RND の実装）。
- [外的と内的の価値の分解の暫定 PRD](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)（Q を外的と内的の 2 つの Head で学ぶ形と、924 との関係）。
- [RND の内的報酬の尺度の暫定 PRD](../../memo/123_rnd_intrinsic_reward_scale_10prd.md)（誤差の統計の窓・warm-up・平均を引かない形）。
- 仕組み: [RND の解説](../../design/212_rnd_lifelong_novelty.jp.md) §3.2〜§3.5（誤差の統計と warm-up）。
- [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）の記録](../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)。
