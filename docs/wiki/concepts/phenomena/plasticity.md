# 可塑性指標と成績

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [現象](../../index.md#現象)

## 今の結論

同じ勾配 step 付近での摩耗は replay ratio が違っても似ていたが、その間に入る新規データ量と成績の推移は異なった。崩壊時に重みが膨らんだのは feature 側だった。
Breakout では SiLU や LN512 で可塑性の指標が改善しても、RR8 の短期比較で成績の改善は確認できなかった。

## 分かっていること

### anet の知見

- 同勾配step付近の損傷はRR間で似たが、付随する新規データ量は異なった。dead ratioは設定内の転換点と、設定間の水準比較で意味が違う。（出典: [2026-08-28_plasticity.md](../../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ。Run: run_20260828-062103_plasticity_rr8_breakout）。
- 設定修正後も同勾配座標の摩耗と、RRごとの崩壊・天井・上昇の違いが見られた。（出典: [2026-08-29_bundle-fix-rebaseline.md](../../../experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md)、現時点のまとめ。Run: run_20260829-190614_a5_breakout_rr4）。
- SiLUやLN512で機序指標を改善できても、RR8短期screeningでscore改善は得られなかった。q_gapの定常値が動かないことから因果を導く主張は撤回済みである。（出典: [2026-08-30_protection-screening.md](../../../experiments/default-dqn/atari/2026-08-30_protection-screening.md)、現時点のまとめ1〜5）。
- 全層spectral_capは重みの実効ノルムを抑えて発散しなかったが、RR4の成績は動かなかった。（出典: [2026-09-08_replay-ratio-mechanism.md](../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ6・7。Run: run_20260908-073326_rr4_capall_munch）。
- Breakout の RR8・5M では、崩壊（eval ピーク後の低下、q_gap −57%）と同期する srank の低下や dormant の増加は無かった。重みを feature 側と readout 側に分けて測ると、膨らんだのは feature 側（RR8・5M で 1.47 倍）で readout 側は 1.10 倍、成長比は全腕で 4.2〜5.6 倍だった（出典: [0063_plasticity_weight_norm_10rfd.md](../../../memo/done/0063_plasticity_weight_norm_10rfd.md)、Context（背景・目的）、初回検証 Run run_20260828-000954_plasticity_rr8_breakout；[2026-08-28_plasticity.md](../../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ3・探索ブロック01）。
- サンプル効率優先の Atari-5 では、readout の重みノルムが 5 本すべてで増え続けた（5-10M → 45-50M で battle_zone 47 → 80、qbert 70 → 140 など）。実時間優先は 34〜41 でほぼ横ばいだった（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック02「内部指標」）。
- MontezumaのTaigaら寄せの構成＋RND（seed 1、30M、learn step約94万）で、probeのdormant率はβ 0.01・Munchausen無効のRunでlearn step 15万〜30万の0.32から90万〜94万の0.54へ上がり続け、β 0.1・Munchausen有効のRunでは同じ範囲で0.19〜0.34だった。前者はGreedyで2500点に届き、後者は届かなかった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ7・探索ブロック07。Run: run_20261110-003002_mz_tg_nmdd_b001_s1、run_20261108-232006_mz_tg_rnd_s1）。
- 同じ構成でRNDをTaigaらの式（β 5e-05）にし、Munchausen無効・Double DQN有効にしたRunは、learn step 2万ごとの窓でprobeのdormant率が0.73から0.96、dead率が0.61から0.94、`04_srank`が211から53、`03_feature_norm`が0.35から0.06になり、Qの最大の平均は0.005〜0.012で平らだった（learn step約8万、2.53Mで止めた）（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック08。Run: run_20261109-221347_mz_tg_taiga_nmdd_s1）。
- 同じ構成のRFD0123の既定（β 0.01・Munchausen無効）で、価値を外的と内的の2つのHeadに分けたRun（seed 1）は、4M窓のprobeのdormant率が16M以降0.46〜0.50、dead率が0.24〜0.25だった。分けない2本（seed 1・2）は同じ窓でdormant率0.30〜0.35、dead率0.11〜0.15。分けたRunはGreedyで30Mまで500点を超えず、分けない2本は26.2M・29.2Mに2500点以上に届いた（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ12・15・23・探索ブロック18。Run: run_20261115-055326_mz_tg_r121_vd_s1、run_20261112-043034_mz_tg_r123_b001_s1、run_20261112-222903_mz_tg_r123_b001_s2）。
- 同じ分解のRunの内的の割引率を0.99に分けてgame overで切らず、外的と内的の損失を別々のAdamで更新したRun（seed 1、probeは5,000 learn stepごと。分解だけのRunは500）は、4M窓のprobeのdormant率が0.59〜0.70、dead率が0.38〜0.52、srankが730〜982だった。分解だけのRunは同じ窓で0.30〜0.50・0.12〜0.25・990〜1,405、分解なしのseed 1はdormant率0.18〜0.35・dead率0.06〜0.15。このRunはGreedyで15.2Mに2500点に届き、20.2〜29.2Mの`53_evalg`で146/160が2500点以上だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ25・28・探索ブロック18・20。Run: run_20261116-005336_mz_tg_g99ne_split_s1）。
- 同じ割引率の分離のまま外的と内的の損失を合わせて1つのoptimizerで更新したRunと、2つの形をseed 2で流したRunを足すと、20〜26Mと26〜30Mの平均で、probeのdormant率は別々の形が0.64・0.62（seed 1）と0.55・0.56（seed 2）、合わせた形が0.48・0.50と0.44・0.45、dead率は別々の形が0.40・0.38と0.33・0.34、合わせた形が0.27・0.27と0.21・0.23だった。20〜30MのGreedy（320ゲーム）の平均は、合わせた形が3,160.3・2,790.9、別々の形が2,394.4・2,265.3だった。seed 2の2本は実行体のbuildがseed 1の2本と違う。合わせた形のreplayの容量を2倍にしたRunは、dormant率0.41・0.48、dead率0.19・0.22で、Greedyの平均は1,919.4だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ29・33・探索ブロック20・21。Run: run_20261116-173557_mz_tg_r121_g99ne_s1、run_20261117-142732_mz_tg_r121_g99ne_s2、run_20261116-005336_mz_tg_g99ne_split_s1、run_20261118-000153_mz_tg_g99ne_split_s2、run_20261118-111145_mz_tg_g99ne_cap4m_s1）。
- qbert では勾配のクリップ閾値を上げた合算で眠ったユニットが最も少なく、Greedy の得点も全変種中で最も低かった → 詳しくは [行動の時間尺度](../methods/temporal_abstraction.md)（出典: [qbert の学習比較](../../../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)、探索ブロック 01・03）。
- 実時間優先 seed 2 の Atari-5 は 10〜20M から 90〜100M に action_churn が全ゲームで 38〜69% 減ったが、q_gap が縮んだのは battle_zone だけ。特徴の重みのノルムは全ゲームで 37〜88% 増えた。終盤の srank_ratio は 0.863〜0.921、dormant_ratio は 0.009〜0.117 だった（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）。
- Breakout の可塑性測定で srank の閾値を3通り、probe batch を512と4096に変えても、崩壊時の変化は −3〜6% だった。今回の損傷を拾う感度は低かった（出典: [Atari / 可塑性メトリクスの較正と保護機構の初回評価](../../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、探索ブロック03。Run: `run_20260829-094914_ab_s2_probe_b4096_5m` ほか）。
- RR8・5M の zero-init residual 試験は `res.init2.mode` が黙殺され、2本とも対照と同一構成だった。zero-init の効果は未検証（出典: [Atari / 可塑性の保護機構 screening（新バンドル・RR8 アッセイ）](../../../experiments/default-dqn/atari/2026-08-30_protection-screening.md)、探索ブロック04。Run: `run_20260830-065617_b_init_he`、`run_20260830-075903_b_init_he`）。
- Breakout の RR4 で Linear512 直後に LayerNorm を入れると45–50M の評価平均は378.4、無保護対照は295.7だった。RR1 の25–50M 評価平均の差は対照のブレ幅に収まった（出典: [Atari Breakout: RR4 天井と LayerNorm](../../../experiments/default-dqn/atari/2026-08-31_rr4-ceiling-ln512.md)、探索ブロック01・02。Run: `run_20260831-050214_i_rr4_ln512_breakout`、`run_20260831-124349_j_rr1_ln512_breakout`）。
- Breakout の τ 整合と LayerNorm を組み合わせた条件では、表現指標を保ちながら15M step 以降の壁突破率が LN512 単独より低下した（出典: [RFD0066](../../../memo/done/0066_policy_churn_metrics_10rfd.md)、Context / Problem Statement）。

### 外の事例

- Transformer の価値関数でモデルを大きくしたときに attention の分布が集中し、性能低下と対応した。 → 詳しくは[Action queryとattentionの証拠](../methods/action_query_attention.md)。（出典: [TQL: Scaling Q-Functions with Transformers by Preventing Attention Collapse](../../../../reports/dong2026_tql_paper_2026-10-05.md)、5. 実験（Experiments）・5.2 価値関数で Transformer の効果的なスケールを妨げるもの（What Prevents Transformers from Scaling Effectively for Value Functions?））。
- 正規化で出力が重みの倍率に依らなくなった重み（スケール不変な重み）では、勾配が重みのノルムに反比例し、実効学習率は SGD とモメンタムで η/‖w‖₂²、RMSProp と Adam で η/‖w‖₂ になる。Adam では、重みを α 倍した系が、学習率を η/α、ε を αε にした系と同じ更新になる（実験は BN 付きの 4 層の CNN・CIFAR-10）。Lyle らの実効学習率の定義も同じ形で、Heo らは単位球の上での実効的な歩幅が重みのノルムに反比例すると導く（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.1）。
- 正規化の種類で、重みがスケール不変になる単位が違う。BN と Weight Standardization ではニューロンごと、LayerNorm では層全体である（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.1）。
- bias の無い線形・ReLU のネットの最後に正規化を置くと、各層の重みの倍率がほかの層の勾配の大きさに効かなくなり、各層の実効学習率はその層の倍率で決まる（Lyle らの付録の導出）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.1）。
- 正規化と組むと L2 正則化（weight decay）には正則化の効果が無く、重みの大きさを通じて実効学習率に効く。正則化が無ければ重みは際限なく大きくなり、実効学習率は 0 へ向かう。RMSProp と Adam では重みの大きさは η/λ の 3 乗根で決まり、交差検証の最適な学習率は SGD とモメンタムで λ に反比例、RMSProp と Adam で λ の平方根に反比例した（BN 付きの CNN・CIFAR-10）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.2）。
- BN 付きのネットを SGD か Adam で学ぶとき、weight decay の汎化の効果は主に実効学習率の引き上げを通じたものだった。weight decay なしのネットの各層のノルムを、weight decay ありのネットの同じ epoch の値へ毎 epoch 合わせると、差のほとんどが消えた（ResNet32・CIFAR-100。この節は SGD を中心に書かれ、Adam でも同様だったと添える）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.2）。
- AdamW では、スケール不変な重みのノルムの 2 乗の期待値は、初期値の項が毎回縮み、更新の項が足される形で、平衡のノルムは √(ηC/2λ) にほぼ等しい（C は重みのベクトルの次元）。平衡の 1 回の更新の角度は η・λ・β1 で決まり、勾配の大きさにも初期値にも依らない（勾配が雑音で占められるランダムウォークの近似）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.2）。
- Adam に L2 正則化を足した形では、スケール不変な重みの平衡が勾配の大きさに依り、層やニューロンごとに回り方がそろわない。AdamW の平衡は勾配の大きさに依らない（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.2）。
- weight decay が無いと、スケール不変な重みのノルムは勾配降下の 1 回ごとに 2 乗が更新の 2 乗だけ増え、実効的な歩幅は小さくなっていく。モメンタムではさらに速く増える。ImageNet の ResNet-18 を weight decay 0 の AdamW（Adam と同じ）で学ぶと、ノルムは最初の epoch の 1.70 から最後の 10.41、半径方向の成分を除く AdamP では 1.29 から 3.02 で、精度は 68.38 対 70.55 だった。音楽のタグ付けの HCNN では 7.35 から 11.56 対 7.30 から 7.62 だった（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.3）。
- weight normalization では、ノルムが増えられることで学習率に頑健になり、学習率が大きすぎてもノルムが速く増えて適当な実効学習率に落ち着く。ノルムの単調な増加は Adam では厳密には成り立たないが、定性的には同じだった（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.3）。
- スケール不変な重みでは、初期のノルムが平衡と比べてどうかで、学習の初めの過渡期の回り方が決まる → 詳しくは [重みの初期値](../methods/weight_init.md)（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.4）。
- 1 回の更新の回り方を直接決める最適化の変種（RV）では、学習の初めの過渡期が無くなり、学習率の warmup の必要が減った（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.4）。
- van Laarhoven は、weight decay が無いときの実効学習率の下がり方は制御しにくく、学習の初めの数 step に大きく左右され、結果の再現を難しくすると書く（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.4）。
- 各非線形の前に LayerNorm を入れた Rainbow（Atari 57 ゲーム、200M フレーム）で、各層の重みを周期的に初期のノルムへ射影して実効学習率を一定にすると、射影しないものより悪いことが多かった。学習率を 6.25e-5 から 1e-6 へ線形に下げると差が埋まり、57 ゲーム・4 seed の集計では、この減衰を付けた射影が、素の Rainbow、LayerNorm だけ、減衰の無い射影を上回った（集計の値は図だけ。付録は Rainbow の既定の学習率を 0.000625 と書き、本文と桁が違う）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.6・§8）。
- 初期値の大きさの差が消える更新の回数と、LayerNorm の付いた構成で初期値を小さくしても学習が速くならない理由を論じた文献は、調査では見つからなかった → 詳しくは [重みの初期値](../methods/weight_init.md)（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§3・§4）。
- 多 agent の QMIX で、RND の新しさと活性なユニットの割合を合図に、Q 側の mixer の Network へ Plasticity Injection を当てる例がある。RND の predictor は初期化しない。 → 詳しくは [内的報酬による探索の比較](../methods/intrinsic_novelty.md)（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§3.3）。
- モデルベースの RL で、agent ではなく world model をリセットして成績が上がった例がある。 → 詳しくは [Model-based Atariの評価](../methods/model_based_evidence.md)（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§3.4）。
- Montezuma's Revenge の RND の agent に shrink+perturb と regenerative regularization を足すと報酬が上がったという報告がある。 → 詳しくは [内的報酬による探索の比較](../methods/intrinsic_novelty.md)（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§3.5）。
- 文献の比較条件: Lyle ら（ICLR 2022）は、DDQN などの DQN 系を Montezuma's Revenge（疎な報酬）と Pong（密な報酬）で学ばせ、特徴のランクが一貫して下がるのは疎な報酬の環境だけで、agent はこの崩れを抜けたあとに初めて課題を解いたと報告する（各組 3 seed）。方策の下で報酬が 0 なら特徴は全状態で 0 のベクトルへ収束するという系も示す。崩れるのは価値の Network の表現で、内的報酬は扱っていない（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.1）。
- 文献の比較条件: Lyle らは、ランダムな目標へ当てはめる力が、密な報酬のゲームを含む多くの環境の checkpoint で少し落ちていたとも報告する（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.1）。
- 文献の効いた条件: Lyle らは、InFeR（補助の Head の出力を初期の値へ引き戻す正則化）で Rainbow の Montezuma's Revenge の成績が大きく上がったと書く。値は図だけにある（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.1）。
- 多 agent の疎な報酬の課題で、RND が成功する経験を見つけても、可塑性を失った Network はそれを方策に変えられないという主張がある。 → 詳しくは [内的報酬による探索の比較](../methods/intrinsic_novelty.md)（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.2）。
- 文献の比較条件: DrM（ICLR 2024、DrQ-v2 が基盤、DeepMind Control は 30M フレーム・4 seed）は、学習の早い時期に agent が動かなくなり、その時期に方策の Network の眠ったユニットの割合が高いと報告する。割合が下がると意味のある動きが始まり、その割合で探索のノイズの大きさと、約 2×10^5 フレームごとのやわらかいリセットの強さを決める。崩れるのは探索の行動で、内的報酬ではない（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.3）。
- 文献の比較条件: Klein らの可塑性のサーベイは、可塑性の喪失が性能の頭打ちや過大評価とともに探索の不足に寄与すると要旨に書き、リセットに可塑性と別の探索の利点がありうること、探索の一部が勾配の学習による頻繁な行動の変化から来うることを挙げる。ここでの探索は方策の行動の変化で、内的報酬は扱っていない（出典: [rnd_dense_reward_reset_collapse_survey_2026-11-27.md](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md)、§4.4）。

## 横断的知見

- 共通点: Breakout の RR8 短期試験では、旧設定の同一構成5本中4本、設定変更後の2本中2本で、序盤の評価ピークから成績が落ちた。2〜3Mの評価点は旧設定の比較対象が165.8、新設定の2本が323.7と239.7だった。条件の違い: 旧設定には崩れない複製が1本ある。新設定では複数の変更が重なり、観測したピークの差を個別の変更には帰せない（出典: [可塑性の較正](../../../experiments/default-dqn/atari/2026-08-28_plasticity.md)、現時点のまとめ・探索ブロック01。[設定修正後の再測定](../../../experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md)、探索ブロック01）。
- 対立点: Breakout の Linear512 後の LayerNorm は、RR8・5M の試験では可塑性指標を動かしても終端成績を改善しなかった。RR4・50M では終盤の評価平均が無保護対照を上回り、同じ処置でも replay ratio と評価時期によって結果が違った。条件の違い: RR8・5M の終端と RR4・50M の終盤窓を比較している（出典: [短期試験](../../../experiments/default-dqn/atari/2026-08-30_protection-screening.md)、探索ブロック02・03、`run_20260830-114547_f_ln512` ほか。[50M 試験](../../../experiments/default-dqn/atari/2026-08-31_rr4-ceiling-ln512.md)、探索ブロック01、`run_20260831-050214_i_rr4_ln512_breakout`）。

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

- 2026-09-08: 全層でノルムを抑えると発散するため両立点が無い → spectral_capの当該構成は抑制と非発散を両立したがscoreは改善しなかった（出典: [RR機序](../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md)、現時点のまとめ6・7）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。
- [可塑性メトリクスは実 forward の branch capture と学習経路外の部分 forward の 2 チャネルで測る](../../../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)。
- [Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)。
