---
title: LunarLander
page_kind: 題材
aliases:
  - LunarLander
  - ll
  - ホバリング
description: 同条件の 3 seed では、ホバリングを抜ける時期に QR と IQN で明確な差はなかった。持続的に T180 に達したのは QR が 1 本、IQN が 3 本だった。
updated: 2026-11-27
---
# LunarLander

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [題材](../../index.md#題材)

## 今の結論

同条件の 3 seed では、ホバリングを抜ける時期に QR と IQN で明確な差はなかった。持続的に T180 に達したのは QR が 1 本、IQN が 3 本だった。
TempoRL の学習中の繰り返しと評価時のスキップは分けて比べている。学習中の設定が同じでも、評価で行動を繰り返すかによって成績が変わる。

## 分かっていること

### anet の知見

- Trainの繰り返しの時期と長さを変えた4群×3seed×2M（分離×合算、J=4）で、UQE τ0.25のsingleの報酬は、Run全体の平均が基準129.9・一様乱択127.6・50万まで121.0・warmupだけ79.8、1.5〜2Mが211.0・199.1・195.2・182.5だった。warmupだけの0.25〜0.5Mは78.1／−37.2／15.6で、基準の112.6／147.4／89.9と範囲が離れ、局の長さは673対155だった。50万までと一様乱択は基準と範囲が重なり、Greedyの1.5〜2Mは基準より高い向きだった（singleが211.1・203.1対200.7、fullが209.7・208.8対200.5）（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ1〜3・報酬・ホバリングの抜け出し。Run: `run_20261105-025614_ll_tw_base_s1` / `run_20261105-012413_ll_tw_wu_s1` / `run_20261105-022454_ll_tw_e05_s1` / `run_20261105-005313_ll_tw_uni_s1`ほか同記録の12本）。
- これまでの比較と合わせると、LunarLanderの得は序盤のTrainの繰り返しで説明できる。学習した長さ、0.5Mより後の繰り返し、補助の勾配（分離×遮断の1.5M〜は209 [176〜231]、分離×合算は209 [189〜228]）、評価時の繰り返しは、それぞれ無くても基準と同じ帯だった。同時に外した組み合わせは測っていない（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ4、[0098_temporl_skip_policy_10rfd.md](../../../memo/0098_temporl_skip_policy_10rfd.md)、§7「段 2 の結果」）。
- 基準のTrainのスキップ長は序盤にほぼ一様で、0〜0.5Mの長さ1の割合は0.20〜0.27、長さ4は0.22〜0.31、平均の長さは2.41〜2.69だった。1.5〜2Mでは長さ1が0.40〜0.43、平均の長さが2.10〜2.15になった（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ3・繰り返しと教材）。
- 基準の同じRunのfullとsingleは立ち上がりでも同じ帯で、0.25〜0.5Mはsingle 112.6／147.4／89.9、full 34.6／109.4／118.6だった（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ5）。
- warmupだけの群と繰り返しの無い過去の群の差は確定していない。Run全体の範囲45.3〜115.9は、スキップなし−6.8〜68.4、k=1対照−32.78〜50.35、STOP対照18.20〜102.58と重なる（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ6・比較に使った過去のRun）。
- 基準と同じ学習設定の過去9本のfullはRun全体が112.5〜143.4、1.5〜2Mが188.6〜228.5で、一様乱択と50万までのfullはこの中に収まった。ただし12本は独立ではなく、基準のs3は10/28のs3と312点すべて一致した（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、比較に使った過去のRun）。
- 行動固定なしのSingle行動変化窓（`t104r1_ac`）は3seedのAUCが73.04／62.65／18.08で、無効（`t104r1_off`）の64.93／4.50／−6.83を同seedで上回った。ただしRFDの範囲による「上回る」は満たさず、終盤平均は156.45対161.83だった。通常TempoRLは平均138.43で、Single全群より高かった（出典: [Singleの7群比較](../../../experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md)、現時点のまとめ・AUC・立ち上がりと終盤。Run: `run_20261029-154259_t104r1_ac_s1`ほか同記録の21本）。
- LunarLander の決定的な比較で、補助損失を変えた場合と、乱数列・target 計算を分けた場合の双方で Q 側の一致を確かめた。 → 詳しくは [行動の時間尺度](../methods/temporal_abstraction.md)（出典: [STOP×Single と無効の bit 比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-stop-single-bit-compare.md)、現時点のまとめ・解釈の限界と残る問い）。
- STOP対照（`t104r1_ctl`）も同seedの無効を3本ともAUCで改善し、平均57.25だったが、この差は無効の別の標本どうしの偶然である（上の項目）。行動変化窓mergeとの優劣はseedで揃わず、補助勾配が改善を生むことは特定できなかった。k=1対照の平均は16.55で、無効を上回る得を再現しなかった。終了窓＋splitの終盤平均は179.99だったが、AUCは−24.37〜78.15と広く、終了窓mergeと同じ帯だった（出典: [Singleの7群比較](../../../experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md)、現時点のまとめ・対照・教材と解釈の限界。Run: `run_20261029-162638_t104r1_ctl_s1` / `run_20261029-230213_t104r1_spl_s3`ほか同記録）。
- 分離×合算（J=4）と無効の2M終了checkpointを評価専用Runで各条件1,600局（400局×4 session）比べると、報酬の3seed平均はUQE τ0.25がfull 215.06・single 214.93・無効168.00、Greedyがfull 214.94・single 215.96・無効167.19だった。同じ重みのsingle−fullは−1.14〜+2.97で符号が揃わず、sessionの範囲も重なった。合算のfull・singleは3seedとも同じ番号の無効の最大より最小が高かった。Greedyの局の長さの平均はfull 158.2、single 176.6、無効314.2 stepだった。singleと無効はどちらも毎step判断するので、この差は評価時の行動の選び方ではなく学習済みの重みの差である（出典: [Singleの7群比較](../../../experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md)、現時点のまとめ7・同じ重みのfullとsingleの評価。Run: `run_20261104-233950_ll_ev_mrg_s1` / `run_20261104-234427_ll_ev_off_s1`ほか同節の6本）。
- RNDの内的報酬（β 0.1、`clipped`・`whiten`）を分離×合算の構成に混ぜた3 seed×2Mでは、UQE full（`21_eval/02_policy_reward`）のRun全体が114.5／129.2／109.9、1.5〜2Mが210.7／225.5／181.1で、RNDなしの基準9本（117.4〜144.0、188.6〜225.7）と同じ帯だった。bonusは採点5〜10万件で消えた（出典: [RNDの害の確認とpredictorの周期リセット](../../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、現時点のまとめ1・2・報酬。Run: `run_20261108-061056_ll_rnd_s1`ほか同記録の3本）。
- predictorを採点50,000件ごとに初期化する暫定の機構を足した3 seed×2Mでは、UQE fullの1.5〜2Mが183.7／196.8／213.5で同じ帯だった。seed 2は0.78〜0.9Mに崩れ、−1,000未満の局が8つ出た。完走した2MのLunarLander 89本で、0.5〜1.5Mにこれが3つ以上あるのはこのRunだけだった。局の長さのRun全体は268／329／233で、RNDだけの192〜225より長かった（出典: [RNDの害の確認とpredictorの周期リセット](../../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、現時点のまとめ4〜6・seed 2の崩れ・学習側の指標と局の長さ。Run: `run_20261108-133857_ll_rnd_rs50k_s2`ほか同記録の3本）。
- matched 3-seedのQR/IQN比較ではホバリング離脱の早さに明確な差はなく、離脱後のT180到達と終盤rewardはIQNが有利な可能性を示した。（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ3・9・10）。
- RB満杯付近の変化はcapacity半減で前へ動き、IQN固有より経験分布切替が有力だった。current N削減によるloss低下は集約尺度も変えるため、過学習改善と断定できない。（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ5〜8）。
- LunarLanderEnv の観測（既存 8 次元）には風・乱流が含まれず POMDP で、エージェントはフレームスタック（stack_count=4）の速度履歴の差分から風を推すしかない。速度差分には自機エンジンの推力も混ざる（出典: [0032_lunarlander_obs_include_action_10rfd.md](../../../memo/done/0032_lunarlander_obs_include_action_10rfd.md)、Context（背景・目的）・確定した設計判断2）。
- NoisyNetの9 Runは有限値で完走したが、3 seedの成績差はseedの幅内だった。σと行動変化率は学習中に縮み、探索置換でホバリング離脱が遅くなったとは言えなかった。（出典: [2026-10-06_noisynet.md](../../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ。Run: run_20261006-233023_ll_qr_off_s1 / run_20261006-233110_ll_qr_nzadd_s1）。
- IQNについて: learner update、eval、full distribution表示、finiteなloss/Q/gradを確認した（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- QR対IQNについて: matched 3-seed A/Bで`T50`は重なった。持続的`T180`到達はQR 1/3対IQN 3/3、終盤policy reward平均は144.3対186.0だった（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- update_warmup_steps=38,400について: QR/IQNの正式比較で一致させた値。LunarLanderの最適値とはみなさず、warmup探索は継続する（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- IQN learner N=32, M=32について: QR 32 quantilesとpair数を揃えやすく、3 seedで学習成立と終盤の健全性を確認した（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- IQN learner N=32, M=64について: `N=64, M=64`よりraw loss/gradを抑え、単一Runでは良好。ただしtarget計算量が多い（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- IQN learner N=32, M=8について: 単一Runで終盤が不安定。論文上の一般傾向だけでLunarLanderへ採用しない（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- UQE個別調整について: tau end、tail mean、epsilon floor、optimistic targetの単発probeでは一貫した改善を確認できなかった（出典: [2026-08-08_iqn.md](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。
- NoisyNet（V/A ストリームの FC と Head の射影）について: QR の matched 3-seed で、spatial UQE への上乗せも ε の置き換えも、1.5M〜2.0M の policy reward 平均（134.0 / 127.3）が OFF（145.1）の seed の幅（96.6〜199.4）に収まった。σ と probe の指標は学習とともに縮んだ（出典: [2026-10-06_noisynet.md](../../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ・対応する探索ブロック）。
- TempoRL（スキップ、J=4）について: IQN＋Munchausen の 3 seed・2M で、評価 policy 報酬の Run 全体の平均は無効 21、分離×遮断 122、分離×合算 137、包含×遮断 96。分離×合算は 3 seed とも約 1.0〜1.1M で評価の EMA が 200 に届いた（無効は 3 seed 中 1 seed だけ）。これを受けて 2026-10-26 から LunarLander の既定を有効（分離×合算）にし、無効の Run は CLI で `run.$=run.@munch` を渡す（出典: [0098_temporl_skip_policy_10rfd.md](../../../memo/0098_temporl_skip_policy_10rfd.md)、§7「段 2 の結果」・§6 の 21）。
- TempoRL 第二弾の段 2 について: 合算の基準に対して分割・`kmax`・学習側のスキップ長の ε-greedy を同じビルドの 3 seed・2M で比べ、評価 policy 報酬の Run 全体の平均は合算 135／113／125、分割 148／133／98、`kmax` 141／135／134、ε-greedy 150／104／84。3 腕とも基準以上だが上回る腕は無く、LunarLander の既定は据え置き。基準は `[greedy]` session を足した分、10/26 の同 seed の Run（131／143／138）と値が違う（出典: [2026-10-28_temporl-split-kmax-epsilon.md](../../../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md)、現時点のまとめ 1・5・探索ブロック 01・02。Run: run_20261028-131118_t103_ll_mrg_s1 ほか 11 本）。
- LunarLander は低い実験コストを利用して、IQN・UQE・Replay や learner の τ 本数の動作確認、仮説生成、回帰確認に使う。DropMerge の採否はその環境の Run で判断する（出典: [DefaultDQNAgent / LunarLander 探索記録](../../../experiments/default-dqn/lunarlander/README.md)、探索環境としての位置付け）。
- current τ の本数を減らしたときの loss や勾配の低下には、IQN loss の集約方法も効く。 → 詳しくは[τ配置とIQN比較](../methods/tau_sampling.md)。（出典: [IQN導入とQR比較の探索履歴](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ。Run: `run_20260809-010602_ll_iqn_ctau32`）。
- 合算条件の再実行を含めると通常のばらつきが広く、kmax 条件の優位とは判定できなかった。 → 詳しくは[評価と比較の条件](../measurement/evaluation.md)。（出典: [TempoRL 第二弾の段 2（分割・k_max の重み・スキップ長の ε-greedy）、LunarLander 探索履歴](../../../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md)、現時点のまとめ。Run: `run_20261028-131118_t103_ll_mrg_s1`・`run_20261028-135410_t103_ll_kmx_s1`）。
- predictor の周期リセットで bonus は持続したが、終盤評価の改善は確認できなかった。 → 詳しくは[内的報酬による探索の比較](../methods/intrinsic_novelty.md)。（出典: [RNDの害の確認とpredictorの周期リセット](../../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、現時点のまとめ。Run: `run_20261108-061056_ll_rnd_s1`・`run_20261108-125542_ll_rnd_rs50k_s1`）。
- 小型 MLP の DQN 更新では AdamW のカーネル起動負荷が測定された。 → 詳しくは [計測の負荷と比較条件](../measurement/diagnostic_cost.md)（出典: [RFD0011](../../../memo/done/0011_fused_adamw_10rfd.md)、Context（背景・目的））。
- 0.5M まで基準と同じ設定の群は、full の点が s1・s3 で 501,248 exp step の直前まで基準と全点一致し、s2 は 14,848 exp step から違った。single の点は s1 が 130,048 exp step から違い、理由は確かめていない（出典: [2026-11-05_temporl-train-repeat-timing.md](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、解釈の限界と残る問い）。
- `ll_tw_base_s3`（2026-11-05。TempoRLのTrainの繰り返しの時期と長さの比較）の実効設定では、LunarLander の `update_warmup_steps` は 8,192 で、最初の学習は 8,448 exp step だった（出典: [2026-11-05_temporl-train-repeat-timing.md](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、条件と判定方法・Runと実効config；[run_20261105-061647_ll_tw_base_s3.txt](../../../experiments/default-dqn/lunarlander/config/run_20261105-061647_ll_tw_base_s3.txt)、`DefaultDQNAgent.learner.update_warmup_steps = 8,192`）。

### 外の事例

なし

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

- Single行動変化窓（補助のlossの勾配を共有部へ返す）は、無効と別の分布か（出典: [STOP×Singleと無効のbit比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-stop-single-bit-compare.md)、解釈の限界と残る問い）。
- 無効の局が長いのはなぜか（出典: [Singleの7群比較](../../../experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md)、同じ重みのfullとsingleの評価の解釈の限界）。
- Trainの繰り返しが効く経路は、訪れる状態が広がることか、TDの学習がしやすくなることか。繰り返す期間は8,192と500,000のあいだのどこまで短くできるか。補助の勾配を止め、長さを無作為にし、繰り返しを序盤だけにする組み合わせでも基準と同じ帯か（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、解釈の限界と残る問い）。
- bonusが続く状態の内的報酬を混ぜたとき、成績は落ちるか（出典: [RNDの害の確認とpredictorの周期リセット](../../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)、解釈の限界と残る問い）。

## 撤回・更新

- 2026-11-06: 「STOPでも立ち上がりが改善し、原因は分からない」→「STOP×Singleは決定的モードのseed 1で乱数とtargetのforwardを分けると無効と全点一致し、無効の別の標本だった。改善に見えた差は偶然である」（出典: [STOP×Singleと無効のbit比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-stop-single-bit-compare.md)、現時点のまとめ1〜4）。
- 2026-11-05: 「通常TempoRLと無効の重みの差を作ったのが、学習中の繰り返しか、補助のlossか、その組み合わせかは分からない」→「序盤のTrainの繰り返しで説明できる。warmupの間だけでは足りず、学習した長さ、0.5Mより後の繰り返し、補助の勾配、評価時の繰り返しは、それぞれ無くても基準と同じ帯だった。同時に外した組み合わせは測っていない」（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ1〜4）。
- 2026-11-05: 「立ち上がりの速さの差が同じ重みの評価でどう分かれるかは分からない」→「基準の同じRunのfullとsingleは立ち上がりでも同じ帯だった」（出典: [Trainの繰り返しの時期と長さの比較](../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md)、現時点のまとめ5）。

## 関連

- [atari](atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [内的報酬による探索の比較](../methods/intrinsic_novelty.md)。
- [campaignにない独自の分析](../../../experiments/default-dqn/lunarlander/README.md#独自の分析)。
- [temporal_abstraction](../methods/temporal_abstraction.md#見立て)（TempoRL の得と最大スキップ長の見立て）。
- [noisynet](../methods/noisynet.md#見立て)（ε ラダーの置き換えとノイズの残り方の見立て）。
- [temporal_abstraction](../methods/temporal_abstraction.md#横断的知見)（TempoRL の得と損の出どころを題材で比べた横断的知見）。
- [intrinsic_novelty](../methods/intrinsic_novelty.md#横断的知見)（内的報酬が消える・下がる局面の並べ）。
