---
title: 重みの初期値
page_kind: 手法
aliases:
  - 初期値
  - 初期化
  - init
  - he
  - xavier
  - orthogonal
  - 直交
  - PyTorch の既定
  - torch の既定
  - variance scaling
  - Kaiming
  - aln_init
description: 重みの初期値を torch の既定に戻した効きは構成で分かれた。qbert のサンプル効率優先（SN 付きの残差 conv、1 seed）と Breakout の Nature 型（seed 2 の 1 対）では上がり、qbert の実時間優先（残差に SN が無く、胴体の後ろに LayerNorm、1 seed）では従来の初期値の 2 本を下回った。
updated: 2026-11-27
---
# 重みの初期値

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

重みの初期値を torch の既定に戻した効きは構成で分かれた。qbert のサンプル効率優先（SN 付きの残差 conv、1 seed）と Breakout の Nature 型（seed 2 の 1 対）では上がり、qbert の実時間優先（残差に SN が無く、胴体の後ろに LayerNorm、1 seed）では従来の初期値の 2 本を下回った。
Atari の DQN 系（DQN・Rainbow・IQN）や IMPALA の ResNet で、He・直交・PyTorch の既定・Xavier・Dopamine の variance scaling を並べて比べた報告は、2026-11-27 の調査では見つからなかった。
qbert の記録が初期値の効きの試算に置いた前提（SN 付きの層は出力も Adam の 1 回の更新幅も重みの大きさに依らない）は外の解析と同じ形だが、更新の割合は測っていない。
DropMerge の `dmfb_47b_scr_ctl`（2026-11-01）の実効設定には、torch の既定を指定した block も SN を指定した block も無かった。

## 分かっていること

### anet の知見

- 初期値の比較: サンプル効率優先の1 seedでは、convとτ埋め込みの初期値をPyTorch既定へ戻した評価の5〜7.5Mが11,450、7.5〜10Mが14,918で、BTRの2本の帯に入った。Munchausenのlog πをonlineへ替えた上乗せは見えなかった。実時間優先の1 seedでは7.5〜10Mが4,448で、従来の2本の7,306・5,068を下回った。構成をまたいで初期値の効果を一般化しない（出典: [BTRとの比較と初期値の記録](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック06・07。Run: `run_20261025-191201_a5_qb_aln_init_s1` / `run_20261025-224454_a5_qb_rt_aln_s1`）。
- `dmfb_47b_scr_ctl`（2026-11-01。DropMerge の TempoRL（scratch 第 1 弾）と、対照の新旧の差の探索履歴）の実効設定では、DropMerge の main_feature は、conv（`DMImpalaConv64`・`DMImpalaConv128`）が `he`、残差 block（`DMImpalaRes64`・`DMImpalaRes128`）が norm なしで、そのあとに TransformerEncoder（`DMImpalaTrans128`。4 層、`norm_first = true`）と `Linear_2048`（`orthogonal`、gain 1.0）が続き、Head の `HeadFC1024` と τ の射影 `IQNTauProj2048` は `xavier` だった。qbert のサンプル効率優先で戻した torch の既定（`init.mode = default`）を指定した block も、SN（`weight_norm.mode`）を指定した block も無かった（出典: [2026-11-01_temporl-and-control-gap.md](../../../experiments/default-dqn/dropmerge/2026-11-01_temporl-and-control-gap.md)、参照 Run；[run_20261101-081421_dmfb_47b_scr_ctl.txt](../../../experiments/default-dqn/dropmerge/config/run_20261101-081421_dmfb_47b_scr_ctl.txt)、`DefaultDQNAgent.net.branch.[main_feature].structure`・`DefaultDQNAgent.net.branch.[tau_embedding].structure`・`DefaultDQNAgent.net.branch.[value_stream].structure` と各 block の定義）。

### 外の事例

- Atari の DQN 系（DQN・Rainbow・IQN）や IMPALA の ResNet で、He・直交・PyTorch の既定・Xavier・Dopamine の variance scaling を並べて比べた報告は、2026-11-27 の調査では見つからなかった。探した範囲は、初期化の比較・ablation の語での web 検索、DER と DrQ(ε) の 12 の hyper-parameter の比較、IMPALA の CNN の Rainbow の改良の一覧、BTR の論文の Appendix H で、どれにも初期化は入っていない（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§4・§6）。
- スケール不変な重みは、平衡より小さいノルムなら平衡より速く、大きいノルムなら遅く回るので、初期のノルムが平衡と比べてどうかで、学習の初めの過渡期に回りすぎるか回らなすぎるかが決まる（AdamW の解析）。学習率の warmup を入れても、過渡期に速く回りすぎる例があった（ResNet-20・CIFAR-10）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.4）。
- weight decay の無い Adam で、初期値の大きさの差がどれだけの更新回数で消えるかを定量した文献と、LayerNorm の付いた構成で初期値を小さくしても学習が速くならない理由を論じた文献は、2026-11-27 の調査では見つからなかった（探した範囲: 初期化の比較・初期値の大きさ・ablation の語での web 検索と、読んだ論文の本文）（出典: [weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§3・§4）。

## 横断的知見

- 対立点: 重みの初期値を torch の既定（`init.mode = default`）に戻した効きは、構成で分かれた。qbert のサンプル効率優先（SN 付きの残差 conv、1 seed）では、評価の 5-7.5M が 5,544 から 11,450 に、ε=0 の評価（5.25M）が 3,994 から 7,865 に上がった。Breakout の Dopamine の IQN に寄せた Nature 型（conv 3 層と Linear 512。`run.@rainbow_like` の Q 側から PER を外した構成で、`run.@rainbow_like` を載せた Montezuma の `mz_rbw_rndtg_s1`（2026-11-17。Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen））の実効設定では、この型の main_feature に SN も LayerNorm も無かった。全層を戻した seed 2 の 1 対）でも、学習側の it 15〜19 の平均が 59.6 から 89.6 に上がった。qbert の実時間優先（残差に SN が無く、胴体の後ろに LayerNorm、1 seed）では、評価の 7.5-10M が 4,448 で、従来の初期値の 2 本（7,306・5,068）を下回った。条件の違い: 戻した層、正規化の有無と種類、指標（qbert は評価、Breakout は学習側のゲームの点）が違い、どれも 1 seed か 1 対である（出典: [2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック 06・07；[0193_variance_scaling_weight_init_10rfd.md](../../../memo/0193_variance_scaling_weight_init_10rfd.md)、§5.3；[2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、目的と比較条件；[run_20261117-043636_mz_rbw_rndtg_s1.txt](../../../experiments/default-dqn/atari/config/run_20261117-043636_mz_rbw_rndtg_s1.txt)、`DefaultDQNAgent.net.branch.[main_feature].structure = StackMerge > AtariConv32 > ReLU > AtariConv64 > ReLU > AtariConv64Final > ReLU > Flatten > AtariLinear512 > ReLU` と各 block の定義）。
- 補完: qbert のサンプル効率優先の記録が初期値の効きの試算に置いた前提（SN 付きの層は出力が重みの大きさに依らず、Adam の 1 回の更新幅も依らないので、初期値が 2.45 倍大きいと 1 回の更新で重みが変わる割合が約 0.4 倍になる）は、スケール不変な重みの実効学習率を SGD で η/‖w‖₂²、Adam で η/‖w‖₂ とする外の解析と同じ形である。Gogianu らは、SN 付きの層を Adam で学ぶと、ε をスペクトル半径倍にした形の更新になると書く。条件の違い: 外の解析は BN 付きの CNN の CIFAR-10 と、DQN の 1 層だけの SN で、Adam の ε を無視した近似を含む。anet は SN 付きの残差 conv の IQN で、試算は seed 1 の比較の説明として置かれ、更新の割合は測っていない（出典: [2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック 06；[weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§5.1・§5.5）。

## 見立て

- qbert のサンプル効率優先で初期値を torch の既定に戻すと序盤が速くなったのは、SN 付きの層では出力が重みの大きさに依らない一方で Adam の 1 回の更新幅も重みの大きさに依らないので、標準偏差が既定の 0.87〜2.45 倍ある he の初期値では 1 回の更新で重みが変わる割合が小さく、学習が遅れていたためではないか。 状態: 未検証
  - 支える事実: (1) コードの照合で、先頭の conv と SN 付き残差 conv の初期値は anet が he、BTR が PyTorch 既定で、標準偏差は anet が 0.87〜2.45 倍であり、記録は、SN 付きの層の出力と Adam の 1 回の更新幅がどちらも重みの大きさに依らないので、初期値が 2.45 倍大きいと 1 回の更新で重みが変わる割合が約 0.4 倍になると書く（出典: [2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック 06；[weight_init_normalization_adam_survey_2026-11-27.md](../../../../reports/weight_init_normalization_adam_survey_2026-11-27.md)、§7.3）。 (2) 同じ構成で初期値だけを戻すと、評価の 5-7.5M が 5,544 から 11,450 に、ε=0 の評価（5.25M）が 3,994 から 7,865 になった（出典: [2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック 06）。 (3) 序盤の遅れは ε の形、env 数、replay の容量、NoisyNet の有無を変えても残り、評価の 5-7.5M は anet の qbert の 20 本以上で最高 8,675、BTR は 9,651〜10,766 だった（出典: [2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、現時点のまとめ 5・探索ブロック 05）。
  - 合わない事実: (1) 実時間優先の構成（残差に SN が無く、胴体の後ろに LayerNorm があり、記録は LayerNorm も手前の層の重みの大きさを打ち消すとして大きい初期値の層を戻した）では、評価の 7.5-10M が 4,448 で、従来の 2 本（7,306・5,068）を下回った（出典: [2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック 07）。 (2) SN も LayerNorm も無い Breakout の Nature 型でも、全層を torch の既定にすると学習側の it 15〜19 の平均が 59.6 から 89.6 に上がった（seed 2 の 1 対。Nature 型は `run.@rainbow_like` の Q 側から PER を外した構成で、`run.@rainbow_like` を載せた Montezuma の `mz_rbw_rndtg_s1`（2026-11-17。Montezuma's Revenge の RND（Taiga ら寄せの構成、β と Munchausen））の実効設定では、この型の main_feature に SN も LayerNorm も無かった）（出典: [0193_variance_scaling_weight_init_10rfd.md](../../../memo/0193_variance_scaling_weight_init_10rfd.md)、§5.3；[2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、目的と比較条件；[run_20261117-043636_mz_rbw_rndtg_s1.txt](../../../experiments/default-dqn/atari/config/run_20261117-043636_mz_rbw_rndtg_s1.txt)、`DefaultDQNAgent.net.branch.[main_feature].structure = StackMerge > AtariConv32 > ReLU > AtariConv64 > ReLU > AtariConv64Final > ReLU > Flatten > AtariLinear512 > ReLU` と各 block の定義）。
  - 条件の違い: qbert の 2 構成は戻した層が違い、実時間優先は残差の 2 本目の conv を 0 から始まる既定のままにした。env 数と replay の容量も違う（サンプル効率優先は 64 env・1M、実時間優先は RR1・128 env・2M）。Breakout は Dopamine 寄せの構成で、指標は学習側のゲームの点である。どれも 1 seed か 1 対。DropMerge の `dmfb_47b_scr_ctl`（2026-11-01）の実効設定の構成（norm の無い conv の残差のあとに pre-LN の TransformerEncoder、SN なし）は、どれとも違う（[2026-11-01_temporl-and-control-gap.md](../../../experiments/default-dqn/dropmerge/2026-11-01_temporl-and-control-gap.md)、参照 Run；[run_20261101-081421_dmfb_47b_scr_ctl.txt](../../../experiments/default-dqn/dropmerge/config/run_20261101-081421_dmfb_47b_scr_ctl.txt)）。
  - 確かめ方: サンプル効率優先の qbert で、層ごとの 1 回の更新での重みの変化の割合を、従来の初期値の Run（btreps）と戻した Run（aln_init）で比べる（その値を記録するメトリクスがあるかは確かめていない。無ければ 10M の 2 本を流して測る。新しい Run が要る）。DropMerge では、main_feature の conv と Linear を torch の既定にした群を既定と比べ、序盤の評価を見る（新しい Run が要る）。
  - 出どころ: AI（開発者の質問を起点、2026-11-04）

## 分かっていないこと

- 実時間優先の構成では、初期値を戻しても評価が上がらなかったのはなぜか（出典: [BTRとの比較と初期値の記録](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、pending・探索ブロック07）？
- DropMerge の `dmfb_47b_scr_ctl`（2026-11-01。DropMerge の TempoRL（scratch 第 1 弾）と、対照の新旧の差の探索履歴）の実効設定の構成（norm の無い conv の残差のあとに pre-LN の TransformerEncoder、SN なし）で、初期値を torch の既定に戻すと、序盤の立ち上がりと成績は変わるか（出典: [2026-11-01_temporl-and-control-gap.md](../../../experiments/default-dqn/dropmerge/2026-11-01_temporl-and-control-gap.md)、参照 Run；[run_20261101-081421_dmfb_47b_scr_ctl.txt](../../../experiments/default-dqn/dropmerge/config/run_20261101-081421_dmfb_47b_scr_ctl.txt)、`DefaultDQNAgent.net.branch.[main_feature].structure`・`net.block.[DMImpalaRes64].res.norm_type = none`・`net.block.[DMImpalaTrans128].tf.norm_first = true`；[2026-10-23_btr-qbert-repro-init.md](../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、探索ブロック 06・07・pending。開発者の発言（2026-11-04））。

## 撤回・更新

なし

## 関連

- [Atari横断](../subjects/atari.md)。
- [Qbert](../subjects/qbert.md)。
- [Breakout](../subjects/breakout.md)。
- [DropMerge](../subjects/dropmerge.md)。
- [可塑性指標と成績](../phenomena/plasticity.md)（正規化と実効学習率の外の事例）。
- [NoisyNetの観測](noisynet.md)（公開実装のノイズ付き線形層の初期値）。
- [RFD0193](../../../memo/0193_variance_scaling_weight_init_10rfd.md)（重みの初期化に、分散を倍率÷fan_in にする一様分布の方式を足す）。
