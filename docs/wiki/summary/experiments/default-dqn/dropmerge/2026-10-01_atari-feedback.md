---
title: DropMerge への Atari 知見のフィードバック実験（FB シリーズ）の探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md|default-dqn/dropmerge/2026-10-01_atari-feedback]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# DropMerge への Atari 知見のフィードバック実験（FB シリーズ）の探索履歴

## 元の材料

[DropMerge への Atari 知見のフィードバック実験（FB シリーズ）の探索履歴](../../../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) blob:f8b8f1af83e143d10ff7b3adcb9c431576e6f530

## 要点

- Munchausen（温度 τ0.03、α0.9、clip −1、Double DQN OFF）は、この系列で最大の単独の改善だった。IQN の 100M checkpoint の評価専用 Run で、Eval1 は 1,524 → 1,796（+18%）、Eval3（Greedy）は 1,572 → 1,831（+16%）。IQN+M の 2 seed の差は 1〜2% だった。
- 8 月から続いていた「IQN は QR51 に負ける」は、評価方策 UQE との相互作用だった。Greedy で評価すると IQN 1,572 対 QR51 1,552 で、IQN が負けるのは UQE で評価したときだけ（1,524 対 1,694）。QR51 は Greedy で評価すると 84〜88% の局が NoDropTimeout で終わる。Munchausen を入れた後は、評価専用 Run の 2 seed 平均で IQN+M が QR+M を上回った（Eval1 1,784 対 1,671）。ただし QR+M は 2 seed で 10〜15% ずれた。
- target の hard 更新（C=200、soft と同じ時定数）は、IQN+M で soft の 2 seed より 3 方策とも低く、NOOP が増えた。soft（0.005）のままにした。
- ε ラダーの上で試した n-step 3 と RR4+SN は、どちらも深い長考型 NEET に入り、評価専用 Run で 10〜16% 下がった。NoisyNet 基準の上で RR4 だけを試すと NEET は出ず、100M の学習中の評価は FB シリーズで最高の 1,851（nzrep 比 +4.1%）だった。同じ実時間では RR1 が大きく上回る（8 時間で 1,777 対 1,548）。
- 後の系列（10/18〜19）で、学習側の平均点には NoisyNet の探索の損が含まれ、勾配ノイズを減らす群は NoisyNet の揺らぎに鈍くなるだけで学習側の点が 2〜3% 上がり、決定的な評価（各ランナー 5,000 局）は変わらないことが分かった。11 の学習側の差（31 と 34 の分岐）は、評価側での確認が要る。学習中の評価と評価専用 Run に基づく結論はこの影響を受けないが、学習中の評価は 10M 窓で ±2% ほどぶれる。
- Run ごとのぶれは、IQN+M の 2 seed で 1〜2%（評価専用 Run）、QR+M の 2 seed で 10〜15%（評価専用 Run）、nzrep と後の系列の seed 2 で約 6%（学習側の平均点）だった。1 Run どうしで 6% に満たない差（RR4 +4.1%、γ0.999、ε ラダー＋NoisyNet、B512/RR2 −5%、nzrep と −20 基準の差）は確定していない。

## 内容

### この文書の読み方

「NOOP 選択率」は Eval1 で NOOP が UQE の 1 位になった step の割合で、評価ランナー自身の step 軸の後半で平均した。

Q の水準（`37_agent_qtd/31_q_max_real_mean`）と Munchausen の指標（`36_agent_munchausen/*`）は経験数の軸で記録される。この文書の値は経験数の窓で引き直した。当時の分析の 10/03〜10/12 の値の一部は学習ステップの窓で切っていて、学習の最初の約 0.4M 経験だけを見ていた。該当するブロックに判断更新として書いた。

### 基準の実効設定

出発点は `run_20261001-191232_dmfb_base` の `config/config_data.txt`（`run.@iqn32_stratified`）。

### 探索ブロック 01: 引き継ぎの照合と Munchausen の 20M スクリーニング

同じ設定の基準 100M（探索ブロック 02）の 10–20M は 1,006 / 948 で、基準 20M との差は Eval1 0.8%、Eval2 1.6% だった。

τ0.01 の 10–20M では、Q の差（h 空間）0.0032 → 0.0125、target と online の行動の食い違い 0.67 → 0.32、dormant 0.012 → 0.004、Munchausen の softmax のエントロピー 0.60（最大 ln59 = 4.08）、clip 率 2.8% だった。

### 探索ブロック 02: 基準 IQN と QR51 の 100M、評価専用 Run による切り分け

90–100M の Eval1 − Eval2 は IQN 131、QR51 26。実空間の Q は 40M 以降 IQN 9.1〜10.0、QR51 5.7〜7.4 だった。

Greedy で選ぶと IQN は QR51 と同等かやや上だった。IQN が負けたのは UQE で選んだときだけだった。

### 探索ブロック 03: 活性化関数を ReLU にする群（50M）

A ReLU は 0–20M に Eval1 で 66〜111 遅れ、40–50M の差は −24 だった。

DropMerge では、活性化関数を ReLU にしても改善は見えなかった。Atari での効き方は行動の入れ替わりを減らして一貫した狙いを保つことだったが、DropMerge はほぼ同価値の行動（隣の列）が多く、点数は数百手の積み重ねで決まる。表現は巨大な共有トランクが支配している。

### 探索ブロック 04: Munchausen τ0.03 の 100M（IQN と QR51、各 2 seed）

IQN+M の 2 seed の差は評価専用 Run で 1〜2%、学習中の評価の 90–100M でも Eval1 で 1.7% だった。QR+M の 2 seed は評価専用 Run で 10〜15% ずれ、学習中の評価は 0–10M から 90–100M まで全窓で seed 2 が 120〜170 低かった。2 Run の設定 dump の差は `run.seed` と `app.run_name` だけで、評価専用 Run の標準誤差は 20 前後なので、測定の揺れではなく学習結果のばらつきだった。

IQN+M は 0–10M から時間切れ型の NEET を持ち続けた。学習側の局の 4〜13% が NoDropTimeout で終わり、Eval1 の NOOP 選択率は 25% だった。それでも 70M で QR51 の 100M の到達点を越えた。

### 探索ブロック 05: target の hard 更新（C=200）

IQN+M hard は 3 方策とも soft の 2 本より低く、soft の低い方の seed との差は 28〜79 だった（soft の 2 seed の差は 21〜42）。学習中の評価も全窓で下で、0–10M は 700 対 924 / 947、90–100M の 3 方策平均は 1,677 対 1,784 / 1,755 だった。

QR+M hard は soft の低い方の seed と同程度だったが、soft の 2 seed がどちらも 0% だった NoDropTimeout が 39〜58% 出た。

### 探索ブロック 06: NoDropTimeout の罰 −20

時間切れの終局はほぼ消えた。学習側の局の NoDropTimeout も 1〜5%（−10 は 4〜13%）に下がった。

待つ行動は減らなかった。Eval1 の NOOP 選択率は 34%（−10 seed 1 は 25%）、学習中の評価の Eval1 の局の長さは 418 step（−10 は 400〜417）。時間切れ型が「締切の直前で落とす」長考型に変わった。

### 探索ブロック 07: n-step 3 と RR4+SN（ε ラダーの基準の上）

2 群とも長考型の NEET が深かった。時間切れの終局は少なく、局が長い。Eval1 の NOOP の差（UQE、後半の平均）は −20 基準 −0.062、n-step 3 +0.001（同点）、RR4+SN +0.059（NOOP が上）だった。

RR4+SN は序盤だけ速かった（0–10M 1,017 / 1,002 / 1,055、−20 基準比 +11%）。5〜10M に NEET が始まり、15〜20M が山で、100M まで抜けなかった。学習中の評価は 70M から頭打ちだった。学習側の平均点は 90–100M で 1,049 対 1,468（−29%）。

### 探索ブロック 08: NoisyNet と NEET の切り分け

nzrep は最初から最後まで NEET が出なかった。NOOP 選択率は全窓で 0%、時間切れと投了も 0、学習中の評価の局の長さは 169 → 289 step。評価専用 Run は −20 基準比 +3.7% / +4.4% / +0.6% で、SpawnBlocked が 94〜97% だった。序盤は遅く（0–10M は −31%）、50M ごろに追いつき、70–80M は +4%、90–100M はほぼ同じだった（3 方策の平均）。

nzadd は nzrep との違いが `uqe_eps_start`（0.6 対 0）だけなのに NEET が出た。16M ごろから長考型に入り（学習中の評価の Eval1 の局の長さは 20–30M で 1,824 step）、32〜36M に抜けた。50M 以降は終盤の時間切れ型が増え、評価専用 Run の NoDropTimeout は 55〜75% だった。成績は 70M 以降 −20 基準と ±3% 以内で、評価専用 Run も同等だった。

### 探索ブロック 09: NoisyNet 基準の上の B512/RR2 と RR4、実時間

RR4 の 90–100M は 1,840 / 1,804 / 1,908 で、FB シリーズの 100M の全 Run で最高だった（次は nzadd 1,791、IQN+M seed 1 1,784、nzrep 1,778、−20 基準 1,770）。経験あたりの速さは 40M までは約 2 倍（30–40M で RR1 の 60–70M に並ぶ）、終盤は約 1.1 倍。最後の 30M も伸び続けた。

### 探索ブロック 10: NN 構造（共有トランクの大きさと Transformer の層数）

TF8 は 10–20M に NEET（学習側の時間切れ 4.9%）が出て一度薄れたが、終盤に再び強まった（90–100M の学習側の時間切れ 7.1%・投了 6.3%、学習中の評価の局の長さ 376 step）。最後の 10M は横ばいだった。

### 探索ブロック 11: RR1 の 300M 継続

最後の 10M は 2,033 だった。nzrep の 8.2 時間と合わせて 25.8 時間で、RR4 の 100M（26.5 時間、1,851）より約 10% 上だった。

### 探索ブロック 12: 100M checkpoint からの 20M 分岐（PER 初期優先度・B512/RR1・B512/RR2）

対照と 300M 継続（探索ブロック 11）の同じ区間の学習側の平均点は 0.15% 差だった。

### 探索ブロック 13: γ0.999

最後の 20M の伸びは 10M あたり +52（nzrep +18）だったが、学習側で同じ点数の高さ（1,680〜1,730）なら +47 対 +43 で同程度だった。

### 探索ブロック 14: 120M からの 50M 分岐（対照と B512/RR1）

34a は 300M 継続（探索ブロック 11）の同じ区間と、学習側の平均点で各 10M 窓 0.8% 以内で一致した。

### 関連文書

Atari: BTR からのフィードバックの群（Munchausen・活性化関数・target 更新）

## 主要な概念

- [DropMerge](../../../../concepts/subjects/dropmerge.md)
- [報酬尺度と割引率](../../../../concepts/methods/reward_scale_discount.md)
- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
- [Target更新の比較](../../../../concepts/methods/target_update.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
