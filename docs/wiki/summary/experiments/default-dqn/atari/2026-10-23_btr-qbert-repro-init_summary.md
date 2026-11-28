---
title: BTR 公開実装の qbert 再現と、序盤の学習の遅れ（初期値） 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md|default-dqn/atari/2026-10-23_btr-qbert-repro-init]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-28
---
# BTR 公開実装の qbert 再現と、序盤の学習の遅れ（初期値） 探索履歴

## 元の材料

[BTR 公開実装の qbert 再現と、序盤の学習の遅れ（初期値） 探索履歴](../../../../../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md) blob:fc14c4651561af14f3cdfcb23b9d9ef79000ad1f

## 要点

- BTR 公開実装は、この環境でも第 8 面を越えた。
- BTR も第 9 面（L3R1）で止まる。
- anet を BTR に寄せても、第 8 面はほぼ越えなかった。
- anet の qbert は、どの設定でも序盤の学習が BTR の半分ほどの速さだった。
- 実時間優先の構成（SN なし、LayerNorm あり）では、初期値を戻しても上がらなかった。
- 初期値を戻し、ε を BTR と同じ経過にした 50M で、第 8 面の突破は BTR と同じ帯になった。
- 30M 以降の train の平均点とゲームの長さの伸びは、第 9 面で終わるゲームが増えたことと、第 9 面で重ねる点が増えたことによる。BTR の 2 本も同じ形だった。

## 内容

### 探索ブロック 01: BTR 公開実装をこの環境で回す

5M 窓では、公表 CSV が 45-50M で 36,529、r0 が 29,405 だった。r1 は 35-40M で 32,054（公表 CSV は 32,533）。 r0 の最後の評価（50M、20 本）は平均 35,931、最高 63,325。

### 探索ブロック 02: anet を BTR に寄せる（btrlike）

45-50M の train の突破は 0.0〜0.2% だった。 第 8 面への到達は 27.5-30M の 66.0% が最高で、そこから上がらなかった。

### 探索ブロック 03: 縮小方式を `pixel_area` にする（btrpa）

50M まで突破は 0 本だった。 train の第 8 面の到達は 40-42.5M 49.6%、47.5-50M 59.8%。ε=0 の評価（100 本）の到達は 45-47.5M で 79%、突破は 0 本。

### 探索ブロック 04: ε を BTR と同じ経過にする（btreps）

25M までは btrpa とほぼ同じだった。両方の ε の経過は 25M までほぼ同じ（直線 4M と時定数 2M の指数）。

### 探索ブロック 05: 序盤の遅れは anet の qbert の全 Run に共通

評価（ε 0.01、2.5M 窓）で、5-7.5M に BTR の帯（9,651〜10,766）へ届いた anet の Run は無かった。

### 探索ブロック 06: コードの照合と初期値の修正

5-7.5M の 11,450 と 11,313 は、探索ブロック 05 の anet の Run の最高（8,675）を大きく超える。

Munchausen online を足した aln_im は、2.5M 以降の 3 窓とも aln_init を上回らなかった（差は 252・137・1,140）。

### 探索ブロック 07: 実時間優先の構成で初期値を戻す

サンプル効率優先の構成（探索ブロック 06）とは逆に、2.5-10M の 3 窓とも従来の 2 本を下回った。

### 探索ブロック 08: 初期値を戻した設定で 50M（qbert）

第 8 面への到達は 27.5M 以降 88.5〜96.5%。btreps（同じ ε、従来の初期値）は 33.1M まで突破 0 本だった（探索ブロック 04）。

### 探索ブロック 09: 第 9 面で終わるゲームの点と長さ（train）

どの Run も、第 9 面で終わるゲームが増えるにつれて、第 9 面で終わったゲームの得点が上がった。第 9 面の突破は増えなかった。

### pending

BTR の第 9 面の壁が、色の戻るルールによるものか、速度と敵の量によるものか（anet の合算の checkpoint の Evaluation Trace では、第 9 面で同じあたりのキューブの色を付けたり戻したりをくり返した。2026-10-29 探索ブロック 04）。

## 主要な概念

- [Qbert](../../../../concepts/subjects/qbert.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
- [315 ニューラルネットワーク](../../../../concepts/framework/315_nn.md)
