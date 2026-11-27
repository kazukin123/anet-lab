---
title: 実時間優先プロファイルの Atari-5 と評価予算
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md|default-dqn/atari/2026-09-21_realtime-profile-atari5]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/subjects/phoenix.md|Phoenix]]"
  - "[[docs/wiki/concepts/subjects/name_this_game.md|Name This Game]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# 実時間優先プロファイルの Atari-5 と評価予算

## 元の材料

[実時間優先プロファイルの Atari-5 と評価予算](../../../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md) blob:02ef8a673202c1bfa4a8aa297f6c9a7fb95890fa

## 要点

- 旧 NN・RR1 の実時間優先構成で Atari-5 を比較した。seed 1 の 40〜50M の ε=0.01 評価は HNS 中央値 358.0%、回帰値 344.8%。5 ゲーム中 4 つが終盤でも伸びた。
- seed 2 では全 5 ゲームが 90〜100M 窓で自己最高となり、回帰値は 444.9%。学習中の最良を用いると 465.5 で、BTR の 555.2 の 83.8%。100M は BTR の 2 倍のフレーム予算である。
- 50M の 5 本では評価の環境 step が学習予算の 83% を占めた。Greedy の評価は発火が少なくても、エピソード数と長さのため総仕事量が大きい。
- name_this_game では Replay 容量を倍にすると経験の古さは増えたが、得点の改善は確認できなかった。Breakout の正の効果との違いにはゲーム・RR・NN が同時に関わる。
- qbert の旧 NN に Pool4 を加えると終盤の評価・学習の両方が低下した。最大層は小さくなったが、実時間の利得は確認できなかった。
- phoenix の Greedy 得点は突破ウェーブ数とともに変動した。qbert は seed 2 の 100M 最終重みでも Greedy 100 ゲーム中 93 が第 8 面に達し、突破は 0 だった。

## 内容

### メタデータ・対象 Run

Breakout で選んだ旧 NN・RR1・hard C=125 に Munchausen と replay_fit を加え、5 ゲームで測った。Atari-5 回帰は、各ゲームの HNS に 1 を足した値の常用対数を係数で加重し、元の尺度に戻す。5 ゲームの単純中央値とは区別する。係数は battle_zone と name_this_game が大きい。集計ツールは実効設定の AtariEnv.game でゲームを対応付ける。

seed 1 の 50M、name_this_game の容量 4M、qbert 20M の Pool4 比較、サンプル効率優先の 10M 対照、seed 2 の 100M、qbert の最終重みの評価専用 Run を区別する。name_this_game の最初の 100M Run は異常に遅く 25.7M で停止したため、集計には再実行分を使った。原因は未確定である。

### 実時間優先 50M と評価予算

ε=0.01・10 エピソードの評価を 10M 窓で平均すると、40〜50M の中央値 358.0%、回帰値 344.8%。前の 20M スイープは中央値 55.4% で、double_dunk・phoenix の 25M 前後の急伸に届いていなかった。Greedy の学習中最良の回帰値は 370.7。BTR の 555.2 と比べる際は、こちらは 1 seed、BTR は 3 seed であり、評価回数・1 回のゲーム数も確認する。ε=0.01 の 199 回から最大を取ると 450.0 だが、1 回 10 ゲームなので、同じ精度の最大値としては扱っていない。

5 本で Greedy の評価は 155.0M、ε=0.01 の評価は 53.8M 環境 step。訓練 250M に対して合計 83% で、学習の停止時間は合計 4.75 時間。Greedy は 10 回、ε=0.01 は 199 回発火した。評価の回数だけでは負荷を説明できない。400 ゲームを 100 にすれば同条件で Greedy の仕事量は約 38.8M になる見込みだった。固定方策を 100 ゲームずつ 5 回測ったセッション間の HNS 変動係数は、name_this_game 0.46%、double_dunk 0.47%、qbert 1.36%、battle_zone 4.32%、phoenix 5.26%。name_this_game の評価 lane を 16 から 128 に増やすと 14 分 45 秒から 11 分 36 秒に短縮した。

### Replay 容量と旧 NN の Pool4

name_this_game の設定で容量だけを 2M から 4M に変えた。lane 窓は 16,384 から 32,768 へ伸び、Replay の age・count は約 2 倍になった。一方、未抽選率は 56.0% と 55.5%、共通する Greedy 評価 8 点の差は平均 −2.2%、終盤の ε=0.01 評価は −2.5%。各 1 seed で、ビルドの違いもある。容量増による悪化を確定する比較ではない。Breakout の先行実験は RR4・BTR 系 NN で、効果の違いをゲームだけに帰せない。

qbert の Pool4 は旧 NN の線形層への入力を 7,744 から 1,024 に減らした。基準と Pool4 を交互に各 2 本流すと、15〜20M の ε=0.01 評価は基準 15,720・14,550、Pool4 10,830・8,550。学習側の得点と Greedy の断面も重ならなかった。q_gap は 19.4%、q_max は 25.5% 低下した。総実時間は平均 0.895 時間と 0.928 時間で、Pool4 自体のばらつきは 11.4%。序盤は群間差が複製幅内にあり、終盤で初めて判別できた。

### seed 2 の 100M

ε=0.01 評価の回帰値は 40〜50M の 358.9 から 90〜100M の 444.9、中央値は 383.4 から 463.3 へ伸びた。ゲーム別の同期間の伸びは phoenix 75.1%、name_this_game 21.3%、battle_zone 20.8%、qbert 12.3%、double_dunk 5.6%。double_dunk の Greedy は 91.1M に 23.0 点で、公表値の上位に近い。qbert の全 93,010 ゲームで第 8 面突破は学習側 2 ゲームだけだった。phoenix の Greedy では 1 ウェーブあたりの得点は比較的安定し、突破ウェーブ数と得点が一緒に上下した。

50M 時点の 2 seed の回帰値は近くても、ゲーム別の HNS 比は 0.76〜1.27。例えば double_dunk と phoenix では急伸の時期が違った。BTR と同予算の seed 2 の最良回帰は 420.7、2 seed の平均で 396.5。BTR の 2 倍の 100M では battle_zone と double_dunk がその公開値に届いたが、phoenix は 46.0%、qbert は 57.5% だった。内部指標では action_churn は全ゲームで減り、q_gap の縮小は battle_zone に限られた。評価予算を減らした seed 2 は 4.1〜5.4 時間で 100M に達した。初期評価がエピソードの上限に張り付くと、特定ゲームの評価予算が膨らむ。

### qbert の画面と残った問い

100M の重みを online・sticky 0.25・ε=0 で再評価した。第 8 面の突破は 0。別に EvalPanel の target・sticky 0 で見ると立方体の側面は黒く、原画面の色だった。到達時の残機が記録されておらず、その面の難しさとそれ以前の消耗は分けられない。100M 以降の飽和点、phoenix の BTR との差、他ゲームの lane 窓下限、RR4 での容量比較、旧 NN の Pool6 もこの記録では未確認である。

## 主要な概念

- [Atari 横断](../../../../concepts/subjects/atari.md)
- [計測の負荷と比較条件](../../../../concepts/measurement/diagnostic_cost.md)
- [Replay ratio と lane 窓](../../../../concepts/methods/replay_ratio_lane_window.md)
- [Qbert](../../../../concepts/subjects/qbert.md)
- [Phoenix](../../../../concepts/subjects/phoenix.md)
- [Name This Game](../../../../concepts/subjects/name_this_game.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
