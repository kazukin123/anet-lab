---
title: サンプル効率優先プロファイルの Atari-5 展開 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md|default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/subjects/battle_zone.md|Battle Zone]]"
  - "[[docs/wiki/concepts/subjects/phoenix.md|Phoenix]]"
  - "[[docs/wiki/concepts/subjects/name_this_game.md|Name This Game]]"
  - "[[docs/wiki/concepts/subjects/double_dunk.md|Double Dunk]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# サンプル効率優先プロファイルの Atari-5 展開 探索履歴

## 元の材料

[サンプル効率優先プロファイルの Atari-5 展開 探索履歴](../../../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md) blob:ab22bcac4a97c53ce7a5183b12309ae9678c46bd

## 要点

- Atari-5 回帰は 458.3（学習中の最良、50M）で、BTR 555.2 の 82.5%。
- 実時間は 5 本合計 35.6h。 実時間優先は 50M で 18.9h（seed 1）/ 11.9h（seed 2）、100M で 24.0h（seed 2）。
- BTR 比はゲームで二分される。 phoenix 94% / name_this_game 102% / double_dunk 101% に対し、 battle_zone 68% / qbert 60%。残りの差はこの 2 本にあり、battle_zone は Atari-5 の重みが最大（0.382）である（探索ブロック 01）。
- phoenix は実時間優先の 2.2〜2.6 倍。
- battle_zone は序盤だけ速い。
- qbert の第 8 面の壁は RR4 でも同じ。
- qbert の `round_clear` = 8 は第 8 面のクリアと一致しない場合がある。

## 内容

### 探索ブロック 01: Atari-5 サンプル効率優先 50M 横断

Atari-5 回帰は 458.3 で、BTR の 82.5%。 学習中の最良（ε=0 断面、100 エピソード）。括弧内は HNS。

※ Run ログから復元した 45.7M の値。メトリクスに残る最良（27,122、40.6M）で計算すると Atari-5 回帰は 450.8。

同じ 50M では実時間優先の 2 seed の両方を上回る。 実時間優先の seed 間の幅（370.7〜420.7）は 50 あり、 サンプル効率優先はその上端より 37.6 高い。実時間優先 seed 2 の 100M とはほぼ同じ（458.3 対 465.5）。

### 探索ブロック 02: ゲーム別の中身と内部指標

RAM メトリクスでは 45.7M の断面で、1 ゲームあたり母艦以外 130.4 + 母艦 32.0 = 162.4 ウェーブを抜けている。 1 ウェーブあたり約 2,460 点で、40.6M（153.4 ウェーブ、約 2,430 点）とほぼ同じ。得点の伸びは抜けたウェーブの数の伸びである。

5.3M の打ち切り組は得点が約 27,000 で、稼がずに生き残る状態だった。10.3M では 0 本に戻る。

BTR に届かない 2 本はどちらも、train の得点が lane の ε で大きく変わるゲームである。qbert では高 ε の lane が深い面へ行けず、 battle_zone では高 ε の lane が 10 万点に届かない。特徴層の痩せは battle_zone と double_dunk に共通で、成績とは連動しない。

### 探索ブロック 03: qbert の ε ラダーを 1e-4〜0.05 へ

第 8 面の突破が 41.5M から出始める。 突破は「`round_clear` = 8 かつ 29,000 点以上」で数えた（下の RAM の項を参照）。 ε 群では 29,000 点以上のゲームがすべて `round_clear` = 8 だった。

`round_clear` = 7 のゲームの得点は ε 群で最高 28,950（第 8 面の奥まで進んで終わったゲーム）。 第 8 面に入ってすぐ終わったゲームだけが 8 を返す機序は確認していない。

効いたのは、同じ ε での群前ではなく、低 ε の lane の割合である。ε 帯ごとの到達率は旧ラダーと変わらず、低 ε の lane を増やした分だけ 第 8 面の経験が replay に入り、41.5M から突破が出始めた。探索ブロック 02 の「高 ε の lane が深い面の経験を飢えさせる」を支持する。 代わりに探索が減った序盤は遅く、5.3M の ε=0 断面は旧ラダーの 60%。

### pending

qbert と battle_zone の続き（ε 群の seed 2・3、BTR 型の ε の経過、env 数と replay 容量、battle_zone の ε ラダー群）の結果は 2026-09-27 にある。

qbert の ε 群は、train の突破が 49-50M で 13.3% に上がったところで 50M が終わった。突破が定着するかは判定できていない。 ε=0 断面は 45.7M で 23,900 に下がっている。1 seed。

qbert の `round_clear` が第 8 面の入口で 8 を返す機序は未確認。

## 主要な概念

- [Atari横断](../../../../concepts/subjects/atari.md)
- [Qbert](../../../../concepts/subjects/qbert.md)
- [Battle Zone](../../../../concepts/subjects/battle_zone.md)
- [Phoenix](../../../../concepts/subjects/phoenix.md)
- [Name This Game](../../../../concepts/subjects/name_this_game.md)
- [Double Dunk](../../../../concepts/subjects/double_dunk.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
