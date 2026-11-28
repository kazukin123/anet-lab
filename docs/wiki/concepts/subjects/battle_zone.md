---
title: Battle Zone
page_kind: 題材
aliases:
  - BattleZone
  - battle_zone
  - bz
description: 低εラダーのε=0最良は118,388、旧ラダーは115,533だった。64 env・容量1Mの2 seedは40–50M評価が94,621・86,616で、128 env・2Mの106,389を下回った。
updated: 2026-11-26
---
# Battle Zone

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [題材](../../index.md#題材)

## 今の結論

低εラダーのε=0最良は118,388、旧ラダーは115,533だった。64 env・容量1Mの2 seedは40–50M評価が94,621・86,616で、128 env・2Mの106,389を下回った。

## 分かっていること

### anet の知見

- サンプル効率優先の序盤の速さは後半に続かず、特徴の痩せだけでは頭打ちの原因を確定できない。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ7〜9。Run: run_20260926-082748_a5_battle_zone_btr50m）。
- サンプル効率優先の ε=0 断面は、BTR の公開 CSV の同じ step に対して 67〜94% で、BTR 自身も 25M 以降は 12〜15 万で横ばいだった。25M 以降の train の得点は lane の ε で大きく変わり（ε 0.01〜0.016 の lane は平均 97,023、ε 0.25〜0.4 は 35,340）、ε 0.1 以上の lane が train のゲームの 44% を占めた（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ8・探索ブロック02「battle_zone」。Run: `run_20260926-082748_a5_battle_zone_btr50m`）。
- εラダーを0.01〜0.4から1e-4〜0.05へ下げた50M比較では、ε=0の最良は115,533→118,388、40–50Mのε=0.01評価は97,553→106,389だった。低ε側は序盤が遅く、10.3Mのε=0は59,858→35,590だった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ3、探索ブロック02。Run: `run_20260926-082748_a5_battle_zone_btr50m`, `run_20260928-000617_a5_battle_zone_btr50m_eps005_end0`）。
- 64 env・容量1Mではseed1のε=0が25.5Mの146,203から45.7Mの97,236へ低下し、seed2の最良も113,299だった。40–50Mのε=0.01評価は94,621 / 86,616で、128 env・2Mの106,389を下回った。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ11、探索ブロック05。Run: `run_20260929-094313_a5_battle_zone_btr50m_eps005_env64`, `run_20260930-184113_a5_battle_zone_btr50m_eps005_env64_seed2`, `run_20260928-000617_a5_battle_zone_btr50m_eps005_end0`）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、eval2 40-50M は 53,335（基準 seed 1 / 2 は 107,497 / 135,890）だった。NoisyNet をラダーに足すと 100,955 まで戻ったが、eval2 が上がり始めるのは 15-20M で、基準（5-10M）より約 10M 遅かった（出典: [2026-10-11_noisynet-realtime.md](../../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・3、探索ブロック 01・02。Run: `run_20261012-171923_a5_battle_zone_rr1_50m_nz_eps0`, `run_20261015-030627_a5_battle_zone_rr1_50m_nz_ladder`）。
- 実時間優先の 40〜50M の HNS は seed 2 が seed 1 を 27% 上回ったが、40.6M の Greedy 評価では 6.2% 下だった。seed 2 は 40.6M から 50.7M の間で 97,500 から 146,960 点へ伸び、どの窓を比べるかで差の向きが変わった（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）。
- 記録は、battle_zone では 64 env・容量 1M の効果は見えないとまとめた。seed 2 は 128 env・2M と同じ推移で、旧ラダーの 128 env・2M の 40–50M の ε=0.01 評価は 97,553 だった（出典: [2026-09-27_btr-gap-qbert-bz.md](../../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ 11・探索ブロック 05）。

### 外の事例

- 調査が BTR の公開 CSV から計算した BattleZone の値は、max（評価 200 回）151,877、max（評価 10 回）145,903、最後の 10 回の平均 135,470 で、論文の Table A2 は 168,340 である（出典: [few_seed_comparison_reporting_survey_2026-11-04.md](../../../../reports/survey/few_seed_comparison_reporting_survey_2026-11-04.md)、4.3）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [atari](../subjects/atari.md#横断的知見)（Atari-5 の BTR 比の残りの差（qbert は後の Run で埋まり、battle_zone は残った））。
- [evaluation](../measurement/evaluation.md#見立て)（battle_zone の BTR 比 68% を Run ごとのぶれと比べる統計量の違いで読む見立て）。
