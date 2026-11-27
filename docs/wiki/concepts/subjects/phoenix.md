# Phoenix

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [題材](../../index.md#題材)

## 今の結論

サンプル効率優先45.7M・ε=0の最良399,895はBTR Table A2比94%だった。同断面の100本中44本が27,000 step上限で打ち切られた。

## 分かっていること

### anet の知見

- 初回20Mでは長時間生存とscore低下が同時に起き、途中でbad allocationにより終了した。（出典: [2026-08-27_atari5.md](../../../experiments/default-dqn/atari/2026-08-27_atari5.md)、現時点のまとめ6。Run: run_20260827-162911_a5_phoenix_apex）。
- サンプル効率優先の45.7M断面では100本中44本が27,000 stepの上限に届き、上限が成績を抑え始めた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ4。Run: run_20260925-174945_a5_phoenix_btr50m）。
- サンプル効率優先45.7M・ε=0の最良は399,895で、実時間優先の2.2〜2.6倍、BTR Table A2の427,481の94%だった。同断面の100本中44本は27,000 step上限で打ち切られ、打ち切り組の平均は454,277だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ4、探索ブロック02。Run: `run_20260925-174945_a5_phoenix_btr50m`）。
- 45.7M の ε=0 断面では 1 ゲームあたり 162.4 ウェーブを抜けており、1 ウェーブあたり約 2,460 点は 40.6M（約 2,430 点）とほぼ同じで、得点の伸びは抜けたウェーブ数の伸びだった。5.3M の断面では 100 本中 59 本が 27,000 step の上限で打ち切られ、得点は約 27,000 点で、稼がずに生き残る状態だった（10.3M では 0 本）（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、探索ブロック02「phoenix」。Run: `run_20260925-174945_a5_phoenix_btr50m`）。
- 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、eval2 40-50M は 62,951（基準 seed 1 / 2 は 114,159 / 86,981）だった（出典: [2026-10-11_noisynet-realtime.md](../../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ1・探索ブロック 01。Run: `run_20261012-090712_a5_phoenix_rr1_50m_nz_eps0`）。
- 実時間優先 seed 2 の Greedy で得点は 50.7M の 185,402 から 70.9M の 109,522 へ落ち、91.1M に 197,025 へ戻った。1 ウェーブ 1,850〜2,098 点で、得点の変動は突破ウェーブ数と同期した。同じ区間の ε=0.01 の窓平均は単調に増えた（出典: [実時間優先の Atari-5 記録](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）。
- 45.7M の ε=0 の最良 399,895 は、BTR の公開 CSV の最大 367,284 の 109% である（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ 4・探索ブロック 02）。

### 外の事例

- 調査が BTR の公開 CSV から計算した Phoenix の値は、max（評価 200 回）367,284、max（評価 10 回）303,923、最後の 10 回の平均 287,479 である。評価 10 回の取り始めをずらすと、評価 200 回の max との比は 1.00〜1.55（中央値 1.19）だった（出典: [few_seed_comparison_reporting_survey_2026-11-04.md](../../../../reports/few_seed_comparison_reporting_survey_2026-11-04.md)、4.3）。
- BTR の Phoenix の表では、Munchausen を外すと action gap が小さくなった → 詳しくは [Munchausen RL](../methods/munchausen.md)（出典: [btr_hyperparams_survey_2026-08-26.md](../../../../reports/btr_hyperparams_survey_2026-08-26.md)、9.2 Table 2 (Phoenix)）。

## 横断的知見

- 共通点: phoenix で Greedy の得点が伸びるとき、実時間優先・サンプル効率優先の双方で突破ウェーブ数が増えた。条件の違い: 実時間優先の 91.1M では 111 ゲームのうち 97 が上限より前に終了した。サンプル効率優先の 45.7M では 100 ゲーム中 44 が上限に達した（出典: [実時間優先](../../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04、[サンプル効率優先](../../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、phoenix のウェーブと上限）。

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](atari.md)。
- [evaluation](../measurement/evaluation.md)。
