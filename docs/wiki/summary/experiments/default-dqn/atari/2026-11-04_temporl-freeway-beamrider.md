# BeamRider と Freeway での TempoRL（PRD103 の再開条件）

## 元の材料

[BeamRider と Freeway での TempoRL（PRD103 の再開条件）](../../../../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md) blob:2fe12d4034fdd5a1ec07cdcfb8025da1d5a3126b

## 要点

- BeamRider のスキップなしは、設定の差が seed と Run 名だけの 2 本で、Greedy の 35.55〜45.65M の 3 回平均が 22,914 と 85,872 に割れた。
- BeamRider の合算 s1 の full は 9,854 で、スキップなしの 2 本のどちらより低かった。
- BeamRider の補助学習のみ s1 は 79,799（single 82,482）で、スキップなし s2 と同じ帯だった。
- スキップなしの割れは 3〜5M に生じた。
- 公開の results.csv の窓平均は 141〜160M フレームで 57,361、161〜180M で 76,001、181〜199M で 87,561 だった。
- Freeway は上限（34）付近で並んだ。
- PRD103 の再開条件（合算がスキップなしを Greedy で上回る）に当てると、どちらのゲームでも満たさなかった。

## 内容

### 目的と比較条件

qbert では合算の変種 8 本がどれもスキップなしを下回った（qbert の記録）。別のゲームで同じ構成の TempoRL を流し、PRD103 §8 の再開条件に当てる。

### 探索ブロック01: BeamRider の Greedy

合算 s2 の 15.35M はセクター 6.9 / 12.4、ボス 0.0 / 0.0。45.65M の最高点は、スキップなし s1 が 99,030 / 74,610、s2 が 247,464 / 238,692、補助学習のみが 180,828 / 182,790、合算 s1 が 28,794 / 79,826。

### 探索ブロック02: BeamRider の ε=0.01 と公開値

BTR の値（exp step × 4 = フレーム）。`results.csv` は BTR のリポジトリに同梱された公開値、手元の Run は `BTR_BeamRider200M_1`（公開実装を手元で 200M フレーム流した、評価 10 ゲーム）。手元の公開実装の評価は、評価の番号 125 までが ε=0.01、それより後が ε=0 で、どちらも NoisyNet を切る。

### 探索ブロック03: BeamRider の学習側

合算 s1 の平均スキップ長は、学習側が 5〜10M の 5.04 から 40〜50M の 4.12、Greedy が 5M 以降の窓で 3.35〜4.30 step だった。

### 探索ブロック04: BeamRider のスキップなしの割れ

学習の進みの窓の平均。各欄は「3〜4M / 4〜5M / 40〜50M」。learn step の軸の指標は learn step × 64 を exp step として窓を切った。

### 探索ブロック07: Freeway

合算の平均スキップ長は、学習側が 0〜5M の窓で 5.76〜6.50、5〜10M の窓で 4.70〜5.81、Greedy が 1.21M の 7.49 から 9.29M の 3.69 step だった。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [BeamRider](../../../../concepts/subjects/beam_rider.md)
- [Freeway](../../../../concepts/subjects/freeway.md)
