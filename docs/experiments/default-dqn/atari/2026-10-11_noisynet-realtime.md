# NoisyNet の実時間優先プロファイルへの投入（Atari） 探索履歴

## メタデータ

| 項目 | 内容 |
|---|---|
| 対象 Agent | DefaultDQNAgent（IQN + Munchausen） |
| 対象 Env | AtariEnv（ALE v0.12.0 直結）。Breakout、Atari-5（battle_zone / double_dunk / name_this_game / phoenix / qbert）、kung_fu_master |
| 開始日 | 2026-10-11 |
| 主目的 | NoisyNet（[PRD 084](../../../memo/084_noisynet_10prd.md)）を実時間優先プロファイルの 50M に入れ、train の ε ラダーを NoisyNet に置き換えた腕（Noisy だけ）と、ラダーに NoisyNet を足した腕（Noisy + ラダー）を基準と比べる |
| 基準 | 実時間優先（[2026-09-21](2026-09-21_realtime-profile-atari5.md)）の seed 1（50M）と seed 2（100M の 0-50M）。Breakout は `run_20260919-235619_rr1_va_hard125_cap2m_100m`（同じ構成の 100M）の 0-50M。kung_fu_master は [2026-09-23](2026-09-23_realtime-profile-100m.md) の seed 1・2 と seed 3 |
| 判定指標 | eval2（ε=0.01、μ-only、`52_eval2/10_game_score_mean`）の 40-50M 平均。Breakout は 25-50M 平均と、学習中のゲームのうち 432 点以上の割合（`42_env/40_game_score_ge432`）も見る。ε=0 断面（`53_evalg/10_game_score_mean`、μ-only） |
| 比較上の注意 | NoisyNet の Run はすべて `run.seed` = 1 で、同じ実行環境で回した。基準は別の実行環境・前の実行ファイルなので、実時間は比べない。各ゲームとも 1 seed。ε=0 断面の間隔は、基準の seed 2・Breakout の基準・kung_fu_master の基準が 10.1M、ほかは 5.05M。非決定論 backend |

## 現時点のまとめ

1. **Noisy だけ（train の ε 0）の 7 本は、どのゲームでも基準のどの seed も下回った。** eval2 40-50M で Breakout 470.1（基準 546.0）、qbert 12,144（20,650 / 20,103）、
   phoenix 62,951（114,159 / 86,981）、battle_zone 53,335（107,497 / 135,890）、double_dunk −23.9（17.1 / 20.5）、
   name_this_game 20,540（22,903 / 21,544）、kung_fu_master 36,431（基準 3 seed 61,757 / 37,892 / 76,604）（探索ブロック 01）。

2. **Breakout で下がったのは ε ラダーを外したぶんで、NoisyNet 自体は ±0 だった。** 25-50M の eval2 は、基準 517.1、Noisy + ラダー 532.4、Noisy だけ 448.6、ε 0・NoisyNet なし 447.8。
   ラダーの有無で約 70 点違い、NoisyNet の有無ではラダーありで +15、ラダーなしで +1（探索ブロック 02）。

3. **battle_zone と double_dunk では、Noisy + ラダーで成績は戻ったが、立ち上がりが 10〜20M 遅れた。** battle_zone は eval2 が上がり始めるのが 15-20M（基準は 5-10M）で、40-50M は 100,955。
   double_dunk は eval2 が正になるのが 40-45M（基準は 20-25M）で、40-50M は 11.5、45-50M は 15.3（探索ブロック 02）。

4. **double_dunk の Noisy だけは、50M まで相転移が起きなかった。** eval2 は 5M 以降 −23.9 前後、ε=0 断面は −0.7〜−1.0 のまま。
   ノイズで行動が変わる割合（probe）は、5M 以降 0.0% だった（探索ブロック 01・03）。

5. **ノイズで行動が変わる割合は、学習の初めに大きく、数 M のうちに数 % まで下がった。** 学習開始時（learn step 0）は 27〜71%、40-50M の平均は 0.0〜4.9%（Noisy だけ）。
   readout weight の ‖σ‖/‖μ‖ は 0.869 から、40-50M の平均で 0.48〜0.81 になった（探索ブロック 03）。

6. **kung_fu_master の Noisy だけは、序盤は速いがそのまま横ばいになり、面を 1 度も抜けなかった。** eval2 5-10M は 31,447（基準 12,141〜21,347）、
   10M 以降は 34,089〜37,596 の範囲で、50M までの学習中のゲームで面クリアは 0（探索ブロック 01）。

## 対象 Run

### NoisyNet の設定

NoisyNet の腕に共通の実効設定:

- `DefaultDQNAgent.head_projection.noisy.enabled` = true、`sigma0` = 1.2
- `net.block.[AtariHeadFC512].noisy.enabled` = true、`sigma0` = 0.87
- `actor.[train]`・`learner.current`・`learner.target` の `nn_forward_key` = `act_sample`、`actor.[eval]`・`actor.[greedy]` は `mu_only`
- `learner.noisy_weight_decay` = 0

腕の train の探索:

| 腕 | NoisyNet | train の ε |
|---|---|---|
| Noisy だけ | あり | 0（`eps_start` = `eps_end` = 0、spatial exploration なし。時間で下げる ε も無い） |
| Noisy + ラダー | あり | 基準と同じラダー 0.4〜0.01（spatial、lane 固定） |
| ε 0・NoisyNet なし | なし | 0（Noisy だけと同じ） |

基準（ラダーあり・NoisyNet なし）との実効 config の差は、表の探索の設定のほかは、基準が 100M の Run のときの `per_beta_step` と終了 step だけ（β は 0.2 で固定なので効かない）。

### Run

| Run | ゲーム | 腕 | 到達 / 実時間 |
|---|---|---|---|
| `run_20261011-182123_breakout_rr1_50m_nz_eps0` | breakout | Noisy だけ | 50M / 7.49h |
| `run_20261012-015046_a5_qbert_rr1_50m_nz_eps0` | qbert | Noisy だけ | 50M / 7.27h |
| `run_20261012-090712_a5_phoenix_rr1_50m_nz_eps0` | phoenix | Noisy だけ | 50M / 8.20h |
| `run_20261012-171923_a5_battle_zone_rr1_50m_nz_eps0` | battle_zone | Noisy だけ | 50M / 8.02h |
| `run_20261013-012055_a5_double_dunk_rr1_50m_nz_eps0` | double_dunk | Noisy だけ | 50M / 8.94h |
| `run_20261013-102213_a5_name_this_game_rr1_50m_nz_eps0` | name_this_game | Noisy だけ | 50M / 7.86h |
| `run_20261013-181350_kfm_rr1_50m_nz_eps0` | kung_fu_master | Noisy だけ | 50M / 7.69h |
| `run_20261014-015532_breakout_rr1_50m_nz_ladder` | breakout | Noisy + ラダー | 23.7M で中断 / 3.58h |
| `run_20261014-121541_breakout_rr1_50m_nz_ladder` | breakout | Noisy + ラダー（上の Run の流し直し） | 50M / 7.57h |
| `run_20261014-194942_breakout_rr1_50m_eps0` | breakout | ε 0・NoisyNet なし | 50M / 7.28h |
| `run_20261015-030627_a5_battle_zone_rr1_50m_nz_ladder` | battle_zone | Noisy + ラダー | 50M / 7.89h |
| `run_20261015-110003_a5_double_dunk_rr1_50m_nz_ladder` | double_dunk | Noisy + ラダー | 50M / 8.32h |

`run_20261014-015532_breakout_rr1_50m_nz_ladder` は 23.7M で外から閉じられた（ログに学習停止の要求が無く、ウィンドウを閉じた記録だけがある）。
checkpoint からの再開は replay・step の系列・評価の予定を戻さないので、同じ設定を最初から流し直した。

### 基準

| Run | ゲーム | seed | 記録 |
|---|---|---:|---|
| `run_20260919-235619_rr1_va_hard125_cap2m_100m` | breakout | 1 | 100M の 0-50M を使う |
| `run_20260921-155634_a5_qbert_rr1_50m` ほか 5 本 | Atari-5 | 1 | [2026-09-21](2026-09-21_realtime-profile-atari5.md) 探索ブロック 01 |
| `run_20260923-163113_a5_qbert_rr1_100m_seed2` ほか 5 本 | Atari-5 | 2 | 同 探索ブロック 04（0-50M を使う） |
| `run_20260923-020945_atari_kung_fu_master`、`run_20260923-095708_atari_kung_fu_master`、`run_20260925-130017_atari_kfm_rr1_100m_seed3` | kung_fu_master | 1 / 2 / 3 | [2026-09-23](2026-09-23_realtime-profile-100m.md)、[2026-10-15](2026-10-15_kfm-boss-route.md) |

## 探索ブロック 01: Noisy だけ（ε 0）× 7 ゲーム

### 探索観点

NoisyNet に探索を任せ、train の ε を 0 にする。BTR は 25M 以降 ε を切って NoisyNet だけで回す（[2026-09-27](2026-09-27_btr-gap-qbert-bz.md) 探索ブロック 01）。

### 探索結果

eval2 の窓平均（Breakout だけ 25-50M も並べる）:

| ゲーム | Noisy だけ 40-50M | 基準 seed 1 | 基準 seed 2 | Noisy だけ 25-50M | 基準 seed 1 25-50M |
|---|---:|---:|---:|---:|---:|
| breakout | 470.1 | 546.0 | — | 448.6 | 517.1 |
| qbert | 12,144 | 20,650 | 20,103 | 10,763 | 17,443 |
| phoenix | 62,951 | 114,159 | 86,981 | 58,922 | 106,247 |
| battle_zone | 53,335 | 107,497 | 135,890 | 45,794 | 96,837 |
| double_dunk | −23.9 | 17.1 | 20.5 | −23.9 | 14.9 |
| name_this_game | 20,540 | 22,903 | 21,544 | 19,160 | 22,348 |
| kung_fu_master | 36,431 | 61,757 | 37,892 | 35,982 | 52,376 |

kung_fu_master の基準 seed 3 は、40-50M が 76,604、25-50M が 61,722。

- **double_dunk は相転移が起きなかった。** eval2 は 5-10M 以降すべての 5M 窓で −23.9〜−24.0、ε=0 断面は 10 点とも −0.7〜−1.0。
  学習中のスコアは 10M 以降 −0.8〜−0.9 で、基準 seed 1 の ε=0 断面が 25.5M に 10.8 へ跳ねた相転移が無い。
- **battle_zone は立ち上がりが遅く、そのまま差が残った。** eval2 の 5M 窓は 0-15M に 2,310〜5,810 で、基準 seed 1（8,510〜54,520）より遅く、15-20M から上がった。
- **name_this_game は序盤が遅く、後半に近づいた。** eval2 の 5M 窓は 0-25M に 1,913〜8,034（基準 seed 1 は 7,082〜20,331）、30-35M に 19,076（同 22,107）、45-50M に 20,884（同 23,039）。
- **qbert は第 5 面より先へ進まなかった。** 学習中のゲーム（54,620 本）と ε=0 断面の `round_clear` は最大 4 で、学習中の最高点は 17,100。
- **kung_fu_master は序盤だけ速い。** eval2 5-10M は 31,447（基準 3 seed は 12,141 / 21,347 / 19,497）、学習中のスコアは 5-10M から 45-50M まで 42,542〜44,248 で横ばい。
  ε=0 断面は 5.3M から 45.7M まで 38,666〜44,468。学習中のゲームで、ボスへの命中・撃破・面クリアは 50M まで 0 本だった。

## 探索ブロック 02: ε ラダーの有無と NoisyNet の有無を分ける

### 探索観点

探索ブロック 01 の Noisy だけは、「NoisyNet を足す」と「ε ラダーを外す」の 2 つを同時に変えている。
Breakout で 2 × 2 の残りの 2 マス（Noisy + ラダー、ε 0・NoisyNet なし）を回し、差の大きかった battle_zone と double_dunk には Noisy + ラダーを回す。

### 探索結果

**Breakout（2 × 2）:**

| 腕 | eval2 25-50M | eval2 40-50M | ε=0 断面 25-50M の平均 | 学習中の 432 点以上（25-50M） |
|---|---:|---:|---:|---:|
| 基準（ラダー、NoisyNet なし） | 517.1 | 546.0 | 570.5（2 断面） | 12.0% |
| Noisy + ラダー | 532.4 | 553.3 | 567.7（5 断面） | 12.7% |
| Noisy だけ（ε 0） | 448.6 | 470.1 | 509.5（5 断面） | 45.6% |
| ε 0・NoisyNet なし | 447.8 | 464.7 | 512.1（5 断面） | 48.6% |

- ラダーを外すと、NoisyNet の有無によらず eval2 25-50M が約 70〜85 点下がった。NoisyNet の有無の差は、ラダーありで +15.3、ラダーなしで +0.8。
- ε 0 の 2 腕の学習中の 432 点以上の割合は、探索のランダム行動が無いので、ラダーありの腕と比べられない。
- 23.7M で中断した Noisy + ラダーは、流し直しと各 5M 窓の eval2 が近い（0-5M 43.2 / 58.9、5-10M 360.7 / 357.8、10-15M 427.3 / 446.1、15-20M 453.2 / 458.5、20-25M 520.1 / 510.8）。

**battle_zone と double_dunk（Noisy + ラダー）:** eval2 の 5M 窓。

| ゲーム / 腕 | 0-5M | 5-10M | 10-15M | 15-20M | 20-25M | 25-30M | 30-35M | 35-40M | 40-45M | 45-50M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| battle_zone 基準 seed 1 | 8,510 | 33,945 | 54,520 | 65,205 | 79,570 | 88,470 | 87,955 | 93,300 | 104,460 | 110,695 |
| battle_zone 基準 seed 2 | 13,750 | 35,550 | 57,260 | 57,240 | 72,170 | 75,950 | 102,130 | 106,610 | 131,560 | 140,220 |
| battle_zone Noisy + ラダー | 3,090 | 3,900 | 7,470 | 31,690 | 58,880 | 71,980 | 84,110 | 89,220 | 98,080 | 103,830 |
| battle_zone Noisy だけ | 2,310 | 3,170 | 5,810 | 19,610 | 30,820 | 39,080 | 38,720 | 44,500 | 49,240 | 57,430 |
| double_dunk 基準 seed 1 | −22.6 | −19.5 | −17.3 | −9.3 | 2.5 | 11.3 | 14.0 | 15.3 | 16.6 | 17.6 |
| double_dunk 基準 seed 2 | −21.8 | −18.8 | −15.0 | −1.2 | 14.1 | 17.9 | 18.2 | 19.4 | 20.1 | 20.9 |
| double_dunk Noisy + ラダー | −23.3 | −23.8 | −21.8 | −20.1 | −19.0 | −19.4 | −11.8 | −4.4 | 7.6 | 15.3 |
| double_dunk Noisy だけ | −23.1 | −23.9 | −23.9 | −23.9 | −23.9 | −23.9 | −24.0 | −23.9 | −23.9 | −23.9 |

- **battle_zone の Noisy + ラダーは、立ち上がりが基準より約 10M 遅れ、50M で基準 seed 1 の少し下まで来た。** 40-50M は 100,955（基準 107,497 / 135,890、Noisy だけ 53,335）。
- **double_dunk の Noisy + ラダーは相転移が戻ったが、約 20M 遅れた。** eval2 が正になったのは 40-45M（基準は 20-25M）。
  ε=0 断面は 25.5M −0.6、30.5M −16.1、35.6M −9.1、40.6M 4.3、45.7M 18.5（基準 seed 1 の 45.7M は 17.3）。

## 探索ブロック 03: NoisyNet の probe

### 探索観点

`47_agent_noisy/10_action_disagreement_ratio`（ノイズだけを変えたときに greedy 行動が μ-only と食い違う割合）と、
`07_sigma_l2_ratio_readout_weight`（readout 群の weight の ‖σ‖/‖μ‖）の推移を見る。どちらも learn step の軸に載るので、
exp step ≈ 200k + 256 × learn step で読み替えた窓の平均を並べる。

### 探索結果

行動の食い違いの割合（%）:

| Run | 学習開始時 | 0.2-1M | 1-2M | 2-4M | 4-5M | 5-15M | 15-25M | 40-50M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| breakout Noisy だけ | 70.4 | 34.8 | 9.7 | 4.4 | 3.2 | 2.9 | 2.4 | 2.3 |
| qbert Noisy だけ | 40.2 | 17.1 | 7.3 | 8.3 | 2.9 | 3.3 | 3.8 | 3.9 |
| phoenix Noisy だけ | 70.9 | 52.4 | 18.3 | 16.6 | 13.7 | 4.4 | 3.5 | 4.9 |
| battle_zone Noisy だけ | 55.6 | 61.4 | 64.1 | 50.2 | 44.1 | 12.0 | 2.6 | 1.4 |
| double_dunk Noisy だけ | 62.4 | 67.4 | 62.3 | 46.1 | 33.0 | 0.0 | 0.0 | 0.0 |
| name_this_game Noisy だけ | 27.2 | 17.1 | 0.4 | 2.5 | 1.2 | 0.7 | 1.9 | 1.8 |
| kung_fu_master Noisy だけ | 51.0 | 35.8 | 28.5 | 12.4 | 3.6 | 6.2 | 5.2 | 4.8 |
| breakout Noisy + ラダー | 70.0 | 31.9 | 11.5 | 5.5 | 6.4 | 3.4 | 3.4 | 3.9 |
| battle_zone Noisy + ラダー | 55.8 | 62.0 | 56.2 | 40.9 | 31.2 | 15.2 | 3.8 | 2.0 |
| double_dunk Noisy + ラダー | 62.2 | 72.1 | 55.5 | 34.0 | 23.4 | 12.0 | 11.2 | 9.2 |

readout weight の ‖σ‖/‖μ‖ は全 Run とも学習開始時 0.869 で、40-50M の平均は breakout 0.572、qbert 0.480、phoenix 0.618、battle_zone 0.779、
double_dunk 0.810、name_this_game 0.660、kung_fu_master 0.726（以上 Noisy だけ）、Noisy + ラダーは breakout 0.593、battle_zone 0.776、double_dunk 0.711。

- **ノイズが行動を変える割合は、breakout・qbert・name_this_game では 1-2M に 10% 以下、phoenix・kung_fu_master では 4-5M に 14% 以下へ下がり、5M 以降はどれも数 % になった。**
  battle_zone と double_dunk は 4-5M にも 23〜44% が残った。
- **double_dunk の Noisy だけは、5M 以降ノイズが行動を 1 度も変えなかった**（5-15M・15-25M・40-50M とも 0.0%）。‖σ‖/‖μ‖ は 0.81 で、7 本の中で最も下がっていない。
  Noisy + ラダーの double_dunk は 40-50M も 9.2% が残った。

## pending

- battle_zone と double_dunk で、Noisy + ラダーの立ち上がりが基準より遅れる理由。
- ε ラダーを外した損と NoisyNet の効きを分けたのは Breakout だけで、ほかのゲームでは分けていない（ε 0・NoisyNet なしの腕は Breakout だけ）。
- double_dunk の Noisy だけで、ノイズが行動を変えなくなったのに ‖σ‖/‖μ‖ が 0.81 に留まる理由。
