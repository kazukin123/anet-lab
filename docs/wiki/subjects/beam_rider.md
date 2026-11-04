# BeamRider

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

RAM の 0x80 はセクター番号、0x94 はボスを魚雷で倒したフレームだけ 141 になり、0xD3 は魚雷の残りだった（エミュレータ調査）。ボスを倒すと残機ぶんのボーナスが入り、その間は残機が減って見える。

サンプル効率優先の 50M では、設定の差が seed だけのスキップなし 2 本の Greedy（35.55〜45.65M の 3 回平均）が 22,914 と 85,872 に割れ、割れは 3〜5M に生じた。低い側はセクター 5〜14 で終わる率が高かった。高い点の Run は、ボスを倒す形（スキップなし s2）と、倒さずにセクターを進む形（補助学習のみ s1）の 2 通りだった。TempoRL の結果は[行動の時間尺度](../mechanisms/temporal_abstraction.md)にまとめた。

## 分かっていること

- サンプル効率優先の 50M で、設定の差が seed と Run 名だけのスキップなし 2 本の Greedy の 35.55〜45.65M の 3 回平均は 22,914 と 85,872 だった（同じ重みの single は 23,626 と 84,636）。セクターは 12.1 と 24.8、ボスは 0.8 と 19.7（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、現時点のまとめ1・探索ブロック01。Run: run_20261031-232353_a5_br_aln_btreps_s1 / run_20261103-080630_a5_br_aln_btreps_s2）。
- 45.65M の Greedy で、スキップなし s2 はセクター 28.5・ボス 22.8 で 104,250、補助学習のみ s1 はセクター 35.0・ボス 0.0 で 106,219 だった。ボスの回数が増え始めた学習側の窓は、スキップなし s2 が 25〜27.5M、s1 が 45〜47.5M で、補助学習のみは 50M まで増えなかった（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、現時点のまとめ4・探索ブロック01・03。Run: run_20261103-080630_a5_br_aln_btreps_s2 / run_20261103-220328_a5_br_aln_t104ac_s1）。
- スキップなしの学習側の平均点は 3.0〜3.5M の窓まで 2 本とも約 10k で、4.0〜4.5M の窓で s1 は 7,845、s2 は 15,416 だった。3〜5M の眠ったユニットの率・Q・loss・勾配ノルム・エントロピー・churn に、この割れに対応する差は無かった（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、現時点のまとめ5・探索ブロック04）。
- 35M 以降の Greedy で、そのセクターの帯に届いたゲームのうちその帯で終わった率は、セクター 5〜9 でスキップなし s1 が 43%、s2 が 4%、補助学習のみ s1 が 2%、10〜14 で 39%、9%、6% だった（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、現時点のまとめ6・探索ブロック05）。
- ボス 1 回の点の推定は約 1,400〜1,600 点だった。スキップなし s2 のボス 0 回のゲームに当てた得点の曲線との差を、ボスの回数で割った値で、ボス 5 回以上の帯で 1,364〜1,640。曲線のセクター 25 での傾きは 3,336 点である（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、探索ブロック05）。
- 公開の `results.csv` の窓平均は 141〜160M フレームで 57,361、161〜180M で 76,001、181〜199M で 87,561 だった。手元で流した BTR の公開実装（評価 10 ゲーム）は同じ窓で 21,861、16,341、17,085、セクターは 11.8、9.4、9.8（181〜200M）だった（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、現時点のまとめ7・探索ブロック02）。
- 40〜50M の Q の最大の平均は、single の Greedy の 40.6・45.65M のクリップ後の報酬の密度（報酬の合計 ÷ step）に 1/(1−γ)=333 を掛けた値に近かった（スキップなし s1 9.34 対 9.22、s2 11.3 対 11.13、補助学習のみ s1 9.50 対 9.37、合算 s1 9.17 対 9.17）。1 セクターは 584〜687 step だった（出典: [2026-11-04_temporl-freeway-beamrider.md](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)、探索ブロック05・06）。
- RAM 0x80 はセクター番号（1 始まり）で、セクターが変わるフレームで 1 ずつ増え、ランダムな行動の 12 ゲームでは死亡とゲームオーバーで増えなかった。15 機を倒しきったあとのボスの場面で死ぬと、復帰と同じフレームで増えた。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0xD4 はそのセクターで残っている敵の数で、セクターの開始で 15、白い円盤を倒すたびに 1 減り、15 機目のあと 0 → 255 になった。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0x94 は、ボスを魚雷で倒したフレームで 1 回だけ 141 になった（セクター 2・3 で 1 回ずつ、得点は 360・390）。倒さない枝と倒していない 2 万フレームでは 141 に入らず、ゲームオーバーでは 140 で止まった。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0xD3 は魚雷の残り（セクターの開始で 3）で、魚雷の行動のフレームだけ 1 減り、死亡では減らなかった。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- ボスを倒すと残機の数だけボーナス（セクター 2 は 120 点、3 は 130 点）が 1 回ずつ入り、そのたびに残機の 0x85 が 1 減って最後に戻った。ALE の `lives()` も 3 → 1 → 3 と動くので、episodic_life を有効にしたプロファイルではボスを倒すたびに残機を失った扱いになる。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- 既定設定の RAM 定義は、1 = `sector_clear`（0x80 inc_count）、2 = `boss_kill`（0x94 reach_count:141）、4 = `torpedo_fired`（0xD3 dec_count）である。ボスは魚雷 1 発で倒れるので 3 は置かない。（出典: [220_atari_env.jp.md](../../design/220_atari_env.jp.md)、§4.9 RAM メトリクス。[082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、D21・Context「beam_rider と freeway の RAM」）

## 分かっていないこと

- セクター 1 と、セクター 4 より先でも、ボスを倒したときに 0x94 が 1 回だけ 141 になるか。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、既知の限界と後続「beam_rider の確認範囲」）

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [220 §4.9 RAM メトリクス](../../design/220_atari_env.jp.md)。
- [PRD082](../../memo/done/082_atari_ram_progress_metrics_10prd.md)。
- [行動の時間尺度](../mechanisms/temporal_abstraction.md)。
- [BeamRider と Freeway での TempoRL の記録](../../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md)。
