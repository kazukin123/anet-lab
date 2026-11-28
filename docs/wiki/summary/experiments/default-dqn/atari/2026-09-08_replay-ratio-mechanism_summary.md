---
title: "Atari Breakout: 高 replay_ratio で成績が落ちる機序 — 2026-09-08"
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md|default-dqn/atari/2026-09-08_replay-ratio-mechanism]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
updated: 2026-11-28
---
# Atari Breakout: 高 replay_ratio で成績が落ちる機序 — 2026-09-08

## 元の材料

[Atari Breakout: 高 replay_ratio で成績が落ちる機序 — 2026-09-08](../../../../../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md) blob:5109f7fd720a54aa1db406d8248d787daa988264

## 要点

- 犯人は再利用回数で、勾配回数ではない。
- 過学習の署名が出ている。 25-50M で `01_loss` が RR4 系 0.36-0.40 対 RR1/RR2 系 0.56-0.64、 `01_td_mean` が 0.29-0.31 対 0.34-0.37、`04_td_std` が 0.37-0.40 対 0.54-0.59。
- 可塑性は成績を予測しない。上下どちらの方向でも。
- 全層 `spectral_cap` は①（重み成長）を止め、発散しなかった。
- PER の「捨てる」チャネルは RR1 と RR2 の間で閉じる。
- 複製幅は群によって違い、DropPath 群では大きい。
- BTR 公開実装は同じ再利用 4 回で崩れない。

## 内容

### 成績（25-50M プール）

50M に届かなかった 2 本は別窓で見る。同じ窓で他の群と揃えた値である。

### 探索ブロック 01: target 鮮度を勾配時間で揃える（C=125）

それでも壁突破率は 8.85% → 6.83%、eval1 は 461.2 → 430.5。 窓別では 5-10M で C=125 がリードし （5.01% 対 4.53%）、10-20M で並び、20M 以降に離される（20-30M 5.79 対 7.50、40-50M 7.49 対 9.30）。 eval1 は 10M 以降 421 → 430 でほぼ平坦である。

この軸が env 時間で決まることは 2026-09-01 探索ブロック 14 / 21 / 23 と `940_target_update_step_axis_10prd.md` で確定していたが、速すぎる側の実測は本ブロックが初めてである。 遅すぎる側（τ=0.001 = env 256K、3.48%）と合わせて、32K は両側から挟まれた内点最適になった。

### 探索ブロック 02: 全層 `spectral_cap`（capall）

2026-08-30 探索ブロック 01 は「①が止まる範囲と発散しない範囲に両立点が無い」と結論していた。 全層 `spectral` は実効ノルムを平坦化した唯一の例だが `q_max` が 2,700 まで発散し、 BTR 範囲（residual conv のみ）の `spectral` / `spectral_cap` は生ノルムと一緒に伸びて実質 no-op だった。

σ が伸び続ける＝cap が噛み続けて除算量を自動的に強めている。参照値は 2026-08-30 の 全層 `spectral` が比 0.50（勾配 28k、発散直前）、BTR 範囲の cap が 0.93（実質 no-op）。 `11_q_max_mean` は全行程で 9.8 前後、`24_grad_clip_ratio` も平常で、781k 勾配まで発散していない （全層 `spectral` は 28k 勾配で飛んだ）。

### 探索ブロック 03: RR2（`replay_ratio` の山の移動）

差 +2.69pt は RR1 の複製差（0.24pt）の 11 倍である。

2026-09-01 探索ブロック 28 の残件も「Munchausen は暗黙 KL でその保護側に当たるため、 高 RR 側の損傷の壁が後退している可能性がある」としていた。RR4 では成立しなかったが、RR2 で成立した。

### 探索ブロック 04: 再利用と勾配回数の分離（B=512）

`U = num_envs × replay_ratio / replay_batch_size`（`dqn_based_agent.cpp:1939`）なので、 batch を上げると再利用回数を保ったまま勾配回数だけ減らせる。RR4 / B512 は 「RR4 の再利用（×4）・RR2 の勾配頻度（128 exp step / 勾配）」にあたる。

なお `replay_ratio` は環境遷移 1 件あたり平均何サンプルを学習に使うかであって更新回数ではない （`earned_credit_` は train step 1 回につき 1 度加算され、train step は num_envs 件の遷移を進める）。

### 探索ブロック 05: 過学習の署名と PER の退化（横断）

replay バッチへの当てはまりが良い群ほど成績が悪い。 再利用 ×4 の 4 本は誤差 0.29-0.31 まで詰めて 成績は 6.8-8.9%、再利用 ×2 / ×1 は 0.34-0.37 までしか詰めず 11.4-14.3% である。

訓練誤差が下がるのに汎化が落ちるという形なので、 再利用 → replay 分布への過学習 → 汎化低下という読みが成立する。 ただし held-out の TD を測る計器が無い（probe チャネルは RB からの一様抽出で同じ分布）ため、 train / held-out ギャップの直接測定はできていない。

### 探索ブロック 06: 網側の正則化 4 本

天井を動かしたのは DropPath 0.1 だけで、しかも RR2 の 14.33% には届かない。 `per_alpha` を BTR の 0.2 から Schaul 2015 既定の 0.6 へ上げると悪化した。 RR4 では `07_evicted_unsampled_ratio` が 0.00012 でほぼ全件が引かれるため、 優先度を尖らせても「引かれない経験が増える」方向には働かない。

### 探索ブロック 07: DropPath の用量反応

15-17M 窓（DropPath 0.2 が 17.7M で止まったため）。

`weight_norm_feature` は用量に単調減少（RR4 素 96.6 → 0.1 で 94.3 → 0.2 で 92.5、learn 231-262k）、 `11_q_max_mean` も単調減少（10.87 → 10.35 → 9.90）した。ただし `44_probe_srank` は 466 / 472 / 475、 `43_probe_feature_norm` は 11.45 / 10.77 / 11.02 で崩れていない。

### 探索ブロック 08: `46_agent_replay_fit` の初測定

`03` を押し上げるチャネルは記憶だけである。PER 選択（S 群は高 TD が濃い）と年齢 （U 群は必ず若い）は両方とも押し下げに働く。したがって 1 を超えていること自体が記憶の証拠であり、 学習を通じて緩やかに増える。

### 探索ブロック 09: 複製幅の再測定

複製幅は群によって 1 桁違う。 DropPath はサンプル単位で枝を落とすので Run 間分散が増える、が素直な読みである。

### 探索ブロック 10: BTR 公開実装のローカル実行

WSL2 Ubuntu-24.04 上で実行した。Windows では `AsyncVectorEnv(context="spawn")` が env 数だけ Python プロセスを起こすためメモリが足りず、`--envs 64` が使えなかった。WSL で env worker だけ `fork` に変えると 64 env が 1.2 秒で生成・WSL 全体 RSS 3.6GB になる。

Double DQN も grad_clip 10 も weight_decay 0 も一致している。

### 判断

高 replay_ratio の劣化は、同じ遷移を繰り返し学習に入れることによる replay 分布への過学習である。

### 残件

BTR との構成差で RR4 未検証のもの: 6×6 maxpool + 共有トランク除去 + 2304 融合、LayerNorm の有無、 spectral norm（BTR は `--spectral 1`、checkpoint の `parametrizations.weight.original` で確認）、 `train.num_envs` 64、探索方式（BTR は単一 ε、本構成は ε ラダー 0.01-0.4）

### 参照

2026-08-17_baseline.md — 旧バンドルの RR 掃引（探索ブロック 23-25、まとめ 30-38）

## 主要な概念

- [Replay ratioとlane窓](../../../../concepts/methods/replay_ratio_lane_window.md)
- [可塑性指標と成績](../../../../concepts/phenomena/plasticity.md)
- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
- [Breakout](../../../../concepts/subjects/breakout.md)
