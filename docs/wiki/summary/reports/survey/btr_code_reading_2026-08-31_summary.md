---
title: "Code Reading: BTR 公開実装 — PER・ネットワーク結線・評価系と anet-lab 現行差分"
page_kind: サマリ
material: "[[reports/survey/btr_code_reading_2026-08-31.md|btr_code_reading_2026-08-31]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# Code Reading: BTR 公開実装 — PER・ネットワーク結線・評価系と anet-lab 現行差分

## 元の材料

[Code Reading: BTR 公開実装 — PER・ネットワーク結線・評価系と anet-lab 現行差分](../../../../../reports/survey/btr_code_reading_2026-08-31.md) blob:d8e85f5f9c6e557dd76b0c791cbac185796db64a

## 要点

- BTR の新規遷移は過去の最大優先度で追加され、最大値の減衰・再計算はない。
- `PER.__init__` は `beta=0.4` を受け取るが（`PER.py:86,118`）、`append_pointer` の while ループ末尾で毎回 0 に潰される。
- 実装の replay ratio は環境数と batch size も合わせて初めて1遷移あたりの抽出回数に換算できる。
- 公開 Breakout CSV の各列は100エピソードの平均である。
- Breakout は1系列のみで、その CSV から seed 間の分散は計算できない。

## 内容

### 1. 調査方法

`results.csv` は 60 ゲーム × 200 列（1M フレームごと、各回 100 エピソード平均）の実測値であり、Python で直接集計した。

### 2. 既存 survey の再確認（新規性なし）

2026-08-26 survey §14 の差分表にあった「BTR の 1/64 は batch 256 ÷ 64 env steps = 4 samples/env-step であり、anet-lab の `replay_ratio = 4` と同義」も、コード側で確認した（§3.4）。

### 3. 新規に判明した点

`max_prio` は `max()` でしか更新されず、減衰も再計算もされない。したがってこれは「現在バッファ内にある最大優先度」ではなく、学習開始以降に一度でも観測された TD 誤差の最大値である。一度大きな値が出れば、以後すべての新規遷移が永久にその水位で入る。

### 4. anet-lab 現行 config との差分（2026-08-31 版）

survey §14 の差分表は Run `run_20260825-183923`（旧バンドル）を右列に取っていた。その後のベースライン修正で複数項目が解消しているため、Run `run_20260831-050214_i_rr4_ln512_breakout` の実行時ダンプで更新する。

### 5. config だけで試せる未検証項目

§4.2 のうち、実装を伴わずに測れるものを影響の大きい順に置く。いずれも 2026-08-31 時点で未検証。

### 6. 再現実験の実行可能性

別 venv（Python 3.11 + torch cu121）の構築が前提で、GPU も占有する。`results.csv` が 60 ゲーム × 200 点の実測を含むため、単純な再現の情報価値は低い。価値があるのは Breakout 単独の ablation（`--munch 0` との対など、論文の ablation は Atari-5 集計のみ）である。

## 主要な概念

- [優先度付き経験再生（PER）](../../../concepts/methods/prioritized_replay.md)
- [Replay ratioとlane窓](../../../concepts/methods/replay_ratio_lane_window.md)
- [τ配置とIQN比較](../../../concepts/methods/tau_sampling.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [324 Agent個別実装 DQN系列](../../../concepts/framework/324_dqn.md)
