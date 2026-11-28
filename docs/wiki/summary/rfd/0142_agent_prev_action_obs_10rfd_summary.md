---
title: 直前に実行した行動を観測へ加える
page_kind: サマリ
material: "[[docs/memo/0142_agent_prev_action_obs_10rfd.md|0142_agent_prev_action_obs_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# 直前に実行した行動を観測へ加える

## 元の材料

[RFD0142　直前に実行した行動を観測へ加える](../../../memo/0142_agent_prev_action_obs_10rfd.md) blob:79ffc2db8a72bfa111eb34fb1259a4cc4377683c

## 要点

- 直前の行動を利用したい Agent に、環境の実装に依存しない観測キーを用意する案である。
- 選ばれた提案と、手動入力などで実際に実行された行動を区別し、後者を参照する。
- 各 lane の1フレームへ行動数と同じ長さの float32 の one-hot を独立キーとして加える。
- frame stack と Replay 保存より前で加え、Actor と学習で同じ意味の観測を使えるようにする。
- 設定を有効にすることと、Network をその入力へ接続することは別に指定する。

## 内容

### Context

032: LunarLanderEnv が 8 次元の `vector` に one-hot(4) を連結する（`E1.obs_include_action = true`）。
本 RFD は env に依存しない恒久版である。env ごとに同じ機構を書き足さずに、任意の離散行動 env（Atari など）で使えるようにする。
`CheckSameStateActionSpec` や他の Agent（MuZero / ImageCls）へ影響しない。

### 確定した設計判断

配置は観測パイプラインのフレーム段。 frame stack の前、ReplayBuffer へ保存する前に、lane の 1 frame へキーを足す。
記録するのは Actor の出力。 強制行動は `MakeAction` の入力として Actor の出力に反映されるので、出力 = 実行された行動である。
ReplayBuffer は obs の storage を最初の `Push` の dict から確保するので、変更が要らない。

### 仕様

入力 dict が既に `prev_action` を持つ場合は `ANET_SYSTEM_ERROR`。
明示の `Reset()` は持たない。lane の初期化は `episode_start` のマスクが担う（stacker と同じ）。
`next_state.obs` が dummy slot へ保存されるので a_t の one-hot になる。
experience に入れる（単位 A）。両側が同じクラスへ同じ行動列と同じ `episode_start` を与えるので、同じキーが出る。

### 非対象

連続行動（構築時エラー。将来は生の行動ベクトルを同じキーで入れる変種）。

### 受け入れ基準

行動選択側の Network 入力と、同じ時刻の ReplayBuffer 保存 frame の `prev_action` が一致する。
Network の入力テンソルが同じになる。同 seed で主要な学習 scalar（loss、q_max 系、評価の episode return）の checksum が一致することを確認する。
連続行動、env に既存の `prev_action` キー、lane 初回の `episode_start = false` は `ANET_SYSTEM_ERROR` になる。

### テスト項目

契約違反: 初回に `episode_start = false` の lane がある、入力 dict が既に `prev_action` を持つ。
normalizer: dynamic 設定でも `prev_action` に normalizer が作られない（stack なしの構成で確認する）。
`stack_keys = vector`: 行動選択 `[N, A]`、Sample `[B, A]` で forward が通る（キーごとに `Flatten` する NN 構成）。

### 実装対象

`core/anet-core/src/rl.cpp` — `ToUnifiedObservation` で `kVector` を先頭に置く。

### 検証

等価 Run は 083 の実装メモ と同じ全点 checksum の手順で比較する。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
