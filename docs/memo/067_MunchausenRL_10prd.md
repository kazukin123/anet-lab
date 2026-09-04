# Munchausen RL (M-DQN / M-IQN) 暫定 PRD

> 番号 999。骨子のみ。設計・実装方針は未着手。

- M-DQN / M-IQN (Vieillard et al. 2020): TD target に scaled log-policy 項 `α·τ·log π` を加算する既存 DQN 系の拡張。
- 変更は learner の target 計算に局所。IQN と直交し併用可(M-IQN は Atari で IQN 超えを報告)。
- 既存の `quantile_mode` 資産にそのまま乗るため実装コストは小さい見込み。
