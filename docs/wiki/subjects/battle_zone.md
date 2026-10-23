# Battle Zone

## 今の結論

低εラダーのε=0最良は118,388、旧ラダーは115,533だった。64 env・容量1Mの2 seedは40–50M評価が94,621・86,616で、128 env・2Mの106,389を下回った。

## 分かっていること

- サンプル効率優先の序盤の速さは後半に続かず、特徴の痩せだけでは頭打ちの原因を確定できない。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ7〜9。Run: run_20260926-082748_a5_battle_zone_btr50m）。

- εラダーを0.01〜0.4から1e-4〜0.05へ下げた50M比較では、ε=0の最良は115,533→118,388、40–50Mのε=0.01評価は97,553→106,389だった。低ε側は序盤が遅く、10.3Mのε=0は59,858→35,590だった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ3、探索ブロック02。Run: `run_20260926-082748_a5_battle_zone_btr50m`, `run_20260928-000617_a5_battle_zone_btr50m_eps005_end0`）。
- 64 env・容量1Mではseed1のε=0が25.5Mの146,203から45.7Mの97,236へ低下し、seed2の最良も113,299だった。40–50Mのε=0.01評価は94,621 / 86,616で、128 env・2Mの106,389を下回った。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ11、探索ブロック05。Run: `run_20260929-094313_a5_battle_zone_btr50m_eps005_env64`, `run_20260930-184113_a5_battle_zone_btr50m_eps005_env64_seed2`, `run_20260928-000617_a5_battle_zone_btr50m_eps005_end0`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
