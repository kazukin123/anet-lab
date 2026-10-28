# LunarLander

## 今の結論

matched 3-seedのQR/IQN比較ではホバリング離脱時期に明確な差がなかった。持続的T180到達はQR1/3対IQN3/3、終盤reward平均は144.3対186.0だった。TempoRL（分離×合算）は IQN＋Munchausen の 3 seed で無効を上回り、2026-10-26 から LunarLander の既定で有効にした。合算に対する分割・k_max の重み・スキップ長の ε-greedy の 3 腕（3 seed）はどれも基準を上回らず、既定は据え置いた（2026-10-28）。

## 分かっていること

- matched 3-seedのQR/IQN比較ではホバリング離脱の早さに明確な差はなく、離脱後のT180到達と終盤rewardはIQNが有利な可能性を示した。（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ3・9・10）。
- RB満杯付近の変化はcapacity半減で前へ動き、IQN固有より経験分布切替が有力だった。current N削減によるloss低下は集約尺度も変えるため、過学習改善と断定できない。（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ5〜8）。
- LunarLanderEnv の観測（既存 8 次元）には風・乱流が含まれず POMDP で、エージェントはフレームスタック（stack_count=4）の速度履歴の差分から風を推すしかない。速度差分には自機エンジンの推力も混ざる（出典: [032_lunarlander_obs_include_action_10prd.md](../../memo/done/032_lunarlander_obs_include_action_10prd.md)、Context（背景・目的）・確定した設計判断2）。
- NoisyNetの9 Runは有限値で完走したが、3 seedの成績差はseedの幅内だった。σと行動変化率は学習中に縮み、探索置換でホバリング離脱が遅くなったとは言えなかった。（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ。Run: run_20261006-233023_ll_qr_off_s1 / run_20261006-233110_ll_qr_nzadd_s1）。

- IQNについて: learner update、eval、full distribution表示、finiteなloss/Q/gradを確認した（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- QR対IQNについて: matched 3-seed A/Bで`T50`は重なった。持続的`T180`到達はQR 1/3対IQN 3/3、終盤policy reward平均は144.3対186.0だった（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- update_warmup_steps=38,400について: QR/IQNの正式比較で一致させた値。LunarLanderの最適値とはみなさず、warmup探索は継続する（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN learner N=32, M=32について: QR 32 quantilesとpair数を揃えやすく、3 seedで学習成立と終盤の健全性を確認した（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN learner N=32, M=64について: `N=64, M=64`よりraw loss/gradを抑え、単一Runでは良好。ただしtarget計算量が多い（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN learner N=32, M=8について: 単一Runで終盤が不安定。論文上の一般傾向だけでLunarLanderへ採用しない（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- UQE個別調整について: tau end、tail mean、epsilon floor、optimistic targetの単発probeでは一貫した改善を確認できなかった（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- NoisyNet（V/A ストリームの FC と Head の射影）について: QR の matched 3-seed で、spatial UQE への上乗せも ε の置き換えも、1.5M〜2.0M の policy reward 平均（134.0 / 127.3）が OFF（145.1）の seed の幅（96.6〜199.4）に収まった。σ と probe の指標は学習とともに縮んだ（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ・対応する探索ブロック）。
- TempoRL（スキップ、J=4）について: IQN＋Munchausen の 3 seed・2M で、評価 policy 報酬の Run 全体の平均は無効 21、分離×遮断 122、分離×合算 137、包含×遮断 96。分離×合算は 3 seed とも約 1.0〜1.1M で評価の EMA が 200 に届いた（無効は 3 seed 中 1 seed だけ）。これを受けて 2026-10-26 から LunarLander の既定を有効（分離×合算）にし、無効の Run は CLI で `run.$=run.@munch` を渡す（出典: [098_temporl_skip_policy_10prd.md](../../memo/098_temporl_skip_policy_10prd.md)、§7「段 2 の結果」・§6 の 21）。
- TempoRL 第二弾の段 2 について: 合算の基準に対して分割・`kmax`・学習側のスキップ長の ε-greedy を同じビルドの 3 seed・2M で比べ、評価 policy 報酬の Run 全体の平均は合算 135／113／125、分割 148／133／98、`kmax` 141／135／134、ε-greedy 150／104／84。3 腕とも基準以上だが上回る腕は無く、LunarLander の既定は据え置き。基準は `[greedy]` session を足した分、10/26 の同 seed の Run（131／143／138）と値が違う（出典: [2026-10-28_temporl-split-kmax-epsilon.md](../../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md)、現時点のまとめ 1・5・探索ブロック 01・02。Run: run_20261028-131118_t103_ll_mrg_s1 ほか 11 本）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [campaignにない独自の分析](../../experiments/default-dqn/lunarlander/README.md#独自の分析)。
