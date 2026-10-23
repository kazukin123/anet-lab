# LunarLander

## 今の結論

matched 3-seedのQR/IQN比較ではホバリング離脱時期に明確な差がなかった。持続的T180到達はQR1/3対IQN3/3、終盤reward平均は144.3対186.0だった。

## 分かっていること

- matched 3-seedのQR/IQN比較ではホバリング離脱の早さに明確な差はなく、離脱後のT180到達と終盤rewardはIQNが有利な可能性を示した。（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ3・9・10）。
- RB満杯付近の変化はcapacity半減で前へ動き、IQN固有より経験分布切替が有力だった。current N削減によるloss低下は集約尺度も変えるため、過学習改善と断定できない。（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ5〜8）。
- NoisyNetの9 Runは有限値で完走したが、3 seedの成績差はseedの幅内だった。σと行動変化率は学習中に縮み、探索置換でホバリング離脱が遅くなったとは言えなかった。（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ。Run: run_20261006-233023_ll_qr_off_s1 / run_20261006-233110_ll_qr_nzadd_s1）。

- IQNについて: learner update、eval、full distribution表示、finiteなloss/Q/gradを確認した（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- QR対IQNについて: matched 3-seed A/Bで`T50`は重なった。持続的`T180`到達はQR 1/3対IQN 3/3、終盤policy reward平均は144.3対186.0だった（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- update_warmup_steps=38,400について: QR/IQNの正式比較で一致させた値。LunarLanderの最適値とはみなさず、warmup探索は継続する（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN learner N=32, M=32について: QR 32 quantilesとpair数を揃えやすく、3 seedで学習成立と終盤の健全性を確認した（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN learner N=32, M=64について: `N=64, M=64`よりraw loss/gradを抑え、単一Runでは良好。ただしtarget計算量が多い（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- IQN learner N=32, M=8について: 単一Runで終盤が不安定。論文上の一般傾向だけでLunarLanderへ採用しない（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- UQE個別調整について: tau end、tail mean、epsilon floor、optimistic targetの単発probeでは一貫した改善を確認できなかった（出典: [2026-08-08_iqn.md](../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ・対応する探索ブロック）。

- NoisyNet（V/A ストリームの FC と Head の射影）について: QR の matched 3-seed で、spatial UQE への上乗せも ε の置き換えも、1.5M〜2.0M の policy reward 平均（134.0 / 127.3）が OFF（145.1）の seed の幅（96.6〜199.4）に収まった。σ と probe の指標は学習とともに縮んだ（出典: [2026-10-06_noisynet.md](../../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md)、現時点のまとめ・対応する探索ブロック）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [campaignにない独自の分析](../../experiments/default-dqn/lunarlander/README.md#独自の分析)。
