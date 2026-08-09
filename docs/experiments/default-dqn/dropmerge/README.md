# DefaultDQNAgent / DropMerge 探索記録

DropMerge の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-07-27〜 | [長期 Run: batch / replay 探索](2026-07-27_longrun-batch-replay.md) | 長期継続学習における batch size、replay ratio、PER、実時間効率 | active |
| 2026-08-09〜 | [IQN 導入・QR 比較](2026-08-09_iqn.md) | IQN 32/32 の成立性、QR51 比較、Q 値バブルと一時的 NEET、fixed-grid control | active |

## 現時点の判断

最終更新: 2026-08-09

| 設定・探索 | 現在の扱い | 根拠の概要 |
|---|---|---|
| `batch_size=512`, `replay_ratio=2.0` | 最終成績優先の主力 | 同一 checkpoint の短期比較と後続長期 Run で、B256/RR1 より高い性能水準を示した。ただし実時間は重い |
| `batch_size=256`, `replay_ratio=1.0` | 実時間効率の対照。性能主力としては見送り | throughput は高いが、cy07 分岐の 107M では後半が停滞し、B512/RR2 の水準を回収できなかった |
| `batch_size=512`, `replay_ratio=1.0` | pending | B256/RR1 と同じ sample budget で optimizer update 回数だけを減らす診断候補。未実行 |
| `alpha=5e-5` / `7.5e-5` | invalidated | checkpoint load により AdamW の param group options が復元され、設定ファイル上の alpha が実効学習率になっていない可能性が高い |
| `per_alpha=0.1` | 採用見送り | 単一分岐では明確な改善がなく、NEET 増加の懸念もあった。確度は低い |
| IQN 32/32 random | 100M基準Run完了。長期主力への採用は保留 | Q/NEETバブルは自力鎮火し正常close。90–100MでQR51よりEval target rewardが約16%低く、Double Suika未観測 |
| IQN fixed-grid control | 次の優先診断 | current / target / train-policy tauを`fixed`へ寄せ、IQN固有のsampling varianceとQRとの差を切り分ける |

詳細な条件、当時の解釈、判断の更新履歴は campaign 文書を参照してください。
