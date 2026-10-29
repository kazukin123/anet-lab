# DefaultDQNAgent / LunarLander 探索記録

LunarLander の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-08-08〜 | [IQN導入とQR比較](2026-08-08_iqn.md) | IQN動作確認、ReplayBuffer capacity、warmup、UQE、quantile sample数、QR/IQN複数seed比較 | active |
| 2026-10-06〜2026-10-07 | [NoisyNet の導入確認（QR）](2026-10-06_noisynet.md) | NoisyNet の学習成立と診断メトリクス、spatial UQE への上乗せと ε の置き換えの matched 3-seed | completed |
| 2026-10-28 | [TempoRL 第二弾の段 2（分割・k_max の重み・ε-greedy）](2026-10-28_temporl-split-kmax-epsilon.md) | PRD103 §7 の段 2。合算の基準に対する分割・`kmax`・スキップ長の ε-greedy の同じビルド 3 seed 比較、`[greedy]` session の判断率、10/26 の同 seed との照合 | completed |
| 2026-10-29 | [TempoRLを行動に反映しないSingleの比較](2026-10-29_temporl-auxiliary-only-stage1.md) | PRD104 ステージ1。7群×3 seed×2M、Single各群と無効の比較、STOP対照、k=1、終了窓＋split | completed |

## 知見のまとめ

- [wikiの知見](../../../wiki/subjects/lunarlander.md)から題材・機構と正本をたどる。

## 独自の分析

以下は探索対象を選ぶための位置付けである。ホバリング離脱・着陸定着の判定とseed差の知見はwikiからcampaignの正本へたどる。

### 探索環境としての位置付け

LunarLanderはDropMergeと比べて実験コストが大幅に低い。
IQN、UQE、ReplayBuffer、learner tau数などの動作確認、傾向分析、仮説生成、回帰確認に有用なため、DropMergeの探索と並行してcampaignを継続する。

LunarLanderで得た性能傾向をDropMergeへ直接一般化しない。
DropMerge投入前の不具合検出、仮説生成、実験条件の絞り込みに使い、最終的な性能判定はDropMerge自身のRunで行う。

詳細な条件、当時の解釈、判断の更新履歴はcampaign文書を参照してください。
