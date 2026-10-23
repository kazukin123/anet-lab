# DefaultDQNAgent / DropMerge 探索記録

DropMerge の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-07-27〜2026-08-12 | [長期 Run: batch / replay 探索](2026-07-27_longrun-batch-replay.md) | 長期継続学習における batch size、replay ratio、PER、実時間効率 | completed |
| 2026-08-09〜 | [IQN 導入・QR 比較](2026-08-09_iqn.md) | IQN成立性、QR51比較、tau sampling 5方式、K/N/M本数探索 | active |
| 2026-10-01〜2026-10-15 | [Atari からのフィードバック](2026-10-01_atari-feedback.md) | Munchausen、IQN と UQE の相互作用、target 更新、NoDropTimeout 罰、NoisyNet と NEET、RR・NN 構造・γ、100M からの分岐 | completed |
| 2026-10-15〜 | [勾配ノイズと batch / replay ratio の機序](2026-10-15_gradient-noise-batch.md) | PRD092 の勾配ノイズ、B・RR・学習率の分岐、B の上限、評価専用 Run による判定、NoisyNet の揺らぎへの感度、Run ごとのぶれと学習中の評価 ×8、学習の中身の多角比較 | active |

## 知見のまとめ

- [wikiの知見](../../../wiki/subjects/dropmerge.md)から題材・機構と正本をたどる。

## 独自の分析

以下はcampaignにない実行契約の補足と、複数の観測を読むための分析である。個別Runの結果要約はwikiへ集約する。

### NOOP中の物理進行

`use_instant_drop=true`はNOOP中の物理進行を消す設定ではなく、現在の構成ではNOOP 1回につき物理1 frameが進む。

stack 4のgrid、`no_drop_timeout_ratio`、prev-action trio / DROP markerがあるため、方策は盤面の移動、直前action、timeout接近を組み合わせて「動いている間は待ち、落ち着いたらDROPする」技能を表現できる。
この段落は、campaignに記録された待機の成績ではなく、観測の組から表現できる技能についての分析である。

### 高度NEETとQ値の読み方

Eval policyがUQEの場合、Q tableの`Mean`最大actionが必ず選択されるわけではない。
risk-tail scoreも含めた選択なので、action highlightとMean順位の差をバグとみなさない。

### NoLegalDropを長期で読むための補助観測

成熟したcy07の175〜200M windowではNoLegal EMAがEval1約46.7%、Eval2約45.2%、SpawnBlockedが約37〜39%まで変化した。
この観測はcampaign本文に同じ数値と条件の記載を確認できなかったため、READMEを正本として残す。
比較対象のscratch B512/RR2 Runの80M終点は、[長期Runの探索ブロック10](2026-07-27_longrun-batch-replay.md#探索ブロック-10-scratch-開始-b512rr2-の-q-バブルと条件付き-neet)を参照する。
NoLegal獲得はEnv実装の成立確認と方策学習を分け、長期指標として観測する。

### 終端分類と投了学習を分ける

NoLegal Phase 2は方策へ投了を直接教えるreward shapingではない。
Agentは、満杯近くの比較的稀な状態で「どのDROPも失敗する」と認識し、SpawnBlockedになるDROPではなくNOOPを選ぶ必要がある。
さらにunsettled時は、裁定までNOOPを維持しなければならないため、n-step 1では不利・有利の伝播に時間がかかる。

Phase 2有効時に`no_drop_timeout_on_candidate≈0`なら、NoDropTimeoutへ詰み盤面が大量に誤分類されている証拠は弱い。
Phase 2は終端契約の修正であり、高度NEETを直接治す機構とは扱わない。

### 診断を進める順序

1. Run artifactの実効設定とstep軸を確認する。
2. NoLegal / NoDropTimeout / SpawnBlockedの終端契約とcandidate指標を確認する。
3. 映像またはepisode forensicで、物性移動中の短い待機と高度NEETを分ける。
4. batch / replay、Q / loss / TD、探索方策など、待機価値を増幅した学習側要因を一軸比較する。

介入の採否と罰の比較は[長期Runの残件](2026-07-27_longrun-batch-replay.md#次campaignへ保留する残件)からたどる。
