# DefaultDQNAgent / LunarLander 探索記録

LunarLander の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-08-08〜 | [IQN導入とQR比較](2026-08-08_iqn.md) | IQN動作確認、ReplayBuffer capacity、warmup、UQE、quantile sample数、QR/IQN複数seed比較 | active |

## 現時点の判断

最終更新: 2026-08-09

| 設定・探索 | 現在の扱い | 根拠の概要 |
|---|---|---|
| IQN | 実装・学習成立を確認 | learner update、eval、full distribution表示、finiteなloss/Q/gradを確認した |
| QR対IQN | 初期発見は同等、発見後はIQN優位傾向 | matched 3-seed A/Bで`T50`は重なった。持続的`T180`到達はQR 1/3対IQN 3/3、終盤policy reward平均は144.3対186.0だった |
| `update_warmup_steps=38,400` | matched A/Bの比較基準 | QR/IQNの正式比較で一致させた値。LunarLanderの最適値とはみなさず、warmup探索は継続する |
| IQN learner `N=32, M=32` | LunarLanderの検証済み比較基準 | QR 32 quantilesとpair数を揃えやすく、3 seedで学習成立と終盤の健全性を確認した |
| IQN learner `N=32, M=64` | 安定性寄りの参考 | `N=64, M=64`よりraw loss/gradを抑え、単一Runでは良好。ただしtarget計算量が多い |
| IQN learner `N=32, M=8` | 採用保留 | 単一Runで終盤が不安定。論文上の一般傾向だけでLunarLanderへ採用しない |
| UQE個別調整 | 明確な改善なし | tau end、tail mean、epsilon floor、optimistic targetの単発probeでは一貫した改善を確認できなかった |

上記のQR/IQN比較は、現在のLunarLander構成に限定した3 seedの傾向である。
「IQNは発見を早める」とは扱わず、発見後の成長・定着に優位性がある可能性として後続実験で更新する。

## LunarLanderでの評価観点

### ホバリング離脱と着陸定着を分ける

LunarLanderの学習は、次の二段階に分けて評価する。

1. 報酬0付近のホバリング局所解から着陸方策を発見するまで
2. 発見後に報酬180〜200付近へ成長し、着陸を定着させるまで

最終報酬だけでなく、次の指標を使う。

- `T50`: ホバリング離脱の機械的proxy
- `T50 → T180`: 発見後の成長速度
- 1.5M〜2.0M exp-stepのpolicy reward window: 終盤の水準と安定性
- 2Mまでに持続的`T180`へ到達したseed数: seed robustness

`T50`は報酬EMAから後付けしたproxyであり、Env状態からホバリングを直接分類した値ではない。

### 部分観測性とseed差

風の位相とランダム地形は観測に直接含まれない。
この部分観測性は、ホバリングから着陸を発見する時点のseed差を増やす有力な作業仮説である。
ただし、探索乱数、NN初期値、ReplayBuffer sampling、backend非決定性も同時に影響するため、風・地形を確定原因とは扱わない。

アルゴリズムやwarmupの比較は、単一seedの立ち上がりで判断せず、実効設定とbudgetを揃えたmatched multi-seedで行う。

## 探索環境としての位置付け

LunarLanderはDropMergeと比べて実験コストが大幅に低い。
IQN、UQE、ReplayBuffer、learner tau数などの動作確認、傾向分析、仮説生成、回帰確認に有用なため、DropMergeの探索と並行してcampaignを継続する。

LunarLanderで得た性能傾向をDropMergeへ直接一般化しない。
DropMerge投入前の不具合検出、仮説生成、実験条件の絞り込みに使い、最終的な性能判定はDropMerge自身のRunで行う。

詳細な条件、当時の解釈、判断の更新履歴はcampaign文書を参照してください。
