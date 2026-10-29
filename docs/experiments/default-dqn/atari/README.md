# DefaultDQNAgent / Atari 探索記録

Atari（ALE 直結）の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-08-14〜2026-08-27 | [ベースライン定義（Breakout / Pong）](2026-08-17_baseline.md) | プロトコル、env数、NN、gradient clipping、PER、spatial exploration、replay ratio、予算 | completed |
| 2026-08-27〜 | [Atari-5 横断スイープ](2026-08-27_atari5.md) | Atari-5横断比較とプロトコル・optimizer設定 | active |
| 2026-08-28〜08-29 | [可塑性メトリクスの較正と保護機構](2026-08-28_plasticity.md) | 可塑性診断の較正、測定コスト、正規化、weight decay | completed |
| 2026-08-29〜 | [設定修正後の再ベースラインと RR の勾配軸決着](2026-08-29_bundle-fix-rebaseline.md) | 設定修正後のベースラインとreplay ratio比較 | completed |
| 2026-08-30 | [可塑性の保護機構 screening](2026-08-30_protection-screening.md) | 活性、LayerNorm、spectral normalization、spectral capの比較 | completed |
| 2026-08-31 | [RR4 天井と LayerNorm](2026-08-31_rr4-ceiling-ln512.md) | RR4/RR1、LayerNorm、target更新、評価指標 | completed |
| 2026-08-31〜 | [BTR ソース差分の腕別スクリーニング](2026-09-01_btr-feedback-arms.md) | BTRとの差分、複製Run、head構造、target更新、評価本数 | active |
| 2026-09-07 | [UQE 探索方策の掃引](2026-09-07_uqe-exploration.md) | UQEとtauラダーの比較 | completed |
| 2026-09-08〜 | [高 replay_ratio で成績が落ちる機序](2026-09-08_replay-ratio-mechanism.md) | replay ratio、target鮮度、batch、正則化、replay fit | active |
| 2026-09-12〜 | [BTR NN構成の移植検証](2026-09-12_btr-network-port.md) | BTR NN構成、Pool6、tau融合、SN、継続Run | active |
| 2026-09-14〜 | [replay の lane 窓と eval コスト](2026-09-15_replay-lane-window.md) | replay容量、env数、lane履歴窓、評価プロトコル、評価コスト | active |
| 2026-09-21〜 | [実時間優先プロファイルの Atari-5 展開](2026-09-21_realtime-profile-atari5.md) | 実時間優先構成のAtari-5横断比較とreplay容量・NN・評価予算 | active |
| 2026-09-23〜 | [Atari-5 外ゲームでの実時間優先 100M](2026-09-23_realtime-profile-100m.md) | Atari-5外ゲームの100M評価と公表値比較 | active |
| 2026-09-25〜 | [サンプル効率優先プロファイルの Atari-5 展開](2026-09-25_sample-efficiency-profile-atari5.md) | サンプル効率優先構成のAtari-5横断比較と探索ラダー | active |
| 2026-09-27〜 | [BTR との差の切り分け（qbert / battle_zone）](2026-09-27_btr-gap-qbert-bz.md) | qbert/battle_zone、探索、env数、容量、同期、UQE、ViTの比較 | active |
| 2026-10-11〜10-16 | [NoisyNet の実時間優先プロファイルへの投入](2026-10-11_noisynet-realtime.md) | NoisyNetのεラダーへの置換と上乗せ、Breakoutの2×2、probe | completed |
| 2026-10-15〜10-20 | [kung_fu_master のボス行きの定着](2026-10-15_kfm-boss-route.md) | kung_fu_masterのボス行き、報酬クリップ、ライフ区切り、sticky、ε、PER、UQE、seed数、batch、ViT | completed |
| 2026-10-23〜 | [BTR 公開実装の qbert 再現と、序盤の学習の遅れ（初期値）](2026-10-23_btr-qbert-repro-init.md) | BTR公開実装の手元での再現、BTR寄せ、縮小方式、BTRのε、コード照合、SN付きconvの初期値 | active |
| 2026-10-29 | [qbertの行動固定を評価時だけ外す比較](2026-10-29_temporl-auxiliary-only-stage0.md) | PRD104段0、同じ終了checkpointでfull/single各200ゲーム | 段0完了・段1保留 |

## 知見のまとめ

- [wikiの知見](../../../wiki/subjects/atari.md)から題材・機構と正本をたどる。

## 独自の分析

### campaignに無い比較の方針

- ViTハイブリッドはCNN以外の基本構成の選択肢として、成績によらず常設する。この方針はAtari固有ではなく、Breakoutの成績だけで採否を決めない。

### 追加探索と複製の運用方針

- BTR NN構成の下でのRR2は測っていないが、どちらのプロファイルの選択も変えないので探索しない。
- 複製を持たない腕の小差を順位付けしないことは、追加測定を待つ保留ではなく運用規則として扱う。全腕の厳密な順位付けは目的ではなく、複製を取るのは判定が要る軸に限る。

これらは探索予算の使い方についての方針であり、未測定の腕の性能を認定するものではない。複製幅の実測値はwikiからcampaignの正本へたどる。
