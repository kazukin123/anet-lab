# 訪問済み状態へ戻る探索

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

有望な状態を記憶し、そこへ戻ってから探索する。シミュレータの復元、行動列の再生、目標条件付き方策では、戻れる環境と必要な環境操作が異なる。

## 分かっていること

### anet の知見

なし

### 外の事例

- Go-Explore は、報酬が消えた探索前線へ戻れなくなる detachment と、戻る途中に探索ノイズで行動列が崩れる derailment を区別した。2019 年版は状態を保存・復元して探索し、成功軌跡の模倣で頑健な方策を作る。後者は計算が重く、Nature 版の該当評価は 11 ゲームだった（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§5.1〜5.2）。
- policy-based Go-Explore は PPO の目標条件付き方策でセルへ戻り、sticky actions 下の Montezuma は 12B フレームで 97,728、Pitfall は 10B で 20,093。セルにはゲームの知識を使い、確率的な報酬は将来課題としている（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§5.2）。
- DTSIL は記憶した軌跡を条件にして戻り、任意のリセットを要さない一方、Atari の結果には RAM の位置を使った。MEGA は達成済みで密度の低い目標へ行ってから探索を増やし、意味のある目標空間を前提とする。Cell-Free Latent Go-Explore は手設計のセルを学習した表現に替えた（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§5.3・5.5）。
- IGE は archive の保存・戻り先・探索行動を基盤モデルに選ばせる。報告された実験は決定的な環境に限られる。ELLM の目標提案や Motif の選好報酬では、状態への復帰は同じ前提になっていない（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§5.4）。

## 横断的知見

- 補完: IGE の方式は訪問済み状態に戻れることを前提にする。公開 BabyAI-Text の固定コードでは、同じ seed から行動列を再実行して復帰し、その後に探索用のカウンタを初期化する。条件の違い: 前者は手法が環境に求める能力、後者は公開された一つの実装での復帰方法と費用の数え方である（出典: [内的報酬の調査](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§5.2〜5.4、[IGE の固定コードを調べた報告](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§4〜5）。

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし

## 関連

- [LLM の外部知識と探索](llm_prior_knowledge.md)。
- [Env 共通基盤の外の事例](../framework/320_env_core.md#外の事例)。
