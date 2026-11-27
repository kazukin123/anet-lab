# Action queryとattentionの証拠

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

Action Q-Transformerの性能上の優位は一部ゲームに限られた。TQLはattention entropyを制御した条件で大きな価値Transformerの改善を報告したが、anetで同じ改善は未確認である。PACの「行動を入力でなく出力クエリにする」利点は連続制御の1比較だけで、全行動を一度に出すDropMergeの今のHeadより良い根拠にはならない。

## 分かっていること

### anet の知見

- PAC（Springenbergら、ICML 2024）は、Perceiver型の共通部を1回だけ通し、行動を出力側のcross-attentionのクエリとしてQを読む。行動を入力に入れる形との比較は連続制御の1件だけ（Control Suite XS で 86.7 対 76.8）で、出力クエリと行動ごとの出力を持つ通常のHeadとの比較は論文に無い。解説の読みでは、DropMergeの今の構成は評価する行動を入力に入れず全行動を一度に出すので、この結果は「今のHeadより行動クエリがよい」の根拠にならない。論文の「value query」はDropMergeの案の価値クエリと別物で、論文のτはタスク記述でIQNのτと無関係である（出典: [springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md](../../../../reports/springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md)、キューの論点への答え 1・6・このプロジェクトの検討に効く点）。

### 外の事例

- Action Q-Transformerはactionごとのqueryで空間特徴を読む先行例だが、IQNとの融合まで同じ手法だとはしていない。（出典: [action_query_papers_2026-08-13.md](../../../../reports/action_query_papers_2026-08-13.md)、結論）。
- AQTの性能上の優位は一部ゲームに限られ、説明可能性という目的とSOTA性能の主張を分けて読む。（出典: [aqt_performance_and_visibility_2026-08-13.md](../../../../reports/aqt_performance_and_visibility_2026-08-13.md)、1. 総合評価）。（出典のリンク切れ）
- TQL論文は大きな価値Transformerの不安定さをattentionの崩壊と関連付け、attention entropyを制御した条件で規模拡大の改善を報告した。anetで同じ改善を確認したものではない。（出典: [dong2026_tql_paper_2026-10-05.md](../../../../reports/dong2026_tql_paper_2026-10-05.md)、アブストラクト）。
- Query2Label では各クラスに1つずつ固定した query が画像特徴を読む。AQT の行動と同様に出力の意味が事前に決まる例で、集合として予測を対応付ける DETR の object query とは異なる（出典: [Action Query 関連論文サーベイ](../../../../reports/action_query_papers_2026-08-13.md)、2. Query2Label）。
- TQL を入れない Transformer では、比較した5環境すべてでモデルを大きくするほど attention エントロピーが下がり、最大モデルではほぼ0になった。価値関数の滑らかさ・成績との観測上の相関である（出典: [TQL: Scaling Q-Functions with Transformers by Preventing Attention Collapse](../../../../reports/dong2026_tql_paper_2026-10-05.md)、付録 A. スケーリングの全分析（Full Scaling Analysis））。
- Set Transformer の PMA は複数の learned seed vector から集合を読み、対応する数の出力を作る。離散行動の one-hot から query を作る AQT と違い、各出力に行動の意味はあらかじめ結び付かない。（出典: [Action Query 関連論文サーベイ](../../../../reports/action_query_papers_2026-08-13.md)、4. Set Transformer）。
- PAC のオフライン actor-critic におけるモデル規模の評価では、学習損失と実環境の性能の違いがスケーリングの計測を制約した。 → 詳しくは[評価と比較の条件](../measurement/evaluation.md)。（出典: [Offline Actor-Critic Reinforcement Learning Scales to Large Models](../../../../reports/springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md)、論文が述べている限界）。
- TQL の比較では、Transformer だけでなく、MLP の FQL とフローマッチングの floq も、容量を増やすと性能が下がった（最小から最大で平均 10.6% の低下。出典: [dong2026_tql_paper_2026-10-05.md](../../../../reports/dong2026_tql_paper_2026-10-05.md)、5.3）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [forward の持ち帰りは鍵要求型の capture 1 つに統一し、module の診断も同じ器で運ぶ](../../../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)。
- [TransformerEncoder の attention エントロピー診断 RFD](../../../memo/done/0093_attention_entropy_diagnostics_10rfd.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。
