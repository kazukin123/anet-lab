# Action queryとattentionの証拠

## 今の結論

Action Q-Transformerの性能上の優位は一部ゲームに限られた。TQLはattention entropyを制御した条件で大きな価値Transformerの改善を報告したが、anetで同じ改善は未確認である。

## 分かっていること

- Action Q-Transformerはactionごとのqueryで空間特徴を読む先行例だが、IQNとの融合まで同じ手法だとはしていない。（出典: [action_query_papers_2026-08-13.md](../../../reports/action_query_papers_2026-08-13.md)、結論）。
- AQTの性能上の優位は一部ゲームに限られ、説明可能性という目的とSOTA性能の主張を分けて読む。（出典: [aqt_performance_and_visibility_2026-08-13.md](../../../reports/aqt_performance_and_visibility_2026-08-13.md)、1. 総合評価）。
- TQL論文は大きな価値Transformerの不安定さをattentionの崩壊と関連付け、attention entropyを制御した条件で規模拡大の改善を報告した。anetで同じ改善を確認したものではない。（出典: [dong2026_tql_paper_2026-10-05.md](../../../reports/dong2026_tql_paper_2026-10-05.md)、アブストラクト）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [forward の持ち帰りは鍵要求型の capture 1 つに統一し、module の診断も同じ器で運ぶ](../../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)。
- [TransformerEncoder の attention エントロピー診断 PRD](../../memo/093_attention_entropy_diagnostics_10prd.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
