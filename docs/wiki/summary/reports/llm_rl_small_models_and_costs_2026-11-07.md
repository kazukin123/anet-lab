---
title: 小型モデルと計算費用から見たLLMとRL
page_kind: サマリ
material: "[[reports/llm_rl_small_models_and_costs_2026-11-07.md|llm_rl_small_models_and_costs_2026-11-07]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/robot_tasks.md|実ロボット課題の選定]]"
updated: 2026-11-26
---
# 小型モデルと計算費用から見たLLMとRL

## 元の材料

[小型モデルと計算費用から見たLLMとRL](../../../../reports/llm_rl_small_models_and_costs_2026-11-07.md) blob:b4999a19c9a8888a6767d3a77aad62a2915dbfbd

## 要点

- VLM-AR3L は、VLM が画像対に付けた好みのラベルから状態と画像間の変化を採点する報酬モデルを学習し、各 step の VLM 問い合わせを省く。
- SCOUT の Qwen2.5-3B は6種類の記号課題で平均0.86、小さい Scout-DQN は0.83だった。2048では最大 tile を正規化しており、平均値を全課題の成功確率とは読めない。
- EARS の主指標 TAC は、実軌跡500本の順位相関を測る。生成した報酬で学習した方策の最終成績はこの値から分からない。合成ラベルを使う条件では、評価軌跡集合の順位に合わせてラベル生成器を学習した。
- ALFWorld・ScienceWorldでは内側がQwen2.5-1.5B、4 seeds。未学習の課題種類を含むL2でも改善する。最後の方策学習の予算は揃えるが、その前の報酬探索費用は追加である。DERL-baseの探索は40〜128回の内側学習、DERL-fullも8本の全学習相当と最終1本を要する。小さな提案器というだけで、探索全体が軽いとは言えない。同 §4.2、Appendix G–H
- 小型モデルの候補を選ぶ際には、モデルのparameter数だけでなく、対象課題の採点を確かめた証拠と、報酬学習・方策学習を含む総費用を並べる必要がある。本冊の結果だけから、7B・8Bを1つ選んで全てのAtariへ適用できるとは結論できない。

## 内容

### 6. VLM-AR3Lの公開コードで確認した条件

したがって、実装を移す場合の費用はVLM問い合わせだけではない。画像保持、2方向の報酬推論、報酬モデル学習、保存済み遷移の再採点も含む。この費用の切り分けは上の実装からの本調査の整理で、anet-lab上の測定値ではない。

### 7. 費用と効果を分けて読む

追加で確認したiLLMの補足資料には、AtariでTransfo-XL 280MとGPT2系・Llama2/3を比べた結果がある。4-bit使用の記載、バッチ実行、内発報酬を外した比較を探索の別冊 §2–3へまとめた。小型モデルの有用性を検討する証拠を、ロボットの画像採点だけに限定しない。

### 8. 調査の限界

学習、GPU使用量、推論遅延、API費用を独立に測定していない。公開コードの設定と、著者の報告を区別した。

## 主要な概念

- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [実ロボット課題の選定](../../concepts/subjects/robot_tasks.md)
