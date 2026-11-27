---
title: LLMの行動助言をRLのどこへ接続するか
page_kind: サマリ
material: "[[reports/llm_rl_action_advice_2026-11-07.md|llm_rl_action_advice_2026-11-07]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# LLMの行動助言をRLのどこへ接続するか

## 元の材料

[LLMの行動助言をRLのどこへ接続するか](../../../../reports/llm_rl_action_advice_2026-11-07.md) blob:3ab948f65b7b09a4dd394380d755f6e60dc29f0b

## 要点

- 「右へ動くとよい」という助言には、少なくとも三つの渡し方がある。方策に参考情報として見せる、探索する行動の確率を変える、学習で比べる行動の集合自体を狭める、である。最後の方法では、行動を選ぶ段階だけでなく、価値の更新でも除かれた行動を比較しなくなる。
- LLM Priors のキャッシュ比較はTomatoの1 training seed。逐次問い合わせ31.1分・2,933回から13.8分・434回へ短縮したが、GPU使用量は約28,340 MB。計算機は40 GB A100を2枚搭載する。24 GB単一GPUの実証とは読めない。候補数を増やす比較もあるが、理論の極限はKL正則化付き目的である。Proposition 1、Appendix 9.3–9.4、Table 12
- 観測にヒントを加える実験では、3種類の単一部屋BabyAIを3Mまたは5M frame学習する。PickupLocの最終成功率は17.2%から29.5%で、差は12.3ポイント、相対改善は約71%。GoToObjの50%到達は1.08Mから120K frameへ減るが、9倍という値は実時間ではない。課題文だけ追加した比較もある。Table 2–3
- APM の主モデルはGPT-4o-mini。ローカルLlama-3-8Bの比較は行動の効果クラスの一致であり、モデルを替えたRL学習の成績と同一視できない。同期実行でAPIは約3〜6秒、Intel i5のCPU上の8Bは約30秒/応答と記し、ゲーム内100秒の最大200問い合わせで最大約20分・100分と見積もる。IEEE本文 §IV-H
- 候補から必要な行動を外す方式では、観測として助言を見せる方式と違い、方策が誤った助言を無視できない。

## 内容

### 2. LLM Priors：候補の中で価値を学ぶ

本調査の解釈では、候補の再利用は費用と提案の揺れを減らす一方、最初の提案の偏りを保持する。この条件を§5で、論文の性能値とは独立に計算する。

### 3. 観測ヒント：従うかどうかを方策が学ぶ

観測ヒント論文 Conclusion and Discussion（2025/10）

### 4. APM：行動の直後の効果を学んで文章にする

この方式の研究上の問いは、「数値をLLMへ直接読ませるか」だけではない。どの状態変化を有用なクラスとするか、その判定をどのデータで学ぶかも含む。距離が減ることと最終的に目的を達成できることは、一般には同じではない。

### 5. 候補数とcacheの小さな計算例

最初の候補集合を同じ観測に対して永久に使い、選択も更新もその集合内に限るとする。この追加条件の下では、最初に欠けた行動は、Qの推定を正確にしても集合へ戻らない。新しい独立候補を毎訪問で作るなら、\(n\)訪問後の一度以上の包含確率は\(1-(1-p)^{kn}\)だが、同じcacheを使うなら\(1-(1-p)^k\)のままである。

## 主要な概念

- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
