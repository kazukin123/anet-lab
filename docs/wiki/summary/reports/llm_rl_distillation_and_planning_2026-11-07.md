---
title: LLMの助言を方策へ蒸留する方法と、世界モデルの先読みに使う方法
page_kind: サマリ
material: "[[reports/llm_rl_distillation_and_planning_2026-11-07.md|llm_rl_distillation_and_planning_2026-11-07]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/methods/model_based_evidence.md|Model-based Atariの評価]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# LLMの助言を方策へ蒸留する方法と、世界モデルの先読みに使う方法

## 元の材料

[LLMの助言を方策へ蒸留する方法と、世界モデルの先読みに使う方法](../../../../reports/llm_rl_distillation_and_planning_2026-11-07.md) blob:1d038dfe4ad97aeca9eb6a0ef1a0dd91cb5c2987

## 要点

- 行動の候補を教える方式には、学習中に小さな方策へ教え込む方法と、実際に行動を決めるときに先読みの出発点として使う方法がある。前者は学習後に教師を外せる。後者は、その場の助言と環境から学んだ予測を組み合わせる。本冊は二つを、問い合わせの場所と誤りを修正する経路から読む。
- LVLM2P は4種類の MiniGrid・各3 seed で、目標 return に達するサンプル効率が平均で PPO の2.56倍、A2C の2.86倍だった。LavaGap では soft な教師分布が hard な分布を上回り、教師の重みが1を超えると成功率が下がった。
- LVLM2P の公開 PPO 分岐は、学生の行動確率にもう一度 softmax をかけ、教師分布も一つの値にまとめる。論文で比較した soft な教師分布とは処理が異なる。調査対象の公開版が論文の実験版かは未確認である。
- PriorZero は、各実状態から始める木探索の根で LLM を呼ぶ。探索木の内部で節点を増やしても LLM の呼び出しは増えないが、新たな環境 step では再び呼びうる。
- LVLM2Pの解析失敗時の挙動、公開分岐と実験版の対応、PriorZeroの環境サーバーと実験コードの版は追加確認が必要である。公開モデルの現在の提供状況や再実行費用は今回測定していない。学習結果を追認した報告としては扱わない。

## 内容

### 1. 助言をいつ使うか

表は本調査の整理である。「LLMを小さい方策へ移す」と「LLMの世界知識を世界モデルへ直接移す」は同じ操作ではない。

### 2. LVLM2P：行動分布を小さな方策へ移す

評価時に教師が不要であることは明確な利点だが、学習時の総費用が安いことや、小型のローカル教師で同じ結果が出ることまで示すものではない。

### 3. 公開PPO実装で異なる教師分布

二重softmaxの違いは小例で分かる。3行動の学生確率が`[1, 0, 0]`に近づいても、その確率へsoftmaxを掛けた最初の成分は\(e/(e+2)\simeq0.5761\)にしかならない。この値は本調査の計算であり、学習結果ではない。教師の分布を一つの値へ潰す処理も含め、softな助言の比較を再現したと扱う前に確認すべき箇所である。

### 5. 知識の転用と実行可能な操作の違い

非LLMの比較を置く意義は、LLMの価値を最初から否定することではない。どの知識を、どの形式で渡すことに効果があるかを特定するためである。この点は画像Atariへ持ち込む場合にも残るが、本調査はその実証を行っていない。

## 主要な概念

- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [Model-based Atariの評価](../../concepts/methods/model_based_evidence.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
