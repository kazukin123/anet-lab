---
title: LLMの言葉をRLの出来事と行動へ結び付ける
page_kind: サマリ
material: "[[reports/llm_rl_language_grounding_2026-11-07.md|llm_rl_language_grounding_2026-11-07]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/model_based_evidence.md|Model-based Atariの評価]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# LLMの言葉をRLの出来事と行動へ結び付ける

## 元の材料

[LLMの言葉をRLの出来事と行動へ結び付ける](../../../../reports/llm_rl_language_grounding_2026-11-07.md) blob:4c1723780d23aa43a33a0857834f8a8de2ef7e05

## 要点

- 「青い鍵で青い扉を開ける」という助言をRLへ渡す場合、少なくとも三つの対応が要る。どの観測が青い鍵なのか、手持ちの行動でどう近づくのか、どの変化を開いた成功と判定するのか、である。本冊でいうgroundingは、このように言葉と環境内の対象・操作・判定を結び付けることを指す。
- ONI ではRL側とLLM側を分ける。RLの基準約32k step/sに対して、注釈の直接検索は約30k、分類モデルを同時学習する場合は約26k。注釈は全観測のごく一部で、直近の候補を優先する。70Bとの比較で8Bが十分だったのはScore・Oracleの条件であり、全ゲームへの小型化保証ではない。ONI v4 §3.1、Appendix C.1
- RLang では、文の内容によって渡す先が違うことである。「扉を開けよ」は行動についての助言だが、「閉じた扉には通れない」は環境の遷移についての知識である。後者まで正の報酬へ押し込むと、本来何を伝えたかったかを見失う。この研究は、その違いを明示的に扱う比較対象になる。
- DECKARD の評価には、次の前提がある。高レベルのcraft操作、inventory、学習済みVPTを使う。付録にはMineCLIPの報酬整形、各subgoalを一つのworld seedで学習すること、必要な道具を初期所持させることも記される。評価はランダムなworld seed。抽象化の自動獲得や確率的な依存関係は対象外である。DECKARD §5.3、§7、Appendix B.2–B.3
- たとえばボス戦で必要なのが相手の動作タイミングなら、正しい攻略文があっても、観測がその動作を区別できなければ効かない。逆に、出来事を正確に取得できても、同じ出来事を繰り返して得点できる補助報酬なら、本来の進行を妨げる可能性がある。前回のMotif・ELLMの失敗例と、今回の報酬整形の反例は、それぞれ別の層の問題として読む。

## 内容

### 1. 言葉が正しいだけでは足りない

これは3研究の比較のための整理である。LLMの文章を受け取る処理が動いても、入口が正しく対応していなければ助言は作用しない。前回のローカルMiniGrid pilotで見つかった語彙不一致は、この問題の一例だった。

### 2. ONI：採点を待たずに学習を進める

本調査では、これを「LLMを頻繁に呼んでも無料」という結果とは読まない。遅い注釈器を速い方策学習から切り離す代わりに、注釈の少なさと遅れを受け入れる方式である。1台のGPUで共有した場合の競合や、文字メッセージがない環境での画像認識費用は、上記のstep/sからは分からない。

### 3. ONIの固定実装で確認した境界

学習中に追加されるラベルを使う場合、同じ出来事の扱いが時間とともに変わりうる。未注釈だったことと、モデルが無益と判断したことも異なる。本調査としては、生成された文章、解析したラベル、学習へ渡した値を区別して読むことが必要だと考える。

### 6. Atariへ持ち込む前に区別すること

以下は本冊と既調査の結果を基にした、検証対象の整理である。anet-labの変更計画や実験開始の合意を表さない。

## 主要な概念

- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [Model-based Atariの評価](../../concepts/methods/model_based_evidence.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
