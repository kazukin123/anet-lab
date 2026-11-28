---
title: LLMの小目標・行動の制約・失敗の振り返りをRLへ渡す
page_kind: サマリ
material: "[[reports/survey/llm_rl_goals_and_reflection_2026-11-07.md|llm_rl_goals_and_reflection_2026-11-07]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/robot_tasks.md|実ロボット課題の選定]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# LLMの小目標・行動の制約・失敗の振り返りをRLへ渡す

## 元の材料

[LLMの小目標・行動の制約・失敗の振り返りをRLへ渡す](../../../../../reports/survey/llm_rl_goals_and_reflection_2026-11-07.md) blob:b51aa85838d5613bc6030f474daa89b7d51bc326

## 要点

- ELLM は画像に加えて状態説明も入力する。学習した文章化器を使う追加試験には人手ラベル847件と合成ラベル900件を要し、単語の重なりによる誤報酬が残った。
- Craftax-Classicの5M step・5 seedではscoreがSGRL 33.8±1.5、PPO 24.8±5.7。一方、速度は1,850対13,530 step/sで、PPOより遅い。SGRLは頻繁にLLMを呼ぶ比較法より速い、という範囲にとどまる。生成はDeepSeek-R1/V3、RL側はRTX 4090という条件であり、小型LLMの同居を実証した結果ではない。§5、Table 1、Appendix B
- LLM-TALE は物体状態と既存の制御器、距離・速度に基づく報酬を使い、ManiSkill と RLBench の計6課題・3 seed で評価した。実機の PutBox は15試行中14成功。比較先の Text2Reward と他の手法では ManiSkill の版が異なる。
- LAGEA では、FuRL の MT10 平均成功率が固定目標で76%から80%、ランダム目標で64.6%から70.4%に上がった。実時間を測った3 seed・9条件の平均は94.90分から92.36分で、固定目標の button-press だけを見ると99.71分から162.12分に増えた。
- 行動の提案と禁止を分ける。 目標を観測に加えるだけなら別の行動も選べる。maskは選択可能な行動そのものを減らす。誤った助言への耐性を問うなら、正しい目標での成績だけでなく、必要な行動が一時的に候補から落ちた場合も別の条件になる。

## 内容

### 1. 何を目標にするかと、どう達成するか

表は各論文と後述の実装を比較するための本調査の整理である。同じ「言語で探索を助ける」でも、画像だけから一から行動を学ぶ方式と、既存の制御器を調整する方式を同列に扱えない。

### 2. ELLM：有用そうな目標を探索の報酬にする

したがって本調査では、「文章化まで無料」「下流課題の情報なし」「新規runでも問い合わせが少ない」という三つの読み方を避ける。何を再利用したかを揃えると、意味のある探索目標を選ぶ効果を考えやすい。

### 3. SGRL：目標生成をコードにし、行動を制約する

この比較から言えるのは、同じstep数での成績と処理速度の交換条件があることまでである。同じ実時間でPPOが何点に達するかは、Table 1だけでは分からない。後述の公開実装には乱数の扱いにも確認点がある。

### 6. 公開実装で確認した再現条件

対象は `monaf-chowdhury/LaGEA@0f8d761729665abc03635410560fd4c1fe853f41`（2026-05-01）。arXiv v3より古いため、最終版との完全一致は未確認である。

## 主要な概念

- [LLMの外部知識と探索](../../../concepts/methods/llm_prior_knowledge.md)
- [内的報酬による探索の比較](../../../concepts/methods/intrinsic_novelty.md)
- [実ロボット課題の選定](../../../concepts/subjects/robot_tasks.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
