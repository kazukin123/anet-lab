---
title: LLMでRLを補助する公開実装の再現条件
page_kind: サマリ
material: "[[reports/survey/llm_assisted_rl_reproducibility_2026-11-07.md|llm_assisted_rl_reproducibility_2026-11-07]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
updated: 2026-11-28
---
# LLMでRLを補助する公開実装の再現条件

## 元の材料

[LLMでRLを補助する公開実装の再現条件](../../../../../reports/survey/llm_assisted_rl_reproducibility_2026-11-07.md) blob:74e00e7fc6458e48b5fe3104ffd60a36ab95291a

## 要点

- 固定 commit で公開コードと配布モデルを照合した静的調査で、学習の再実行は行っていない。
- 短い候補評価と本格学習の結果が混在しうるため、記録された評価段階も結果の一部になる。
- BIRDの既定設定はmockの生成器と評価器である。READMEのtesterで動作確認が済んでも、実LLM・実RLの成績を再現したことにはならない。生成器と評価器は別々にproviderとmodelを持つため、片方を変えた条件を両方変更した条件と取り違えない。既定設定、README
- 報酬モデルの推論例では映像全体からフレームを間引いて使う。
- モデルや評価器を置き換えただけでは論文の成績を再現したことにならない。

## 内容

### 1. 確認対象と読み方

論文の実験条件、配布コードの既定値、今回実行して確かめたことを区別する。最後の区分に入る学習結果はない。コードから決定的に読める分岐と、実際に特定環境で発生した障害も区別する。

RoboRewardは著者のモデルカードを確認した。以下でコードを引用するリンクはcommitを固定し、モデルカードは確認日を基準とする。

### 2. LLM-Explorer

LLM-Explorer の Atari DQN は、直前のエピソードの報酬と間引いた行動列を LLM に渡す。公開版の既定値は毎エピソード更新、総31万 step で、論文の10万・50万 step の条件を再現するには設定を指定し直す必要がある。

### 3. LIMEN

LIMEN の配布設定は500反復・Claude Sonnet 4、論文は30反復・Claude Sonnet 4.6。候補をまず50万 step 学習し、閾値を通ったものを100万 step・3 seed で評価する。本格学習に失敗すると短い学習の指標が返るため、比較には成績と評価段階の両方が要る。

### 配布モデルと採点範囲

BIRD は既定の生成器と評価器に模擬実装を使う。実際の比較では、それぞれのモデルと方法別の方策数を記録する。Robo-Dopamine の動画推論例は応答を数値にできなければ0とする。RoboReward の配布モデルは目標と動画から最後の状態を1〜5で採点する。

### 7. 再現結果を比べるために残す情報

論文と配布版を比べるには、実行したコードの版、モデル、探索の反復数、外側の探索 seed と各候補を評価した seed を記録する。候補の再生成や棄却に使った時間と、最終方策を学習した時間も区別する。

### 8. 未確認事項

動画採点モデルをAtariへ移した場合の妥当性、終端や後退を含む採点の一貫性。

外部リポジトリの全コード、全入口、全失敗経路の検査。確認箇所を超える品質判定はしていない。

## 主要な概念

- [LLMの外部知識と探索](../../../concepts/methods/llm_prior_knowledge.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [内的報酬による探索の比較](../../../concepts/methods/intrinsic_novelty.md)
