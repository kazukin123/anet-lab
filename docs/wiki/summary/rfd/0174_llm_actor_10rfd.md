---
title: LLM の事前知識を使うゲーム Actor の候補
page_kind: サマリ
material: "[[docs/memo/0174_llm_actor_10rfd.md|0174_llm_actor_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# LLM の事前知識を使うゲーム Actor の候補

## 元の材料

[RFD0174　LLM の事前知識を使うゲーム Actor の候補](../../../memo/0174_llm_actor_10rfd.md) blob:aca4ba931bf0feaade8726cf0926a42e13fcb547

## 要点

- LLM を単独の Actor または DQN と組み合わせて使う暫定案である。どの構成を実装するかは未決である。
- Kung-Fu Master のゲーム目的や表示の意味のような事前知識を、画像を読むモデルから行動へ取り込む案である。
- LLM だけで行動を決める Actor と、DQN の探索や方策を補う方式を比較する。
- ハイブリッドの初期候補には、ε探索で選ぶ行動の分布を LLM が決める方式がある。
- 候補ごとに DQN が学ぶ対象、推論を行う頻度、ゲーム進行に与える負荷を区別する。
- 個別ゲームでの結果と他の環境への一般化を分け、採用する方式は実験で判断する。

## 内容

### 目的と問題

LLM（画像を読めるもの）がゲームの常識で行動を選ぶ Actor を作り、KFM のボス討伐のように「何をすべきか」の知識が足りない場面に事前知識を持ち込む。形は 2 つある。
LLM の好みや提案を報酬にすること（Motif・ELLM 型）。サーベイでは Atari での効果が条件付きなので、必要なら別 RFD にする。

### 背景: KFM のボス討伐

ボスは距離で出る。左へ歩くだけで約 3,000 フレームでボスの位置に着く。雑魚は動いているときに出てくる。
ボスの手前から始めること（状態の保存と復元）と、ボス体力の減りへの報酬は「ゲーム仕様を別にするのはやり過ぎ」で採らない。sticky action を切るのは本来のゲームに戻すので可。エージェント側の汎用の工夫は歓迎。LLM の事前知識はエージェント側の知識であり、ゲームの仕様の変更ではない。

### 起草時点の現状

anet の実行系に LLM や HTTP の呼び出しは無い（LLM を使うのは CI のレビューだけ）。
Atari の env は、表示用に元の RGB 画面（210×160）を保持できる（`AtariEnv.retain_rgb_frame`、既定 true。AtariEnv.cpp の `rgb_frame`）。LLM に渡す画面はここから取れる。行動の一覧は ALE の最小行動集合（`getMinimalActionSet`）。

### 提案の骨子

`LLMAgent`（学習しない Agent）を作る。現在の画面、直近の数枚、行動の一覧、短い指示文を LLM に渡し、行動を受け取る。
呼び出しを減らすため、LLM は「行動と、それを続ける step 数」を返し、その間は呼ばない（既定: 最大 16 step）。
学習しないので、env を LLM の応答まで待たせてよい。評価用の Runner（少数 lane）で回す。

### User Stories

研究者として、LLM の事前知識だけで KFM のボスを倒せるかを、評価と同じ指標で測りたい。
研究者として、LLM のプレイをデモとして保存し、DQN の学習に消えない形で混ぜたい。
研究者として、LLM の待ち時間で学習のループを止めずに、一部の lane だけ LLM に任せたい。

### 背景の補足

デモの別枠と、別枠から混ぜて引く仕組みは、人間のプレイを学習に使う路線（0160_input_target_train_forced_action）とも共通の部品になる。データの出どころ（LLM か人間か）に依らない形で作る。

### 候補を採る場合の成功の測り方と受入の目安

段階 B1: デモを混ぜた DQN（実時間優先 30M、sticky 0、2 seed）で、学習中のボス命中が 10M 以降も続き、撃破 ÷ 命中が上がる。基準では 6 本に 1 本しかボス行きに入らないので、命中と勝率の推移を主に見る。DQN の評価の点数が先生（パターン A）を超えるかも見る。
段階 B2: 学習のループの速さが、LLM lane なしの場合と測定のぶれの範囲で同じ（LLM を待っていない）。

## 主要な概念

- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
