---
title: Conv2d 可視化「実Forwardトレース化」実装指示書
page_kind: サマリ
material: "[[docs/memo/done/0006_conv2d_visual_trace_10rfd.md|0006_conv2d_visual_trace_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# Conv2d 可視化「実Forwardトレース化」実装指示書

## 元の材料

[Conv2d 可視化「実Forwardトレース化」実装指示書](../../../memo/done/0006_conv2d_visual_trace_10rfd.md) blob:1ea13343a6aa715fe64754240fa0c59bd9f7d2d5

## 要点

- Conv2dPanel は推論で実際に得た activation を表示する。
- 従来の可視化は、Actor が蓄積したフレーム履歴と正規化後の入力を再現できなかった。
- 本番 Forward 中の値を trace して Observer に渡す。
- 可視化のために別の履歴を作って推論をやり直す経路を外す。
- 状態を変えて比較する sweep の TensorDict 対応は別の作業に分ける。

## 内容

### 背景（直すべきバグ）

現状の可視化は「推論に入力された FrameStack 済み Observation」を再現できていない。
実推論の入力は `Actor::MakeAction` で `context_->PushObservation(state)`（実フレーム履歴を保持する `DictFrameStacker`）→ `obs_norm_->Normalize` を通った 本物のスタック済み・正規化済み obs。

### 方針（確定：Case D = 実Forwardトレース）

再 Forward／再現をやめ、本番 Forward 中に conv2d 中間 activation を sink で横取りする。
横取りした env0 の activation を `BatchActionInfo` の AuxData（transient・非永続） に積み、`TrainEvent` 経由で Observer/Panel が読む。

### スコープ

Forward チェーン4関数に `sink` 引数（デフォルト空）を通し、`NetworkStruct::Forward` に唯一の捕捉点を置く。
アクション経路（`SelectAction` / `ForwardForAction`）に `sink` を通し、`Actor::MakeAction` / `DefaultDQNAgent::MakeAction` で trace を AuxData に積む。

### 落とし穴・注意点

env0 のみ・clone 必須：sink は `a.slice(0,0,1)` で先頭 env だけ抽出し `clone()` で独立させる。view のまま aux に載せるとフルバッチ storage を保持してしまう。
`"00_Input"` の無条件捕捉：旧 `GetConv2dOutputs` は入力を無条件で `"00_Input"` 登録していた。Visualizer 側は 3次元(squeeze後)以外を弾くため、非画像 branch の `"00_Input"` は描画段で無視される＝従来挙動と同じ。

### 検証手順

runner GUI で `Conv2dPanel` を開き、FrameStack 枚数ぶんの異なるフレームが並んで表示されること（従来は同一フレームの複製だった）。
`Conv2dVisualizationObserver` の episode_interval 録画が従来どおり開始/終了し、画像が TensorBoard に出ること。

## 主要な概念

- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
