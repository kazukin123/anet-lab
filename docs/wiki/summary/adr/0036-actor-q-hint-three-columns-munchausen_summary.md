---
title: Actor Qヒントを3列へ拡張し、近似Actor初期優先度をMunchausenターゲットと同型にする
page_kind: サマリ
material: "[[docs/adr/0036-actor-q-hint-three-columns-munchausen.md|0036-actor-q-hint-three-columns-munchausen]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
updated: 2026-11-28
---
# Actor Qヒントを3列へ拡張し、近似Actor初期優先度をMunchausenターゲットと同型にする

## 元の材料

[ADR 0036](../../../adr/0036-actor-q-hint-three-columns-munchausen.md) blob:94c2ffc40737ea053853574cf753e4246cd6468a

## 要点

- Actor の近似初期優先度を Munchausen target と同じ形にするため、行動価値・状態価値に Munchausen 項を加えた。
- ヒントの転送と推定の責務分離を保ち、Actor は既存の推論出力を使う。
- 無効な列には0を入れ、有効・無効によって payload の幅を変えない。
- 価値の分解では内的 Head の2列が追加され、ヒントは常に5列になった。
- 強制行動は Actor が先に確定し、実際の行動に対応する値を計算する。

## 内容

### 必要な追加値

従来の2列だけでは Munchausen 項を表現できず、Actor による近似優先度と学習後の優先度が別の式になってしまう。Actor に学習設定全体を渡さず、ヒントに必要な部分だけを共有する。

### スキーマの履歴

有効時だけ列を追加する形は検証と復号を複雑にする。後の5列化でも無効列は0で同じ形を保つ。実行行動の確定を Actor に移したことで、強制変更に備えて全行動の項を保存する必要がなくなった。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
