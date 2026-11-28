---
title: Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)
page_kind: サマリ
material: "[[docs/memo/done/0057_native_stack_dim_handling_10rfd.md|0057_native_stack_dim_handling_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)

## 元の材料

[Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)](../../../memo/done/0057_native_stack_dim_handling_10rfd.md) blob:206050a5fc624e56b95a28a65d94d7e4ba3a88a9

## 要点

- 複数フレームの実入力には独立した時間軸があったが、構築用のダミー Tensor は特徴軸へ畳んでいた。
- LunarLander の4枚・8特徴の例では、ダミーが (1,32)、実データが (N,4,8) だった。
- ネットワーク構築に使う spec で stack を独立軸として表し、実データと rank を合わせる。
- Flatten や StackMerge は必要なネットワークだけが使い、時間方向の Conv1D を自然に記述できる。
- 行動側と ReplayBuffer の実データが持つ形を保ち、構築時の推論との一致を検証する。

## 内容

### Stack 軸と構築時の形状

4枚積んだ LunarLander の8特徴は、行動時・学習時ともバッチを含めて `(N,4,8)` になる。一方、従来の入力仕様は stack を特徴次元へ畳んで `(1,32)` のダミーでネットワークを構築するため、時間方向の Conv1D に入力するには形状を読み替える必要があった。

入力仕様に `[stack, *orig_shape]` を持たせ、ダミーも `(1,4,8)` にする。時間軸の位置と意味が行動時・学習時・形状推論で一致し、ブランチの先頭に一律の Flatten と Reshape を置かずに済む。画像を CNN に渡す場合は、そのネットワークの StackMerge が時間軸とチャンネル軸をまとめる。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
