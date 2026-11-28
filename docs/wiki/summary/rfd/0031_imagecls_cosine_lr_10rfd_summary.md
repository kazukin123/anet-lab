---
title: ProfiledValue による ImageCls learning_rate 制御
page_kind: サマリ
material: "[[docs/memo/done/0031_imagecls_cosine_lr_10rfd.md|0031_imagecls_cosine_lr_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-28
---
# ProfiledValue による ImageCls learning_rate 制御

## 元の材料

[ProfiledValue による ImageCls learning_rate 制御](../../../memo/done/0031_imagecls_cosine_lr_10rfd.md) blob:839a03e03df878b2a6e56e07c287b691f1a0f34c

## 要点

- ImageCls の学習率を学習の進み具合に応じて変える。
- 値の時間変化を表す共通部品 ProfiledValue を設け、最初の用途を学習率とする。
- 一定値のほか、linear warmup や cosine decay などの曲線を設定で表せるようにする。
- 各更新時に現在の位置から値を求め、optimizer の学習率へ反映する。
- 学習率専用のスケジューラーに閉じず、別の設定値でも同じ表現を使えるようにする。

## 内容

### 時間変化する設定値

`ProfiledValue<T>` は呼び出し元が渡す step に応じて値を返す。最初の用途は ImageCls の学習率で、更新に使用する値を Agent の `$agent learning_rate @learn` として記録する。部品自身は経験数や更新数のどちらを数えるかを決めないため、ほかの量にも同じ表現を使える。

### 曲線と並び順

定数は `value` をそのまま使う。線形・cosine・cosine restart は開始値、終了値、区間の長さで表す。restart の `cycle_mult` を2とすれば周期は毎回2倍になり、0と1の間なら1ステップを下限に短くなる。`phased` は `phases` リストに指定した順に一つずつ区間を進めるため、設定ファイル内の定義順には依存しない。

### 設定の読み取り

ImageCls では `ImageClsAgent.learning_rate.*` と、その設定に対する override を通常の prefix 解決順で読み取る。複合型の設定に限り専用 reader を使い、通常型には従来の読み方を保つ。一つの数値しか受けない旧 `ImageClsAgent.learning_rate` キーからの読み替えは設けない。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
