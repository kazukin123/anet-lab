---
title: 選んだモードや意図した休止で値が結果を変えない設定キーは、知らせずに値域だけを検証する
page_kind: サマリ
material: "[[docs/adr/0091-config-keys-idle-under-chosen-mode-or-intended-pause-are-range-checked-not-notified.md|0091-config-keys-idle-under-chosen-mode-or-intended-pause-are-range-checked-not-notified]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-26
---
# 選んだモードや意図した休止で値が結果を変えない設定キーは、知らせずに値域だけを検証する

## 元の材料

[ADR 0091](../../../adr/0091-config-keys-idle-under-chosen-mode-or-intended-pause-are-range-checked-not-notified.md) blob:59a47ae19fd9c3f5fa8edaa1a30d039a386fe218

## 要点

- 選ばなかったモードや意図した休止のため効かない設定キーについては、個別の通知を出さない。
- キー単独の値域は休止中も検証し、不正なら止める。
- 他の設定との組み合わせは該当する機能が有効なときに検査する。
- 両方が効いて結果をゆがめる組み合わせへの警告と、休止自体の通知は残す。
- 存在するが非選択のキーと、どこからも読まれないキーの検出を区別する。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
