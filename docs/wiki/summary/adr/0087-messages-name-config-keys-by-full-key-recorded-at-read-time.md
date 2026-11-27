---
title: 通知が示す設定キーは完全キーで書き、完全キーは読み口が読むときに記録して部品へ運ぶ
page_kind: サマリ
material: "[[docs/adr/0087-messages-name-config-keys-by-full-key-recorded-at-read-time.md|0087-messages-name-config-keys-by-full-key-recorded-at-read-time]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/312_util.md|312 ユーティリティ]]"
updated: 2026-11-26
---
# 通知が示す設定キーは完全キーで書き、完全キーは読み口が読むときに記録して部品へ運ぶ

## 元の材料

[ADR 0087](../../../adr/0087-messages-name-config-keys-by-full-key-recorded-at-read-time.md) blob:ffd64f482a04a397fc7f042657f7c9d2883a21f6

## 要点

- 設定についてのエラー・警告・通知では、実際に読み取った完全キーを示す。
- 共通部品で相対キーや特定の Agent 名を組み立てると、貼り戻しても効かないキーを案内してしまう。
- 既定・上書き・共有設定のどこから得たかは設定の読み取り時に記録する。
- 記録したキーを設定値の型とともに利用部品へ渡す。
- 利用側で読まなかった設定について完全キーを作り出さない。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [312 ユーティリティ](../../concepts/framework/312_util.md)
