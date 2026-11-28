---
title: ProfiledValue の定数は裸のスカラー 1 行で書き、root の `.value` 記法は置かない
page_kind: サマリ
material: "[[docs/adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md|0064-profiledvalue-bare-scalar-reads-as-value-field]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-28
---
# ProfiledValue の定数は裸のスカラー 1 行で書き、root の `.value` 記法は置かない

## 元の材料

[ADR 0064](../../../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md) blob:8fb5162294486b89cd6ae759deeb4a520960e371

## 要点

- ProfiledValue の定数は従来と同じ X = v の1行で書く。
- root に X.value の別記法を設けない。
- 平らな設定キーでも X と X.type は共存できる。
- 区間内の定数では phase の value を使う。
- 定数があるだけで型を固定すると、後から schedule を指定できなくなるため採らない。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
