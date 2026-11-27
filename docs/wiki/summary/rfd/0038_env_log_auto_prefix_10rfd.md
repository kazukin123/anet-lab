---
title: Envログ name prefix 自動付与 PRD
page_kind: サマリ
material: "[[docs/memo/done/0038_env_log_auto_prefix_10rfd.md|0038_env_log_auto_prefix_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/312_util.md|312 ユーティリティ]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-26
---
# Envログ name prefix 自動付与 PRD

## 元の材料

[Envログ name prefix 自動付与 PRD](../../../memo/done/0038_env_log_auto_prefix_10rfd.md) blob:a2f0622dae1a5916d01517873db78d5f1dddbf87

## 要点

- 各 Env が名前をログ本文へ個別に連結すると、付け忘れや書式の違いが生じる。
- Logger の構築時に Env 名に応じた prefix を一度だけ決める。
- 各ログ行は prefix の詳細を知らずに本文を書ける。
- lane 名がすでに角括弧を持つ場合の二重の囲みを取り除く。
- immutable な名前と組み合わせ、学習と評価のログが混じっても発行元を読めるようにする。

## 内容

### 問題

`[<Env name>] `の囲み`[ ]`は、lane表記(`train[37]`)が既に`[ ]`を含むため二重になり(`[train[37]]`)、視覚ノイズが大きい。本文にも`Rank [ 5 ]`のような`[ ]`過多の箇所がある。

### 解決方針

レベルメソッドは`LOG::info()`等と同じレベル対応(`info`=`wxLOG_Message`、`verbose`=`wxLOG_Info`、`warn`=`wxLOG_Warning`、`error`=`wxLOG_Error`)とする。

### 実装上の決定

prefix文字列はEnv構築時に一度だけ構築する。出力行ごとの追加コストはprefix 1個のstream書き込みのみで、従来の手書き3項連結(`"[" << GetName() << "] "`)より減る。ログレベル無効時にも整形が走る点は既存`LOG::`直用と同一であり、悪化しない。debug系はguardにより無効時コストゼロを維持する。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [312 ユーティリティ](../../concepts/framework/312_util.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
