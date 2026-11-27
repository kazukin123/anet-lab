---
title: 設定キーの root と値の所有者の対応
page_kind: サマリ
material: "[[docs/memo/0155_config_root_ownership_alignment_10rfd.md|0155_config_root_ownership_alignment_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-26
---
# 設定キーの root と値の所有者の対応

## 元の材料

[RFD0155　設定キーの root と値の所有者の対応](../../../memo/0155_config_root_ownership_alignment_10rfd.md) blob:483f0ecad9203bebe33dee46bb4036b3914ed4c9

## 要点

- RFD0061 で扱った設定名と、所有者に合わせる一般則の候補を区別する。本書でほかの設定キーの移行先は確定していない。
- RunManager が読む値に train の prefix が付くなど、設定の階層が実際の所有者とずれている箇所を棚卸しする。
- Run の seed、各 Runner の種類、Run 名などの読み手と用途を調べて候補を分ける。
- 起動アプリ全体の値、Run に属する値、Train Runner 固有の値を区別する。
- 既存設定や保存した Run を参照する道具まで含めて、名前の変更の影響を追えるようにする。

## 内容

### 問題

現用 root を「読み口(消費 Config の default_prefix)」「所有者(生成・保持するオブジェクト)」「フレームワーク概念」で並べると、ずれは 2 件に留まらない。
Module Config の dump は prefix 名がそのままファイル名になる(`config/<prefix>.txt`。metrics_logger.cpp `Log(const Config&)` が `GetConfigPrefix()` を tag にする)。root を変えると dump のファイル名も変わる。

### 規約案(裁定対象)

案 O(所有=オブジェクト階層をそのまま写す): 止め方がないと RunManager が所有する `env` / `agent` も `app.run.env.class_id` のように app 配下へ沈む。深くなるだけで概念は増えない。
案 C(消費 Config クラスの略称を root にする): 全読み口が root 直下に並ぶフラットな体系になる。`net` を Agent 配下へ移した ADR 0030 と、スロット(`train.eval.[tag]`)が `env.*` 上書き層を所有する ADR 0038 に逆行する。読み口が 2 つある設定(具象 Env の既定 prefix とスロットの override prefix)を表せない。
現に ADR 0030 / 0038 / 059 が採っている形は両者の折衷であり、それを規約として明文化する。

### 棚卸し結果と候補

`train.eval.[eval_panel].*`(EvalPanel が鏡写しする configured eval tag)も機械的に `run.eval.[eval_panel].*` になり、`app.eval_panel.eval_config_tag = eval_panel` の名前参照は root をまたぐ参照のまま成立する。
MetricsLogger は app が所有する process singleton で、R2 のとおり `app.` 配下が正しい。
`env.*` / `agent.*`: スロット(`class_id` + 実行方式・device)。R1 により root は概念単位なので、RunManager 所有でも `run.env` にはしない。
`ImageDataset.[key]`: process 横断のカタログで、差し替わらない部品の class 名 root(R4 の境界例)。ImageCls 以外の利用者が出るまで据え置き。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
