---
title: device 指定の auto 対応とキー体系統一
page_kind: サマリ
material: "[[docs/memo/done/0070_device_spec_auto_10rfd.md|0070_device_spec_auto_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# device 指定の auto 対応とキー体系統一

## 元の材料

[device 指定の auto 対応とキー体系統一](../../../memo/done/0070_device_spec_auto_10rfd.md) blob:f2ec043304405ecca345770ca1feceacfedab842

## 要点

- CUDA を要求した設定では、CUDA を使えない機器で正常に起動できなかった。
- Agent、Env、評価のデバイス指定を文字列のキーに統一する。
- auto、cpu、cuda、番号付き cuda を認め、値の前後空白と大文字・小文字を正規化する。
- 自動選択と明示要求を区別し、不正な値はエラーにする。
- 非搭載機での致命エラー表示は別 RFD に分け、ここでは環境に合った設定を選べることを扱う。

## 内容

### Context（背景・目的）

auto が無い。 実行環境に GPU があるかどうかを設定側から表現できない。

### 比較履歴

案 A を採用する。今回の詰まりの遠因が「同じ概念が 2 系統 3 組に散っていること」なので、キーを減らす方向が本筋。

### 受入基準

CPU 機で `agent.device = cuda` を明示すると NN 構築前に停止し、cuda を指定している全キーが列挙される。
`env.device` 未指定の Run は従来どおり CPU env で、既存 Run と挙動が一致する。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
