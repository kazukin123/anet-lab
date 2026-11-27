---
title: Atari の本流 Run プロファイルの整理
page_kind: サマリ
material: "[[docs/memo/0118_atari_run_profile_flatten_10rfd.md|0118_atari_run_profile_flatten_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-26
---
# Atari の本流 Run プロファイルの整理

## 元の材料

[RFD0118　Atari の本流 Run プロファイルの整理](../../../memo/0118_atari_run_profile_flatten_10rfd.md) blob:30129ea28fd795cb1de993795bef18ceb56d3268

## 要点

- サンプル効率優先と実時間優先の各本流を、一つの平らな Run プロファイルで読めるようにする暫定案である。
- 本流の構成が他のプロファイルを連結する順番に依存しない形にする。
- 本流・機能・予算への三分類、個別の移行方法と検査は、未決表の既定を採る場合の案である。
- 実験時に切り替える機能と予算を区別し、同じ本流から設定を選べるようにする。
- 移行の前後で解決した全設定キーと文字列値を比較し、実効設定が保たれたか確かめる。

## 内容

### 起草時に困っていたこと

Atari の実時間優先の既定は11個、サンプル効率優先のバッチ側の構成は14個のプロファイルを連結していた。組み合わせを展開するまで、実際に選ばれる値や二つの本流の違いを読み取りにくかった。

### 未決表を前提とした構成案

起草時の案では、土台との差を本流に並べ、参照される機能だけを後から足す。予算は終了ステップなど実行長に必要な値に絞る。Rainbow 寄せをさらに本流にするかなどは未決で、原本の仕様・実装契約はこの提案を採用する場合の記述である。

### 移行と検証の案

バッチでの切り替えを本流・機能・予算の新しい名前に対応させる。移行前に解決された全キーと文字列値を保存し、同じ用途の新しい組み合わせの出力と比較して、プロファイル整理が学習条件を変えていないかを確かめる。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
