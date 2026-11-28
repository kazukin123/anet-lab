---
title: Atari の観測の縮小方式を設定で切り替える PRD
page_kind: サマリ
material: "[[docs/memo/done/0099_atari_resize_mode_10rfd.md|0099_atari_resize_mode_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-28
---
# Atari の観測の縮小方式を設定で切り替える PRD

## 元の材料

[Atari の観測の縮小方式を設定で切り替える PRD](../../../memo/done/0099_atari_resize_mode_10rfd.md) blob:b306131b9bce313a597bcb51e4f9cf142ccad8a9

## 要点

- Atari 観測を210×160から縮小するときの方式を設定で選び、Qbert で前処理だけの差を比べる。
- 従来の adaptive average pooling は、非整数比の窓に入る元画素を等しい重みで平均する。
- 面積加重方式は元画素と出力区間の重なりに応じて寄与を変え、OpenCV INTER_AREA と対応づけて検証する。
- 既存方式を既定にし、Reset と Step の両方で同じ処理が使われるようにする。
- 同じ入力での数値確認を機能の受入とし、Qbert の学習改善は別途確かめる仮説として扱う。

## 内容

### 根拠と確認状況

現行のResizeGrayscaleは、入力をfloat32にして`interpolate(mode=torch::kArea)`を実行し、`round`・0〜255への制限・uint8への変換を行う。AtariEnvには、Resetから使うCaptureObservationとStepの2経路に呼び出しがある。

### 確定事項と実装契約

各出力画素に対する重みは非負で合計1、整数比では通常の箱平均になる。出力84×84では元の各軸の寄与は最大4画素以内だが、この上限を小さいscreen_sizeにも適用しない。より広い元領域を覆う場合も同じ面積の規則で計算する。

### 性能の見積もりと内部方式を決める条件

通常の1 laneのStepは1回縮小する。終端してResetも起こる場合は、その観測生成分が加わる。210×160→84×84で重なる元画素の組を直接数えると60,480寄与／lane、64 laneのBatch Stepなら3,870,720寄与である。これは画素数と区間の重なりからの計算量の見積もりで、採用する計算の分け方を指定しない。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [Qbert](../../concepts/subjects/qbert.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
