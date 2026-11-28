---
title: Replay の複数バッチ先読みと決定性
page_kind: サマリ
material: "[[docs/memo/frozen/0133_async_loader_ndeep_10rfd.md|0133_async_loader_ndeep_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# Replay の複数バッチ先読みと決定性

## 元の材料

[RFD0133　Replay の複数バッチ先読みと決定性](../../../memo/frozen/0133_async_loader_ndeep_10rfd.md) blob:7756a8ee38cecb2de35cefb3237704d2e32067b9

## 要点

- 複数バッチの先読みは凍結中である。既存の1バッチ先読みが律速だと実測できた場合に再開する。
- 現在の1バッチ先読みでは Sample と転送が一体で、優先度更新との順序を待ち合わせで決める。
- 複数バッチを先読みすると、全件待てば処理が直列になり、一部だけ待てば優先度木を読む順序が変わりうる。
- 候補は caller が同期的に抽選し、データ抽出と転送だけを並行化する構成である。
- 深さを増やすには完了キュー、古いバッチの扱い、Push と抽出の整合も決める必要がある。

## 内容

### なぜ wait だけでは N-deep にスケールしないか

つまり wait 方式は深さ N で「全待ち＝直列化」か「一部待ち＝非決定」の二択になり破綻する。

### なぜ split

split＝`SampleIndices` を caller thread で同期実行 / `Extract`+`To` を async。現行 1-deep はこの seam を持っていないため、N-deep 化ではまず narrow な plan/fetch seam を追加する必要がある。

### 移行時にやること

`UpdatePriorities`-wait を撤去（split で不要）。整合性は storage lock のまま。
再現性テストを N-deep でも：同 seed 2 run の sampled index 列一致。
stale 深さが N に応じて深くなる（新規経験のサンプル遅延が増える）点を学習側で許容できるか確認。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
