---
title: sampleable index 列の確保を持続バッファへ移す
page_kind: サマリ
material: "[[docs/memo/done/0077_valid_index_scratch_reuse_10rfd.md|0077_valid_index_scratch_reuse_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# sampleable index 列の確保を持続バッファへ移す

## 元の材料

[sampleable index 列の確保を持続バッファへ移す](../../../memo/done/0077_valid_index_scratch_reuse_10rfd.md) blob:f497f3ced8c7fbdb683db58a73f3d7fbacb7d440

## 要点

- 抽選可能な index の列挙では、一時 vector と Int64 Tensor を作り、同じ値をコピーしていた。
- ReplayBuffer が大きいと二つの領域が同時に生存し、確保のピークが増える。
- 必要な作業領域を持続バッファに移し、繰り返す大きな確保を抑える。
- 実際に列挙した有効な範囲だけを後の利用者へ渡す。
- 抽選可能な index の順序と内容を保ち、容量や lane 数を変えた条件でも検証する。

## 内容

### 列挙時のメモリ使用

ReplayBuffer が抽選できる index を求めるたびに、容量ぶんの vector を確保し、有効な値をさらに Int64 Tensor へコピーしていた。容量1,572,864では一方だけで約12.6 MB、両方が同時にあると約25 MBになる。4.318時間で停止した Run の `bad_alloc` はこの確保で発生し、checkpoint も残らなかった。特定したのは失敗した場所で、外部のメモリ圧そのものを解消する変更ではない。

### 持続バッファと返り値の寿命

`ValidIndexManager` の構築時に CPU の Int64 Tensor を一度確保し、有効な index を従来順に書き込む。呼び出し元には使用した長さのビューを返すので、抽選のたびに容量ぶんを確保したりコピーしたりせずに済む。空のときも要素数0の Tensor を返し、バッチに必要な件数との比較は同じ意味を持つ。

返り値の内容を保証するのは同じ管理器で次に列挙するまで。`GetTensorVector` と `DumpToLog` はロック解放後も内容を使うため、`metadata_mutex_` を保持している間に `clone()` する。複製後の管理器が同じ作業領域を共有しないよう管理器のコピーも禁じる。外部 vector を指す `from_blob` は寿命管理を呼び出し元へ負わせるため採らない。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
