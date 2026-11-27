---
title: 環境遷移変更後の性能再測定と DropMerge の追加費用
page_kind: サマリ
material: "[[docs/memo/0179_rfd0097_performance_remeasurement_10rfd.md|0179_rfd0097_performance_remeasurement_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
updated: 2026-11-26
---
# 環境遷移変更後の性能再測定と DropMerge の追加費用

## 元の材料

[RFD0179　環境遷移変更後の性能再測定と DropMerge の追加費用](../../../memo/0179_rfd0097_performance_remeasurement_10rfd.md) blob:a03692b73c213e8e4353d1fc46b853cac995a33d

## 要点

- 性能の再測定を扱うことは採用された。原本は暫定で、計測済みの比較と、値の所有方式を変えた後の未測定の性能を区別する。
- RFD0097 の挙動の比較では7構成で scalar・trace 計728,534点が一致したが、性能はばらつきのため判定できない条件が残った。
- 同じ構成で5反復し、各 Run の後半の速度中央値を取り、その中央値を構成値とする。
- 総実行時間に依存せず、基準側のばらつきも見て比較できるかを判断する。
- DropMerge の serial 構成で観測した中央値6.9155%の低下は独立に原因を調べる。
- native 値を保持して要求時に Tensor 化する修正後の性能は未測定として扱う。

## 内容

### 背景と問題

各Runは速度の最終報告stepの半分を超える区間の中央値、構成値は5反復の中央値。総実行時間は指標に使わない。DropMerge serialの中央値6.9155%低下は、beforeの大きなばらつきと別に調査対象として残す。

### 分かっていることと証拠

正式性能46Runはseed=1・512000経験・厳密な決定的backend・同期評価で正常終了し、凍結実行体と引数を監査済み。4構成ともbefore全5本の時間/資源以外のscalar/trace全点が完全一致した。数値系列の差による速度差ではない。
Gridの既存NVTX診断ではEnv/学習の重なりが保たれ、WARN量の急増も見られない。正式反復のばらつきの原因は未確定。

### 取り組むこと

保存済みの正式条件と資材を保持し、4構成のばらつきを再現・調査する。資源時系列、外部負荷、実行順等を追い、原因が分かる前に測定値の選別や除外を増やさない。
条件を安定させたうえで、同じbefore/after交互手順・指標・3組/不安定時5組・2%基準で再測定する。条件を変更する場合は内容と理由を記録し、両側へ同じ条件を適用する。

## 主要な概念

- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
