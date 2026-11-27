---
title: コア処理の性能改善候補の再点検
page_kind: サマリ
material: "[[docs/memo/0180_core_perf_candidates_202607_10rfd.md|0180_core_perf_candidates_202607_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-26
---
# コア処理の性能改善候補の再点検

## 元の材料

[RFD0180　コア処理の性能改善候補の再点検](../../../memo/0180_core_perf_candidates_202607_10rfd.md) blob:9ffa298fd7fac667266301330c89d5a4368191b8

## 要点

- 既存のレビューが挙げた5件を再点検する暫定案である。各項目が現行コードに当てはまるか、改善に値するかを判断する前の候補である。
- 過去の調査で挙がった五つの候補は、コード変更後も残っているかと実際の費用を確認してから扱う。
- ImageCls の更新中の GPU 同期と、Network の中間値の取り出しが通常時にもかかるかを調べる。
- BatchEnv の lane ごとの Tensor 操作と結果確保、Double DQN を使わない経路の target forward の回数を調べる。
- MetricsLogger の Flush は学習ループ本体の外にあるため、呼び出し側が何 step ごとに戻るかを追う。

## 内容

### 目的と問題

2026-07 のコアの詳細レビューが挙げた性能の改善候補 5 件を、repo の中に正本として置き、今も残っているかを確かめたうえで、直すかどうかを 1 件ずつ決められるようにする。
候補が Claude Code の memory にだけあり、Codex からも、ほかの RFD づくりからも引けない。

### 起草時点の現状

2026-07 の観察を、今のコードで軽く確かめた結果である。費用は測っていない。

### 提案の骨子

5 件それぞれについて、今のコードで残っているかを確かめる。残っていないもの（直っている、形が変わって当てはまらない）は、理由を本文に書いて閉じる。

### 候補を採る場合の成功の測り方と受入の目安

5 件それぞれに「残っている／直っている／当てはまらない」の判定と根拠（ファイルと行）が、本文にある。
直したものは、step あたりの時間が、交互に回した複数の Run のブレ幅を超えて縮み、同 seed の metrics checksum が直す前と一致する。

## 主要な概念

- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
