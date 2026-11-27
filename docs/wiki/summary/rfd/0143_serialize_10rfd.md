---
title: Run 全体の保存と再開の契約案
page_kind: サマリ
material: "[[docs/memo/0143_serialize_10rfd.md|0143_serialize_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-26
---
# Run 全体の保存と再開の契約案

## 元の材料

[RFD0143　Run 全体の保存と再開の契約案](../../../memo/0143_serialize_10rfd.md) blob:dd57470f502eba058227f320180afaa646444ff7

## 要点

- Run 全体の保存・復元の要求を記録した暫定メモである。設定名、保存形式、導入順は未決である。
- 重みを読み戻した後、Run に指定した学習率が optimizer に保存された値で上書きされた事例を起点とする。
- 同じ計算を続きから再開する使い方と、既存の重みを使って条件を変える使い方を区別する。
- Network、optimizer など、復元する状態と新しい設定で指定する値の優先順位を整理する。
- optimizer のパラメータ群を再設定する範囲は RFD0091 に切り出す。

## 内容

### Problem Statement

現行の checkpoint は Agent、より正確には `DefaultDQNAgent` 固有の archive であり、Run 全体を継続する仕組みではない。
この曖昧さが、今回 Adam/AdamW の learning rate で実害になった。

### 今回確認した Adam/AdamW の事実

新しい Run の Config から、現在指定された `learner.alpha` を持つ optimizer を構築する。
`Learner::Load()` が LibTorch の optimizer serialization を使って AdamW を load する。

### Solution Direction

設定では、単一の `auto_load_file` だけでなく、少なくとも次を表現できるようにする。
多数の個別 boolean を追加するのではなく、Run-level manifest と少数の profile、State group ごとの override で表現する方向を優先する。

### State Group の暫定分類

非同期 queue や prefetch buffer のような一時的 State は、そのまま dump することを既定にしない。Save 前に処理を quiesce し、所有モジュールの authoritative state へ反映したうえで保存するか、再構築可能であることを contract に明記する。

### Adam/AdamW に必要な contract

現在 Config と checkpoint options が異なる場合、checkpoint が黙って勝つのではなく、不一致と最終実効値を明示する。
learning rate など、retune 対象として明示した options は現在 Config を load 後に適用する。
options 間に互換性がない場合は、黙って一部だけ適用せず fail-fast する。

### Manifest と実効値の要求

checkpoint または Run serialization bundle は、データ本体とは別に manifest を持つ。manifest には少なくとも次を含める。

### Save / Load の安全性

Load は、schema、必須 group、Network shape、parameter identity、Config compatibility、source generation の整合を 全対象へ事前検証してから State を変更する。
明示した group を保存できない場合、黙って省略しない。`auto` profile だけが WARN 付き fallback を定義できる。

### 保存時と異なる device への Load

2026-10-15、GPU が学習中の Run で埋まっていたため、DropMerge の継続設定の smoke を `agent.device=cpu` で回したところ、次が判明した。
B256 の Run 2 つと B512 の Run 1 つが保存した checkpoint の 3 つとも同じ結果だった。同じ checkpoint を GPU の Run で読む継続は正常に動く。

### 候補を採る場合の決定性と等価性検証

同一の保存済み実行体を、同一 seed・同一 config・determinism 既定 ON で 2 回実行しても、`agent_close.anet` の raw SHA-256 は一致しない。
group payload に保存時刻、絶対 path、ホスト名、非決定な走査順（`unordered_map` 等）を混ぜない。可変情報は manifest 側へ寄せる。

### User Stories

長期 Run 実施者として、Run を停止しても Step と schedule を含めて同じ進行位置から再開したい。
実験者として、Network と Adam moment は引き継ぎつつ learning rate だけを変更したい。
実験比較者として、Config に書いた値と load 後の実効 optimizer 値が異なる場合に、その差を見落としたくない。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
