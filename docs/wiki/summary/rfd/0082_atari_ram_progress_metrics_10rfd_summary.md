---
title: Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD
page_kind: サマリ
material: "[[docs/memo/done/0082_atari_ram_progress_metrics_10rfd.md|0082_atari_ram_progress_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-28
---
# Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD

## 元の材料

[Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD](../../../memo/done/0082_atari_ram_progress_metrics_10rfd.md) blob:5c83241a1bed08a1a4cfb9ebefc2a6f18bba1521

## 要点

- Atari のゲーム固有の進行を、スコアだけでは識別できない場面で測る。
- Kung-Fu Master の長いゲームでは、制限時間まで持ちこたえたか、ボスに当てたか、面を抜けたかが従来の記録から分からなかった。
- ゲームごとに RAM 番地と意味のあるラベルを設定し、汎用のメトリクスに対応づける。
- 毎フレーム読む値を、1ゲーム内の最大・最小や状態遷移の回数へ集約する。
- ALE の RAM を使った調査条件と通常の学習 Run を区別し、番地が実際に表すイベントを確認する。

## 内容

### Context / Problem Statement

ボス討伐を目標に置くため。 ゲームごとに「ボスに当てた回数」「倒した回数」「抜けた面」を数えたい。
スコアの平均では見えない、方策の質的な違いを行動で比べるため。 例えば NN 構成の違いを、「先へ進むか」「ボスと戦うか」で比べられるようにしたい。

### 契約

ブロックは今のゲーム以外も同じファイルに並べてよい。env は `AtariEnv.game` に一致するブロックだけを使い、他のブロックは検証だけする。
v0 を `reset_game()` 直後の値、v_t を t フレーム目の `act()` 後の値とする。
番号は metrics 側の固定の接点だが、全ゲーム共通の大まかな意味を持たせる。trace の列をゲームをまたいで同じ読み方で読めるようにするためである。

### 設定例

kung_fu_master では `ram_metric.[1]` が 1 以上なら面 1 を抜けている、`ram_metric.[2]` が 1 以上なら少なくとも 1 回ボスを倒している、`ram_metric.[3]` が 1 以上ならボスに当てている。
phoenix の `ram_metric.[1]` は母艦ウェーブの突破を数えない。抜けたウェーブの総数は `ram_metric.[1]` + `ram_metric.[2]` になる。
breakout の `ram_metric.[1]` は 0 か 1 で、`game_score.ge.[432]`（1 枚目の壁を消した）と一致する。`@breakout` の ge432 系は `mean.ram_metric.[1]` で置き換えられる。ge600 もスコアの百の位 0xCC の `reach_count:6` で同じ値になる（`score_ge600`）。残すかは決めていないので番号を付けていない。付けるなら慣例の 1〜3 に当たらないので 4 以降になる。
tag 番号は 50 番台を使い、mean と EMA の並べ方は既存の 20/21・22/23 と同じにする。max は 56〜58 で、既存の 16・24 と同じく組の後ろに置く。40 番台は `@breakout` の `game_score.ge` 系が使っている。

### 実装指針

畳み方は純粋な状態構造体（`Begin(v0)` / `Observe(v)` / `Value()`）にし、ROM 無しで単体テストできる形にする。
`ale_->act()` の 5 箇所（`ApplyFireReset` の 2 回、`ApplyResetActions` の NOOP、soft reset の NOOP、`Step` のループ）を 1 つの補助関数へ集約し、フレームごとに観測する。`reset_game()` の直後（`ApplyFireReset` 内 2 箇所、`ApplyResetActions` 内 1 箇所、`Reset` の hard / soft 経路）で仕切り直す。
`ANET_PROFILE_SCOPE` は既存の `Step` の範囲で足りる。フレームごとの比較は細粒度なので別スコープにしない。

### テストの方針

ROM あり（無ければ SKIP。先行例は `AtariEnv_test.cpp` の Pong spec テスト）。
既存の trace / scalar が変わらないこと: `ram_metric` キーを使わない構成で行が変わらない。ADR 0037 の受入方式（編集前 baseline との同 seed 比較）に従う。
既定の scalar 定義: `Atari.txt` のプリセットテストで、4 群 33 本（mean 12・EMA 9・max 12）の tag・`source_key`・EMA の有無と係数を確かめ、残りの項目（event・eval 名など）が同じ群の `game_score` の定義と一致することを見る。

### 決定の記録

番地をキーに直書き（原案 A、`ram_min.[0xCC]`）: キーから意味が読めず、metrics 定義がゲーム別になる。
ラベルをキーにする（`ram.[boss_hit]`）: イベントの数だけ metrics が増え、ゲームを変えるたびに定義を書き換える。
購読キーからの自動追跡: env は metrics 解析より先に構築され、配線が無い。

### 既知の限界と後続

phoenix の周回。 0xCA と 0xFC の意味は 2 周ぶん（母艦 2 回）しか見ていない。
beam_rider の確認範囲。 ボスを倒せたのはセクター 2・3 の 1 回ずつで、セクター 1 は魚雷の総当たりでも倒す枝が見つからなかった。セクター 4 より先は見ていない。実 Run で `boss_kill` が 1 ゲームの中で `sector_clear` を超えないことで確かめる。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
