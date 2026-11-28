---
title: "メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す"
page_kind: サマリ
material: "[[docs/memo/done/0069_metrics_trace_channel_10rfd.md|0069_metrics_trace_channel_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す

## 元の材料

[メトリクス trace チャネル: 統計（scalar）とは別に「1 件 = 1 レコード」を識別子付きで出す](../../../memo/done/0069_metrics_trace_channel_10rfd.md) blob:90de8911a4868fb96f1b4a1f2c054b46579fdd5d

## 要点

- scalar は step・tag・値だけを持ち、平均や最大へまとめた後に個々のエピソードを識別できない。
- trace チャネルは個体やセッションの識別情報を付け、1件の出来事を1レコードとして metrics.jsonl に書く。
- 定義、書き込み API、出力設定を統計用 scalar と分ける。
- 評価で採用された個々のエピソードとセッション全体の終了を別々に通知する。
- 複数 lane・複数エピソードでも、集計したスコアから失われていた結果の分布を追えるようにする。

## 内容

### 現行コードで確定している事実（2026-09-05 時点）

Atari-5 はエピソード長がゲームごとに違うため、同じタグがゲームごとに違う集約になる。
eval の trace は「セッションあたり N 行」なので、これよりさらに小さい。

### 問題

Atari-5 では 5 ゲームで λ が異なり、`mean.game_score` の意味がゲームごとに変わる。エラーも警告も出ない。
eval は N 本が 1 点に畳まれ、`std.` を足しても時間方向の分布（p10、閾値越え率）にはならない。
ゴール: 統計（scalar）と別に、「1 件 = 1 レコード」を識別子付きで `metrics.jsonl` へ出す trace チャネルの出力側を実装する。(a) 行契約、(b) JSONL 専用の書き口、(c) 設定 DSL での宣言、(d) チャネル別の定義レコード、(e) producer 第 1 号 = eval の採用エピソード（train は同じ仕組みで宣言だけ足せば動く状態にする）。
非ゴール: inspect_run / Viewer の trace 読み取り、912 の `model_version`、932 の forensic レベル、走行中に裾を見ること（移動窓統計は別件）、既存 scalar 集約タグの廃止、`min.` / `std.` 以外の集約 prefix 追加。

### 確定契約

裸トークンは全部キー。1 個以上、重複不可。値の取得順と定義レコードの `keys` 配列は宣言順とし、`data` オブジェクトのキー順序は保証しない。既存の `anet::json = nlohmann::json` を使い、順序保持専用の JSON 型・書き込み経路は追加しない。scalar の「最後のキー指定を採る」挙動は維持する。
`$ema` / `ema_alpha:` / `clip:` / `interval:` / `key:` は設定読み込み時に fail-fast（キーは裸トークンで書く）。`interval:1` のように既定値と同じ指定も拒否する。
`$step` の既定は scalar と同じ（`@episode_end` → `exp_step`）。scope 既定は `$train`。
eval scalar を `@episode_end` に残すと、採用エピソードごとに `EvalSessionEnv::GetScalar("mean.game_score", -1)` の途中経過の平均が点として出る。誤読を生むので黙って通さない。
step 軸既定: `SESSION_END` は `EXP`（`@episode_end` と同じ）。`OwningRunner` は変更不要（EVAL + `@train` 以外は `"train"`）。`metrics_def_names::EventToken` に `"session_end"`。

### 実装ノート

`static` helper は状態非依存の純粋パースに限る。テストのために本体へ test-only API を足さない（`LastAdoptedGroups()` は EvalRunner が使う production API）。

### 受入条件

scalar の挙動不変: `$eval.[x] @session_end $env mean.score` と `$runner mean.episode_return @session_end $eval.[x]` が、変更前の `@episode_end` と同じセッション集約を1点だけ出す。途中経過を scalar として記録しない。
読み手の無害性: 変更後 Run を Metrics Viewer が `ERROR` にせず取り込み、既存 scalar を利用できること、`inspect_run.py runs` / `tags` が警告なしで通ることを確認する。trace / scalar の同名 tag ケースも reader 検証に含める。

### フェーズとゲート

trace の `@train` / `@learn`（per-step lane fan-out、per-sample update_result）。
trace の `@session_end`（セッション 1 行に `model_version` 等）。
inspect_run の trace 集約（分位点・閾値越え率）。

### 却下した案（経緯）

eval scalar のトークン `@episode_end` を残して内部で SessionEnd に束縛: 設定は無傷だが「同じトークンがチャネルと scope で別イベントを指す」ことになり、型を分けた意味が薄れる。
省略・重複指定の扱いが未確定。後勝ちや既定値で指定の欠落を隠さない。scalar の既存挙動は維持。

### スコープ外（再掲）

走行中の viewer で裾を見ること（移動窓統計は別件。trace があれば後から consumer として乗る）。
`min.` / `std.` 以外の集約 prefix。集約 prefix の std は「同時完了した env 間のばらつき」であり、欲しいのは時間方向の分布で、それは trace から算出する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
