---
title: eval 定義とスケジュールの namespace 分離
page_kind: サマリ
material: "[[docs/memo/done/0052_eval_schedule_separation_10rfd.md|0052_eval_schedule_separation_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-26
---
# eval 定義とスケジュールの namespace 分離

## 元の材料

[eval 定義とスケジュールの namespace 分離](../../../memo/done/0052_eval_schedule_separation_10rfd.md) blob:d8c1ac2ed3702dfbd88172221bb49bcc8856c171

## 要点

- 評価用 Env・Runner の定義と、自動評価をいつ実行するかの設定を分離する。
- train.eval のタグは評価内容を表し、train.eval_schedule の同名タグがその起動条件を表す。
- interval とバックグラウンド実行の設定をスケジューラーの名前空間へ移す。
- 同じタグへのスケジュールは設定キーの一意性で一つに決まる。
- 自動スケジュールを付けずに評価定義を利用でき、定義と起動条件を個別に変更できる。

## 内容

### Context（背景・目的）

`train.eval.[tag].*` はキーが 1 つでも存在するとタグが実 eval として列挙され、`interval` 省略時は既定 100 で背景 EvalRunner が起動する。このため、EvalPanel 用の設定キャリアとしてだけ `[eval_panel]` タグを書いていた LunarLander.txt（`env.limit_step` と `eval_config_tag` のみ、interval 無し）で、全 Run に意図しない背景 eval が寄生していた。Atari では同構造が SDL 音声デバイスの取り合い（先に生成された背景 eval 側がデバイスを握り EvalPanel が無音）として顕在化し発見された。

### 契約

スケジュール `train.eval_schedule.[tag]`: キーは `interval`（必須、int、`>= 0`。`0` = 明示 OFF、負値エラー）と `use_background`（省略可、既定 true）。消費者は EpisodeEvalObserver のみ。
name 予約（`RegisterEnvName`）+ `ValidateConfig` は行う。Env / Runner / Observer / background worker は生成しない。

### 実装範囲

EpisodeEvalObserver（observers.cpp / observers.hpp）は変更しない。コンストラクタ引数 `eval_interval` の意味・消費は不変。

### config 移行インベントリ（挙動等価、機械的移動）

移行後、リポジトリ管理下に `train.eval.[tag].interval` / `use_background` / `train.eval_interval` の非コメント記述が残らないこと（grep で確認）。
補足: 既発行の暫定止血タスク「LunarLander に interval=0 を 1 行追加」は本設計の実装で不要化される。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
