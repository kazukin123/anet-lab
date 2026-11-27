---
title: Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)
page_kind: サマリ
material: "[[docs/memo/done/0046_workspace_10rfd.md|0046_workspace_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)

## 元の材料

[Workspace 機構(Run 分類の物理フォルダ化と Runner/Viewer/optuna 統合)](../../../memo/done/0046_workspace_10rfd.md) blob:d50e7ad59d38966f964f544448393a458d147752

## 要点

- 異なる環境・探索の Run を物理フォルダで整理する workspace を導入する。
- workspace は入力設定、Run の成果物、Optuna の探索結果をまとめる。
- Runner は選択した workspace を使い、未指定の利用には既定の workspace を用意する。
- Viewer と Optuna にも選択した workspace を渡し、手動の退避フォルダを個別に探す運用を減らす。
- Run をフォルダ単位で移動・退避・復元できる性質を保つ。

## 内容

### Run 分類の単位

入力設定、Run の成果物、Optuna の探索履歴を一つの workspace フォルダに置く。内部の Run を別の workspace に移したり、workspace 全体を退避したりすれば分類も一緒に移る。フォルダの外に分類用タグや DB を持たない。アプリが記録する履歴は最近開いた場所だけを表す。

### Runner の選択と設定

選択の優先順は CLI の `--workspace`、起動時のダイアログ、最近使った workspace、`_default`。相対パスは Runner の `workspaces/` を基準にし、絶対パスも受け付ける。新規作成時と、明示的に選んだ既存フォルダに `config/_main.txt` がなければ `apps/runner/config/_workspace_template.txt` から補う。既存の入力設定や Run は保持する。履歴と選好はそれぞれ `history.txt` と `prefs.txt` に置く。

workspace モードでは最終的な `app.runs_dir` が選択先の `runs/` を指すことを設定合成の後に確認する。`--config` を指定した起動はそのファイルだけで設定を解決する。workspace 選択との同時指定をエラーとし、Optuna が生成した設定への二重適用を防ぐ。

### Metrics Viewer の切り替え

Viewer は workspace 一覧ディレクトリ直下の一つを開く。SQLite のロックを保証できない UNC パスを拒否する。各リクエストと取り込み周期は開始時に、workspace、切替番号、メモリ上のキャッシュをまとめて取得し、終了まで同じ組を使う。別 workspace に同名 Run があってもキャッシュが混ざらない。

取り込みと切替を直列化し、同じ切替先が同時に要求されても一度だけ切り替える。以前の workspace を使っている処理には完走を許し、最後の利用者が離れた時点で gzip ストリームなどを閉じる。同じ workspace に戻ったときも同じ DB へのアクセスを直列化できるよう、DB のロックは絶対パスごとに共有する。

### Optuna のパスと事前検証

`<ws>` を解決済みの workspace とすると、Run は `<ws>/runs` に固定する。Optuna の既定値は SQLite の保存先が `<ws>/optuna/optuna.db`、成果物が `<ws>/optuna/artifacts`、実行ログが `<ws>/optuna/harness.log`。ログを作るのは `run-trial` と `run-study` に限る。run 系から `--runs-dir` を廃止し、storage・成果物の個別指定も `<ws>/optuna` 配下に限る。

`summarize-study` は複数 workspace の集約が目的なので、source と target を別に指定できる。target の DB・成果物の指定を省くと対応する source を継承する。各コマンドは、パスの解決と必須入力の存在確認をすべて済ませてから出力先を作成する。入力に誤りがあるだけでフォルダや DB が新規作成されることを防ぐ。

### 移行と検証

既存 Run と各 trial の代表フォルダは選んだ workspace の `runs/` に、Optuna DB・ログ・成果物は `optuna/` に手動で移す。移動前に SQLite の利用元を停止し、`optuna.db-wal` や `optuna.db-shm` が残っていれば DB 本体と一緒に移す。Viewer の切替では同名 Run の分離、切替直後の処理中クエリ、旧ファイルが閉じられて Windows 上でも移動できることを確認する。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
