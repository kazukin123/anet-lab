# PRD070 device 設定統一 実装メモ

## 概要

`agent.device`、`env.device`、`run.eval_device` を `auto | cpu | cuda | cuda:N` に統一する。既定値は Agent と評価が `auto`、Env が `cpu`。`config_data` と `config/*.txt` は設定値を保持し、採用した device は各所有クラスの個別 JSON に記録する。このメモは [PRD070](070_device_spec_auto_10prd.md) と 2026-09-23 の合意を反映した実装の正本とする。

## 主な変更

- `anet::ParseDevice(const std::string&) -> torch::Device` を `tensor_util` に追加する。前後空白を trim し、大小文字を無視する。`cuda:N` は非負整数だけ受け、`auto` は `torch::cuda::is_available()` だけで `cuda`（current device）か `cpu` へ解決する。`MakeDevice(int, int)` と旧メンバを削除する。
- `InitRL(const BackendConfig&, const ConfigData&)` は NN 初期化前に3キーを検証する。明示 CUDA が利用不可なら、明示した全キーと `auto` への変更方法を1件のエラーに含める。旧キー専用検出・互換読みは置かない。
- `RunManager`、`BatchEnvBuilder`、`DefaultAgentFactory` がそれぞれ採用した device を `LOG::info` と JSON に記録する。`json/run.json.data.effective_eval_device`、`json/env.json.data.effective_device`、新設する `json/agent.json.data.effective_device` を使う。各 JSON の `eval_device` / `device` と `config_data`、`config/*.txt` は指定値のままにする。`auto`→CPU は WARN しない。
- 現用コード・同梱設定・現行設計文書・現用テストを新キーへ移す。共通設定と CartPole の CUDA 固定は `auto`、GridMaze MuZero の明示 CPU は `cpu` とする。過去の Run artifact、固定 golden、過去データ読取テストは変更しない。

## テスト

- Public interface / surface: `ParseDevice`、設定読込から `InitRL` と `RunManager` / builder / factory まで、Run ログと `json/run.json` / `json/env.json` / `json/agent.json`、`config_data` と `config/*.txt`。
- 優先 behavior: `auto` の解決と所有クラスの個別 JSON 記録、文法と `cuda:1`、明示 CUDA 利用不可時の全キー列挙、Env の CPU 既定、同梱設定の移行。
- TDD 順序: 最初に設定読込から実効 device の個別 JSON 記録までを1テストで通す。その後は各 behavior ごとに1テストの RED → 最小実装の GREEN を繰り返す。テストは private method でなく公開面と生成 artifact を観測する。
- `config_data` と `config/*.txt` は `auto` を保持し、個別 JSON は実効値を持ち、`auto`→CPU で WARN が出ないことを確認する。

## 検証

- `VsDevCmd.bat` 経由で `x64-Debug` の `anet-core-test` をビルドし、device 関連 Catch2 テストを実行する。CUDA 非表示の別プロセスで CPU 分岐を確認する。
- `check_default_leaves.py` と `check_default_leaves_test.py` を実行する。現用コード・設定・文書の旧キーを検索し、設定比較では device キーの意図した差分だけを確認する。
- `git diff --check`、変更対象の UTF-8 / LF、保存した PRD と本メモの内容を確認する。

## 前提

- 評価キーは `run.eval_device`。旧キーは AGENTS.md のクリーンブレーク方針に従い専用検出しない。
- 実効値は設定値と区別し、個別 JSON の追加フィールドに置く。複数 GPU 自動選択、CPU-only ビルド、無言終了は PRD070 の範囲外。
- 既存の未コミット変更を保持する。
