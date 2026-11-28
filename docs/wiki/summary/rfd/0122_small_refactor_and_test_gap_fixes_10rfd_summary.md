---
title: 小さなリファクタリングと学習契約の回帰試験
page_kind: サマリ
material: "[[docs/memo/0122_small_refactor_and_test_gap_fixes_10rfd.md|0122_small_refactor_and_test_gap_fixes_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/325_muzero.md|325 Agent個別実装 MuZero系列]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-28
---
# 小さなリファクタリングと学習契約の回帰試験

## 元の材料

[RFD0122　小さなリファクタリングと学習契約の回帰試験](../../../memo/0122_small_refactor_and_test_gap_fixes_10rfd.md) blob:3e66ec3cd146a95f565383979c936ba5b58b4458

## 要点

- DQN 系のヘッダの形式など、挙動を変えずに整えられる箇所を扱う。
- 報酬スケーラの自動 post-scale と観測正規化の robust update・SymLog を試験経路に含める。
- MuZero の Replay では終端による価値ターゲットの打ち切りと unroll 窓のマスクを検証する。
- 既定のプロファイルで使われても単体試験に現れなかった条件を、公開経路から観測する。
- 想定した壊し方で各試験が失敗することを確認し、単に処理を通すだけの被覆を避ける。

## 内容

### 問題

ReplayBuffer のコンストラクタは、PER では priority store と初期優先度の completer を対で持ち、uniform ではどちらも持たないことを検査している。片方だけを渡したときに止まることを確かめるテストが無い。
報酬の scaler の `use_auto_post_scale`、観測の正規化の robust update と SymLog の後処理は、DefaultDQNAgent の既定のプロファイルで有効だが、テストはどれも通らない。
core のヘッダで、`dqn_based_agent.hpp` だけがインクルードガードを持たず、先頭のファイル名のコメントが旧名のままである。
`QValuePanel::Update` が約 310 行あり、統計の計算、Grid の更新、HeatMap 用のデータの加工を 1 つの関数で続けて行う。加工は画面の部品を直接読むので、画面を作らないと動かせず、テストが無い。表示のモードを足すときに壊しても気づきにくい。

### 起草時点の現状

短い Run の前後比較の手順: RFD0098 の検証資材 の「ステージ 0・smoke」が、変更前の実行体を保存して、4 環境の短い Run の全 scalar の点を比べる手順を持つ。RFD0105 と RFD0109 が同じ手順で受入を取った（RFD0109 の受入 7）。
課題の状態: `解決` にするのは開発者で、AI は対応内容と AI の判定（`ai_judge`）を書く（anet-update-issue）。
リポジトリの中で `wxEVT_CHAR_HOOK` を受けるのは 3 か所である（`git grep -n "wxEVT_CHAR_HOOK" -- core apps` のうちテストを除く当たり）。
`Panel` より上の window で `wxEVT_CHAR_HOOK` を受ける部品は無い。`Panel` を中に置くダイアログも無い。

### 確定事項

d4aa77 のテスト。 `DefaultReplayBuffer` のコンストラクタへ priority store と completer の片方だけを渡す 2 通りが `ANET_SYSTEM_ERROR` で止まり、文言に `must be configured together` が入る。両方ありと両方なしは構築できる（この 2 通りはファクトリ経由でもよい）。
03a1b6 の直し方。 先頭に `#pragma once` を足し、先頭のコメントを `// dqn_based_agent.hpp` にする。
741251 の分け方。 `GetScalar` の本体と、`GetScalar` だけが呼ぶ補助 4 つ（`GetPlasticityScalar`・`ValidateSpectralNormSentinel`・`TransQToCpu`・`TransRealQToCpu`）の本体を、dqn_based_agent.cpp へ移す。名前の解析か key の配列で受ける 8 つの群は、群ごとの関数にして、今の並びの順で先に引く。各関数は、自分の群の key なら今と同じ値を、自分の群の key でなければ「当たらない」を返す。どの群にも当たらなければ、直接比べていた 43 個の key を、key の名前と値の作り方を 1 行ずつ並べた 1 つの表から引く。表にも無ければ、未知の key として `std::nullopt` を返す。表に同じ key を 2 行書いたら、ビルドかテストで止まるようにする（手段は実装計画で決める）。key の名前、返す値、止まる条件と文言は変えない。ヘッダのクラスの定義には、`GetScalar` の宣言だけを残す。
ステージ 1 で振る舞いが変わらないことの確かめ方。 性能の受入は置かない。ステージ 1 の前後で、RFD0098 の検証資材の短い Run（4 環境、seed 1、決定的モード）を流し、時間と資源の tag を除く全 scalar の点が一致することを受入にする。

### 実装契約

入力と出力だけで結果が決まり、画面の部品とメンバを読まない。
`GetScalar` の key の名前と、それを参照する `metrics.scalar` の設定。
過去の RFD・実装記録・課題の単票の題と事象に書かれた当時の行番号。
`DefaultReplayBuffer`・scaler・MuZero の Replay のクラスと、`InitRL` の関数の宣言は変えない。

### 検証と受入

環境変数 `CUDA_VISIBLE_DEVICES` を `-1` にして `core\anet-core\bin\Debug\anet-core-test.exe "[rfd0070]"` を流すと、skip が 0 件で、すべて通る。CUDA が見える状態で流したときの skip を、合格に数えない。
Run は、環境ごとに変更前 1 本と変更後 1 本である。条件は RFD0098 の検証資材 の「ステージ 0・smoke」の短縮条件（GridMaze・LunarLander・Atari qbert・DropMerge、seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価停止、FP32、決定的 backend）で、変更前はステージ 1 に入る前に保存した Debug の実行体、変更後はステージ 1 を終えた Debug の実行体を使う。
検証資材の手順が今の設定で動かないときは、同じ条件と同じ比べ方（全 scalar の点を tag・step・値で突き合わせる）を満たす手順を実装計画に書き、それで確かめる。比較を省いて合格にしない。
人が Runner で確かめる: 評価を流し、Q-Values の表示が出て、Histogram・Advantage・Log・Auto Range の切替で絵が変わる。行の選択とクリックで、ハイライトと選ばれる行動が直す前と同じである。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [325 Agent個別実装 MuZero系列](../../concepts/framework/325_muzero.md)
- [316 設定管理](../../concepts/framework/316_config.md)
