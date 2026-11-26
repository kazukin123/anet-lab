# Runner 起動時の RunManager 構築

## 元の材料

[RFD0109　Runner 起動時の RunManager 構築](../../../memo/done/0109_runner_startup_background_init_10rfd.md) blob:112ea38ad4f4bf7bb2cc299d541bf331ff5e92c4

## 要点

- 起動時に RunManager を構築する間、GUI のイベントループが始まらず、表示済みの窓が描画に反応しなかった。
- 窓の初期表示と時間のかかる構築処理を分け、RunManager は別スレッドで構築する。
- 初期化中もメインスレッドが描画とイベントを処理できるようにする。
- 準備の完了後に、GUI 側で構築済みの RunManager と表示・操作を接続する。
- 起動中の終了や構築失敗も対象にし、利用可能になる前の操作から未構築の部品へ触れないようにする。

## 内容

### 問題

Run ログの時刻から、表示から OnInit の終わりまでを区切った。
約 11.5 秒のうち、順 4 の RunManager の構築が約 11.1 秒を占める。順 1〜3 は、Run 名の時刻（MetricsLogger の初期化。21:55:54 台で、秒までしか分からない）から順 3 の行（21:55:55.073）までで、長くても約 1.07 秒である。
順 4〜7 の間はイベントループが回らないため、描画のメッセージが処理されない。表示した Frame の中は描かれず、前の画面の残りや白・黒の領域が残る。ほかのウィンドウを重ねたり Frame を動かしたりすると、乱れが増える。

### 成功の測り方

起動にかかる時間を、Run ログの `Run initialized. elapsed_sec=` で読めるようになる。

### 確定事項

構築スレッドの名前は `RunInitThread` とする。プロファイラとログに出るスレッド名である。

### 実装契約

直さないもの: 過去の Run の記録と、PRD105 の検証資材の記録。
`SlowInitTestAgent` だけが読む、gui-test の設定である。製品の設定ファイルには置かない。

### 受入基準

ビルドと既存のテスト: Debug 全体と `AnetRLRunner-gui-test` がビルドでき、`anet-core-test` と `AnetRLRunner-gui-test` の既定のスイートで、落ちるテストの集合が変更前と同じである（変更前から落ちているテストは増えも減りもしない）。
移したテスト: `AnetRLRunner-gui-test "[eval_steps_menu]"` と `AnetRLRunner-gui-test "[prd105_ui_keys]"` が、それぞれ別プロセスで通る。
構築中の描画とログ（`[.][prd109_startup]`、1 プロセス）: `agent.construction_delay_ms = 3000` の `SlowInitTestAgent` を使う CartPole の設定で RunnerApp を起動すると、次がすべて成り立つ。
`SlowInitTestAgent` が構築の途中で出したログ行（`SlowInitTestAgent waiting.`）が、初期化が済む前に LogPanel のログ欄の文字列に現れる。

### 関連

068（fatal 経路と終了コード）、071（捕まらない fatal の可視化）。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
