# PRD082 RAM メトリクスの検証

`AtariEnv-test` の `[ram_metric]` は、設定検証、5 種の畳み方、Pong の skip 窓内遷移、kung_fu_master の実値、episodic-life の soft / hard reset、AuxData、完了ログを検証する。ROM 依存ケースは `ATARI_ROM_DIR` に対象 ROM がない場合 SKIP する。全 `[atari]` は既存の `game_score`、報酬、reset の回帰も含む。

リポジトリルートの PowerShell から実行する:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AtariEnv-test'
```

```bash
& .\core\envs\atari1\bin\Debug\AtariEnv-test.exe '[atari]' --rng-seed 123 --reporter compact
```

既定 `Atari.txt` の変更時は、プロトコルプリセットの値と、群ごとの RAM 指標の揃い（番号、`source_key`、得点の指標との軸・event の一致）を anet-core-test の `[shipped_config][atari]`（`config_baseline_test.cpp`。ALE 不要）で確認する。本番設定を読むテストが固定してよいものの線引きは [実装 品質確認観点](../../../../docs/quality/impl.md#テスト)「テスト」の規約に従う。baseline 比較の実施範囲と waiver は [実装メモ](../../../../docs/memo/082_atari_ram_progress_metrics_20impl.md) に記録する。
