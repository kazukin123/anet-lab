# WeightInitConfig.mode の文字列化

`WeightInitConfig::mode` は int（0:Default, 1:Xavier, 2:He, 3:Orthogonal, 4:Constant）だったが、ConvNeXt導入で `trunc_normal` を追加すると6値目になり、数値と初期化手法の対応が設定ファイルを読む人間に伝わらない水準に達した。そこで `mode` を `std::string`（`"default"`/`"xavier"`/`"he"`/`"orthogonal"`/`"constant"`/`"trunc_normal"`）へハード改名し、旧数値との後方互換・fallbackは入れない。

## Consequences

- `net.block.[XXX].init.mode` は数値ではなく文字列で書く。既存の全 `.txt` 設定・コード内デフォルト値を新規格へ一括書き換える（`027_weight_init_mode_string_10prd.md`）。
- 未知の文字列が渡された場合は `ANET_SYSTEM_ERROR` で即座に失敗する（旧数値がそのまま残っていた場合もこの経路でfail-fastする）。
