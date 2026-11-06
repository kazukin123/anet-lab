# Dueling DQN V/A 分岐 Head 実装計画

## Summary

`docs/memo/009_dueling_dag_head_10prd.md` に沿って、Dueling 系 head が `value_feature` / `adv_feature` を opt-in で読み、未指定なら従来どおり `features` 一本へ fallback する適応型 head にする。`use_dueling_net` / `use_qr` の factory 選択や `NetworkBody` 側の DAG 出力機構は変更しない。

## Key Changes

- `core/anet-core/src/dqn_based_heads.cpp` の `DuelingHead` / `QuantileDuelingHead` を、V 入力キー・A 入力キー・各入力次元を保持する構造へ変更する。
- `CreateHead()` で `dummy_features.Get("value_feature")` と `dummy_features.Get("adv_feature")` を判定する。
  - 両方あり: 分岐モード。V は `value_feature`、A は `adv_feature` を読む。
  - 両方なし: 共有モード。従来どおり `features` を V/A 双方で読む。
  - 片側だけあり: `ANET_SYSTEM_ERROR` で明示エラー。
- `Forward()` と `GetTensorDictFunction()` は保存した入力キーを使う。`forward` / `forward.q` / `q_values` / `forward.v` / `v_values` / `forward.a` / `a_values` / QR の `forward.dist` は既存の public function key を維持する。
- `GetGraphVizInfo()` には `mode = shared|branched`、`value_input_key`、`adv_input_key` を追加し、`show_head_info` 時に head が何を読んでいるか確認できるようにする。
- `CONTEXT.md` には実装仕様ではなく用語だけ追加する。対象は「価値ストリーム」「アドバンテージストリーム」。
- `apps/runner/config/DropMerge.txt` にはコメントアウトされた分岐モード設定例を、既存の未コミット変更を壊さない位置へ追記する。

## Public Interfaces

- `DuelingHeadFactory` / `QuantileDuelingHeadFactory` の public constructor と header 宣言は変更しない。
- `default_dqn_agent.cpp` / `rainbow_agent.cpp` の head factory construction site は変更しない。
- `NetworkBody` / `NetworkConfig` / `Network::Forward` の public behavior は変更しない。
- 新しい config-facing key は `net.body.output.[value_feature]` と `net.body.output.[adv_feature]`。既存の `net.body.output.[features]` は共有モード用として残す。

## Test Plan

- `core/anet-core/src/nn_test.cpp` に head factory regression を追加する。
  - `features` のみで Dueling / QuantileDueling head が従来 shape を返す。
  - `value_feature` と `adv_feature` の両方ありで、異なる入力次元から V/A head が構築される。
  - 片側だけ指定すると `CHECK_THROWS` で失敗する。
  - GraphViz detail に入力キーと分岐モードが出る。
- 既存 head GraphViz テストは、追加 detail が `show_head_info=false` では漏れないことを保つ。
- 検証コマンドは MSVC 初期化付きで実行する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe
```

## Assumptions

- 片側だけの `value_feature` / `adv_feature` は設定ミスとして fail-fast する。
- `DropMerge.txt` への設定例はコメントアウトで追加し、既定動作は変えない。
- ADR は追加しない。この変更は head の局所拡張で、後戻り不能な設計決定ではない。
