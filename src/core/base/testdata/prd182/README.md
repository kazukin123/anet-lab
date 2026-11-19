# PRD182 検証資材

対象は [PRD182](../../../../../docs/memo/182_source_layout_10prd.md)。検証の実施記録は [実装計画](../../../../../docs/memo/182_source_layout_20impl.md) に置く。

## 学習挙動の比較

`run_comparison.py` は、変更前の Debug Runner、GUI テスト実行体、DLL、設定から PRD128 の34構成を採取し、scalar・設定記録・WARN/ERROR を照合する。Runner と設定を `snapshot` で保存し、`repeat` で再現性を確認する。各ステージの実行体をビルドしたら `compare --label stage2` のようにステージ名を指定する。同じ label は再利用せず、結果は `.scratch/prd182/comparison/` に残す。

時刻と資源の scalar、ログの時刻・Run名・Runパス・ソース位置を比較から除く。構成や出力の不足は失敗とする。

## DQN ファイル分割

`dqn_definitions.py` に移動前のヘッダ・実装、`--after` に移動後の10本を渡すと、定義のトークン列の件数と差を表示する。テストは `--test-cases` を指定し、移動前のテストと移動後の6本を比較する。コメント、include、名前空間の開閉、外から呼び出すために変更した宣言とリンケージ指定を比較から除く。

## ライブラリの依存

Debug の configure で `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` を指定し、`--graphviz=out/build/x64-Debug/prd182-targets.dot` で直接の依存グラフを出す。リポジトリの根から、`check_stage3_dependencies.py` の第1引数に DOT、第2引数に `out/build/x64-Debug/compile_commands.json` を指定する。系統・Env の依存先、Runner とテストの依存、基盤の内部 include パスを検査する。

## 実機のデータを移す前の確認

[CHANGELOGの移行手順](../../../../../CHANGELOG.md#ワークスペースと-appdata-の移行prd182-ステージ4)を参照する。`move_runtime_data.py` の既定実行は旧・新のパスとフォルダ数を表示する。合成データの確認は `move_runtime_data_test.py` を使う。実機での改名とworkspaceの再選択は開発者が行う。
