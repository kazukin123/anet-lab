# Envインスタンスname PRD

## 元の材料

[Envインスタンスname PRD](../../../memo/done/0037_env_instance_name_10rfd.md) blob:a71315be9e8d81790e66e011e9c07099d02a4f18

## 要点

- 複数の Env が同時にログを出すとき、どの学習・評価・GUI 個体から来たかを識別できるようにする。
- RunManager が BatchEnv の名前を決め、生成経路を通して渡す。
- 同じ Run の名前衝突は Env を構築する前に検出する。
- 各 lane にも個体を識別できる名前を与え、Env が上位の Runner 構造を解釈せず使えるようにする。
- 名前のための利用者設定を増やさず、既存の生成経路から固定規則で決める。

## 内容

### 問題

一方、EnvへTrain/Eval、Runner、configured Eval tagなどの構造化コンテキストを渡すと、Envが上位の実行構造を認識し、その値で挙動を分岐する余地が生まれる。Envが本来扱う環境固有ロジックへ上位層の複雑さを持ち込みたくない。

### 目的

main Train、configured Eval、EvalPanel、およびlane番号をtext log上で識別できるようにする。
設定項目を追加せず、既存の生成経路から固定規則でnameを決定する。

### 前提

Env class IDは同一Run内のEnvインスタンス識別には不要なため、nameへ含めない。
main Train、各configured Eval、EvalPanelは、それぞれ別のBatchEnvを所有する。

### 解決方針

testや単独ツールから直接生成する場合は、その呼び出し側が人間向けのnameを明示する。
`train`と`EvalPanel`はconfigured Eval tagでは使用できない予約名とする。いずれかと一致するtagは`ANET_SYSTEM_ERROR`でfail-fastする。

### 実装上の決定

具象single Envは`SingleDiscreteEnvBase`、具象BatchEnvは`BatchEnvBase`を継承し、name accessorを独自実装しない。batch-native Envも`BatchEnvBase`へnameを渡すだけとする。
PRD 037実装時の旧top-level APIを`BatchEnvFactory::CreateBatchEnv(name, seed, num_envs)`とする。

### テスト方針

configured Eval tagが`train`または`EvalPanel`の場合、最初のBatchEnv構築前に`ANET_SYSTEM_ERROR`となることを検証する。
既存の具象Env testとcore testを実行し、必須name引数追加による生成経路の漏れがないことを確認する。

### 受入条件

EvalPanel用single Envが固定値`EvalPanel`由来のnameを持ち、selected configured Eval tagまたは`RunMode`に依存しない。
PRD 037単独の現行direct configured Eval経路で`name=tag`が渡り、PRD 034のBuilder移行後も同じnameを維持する。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
