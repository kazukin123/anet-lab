# Envインスタンスname 実装メモ

## 概要

`SingleDiscreteEnv`と`BatchEnv`を状態を持たないinterfaceとして維持し、`env.hpp`の`SingleDiscreteEnvBase`と`BatchEnvBase`へ人間向けのimmutableな`name`保持を追加する。nameはsingle wrapper、factory、`RunManager`、具象Envのruntime logまで伝播する。同一Run内のBatchEnv nameは`RunManager`がcase-sensitiveに一意性を保証し、重複、空name、lane範囲外を常時fail-fastする。

PRD 034のBuilder再編、batch-native ImageCls、既存Viewへの具体的な表示変更は本実装に含めない。

## 主な変更

- `SingleDiscreteEnv`はpure virtualな`GetName()`を公開し、`SingleDiscreteEnvBase(std::string name)`がnameを保持してaccessorを`final override`する。
- `BatchEnv`はpure virtualな`GetName()`と`GetEnvName(lane_index)`を公開し、`BatchEnvBase(std::string name, int num_envs)`が構築時に`<name>[0..N-1]`を生成・保持して両accessorを`final override`する。
- 空name、`num_envs <= 0`、範囲外lane indexは`ANET_CHECK_MSG`で常時fail-fastする。laneエラーにはname、index、num_envsを含める。
- `BatchEnvFactory::CreateBatchEnv(name, seed, num_envs)`と`SingleDiscreteEnvFactory::CreateSingleEnv(config_data, device, name, seed, config_prefix)`でnameを必須引数として伝播する。
- Vectorized/ThreadPool wrapperはfactory直後に必須`name`を受け、各single Env factoryへ完成済みlane nameを渡す。
- CartPole、GridMaze、DropMerge、LunarLanderは`config, device, name, seed`、ImageClsは`config, name, seed`の順で具象constructorへ伝播する。
- `RunManager`はconstructor冒頭で`train`、全configured Eval tag、`EvalPanel`を一括検証し、成功済みnameだけをprivateなrun-local registryへ登録する。
- main Trainは`train`、configured Eval direct経路は`name=tag`、`CreateEvalRunner`は受け取ったnameを無加工でfactoryへ渡す。重複時は第二Envを構築せず既存runnerを維持する。
- `EvalPanel`はconfigured Eval tagの予約名とするが、`CreateEvalRunner("EvalPanel", ...)`による最初のEvalPanel生成は許可する。
- DropMerge、ImageCls、LunarLanderのactiveなEnv本体ログへ`[<GetName()>] `を付ける。metrics key、設定検証例外、View、factory、コメントアウト済みログは変更しない。
- 4つのBatchEnv test doubleは`BatchEnvBase`を継承してaccessorをoverrideせず、constructorと呼び出し側から明示nameを渡す。

## テスト

- Public interface / surface: pure interfaceの`SingleDiscreteEnv::GetName()`、`BatchEnv::GetName()`、`BatchEnv::GetEnvName()`、共通実装の`SingleDiscreteEnvBase`／`BatchEnvBase`、wrapper/factory生成API、`RunManager` constructor、`RunManager::CreateEvalRunner()`、具象Env runtime log。
- 優先 behavior:
  1. BatchEnvのname、B=1 lane name、空name・不正num_envs・範囲外laneのfail-fast。
  2. Vectorized B>1、ThreadPool、worker数変更でのlane name一致とsingle factoryへの完成済みname伝播。
  3. main Train、configured Eval、EvalPanelのname決定と同一Run内一意性。
  4. reserved tag、main/configured/dynamic name重複、同名2回、case-sensitive比較、生成失敗後の再試行、別RunManagerでの再利用。
  5. nameだけを変えた場合のEnv結果・seed・RNG列の不変性。
  6. DropMergeの決定的な最大step Verboseログでprefix、本文、levelが維持されること。
- TDD順序: 上記を1 behaviorずつRED→最小GREENで実装し、GREEN後だけ重複整理する。DropMergeのInfo/Errorは全siteへ同じprefix式を適用した差分確認で保証し、test-only seamや不安定な物理シナリオは追加しない。

## ドキュメント

- PRD 037へ`ANET_CHECK_MSG`、正確な`CreateSingleEnv`引数順、5具象factoryと4 test doubleの影響範囲、空name契約を追記する。
- `CONTEXT.md`へ人間向け・不透明・挙動非依存なEnv nameを追加する。
- runtime/configuration、environments、observability設計文書へname決定、lane、Run内一意性、ログprefix、metricsとの分離を反映する。
- PRD 034とADR 0009は既に後続義務が整合しているため変更しない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test LunarLanderEnv-test ImageClsEnv-test DropMergeEnv-test'
ctest --preset x64-Debug --output-on-failure
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target anet-core-test'
core\anet-core\bin\Release\anet-core-test.exe "[env_name]"
git diff --check
```

## 前提

- `name`は人間向けの不透明な文字列であり、Envが意味を解析しない。
- interfaceとBaseの`GetName()`／`GetEnvName()`は`const std::string&`を返し、Base実装は`final`とする。
- factory、wrapper、Envは一意性registryやowner情報を持たない。
- `RunManager`の既存thread-safety contractは変更せず、並行`CreateEvalRunner`対応は対象外とする。
- 設定キー、metrics schema、artifact path、serialization、seed domainを変更しない。

## 修正ファイル

### 仕様・設計文書

- `CONTEXT.md`
- `docs/memo/037_env_instance_name_10prd.md`
- `docs/memo/037_env_instance_name_20impl.md`（本実装メモ）
- `docs/memo/034_imagecls_batch_input_10prd.md`
- `docs/adr/0009-imagecls-batch-env-seam.md`
- `docs/design/100_runtime_and_configuration.jp.md`
- `docs/design/120_environments.jp.md`
- `docs/design/140_observability.jp.md`

### Env共通基盤・RunManager

- `core/anet-core/include/anet/rl.hpp`
- `core/anet-core/src/rl.cpp`
- `core/anet-core/include/anet/env.hpp`
- `core/anet-core/src/env.cpp`
- `core/anet-core/include/anet/trainer.hpp`
- `core/anet-core/src/trainer.cpp`

### 共通基盤テスト

- `core/anet-core/src/env_test.cpp`
- `core/anet-core/src/trainer_test.cpp`
- `core/anet-core/src/dqn_based_agent_test.cpp`
- `core/anet-core/src/episode_end_test.cpp`
- `core/anet-core/src/observers_test.cpp`

### 具象Env

- `core/envs/cartpole2/src/CartPoleEnv.hpp`
- `core/envs/cartpole2/src/CartPoleEnv.cpp`
- `core/envs/gridmaze1/src/GridMazeEnv.hpp`
- `core/envs/gridmaze1/src/GridMazeEnv.cpp`
- `core/envs/dropmerge1/CMakeLists.txt`
- `core/envs/dropmerge1/src/DropMergeEnv.hpp`
- `core/envs/dropmerge1/src/DropMergeEnv.cpp`
- `core/envs/dropmerge1/src/DropMergeEnv_test.cpp`
- `core/envs/lunarlander1/src/LunarLanderEnv.hpp`
- `core/envs/lunarlander1/src/LunarLanderEnv.cpp`
- `core/envs/lunarlander1/src/LunarLanderEnv_test.cpp`
- `core/envs/imagecls1/src/ImageClsEnv.hpp`
- `core/envs/imagecls1/src/ImageClsEnv.cpp`
- `core/envs/imagecls1/src/ImageClsEnv_test.cpp`
