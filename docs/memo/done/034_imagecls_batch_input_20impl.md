# PRD 034 ImageCls batch-native 入力 実装メモ

## 概要

- `034_imagecls_batch_input_10prd.md` を仕様の正本とする。
- `ImageClsEnv` を `SingleDiscreteEnv` と内部 vectorize に依存する構成から、dataset を直接 batch tensor 化する native `BatchEnvBase` 実装へ移行する。
- `ImageDatasetManager` は process singleton とし、catalog 登録、manifest 検証、dataset共有、cacheを一元管理する。Env固有のSourceは各ImageClsEnvが専有する。
- TDD の縦スライスで実装し、既存の Env name、Runner、他 Env の回帰を防ぐ。

## 主な変更

### Dataset と設定

- `ImageDataset.*` にdataset catalogとdataset-key単位の定義を導入する。`ImageClsEnv`は固定RunModeにかかわらず`ImageClsEnv.train.dataset_key` / `ImageClsEnv.train.augment.*`と`ImageClsEnv.eval.dataset_key` / `ImageClsEnv.eval.eval_window.*`の標準Train/Eval Source設定を必須の組として持ち、`data_source` / `common`層は設けない。
- config-facingなdataset keyは`std::string`とし、DatasetKey専用型、`std::optional`、`std::variant`、未設定sentinelを設定構造へ導入しない。`eval.eval_window.rotating.size`はmodeにかかわらず正数を保持する。
- `ImageDataSource`はTrain/Eval別Configを受けるconstructor overloadで役割を確定し、内部では共通decode/cache/collateと役割別sampling/augment/windowを扱う。RunModeは受け取らない。
- `ImageDatasetManager` は catalog 全体を検証して原子的に登録する。登録失敗は状態を公開せず、同一失敗を保持して後続アクセスでも再送出する。
- 標準Train/Evalが参照する両manifestは起動時に個別検証し、各datasetのclass、path、shape、設定値、未知キー、不整合な組み合わせを`ANET_SYSTEM_ERROR`でfail-fastさせる。Train/Eval間のspec受理判断はAgentへ委ね、画像decodeとpayload/cache確保は使用時まで遅延する。
- cache は `NoCache`、`FullRam`、`auto` を実装する。明示 `FullRam` の確保不能はエラー、`auto` は既定上限 4 GiB 内で選択し、実行不能時のみ対象ごと1回 WARNして実行可能な方式へ固定する。
- bundled config を新スキーマへ移行し、旧キーの暗黙 alias や silent fallback は設けない。

### Native batch Env

- 画像読込、decode、変換、cache の実体を `ImageData.cpp` 側へ置き、公開ヘッダには必要最小限の型と契約だけを残す。
- `ImageClsEnv` を native `BatchEnvBase` として構築し、各 `Reset` / `Step` が呼出し側に所有権を渡せる fresh tensor を返す。
- train は batch 全 lane を学習サンプルとして扱い、eval は lane 0 を代表 lane として episode 境界を通知する。
- eval window は PRD の full / rotating 契約に従い、指定サイズを厳密に満たし、末尾不足時は規定どおり padding する。`eval_batch_size` は Env 構築時に確定させる。
- accuracy、epoch、sample count は dataset 全体の進行を表す metric とし、lane ごとの一時値や単純 mean に退化させない。
- 標準Train/Eval Dataset pairのinput shapeとclass_names順序をImageCls内で検証する。DatasetKeyを`EnvSpec.info`へ特別格納せず、`info`は参考metadataに留める。
- `Module::GetConfigData()`を共通の設定参照seamとして追加する。include・継承・override解決後に注入された構築後不変の設定を`std::optional<ConfigData>`で返し、実動情報は将来のPropertyへ分離する。複合Envは子設定をscope付きkeyでmergeする。
- batch size、画像数、decode、tensor 化、device 転送に比例する主要境界へ既存命名規則に従う `ANET_PROFILE_SCOPE` を追加する。

### Env API、Factory、Runner

- `RunMode` を Env 構築時の固定属性にし、`Reset` / `Step` から引数を削除する。すべての Env、wrapper、呼出し側を同じ契約へ移行する。
- 旧 top-level `BatchEnvFactory` interface を廃止し、汎用 builder は `DefaultBatchEnvFactory` から `BatchEnvBuilder` へ改名する。
- Env クラスごとの factory と `EnvRepository` の variant dispatch を導入し、static auto-registration を除去して構成を明示化する。
- configured evalは設定されたEnv tagと`eval_batch_size`からbuilderを生成する。無効化されたdormant tagは構築せず、参照metricは対象ごと1回WARNしてskipする。ImageClsのtag無しEvalは標準`ImageClsEnv.eval.*`を使用し、tag指定時だけoverlayを適用する。
- `RunnerFrame`はEnv種別に応じたeval tagの必須・禁止を判定しない。`CreateEvalRunner` interface全体の再設計は別件へ延期し、今回はImageCls固有分岐の除去に留める。未コミットの`RunnerFrame` UI変更は保持する。
- RunManagerからImageCls固有のseed domain、DatasetKey、EnvSpec互換性判定を除去する。seedは全Env共通domainだけを渡し、ImageCls固有のsampler/augment派生はImageCls内で行う。
- Agentの`CreateActor`へ既存引数順を維持して`BatchEnvSpec`の後に`EnvSpec`を渡し、接続可否はAgentが判断する。通常の同一state/action契約を検査する`CheckSameStateActionSpec`相当helperは`ANET_CHECK_*`を用い、`info`を比較しない。
- RunManagerは生成済みBatchEnvの`GetConfigData()`を使い、`config/env.<Env name>.txt`へ設定を共通dumpする。`nullopt`はEnv nameごとに1回WARNしてskipする。Run直下`config.txt`は廃止し、`config/config_data.txt`とtag別fileを維持する。
- `CONTEXT.md` の stale な `eval_samples=all` 記述、ADR 0009 の後続決定、関連する Env / Runner / config 設計資料を実装契約へ同期する。

## 公開インターフェース

- `SingleDiscreteEnv` / `BatchEnvBase` 系の `Reset` / `Step` から `RunMode` 引数を削除し、constructor または生成設定で固定する。
- 汎用 batch 生成 API の名称を `BatchEnvBuilder` に統一し、Env 固有 factory を repository の明示的 variant として扱う。
- Dataset catalog、dataset key、cache mode、eval window、`eval_batch_size`、`app.eval_panel.eval_config_tag` を設定上の正式な契約として公開する。
- 旧 ImageCls 設定構造と旧 factory interface は互換 shim を残さず削除し、残存利用は設定読込またはコンパイル時に検出する。
- `Module::GetConfigData()`は段階導入中だけdefaultで`nullopt`を返すvirtual methodとし、既存Module全体への波及を避ける。`Module`をpure interfaceとする原則は維持し、全Module対応時のpure virtual化を後続課題とする。

## テスト

- Public interface / surface: Dataset catalog の設定読込、`ImageDatasetManager`、native `ImageClsEnv`、Env の `Reset` / `Step`、factory / repository、configured eval、EvalPanel tag。
- 優先 behavior:
  1. catalog登録から標準Eval Sourceおよびconfigured eval overlayのnative `ImageClsEnv`を1 batch実行し、shape、dataset key、done、accuracyまで観測する。
  2. catalog の原子登録、重複、未知参照、壊れた manifest、sticky failure を検証する。
  3. `NoCache`、明示 `FullRam`、`auto` 選択、4 GiB上限、WARN一回、明示要求失敗を検証する。
  4. train/eval window、rotating、padding、fresh tensor、lane 0 done、epoch / accuracy集計、標準Dataset pairのeager manifest検証を確認する。
  5. `RunMode`固定、factory variant、dormant tag非構築、tag無し標準Eval、configured eval overlay、metric skip WARNを検証する。
  6. AgentによるEnvSpec受理、`CheckSameStateActionSpec`、Module設定merge、Env別dump、`nullopt` WARN、artifact名衝突、Run直下`config.txt`非生成を検証する。
- TDD 順序: tracer bullet から始め、各 behavior を1テスト追加、失敗確認、最小実装、GREEN確認の順で進める。refactor は GREEN 後だけ行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test ImageClsEnv-test'
core\anet-core\bin\Debug\anet-core-test.exe
core\envs\imagecls1\bin\Debug\ImageClsEnv-test.exe
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
git diff --check
```

Food101 は短時間 smoke test に限定し、長時間学習評価は実装完了条件から分離する。

## 前提

- `ImageDatasetManager` の lifetime は process singleton とし、run-scoped session への再設計は行わない。
- 背景 eval snapshot、汎用 pool の例外機構、supervised runner、mmap cache、長時間の学習比較は対象外とする。
- ユーザーの既存未コミット変更、過去の `2ximpl` memo、生成物、`third_party/` は変更しない。
