# Envログprefix自動付与 実装メモ

## 概要

`anet::log::Logger`へprefix処理を集約し、Env基底から`log.info() << ...`形式で利用可能にする。prefixは`<Env name>: `へ統一し、通常ログとdebugログの既存level・遅延評価・flush契約を維持する。

## 主な変更

- `anet::log::Logger`を追加し、`info()`、`verbose()`、`warn()`、`error()`、`prefix()`を公開する。prefix再代入は許可せず、default構築時だけ空prefixを許容する。
- `WxLogStream`にはLogger専用の非公開初期本文constructor seamを設け、prefix入りstreamを直接prvalue生成する。既存公開APIを変えず、move元の二重・途中flushを構造的に防ぐ。
- `SingleDiscreteEnvBase`と`BatchEnvBase`へprotectedな`log`値メンバを追加し、`name_`とlane名より後に宣言する。constructorでそれぞれ`<lane name>: `、`<BatchEnv name>: `を確定する。
- `ANET_LOG_DEBUG_PREFIXED(expr)`を追加し、既存`ANET_LOG_DEBUG`へ委譲してguard、source情報、無効ビルド時の式非評価を維持する。
- DropMerge、ImageCls、LunarLander、batch wrapperのactiveログを新記法へ移行する。DropMergeの`Rank [ n ]`は`Rank n`へ整理し、その他の本文とlevelは変えない。
- `docs/design/120_environments.jp.md`と`docs/design/140_observability.jp.md`へLogger、記法規約、prefix書式を反映する。`CONTEXT.md`とADRは変更しない。

## テスト

- Public interface / surface: `anet::log::Logger`、`SingleDiscreteEnvBase`、`BatchEnvBase`、`ANET_LOG_DEBUG_PREFIXED`、具象Envの観測可能なtext log。
- 優先 behavior: LoggerのInfo prefix exactly-once、各level対応・空prefix・immutable prefix、single/batch Baseのname-bound log、具象Envとbatch wrapperのprefix移行。
- TDD順序: LoggerのInfoログ捕捉をtracer bulletとして1テストでRED→最小GREENにし、各level、prefix accessor、空prefix、再代入不可、Env Baseの順に1 behaviorずつ進める。GREEN後だけ整理する。
- DropMergeは決定的な最大step Verboseを動的検証する。Info/Errorは不安定な物理シナリオを追加せず、Logger単体のlevelテストと全siteの静的検索で担保する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test DropMergeEnv-test LunarLanderEnv-test ImageClsEnv-test'
core\anet-core\bin\Debug\anet-core-test.exe "[log]"
core\anet-core\bin\Debug\anet-core-test.exe "[env_name]"
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe
core\envs\lunarlander1\bin\Debug\LunarLanderEnv-test.exe
core\envs\imagecls1\bin\Debug\ImageClsEnv-test.exe
ctest --preset x64-Debug --output-on-failure
git diff --check
```

## 前提

- Env name、lane name、RunMode、seed、metrics identity、ログformatter、warning以上の即時flush契約は変更しない。
- PRD 034で予定される`GetScalar`無prefix WARNのfail-fast化は範囲外とし、今回は現行WARNへBatchEnv prefixを付ける。
- Runner、Agent、View、コメントアウト済みログへのLogger適用は行わない。
- `CONTEXT.md`とADRは変更しない。
- 既存・無関係な未コミット変更には触れない。
