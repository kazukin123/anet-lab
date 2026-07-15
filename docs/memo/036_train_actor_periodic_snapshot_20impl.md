# PRD 036 DefaultDQN Train Actor 定期 network snapshot 実装計画

## 概要

- `DefaultDQNAgent` の Train Actorだけに、既定無効の private network snapshot と `train_step` 周期同期を追加する。
- Serial/Pipeline、保存形式、PRD 035、Rainbow・ImageCls・MuZero・Evalの既存挙動は、明記された共通API変更以外維持する。

## 公開API・設定

- `Agent::CreateActor()` の必須 `bool clone_model` を、既定 `std::nullopt` の `std::optional<bool> clone_model_override` へ変更する。
- Train Runnerは `nullopt`、Eval Runnerは既存の明示boolを渡す。DefaultDQNはTrain時のみAgent設定を既定値に使い、Rainbow/ImageClsは `nullopt=false`、MuZeroは全指定を従来どおり無視する。
- effective policyがsharedの場合のdevice整合性は各Agentで検証し、Runner側の明示 `false` に対する早期検証も維持する。
- `DefaultDQNAgentConfig` に以下を追加する。
  - `train_actor.clone_model = false`
  - `train_actor.sync_interval.type = constant`
  - `train_actor.sync_interval.value = 400`
- 新規subtreeの明示raw値をDQN固有の厳格parserで先に検証する。既存の桁区切りカンマは許可するが、空値、負のunsigned値、末尾文字、overflow、nonfinite値は拒否する。active profileは周期1以上、必要なstepsとphase定義、正の`cycle_mult`を要求し、clone無効時も検証する。
- `Actor::MakeAction()` と `Actor::Sync()` に、同一Actorへ並行呼び出しできない契約をDoxygenで明記する。

## 実装変更

- DefaultDQNのclone生成をAgent mutexのshared lock内で行い、clone時のcopyを `train_step=0` の初回同期とする。
- DQN Actorへ、周期snapshot有効時だけ使う `ProfiledValue<step_t>`、最終同期step、強制同期後pending状態を持たせる。
- `MakeAction()` のforward前に、`exp_step`によるprofile更新、pending基準更新、age判定、必要な`CopyTo()`を順に行う。同期actionのageは0とし、追加network forwardやCUDA同期は行わない。
- `Sync()` はclone Actorを必ず即時copyし、周期Train Actorでは次回actionをage 0の新基準にする。Pipeline Train Runnerの毎step `actor_->Sync()` は削除する。
- `DQNActionInfo` に専用のoptional snapshot診断値を保持させ、DQN内部のActionPolicy戻り値を型付けしてActorから設定する。DefaultDQNのshared/Evalでは両キーを `NaN`、snapshot有効時は同期後のinterval/age、Rainbowでは診断値自体を持たせず `std::nullopt` とする。`To()` と `WithAction()` は診断値を保持する。
- archiveにはsnapshot network、profile runtime、同期step、pending状態を追加しない。`auto_load_file` 後にload済みonline networkから新しい初期snapshotを作る。
- 既存の `CONTEXT.md`、ADR 0013、ownership guideline、`metrics.scalar.full` のユーザー変更は保持する。実装後に設計資料のRunner境界、Actor ownership/API、snapshot metric契約を現行仕様へ更新する。新規ADRは追加しない。

## TDD・テスト

1. Tracer bulletとして、既定shared Train Actorがsource更新を即時反映し、Train Runnerが `nullopt` を渡す挙動をRED→GREENで通す。
2. snapshotの初期固定、`0..399`、`400`直前同期、age 0、actionごとforward 1回を公開Actor経路で検証する。
3. 短縮時即時同期、延長時寿命延長、全profile種別、整数切り捨て、正値検証を1 behaviorずつ追加する。
4. 強制`Sync()`と次actionの重複copyなし、生成直後の重複copyなしを、copy回数用production APIではなく同期間のsource変更が反映されないことから検証する。
5. DefaultDQNのTrain/Eval × `nullopt/true/false`、shared device不整合、Rainbow/ImageCls/MuZeroの互換性を検証する。
6. Pipelineが毎step強制同期しないこととSerial/Pipelineのtrigger境界一致をRunnerの公開実行経路で検証する。
7. snapshot scalar、無効時`NaN`、Rainbowの`nullopt`、`To()`/`WithAction()`保持、fullのみへのcatalog登録を検証する。
8. 不正bool、負値、末尾文字、overflow、nonfinite、未定義phase、clone無効時の不正intervalをfail-fastテストする。
9. 保存済みAgentを `auto_load_file` で読み込んだ後、復元networkから初期snapshotが作られることを検証する。
10. 各RED→GREEN後に関連テストを実行し、全GREEN後だけ重複整理を行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][actor]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][config]"
core\anet-core\bin\Debug\anet-core-test.exe "[profiled_value]"
core\anet-core\bin\Debug\anet-core-test.exe "[trainer]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- 現在のdirty worktreeと未追跡PRD・ADRをユーザー変更として保持し、無関係な差分を戻さない。
- snapshot周期のschedule軸は`exp_step`、ageと同期判定単位は`train_step`で固定する。
- CUDA default stream、既存mutex、`Network::CopyTo()` のprofiling契約を維持し、別stream、Event、明示同期は追加しない。
