# PRD065 NN Spectral Normalization 実装計画

## 概要

- 実行開始時に本計画を `docs/memo/065_nn_spectral_norm_20impl.md` へ保存し、以後の実装正本とする。
- `weight_norm.mode = none | spectral | spectral_cap` を Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder に導入する。
- Phase 1 で SN 本体・seed・buffer 不変量を完成させ、Phase 2 で実効ノルム、σ、validity sentinel を追加する。効果実験は Phase 2 完了後とする。

## 公開契約と実装変更

- `WeightNormConfig` を `WeightInitConfig` と同じ共通設定層へ追加する。各 factory Config が `weight_norm` として合成し、既定 `none`、未知値はキー・値・許容値を含めて fail-fast する。
- `NetworkBuilder::BuildNetwork(config, input_specs, head_factory, seed, device)` へクリーンブレークする。
  - DefaultDQN / Rainbow / ImageCls は Agent seed から `"network"` を導出する。
  - MuZero は `"network.rep"` / `"network.dyn"` / `"network.pred"` を導出する。
  - `Network` は構築 seed を保持し、Clone 時に再利用する。全直接呼び出しとテストを同時移行する。
- `ModuleRandomSource` を 1 Network 1 registry として追加し、purpose ごとに `RandomGenerator` を遅延生成・キャッシュする。SN は `"spectral_norm"` を使用し、parameter 初期化の global RNG を消費しない。`NetworkBodyBuilder` / `NetworkStructBuilder` も registry を既定値なしの必須引数とし、seed 0 の暗黙fallbackを置かない。
- SN 共通処理は FP32・Autocast OFF で実装する。
  - u/v を named buffer とし、専用 RNG で初期化後、保持ガード付き power iteration を15回実行する。
  - u/v 更新は `is_training() && GradMode::is_enabled()` の forward のみ。
  - `spectral` は `W/σ`、`spectral_cap` は `W/max(1,|σ|)`。σは毎 forward 再計算し、勾配を detach しない。
  - `spectral` の退化 init は説明付き fail-fast、`spectral_cap` の zero-init は正常系とする。
- 6 block は既存 module を保持しつつ、ON 時のみ functional Linear/Conv/SDPA 経路へ分岐する。ResBlock は conv1/conv2/downsample、CNBlock は dwconv/pwconv1/pwconv2、Transformer は layer ごとの Q/K/V/out_proj/linear1/linear2 を対象とする。Transformer の SN × `use_sdpa=false` は fail-fast する。
- `NetworkModule::GetSpectralNormEntries()` を既定空の仮想関数として追加する。Network 側は branch/block walk で完全名を付け、`dynamic_cast` や buffer 名解析を使わない。
- `Network::SoftCopyTo()` は、source/target のいずれかに SN があれば変更前に τ を検証する。許容集合は `{0≤τ≤0.1} ∪ {τ=1}`。成功時は buffer lerp 後に target u/v を再正規化し、失敗時は target を一切変更しない。
- DQN `NetworkModel` は SN を含む soft-update 構成を起動時にも同じ規則で検証する。hard-update 構成の未使用 τ は検証しない。
- `NetworkParameterNormSplit` を生ノルム2値、実効ノルム2値、max σ 2値、online invalid-count へ拡張し、target invalid-count は別の device scalar helper で得る。
  - 実効ノルムは生の二乗和から各 SN weight の生値を引き、実効値を加えて置換する。bias・norm affine・非SN parameterは生値を維持する。
  - SN 層なしでは 63/64 = 61/62、65/66 = NaN。
  - DQN は6公開値＋online/target sentinelの8要素、ImageClsは6公開値＋online sentinelの7要素を既存lazy D2H packへ載せる。
  - sentinel 異常時だけ cold-pathで再walkし、network側と層名を含めて fail-fastする。
- metricsへ `63_weight_norm_*_effective` と `65/66_spectral_sigma_*` をコメントアウト状態で追加する。`nn.txt` には全型共通キーを記載し、設計文書はNN、DQN、分析ガイドの3ページを同期する。既存の `CONTEXT.md` と ADR 0032 は変更しない。

## TDD 順序

各項目を1テストずつ RED→GREEN とし、GREEN 後のみ整理する。

1. Tracer bullet: Linear の `spectral` をConfigから構築し、専用seed、u/v buffer、実効forward、entry取得まで通す。
2. 数理契約: 既知行列のσ、解析勾配、対角行列、中心差分、FP32/autocast、使用時normalizeを検証する。
3. `spectral_cap`: zero-init恒等、subgradient、保持ガード、非ゼロ遷移後のcap発動を検証する。
4. Conv1d/Conv2d、ResBlock、CNBlock、Transformerを順に追加し、対象weight集合、noneビット一致、Q/K/V独立σとpacked勾配を確認する。
5. seed契約: purpose stream独立性、同seed再現、mode変更時のparameter一致、Clone/Copy/serialize復元を確認する。
6. SoftCopy: τ境界、NaN/範囲外の変更前失敗、連続lerp後のu/v単位ノルム、SNなしの既存挙動を確認する。
7. DQN起動時検証とonline/target validity sentinelを追加し、異常層名、lazy一括D2H、非測定時NaNを確認する。
8. ImageCls pack、4 source key、metrics定義、config dumpを追加する。
9. 関連テストを再実行後、重複だけを整理し、全core testへ進む。

## 検証

- コード編集前にReleaseをビルドし、baseline実行体をignored build artifactとして保存する。固定 workload は次とする。

```powershell
Push-Location apps\runner
$prd065RunnerArgs = 'bin\Release\AnetRLRunner.exe --workspace plasticity "run.$=run.@breakout_rr1_100m>run.@plasticity>run.@pl_check" train.seed=65065 app.exp_exit_step=100000 backend.cudnn_benchmark=false backend.cudnn_deterministic=true backend.deterministic_algorithms=true app.run_name=run_{t}_prd065_off'
cmd /s /c $prd065RunnerArgs
Pop-Location
```

`--config config\Atari.txt`の直接指定は共通`_main.txt`を合成せず、`DefaultDQNAgent.@baseline`を解決できないため使用しない。baseline/newとも同じ`plasticity` workspace解決経路を使う。

- base/new双方で解決済みconfigを確認し、`loss`、`q_max_mean/max`、61/62の全 `(tag, step, value)` をtimestamp抜きでcanonical化したSHA-256が一致することをOFF完全不変の合格条件とする。`agent_close.anet` のraw SHA-256は既存serialize非決定性があるため観測値として記録するが、合否ゲートにしない。
- MSVC Debugで全体をビルドし、`anet-core-test.exe "[spectral_norm]"`、関連NN/DQN/ImageCls/observer test、最後に全 `anet-core-test.exe` を実行する。
- Atari ON smoke:
  - `spectral`: `run.@pl_he`を重ね、AtariImpalaConv32/64、AtariImpalaRes32He/64He、AtariLinear512をONにする。
  - `spectral_cap`: plain Res32/64を含む同範囲をONにし、zero-initを維持する。
  - 63〜66をCLIで購読し、`inspect_run.py tags`で61〜65が`status=ok`かつ`count>0`、66はreadout側にSN対象層がある場合のみ`count>0`（対象層なしではNaN・`count=0`）、lossがfinite、config dumpが指定modeであることを確認する。
  - 両modeを同seedで各2回実行し、学習系列の一致を確認する。checkpointはserialize / load後のparameter・buffer復元を単体テストで確認し、raw archive checksumは合否に使わない。
- Release throughputは保存したbase実行体と新実行体をA/B/A/Bで交互実行する。20k〜100k stepの安定窓でOFF差分 `<2%` をゲートとし、新OFF/全層spectralも各2回交互実行してコストとSN profile rangeを記録する。長時間screening/confirmationは実施しない。

## 前提

- libtorch 2.11.0 の `parametrizations.spectral_norm` を参照実装とする。
- 旧 `BuildNetwork` overload、旧設定alias、互換分岐は残さない。
- Head、embedding、bias、normalization affine、layerscale、cls tokenはSN対象外。
- per-weight override、旧MHA対応、層別σ、actor σ cache、Dropout/DropPathのpurpose stream化、長時間効果実験は対象外。
- 既存の未コミット変更を保持し、Git staging・commit・pushは行わない。

## 実装・検証結果（2026-08-30）

- Phase 1 / Phase 2 と6 block、seed契約、SoftCopy、DQN / ImageCls metrics pack、設定・設計文書の同期を完了した。
- 第1ラウンドでは、Debug全core testが `520 cases / 5190 assertions`（`518 passed / 2 failed as expected`）で完走し、Release全体ビルドも成功した。
- レビュー修正後は、`cmake --build --preset x64-Debug` による全targetの再ビルドが成功した。対象テストは `[spectral_norm]` が `18 test cases / 103 assertions`、`[nn]` が `92 test cases / 875 assertions` で、いずれも全件成功した。
- レビュー修正後のDebug全core test再実行は、並行稼働中の `AnetRLRunner_ab.exe` とCUDA資源が競合して長時間停止したため中断した。テスト失敗は観測していないが、レビュー修正後の全件case / assertion数は未確定であり、上記 `520 cases / 5190 assertions` は第1ラウンドの結果として扱う。
- OFFの `loss`、`q_max_mean/max`、61/62はbase/new各935行が完全一致し、canonical SHA-256は `77A80741B636BDDA529445ABBCA12FC7A3C22C1B1F931A899CC7FD534497F855` だった。
- OFF throughputの20k〜100k step中央値はbase平均 `5800.997`、new平均 `5758.811`、差分 `-0.727%` で `<2%` gateを満たした。
- 最終ReleaseでのON smokeは両modeとも100k stepを完走し、lossは全311点finiteだった。`spectral` は先行run/final run各938行のcanonical SHA-256が `644D1B0AA1206B2FF5A980F8457EE483A433DAD8DBA5E39AB855BA017E216549`、`spectral_cap` は `CC81B9CF508746EC429123D8DCAB18C2815B26BA41029F3C1B6E791EC39AEC25` でそれぞれ完全一致した。
- SN構成のthroughputは同じ窓でおよそ `spectral: -16.4%`、`spectral_cap: -16.3%`（base比）だった。
- `agent_close.anet` のraw SHA-256は同じ保存済みbaseline実行体を同じseed・configで再実行しても一致しなかった。全checkpointのサイズは同じで、base/newとbase/base-repeatの最初の差分位置も同じmodel archive内 offset `0x025A7EE4` だったため、raw checkpoint SHA完全一致は既存serializeの非決定性により判定不能とした。学習系列は完全一致している。
- 65は `status=ok` かつ1点を記録した。66はreadout側にSN対象層がないため、公開契約どおりNaNとなり `status=ok` / count 0だった。「SN層なしでは65/66 = NaN」と「61〜66すべてcount>0」は同時に満たせないため、前者を優先した。
- `run.@pl_he` の既存 `res.init2.mode=he` はResBlock factoryの現行キー `init2.mode` と一致しない。検証runではCLIで正しいキーを明示し、実行中のユーザーbatchに関係する `Atari.txt` は変更しなかった。
