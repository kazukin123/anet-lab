# 043 impl: リリースパッケージング実装プラン

PRD: [043_release_packaging_10prd.md](043_release_packaging_10prd.md)。実施順に記載。1 ステップ=1 コミット目安。

## S1. リポジトリ再配置

1. `git mv viewers/metrics-viewer apps/metrics-viewer`、`git rm -r apps/metrics-viewer/docs`。
2. `apps/runner/*.bat` → `apps/` へ git mv し、各 bat の内部パスを修正:
   - 10/11(runner 起動系): `cd /d "%~dp0runner"` を先頭に追加。EXE パスは `bin\Release\...` のまま。
   - 21(Python viewer): `cd /d "%~dp0runner"`(RUNS_PATH=logs が runner 基準)、venv/script 参照 `..\..\viewers\metrics-tools` → `..\viewers\metrics-tools`(cd 後は runner 基準なので `..\..` のままか要確認 — cd 先基準で解決する)。
   - 22(Java viewer): jar → `%~dp0metrics-viewer\target\metrics-viewer.jar`。
   - 23/31/32/41/42/80/81/90/91: 参照パスを実物確認して同様に追随。
3. `git mv docs/optuna.md docs/design/optuna.md`。`grep -r "optuna.md"` で参照追随。
4. `grep -r "viewers/metrics-viewer"`(AGENTS.md、docs、.github、build-jar.bat 等)で旧パス追随。
5. `apps/README.md` 新設(PRD §3)。

## S2. licenses

1. 追加(公式ライセンス文): `cudnn_LICENSE.txt` / `wxWidgets_LICENCE.txt` / `intel_openmp_LICENSE.txt` / `box2d_LICENSE.txt`(third_party/box2d-2.4.2/LICENSE から) / `nlohmann_json_LICENSE.txt`(third_party/nlohmann から) / `tracy_LICENSE.txt`(third_party/tracy から)。
2. `git rm licenses/onnxruntime_LICENSE.txt`。
3. `apps/metrics-viewer/pom.xml` に license-maven-plugin(aggregate-add-third-party を package にバインド)。出力 `target/generated-sources/license/THIRD-PARTY.txt`。

## S3. ADR 0017 作成(PRD §5)

## S4. release.yml 作成(PRD §4)

- job 1 本(build-package)。steps: checkout → CUDA → LibTorch cache/DL → wx cache/build → cmake configure/build(Release) → setup-java + mvn package → doxygen → staging 構築(PowerShell) → zip 2 本 + サイズガード → (tag 時) gh release create / (dispatch 時) upload-artifact。

## S5. 検証

1. ローカル bat 動線(10_run / build-jar → 22)。
2. commit + push(main)。
3. `gh workflow run release.yml` → run 監視 → artifact サイズ確認。
4. tag push は人間。
