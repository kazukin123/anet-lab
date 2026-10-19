# 043: GitHub Actions リリースパッケージング(実行用 bin パッケージ)

- 発端: AnetRLRunner と MetricsViewer を「実行用パッケージ(bin)」として GitHub Release で配布したい。ソース非同梱・ドキュメント(design)とライセンス同梱。
- 実装: Claude(本セッションで PRD → impl → 実行まで)。本書は self-contained。
- 前提実測: `apps/runner/bin/Release` の exe+DLL 一式(展開後約 2.5GB)を zip 圧縮すると**約 1,798MB**。GitHub Release の 1 ファイル 2GiB 制限に収まるため、LibTorch CUDA ランタイムは**同梱**する(DL ヘルパーは後回し。溢れたら runtime 分離 zip へ移行)。

## 0. 決定事項(グリル済み)

1. **リリース物の位置付け = 実行用パッケージ(bin)**。ソース非同梱。
2. **zip 内レイアウト = リポジトリ配置のミラーサブセット**。平坦化しない。開発とリリースで同一 bat・同一パス(`target/metrics-viewer.jar` もそのまま)。→ ADR 0017。
3. トリガー: **tag push `v*` で Release 作成**。**workflow_dispatch はドライラン**(artifact アップロードのみ、Release 作成なし。バージョン文字列は入力欄、既定 `dev`)。
4. asset 構成: **本体 zip + Doxygen docs zip の 2 asset** を同一 Release に添付。
   - `anet-lab-<tag>-win64.zip`(実行一式)
   - `anet-lab-docs-<tag>.zip`(Doxygen 生成 html。`docs/html` は git 未管理なので CI で doxygen+graphviz を入れて生成)
5. bat 群は **`apps/` 直下へ集約移動**(選別・削減しない)。ルート薄 bat は作らない。
6. **MetricsViewer は `viewers/metrics-viewer` → `apps/metrics-viewer` へ移動**。`apps/metrics-viewer/docs/` は内容が古いので削除。
7. README: **`apps/README.md` を新設**(bat 説明+実行要件)。zip ルートに README は置かない(ルートは apps/docs/licenses のみで自明)。
8. 同梱 docs = **`docs/design/` 丸ごと**(選別フィルタなし)。`docs/optuna.md` は `docs/design/` へ移動して含める。
9. licenses: 不足分(cuDNN / wxWidgets / Intel OpenMP / box2d / nlohmann / tracy)を手動追加。Java 依存は **license-maven-plugin で THIRD-PARTY 一覧を自動生成**して同梱。`onnxruntime_LICENSE.txt` は非同梱物(onnxruntime.dll は bin に無い)なので削除。
10. `apps/runner/tools/` は**同梱する**。
11. 本番 Release を作る tag push は**人間が実施**(CI 検証は workflow_dispatch ドライランまで)。

## 1. zip 同梱内容(本体)

```
anet-lab-<tag>-win64.zip
├── apps/
│   ├── README.md                     (新設: bat 説明+実行要件)
│   ├── *.bat                          (集約移動後の全 bat)
│   ├── runner/
│   │   ├── bin/Release/               (exe + DLL。*.ilk / *.pdb 除外)
│   │   ├── config/
│   │   └── tools/
│   └── metrics-viewer/
│       └── target/metrics-viewer.jar
├── docs/
│   └── design/                        (丸ごと。optuna.md 移動済み)
└── licenses/                          (補完後 + metrics-viewer_THIRD-PARTY.txt)
```

除外: ソース全部、ルート README.md(開発者向け)、`runs*` / `logs`(アプリが自動生成: RunnerApp.cpp:353 / app_util.cpp:107)、`viewers/`(metrics-tools は Python 開発用)、`third_party/`、`docs/html`(別 zip)。

## 2. リポジトリ再配置

- `git mv viewers/metrics-viewer apps/metrics-viewer` + `apps/metrics-viewer/docs/` 削除。
- `apps/runner/*.bat` 全部を `apps/` へ移動。パス修正:
  - runner 系は `cd /d "%~dp0runner"` してから実行(runs/logs の出力先が CWD 依存)。
  - 22 番の jar パス → `%~dp0metrics-viewer\target\metrics-viewer.jar`。
  - `..\..\viewers\metrics-tools` → `..\viewers\metrics-tools`(21/23 番)。
- `git mv docs/optuna.md docs/design/optuna.md`(参照リンクは grep で追随)。
- 旧パス `viewers/metrics-viewer` への参照(AGENTS.md・docs 等)を grep で追随修正。

## 3. 実行要件(apps/README.md に記載)

- Windows x64、NVIDIA GPU + CUDA 13 世代ドライバ(R580 以降)
- MetricsViewer: Java 17 以降
- 起動: `apps/10_run.bat`(Runner)、`apps/22_metrics_viewer_java.bat`(Viewer)

## 4. リリースワークフロー(.github/workflows/release.yml 新設)

- windows-2022、Release 構成のみ。windows-ci.yml の手順を流用し **cache キーを共有**:
  - LibTorch: `libtorch-2.12.0-cu130-windows-v1`(DL ステップも同一、release/debug 両方 DL してキャッシュ共有を維持)
  - wxWidgets: `wxwidgets-3.3.1-vs2022-Release`
- CUDA Toolkit(Jimver/cuda-toolkit @ 13.0.0)→ cmake configure/build(Release)→ POST_BUILD で `apps/runner/bin/Release/` に DLL 一式が揃う(既存機構)。
- setup-java(temurin 17)+ `mvn package -DskipTests`(apps/metrics-viewer)。
- `choco install doxygen.install graphviz` → `docs/` で doxygen → `docs/html`。
- staging に §1 の構成をコピーして zip 2 本作成。
- **サイズガード**: 本体 zip > 2,000MB で fail(2GiB 制限の事前検知)。
- tag push 時のみ `gh release create <tag> --generate-notes` + asset 2 本。dispatch 時は actions/upload-artifact。

## 5. ADR

- `docs/adr/0017-release-zip-mirrors-repo-layout.md`: 決定=リリース zip はリポジトリ配置のミラーサブセット(同一 bat、`target/` パスもそのまま)。代替=平坦レイアウト+専用起動スクリプト。理由=開発/配布の二重メンテ回避。

## 6. 検証

1. ローカル: 移動後の `apps/10_run.bat` で Runner 起動・run 作成、`build-jar.bat` → `apps/22_metrics_viewer_java.bat` で Viewer 起動を確認。
2. コミット + main push(workflow_dispatch はリモートに workflow が必要)。
3. `gh workflow run` でドライラン → artifact サイズ ≤ 2,000MB、クリーン展開して bat 起動、docs zip の `html/index.html` 確認。
4. tag push は人間。
