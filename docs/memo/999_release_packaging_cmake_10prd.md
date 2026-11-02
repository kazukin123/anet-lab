# リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする PRD

> 2026-10-31 起票（暫定・999。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍の記述で探し直す。
> 起点: 2026-10-31 の会話。VS Code の CMake の画面で CPack とワークフロー preset を知り、GitHub Actions で作っているリリース物の手順を CMake 側で整理できないかを調べた。ユーザーの望みは「GitHub に依存せず、CMake に統合して、リリース物を手元でも作れること。環境整備以外の実行手順を CMake に統合したい」。
> 関連: [PRD043](done/043_release_packaging_10prd.md)（今のリリースの形）、[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md)・[ADR 0020](../adr/0020-release-archives-carry-project-license.md)・[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md)、[release.yml](../../.github/workflows/release.yml)、[CMakePresets.json](../../CMakePresets.json)、[999 CI/CD 改善](999_cicd_improvement_10prd.md)（CI のビルド確認。リリース物の作成は対象外としている）。

## 1. 目的と問題

リリース物（実行用 zip と Doxygen zip）は、GitHub Actions の [release.yml](../../.github/workflows/release.yml) の中でしか作れない。中身の組み立て・検査・zip 化は、その中の約 90 行の PowerShell（181〜272 行）にあり、CMake が持つのは exe を作って DLL を隣にコピーするところまでである。

このため次のことが起きる。

- 手元でリリース物を作って確かめられない。確かめる手段は workflow_dispatch のドライランだけで、1 回ごとに CI のビルド（CUDA・LibTorch・wxWidgets の準備を含む）を待つ。release.yml は 2026-08-07 に新設され、翌日までに 5 回の変更が入った（wx の DLL を bin へコピーする修正を含む。`git log -- .github/workflows/release.yml`）。
- 中身の決まりが GitHub の書式のファイルにあり、ほかの CI や手元へ持ち出せない。
- 何を入れるかの定義が、それを作るビルドの定義（各 CMakeLists）から離れている。ディレクトリを動かしたときは release.yml のコピー対象を直す（[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md) の Consequences）。

## 2. 現状（2026-10-31 に確認したこと）

1. [release.yml](../../.github/workflows/release.yml) の流れ: tag push（`v*`）で Release を作り、workflow_dispatch はドライランにする（3〜12 行）。版番号は tag 名から先頭の `v` を除いたものか、dispatch の入力（既定 `dev`）（31〜44 行）。CUDA Toolkit の導入、LibTorch の取得とキャッシュ、wxWidgets のビルドとキャッシュ（49〜124 行）。configure は preset を使わずに引数を直に書き、`-DANET_ENABLE_ATARI=OFF` を渡す（128〜144 行）。Maven で jar を作り（152〜162 行）、choco で Doxygen と Graphviz を入れて `docs/` で doxygen を回す（166〜176 行）。
2. release.yml の組み立て（181〜272 行）: 作業フォルダへ次をコピーする。直下の `LICENSE`・`NOTICE`、`apps/*.bat`（名前に `atari` を含むものを除く。203 行）と `apps/README.md`、`apps/runner/bin/Release/*` から `.ilk`・`.pdb` を除いたもの（208 行）、`apps/runner/config`・`workspaces`・`tools`、jar、`docs/design`、`licenses` と Maven が作った THIRD-PARTY.txt。`tar.exe` で `anet-lab-<版>-win64.zip` と `anet-lab-doxygen-<版>.zip` を作る（240〜246 行）。検査は、法的文書と必須の entry（192〜194 行・250〜262 行）、Atari 用の bat の混入（255〜258 行）、wx の core DLL（215〜217 行）、THIRD-PARTY.txt（232〜237 行）、2,000MB の上限（265〜267 行）。tag なら `gh release create --draft --verify-tag --generate-notes`、dispatch なら作業フォルダの木を artifact へ上げる（275〜302 行）。
3. [windows-ci.yml](../../.github/workflows/windows-ci.yml) の configure も、同じ引数を別に書いている（147 行付近）。
4. [CMakePresets.json](../../CMakePresets.json) は schema の version 3 で、`x64-Debug`・`x64-RelWithDebInfo`・`x64-Release` の 3 つを持つ（Ninja。MSVC の環境は外で用意する前提の `strategy: external`）。ワークフロー preset と package preset は schema 6（CMake 3.25 以上）が要る。ワークフロー preset は引数を取らない。2026-10-31 の作者の手元の CMake は 4.1.2、Visual Studio 同梱は 3.31.6。GitHub の windows-2022 の CMake の版は確かめていない。
5. 実行ファイルの出力先は、ソース木の中に構成名で固定されている（[apps/runner/CMakeLists.txt:76-80](../../apps/runner/CMakeLists.txt)、[core/anet-core/CMakeLists.txt:91-95](../../core/anet-core/CMakeLists.txt)、各 Env の CMakeLists、例 [lunarlander1/CMakeLists.txt:77-79](../../core/envs/lunarlander1/CMakeLists.txt)）。preset でビルドフォルダを分けても、Release 構成のビルドは `apps/runner/bin/Release` などへ書く。静的ライブラリでは、anet-core だけが同じくソース木の `core/anet-core/lib/<構成>` へ出す（[core/anet-core/CMakeLists.txt:85-89](../../core/anet-core/CMakeLists.txt)）。ほかの静的ライブラリの出力先はビルド木である。
6. exe の隣の DLL は、ビルド後のコピー（POST_BUILD）で置く。LibTorch は `lib` の `*.dll` を全部（[apps/runner/CMakeLists.txt:95](../../apps/runner/CMakeLists.txt)）、wxWidgets は 4 つ（[CMakeLists.txt:198-211](../../CMakeLists.txt)）、`third_party/runtime_dlls` の `*.dll` を全部（[apps/runner/CMakeLists.txt:113-128](../../apps/runner/CMakeLists.txt)。注記は「実行時ロードされる DLL（SDL2.dll 等）」）。`third_party/runtime_dlls` の管理下のファイルは README.md だけである（`git ls-files`）。
7. anet 自身の CMake に install のルールは無い。第三者のうち box2d だけが、ヘッダーと静的ライブラリの install のルールを持つ（[box2d-2.4.2/CMakeLists.txt:31](../../third_party/box2d-2.4.2/CMakeLists.txt)、[box2d-2.4.2/src/CMakeLists.txt](../../third_party/box2d-2.4.2/src/CMakeLists.txt) の 122・139・147・160 行）。[CMakeLists.txt:225](../../CMakeLists.txt) で取り込んでいる。
8. Atari: `ANET_ENABLE_ATARI` は既定が `AUTO` で、`ALE_ROOT` があれば有効になる（[CMakeLists.txt:35-77](../../CMakeLists.txt)）。[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md) は、リリースを `OFF` でビルドし、リリースに Atari を含めないと決めている。
9. 作者の手元の作業ツリー（2026-10-31）: `apps/runner/bin/Release` に、`AnetRLRunner_ab.exe` など試行用の exe の複製が 10 本と、CI のビルドでは入らない DLL（jpeg62・libpng16・tiff・zlib1。手元の環境変数 `CMAKE_TOOLCHAIN_FILE` は vcpkg を指している）がある。`apps/runner/workspaces` には Git が無視する `runs/` と管理外の workspace があり、`apps/runner/tools` には `__pycache__/` が、`third_party/runtime_dlls` には管理外の `SDL2.dll` がある。今の release.yml のように bin やフォルダを丸ごとコピーする方式は、CI の checkout がまっさらだから成り立っている。また、手元の `Torch_DIR_RELEASE` は LibTorch 2.11.0+cu130 を指し、CI は 2.12.0+cu130 である（release.yml の `LIBTORCH_VER`、[040 開発環境構築ガイド](../design/040_development_environment.jp.md) §3.3）。
10. Metrics Viewer: jar は `apps/metrics-viewer/target/metrics-viewer.jar`（[pom.xml:86](../../apps/metrics-viewer/pom.xml)）。THIRD-PARTY の一覧は license-maven-plugin が `${project.build.directory}/generated-sources/license` に作る（pom.xml の 101・173 行）。Maven wrapper（[mvnw.cmd](../../apps/metrics-viewer/mvnw.cmd)）がある。ソース木の jar は、日常の Viewer の起動（`apps/22_metrics_viewer_java.bat`）が使う。
11. Doxygen: `doc` ターゲットは Doxygen が見つかったときだけ作る（[CMakeLists.txt:244-255](../../CMakeLists.txt)）。[Doxyfile](../Doxyfile) は `OUTPUT_DIRECTORY` が `.`（`docs/` で実行するので出力は `docs/html`。77 行）、`HAVE_DOT` が `YES`（2642 行）で、Graphviz の `dot` が要る。
12. PDB: Release 構成はダンプ解析用に完全な PDB を作る（[CMakeLists.txt:111-121](../../CMakeLists.txt)）。リリースの zip は `.pdb` を除く（release.yml 208 行）。[symbolize.py](../../apps/runner/tools/symbolize.py) は exe と同じフォルダの PDB を探す（1〜4 行）。手元の `AnetRLRunner.pdb` は約 78MiB（81,760,256 bytes）。anet-core と各 Env は静的ライブラリである（例 [core/anet-core/CMakeLists.txt:28](../../core/anet-core/CMakeLists.txt)）。
13. 大きさ: bin の一式は展開して約 2.5GB、zip で約 1,798MB（[PRD043](done/043_release_packaging_10prd.md) 冒頭）。GitHub Release の 1 ファイルの上限は 2GiB。
14. Windows 専用の部分: core の共通基盤に `windows.h` を読むファイルがある（`app_util.cpp`・`config.cpp`・`env.cpp`・`exception.cpp`・`profile.cpp`）。起動の launcher は bat だけである（`apps/*.bat`）。
15. [apps/README.md](../../apps/README.md) の「リリース zip について」（42〜52 行）は Doxygen の zip を `anet-lab-docs-<version>.zip` と書いているが、release.yml が作る名前は `anet-lab-doxygen-<版>.zip`（241 行）である。
16. リリースの実績: tag は v0.1.0（2026-06-26）・v0.2.0（07-01）・v0.3.0（08-09）。tag push は人間が行う（[PRD043](done/043_release_packaging_10prd.md) §0 の 11）。

## 3. ゴール・非ゴール・成功の測り方

**ゴール**

- G1: 依存（CUDA Toolkit・LibTorch・wxWidgets・JDK・Doxygen・Graphviz）と MSVC の環境を用意した Windows の機械なら、GitHub を使わずに 1 つのコマンド（`cmake --workflow --preset x64-ReleasePackage`）でリリース物を作れる。
- G2: 実行用 zip と Doxygen zip の中身（ファイルの entry の一覧）は、今の release.yml が作るものと同じである。手元で作っても、同じコミット・同じ依存の版なら CI と同じになる。
- G3: GitHub Actions に残るのは、trigger、版番号の決定（tag 名から `v` を除く・dispatch の既定 `dev`）と環境変数での受け渡し、依存の準備、MSVC の環境の用意、同じコマンドの実行、GitHub Release への添付と artifact の保存だけになる。release.yml から組み立ての PowerShell が消える。
- G4: 手元でリリース物を作っても、ソース木の中（`out/` を除く）のファイルを作りも変えもしない。日常に使う `apps/runner/bin/Release/AnetRLRunner.exe`（Atari 入り）や Viewer の jar を上書きしない。
- G5: 配った exe のクラッシュダンプを読むための PDB の zip と、各 zip の SHA256 のファイルを一緒に作る。

**非ゴール**

- 依存の準備を CMake で行うこと（FetchContent で LibTorch を取るなど）。ユーザーが「環境整備以外」と範囲を決めた。消える案は依存の取得まで 1 つのコマンドに入れる案で、本当の制約ではない（LibTorch の取得は CMake でも書ける）が、CUDA Toolkit の導入は CMake では行えないので、別の関心として分ける。
- zip の配置や中身を変えること（G5 の zip と SHA256 を足すことを除く）。消える案は配布用の平坦な配置で、[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md) が退けた形である。本当の制約ではない（クリーンブレークで変えられる）が、本書の問題（手元で作れない）とは別の関心である。
- Linux・macOS のリリース物。消える案は CPack の DEB・RPM・DMG。本当の制約ではないが、先にビルドの移植が要る（§2 の 14）。
- インストーラ（NSIS・WiX・Inno Setup）。消える案は CPack のインストーラの生成器。本当の制約ではない（生成器を足せば作れる）が、2GiB の上限は形式で変わらず、今は要求が無い。
- Atari をリリースに含めること。[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md)（GPL の隔離）による本当の制約である。
- windows-ci.yml の configure を preset へ寄せること。消える案は CI のビルド確認も本書の preset で回す案で、本当の制約ではない（同じ書き方で寄せられる）が、CI のビルド確認の関心なので [999 CI/CD 改善](999_cicd_improvement_10prd.md) の L1 へ回す。
- GitHub Release への添付を CMake から行うこと。消える案は package の工程のあとに CMake のスクリプトから `gh release create` を呼ぶ案で、本当の制約ではない（書ける）が、GitHub に依存する工程なので G1（GitHub を使わずに作る）の外に置く。
- Runner に版番号を埋め込むこと。消える案は `ANET_RELEASE_VERSION` を exe の版の情報や定数に入れる案で、本当の制約ではない（同じ cache 変数から作れる）が、配布物の作り方ではなく exe の機能なので別の関心として分ける。

**成功の測り方**

- G1〜G5: §7 の受入 1〜10 が通る。
- 次にリリースの中身を変えるとき（ファイルを足す、DLL が増える）、CI のドライランを待たずに手元で zip を作って確かめられる（使う場面として記録し、合否にはしない）。

## 4. 確定事項（暫定の既定。会話で合意した決定ではない。§9 とあわせてグリルで見直す）

- **D1. 入口はワークフロー preset `x64-ReleasePackage` の 1 つにし、configure → build → package の 3 つの工程を回す。** test の工程は持たない（今の release.yml も test を回していない。テストの自動化は [999 CI/CD 改善](999_cicd_improvement_10prd.md) の L1 の関心。再開条件は §8）。CMakePresets.json の schema を 3 から 6 へ上げる。理由: G1。依存: §2 の 4（CMake 3.25 以上。ワークフロー preset は引数を取らない）。
- **D2. リリースの構成は、cache 変数 `ANET_ENABLE_RELEASE_PACKAGE`（BOOL、既定 OFF）の 1 つで表す。** ON のときだけ、(a) 版番号を要求し、(b) Atari が有効なら止め、(c) JDK・Doxygen・Graphviz を要求し、(d) 今ソース木に書いているビルドの出力をビルド木に出し（D3）、(e) jar と Doxygen をビルド木で作るターゲットを持ち（D7）、(f) CPack を設定する。OFF は今と同一にする。理由: 日常の構成に JDK・Doxygen・版番号を要求しない。出力先の切り替えを別のキーにせず ON に結び付けるのは、リリースの構成で日常の実行物を上書きする組み合わせを作れなくするためである（§2 の 5）。依存: D3。
- **D3. ON のときは、今ソース木に書いているビルドの出力（exe・テストの exe・anet-core の静的ライブラリ）を、各ターゲットのビルド木の既定の場所に出す。OFF のときは今のソース木の `bin/<構成>`・`lib/<構成>` のまま。** ビルド後の DLL のコピー先は exe の場所に従うので、DLL もビルド木に置かれる。理由: G4。依存: §2 の 5。
- **D4. 中身は install のルールで書き、3 つの component（`runtime`・`doxygen`・`symbols`）を 3 本の zip に 1 対 1 で対応させる。CPack が包むのはこの 3 つだけで、box2d の install のルールの分は入らない。zip の中に最上位のフォルダを作らない（今の zip と同じく、直下に `LICENSE`・`NOTICE`）。** 理由: 何を入れるかを、それを作るターゲットの CMakeLists の隣に書ける（Atari 用の bat を入れるかも、ビルドの設定と同じ場所で決まる）。CPack は component ごとに zip を分けられる。依存: §2 の 7。
- **D5. runtime の中身は §5.3 の表（今の release.yml と同じ）にする。ビルドの出力以外は Git の管理下のファイルだけを入れる。** 管理下のファイルに未コミットの変更があれば、作業ツリーの内容のまま入る（§9 の 1）。理由: G2。手元の作業ツリーには管理外・Git が無視するファイルがある（§2 の 9）。CI の checkout は管理下のものだけなので、管理下に絞ると手元と CI がそろう。依存: D4。
- **D6. exe の隣に置く DLL の一覧は 1 か所で作り、ビルド後のコピーと install の両方がそれを使う。** LibTorch は `lib` の `*.dll` を全部、wxWidgets は 4 つ、`third_party/runtime_dlls` はビルド後のコピーでは全部・install では管理下のものだけ。exe の import の表から自動でたどる方式は使わない。理由: 2 つの経路の一覧を別々に書くと、片方の書き漏らしが日常の開発で見つからない（[ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md) の「開発とリリースが同じ経路を通る」性質を、DLL について保つ）。自動でたどる方式は、実行時に読み込む DLL を拾えない（§2 の 6）。依存: D5。
- **D7. jar と Doxygen を CMake のターゲットにし、ALL から外して、リリースの build preset だけが作る。** jar は Maven wrapper（`mvnw.cmd`）で作り、出力先をビルド木にする（pom の出力先を property で上書きできるようにし、既定は今の `target`）。Doxygen は今の Doxyfile を使い、出力先だけをビルド木に替える。zip の中の置き場は今と同じ（`apps/metrics-viewer/target/metrics-viewer.jar`、`licenses/metrics-viewer_THIRD-PARTY.txt`、`html/`）。理由: G1（Maven と Doxygen も同じコマンドで回す）、G4（ソース木の jar と `docs/html` を作り直さない。§2 の 10・11）。依存: D2。
- **D8. 版番号は cache 変数 `ANET_RELEASE_VERSION` で受け取り、preset は同じ名前の環境変数の値を渡す。ON のとき、空か許さない形（§5.2）なら configure で止める。空を `dev` に読み替えない。** tag 名から `v` を除く処理と、dispatch の既定 `dev` は workflow に残す。理由: ワークフロー preset は引数を取らない（§2 の 4）。黙った代替をしない（[AGENTS.md](../../AGENTS.md) §設定値の扱い）。tag の書き方は GitHub の運用の語である。依存: D1。
- **D9. 包んだあとに §5.4 の検査をし、1 つでも外れれば失敗で終わる。** 今の release.yml の検査に、DLL の一致と第三者の開発用ファイルの不在を足す。検査は、リリースのビルドを待たずに小さな zip で単体で試せる形にする。そのために、ビルドの結果に依る比べる相手（DLL の名前の一覧と大きさの上限）は検査の入力として受け取り、検査自身はビルド木を読まない（§5.4 の 4・6、§7 の受入 7）。理由: 今の検査を手元でも回す（G1・G2）。DLL の一致は、D6 の一覧を通らない追加を見つける。依存: D4〜D6。
- **D10. 各 zip の隣に SHA256 のファイル（`<zip の名前>.sha256`）を作り、GitHub Release にも添付する。** 理由: 手元と CI の zip、配った zip と手元の zip を突き合わせられる（G2 の確かめにも使う）。
- **D11. symbols の zip（`anet-lab-symbols-<版>-win64.zip`）には、`AnetRLRunner.pdb` をリポジトリと同じ置き場（`apps/runner/bin/Release/`）で入れ、直下に `LICENSE`・`NOTICE` を置く。** runtime zip と同じフォルダへ展開すると exe の隣に PDB が来て、[symbolize.py](../../apps/runner/tools/symbolize.py) がそのまま使える。理由: G5。Release は完全な PDB を作っているが配っていない（§2 の 12）。法的文書は [ADR 0020](../adr/0020-release-archives-carry-project-license.md)（各 zip が単独で再配布できる）に合わせる。依存: D4。
- **D12. release.yml は、G3 に挙げたもの（trigger、版番号の決定と環境変数での受け渡し、依存の準備、MSVC の環境の用意、`cmake --workflow --preset x64-ReleasePackage`、GitHub Release への添付と artifact の保存）だけにする。組み立ての PowerShell（§2 の 2）は消す。tag push は人間のまま。** 理由: G3。クリーンブレーク（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）で、2 つの組み立てを残さない。依存: D1〜D11。
- **D13. 既存の ADR との関係。** [ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md) の採用理由「パッケージングは対象パスをコピーして zip にするだけ」と、Consequences の「リポジトリのディレクトリを動かす変更は release.yml のコピー対象への追随が必要」を置き換える。配置のミラーは保ち、追随先が CMake の install のルールになる。[ADR 0020](../adr/0020-release-archives-carry-project-license.md) の Consequences「Release workflow は zip 作成後に必須 entry を検証」の担い手を、CMake の包む工程へ移す。法的文書を直下に置く判断は保つ。[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md) の「リリースは OFF でビルド」は、preset と D2 の検査で守る。ADR は、本書をグリルして D3〜D6 が決まってから起こす（§9 の 8）。理由: 本書の確定事項は会話で合意していない案であり、ADR に判断として残すのは決まってからにする。

## 5. 実装契約

### 5.1 preset と入口

| 種類 | 名前 | 中身 |
|---|---|---|
| configure | `x64-ReleasePackage` | Ninja、ビルドフォルダ `out/build/x64-ReleasePackage`、`CMAKE_BUILD_TYPE=Release`、`ANET_ENABLE_ATARI=OFF`、`ANET_ENABLE_RELEASE_PACKAGE=ON`、`ANET_RELEASE_VERSION=$env{ANET_RELEASE_VERSION}`。environment・architecture・toolset は `x64-Release` と同じ |
| build | `x64-ReleasePackage` | 作るターゲットは `AnetRLRunner`、jar のターゲット、`doc` |
| package | `x64-ReleasePackage` | 生成器は ZIP。出力先は `out/build/x64-ReleasePackage/package` |
| workflow | `x64-ReleasePackage` | 上の 3 つを configure → build → package の順に回す |

既存の 3 つの preset は変えない。依存の場所は、手元の開発と同じ環境変数（`Torch_DIR_RELEASE`・`wxWidgets_DIR`・`CUDA_PATH`。[040 開発環境構築ガイド](../design/040_development_environment.jp.md) §3.8）で渡し、preset に機械のパスを書かない。

手元での使い方（MSVC の環境を用意したシェル。040 §6.1 と同じ）:

```powershell
$env:ANET_RELEASE_VERSION = "dev"
cmake --workflow --preset x64-ReleasePackage
```

### 5.2 cache 変数と判定の網羅

| 名前 | 型・値域 | 既定 | 外れたとき |
|---|---|---|---|
| `ANET_ENABLE_RELEASE_PACKAGE` | BOOL（CMake の真偽の定数） | OFF | 真偽の定数でなければ configure で止める（名前と値を出す） |
| `ANET_RELEASE_VERSION` | 文字列。`^[0-9A-Za-z][0-9A-Za-z._-]*$` に合うもの | 空 | ON で空か形に合わなければ configure で止める（名前・値・許す形を出す）。OFF で空でなければ、読まないことを WARN で 1 回出す |

configure の判定（上から順に当てはめ、最初に当たった行で決まる）:

| # | `ANET_ENABLE_RELEASE_PACKAGE` | 条件 | 結果 |
|---|---|---|---|
| 1 | OFF | どれでも | 今と同一。`ANET_RELEASE_VERSION` が空でなければ WARN を 1 回 |
| 2 | ON | Atari が有効（`ANET_ENABLE_ATARI=ON`、または `AUTO` で ALE が見つかった） | 止める（`ANET_ENABLE_ATARI` と ADR 0025 を出す） |
| 3 | ON | 版番号が空か、形に合わない | 止める |
| 4 | ON | `java`・`doxygen`・`dot` のどれかが見つからない | 止める（見つからない道具をすべて挙げる） |
| 5 | ON | 2〜4 のどれでもない | リリースの構成にする |

JDK の版（17 以上）は Maven が検査する。ON でない構成で package を回すと、CPack の設定が無いので CPack 自身のエラーで止まる。

### 5.3 包む中身

runtime（`anet-lab-<版>-win64.zip`）:

| zip の中のパス | 元 | 条件 |
|---|---|---|
| `LICENSE`・`NOTICE` | リポジトリ直下 | |
| `apps/*.bat` | `apps/` | 管理下のもの。名前に `atari` を含むもの（大文字小文字を区別しない）を除く |
| `apps/README.md` | `apps/` | |
| `apps/runner/bin/Release/AnetRLRunner.exe` | リリースのビルド | |
| `apps/runner/bin/Release/*.dll` | D6 の一覧 | `third_party/runtime_dlls` の分は管理下のものだけ |
| `apps/runner/config/**`・`apps/runner/workspaces/**`・`apps/runner/tools/**` | 同じパス | 管理下のもの（workspaces は今は各 workspace の `config/_main.txt`） |
| `apps/metrics-viewer/target/metrics-viewer.jar` | Maven の出力（ビルド木） | |
| `docs/design/**` | 同じパス | 管理下のもの |
| `licenses/**` | 同じパス | 管理下のもの |
| `licenses/metrics-viewer_THIRD-PARTY.txt` | license-maven-plugin の出力（ビルド木） | |

doxygen（`anet-lab-doxygen-<版>.zip`）: `LICENSE`・`NOTICE`・`html/**`（Doxygen の出力。ビルド木）。

symbols（`anet-lab-symbols-<版>-win64.zip`）: `LICENSE`・`NOTICE`・`apps/runner/bin/Release/AnetRLRunner.pdb`。

それぞれの zip の隣に `<zip の名前>.sha256` を置く。runtime に `.pdb`・`.ilk`、テストの exe、第三者のヘッダーと静的ライブラリは入らない。

### 5.4 包んだあとの検査

1. runtime に `LICENSE`・`NOTICE`・`apps/`・`docs/`・`licenses/` がある（今の release.yml の 250〜253 行）。
2. runtime の `apps/` の直下に、名前に `atari` を含む bat が無い（大文字小文字を区別しない。今の release.yml の 254〜258 行）。
3. runtime の `apps/runner/bin/Release/` に wx の core DLL（`wxmsw*_core_*.dll`）がある（今の release.yml の 215〜217 行）。
4. runtime の `apps/runner/bin/Release/` の DLL の名前の集合が、検査の入力として渡された DLL の名前の一覧と一致する（D6）。リリースの工程では、CMake がこの一覧を、リリースのビルドで exe の隣に置かれた DLL の名前から `third_party/runtime_dlls` の管理外の DLL を除いて作り、渡す。差があれば、足りない DLL と余る DLL の名前を出す。
5. runtime に、`.pdb`・`.ilk`・`.lib`・`.exp` の entry と、`include/` で始まる entry が無い。
6. runtime の大きさを 1MiB 単位で四捨五入した値が 2,000 以下である（今の release.yml の 265〜267 行と同じ定義）。上限は検査の入力で、CMake は 2,000 を渡す。
7. doxygen に `LICENSE`・`NOTICE`・`html/index.html` がある。
8. symbols に `LICENSE`・`NOTICE`・`apps/runner/bin/Release/AnetRLRunner.pdb` がある。
9. 各 `.sha256` の値が、その zip の SHA256 と一致する。

外れたら、外れた検査の番号と entry を出し、package の工程を失敗で終える。zip は消さない（中を見て直すため）。CI はこの失敗で止まり、添付しない。

### 5.5 GitHub Actions に残すもの

G3 と D12 の列挙を、今のステップに当てるとこうなる。

- 残す: trigger、版番号の決定（tag 名から `v` を除く・dispatch の既定 `dev`）、依存の準備（CUDA Toolkit の導入、LibTorch の取得とキャッシュ、wxWidgets のビルドとキャッシュ、JDK の用意、Doxygen と Graphviz の導入）。
- 足す: MSVC の環境を用意して `cmake --workflow --preset x64-ReleasePackage` を回すステップ。環境変数での受け渡しとして、`Torch_DIR_RELEASE`・`wxWidgets_DIR`・`ANET_RELEASE_VERSION` と、Graphviz の PATH を置く。
- 変える: tag のときの `gh release create` は 3 本の zip と 3 つの `.sha256` を添付する（`--draft --verify-tag --generate-notes` は今のまま）。dispatch のときの artifact の形は §9 の 4。
- 消す: Configure CMake、Build、Build MetricsViewer jar、Generate Doxygen docs、Package release zips の各ステップ。

### 5.6 移行対象

- [release.yml](../../.github/workflows/release.yml): §5.5。
- [CMakePresets.json](../../CMakePresets.json): schema を 6 にし、§5.1 の 4 つを足す。
- [CMakeLists.txt](../../CMakeLists.txt): D2・D8 の cache 変数と §5.2 の判定、CPack の設定、`doc` の出力先（D7）、jar のターゲット、DLL の一覧を作る共通の処理（D6）、§5.4 の検査の呼び出し。
- [apps/runner/CMakeLists.txt](../../apps/runner/CMakeLists.txt): 出力先（D3）、DLL の一覧（D6）、install のルール（§5.3）。
- [core/anet-core/CMakeLists.txt](../../core/anet-core/CMakeLists.txt) と `core/envs/*/CMakeLists.txt`: 出力先（D3。ON の構成では、テストの exe と anet-core の静的ライブラリもソース木へ書かない）。
- [apps/metrics-viewer/pom.xml](../../apps/metrics-viewer/pom.xml): 出力先の property（D7。既定は今と同じ）。
- [040 開発環境構築ガイド](../design/040_development_environment.jp.md) と [英語版](../design/040_development_environment.en.md): §4.2 の preset の表、§6 にリリース物を作る節（MSVC の環境、環境変数、コマンド、出力先、要る空き）、§3 に JDK・Doxygen・Graphviz がリリース物を作るときに要ること。
- [AGENTS.md](../../AGENTS.md) §ビルドの「利用可能な preset」: `x64-ReleasePackage` を足す。
- [apps/README.md](../../apps/README.md) の「リリース zip について」: Doxygen の zip の名前を今の名前へ直し（§2 の 15）、symbols の zip と `.sha256` を足す。
- [動線マップ](../agents/README.md) の CI の行は、必要なら `anet-harness-map` で作り直す。
- 変えないもの: windows-ci.yml（非ゴール）、CHANGELOG、PRD043、ADR の本文（注記を足すのは §9 の 8）。
- 互換層は置かない。古い組み立てをドライランの経路にも残さない。

### 5.7 性能と資源

- Runner の実行時の性能は変わらない。コンパイルの設定は CMakeLists が決めていて変えない。CI の generator は Visual Studio から Ninja に変わるので、exe のバイト列は前と同じにならない（受入 3 はファイルの一覧で比べる）。ホットパスは触らない。
- 手元の資源: ビルド木の分に加えて、CPack が install した一式（約 2.5GB）と 3 本の zip（約 1.8GB と Doxygen と PDB 約 78MiB）を `out/build/x64-ReleasePackage` に置く。空きは 5GB 以上を目安にする（見積もり。受入 2 で実測して 040 に書く）。
- 実時間: 手元では、初回の Release の全体のビルドが大半を占める見込み。CI の実時間は受入 10 で記録する。

### 5.8 クラス一覧

なし（C++ のクラスを足さない・変えない・消さない）。

## 6. 細部（暫定で置いた既定）

1. preset の名前は `x64-ReleasePackage`（理由: 既存の `x64-<構成>` の並びに置き、何をする preset かを名前で示す）。
2. cache 変数は `ANET_ENABLE_RELEASE_PACKAGE`・`ANET_RELEASE_VERSION`（理由: 既存の `ANET_ENABLE_PROFILE`・`ANET_ENABLE_ATARI` と同じ形。版番号の環境変数も同じ名前にし、preset で 1 対 1 に渡す）。ドメインの語ではないので、[CONTEXT.md](../../CONTEXT.md) は変えない。
3. component の名前は `runtime`・`doxygen`・`symbols`（理由: zip の名前の語に合わせる。実行用の zip は名前に語を持たないので `runtime`）。
4. symbols の zip の名前は `anet-lab-symbols-<版>-win64.zip`（理由: exe と同じ機械向けなので `win64` を付ける。機械によらない Doxygen の zip に `win64` が無い今の形に合わせる）。
5. jar のターゲットの名前は `metrics-viewer-jar`（理由: 既存の `anet-core-test` などと同じ、小文字とハイフンの形）。
6. zip の出力先は `out/build/x64-ReleasePackage/package/`（理由: 既存の preset のビルドフォルダの並び）。
7. 版番号の形は `^[0-9A-Za-z][0-9A-Za-z._-]*$`（理由: ファイル名と GitHub の asset 名で困らない文字に限る。`dev`・`0.4.0`・`0.4.0-rc1` が通る）。
8. CI の MSVC の環境は、1 つのステップで `VsDevCmd.bat` を call した cmd の中で workflow を回して用意する（理由: 第三者の action を増やさない。[AGENTS.md](../../AGENTS.md) と 040 §6.1 の手元の形と同じ）。
9. PDB は `AnetRLRunner.pdb` だけを入れる（理由: anet-core と各 Env は静的ライブラリで、exe の PDB に入る。§2 の 12）。
10. `.sha256` の中身は `sha256sum` と同じ形（小文字の 16 進、空白 2 つ、zip のファイル名）にする（理由: `sha256sum -c` でそのまま確かめられる）。

## 7. 検証と受入

**使う場面**: 次のリリースの前に、手元で `cmake --workflow --preset x64-ReleasePackage` を回して zip を確かめ、CI の workflow_dispatch と突き合わせる。

**受入（外から観測できるもの）**

1. OFF で今と同一: 実装の前と後で、`x64-Debug` と `x64-Release` の configure と build が通り、`apps/runner/bin/<構成>/`・`core/anet-core/bin/<構成>/`・`core/anet-core/lib/<構成>/` に作られる exe・DLL・静的ライブラリのファイル名の一覧が同じである（前後の一覧の差が空）。
2. 手元で作れる: 依存を用意した機械の MSVC の環境のシェルで、作業ツリーに未コミットの変更が無い状態で、`ANET_RELEASE_VERSION=dev` を置いて `cmake --workflow --preset x64-ReleasePackage` を回すと、終了コード 0 で終わる。`out/build/x64-ReleasePackage/package/` に `anet-lab-dev-win64.zip`・`anet-lab-doxygen-dev.zip`・`anet-lab-symbols-dev-win64.zip` と、それぞれの `.sha256` がある。§5.4 の検査がすべて通り、`.sha256` の値が `Get-FileHash -Algorithm SHA256` の値と一致する。
3. 移行の前後で中身が同じ: 移行前の release.yml を、移行の直前のコミットで workflow_dispatch した artifact（作業フォルダの木）と、移行後の release.yml の workflow_dispatch で作った runtime zip について、ファイルの entry のパスの一覧（フォルダの entry を除いて並べ替えたもの）が一致する。ビルドの出力（exe・DLL・jar・THIRD-PARTY.txt）と、本書の変更で直したファイル（`docs/design/040_*`・`apps/README.md`）を除くファイルは、SHA256 も一致する。Doxygen の zip は `LICENSE`・`NOTICE`・`html/index.html` があることだけを合否にし、`html/` の下の一覧の差は Doxygen の版の差として記録する（§9 の 7）。
4. 手元と CI が同じ: 同じコミットで未コミットの変更が無いとき、手元（受入 2）の runtime zip と、CI の workflow_dispatch の runtime zip の、ファイルの entry のパスの一覧が一致する。ビルドの出力を除くファイルは SHA256 も一致する。前提として、手元の LibTorch と wxWidgets の版は CI と同じにする（040 §3.3・§3.4。2026-10-31 の作者の手元は LibTorch の版が違う。§2 の 9）。
5. ソース木を変えない: 受入 2 の実行の前と後で、ソース木の下（`out/` と `.git/` を除く）の全ファイルのパス・大きさ・更新時刻の一覧が同じである。とくに `apps/runner/bin/Release/AnetRLRunner.exe` と `apps/metrics-viewer/target/metrics-viewer.jar`（あれば）の SHA256 が前後で同じである。
6. 止まる場面（ビルドフォルダを別に切って configure する）:
   - ON で版番号が空: configure が非 0 で終わり、出力に `ANET_RELEASE_VERSION` が出る。
   - ON で版番号が `1.0/x`: 止まり、値と許す形が出る。
   - ON で `ANET_ENABLE_ATARI=ON`（`ALE_ROOT` がある機械）: 止まり、`ANET_ENABLE_ATARI` が出る。
   - ON で PATH から doxygen を外す: 止まり、`doxygen` が出る。
   - `ANET_ENABLE_RELEASE_PACKAGE=maybe`: 止まり、名前と値が出る。
   - OFF で `ANET_RELEASE_VERSION=dev`: configure は通り、WARN が 1 回出る。
7. 検査の単体: §5.4 の検査を単体で回し、手で作った次の小さな zip を与えると、それぞれ非 0 で終わり、外れた検査の番号と entry が出る。`LICENSE` の無い runtime、`apps/Atari_run.bat` を含む runtime、`include/box2d/b2_body.h` を含む runtime、手で書いた DLL の名前の一覧より DLL が 1 つ少ない runtime、上限を 1 にして与えた runtime、`html/index.html` の無い doxygen、PDB の無い symbols、値の違う `.sha256`。すべてを満たす組を与えると 0 で終わる。
8. PDB が使える: 受入 2 の runtime zip と symbols zip を同じ空のフォルダへ展開すると、`apps/runner/bin/Release/` に `AnetRLRunner.exe` と `AnetRLRunner.pdb` が並ぶ。`python apps/runner/tools/symbolize.py <展開した exe> <image base> <entry point の RVA>`（値は `dumpbin /headers` で読む）が関数名を返す。
9. CI: 移行後の release.yml を workflow_dispatch で回すと成功し、3 本の zip（または §9 の 4 で決めた形）と `.sha256` が artifact に残る。release.yml に組み立ての PowerShell が無く（`tar.exe` と `Copy-Item` を含む行が 0）、`cmake --workflow --preset x64-ReleasePackage` を呼ぶ行がある。実行の実時間を記録する（合否にしない）。tag push の経路は、次のリリースで人間が確かめる。
10. 文書: 040 の日本語版と英語版に、リリース物を作る手順（MSVC の環境、環境変数、コマンド、出力先、要る空き）がある。AGENTS.md の preset の一覧に `x64-ReleasePackage` がある。apps/README.md の zip の名前が、リリースで作る 3 本の名前と一致する。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由・再開条件 |
|---|---|---|
| 依存の準備（LibTorch の取得など）を CMake で行う | 範囲外 | ユーザーが「環境整備以外」と範囲を決めた。CUDA Toolkit の導入は CMake で行えない。再開条件: 環境整備も 1 つのコマンドにしたくなったとき |
| configure の引数だけを preset へ寄せる案（package とワークフロー preset は使わない） | 不採用 | 手元でリリース物を作れない（G1 を満たさない） |
| `apps/runner/bin/Release` やフォルダを丸ごと包む形 | 不採用 | 手元の作業ツリーでは、試行用の exe、管理外の DLL、Git が無視する `runs/` が混ざる（§2 の 9） |
| exe の import の表から DLL を自動でたどる形 | 不採用 | 実行時に読み込む DLL（`third_party/runtime_dlls` の SDL2 など）を拾えない（§2 の 6） |
| インストーラ（NSIS・WiX・Inno Setup） | 先送り | 2GiB の上限は形式で変わらず、今は要求が無い。再開条件: 展開ではなくインストールで配る需要が出たとき |
| 7Z など圧縮率の高い形式、LibTorch の DLL を別の zip に分ける形 | 先送り | 今の zip は上限の内（約 1,798MB。[PRD043](done/043_release_packaging_10prd.md)）。再開条件: runtime zip が §5.4 の 6 の上限に近づいたとき（component を分ければ作れる） |
| Linux・macOS のリリース物 | 範囲外 | 先にビルドの移植が要る（§2 の 14）。再開条件: Linux でビルドが通ったとき |
| windows-ci.yml の configure を preset へ寄せる | 範囲外 | CI のビルド確認の関心で、[999 CI/CD 改善](999_cicd_improvement_10prd.md) の L1 で扱う。再開条件: 本書の preset が入ったあと |
| リリースの工程で test を回す | 範囲外 | 今の release.yml も回していない。テストの自動化は [999 CI/CD 改善](999_cicd_improvement_10prd.md) の L1 の関心。再開条件: L1 で CI のテストが回るようになり、リリースの前にも同じテストを通したくなったとき（workflow preset に test の工程を足せば作れる） |
| GitHub Release への添付を CMake から行う | 不採用 | 本当の制約ではない（CMake のスクリプトから `gh` を呼べる）が、GitHub に依存する工程で、G1 の外に置く |
| Runner に版番号を埋め込む | 範囲外 | 本当の制約ではない（`ANET_RELEASE_VERSION` から作れる）が、exe の機能で別の関心。再開条件: exe から版を知りたい需要が出たとき |

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | 作業ツリーに未コミットの変更があるときの扱い（そのまま包む／止める） | 管理下のファイルを作業ツリーの内容のまま包み、包むファイルに未コミットの変更があれば、その一覧を WARN で 1 回出す。正式なリリースは CI（まっさらな checkout）で作る |
| 2 | 管理下のファイルに絞る方法（`git ls-files` で並べる／除外のパターンを書く） | `git ls-files`。git が無い（ソースの tarball など）ときは止める |
| 3 | 名前（preset・cache 変数・component・symbols の zip・jar のターゲット・出力先） | §6 の 1〜6。component の `runtime` は、[CONTEXT.md](../../CONTEXT.md) の _Avoid_ の注記で「実行中の状態・操作の語」とされているので、グリルで見直す（候補は `app`。2026-10-31 のレビューの再確認ポイントから） |
| 4 | workflow_dispatch の artifact の形。今は作業フォルダの木を上げて、取り出したときに zip の中に zip が入らないようにしている（コミット 56b63076） | zip と `.sha256` をそのまま上げる（取り出すと zip の中に zip が入る）。検査した zip そのものを残すため。木を上げる形を保つなら、dispatch のときだけ runtime の component を install して木を作り、それを上げる |
| 5 | 出力先の切り替えを ON に結び付けるか（D2・D3）、別の cache 変数にするか | ON に結び付ける。別にすると、リリースの構成でソース木へ書く組み合わせを作れる |
| 6 | CI の CMake が 3.25 以上か、`VsDevCmd.bat` の環境に Ninja があるか（移行後の最初の workflow_dispatch で確かめる） | あると置く。CMake が古ければ action で入れる。Ninja が無ければ、CI だけ Visual Studio の generator の configure preset を分ける |
| 7 | Doxygen の版（choco は最新を入れるので、手元と CI で html がずれうる） | 固定しない（今と同じ）。受入 3・4 は html の中身を比べない |
| 8 | ADR を起こすこと（D13） | グリルで D3〜D6 が決まったら、新しい ADR「リリース物の中身は CMake の install のルールで定め、CPack で包む」を起こし、ADR 0017 と ADR 0020 の先頭に注記を足す（戻しにくく、文脈が無いと意外で、別案との比較の結果なので、ADR の 3 条件に当たる見込み） |
| 9 | symbols の zip を毎回 Release に添付するか | 添付する（約 78MiB。§2 の 12） |
| 10 | Maven の出力先をビルド木にする方法（pom の property／ソースをビルド木へ写して作る） | pom に出力先の property を足し、既定は今の `target` |
| 11 | G5（PDB の zip と SHA256。D10・D11）を本書に入れるか、別の PRD にするか。会話で示されたゴール（GitHub に依存せず、CMake に統合して手元でも作る）の外から、CPack を入れる利点として足したもの | 本書に入れる（同じ component と CPack の設定に乗る小さな追加で、別にすると install のルールを 2 度触る） |

## 10. 依存と順序

- 先に要る PRD は無い（[PRD043](done/043_release_packaging_10prd.md) の実装の上に乗る）。
- [999 CI/CD 改善](999_cicd_improvement_10prd.md) の L1（windows-ci.yml の改修）とは独立している。L1 で preset を使うなら、本書の preset の書き方（schema 6）に合わせる。
- 次のリリース（v0.4.0）の前に入れると、tag push の経路をそのリリースで確かめられる。

## 11. 参照

- [PRD043 リリースパッケージング](done/043_release_packaging_10prd.md)、[999 CI/CD 改善](999_cicd_improvement_10prd.md)
- [ADR 0017](../adr/0017-release-zip-mirrors-repo-layout.md)、[ADR 0020](../adr/0020-release-archives-carry-project-license.md)、[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md)
- [040 開発環境構築ガイド](../design/040_development_environment.jp.md) §3・§4.2・§6
- [311 開発ツール（wiki）](../wiki/categories/311_dev_tools.md)
