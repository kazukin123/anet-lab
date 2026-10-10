# PRD084 レビュー修正の検証記録

承認済み計画は docs/memo/084_noisynet_21impl.md。修正前の測定は verification.md に当時の条件のまま保持する。今回の出力は .scratch/prd084/review-fix/ に分離する。

## 基準と実装

- 編集前の Runner、依存 DLL 50本、設定、seed、checkpoint、差分を before/manifest.json と before/ に保存。
- 既存 Noisy テスト40件・10414 assertions を GREEN と確認してから共通化。
- Body と6 Head の具象設定、検証、保持、抽選を LinearProjectionModule の private 実装へ集約。μ・σ の登録名と初期化順、εの slot は native Linear の実体で維持。
- 内容別の不変設定キャッシュは module 内で同期し、Tensor 演算前にロックを解放する。
- NetworkForwardRowMapping は CPU 宣言を一度検証し使用 device へ転送。射影は SelectRows で再利用し、検査用 GetIndices は複製を返す。
- 参照 profile の AMP と call/batch を継承。結合 Run の Actor だけ episode/sample にする。
- NN 内の RL 用語は今回変更しない。後続で episode_start、時計の区分、保持値 episode を検討する。

## RED→GREEN

| 挙動 | RED | GREEN |
|---|---|---|
| CPU 行対応の所有 | row-mapping-red-build.log の API 未定義 | row-mapping-green.log: 1件2 assertions |
| index・型・rank・元行数の検証 | row-range-red.log: 8 assertions 失敗 | row-range-green.log: 2件10 assertions |
| 同名・異内容、設定寿命 | projection-cache-red-build.log の部品 API 未定義 | projection-green.log: 43件10427 assertions |
| 検査用 Tensor による不変性 | mapping-immutable-red.log: 1 assertion 失敗 | mapping-immutable-green.log: 44件10428 assertions |

LNK1163 は生成済み Head オブジェクトを一つ再生成して解消した。新規 CUDA テストの capture 引数のコンパイル誤りは現行 API に合わせて修正した。

## 追加受け入れ

- Debug 全体ビルド: debug-final-build.log / exit0。
- Noisy 関連: noisynet-final-green.log / 47件10497 assertions / exit0。
- 関連 NN・DQN・Head・Munchausen: related-final.log / 304件27178 assertions / exit0。
- CUDA autocast の Body force_fp32 ON/OFF、Head 出力と SN 推定の FP32 を確認。
- rank2/3・CPU/CUDA の結合行で参照出力が一致。保持中 forward の aten::_to_copy と aten::_local_scalar_dense は各0件。
- 同時4呼び出しの専用箱による再現と分離、同名・異内容と設定寿命、独自 Module/Head の契約を確認。
- 指定11ファイルの UTF-8 BOM と LF を bom-audit.json で確認。

- 全 core: all-core-final.log / 665件中664件成功・1件skip / 34680 assertions / exit0。skipはCUDA利用可能環境で非該当となるPRD070ケース。
- Release Runner: release-runner-build.log / exit0。SHA-256 F7EB69FD918116E3EAAC93BF296F6CD7E6FF00C24C13F03A0881059AD48BF8B4。

OFF3本、結合2本と旧checkpointの受け入れは完了。性能12比較48 Runも完了し、今回の全受け入れ条件を満たした。OFF初回は余分なCLIで既存の未使用終了予算キーまで上書きして設定照合が失敗したため中止し、after/ に記録を保持する。不要なCLIを外した再実行は after-verified/ に分離する。Atari既定の起動時照合では全既存値が一致している。

既定葉監査: Assignments 24877 / audited 3423 / errors 0。

## OFF 回帰

after-verified/ の3本は各500k経験・seed1・deterministic backendで正常終了。数値・点数・旧設定の全既存値が一致した。性能・資源・追加 Noisy の指標を従来どおり除外する。

| 構成 | 数値点 | タグ数 | SHA-256 |
|---|---:|---:|---|
| Atari既定 | 78986 | 114 | 9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29 |
| SN付き効率優先 | 122332 | 114 | 5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e |
| CPU LunarLander | 88184 | 86 | b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5 |

CPU初回はAtari専用greedy_dist指定による未知タグで起動前に停止した（exit255）。失敗Runとexecution/stdout/stderrのfailed-01ファイルを保持し、成功したAtari2本を再実行せずCPUだけ再開した。CPUへのAtari専用CLIは削除し、未完了Runへの再開も拒否する。

## Atari 結合と旧 checkpoint

after-verified/acceptance-summary.json の一括判定は6caseすべてpass。2本ともBreakout・seed1・128 env・500k経験でexit0。

- nz_learning: AMP継承・call/batch。eval2回、probe10回。readout weight/bias mean-absとweight ratioは10回、bias ratioは初回の正当なNaNを除き9回の成立値。
- nz_reference: Learner BF16 AMPを継承、Body2FCだけforce_fp32=true、Actorだけepisode/sample。Learner current/targetはcall/batch。eval2回、readout4統計・probe3指標は各10回の成立値。
- 全11キーを購読し、feature4統計は正常なNaNで成立値が出ない。feature成立値の契約テストもGREEN。
- 旧CPU OFF checkpointのSHAは43674EF8A169FA7C16A388FC4F2532C9D4F4537F3C5B1AF397606994B22D35FE。LoadNetworkログでmodel_size=1276270、learner_size=1192239を読み込み、256経験のsmokeはexit0。

## 性能12比較・48 Run

performance/summary.json と summary.md の集計はexit0。全48 RunはBreakout・seed1・128 env・224k経験で正常終了し、warmup 200kと学習スケジュールを維持した。各比較のA/B/A/B、実行の非重複、同一実行体、比較軸以外の全解決済み設定の一致を確認した。診断ONはprobe3指標を実際に出力し、OFFでは出力が無い。

下表は各側2回の平均。時間は起動から終了まで、学習中経験/sはwarmupをまたぐ区間を除いたメトリクス平均。CPU追加量はプロセスのpeak private bytesのB-A、GPU追加量はデバイス全体の5秒間隔peak-minus-startのB-Aであり、allocatorやNoisy単体への帰属は行わない。速度・スコアの改善は合否条件にしない。

A/Bの対応: offはNoisy ON/OFF、actor_shareとlearner_shareはbatch/sample、hold_count32はcall/count32、hold_episodeはcall/episode（保持比較は両側sample）、sn_spectralとsn_capはSNなし/各SN、precisionは同じAMPでBody force_fp32 true/false、diagnosticsはON/OFF、learning_sn_*は各SN条件でNoisy OFF/ON。

| 比較 | A 秒 | B 秒 | A 経験/s | B 経験/s | A 学習中経験/s | B 学習中経験/s | CPU追加 MiB B-A | GPU追加 MiB B-A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| actor_share | 79.65 | 80.60 | 2813.4 | 2779.3 | 1981.7 | 1929.2 | 2.59 | 4.50 |
| diagnostics | 79.50 | 77.16 | 2818.7 | 2903.0 | 1983.7 | 2069.0 | 9.79 | 0.50 |
| hold_count32 | 79.86 | 80.46 | 2805.1 | 2788.0 | 1937.3 | 1876.6 | 0.60 | 126.50 |
| hold_episode | 82.62 | 138.86 | 2711.6 | 1983.7 | 1919.0 | 1210.3 | -34.10 | 116.00 |
| learner_share | 80.26 | 81.39 | 2791.3 | 2752.7 | 1918.5 | 1862.3 | 35.98 | 592.50 |
| learning_sn_none | 76.74 | 78.84 | 2918.9 | 2841.3 | 2052.2 | 1968.9 | 66.79 | 53.00 |
| learning_sn_spectral | 78.47 | 82.22 | 2854.5 | 2724.7 | 2068.1 | 1876.3 | 61.42 | 39.50 |
| learning_sn_spectral_cap | 82.91 | 87.40 | 2705.6 | 2563.1 | 1871.3 | 1775.6 | 39.74 | -74.50 |
| off | 85.99 | 79.43 | 2630.1 | 2828.9 | 1505.4 | 2159.1 | -34.96 | -41.50 |
| precision | 78.60 | 78.44 | 2850.1 | 2855.7 | 2067.3 | 2011.0 | 3.54 | 69.50 |
| sn_cap | 78.22 | 78.62 | 2863.9 | 2849.3 | 2034.0 | 1906.1 | 7.38 | 5.50 |
| sn_spectral | 79.91 | 79.34 | 2803.6 | 2823.7 | 1996.0 | 1947.3 | 21.04 | -16.00 |

## 最終照合

PRD・20impl・21impl・README・本記録を再読し、保存済みSHA-256とmtimeを確認。PRDの完了状態行は無し。指定11ファイルのBOM/LF、Python/PowerShell構文、git diff --checkは合格。48性能Runの学習中throughputは各4点以上。同一Runner SHAと、最終Releaseビルド後のproduction source未変更を確認。設定・状態の具体型はLinearProjectionModule::Implの一か所だけにあり、共通のNoisy設定型・旧Tensor行対応APIは追加していない。

## 時計の種類と残件の追加修正（22impl）

承認済み計画は docs/memo/084_noisynet_22impl.md。時計はkNoneに加えてkTransientRows/kContinuingRowsの2種類へ統合した。episode_startとepisode保持は維持する。任意のB3は今回は入れない。

上記48本の性能結果は改名前の実行体（F7EB69FD…）での測定。offとhold_episodeのみ新実行体で3反復A/B交互に取り直した。各Run値と中央値は後述の追加修正結果を参照。他10組、旧Runとsummary.jsonは当時の測定として保持する。同じ参照構成・Noisyありの14本でも学習中exp/sは1,005〜2,081だったため、数%の差はこの測定では読めない。

### 追加修正の受け入れ結果

- Debug全体と最終Release Runnerのビルドはexit0。最終実行体SHA-256は 3089EA77E5B79EF027365C8CDB6497BF0C55A2AA10825D818336B083F1612F52。
- 改名前/後のNoisyテストは双方47件・10497 assertionsでGREEN。全coreは665件中664成功・1skip、34680 assertions、exit0。skipはCUDA利用可能環境で非該当のPRD070ケース。
- 定数・エラー文言・enum定義/コメント・局所rows名以外のproduction差分は無し。NN指定6ファイルのActor|Learner|Probe|laneは0件。テストの挙動・assertion・fixtureは変更していない。
- dqn_based_heads.cpp/nn_impl.hppのBOMを復元。固定11本ではなく今回編集した全trackedファイルをHEADのBOM/LFへ照合し、HEAD未登録ファイルは編集前の形式を保持した。
- OFF3本は最終Release・seed1・各500kで正常終了。checksum・点数・全既存設定文字列が旧基準と一致した。

| 構成 | 数値点 | checksum SHA-256 |
|---|---:|---|
| atari_default | 78986 | 9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29 |
| atari_efficiency | 122332 | 5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e |
| lunarlander_cpu | 88184 | b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5 |

- 性能はoff/hold_episodeだけ各3反復A/B、合計12 Run、224k経験、200k warmup。全exit0、比較軸以外の全設定一致、同一実行体、交互順/非重複を確認。
- GPUはRTX 5090 / GPU-0c44f747-7f1b-2d07-cd6d-a9246d000f89。各Runの前後で別学習プロセス0を記録。測定保留は無し。GPUメモリは5秒間隔のデバイス全体peak-minus-startであり、処理別の帰属ではない。
- offはA=Noisy ON、B=OFF。hold_episodeはA=call、B=episode、両側sample共有。以下の時間は起動から終了、学習中経験/sはwarmupをまたぐ区間を除く値。各中央値は列ごとに算出する。
- 今回もoffのNoisy ONは学習中1494.8〜3675.9経験/sと幅がある。旧参照構成14本の1005〜2081経験/sというばらつきと併せ、数%の差や改名による速度変化をこの測定から断定しない。

## Run ごとの値

| 比較 | 反復 | 側 | 秒 | 学習中経験/s | CPU peak MiB | GPU追加 MiB |
|---|---:|---|---:|---:|---:|---:|
| off | 1 | a | 73.279 | 3675.9 | 23189.77 | 5881.00 |
| off | 1 | b | 71.000 | 2544.2 | 23157.06 | 5668.00 |
| off | 2 | a | 75.239 | 1494.8 | 23218.39 | 5707.00 |
| off | 2 | b | 63.672 | 2406.8 | 23167.88 | 5578.00 |
| off | 3 | a | 59.240 | 3617.7 | 23224.84 | 5726.00 |
| off | 3 | b | 57.086 | 3818.9 | 23135.75 | 5451.00 |
| hold_episode | 1 | a | 61.263 | 3629.1 | 23234.97 | 5637.00 |
| hold_episode | 1 | b | 58.438 | 3734.4 | 23168.38 | 5635.00 |
| hold_episode | 2 | a | 60.328 | 3492.8 | 23172.52 | 5499.00 |
| hold_episode | 2 | b | 59.492 | 3519.9 | 23163.08 | 5635.00 |
| hold_episode | 3 | a | 59.049 | 3632.6 | 23170.12 | 5637.00 |
| hold_episode | 3 | b | 59.266 | 3524.1 | 23172.19 | 5673.00 |

## 3反復の中央値

| 比較 | 側 | 秒 | 学習中経験/s | CPU peak MiB | GPU追加 MiB |
|---|---|---:|---:|---:|---:|
| hold_episode | a | 60.328 | 3629.1 | 23172.52 | 5637.00 |
| hold_episode | b | 59.266 | 3524.1 | 23168.38 | 5635.00 |
| off | a | 73.279 | 3617.7 | 23218.39 | 5726.00 |
| off | b | 63.672 | 2544.2 | 23157.06 | 5578.00 |

### 保留・対象外・失敗記録

保留は無し。B3は小さな最適化であり、今回の挙動不変の改名検証に混ぜない判断で未実施。性能の他10組は指定どおり取り直さず、改名前の実行体での測定として上の旧記録/Run/summary.jsonを保持した。episode/episode_startの改名、新語導入、LinearProjectionModuleの構造変更、CONTEXT.md/ADR0048、今回対象外の結合2本と旧checkpoint再読込には手を付けていない。git add/commit/pushは未実施。

最初の検証スクリプトはPowerShell5が通常stderrログをエラー扱いして停止したため、終了コード判定へ修正。次のNoisyテストでは2か所の部分一致文字列にActor clockが残り2case/13assertionsが失敗し、指定のcontinuing rowsへの置換後に全47件がGREENになった。failed-01/failed-02ログは保持し、成功結果と区別する。

詳細artifact: .scratch/prd084/terms-residuals/ のstages.jsonl、regression/acceptance-summary.json、performance/summary.json/mdと各execution.json（GPU前後記録）、final-audit.json。


### 今回変更したファイル

- `core/anet-core/include/anet/nn.hpp`
- `core/anet-core/src/nn_modules.cpp`
- `core/anet-core/src/dqn_based_agent.cpp`
- `core/anet-core/src/nn_test.cpp`
- `core/anet-core/src/dqn_based_test.cpp`
- `core/anet-core/src/dqn_based_agent_test.cpp`
- `core/anet-core/src/dqn_based_heads.cpp`
- `core/anet-core/src/nn_impl.hpp`
- `docs/memo/084_noisynet_10prd.md`
- `docs/design/130_neural_networks.jp.md`
- `docs/design/130_neural_networks.en.md`
- `core/anet-core/testdata/prd084/README.md`
- `core/anet-core/testdata/prd084/review_verification.md`
- `core/anet-core/testdata/prd084/run_performance.ps1`
- `core/anet-core/testdata/prd084/summarize_performance.py`
- `docs/memo/084_noisynet_22impl.md`

検証用に新規作成したローカルスクリプト: `.scratch/prd084/terms-residuals/validate.ps1`, `.scratch/prd084/terms-residuals/run_off.ps1`, `.scratch/prd084/terms-residuals/summarize_off.py`。実行体・DLL・設定コピー・Run・ログ・JSON監査は同じ出力先に保持する。

## Context の追加レビューと最終実行体での OFF 回帰（2026-10-05）

- 行対応の変更は WithRowMapping() で複製。const Params を一括コピーし、行対応以外と将来追加する項目を保持する。TD・分位点の target 構築2か所の手写しを削除。
- コンストラクタの時計種別チェックとその専用テストを削除。保持・時計など利用条件の検証は計算部品に維持。
- PRD 冒頭のクラス形の説明を削除し、設計文書130の日英へ集約。
- Debug 全体ビルド成功。NoisyNet 49件・10,521 assertions 成功。全 core は666件成功・1件スキップ、34,704 assertions 成功。スキップは CUDA が利用可能な場合に対象外となる既存テスト。
- 最終 Release ビルド後にコードを変えず、保存した同一実行体・DLL 50本で OFF 回帰3構成を各500k経験・seed=1で実行。deterministic backend と greedy_dist 無効化を旧基準と揃えた。
- 3本とも終了コード0、close checkpoint を生成。旧基準と数値メトリクスの checksum・点数・既存の解決済み設定値が一致。
- checksum は (tag, step, value) を正規化。従来どおり時刻・性能・資源計測・追加 Noisy 指標を除外。欠損・点数差なし。
- Release SHA-256: DC7218608A8470F1E1139A554271A89D921E83E5D16BA9A2527A4B0AEED8E9EC。
- 出力と実行体は .scratch/prd084/context-review/regression/、ビルド・テスト・実行ログは .scratch/prd084/context-review/ に保存。前回の Context 不変化直後に学習回帰を省略した記録は当時の記録として保持し、今回の最終実行体での3本を現行の回帰根拠とする。
- 性能の再測定は今回の依頼範囲に含めず、過去の測定 artifact は保持。git add / commit / push は実行していない。

| 構成 | 数値メトリクスの点数 | 正規化 checksum（旧基準と一致） |
|---|---:|---|
| atari_default | 78986 | 9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29 |
| atari_efficiency | 122332 | 5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e |
| lunarlander_cpu | 88184 | b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5 |

変更ファイル:

- docs/design/130_neural_networks.en.md
- docs/design/130_neural_networks.jp.md
- docs/memo/084_noisynet_10prd.md
- core/anet-core/src/dqn_based_agent.cpp
- core/anet-core/src/nn_impl.cpp
- core/anet-core/src/nn_test.cpp
- core/anet-core/include/anet/nn.hpp
- core/anet-core/testdata/prd084/review_verification.md


## 再レビュー指摘の修正と検証（2026-10-05）

前回の指摘5点に対応した。PRD冒頭に改訂日と概要を記載し、計画20・21・22を「概要」「主な変更」「テスト」「検証」「前提」の5節へ整理した。agent.txtの15行とmetrics_scalar.txtの11行は、今回追加したプロファイル境界だけを : に変更した。

optimizerの3テストは、実際のTDLearnerの公開UpdateFromBatchを使う形へ変更した。通常/Fused AdamWでσの独立decay・μ-only凍結、σを含む全体normとclip、NetworkModel/Learnerの公開Save/Loadによるgroup・step・moment復元を確認する。新規追加したTestLearner::SetupOptimizerの公開化は削除した。

実行・集計の予算契約はOFF3本が500k、旧checkpointが256、結合2本が500k以上へ統一した。ExperienceBudgetの延長は結合2本だけへ渡す。集計スクリプトをmainへ分離した既存契約のGREEN、600k結合予算のRED、受け入れ条件修正後のGREENを記録した。予算・終了code・seedの4つのテストが成功し、PowerShellの構文と6caseの予算式を学習Runを起動せず確認した。

| 検証 | 結果 |
|---|---|
| MSVC初期化経由 Debug 全体ビルド | exit 0 |
| NoisyNet | 49件・10,690 assertions 成功 |
| 全 core | 666件成功・既存1件スキップ、34,873 assertions 成功 |
| Python予算契約 | 4件成功、延長受け入れのRED→GREENを保存 |
| 設定既定葉監査 | 24,939 assignments、3,417 audited、errors 0 |
| 公開ConfigManagerによる6構成の前後比較 | 全キー・文字列値・型付き設定が一致、44 assertions 成功 |
| 集計CLIによる既存6caseの再判定 | OFF3・結合2・旧checkpointすべて成功 |
| 形式・責任境界の監査 | 編集前/HEADのBOM・LFを維持、指定NN6ファイルのActor/Learner/Probe/laneは0件 |

ビルド・テスト・設定のコマンドは `.scratch/prd084/review-followup-20261005/validate.ps1`、各終了codeはstages.jsonl、ログはdebug-build.log、noisynet.log、all-core.log、budget-final.log、default-leaves.log、config-final.logに保存した。予算テストのRED/GREENはbudget-red.logとbudget-green.log。集計CLIは同じ場所のsummary-e2e.logとsummary-e2e/acceptance-summary.jsonへ出力した。summary-e2eは既存の実行体・DLL・実行記録を複製し、元のRunを読み取るだけとした。過去のartifactは上書きしていない。

本体のソース4本のSHAはContext最終検証時と同一。現在のRelease SHA-256も DC7218608A8470F1E1139A554271A89D921E83E5D16BA9A2527A4B0AEED8E9EC のままである。今回の変更はテスト、設定の区切り、検証補助、文書に限定し、全解決値一致を確認したため、学習・性能Runは追加していない。現行Releaseの回帰根拠は上のContext最終検証のOFF3本を維持する。過去の全体FP32/改名前の測定と今回の検証は区別する。

途中の失敗記録: 古いPRD061採取manifestにあるrun.@nature_dqnは現行設定に存在せず採取が停止したため、manifestへ手を入れず、今回の6構成を公開ConfigManagerで直接比較した。移行中のoptimizer fixtureではReplayBufferの最小容量と先頭状態の投入、Save/Loadの返却サイズの前提を修正した。設定比較テストのwxCmdLineParserに必要なincludeを追加し、最終ビルドと全テストを成功させた。途中の失敗ログは成功結果と分けて保持する。

保留は無し。学習/性能の再実行、B3、episode/episode_startの改名、対象外の設計変更、git add/commit/pushは行っていない。今回の監査結果はaudit.jsonに保存した。

今回の変更ファイル（12本）:

- docs/memo/084_noisynet_10prd.md
- docs/memo/084_noisynet_20impl.md
- docs/memo/084_noisynet_21impl.md
- docs/memo/084_noisynet_22impl.md
- apps/runner/config/agent.txt
- apps/runner/config/metrics_scalar.txt
- core/anet-core/src/dqn_based_agent_test.cpp
- core/anet-core/testdata/prd084/run_review_validation.ps1
- core/anet-core/testdata/prd084/summarize_review_validation.py
- core/anet-core/testdata/prd084/summarize_review_validation_test.py（新規）
- core/anet-core/testdata/prd084/README.md
- core/anet-core/testdata/prd084/review_verification.md

## Forward 事前準備の最終検証（2026-10-10）

[事前準備リファクタリングの検証記録](forward_refactor_verification.md) に、最終ビルド・テスト・OFF3構成の checksum、代表 ON 6 Run の値・中央値・成立条件と変更ファイルを保存した。過去の受け入れ・測定は当時の契約の記録として保持する。
