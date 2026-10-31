# GridMaze の迷路seedを指定値またはエピソード・プロセス単位で選ぶ PRD

> 2026-10-24 起票（Codex、anet-create-prd）。初回の3モード案はレビューOK。その後、正式番号102へ採番して実装を開始した。
> 2026-10-24 ユーザーの仕様調整を反映。seedだけで固定できるようにし、seed_modeからfixedを除外する。seedは未指定を区別して保持し、0と最大値も通常の指定値として扱う。内部の未指定はNaNではなく整数の精度を保つoptionalで表す（内部型の詳細は実装計画）。以前の「自動モードではseedを無視する」という決定は撤回した。

## 1. 問題・ゴール・非ゴール

起動ごとに一つの迷路を選び、学習・評価・GUIで使い続けたい。一方、盤面の再現にはseedだけを指定すれば足りるようにし、seed_modeもfixedへ変える二重指定を不要にする。

- G1: seedが指定されていれば、その値で固定する。未指定ならseed_modeでprocess・episodeを選び、既定はprocessにする。
- G2: processの自動入力seedはプロセス内の全GridMaze Envで一つを共有し、同じ迷路設定なら全個体・全エピソードで同じ盤面にする。
- G3: PRD100の到達可能性保証・解けるseedの盤面と乱数消費順・採用seedからの再現を維持する。

非ゴールは他Envのモード統一、Run master seedの変更、迷路の難易度制御、生成アルゴリズムの置換。技術的に変更不能な制約ではなく、今回の採番と指定方法とは別の関心である（§8）。

## 2. 現行の事実と既存部品

- [GridMazeEnvConfig](../../core/envs/gridmaze1/src/GridMazeEnv.hpp)と[Reset](../../core/envs/gridmaze1/src/GridMazeEnv.cpp)の変更前は、seedの既定0とrandomize_seedで固定・毎Resetの自動採番を選んでいた。コンストラクタもResetする。
- seedは[seed_t](../../core/anet-core/include/anet/random.hpp)で表す符号なし64bit整数。NaNは表せず、浮動小数点へ変えると大きな値の精度を失う。未指定を別に表し、値域を削らない。
- [ConfigDataの数値変換](../../core/anet-core/src/config.cpp)は負数・範囲外・非整数を拒否する。この共通契約を使う。[Configの読み取り](../../core/anet-core/include/anet/config.hpp)には未指定を保持する共通処理を足し、GridMazeに重複する数値パーサーを作らない。
- [SeedMaker::MakeAutoSeed](../../core/anet-core/src/random.cpp)は既存の自動採番入口で、run.seedからの派生とは別。[実行基盤と設定](../design/100_runtime_and_configuration.jp.md)のRun seed契約を変更しない。
- [PRD100](100_gridmaze_solvable_maze_10prd.md) §4・§5と[環境設計](../design/120_environments.jp.md) §7.4の到達可能性保証・決定的な最大1000回の引き直し・採用seedの公開を維持する。PRD100 D8の通知抑制だけの共有から、今回は入力seedの共有だけを追加する。盤面と乱数器は個体所有のまま。
- [GridMaze.txt](../../apps/runner/config/GridMaze.txt)にはユーザーの有効な実験用seed指定がある。これは保持し、新契約では固定として働く。[GridMaze_muzero.txt](../../apps/runner/config/GridMaze_muzero.txt)のseed既定行は外す。共通のseed既定値が残ると「未指定」がなくなり、process既定が働かないため。
- [DropMerge](../../core/envs/dropmerge1/src/DropMergeEnv.cpp)のseed_modeはそのEnv固有の契約であり、変更しない。

参照した地図は[322 Env個別実装](../wiki/categories/322_env_individual.md)、[320 Env共通基盤](../wiki/categories/320_env_core.md)、[312 ユーティリティ](../wiki/categories/312_util.md)。PRD100由来の盤面golden・採用seed復元・固定条件の短Run比較を継続する。

## 3. 機構の寄与と成功の測り方

seedの未指定を保持することで、指定があるときだけ固定するG1を満たす。process共有はG2、共通INFOと採用seedの公開はGUIなしの取得とG3の再現を支える。

成功は§7の受入がすべて通ること。標準の未指定動作が変わるのは意図した変更であり、全scalar一致は同じseedを指定した固定条件で要求する。学習成績向上や性能改善は目的にしない。

## 4. 確定事項

| # | 決定 | 理由・前提 |
|---|---|---|
| D1 | seed指定があれば固定し、未指定時だけseed_modeを使う | G1。再現時にseedだけ指定できるようにするというユーザーの選択 |
| D2 | seed_modeはprocess・episodeだけで、既定process | D1によりfixedは重複するため除く。seed未指定の標準用途は起動単位の迷路選択 |
| D3 | 未指定は数値の外で表し、0・最大値を含む整数全域を固定値として使える | G3。NaNへの型変更やmagic値で精度・値域を失わない |
| D4 | processの入力seedは最初の利用時に自動採番し、プロセス終了まで全GridMaze Envで共有 | G2。個体・BatchEnv・Runを破棄して作り直しても同じ。別プロセスへ保存・引継ぎしない |
| D5 | 共有するのは入力seedだけで、各Envの盤面・乱数器は共有しない | 同じ設定の再現と、異なる設定間の独立を両立する。PRD100 D8の前提をこの範囲だけ改める限定例外 |
| D6 | 自動採番は既存方式を使い、run.seedから派生させない | 既存の自動seedを維持し、共通乱数基盤と他Envの契約を変えない |
| D7 | 共通入力seedをINFOへ1回記録し、画面・auxには採用seedを出す | G3。GUIなしでも取得でき、Env数に比例するログを避ける |
| D8 | seedと有効なseed_modeの併用はseedを優先して固定し、WARNなし | 共通設定にmodeを置いたままseedだけで再現できる。以前のseed無視とは異なる |
| D9 | randomize_seedと暫定実装のfixedモードを廃止し、現用を同じ変更で移す | [AGENTS.md](../../AGENTS.md)のクリーンブレーク方針。互換別名・変換・旧キー専用警告やtripwireは残さない |

新しいドメイン用語は導入しない。seed・プロセス・エピソードという既存の語で説明できるためCONTEXT.mdに実装詳細を足さない。既存ADRとの矛盾はなく、新規ADRは変更困難という条件を満たさないので起こさない。

## 5. 実装契約

### 5.1 設定と優先関係

| キー | 型・有効値 | 未指定時 | 指定時 |
|---|---|---|---|
| GridMazeEnv.seed | 0〜18446744073709551615の整数 | 未指定を保持し、自動採番へ進む | この値で固定。0も通常値 |
| GridMazeEnv.seed_mode | 文字列process / episode | process | seed未指定時の採番単位 |

```text
# 再現するときはseedだけで固定。
GridMazeEnv : seed = 123
```

```text
# seed指定のない設定で、エピソード単位の自動採番。
GridMazeEnv : seed_mode = episode
```

共有するEnv既定は`GridMazeEnv.seed_mode ?= process`だけを置く。seedの既定行は置かない。modeは小文字の2値だけを受け付ける。seedがあっても、読み取り対象としたmodeの未知値・空文字は検証して失敗させる。`seed_mode=fixed`も通常の未知値として扱い、旧値専用の検出を加えない。

未指定とは実効設定にseedのキーがないこと。明示された空文字・nan・負数・範囲外・非整数を未指定へ読み替えず、既存数値変換と同様にエラーにする。既定値・上書き設定・CLIの解決は現行設定基盤に従い、継承元のseedも実効設定に存在すれば固定として扱う。自動採番へ戻すにはそのseed定義を外す。seedを消す新しい構文・magic値は導入しない。

入力キーがなくDTOの初期値も未指定なら、DTOは未指定のままで、GetConfigData・GetScopedConfigData・JSONのいずれにもその葉を追加しない。nullや空文字として記録しない。入力キーがなくDTOに初期値があれば、その値を保持し、通常値と同じ値・型で設定記録へ出す。明示された空文字・false・空vectorは、それぞれの型で有効な指定値として保持し、未指定へ変換しない。

共通の未指定保持は、ConfigDataが直接読み取る葉値を対象にする。複合ConfigReaderを包む拡張は今回の要求に含めず、誤って未指定扱いにせず型の段階で拒否する。

### 5.2 寿命・共有・失敗

- seed指定時は構築・各Resetで同じ入力seedを使う。指定がないepisodeは、構築時の最初のResetを含めて毎Resetで自動採番する。一意番号の保証は設けない。
- seed指定がないprocessは、最初の該当Env構築時までに共通入力seedを確定する。学習・configured eval・後から作るGUI、別BatchEnv、同一プロセス内の別Runも共有する。全個体破棄後の再構築でも維持する。
- seed指定のあるEnvとepisodeだけを使うプロセスでは、共通seedを初期化せず、そのINFOも出さない。それらを先に使っても後のprocessの共有値を変えない。
- 同時構築でも共通の採番と公開は一度だけ。途中の値を公開しない。初期化失敗は保存して後続へ同じ失敗を返し、黙ってretry・代替しない（AGENTS.mdのFail-Fast原則）。
- 寸法・密度・座標等が異なると共通入力seedでも盤面・引き直し・採用seedは異なり得る。片方のResetが他方の盤面・乱数列を変えてはならない。
- modeは設定読み込み時と、設定オブジェクトからの構築時に同じ設定所有の検証を使う。未知値は盤面生成前にANET_SYSTEM_ERRORで停止し、キー・指定値・process, episodeを診断する。数値検証は共通処理を使う。
- 共有seedは各Envの構築時に取得し、通常のStepとResetへ共有seed用ロックを足さない。汎用のEnv・Runner・seed基盤へGridMaze専用分岐を加えない。

### 5.3 生成・公開値・ログ

全モードでPRD100の到達可能性判定、解けるseedのセル生成と乱数消費順、解けない候補の決定的な引き直し、初回とは別に最大1000回、不正寸法・座標・候補なしの失敗を維持する。観測・報酬・GetScalarの鍵も変えない。

auxのseedと画面のSeedは実際の採用seedを返し、画面は全範囲を符号なし十進表示する。採用seedを設定へ戻すだけで、同じ迷路設定の盤面が再現できる。

processの共通入力seedの確定時にINFO本文`GridMaze process seed initialized: seed=<値>`を1回出す。共通入力seedを設定へ戻しても同じ引き直し規則で再現できる。seed指定時の引き直しINFOは入力seedごとにプロセス1回というPRD100の契約を維持する。自動採番時は個体・エピソードごとの引き直しINFOを追加しない。

### 5.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| Config | 変更 | 未指定を保持する設定の共通読み取りを提供する | Config（316 設定管理） |
| GridMazeEnvConfig | 変更 | seedの指定有無とmodeを読み、設定の値域を検証する | GridMazeEnv（322 Env個別実装） |
| GridMazeEnv | 変更 | 指定seed優先の選択、プロセス共有、盤面生成と採用seed公開を担う | GridMazeEnv（322 Env個別実装） |

共通設定の読み取りは利用側の名前に依存させない。内部の分割・保持方法は実装計画へ記録する。

## 6. 移行対象と合意した細部

- [Config共通読み取り](../../core/anet-core/include/anet/config.hpp)と[設定テスト](../../core/anet-core/src/config_test.cpp): 未指定・0・最大値・prefixの上書き・不正値を検証する。
- [GridMazeEnv.hpp](../../core/envs/gridmaze1/src/GridMazeEnv.hpp)・[cpp](../../core/envs/gridmaze1/src/GridMazeEnv.cpp)・[テスト](../../core/envs/gridmaze1/src/GridMazeEnv_test.cpp): 旧boolを削除し、seed指定有無と2モードへ移行する。
- [GridMaze.txt](../../apps/runner/config/GridMaze.txt)・[GridMaze_muzero.txt](../../apps/runner/config/GridMaze_muzero.txt): mode既定をprocessへ移し、共有seed既定を外す。ユーザーの明示的な実験用seedは保持して固定として扱う。
- [PRD100 driver](../../core/anet-core/testdata/prd100/run_validation.py)・[README](../../core/anet-core/testdata/prd100/README.md): 固定条件はseedだけを指定する現行手順へ移す。
- [実行基盤・設定設計日本語](../design/100_runtime_and_configuration.jp.md)・[英語](../design/100_runtime_and_configuration.en.md): 共通optionalの初期値・省略記録・型変換・対応範囲を更新する。
- [環境設計日本語](../design/120_environments.jp.md)・[英語](../design/120_environments.en.md): 2モードとseed優先、共有寿命、再現例、領域テストへの参照を更新する。
- 現用の追加検索結果も同じ変更で移す。PRD固有の検証手順は正式番号のtestdata/prd102へ置く。

過去Run・ADR・実験記録・履歴PRD/実装メモ・PRD072保存baselineを改変せず、互換読替えを足さない。旧識別子は現行契約に含めず、基盤が一律検出しなければNoCareとする。

合意した細部は、共通INFOの英語文言、seedとmode併用時のWARNなし、未知modeのキー・値・有効値を含む診断、並行8個体・episodeの50 Reset・共有検証の独立プロセス実行である。回数はテストの規模であり運用上限ではない。

## 7. 受入基準

実装時はDebugビルドと公開構築・GetSpec・Reset・Step・aux/GetScalar、設定/Runner経路を検証し、結果とログを実装計画へ残す。productionにtest-only APIを追加しない。

| # | 操作 | 観測結果 |
|---|---|---|
| A1 | seedとmodeを省略したEnv、seedなしprocess明示Envを反復Reset | 採用seed・盤面が一致し、共通INFO1件 |
| A2 | 新しいテストプロセスで8個体を同時構築し、全個体破棄後に再構築 | 同じ設定なら同じ採用seed・盤面、INFOは1件のまま |
| A3 | 同一プロセスのRunManagerで学習・configured eval・GUI用Env・別Runを構築 | seed未指定・同じ迷路設定で同じ採用seed・盤面 |
| A4 | seed未指定processで異なる迷路設定を使い、共通入力seedだけを各設定へ指定して再現 | それぞれの盤面が一致し、片方のResetが他方に影響しない |
| A5 | seed指定Env・episodeを先に使い、後からseed未指定processを使う | process利用前の共通INFO0件、利用時1件 |
| A6 | 独立した試験プロセスを2回起動 | 各プロセスで共通INFO1件。永続保存・読込なし。偶然同じseedでも失敗扱いにしない |
| A7 | seed未指定episodeで50回Reset | 全盤面が到達可能で、各採用seedだけを指定して復元できる。毎Resetの再採番経路をレビューでも確認し、偶然の重複を不合格にしない |
| A8 | seedだけに0・2・6545537233705677626・最大値を指定 | 反復・再構築で固定し、PRD100の25セルgolden・生成後の乱数順を維持。0を未指定扱いしない |
| A9 | 自動モードの採用seedと画面値を、seedだけで設定へ戻す | 同じ迷路設定で再現し、全範囲が負数で表示されない。共通ログの入力seedからも再現できる |
| A10 | seedとprocess/episodeを併用 | 指定seedで固定し、採番のINFO/WARNは出ない。seedだけの場合と盤面が同じ |
| A11 | modeに空文字・未知値、seedに空文字・nan・負数・上限超過・非整数を指定 | 生成前に明示的に失敗。seedがあっても未知modeは失敗し、黙って代替しない |
| A12 | seed指定で引き直しなし・seed2を複数個体で実行し、自動採番も反復 | 引き直しINFOは指定seed条件で0件・1件、自動採番では追加しない |
| A13 | 旧固定条件の短Runと、新しいseedだけ指定した短Runを比較 | 時間・資源以外の全scalarをtag・step・値で完全一致。Noisyを含む被覆、件数、SHA256を記録 |
| A15 | 共通optional DTOで入力なし・初期値なし/あり、明示0・空文字・false・空vectorとprefix上書きを読む | 初期値なしは各設定記録に葉がなく、初期値ありはその値が記録される。明示空値・false・0は指定値として残り、設定記録から再読込して一致する。不正値は未指定扱いされず失敗する |
| A14 | 全キー/値比較と現用検索 | 変更はseed未指定対応・mode/旧bool置換という意図した差分だけ。旧契約の現用参照なし、履歴は保持 |

## 8. 選ばなかった形・別の関心

| 案 | 扱い・理由 |
|---|---|
| seedとfixedモードの二重指定 | 不採用。再現をseedだけで行うG1に反する。ユーザーが今回変更を求めた |
| NaNや特定の整数をseed未指定の印にする | 不採用。整数型でNaNは表せず、整数のsentinelは固定値域を狭める。未指定を別に表せば足りる |
| 浮動小数点のseed | 不採用。全64bitのseedを正確に保持できずG3の復元を損なう |
| 個体・BatchEnvごとの一度だけの採番 | 不採用。学習・評価・GUIをそろえるG2を満たさない |
| seed指定を自動モードで無視する | 不採用。モード変更なしでseedだけ指定するG1を満たさない |
| run.seedからの派生 | 不採用。既存自動採番と違う契約を持ち込む。再現はseed指定で行う |
| 他Envの統一や迷路難易度制御 | 別の関心。同じ共有要件や難易度比較の必要が出たら別PRDで検討する。クリーンブレーク不能な制約ではない。今回の起票候補なし |

## 9. 未決と経過

未決: なし。

- 2026-10-24: 初回PRDのグリル・自己点検・独立レビューを完了した。
- 2026-10-24: 正式番号102へ採番し、20implを保存。変更前のDebugビルド・固定Runを完了し、66タグ・20095点の基準を保存した。
- 2026-10-24: 旧仕様のprocess単体tracer bulletはRED→GREEN。並行試験の診断中にユーザーがseedだけでの固定へ仕様変更を求めたため、本体の続行を止めてPRDと実装計画を改訂した。旧TDD記録は履歴として保持し、新仕様で検証し直す。
- 2026-10-24: ユーザーが共通optional対応を含む新仕様での続行をYESで承認。PRD品質観点の全項目を自己点検し、目的のずれと旧決定の残滓を除去して改訂レビューへ出す。
- 2026-10-24: 改訂PRDレビューの不足2件を修正。共通optionalの初期値/記録契約とA15、実行基盤設計の日英版を移行対象へ明記した。
