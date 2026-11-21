# GridMaze 到達可能性保証 実装メモ

## 概要
PRD100に従い解けるseedの盤面を保持し、解けない候補だけ派生seedで引き直す。ユーザーが2026-10-24に提示計画の実装を承認した。公開API・設定キー・観測・報酬は変更しない。

## 主な変更
- GridMazeEnvの候補生成順と乱数消費を保持し、開始決定後に4近傍で壁と穴を避ける探索を行う。初回seedと splitmix64(seed0 + k), k=1..1000 を順に試し、全失敗はANET_SYSTEM_ERROR。
- 正の寸法、座標の -1 または範囲内、固定座標の重複、開始候補なしをfail-fastにする。固定開始とランダムゴールの偶然一致は引き直す。
- 採用seedをcurrent_seed_とauxに公開し、設定seedは保持する。固定seedの引き直しだけ設定seed・回数・採用seedをlog.info()で通知する。
- GridMazeEnv.cpp内のmutexと出力済みseed集合はユーザー承認済みの限定例外。プロセス全体で設定seedごとに1回とし、迷路生成には用いない。
- 既存の穴終端テストを3x2、壁0、穴0.5、seed58（.OG / ...）へ移す。PRDとEnv設計文書の日英版、関係wikiを追随させる。

## テスト
- Public interface / surface: GridMazeEnv構築、Reset、Step、結果のaux、Step統計、Runnerログとmetrics。
- TDD順序: 既存穴テストの移行確認後、seed2の構築→Reset→auxを最初のtracer bulletにする。各behaviorを1テストずつRED→GREENで進め、GREEN後だけ整理する。
- 優先behavior: seed2の到達可能性、同一seedの反復と別個体の決定性、採用seedによる復元、既定seedの25セルgolden、上限失敗、座標と寸法の異常、候補なし、ランダムゴールとの一致、自動seed50回と部分ランダム座標、並行生成を含むログ抑制。
- 到達可能性はテスト側の探索でauxから独立に調べ、得た経路を公開Stepで進めてゴール到達も確認する。private methodやtest-only production APIは使わない。

## 検証
- 着手時のdirty一覧、HEAD、対象ファイル、現在ソースのDebugビルドを.scratch/prd100へ保存する。変更前Runner/DLLとconfigを固定する。
- PRD098段0相当のGridMaze短縮Run: run seed1、8 lane、8704 experience、CPU、FP32、決定論backend、skip無効、既定迷路seed。変更前後の全scalarをtag/step/値で正規化し完全一致を確認する。時間・資源の90_perf/、91_resource/、92_resource/だけ除外し、Noisy指標を含める。checksumと点数、tagを記録する。
- ログは別プロセスで検証する。既定seedの通知0件、seed2で学習256 laneと評価用Envを作る短いRunの通知1件。専用テストで反復・並行生成と自動seedを確認する。
- MSVC初期化はAGENTS.mdのVsDevCmd経由。Debug全体ビルド、core/envs/gridmaze1/bin/Debug/GridMazeEnv-test.exe "[gridmaze]" を実行する。
- 専用の再実行資材はcore/anet-core/testdata/prd100/に置き、Env設計のテスト章から参照する。品質確認観点を全項目自己点検し、独立an-reviewに計画と実装を依頼する。改訂PRDは別のprdレビューへ出す。原文・差分・検証ログを.scratch/prd100へ保存し、修正と再レビューを行う。

## 前提
- ユーザー判断が必要なブロッカー0。共有ログ状態の限定例外と、固定開始・ランダムゴール一致時の引き直しは明示回答済み。
- 上限は初回生成に加えて最大1000回の引き直し。seed加算はseed_tのunsigned演算。
- 設定の密度和が1を超えることだけでは拒否しない。既存のセル抽選を保持し、解けなければ生成上限で失敗する。
- ランダム開始の候補が空なら、構築中の初回Resetを含め、その候補を選ぶ境界でfail-fastする。黙った(0,0)代替は削除する。
- 既存のdirty変更と過去Runは保持する。難易度制御、長時間学習、PRD098の崩壊調査、性能受入、新しいADR/用語は範囲外。
- コミット、staging、pushは行わない。

## 経過
- 2026-10-24: 計画を保存。変更前の検証を開始する。

## 検証・レビュー結果
実施済みの結果を以下へ記録する。Run比較と実装レビューは完了。

- 変更前Debug全体ビルド成功。既存テスト42 assertions、穴テスト移行後46 assertions成功。
- before Run正常終了: 134.0秒、66 tag、20095点、SHA256 54cb0e97abb891ac9799018ee5dec88d7de014e536d24cde7704840e44034ea8。
- seed2 tracer bulletのREDを確認（seedが2のまま・経路なし）。到達判定と引き直し後のGREENは57 assertions/2 cases成功。
- 改訂PRDの独立レビュー1往復目はOK、指摘・未決なし。原文: .scratch/prd100/review-20261024-0913-prd.md。

### 監査で見つかったD3の表示不整合
GridMazeView.cppの既存seed表示は%lldで、上位bitが立つseedを負数にし、uint64設定読込は負数を拒否する。D3の再現経路に必要な局所修正として%lluとunsigned long longへの変換へ移す。盤面の描画方法・auxの表現・公開APIは変更しない。軽微な書式変更のため専用の実装を写すテストは作らず、既存の最大seed復元テストを設定の文字列読込経路へ強化し、コードと独立レビューで表示書式を確認する。

- ログ抑制の共有集合は初回の引き直しResetでだけ照会し、個体に照会済みを保持する。固定seedではコンストラクタの初回Resetで完了するため、以後のStep内の自動Resetへ共有ロック待ちを追加しない。共有ロック内ではseed集合への挿入だけを行い、INFO出力はロック外に置く。
- GenerateGridMapとHasReachableGoalにANET_PROFILE_FUNCを置き、Reset内の候補生成と到達判定を計測可能にした。

- 固定座標重複、負の寸法、座標範囲、開始候補なしはそれぞれRED→最小実装→GREENを確認。ログは並行8個体の採用seed一致を確認した上で通知0件のREDを確認した。
- 通知実装前の回帰: 9 cases / 2397 assertions成功。自動seedは9構成×各50Reset、密度和>1の解ける盤面も含めて確認。
- 最大uint64 seedを設定の十進文字列読込へ戻して復元するテストを強化した。
- Noisy11定義を明示選択してbaselineを採り直した。実効設定のNoisyは無効で値は未出力。全scalarの66 tag / 20095点とchecksumは初回と同じ。新Runの識別で同名suffixが2件となったため、正常終了0の記録と旧Run pathを照合して新Runを一意に選び直した。以後のdriverはsuffixに時刻を付ける。
- D3補足後のPRD独立レビュー2往復目OK。修正要求・範囲外検出・未決なし。原文: .scratch/prd100/review-20261024-0932-prd.md。

## 実装後の検証結果

| 検証 | 結果 | 記録 |
| --- | --- | --- |
| MSVC経由Debug全体ビルド | 成功 | .scratch/prd100/final-debug-build.log |
| GridMazeEnv-test.exe "[gridmaze]" | 9 cases / 2397 assertions成功 | .scratch/prd100/final-gridmaze-test.log |
| GridMazeEnv-test.exe "[prd100_log]"（別プロセス） | 1 case / 14 assertions成功。並行8個体と反復Resetでも1通知 | .scratch/prd100/notice-green.log |
| 変更後GridMaze短いRun | 正常終了、66 tag / 20095点 | .scratch/prd100/after-validation.log |
| before / after全scalar比較 | 完全一致。SHA256 54cb0e97abb891ac9799018ee5dec88d7de014e536d24cde7704840e44034ea8 | .scratch/prd100/comparison.json |
| 256学習Env＋既定評価設定、既定seed | 引き直し通知0件、正常終了 | .scratch/prd100/log-default/log-check.json |
| 同上、seed2 | 通知1件。retries=1、actual_seed=2092789425003139053、正常終了 | .scratch/prd100/log-retry/log-check.json |
| LF・BOM・Python構文・文書リンク・diff whitespace | 成功 | .scratch/prd100/self-check.json、git diff --check |

Run検証は .venv/Scripts/python.exe core/anet-core/testdata/prd100/run_validation.py の before / after / compare / log-default / log-retry を使用した。beforeは出力識別の補正を上記のとおり記録し、Runner終了0と採取設定を確認したうえで正規化した。after以降の4条件はdriverがすべて終了0。Noisy11定義を選択済みで、無効Noisyの値が未出力であることをbefore/afterのnoisy-coverage.jsonへ記録した。

GUIの実画面を操作する目視確認は未実施。seed表示の1行はコード照合、最大uint64 seedの設定文字列読込・復元テスト、独立レビューで確認する。盤面描画方法は変更していない。性能ベンチと長時間学習はPRDの範囲外であり実行していない。

## 自己点検
実装品質確認観点を全項目、適用範囲に従って点検した。C++の共有状態はD8とユーザー回答による限定例外。private APIだけを追加し、公開ヘッダ、設定、Agent/Runnerの契約、旧Run artifactは変更していない。TDDの失敗・成功、境界、自動seed、上限、ログ、checksumを上記へ記録した。未決なし。実装計画と実装の独立レビューは1往復目で両方OK。

## 完了報告

- 実施内容: 到達可能性判定と最大1000回の決定的な引き直し、構築時検証、候補なしのfail-fast、採用seed公開、プロセス全体の通知抑制を実装した。既定盤面・乱数位置を保持した。D3のseed表示だけを符号なし十進へ修正した。
- 変更ファイル: GridMazeEnv.cpp/.hpp、GridMazeEnv_test.cpp、GridMazeView.cpp、testdata/prd100のdriverとREADME、Env設計文書の日英版、PRD100、実装計画、wiki322とlog。計12ファイル。既存の無関係なdirty変更と過去Runを保持した。
- レビュー結果: PRDは2往復目でOK。計画・実装は1往復目で両方OK。修正要求、範囲外の検出、開発者への確認事項なし。GUI目視は未実施で、seed表示の静的照合と最大値の設定復元テストで確認した。範囲に当たらない観点の扱いはレビュー原文に記録した。
- 検証結果: Debug全体ビルド、9ケース2397 assertionsと別プロセス1ケース14 assertions、変更前後の66tag/20095点完全一致、256学習Envの通知0件/1件、文字コードと文書リンクの確認が成功した。
- 決まらなかったこと: なし。記録する課題の候補はない。
- 判断してほしいこと: なし。
- 次の工程: コミットは人間が実施する。staging・commit・pushは行っていない。

実装レビュー原文: .scratch/prd100/review-20261024-0941-impl.md。
