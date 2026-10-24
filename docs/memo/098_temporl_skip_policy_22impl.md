# PRD098 設定ベース化 round2 実装メモ

## 概要

最新の直接依頼 `codex-request-20261024-config-round2.md` と10/24裁定済みPRDを正本とし、round1のprofile配置をbaselineの既定とenv/CLIの葉へ置き換える。Aを先に受入1〜5まで確保し、B・Cと追加の学習確認も完了させる。質問不要というユーザー指示を継承し、暫定判断は本書へ記録する。

## 主な変更

- agent.txtの@baselineへskipの無効既定・配線・学習既定を?=で置く。SkipFC1/SkipFC2をnn.txtへ置き、envはJ・融合済みskip_context・run.@skipのenabled/metrics/Traceだけを持つ。GridMaze切替はOFFへ戻す。
- primaryはhead.use_dueling、skipはskip.head.use_duelingのboolへ移行する。旧文字列と互換層を削除する。SkipNetworkMode enumは保持し、enabled時のaction bind整合は構築境界で検証する。
- 公開ヘッダ `core/anet-core/include/anet/agent.hpp` のSkipConfigをbool Head・J未指定0へ、`core/anet-core/include/anet/default_dqn_agent.hpp` のConfig読込みを新しいboolキーと値域検証へ移行する。公開利用側とテストも同じ変更で更新する。
- 無効時はskip網を構築しない。IQNのtaus bind必須検査を除き、融合済みrank3の特徴をHeadへ渡す。max_lengthは実装既定0（未指定）、有効時未指定と明示0以下を拒否し、明示1は許す。
- B: skip_q.[j]のlane別Trace、SkipNetworkDefinitionとActorの名前、指定箇所の日本語段階コメントを更新する。
- C: skip網のshape dumpとパラメータ数、分布出力skip_q_distのQR分位数/IQN可変軸をGraphVizへ出す。
- 現用コード・設定・テスト・補助ツール・設計文書と20implを追随させる。PRD/ADR/CONTEXT、過去Run・golden・21implは編集しない。

## テスト

Public interface / surface: ConfigData/DefaultDQNAgent構築、MakeAction/GetScalar、HeadのForward/GraphViz、Runnerの解決値・metrics・DOT・Trace。

優先behaviorとTDD順序（1behaviorずつRED→GREEN）:
1. tracer bullet: skip無効なら明示された不正な網を構築しないことをAgent構築で確認する。既知設定値の不正は無効でも拒否する。REDのあとA3だけを最小修正しGREENにする。
2. Head boolとJ既定の設定読込みをRED→GREEN。IQNの融合済み入力を構築・ForwardでRED→GREEN。enabledのaction bind2方向と明示J=1を維持する。
3. IQN包含型ではrank3の特徴とrank2のactionを末尾結合する公開NN/AgentテストをRED→GREEN。結合軸は既存bind_concat_dim=-1を既定へ明示し、汎用のfeature-last concatでrank差のbroadcastを扱う。
4. GetScalarのskip_q.[j]をRED→GREEN（j=1..Jのlane別値、範囲外、無効NaN）。既存cacheにQ値を同梱して読取り同期を増やさない。
5. GREEN後に名前と段階コメントを整理し、可視化の出力軸を確認する。実装本体にtest-only APIを増やさない。

## 検証

着手前を `.scratch/prd098/config-20261024-r2/source-before/`、configをbaseline-config、現在ソースのDebugをbaseline-binへ固定する。変更前4envのOFFを新規採取し、変更後と全scalar完全比較する。7env全解決値も前後採取し、指定のキー置換・既定追加・配置変更とGridMaze既定OFFだけを許す。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
```

- Debug/core全通常テスト、5envテスト、関連Python回帰、check_default_leavesを実行する。
- 4env × default/merge/dueling/action_inの16smoke、1RunのSkipFC1=256とJ=1、Trace Q列、ログのskip形、QR/IQN DOTを確認する。実行体を固定し、比較Runへビルド/単体テストを重ねない。
- GridMaze seed1の500kを分離stop・包含stopで取り直す。既存動作確認と同条件のRelease実行を優先し、eval return/steps、decision/length、有限の学習指標とaction bindを記録する。単一seedから一般的な学習効果は断定しない。
- 品質全項目を自己点検し、独立an-reviewへ計画・実装を渡す。原文を指定scratchへ保存し、不足を直して同じreviewerへ再提出する。

## 前提

- 実装変更に必要な判断待ちは0。最新round2がround1と21implの異なる方針を置き換える。
- 実装のmax_length=0は未指定を表す内部値。設定で明示0は常にエラーにする。
- Rainbowは今回もprimaryの既存use_dueling_netを保持する（round2の対象はDefaultDQNとskip）。
- 受入4のprofile除去はAgent/skip.net/app/env変種が対象。metrics.scalar.@skipはA1で明示的に保持する例外である。
- Trace Qのcache追加はActor infoの購読時だけN×J個のfloat32を既存packへ同梱し、同期回数を増やさない。N=128/J=10なら追加5,120 bytes、未購読なら0。hot pathのforwardやCPU往復を増やさない。
- C2はskip_qが分位平均である現行出力を正確に示すため、skip_qの平均shapeとskip_q_distの分位軸を別出力に示す。IQN可変Kはprimaryと同じ-1表記を使う。
- GridMazeの着手時run.$は包含型ON。比較Runはrun選択を検証profileだけにし、skip.enabled=falseとしてOFFを採る。7env比較もsnapshot-configでGridMazeの切替行だけをOFFへ正規化する。着手時のON解決値はresolved-before-rawへ別途保持する。A7の切替変更自体はsource-before差分で確認する。
- 今回の500k確認はround2受入6に明示された追加実行で、当初の長時間学習準備のみという制限のこの範囲での更新として扱う。段2以降やseed追加は実行しない。

## 経過・結果

着手前snapshotを保存。A→B→Cの順で進め、検証とレビュー結果を追記する。

## 実装中に確認した必要な補完

A1の融合済み特徴とactionの2葉を既存bindへそのまま渡すと、[B,K,F]とone-hot後の[B,A]が異なるrankであるため結合できない。またbind_concat_dimの既存既定1はIQNではK軸を指す。依頼どおりの線形網とCLI2葉を成立させるため、baselineへ `skip.net.branch.[skip_feature].bind_concat_dim ?= -1` を追加する。

汎用NetworkBranchの末尾feature結合に限り、低rank項のbatch直後へsingleton軸を挿入し、既存の中間軸へviewのexpandでbroadcastする。連結軸は最大rankで正規化する。既存の同rank concat、batch一致・batch軸禁止・範囲外軸のfail-fastは保つ。rank差が無い通常経路では追加tensor演算をせず、rank差がある場合もunsqueeze/expandはviewで、catは従来同様1回である。新しい設定モードやAgent専用の汎用分岐を足さない。

ADR0018とwikiのNNカテゴリ、設計130を確認した。専用IQN網を増やさずbind DAGを使う判断に従う補完で、ADRの判断と矛盾しない。実装対象へnn_impl.cpp/nn_test.cppと設計130のconcat契約を追加する。旧『低rank項への末尾結合を拒否する』テストは新契約の正例へ移し、batch・範囲外・非末尾の不整合の拒否を維持する。

### Trace QとDOTの縦スライス

- 21-qtrace-red: 未対応のskip_q.[1]で1失敗。23-qtrace-green: lane別値・範囲外・無効NaN・TraceBuilderの42assertionsが成功。既存診断cacheへQを同梱し、同期回数は既存どおり初回1回。
- 25-dot-red: QR/IQNの両action modeで分布軸4失敗。平均skip_qのshapeを保ち、分布skip_q_distへQRのK/IQNの可変-1を追加する。

### 検証・レビュー結果

- DOTの27-dot-greenは12assertions成功。B2/B3/C1の整理後、全体ビルドでSkipNetworkDefinitionのテスト側4箇所の旧designatorを検出して追随した。外側Pythonからのcmd引用もPowerShell-safeな起動へ直し、最終Debugは29b-final-debug-buildで終了0。
- 36-static-selfcheck: 今回の31ファイルについて着手前からのBOM保持・LF・禁止パターンを確認した。GridMazeのHEADとの差のBOMは着手前からあり、その状態を保持した。
- PRDは並行して§7/§9の実験経過が追記された。自分ではPRD/ADR/CONTEXTを編集せず、その差分を今回の実装スコープに含めない。5×5迷路の未解明事項はこの設定修正で解決したと扱わない。
- 35-audit-and-recipesは設定監査errors=0、長時間レシピ24本とCLI変種指定を検査した。段4は指定checkpointが無いため未生成。レシピ自体からは起動していない。
- 全通常テスト・環境テスト・7環境設定比較・OFF4比較・17smoke・GridMaze2腕の結果は下記に記録した。独立レビューの正式結果は末尾の完了報告に記録した。

- 全通常core: 770成功・1skip、166,906 assertions成功（30-core-full、522.1秒）。5環境はAtari47/DropMerge29/GridMaze1/ImageCls17/LunarLander7ケースがすべて成功。Python監査回帰15件成功。7環境の全解決値は指定変換後の想定外差分0（34-config-comparison）。
- 37-readback: CUDA N=64/J=10、warmup20＋200回。Q同梱cacheのActor平均2.101ms、Learner平均0.0955ms、440 assertions成功。列ごとの同期追加は無い。初回実装時の88-review-testsとは別測定であり、実時間性能の比を示すものではない。N=128への単純な線形スケールを仮定した事前見積もりはActor4.3ms/回、Learner0.1ms/回、2 actor step/updateなら8.7ms/update（上限保証ではない）。全体の性能再測定は今回の受入に含めない。

### OFF4環境の完全一致

38-after-runsは4環境とも終了0、時間・資源以外の全scalar計78,278点が完全一致した。46-off-model-checkで、両側のonline/targetのテンソル数が一致し、skip shape dumpが無いことも確認した。

| 環境 | 点数 | SHA-256（両側共通） |
| --- | --- | --- |
| GridMaze | 20095 | 54cb0e97abb891ac9799018ee5dec88d7de014e536d24cde7704840e44034ea8 |
| LunarLander | 16494 | 9ea069dbb542d46971b111524778d10c29c6cf7df093b2401879312a207e7edf |
| Atari | 18442 | cfd00ecc860aa8098ae5fd0b8c96836cc4c55a98954e7b63f14555f771eb9b66 |
| DropMerge | 23247 | 08c721d17951a9263a7230c40f10deadb2d8fd36d69b4b725f96922a933d03b3 |

### CLI smokeと最終ビルド

39-profile-runsは4環境×4変種と幅256/J=1の全17条件が成功。最新の厳密な生成物検査47でも17条件すべて成功し、C1の列挙したprimary+skipの本数とonline/targetの合計、C2の平均shapeと分位軸、IQN融合入力、action bind、任意Q列、廃止profile不在を確認した。48の最新設定監査は28226 assignments、3561 audited、errors=0。

最終整理（窓の起点を「サンプル起点」と正確にするコメント2箇所、Head出力の指示付き初期化）は41bのDebugビルドと42の `[prd098]` で再検証し、29ケース・16886 assertions成功。コメント以外が不変なことは40にも記録した。最初の41ではcmd自体が起動せずexit markerが空になった。scratchのPS1をWindows PowerShell 5向けのUTF-8 BOM付きへ直した41bで解消した。実装コードのコンパイル失敗ではない。

43のRelease Runnerビルドは終了0。44で長時間レシピ24本のargvをReleaseへ更新した。17smoke・OFF比較・単体テスト・ビルドは順次実行し、比較Runとビルド/単体テストは重ねていない。

### GridMaze seed1・500kの取り直し

45-learning-runsはReleaseで分離stop・包含stopを順次実行し、両方終了0。迷路6×10、J=7、500k予算を解決値で確認し、両方のagent_close.anetを保存した。9指標・ログ・DOTの検査も成功。包含型は実際にactionをbindし、退化した旧腕ではない。Runと全統計はlearning-summary.json、確認は49-learning-evaluation-check.json。

| 構成 | 評価報酬最終 | 評価報酬平均 | 評価ep_steps最終 | 最初の正報酬exp step |
| --- | --- | --- | --- | --- |
| 分離 stop | 0.93 | -0.304 | 8 | 251,136 |
| 包含 stop | 0.93 | -3.778 | 8 | 311,296 |

両型とも最終到達水準は10/23の分離型と同じ+0.93／8stepになり、包含型でも学習が進むことを確認した。一方、このseedでは包含型の最初の正報酬が60,160exp step遅く、評価報酬の全体平均も悪い。学習速度が同等・優位とは報告しない。単一seedの取り直しであり、一般的な学習効果・迷路依存・安定性の結論ではない。5×5既定迷路の未解明事項も解決したと扱わない。

### 自己点検と未実施

実装品質の全62項目を適用範囲に沿って自己点検した。今回の差分は31ファイル。LF/BOM・禁止パターン・契約の移行・エラーとNaN・所有と依存・段階コメント・計測・publicテスト・設定監査・設計追随を確認し、機械検査は36/47/48に保存した。新規C++ファイル/クラス、借用メンバ、互換層、worker例外の変更、UIロック、取り出し口、skill変更のない項目は対象外。独立レビューの正式結果は末尾の完了報告に記録した。

GUIの目視は未実施。Q列はRunの解決値と、GUIが利用する公開StepTraceBuilderの実値出力で確認した。他の長時間学習（追加seed、段2以降）と全体の性能再測定は今回実行せず、長時間レシピ24本をRelease用に準備した。段4は選択された50M checkpointが未指定なので未生成。今回の設定・実装修正に追加判断が必要な未決はない。

## 完了報告

- 実施内容: round2のA→B→Cを実装し、TDD、Debug/Release、全通常テスト・環境テスト、設定監査、7環境解決値、OFF4環境完全一致、17smoke、指定GridMaze2腕を完了した。
- 変更ファイル: source-before基準の31ファイル。scope.jsonに一覧、設計110/130/140/200、20impl/本計画と検証READMEも同じ変更で追随した。並行PRD変更と既存dirtyは保持した。
- レビュー結果: round2正式レビュー1回（静的確認後に最終証跡を追送）。計画・実装ともOK、差し戻し0。原文: `.scratch/prd098/review-20261024-0804.md`。最終到達水準の一致と速度同等未立証、GUI等の未実施、着手前からのGridMaze BOMを再確認した。見送った指摘はない。
- 決まらなかったこと: なし。新たな起票候補もなし。単一seedの学習結果の限界と未実施の検証は上記のとおり保持する。
- 判断してほしいこと: なし。
- 次にできること: 人間による差分確認・コミット。stage/commit/pushは実行していない。この工程の作業は完了し、停止する。

## 追加修正: 設定キーとDTOの一致

ユーザーの追加指摘により、固定キーは特別な理由がない限りANET_READ_CONFIGを使う。Head初期化の文字列キーと平坦なメンバの対応付けをやめ、primaryのhead.init/head.use_dueling/head.projectionとskip.head.init/skip.head.projectionへDTOを揃える。既存のskip.head.use_duelingも同じ階層に保持する。primary→skipの初期化・射影の継承と各既定値、設定の文字列キーは不変。enumへ変換するskip.network_modeと動的なactorキーはReadConfigが必要な理由があるため維持する。

検証: 既存の設定継承・Head構築テストを新しい公開メンバへ移し、Debugビルドと関連テストで値と継承、Headへの受渡しが不変なことを確認する。これは設定の外部挙動を変えない構造整理であり、直前のGREENから既存テストを維持する。結果と独立レビューは後続に記録する。

### 追加修正の検証結果

検証ログは `.scratch/prd098/config-macro-20261024-161604/`。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 2'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[prd098]'
```

- 02-debug-build: 終了0。03-prd098-tests: 終了0、29ケース・16,886 assertions成功。既存の設定継承、Head構築、動作のテストがすべて通った。
- 04-config-audit: 28,363 assignments、3,561 audited、errors=0。
- 05-contract-check: 固定キー114個の集合が完全不変。利用側とテストはメンバパスの置換のみで、13マクロが元の明示キーと一致。旧メンバ名とHeadの文字列ReadConfigの残存0。
- 全62項目を適用範囲に従って自己点検。6差分のBOM/LF保持、git diff --check、所有権、依存、継承順、初期値、公開利用側の移行を確認。構築時の設定配置整理であり、step内の参照・計算・乱数・同期は変更していない。新しい共通機構・worker/UI・ログ・skillの変更は対象外。
- ソース占有で途中の編集が止まったため、先に開始していた自分の01ビルドを停止し、編集完了後の02で検証をやり直した。01は成功に含めない。
- 外部設定キーと挙動が不変の構造整理のため、今回の追加修正ではRelease/全通常テスト/OFF比較/長時間Runを繰り返していない。前節のround2結果は当時の検証として保持する。
- 追加の判断待ちはない。独立レビューは正式判定OK。結果と原文は下記に記録した。

並行編集の注記: 05-contract-checkが全文一致を確認した16:21:45の後、default_dqn_agent.cppのinclude順序が16:23:53に並行して変更された。今回の編集スクリプトはメンバパスの置換のみでincludeを変更していない。06-parallel-include-changeで、自己点検時SHAはBOMも保持した期待される置換結果と一致し、現物との差は同じinclude集合の並べ替えだけであることを確認した。並行変更は保持し今回の判定対象に含めない。同ファイルの今回だけの差分はour-default_dqn_agent.cpp.diff、他はdiff/を使う。ビルドとテストは今回の変更を検証したもので、後からの並行include整理の検証まで主張しない。

### 追加修正の完了報告

- 実施内容: Headの設定階層を実装と一致させ、固定キー13箇所をANET_READ_CONFIGへ移行。既存の継承と外部キーは不変。
- 変更ファイル: 今回6ファイル。config-macro-20261024-161604/scope.jsonと今回だけの差分を参照。
- レビュー結果: 追加修正の正式レビュー1回、OK、差し戻し0。原文: `.scratch/prd098/review-20261024-1626.md`。キー・継承・公開利用側と検証証跡を再確認。見送った指摘はない。並行include整理は上記のとおり範囲外。
- 決まらなかったこと: なし。新たな起票候補もなし。
- 判断してほしいこと: なし。
- 次にできること: 人間による差分確認・コミット。stage/commit/pushは実施していない。追加修正は完了し停止する。

## 追加修正: WeightInitConfigの共通検証

ユーザーの追加指摘に基づき、WeightInitConfig固有のmode・数値の有限性・trunc_stdと切断範囲の検証をNN共通へ集約する。nn.hppでValidateWeightInitConfig(config, owner)を公開し、実装は既存のnn_impl.cppに置く。Agent固有名を共通層へ持ち込まず、呼出元が論理キーをownerとして渡す。DefaultDQNのprimary/skip設定読込で呼び、skip無効時も認識済み不正値の拒否を維持する。WeightInitializerの初期化前にも同じ関数を呼び、従来の未知mode/TruncNormal専用の重複検査を削除する。projectionやnonlinearityの用途別解釈は今回変更しない。

TDD: primaryの未知modeをskipの正しい上書きで隠せる設定を公開ConfigData経路でREDにし、共通検証からprimary/skipを検査してGREENにする。続けて共通公開関数と通常のWeightInitializerについてmode・有限性・範囲・診断キー・初期化前失敗を検証する。正常初期化の乱数・重みは既存torch比較テストで維持する。Debugビルド、[nn][init]と[prd098]、自己点検、同じ独立reviewerによるレビューまで進める。公開面の追加は上記関数1本だけ、公開ヘッダnn.hpp/default_dqn_agent.hppとnn_impl.hpp/.cpp、近傍テスト、設計130英日を対象とする。

### WeightInitConfig共通化の検証結果

ログは `.scratch/prd098/weight-init-20261024-164343/`。実装コードのSHAはレビュー依頼時から変わっていないことも確認した。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 2'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[nn][init],[prd098]'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[nn]~[init]~[prd098]'
```

- 01-red-build終了0、02-red-testは1ケース1assertionが期待した理由（例外が出ない）で失敗、exit42。primaryの未知modeをskip側の正しいmodeで上書きしても、設定読込で拒否されるべきことを再現した。数値パーサは既にNaN/Infを拒否するため、最初に考えたNaNではなく未知modeへ実行前に絞った。
- 03はVsDevCmdパスの誤記で起動失敗。正しい03b-green-buildは終了0。03をコンパイル成功として扱わない。
- 04-green-test終了0、35ケース・16,938 assertions成功。正常なlibtorch初期化との一致、共通APIの全6mode・5数値のNaN/Inf・範囲・owner診断・適用前の重み不変、無効skipでの明示不正値拒否、primaryの未知mode拒否を含む。
- 05-other-nn-tests終了0、残りNNの131ケース・2,391 assertions成功。上のフィルタと重複せず、合計166ケース・19,329 assertions成功。
- 全62項目を適用範囲に沿って自己点検。共有層はWeightInitConfigの共通条件だけを持ち、Agent名やskip固有キーを持たない。認識した不正値のfail-fast、キー/値/期待値の診断、primary→skip継承、初期化前の検証、正常時の乱数順、所有・依存・公開面の追随を確認した。新しい状態・クラス・worker/UI・GetScalar・ログ・skillを追加していない項目は対象外。
- LF/BOM保持と禁止表記の検査は成功。HEADのdiff --checkで指摘するnn_impl.cppの空白行は着手前からのdirtyで、今回追加した差分の空白違反は0（incremental-whitespace.json）。無関係な既存差分は保持した。
- 外部設定キー・ファイルを変更していないため設定ファイル監査は今回対象外。Release/全通常core・環境テスト/比較Run/長時間学習は繰り返していない。共通検証の境界変更は今回のDebug・初期化・全NN・PRD098で検証し、過去Runは当時の記録として保持する。
- 未解決の判断待ちはない。独立レビューは正式判定OK。結果と原文は下記に記録した。

### WeightInitConfig共通化の完了報告

- 実施内容: WeightInitConfigの共通検証をnn.hppへ公開し、既存nn_impl.cppへ実装を集約。Agentのprimary/skip読込とWeightInitializerから呼び、重複guardを削除した。REDからGREEN、Debugビルド、関連166ケース・19,329 assertions成功。
- 変更ファイル: 公開/内部ヘッダ、実装、近傍テスト、設計130英日、本メモの9ファイル。今回だけの差分・scope.jsonをログrootへ保存した。
- レビュー結果: 追加修正の正式レビュー1回、OK、差し戻し0。原文: `.scratch/prd098/review-20261024-1659.md`。検証の集約・順序・診断と全証跡を再確認。見送った指摘はない。
- 決まらなかったこと: なし。新たな起票候補もなし。
- 判断してほしいこと: なし。
- 次にできること: 人間による差分確認・コミット。stage/commit/pushは実施していない。追加修正は完了し停止する。

## 追加修正: DefaultDQNAgent保存形式の版と定数

ユーザー指摘に基づき、機能名/機能内バージョンであるDefaultDQNAgent/skip-v1をやめ、DefaultDQNAgent全体の保存形式の第1版/第2版として定義する。既存primary-onlyの第1版はDefaultDQNAgent、拡張レイアウトの第2版はDefaultDQNAgent-2。default_dqn_agent.cpp内の不変定数に識別子を一元化し、書込・読込・不正値の診断で共有する。識別子以外のバイト配置は変更せず、第1版の保存と既存primary checkpointからの分岐要件は維持する。旧skip-v1のalias/専用検出は残さず、過去Run artifactは書き換えない。

公開Saveからヘッダを読むテストを先にREDにし、第2版の識別子を定数へ移してGREENにする。Debugのanet-core-test/AnetRLRunner、[dqn][serialize]と[prd098]を実行し、現用参照、LF/BOM、全品質項目を点検する。同じ独立reviewerへ今回4ファイルだけの差分を提出する。設計200と本記録を追随させる。新しい公開APIや設定キー、追加の状態フラグ、汎用ArchiveHeaderの版変更は行わない。直前の設定読込の3フェーズ案は説明段階のままとし、この修正には混ぜない。

### 保存形式の版と定数の検証結果

ログは `.scratch/prd098/checkpoint-format-20261024-171158/`。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 2'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[dqn][serialize],[prd098]'
```

- 01-red-build終了0。02-red-testは2assertions中、第1版が成功、第2版の旧識別子が失敗、exit42。公開Saveからヘッダを読んで実際の値の違いを確認した。
- 定数化後の03-green-build終了0、04-green-testは34ケース・20,026 assertionsすべて成功、exit0。第2版の復元、第1版からskip有効への分岐と次のprimary更新の一致、逆方向の拒否を含む。
- レビューの静的指摘1件（新規テストのBatchEnvSpecは集成体なので指示付き初期化にする）へ局所対応した。変更は `{1, 1}` を `{.num_envs = 1, .num_threads = 1}` にした1箇所だけ。05-final-buildでanet-core-testをDebug再ビルドし終了0、06-final-header-testの `[checkpoint-format]` も成功した。productionと設計のSHAは最初のレビュー提出から不変で、テストもその記法変更だけであることを照合した。
- 全62項目を適用範囲に従って自己点検。第1版/第2版の定義一元化、writer/reader/診断の一致、SaveのConfig以降のpayloadコード不変、既存checkpointの復元契約、const状態、公開Saveテスト、設計追随を確認した。新しい公開API・設定キー・状態・worker/UI・GetScalar・skillはない。
- selfcheck.jsonで4ファイルのLF/BOM保持・追加空白エラー0、第2版のproduction文字列1定義、現用core/include/designの旧機能別識別子残存0を確認。以前のdirtyやRun artifactは保持した。
- 識別子以外の保存レイアウトと実行時の計算は変更していないため、Release/全通常core・環境・NNテスト/比較Run/長時間学習は今回は繰り返していない。設定ファイルも変更していないため、その監査は対象外。
- 追加の判断待ちはない。独立レビューの正式結果はOK。原文と完了報告は下記に記録した。


### 保存形式の版と定数の完了報告

- 実施内容: Agent全体の保存形式を非公開のkCheckpointFormatV1/V2へ定数化し、第2版をDefaultDQNAgent-2へ変更。書込・読込・診断で共有し、第1版とpayloadの配置を維持した。
- 変更ファイル: default_dqn_agent.cpp、dqn_based_agent_test.cpp、設計200、本メモの4ファイル。今回だけの差分と検証ログは上記ログrootに保存した。
- レビュー結果: 正式レビュー1回、OK、差し戻し0。途中の静的指摘1件は新規テストの指示付き初期化へ対応し、再ビルド・再テスト成功。見送った指摘はない。原文: `.scratch/prd098/review-20261024-1722.md`。
- 決まらなかったこと: なし。新たな起票候補もなし。
- 判断してほしいこと: なし。
- 次にできること: 人間による差分確認・コミット。stage/commit/pushは実施していない。今回の追加修正は完了し停止する。
