# RND の補助学習の刻みと件数を揃える実装メモ

初回実装は [20impl](116_rnd_lifelong_novelty_20impl.md)、採点と補助学習の独立化は [21impl](116_rnd_lifelong_novelty_21impl.md) に記録する。

## 概要

[PRD116](116_rnd_lifelong_novelty_10prd.md) の2026-11-07改訂と `.scratch/prd116/request-20261107-schedule.md` に対応する。基準コミットは `08bf1e2e62a16cafa2e44c5ed4ae70015d87acf2`。作業ツリーには21implまでの変更と別作業があり、それらを保持する。PRD本文はユーザー側の更新を入力とし、こちらでは編集しない。

補助学習はPush側に置いたまま、facadeのcreditを `N × learn_ratio / k`（`k=min(learn_batch_size,N)`）ずつ加算する。学習は1 intakeにつき最大 `floor(N/k)` 回、k laneずつ重複なく渡す。既定はratio=0.25、batch size=32、warm-up=12,800観測。N不変の回数はNがbatch size以上で、intake上限が制約しない条件について検証する。小さいNではkへ縮小し、割り切れないNでは未消費の整数creditも残る。

## 主な変更

- `core/anet-core/include/anet/intrinsic.hpp`、`src/intrinsic_rnd.hpp/.cpp`: `Learn`をbool戻り値へ変更し、optimizerを進めたときだけtrue。warm-upはfalse。抽選・乱数・`train_fraction`を除き、渡された観測すべてで学習する。採点件数を累積し、RND保存形式を `RndNoveltyEstimator-2` にする。
- `core/anet-core/include/anet/default_dqn_agent.hpp`、`src/default_dqn_agent.cpp`: facade設定2葉を読み検証する。credit・名前付きseed `intrinsic.learn` の乱数・成功回数を所有する。1 intakeで1つの順列を作り、k件ずつ切り出す。成功の可否によらずcreditを消費する。RND本体の後にcredit・乱数・累計成功回数を保存する。成功回数も保存するのは再開後も累計を継続するため。
- `core/anet-core/src/intrinsic_test.cpp`: publicな設定・`CreateLearner()->UpdateFromBatch`・`GetScalar`・`Save/Load`と実物のRNDで検証する。偽の推定器やtest-only本体APIは作らない。
- `apps/runner/config/agent.txt`: facadeの2葉とwarm-up既定を更新し旧葉を削除。`Atari.txt`・`GridMaze.txt`・`LunarLander.txt`の `run.@rnd` に `32_learn_count` を追加。
- `docs/design/212_rnd_lifelong_novelty.jp.md`: §3.4〜3.6・3.8・4.2・4.4・4.5・5.1〜5.2を更新。`200_dqn_agents.jp.md/.en.md`: §6.2・7.2と保存・診断の関連記述を更新し、内容照合後に英語版の翻訳元情報を更新。
- `core/anet-core/testdata/prd116/README.md`の案内、`docs/management/issues/issue-20261106-16b7af.md`の対応内容を更新する。

## テスト

- 面: RNDの `Estimate/Learn/Diagnostics/Save/Load`、facadeの設定読み込み・Learner入口・GetScalar・checkpoint。
- TDD1: N=2とN=4、同じexp step数で成功回数と学習件数が揃うテストを先に失敗させ、facadeのcreditとRNDの全件学習を実装して通す。
- TDD2: warm-up件数とbool結果、falseでもcreditを消費することをテストし、件数への変更を実装する。境界は既存の「閾値までのintakeでは学習しない」を保持し、採点前件数でα、採点後件数が閾値を超えたときLearnを進める（0なら単独Learnも進める）。
- TDD3: 実物のRNDと保存状態を用い、重複しない選択、kへの縮小、上限・繰越、乱数列・端数・成功回数の保存再開を確かめる。checkpoint実装を通す。
- 値域（0・負・上限超過・非有限・非整数）、無効時の葉の未読、loss保持、RND書式の不一致拒否、CPU/CUDAと既存回帰も確認する。

## 検証

各TDDサイクルは同じMSVC初期化でtest targetをビルドし、個別タグを実行する。標準出力・標準エラーは `.scratch/prd116/schedule-*.log` へ保存し、モデルへは終了値と末尾の要約だけ返す。待機は30〜60秒を使い、進捗はビルド出力・終了値で判断する。時間だけでは停止と判断しない。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug' *> .scratch/prd116/schedule-build.log
core/anet-core/bin/Debug/anet-core-test.exe '[prd116],[scaler]' *> .scratch/prd116/schedule-test.log
core/anet-core/bin/Debug/anet-core-test.exe '[shipped_config]' *> .scratch/prd116/schedule-shipped-test.log
```

設定はPRD072の既定葉検査と変更前後の全キー・文字列値比較で意図した差分だけか確認する。旧名は現用のコード・設定・設計文書を検索する。履歴の20impl・21impl・PRD改訂記録・wiki log・Runの写しは保持する。他機能の一般語 `pending`・`Multiplier` まで削除しない。BOM・UTF-8/LF・差分空白・Markdownリンクを確認する。

## 前提

- 2026-11-07ユーザー回答: Learnはoptimizerが進めばtrue、warm-upではfalse。facadeはtrueだけを数え、creditは結果によらず消費し、warm-up分の追いつき学習をしない。
- 未決事項監査: ブロッカー0。Learnerの `earned_credit_` / `update_credit_` と同じ加算・消費の形を使うが、状態はfacadeに置く。RND型・設定の葉はfacadeへ漏らさない。
- 名前はPRDの指定を採用する。`Learn`・`Estimate`は前回の公開API、creditは既存Learnerの用語。CONTEXT・ADRの新設は範囲外。
- API追加はLearnの戻り値だけ。無効時・呼び順・ロック・ネット構造・αの式は保持する。学習が無いintakeでは直近lossを保持する。
- 抽選indexはCPUでまとめて作り、入力deviceへの転送は学習ループの前で1回にする。既存のLearn内の入力転送・誤差readbackは1 optimizer stepごとの固定費で、laneループには入れない。N=128・既定では1 intakeあたり従来と同じ1 step、index転送は最大N個のint64（1 KiB）。PCIe 10 GB/s仮定の帯域分は約0.1µs/回だが起動・同期費用は未測定。実測はRun競合のため保留する。
- 補助学習の同期転送の頻度は通常1 intakeあたり平均N×ratio/k回（上限floor(N/k)）。float画像84×84、k=32、PCIe 10 GB/s仮定では、既存のCPU正規化後の入力H2Dは約0.09ms/step、入力もCUDAなら正規化前D2Hがさらに約0.09ms/step、誤差k個のD2Hの帯域分は約0.013µs/step。ここへ転送ごとの未測定の起動・同期時間が加わる。N=128・既定なら各々毎intake1回、ratio=1なら4回となる。これは帯域だけの見積もりで性能保証ではない。CPU計算・CPU入力ならこれらのdevice転送はない。既存のCPU正規化・readbackは今回の変更対象外であり、k件をまとめて扱いlaneごとの同期は足さない。
- 4環境checksumはRunの切れ目まで保留。smoke不要、性能測定はしない。15:56の実行体・走行中Run・runs配下に触らない。

## 経過と結果

- 着手: 依頼とPRD・現行実装を照合。Learnのbool契約とcredit消費はユーザー回答で確定した。ステージ操作は今回の実装には含めない。
- TDD1 RED: N=2・4の `GetScalar(intrinsic.learn_count)` が未認識で2 assertions失敗（終了42）。テストの初回コンパイル時のnarrowing conversionを直してから、この挙動の失敗を確認した。GREEN: credit・全件学習の実装後、1 test /28 assertions成功（`schedule-red-*`・`schedule-green-*`）。
- TDD2 RED: 観測件数4のwarm-upが呼出回数で数えられていたため8 assertions失敗（終了42）。GREEN: B件ずつ累積する変更後、関連23 tests /1055 assertions成功（`schedule-warmup-red-*`・`schedule-warmup-green-*`）。
- TDD3 RED: facadeのcreditと抽選状態が復元されず、保存再開テストの42 assertions中31件が失敗（終了42）。GREEN: RND本体の後へcredit・乱数・成功回数を保存した後、1 test /42 assertions成功（`schedule-save-red-*`・`schedule-save-green-*`）。
- 境界テストを追加。最初の全体Debugビルドは成功したが、抽選テストの継続intakeにもepisode_start=trueを渡したため、Replayの契約検査で4通りが停止した（27/28 tests成功）。継続intakeのepisode_startをfalseへ直した。本体の抽選結果と許される全順列の重み照合は、この時点でも成功していた。
- 修正後のDebug全体ビルドと関連28 tests /1214 assertionsが成功（`schedule-final-build.log`・`schedule-final-test.log`）。抽選経路のCUDA入力も追加して最終確認する。
- CUDA抽選の初回検証では観測だけをCUDAへ移したテスト入力がdone/rewardと不一致になり、4条件がmasked_fillで失敗した（`schedule-device-test.log`）。通常の `BatchExperience::To(input_device)` で全fieldを揃えるようテストを修正。本体変更は不要。独立レビューでも同じ懸念が挙がり、修正と再検証中であることを共有した。
- 最終のDebug全体ビルド成功、CPU/CUDA入力のfacade抽選を含む関連28 tests /1293 assertions成功、いずれも終了0（`schedule-device-final-build.log`・`schedule-device-final-test.log`）。検証節の全体ビルド・関連テストと同じコマンドを使い、出力先だけこの名前に分けた。テスト内の全fieldのdevice不一致は解消した。
- 出荷設定は `core/anet-core/bin/Debug/anet-core-test.exe '[shipped_config]'` で11 tests /2583 assertions成功、終了0（`schedule-shipped-test.log`）。
- 設定7種類の全キー・文字列値を変更前の写しと比較し、意図した28差分だけを確認した。`./.venv/Scripts/python.exe .scratch/prd116/schedule_checks.py compare` は終了0。記録は `.scratch/prd116/schedule-config/changes.json`。3環境の32_learn_count宣言は差分と出荷設定検査でも確認する。
- `./.venv/Scripts/python.exe core/anet-core/testdata/prd072/check_default_leaves.py` は終了1、31,593 assignments・3,886対象中2件を検出。Atari.txtの既存 `app.online.eval_panel.fps ?= 14.5` と `app.online.train_panel.fps ?= 10`（現在1541・1545行）。変更前の写しにも同じ行があり、今回の差分は32_learn_countの1行だけ。この2件は別件として保持し、新規違反は0。
- 200日英のRND節（6.2・7.2・8.2・9.1）は対応を確認。日本語版の別作業による2章のアルゴリズム解説追加と関連参照は今回の英訳範囲外であるため、翻訳元hashは現在の日本語版に合わせ、progressは `chapter 1/11` として未訳を明示した。今回更新済みのRND節は補足コメントに記した。
- レビュー中も200日本語版の別作業の2章は更新が続いた。RND関連の追加行がレビュー時の写しと一致することを再確認し、英語版は確認済みの翻訳元blob `672ef1368d4cf58ffbdae3604f33dc2554f75d2e` と未完了表示を保持した。未訳部分の変更に追随してhashだけを進めない。静的検査も、全文hash一致か、未完了表示と今回確認したRND節の一致で判定する。

## 自己点検

実装品質確認観点の全項目を走査した。公開APIは計画どおりbool結果へ変更し、設定値域・非整数・無効時未読はpublicなConfigで確認する。Tensorは先頭軸がlaneで、同じindexを各fieldへ適用し、既存RNDのfeaturesは[B,F]検証を維持する。新しい所有状態はfacadeに閉じ、GetScalarは既存atomic経由、呼び順とロック範囲は維持した。旧名から新名は、`rnd.train_fraction`廃止→facadeの`intrinsic.learn_ratio`・`intrinsic.learn_batch_size`、`intrinsic.train`→`intrinsic.learn`。Estimate/Learnへの改名は21implの変更を引き継ぐ。

現用コード・設定・設計文書・CONTEXT・wikiの実装の地図に旧キー・旧seed・pending_errors・旧Multiplier呼出しは0件。履歴資料は保持した。共有プロセスの非依存、ヘッダのusing namespace、production dynamic_cast、TODO表記も違反0。BOM15ファイル保持、新規1ファイル、UTF-8/LF・Markdownリンクとfence・diff空白・翻訳元hashは検査成功。文字列検査の候補は設定キーと設定キー用の葉名で、定数化除外条件に該当する。

実装・テスト・設定・文書・記録の16ファイルを計画と照合した。PRDはユーザー側の変更を参照し編集しない。200日本語版の別作業の2章ほかの追記、既存20impl・21impl、bat、wiki、未追跡資料とindexの他者による変更は保持する。

### 受入条件と確認する値

| 条件 | 入力・観測対象 |
| --- | --- |
| 刻みと件数 | N=2/4、ratio=0.25、k=2、warm-up=0、合計32経験。各intakeのGetScalarを照合し、どちらも4 step・計8観測。1 stepの全入力がk件であることは抽選テストの重み照合で確かめる |
| 抽選 | CPU/CUDA入力それぞれでN/k=2/32（縮小後2/2）、3/2、4/2、4/1、ratio=1。全順列から作る重複なしk件の各学習後のpredictor保存重みと一致する組があること、step数がfloor(N/k)であることを照合。乱数アルゴリズムや期待乱数列は写さない |
| creditの上限・繰越 | N=3,k=2で3 intake後の成功回数3と保存credit=1.5を確認。他の3条件ではcredit=0。小さいNのWARNは構築から3 intakeを通して1件 |
| warm-up | N=2/4、閾値4観測でαとboolを確認。facadeは閾値8・ratio=0.5・k=2で8 intakeを流し、5 intake目は0 step、6回目から1 stepになることと、7回目のloss保持を確認 |
| 保存再開 | N=2,k=1,ratio=0.75、credit=0.5・1成功stepで保存。復元直後の成功回数、以降8 intakeのcount・loss・error_step・bonus_meanが継続実行と一致 |
| 不正設定・書式 | ratioの0・負・1超過・nan・inf、batchの0・負・非整数をenabled時だけ拒否。RNDの旧版・未知版をpayload読込前に拒否 |
| 既存回帰 | 関連スイートで固定教師、100回学習、統計の順序、独立Learn、CPU/CUDA、done/truncated、保存header、optimizer設定、正規化を継続確認 |

## レビュー

- 独立レビュー1回目: 計画OK、実装は初期化順序1件で差し戻し（原文 `.scratch/prd116/review-20261107-0314.md`）。抽選RNGを有効時だけ生成する条件を保ち、既存seed・probe RNGの初期化群へ移動した。推定器・診断の初期化は指定どおりRuntimeVars生成後・RewardScaler生成前に保持する。検証後、同じレビューエージェントへ再提出する。
- 修正後のDebug全体ビルドと関連28 tests /1293 assertions成功、終了0（`schedule-review-build.log`・`schedule-review-test.log`）。検証節と同じコマンドでログ名だけを変更した。静的検査も成功（`schedule-review-static.log`）。未決0として同じエージェントへ再提出する。
- 独立レビュー2回目: 計画・実装ともOK。不足・範囲外の新規検出・開発者への確認事項はなし。初期化位置、生成条件、検証記録、翻訳済み範囲を重点再確認した（原文 `.scratch/prd116/review-20261107-0323.md`）。根拠不足で見送った指摘はない。

## 完了報告

- 実施内容: creditによる刻みと非重複抽選、全件Learnのbool結果、観測件数warm-up、loss保持、成功回数、保存再開を実装。設定・212・200日英のRND節・READMEを更新した。
- 変更ファイル: 主な変更に記した16ファイル。PRD・別作業の200日本語版2章など・既存stageは保持し、ステージ・commit・pushは行っていない。
- 検証: 最終Debug全体ビルド、関連28 tests /1293 assertions、出荷設定11 tests /2583 assertionsが成功。設定7種類は意図した28差分のみ。既定葉検査は既存Atari fps2件のみ検出、今回の新規違反0。静的検査成功。
- レビュー結果: 2往復目で計画・実装ともOK。RNG初期化の位置を修正し、生成条件・named seed・推定器と診断の位置を保持した。見送った指摘なし。
- 課題: 索引issueのai_judgeは実装済み、ai_judgedを2026-11-07へ更新。対応内容を追記し、INDEXとwikiの16派生区画を再計算した結果は現状と一致した。status・監理の項目は変更していない。
- 決まらなかったこと・判断してほしいこと・新規の起票候補: なし。
- 未実施: 無効時4環境checksumはRunの切れ目まで保留、性能測定はユーザー指示で実施しない。smokeは不要。200の別作業による未訳の2章と関連参照は今回の翻訳範囲外で、未完了として表示を保持する。
