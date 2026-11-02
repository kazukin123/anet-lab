# TempoRL の skip 側の NN 実行設定 実装メモ

## 概要

正本は [PRD106](106_temporl_skip_forward_noise_key_10prd.md)。Actor ごとの `skip.nn_forward_mode=shared|dedicated` と `skip.nn_forward_key` を導入する。既定は `shared`。Q 側、Learner、seed、時計、forward の回数と順序を維持する。2026-10-29 にユーザーが対象ファイルを確認した。

基準コミット: `8213f2f52aef349cedf939b9fe174e70860cafe4`。着手前からある `.gitignore`、`CONTEXT.md`、PRD104 の計画、wiki などの変更は保持し、今回の差分に含めない。

## 主な変更

- `core/anet-core/include/anet/agent.hpp`: `DQNActorConfig.skip` にモードと省略可能な参照キーを加える。公開ヘッダの変更は PRD §5.6 の範囲。
- `core/anet-core/include/anet/default_dqn_agent.hpp`: モードを常に検証し、明示された参照は skip 無効・shared でも検証する。skip 有効・dedicated で参照が無い全 Actor をまとめてエラーにする。skip 有効・shared の明示参照だけ Actor ごとに一度 WARN。
- `core/anet-core/src/default_dqn_agent.cpp`、`dqn_based_agent.hpp/.cpp`: 生成時に実行設定を選び、`SkipActorResources` で搬送して既存の skip executor を準備する。不透明な実行設定は Agent の Resource。通常の forward 経路に設定検索、同期転送、追加の forward は足さない。
- `core/anet-core/src/dqn_based_agent_test.cpp`: 設定、ノイズの設定選択、τ に依存する公開特徴量を持つ IQN の専用 skip 方策を検証。資源を直接組む既存テストへ実行設定を渡す。
- `apps/runner/config/agent.txt`: `[train]` と `@eval_base` の `skip.max_length_mode` の隣へ `skip.nn_forward_mode ?= shared` を加える。Env 設定は変えない。
- `docs/design/200_dqn_agents.jp.md`: §6.5、§7.3 とテストの章へ追随する。英訳は PRD §5.4 に従い翻訳工程へ渡す。用語は既存 `CONTEXT.md` の NN 実行設定と一致し、追加更新不要。新しい ADR は不要。
- `core/anet-core/testdata/prd106/`: 独立した baseline 採取、7 構成比較、8 入力の設定比較、評価レシピと集計、および README を置く。過去の PRD の凍結資材・Run artifact は書き換えない。

## テスト

- Public interface / surface: `DefaultDQNAgentConfig(ConfigData)`、`DefaultDQNAgent::CreateActor()`、Actor の `MakeAction()` と aux `q_values` / `skip_q_values`、NN executor の公開 `Forward()`、公開 ConfigManager resolver、Runner の config dump / scalar / episode trace。
- 優先 behavior と TDD 順序:
  1. tracer bullet: 宣言済み dedicated キーを設定から読み、実際の Actor の skip 値まで設定が届くことを一つのテストで確認。RED を確認後、設定搬送と executor 選択の最小実装で GREEN。
  2. 未知モードは skip 有無によらず拒否。次のテストで RED を確認し、値域検証を追加して GREEN。
  3. 明示された未知参照は全モード・skip 有無で拒否。RED→GREEN。
  4. 有効・dedicated の欠落 Actor をすべて報告し、無効なら許す。RED→GREEN。
  5. 有効・shared の明示キーだけ一度 WARN、無効なら WARN なし。RED→GREEN。
  6. Noisy な Q・skip Head の構成で固定入力を二度 forward し、shared の追従、dedicated の独立、Q 側の不変を確認。同じキーの dedicated と shared の数値一致も確認する。足りない behavior ごとにテストを追加し、GREEN 後だけ整理する。
  7. IQN の `skip_context` を τ 融合出力に bind し、Q random / skip fixed と逆の組で、同じ行動の lane の aux 値を確認。既存経路が通れば修正しない。fixed/fixed と shared fixed の一致も確認し、不一致なら τ 評価の基準 G' を用意する。経路を直した場合、実装前の τ の評価は読みに使わず、PRD §6.7 に従って取り直す。
- テストのための本体 API、private へのアクセス、追加の診断指標は作らない。必要な観測は test-only の部品と既存公開経路で組む。

## 検証

1. コード・設定変更前に `.scratch/prd106/` へ Debug の Runner、DLL、test 実行体、Release 実行体、設定一式と SHA256 を保存する。既存保存先は上書きしない。τ 評価の旧ビルドと既存 trace は別の組として記録する。
2. VsDevCmd 経由で Debug ビルド、behavior ごとのテスト、および `[prd098]` / `[prd103]` / `[prd104]` と関連設定の回帰を実行する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

3. PRD §7 受入1 の7構成を、保存した同じ入力で before/after に実行する。8 lane、seed1、8704 experience、FP32、決定論、評価停止。`90_perf/`・`91_resource/`・`92_resource/` 以外の全 scalar 点を完全一致比較する。
4. 同じ resolver で4 env×既定/skipの8入力を比較する。許す差は宣言済み Actor の `skip.nn_forward_mode=shared` の追加だけ。既存キーの変更・削除と `skip.nn_forward_key` の出現は許さない。`check_default_leaves.py` も実行し、既存指摘と区別する。
5. Release をビルドし、PRD §7 の8条件の評価を実行する。分割は評価 seed1/2 各200ゲーム、合算は seed1 の200ゲームから。使用する実行体・checkpoint・解決設定と trace 200件を照合する。既存 τ の結果は受入3を通り、同じ旧ビルドの組を確認できた場合だけ再利用する。
6. 集中面、通過率、全体突破率、面別脱落、平均点・判断率・平均長、h/k/k' と和・差、G4 の差を PRD §6.11〜19 の10,000回の百分位 bootstrap で計算する。集中面は元の G（必要なら G'）で固定。比の分母が非正の抽選は除き、割合を記録する。合算は規定に当たる場合だけ seed2 を足す。`single` の既存値と固定の寄与の目安も併記する。
7. 実験の正本に結果と使用資材を残し、G4 の判断に必要な数値をユーザーへ示す。G4 は数値を見たユーザーの選択と、その後の段取りの承認まで開始しない。
8. `docs/quality/impl.md` の全項目を適用範囲別に自己点検し、計画・実装を独立した `an-review` へ出す。指摘をまとめて修正し、同じ reviewer へ再依頼する。raw 応答と実行結果を保存する。

## 前提

- 未決事項監査: ユーザー判断が必要な実装ブロッカー0件。D1〜D9、§5、§6、§7 を実装契約とする。
- repo evidence: `DefaultDQNAgentConfig` の既存参照検証、`DefaultDQNAgent::CreateActor()`、`Actor` の専用 skip seed と時計、既存 dedicated τ の追加と読み取り、PRD103/104 の検証資材に沿う。
- `shared` は executor・乱数状態を共有する意味ではなく、同じカタログ設定を選ぶ。`dedicated` で Q と同じキーを指定すれば現状と同一。
- モード列挙とキー参照は常に検証する。参照の必須と unused WARN だけ skip 有効時に限定する。エラー・WARN の文言は PRD §6.3 に従う。
- 学習の既定変更、Learner skip forward の分離、`skip.policy` のモード化、Greedy 型の規則変更、新しい診断指標は PRD 範囲外。
- G4 の学習を開始するかはステージ0後の判断事項であり、実装の未決とは扱わない。
- コミット・push は行わない。

## 経過

- 2026-10-29: 対象確認、現行コード・関連設計・ADR0048/0051・用語と検証資材の照合を実施。実装品質確認観点の実装計画と編集種類別の規約を確認。計画を保存。
- baseline は141ファイルを保存。manifest の基準は着手前HEADと一致し、test実行体は19:33:35のもの。7構成の before を開始。
- tracer RED: Debugビルド後、`anet-core-test.exe '[prd106-forward]' --out .scratch/prd106/tracer-red.log` が1 case / 3 assertions中、skip値の変化を期待する1 assertionで失敗（Q値は一致）。設定搬送とskip executorへの適用を実装。
- tracer GREEN: 同じフィルタで `.scratch/prd106/tracer-green.log` に3 assertions / 1 case成功。mode REDは2 assertionsで例外なしの失敗を確認し、列挙検証後の `[prd106]` は `.scratch/prd106/mode-green.log` に成功。
- `./.venv/Scripts/python.exe core/anet-core/testdata/prd106/verify_config.py`: 8入力すべて unexpected=0。既存値の変更・削除なし、全Actorがshared、既定のskip.nn_forward_keyなし。証跡 `.scratch/prd106/config/comparison.json`。
- reference REDは4通りとも例外なしで失敗。既存 `ValidateForwardReference` の再利用後、`reference-green.log` は9 assertions / 3 cases成功。
- required REDは無効状態が成功、有効状態が例外なしで失敗。全Actorの欠落を集めた後、`required-green.log` は11 assertions / 4 cases成功。
- beforeの7構成すべてexit0。正規化点数は順に26946 / 24399 / 23185 / 31152 / 38349 / 34628 / 34545。実行記録とhashは `.scratch/prd106/validation/before_*/` と `before-before-summary.json`。
- `check_default_leaves.py`: 29800 assignments / 3724 audited / errors2。既存のAtari.txtの `app.online.eval_panel.fps` と `app.online.train_panel.fps` だけ（旧PRD103 READMEにも同じ2指摘）。今回追加行の指摘なし。
- 集計の単体テストも1 behaviorごとにRED→GREEN。面8の数え方・集中面除外と、害の比と全体突破率の差を別々に計算する2 tests成功。bootstrapの乱数seedは106に固定する（計算の再現用で、評価のseedとは別）。
- 独立τ抽選の不足3本を旧ビルド（Release SHA256 `dac16bef2c508724e9391a385c90d7d6ffa8511c76fc1d294bb4965640528b37`）で開始。入力は `.scratch/evaldiag_noise_tau/config` の凍結設定、実行体は今回保存したbaseline-release-bin。採用はτの経路テスト成功後。
- warning REDは12 assertions中、必要な2 WARNが無いため失敗。追加後の `warning-green.log` は23 assertions / 5 cases成功。
- Debug全体ビルド成功（VsDevCmd経由 `cmake --build --preset x64-Debug --parallel 4`）。Release Runnerも成功（`--preset x64-Release --target AnetRLRunner --parallel 2`）。
- Noisy選択はQの2キー×skipの3選択×clone有無の12通りで36 assertions成功。τテストは最初のfixtureのReplay容量不足だけを修正し、`tau-path.log` に10 assertions / 1 case成功。τの問い合わせ・読み取りの本体コードは変更していないため、実装前のτ評価を採用する。
- 旧ビルドのTB_indepは3本すべてexit0・200件。`analysis/initial-tau.json` の暫定集計で合算τの区間が0.5を含み、規則どおり旧ビルドで合算のG/TQ/TS/TB/TB_indepのseed2を追加開始。ノイズ側の評価は新Releaseと設定を固定して開始。
- 結論の保存先はPRD §7が実験記録と指定済み。追加の保存先選択は不要。記録後は `anet-ingest-wiki` を適用する。判断率・平均長はseed別のsession scalarを併記し、分母が分からない平均どうしを単純平均しない。
- 単独skip forwardはtest-only Policyで行動選択を通さず、観測・公開特徴量・行動を固定してmu-only/sampleを検証。factoryではQ2キー×skip3選択×clone有無に加え、sharedと同じキーのdedicatedの値を2回ずつ比較した。別Agentの比較は `torch::manual_seed(106)` で初期重みも揃える（Actor seedとは別）。
- `anet-core-test.exe '[prd106]' --success --out .scratch/prd106/contracts-seeded.log`: 78 assertions / 8 cases成功。`shared_vs_dedicated_fixed_grid_equal=1` を記録。G'は不要。τの既存経路は変更なし。
- `anet-core-test.exe '[prd098],[prd103],[prd104],[prd106]' --rng-seed 106 --out .scratch/prd106/regressions.log`: 20164 assertions / 62 cases成功。
- Pythonの数値テストは4 tests成功。面8の失敗、集中面、hとG4差の分母の区別、分母0時に比を作らないこと、集中面を除いてもG4の全体差が変わらないことを確認。
- 前後比較の7構成すべてexact_match=trueを確定。新しい既定設定の実効dumpもGridMazeとAtariで確認した。
- 自己点検: `docs/quality/impl.md` の全項目を適用範囲別に確認。既存8ファイルのBOM保持、LF、共有契約の非依存、productionのdynamic_cast/TODO不在、diff --checkを確認。設定監査の既存2指摘は上記のとおり。旧契約の廃止・互換層・GetScalar変更・skill変更・新規productionファイル・新しい借用・同期転送はなく、それらの個別観点は対象外。
- 自己点検後の実装判断の未決なし。G3/受入5は評価が進行中なので未完了として残し、評価完了後に結果の自己点検とレビューを追加する。現時点の計画・実装・検証資材を独立an-reviewへ依頼する。
- レビュー1回目: 計画OK、実装は追加テストのTensorDict鍵の定数化で差し戻し。原文 `.scratch/prd106/review-20261029-2227.md`。追加範囲だけを既存の `DQNActionInfo::kSkipQValues`、`BatchActionInfo::kKeyDecision`、`nn::kKey_DefaultOutput` とtest-localなQ値の鍵定数へ置き換え、再ビルドする。
- 見送った指摘はなし。レビューの範囲外の検出（既存テストの同種の直書き）は下の候補へ積み、既存テストの無関係な行は変更しない。
- レビュー1の修正後、Debugのtest targetを再ビルドし、`anet-core-test.exe '[prd106]' --out .scratch/prd106/contracts-review1.log` は78 assertions / 8 cases成功。本体・評価の凍結実行体の変更なし。
- 旧ビルドのτは分割・合算ともseed1/2の400ゲームが揃った。分割の全体突破率差 `R_TQ−R_TB` は+7.5ポイント [0.75, 14.5]、合算は+10.25ポイント [4.5, 16.25]。kの主因の読みは両方とも「決めきれない」。新ビルドの分割のノイズは400ゲームが揃い、`R_NQ−R_NB` は+18.5ポイント [11.5, 25.0]、hの読みは「決めきれない」。合算のノイズはまだ実行中。暫定集計は `analysis/progress-2236.json` と `analysis/review1-audit.json`。
- 実験記録へ、全8条件のNN設定・τの共有と独立、ε=0、batch16・200ゲーム、環境条件を明記する。学習元2本の保存configは元Runの `config/config_data.txt` と一致。wikiの起票時点の実行設定の記述を過去形へ是正し、現在の指定は設計文書へ案内した。他会話のPRD105のwriteback行は保持した。
- `analysis/protocol-audit-2303.json`: 合算ノイズの初回200ゲームも完了。h_Q=0.667 [0.416, 0.998]で0.5を含むため、規則に従い新ビルドの合算G/NQ/NS/NBへseed2を各200ゲーム追加した。初回の全体突破率差は+8.00 [-1.00, 17.00]ポイント。追加完了前の値を最終判定に使わない。
- 集計の採用時に、全Runの実効dumpからseed・保存無効・ε=0・batch16・200ゲーム・J=10・frame skip4・sticky0.25・noop30・上限108,000 framesを照合する検査を追加。新Runの実効checkpointパスとSHA256も照合。上の暫定集計とPythonの数値4 testsが成功。
- [実験記録](../experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md)を暫定値で保存し、合算ノイズのseed2追加中と明示。本文を読み直し、SHA256・mtime・Git差分を確認した。完了済みの分割ノイズ・両checkpointのτをwikiの4ページと目録に取り込んだ。出典リンクの欠落0、リンク切れ0、公開できない値の検出0。合算ノイズ完了後、記録と取り込みログを最終値へ更新する。
- 全評価完了後の自己点検: 36 Run・7200 trace、pendingと追加要求はなし、h/k/k'の分母から除いた回はすべて0%。出典・相対リンク・LF・目録4ページの掲載・公開できない値を再確認した。実験記録の最終blob `969f63c842fc9ab82db134d7dc0dc8716ffbc62b` とwikiの最後のingest行が一致。本文・hash・mtime・Git差分を確認し、末尾空行も是正した。
- レビュー2回目: 前回と同じan-reviewに続けて依頼し、計画・実装ともOK。原文 `.scratch/prd106/review-20261029-2343.md`、差分 `.scratch/prd106/diff-20261029-2339/`。指摘の見送りなし、新しい範囲外の検出・開発者への確認事項なし。G3/受入5、集計・実験記録・wikiの整合、G4を未開始とする扱いまで確認された。

## 検証結果・レビュー結果

- 受入1: `./.venv/Scripts/python.exe core/anet-core/testdata/prd106/run_validation.py compare --label after`、7構成すべてexact_match=true。点数は26946 / 24399 / 23185 / 31152 / 38349 / 34628 / 34545。各 `.scratch/prd106/validation/after_*/comparison.json` にSHA256を記録。
- 受入2・3: `[prd106]` の78 assertions / 8 cases成功。固定τの格子は一致、G'不要。関連回帰は20164 assertions / 62 cases成功。実行コマンドとログは「経過」のとおり。
- 受入4: 設定8入力の全キー比較でunexpected=0、`verify_dump.py` もexit0。現行設定のGridMaze 512 experience（exit0）と完了したAtari Gの実効dumpで全Actorのshared・skipキーなしを確認。Atariはgreedyも含む。
- 受入5・G3: 評価・集計・記録は完了。2checkpointともノイズ4条件とτ5条件を各400ゲームで計算した。Gをビルド間で混ぜず、全36 Runの53_evalg/episode各200件（計7200件）・実効設定・checkpointを照合。集中面は分割6・合算5。h・k・k'は両checkpointとも「決めきれない」。分母の区間はすべて正で、比から除いた回は0%。`analysis/final-2329.json` のpendingとadditional_seed2はすべて空。実験記録とwiki4ページ・目録・取り込みログへの反映も完了し、独立レビューで最終確認する。
- G4: ステージ0の数値提示後のユーザー判断待ちになる。学習は開始していない。
- ビルド: Debug全体とRelease Runner成功。レビュー1の修正後のtest targetも成功。最終の自己点検は、実装品質確認観点の全項目を適用範囲別に再確認した。productionは前回以降変更なし。追加・更新文書の出典、相対リンク、公開できない値、LF、Git差分を確認。独立レビューは2往復で計画・実装ともOK。

### ステージ0の最終値

差はskip側だけを切った全体8面突破率の差で、単位はポイント、括弧は95%区間。

| checkpoint | ノイズ R_NQ−R_NB | τ R_TQ−R_TB |
|---|---:|---:|
| 分割 | +18.50 [11.50, 25.00] | +7.50 [0.75, 14.50] |
| 合算 | +16.50 [10.00, 22.75] | +10.25 [4.50, 16.25] |

全体突破率差の区間は4つとも正。ただし主因の比のQ側の区間が0.5を含み、PRDの主因の読みはすべて「決めきれない」。学習への効果はまだ確かめていない。既存singleとfullの損の差は、分割・合算とも約−0.25ポイントで、ビルドと採取時点が違う点推定の目安として記録した。

評価・集計の実行コマンド（すべてリポジトリの `.venv` のPython）:

- `core/anet-core/testdata/prd106/run_evaluation.py run --build old --conditions TB_indep`: 分割seed1/2・合算seed1の不足3本、すべてexit0。
- `core/anet-core/testdata/prd106/run_evaluation.py run --build old --checkpoints mrg --conditions G TQ TS TB TB_indep --seed 2 --jobs 2`: 合算τの追加5本、すべてexit0。
- `core/anet-core/testdata/prd106/run_evaluation.py run --build new --conditions G NQ NS NB --jobs 2`: ノイズの初回12本、すべてexit0。
- `core/anet-core/testdata/prd106/run_evaluation.py run --build new --checkpoints mrg --conditions G NQ NS NB --seed 2 --jobs 2`: 合算ノイズの追加4本、すべてexit0。
- `core/anet-core/testdata/prd106/analyze_evaluation.py --label final-2329`: exit0、10,000回のbootstrap、追加要求なし。旧ビルドの既存12本はPRD §10で指定されたものだけを再利用した。

最終値の正本は[実験記録](../experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md)。実行体・入力は `evaluation-runtime.json`・各 `evaluation/*/recipe.json` と `execution.json`、採用metricsのSHA256は集計JSONに保存。新Release SHA256は `2c7d6ea4e5b0d443e997fb9f72b174bbab30a5f4005a4ed442c42ae211d7e267`、旧Releaseは `dac16bef2c508724e9391a385c90d7d6ffa8511c76fc1d294bb4965640528b37`。

## 決まらなかったこと

G4 はステージ0の結果を提示後、ユーザーが選ぶ。

起票候補（まだ起票しない）: 既存のDQNテストに残るTensorDict鍵の直書きを契約定数へ揃える。レビュー1で見つかった例は `dqn_based_agent_test.cpp` の旧2950・2957・2965行のdecision、旧3079行のq_values。今回の実装の合否には影響しない。質問していない（今回の変更の範囲外としてレビューで検出し、工程末の候補に貯めた）。工程末に提示し、ユーザーが選んだ場合だけ課題として記録する。

## 実装・ステージ0の到達点

- 実施内容: 保存済み計画に沿う実装、受入1〜5、自己点検、独立レビューを完了。
- 変更ファイル: 「主な変更」の実装・設定・設計文書・検証資材と、本メモ、実験記録、wiki4ページ・目録・取り込みログ。別会話の変更は保持した。
- レビュー結果: 2往復で計画・実装ともOK。初回の追加テストの鍵の定数化は修正・再検証済み。見送った指摘なし。再確認点は、主因の比とG4の全体差の区別、同じビルドのGの使用、評価から学習への効果を推測しないこと。
- 決まらなかったこと: 工程末の起票候補1件（上記、見送り推奨）。まだ起票していない。
- 判断してほしいこと: 候補の採否。その後に、提示済みのステージ0の数値を見てG4を流すか選ぶ。学習は、選択後の段取りの承認まで始めない。
- 次にできること: 候補の採否を受けて、選ばれた場合だけ記録し、G4の判断へ進む。コミット・pushはしていない。

## 追加レビューと指摘の扱い（2026-10-30）

- cc-review: Codexは計画・実装ともOK。Claudeは計画OK、実装は今回専用の検証Pythonの段階説明コメント不足1件で差し戻し。原文は `.scratch/prd106/review-20261030-0026.md` と `.scratch/prd106/review-20261030-0026-claude.md`。
- ユーザー判断: 今回専用のPythonはソースを読む運用ではないため、Claudeが求めたコメント追加は見送る。対象は今回の指摘に挙がった検証Pythonであり、一般の実装規約を変更するものではない。
- Claudeの原文判定は保持し、指摘への対応だけをユーザー判断で見送った。本体・検証Python・実験結果は変更していない。
