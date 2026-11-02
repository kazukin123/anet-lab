# PRD097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査

> 2026-10-20 起票（暫定。グリル前）。PRD097工程末の候補①をユーザーが採用した。
> native値所有・要求時Tensor化は22implで実装済みだが、修正後の速度は未測定。

## 背景と問題

[PRD097](097_env_auto_reset_after_notification_10prd.md)の再実装では、全7構成728534点のscalar/trace一致、ビルド・テスト・手動画面確認を終えた。性能はLunarLanderが合格し、下表の4構成はbeforeを5反復してもspread（最大/最小−1）が2%を超えたため判定不能となった。改訂PRD§7受入4に従って性能合否から除外し、本課題へ持ち越す。合格への読み替えは行わない。

| 構成 | before中央値 exp/s | after中央値 exp/s | after変化 | before spread |
|---|---:|---:|---:|---:|
| gridmaze | 17201.179 | 17436.845 | +1.3701% | 5.1521% |
| dropmerge | 3351.936 | 3314.311 | -1.1225% | 2.6296% |
| dropmerge_serial | 2889.554 | 2689.727 | -6.9155% | 19.2386% |
| atari1 | 4978.385 | 4973.180 | -0.1045% | 12.0979% |

各Runは速度の最終報告stepの半分を超える区間の中央値、構成値は5反復の中央値。総実行時間は指標に使わない。DropMerge serialの中央値6.9155%低下は、beforeの大きなばらつきと別に調査対象として残す。

## 分かっていることと証拠

- 正式性能46Runはseed=1・512000経験・厳密な決定的backend・同期評価で正常終了し、凍結実行体と引数を監査済み。4構成ともbefore全5本の時間/資源以外のscalar/trace全点が完全一致した。数値系列の差による速度差ではない。
- Gridの既存NVTX診断ではEnv/学習の重なりが保たれ、WARN量の急増も見られない。正式反復のばらつきの原因は未確定。
- Dropの時間限定NVTX診断で、結果が各laneのauxを所有するため描画Tensor生成が増えた。pipelineのRoot中央値80.408→80.195ms、Env batch平均20.978→21.951ms、Env/学習の重なり99.766→99.788%。serialのRoot中央値83.167→85.176ms、Env batch平均20.806→21.964ms。追加費用は観測されたが、正式serial中央値6.9%低下の全原因は説明できない。
- 診断は各1本・90秒待ち/30秒採取で、同一step範囲ではなく同期評価も含む。inclusive時間をwallへ加算できない。最初のbefore診断はアプリ終了コード未捕捉（Nsight終了0・通常完走ログのみ）。正式性能とは別保存である。
- Atariも同じbefore実行体・設定・全数値系列でspread12.10%。正式反復の資源時系列を取得しておらず、clock/温度/cold cache等へ原因を断定できない。
- 詳細と実行体・設定・ハッシュ・除外Runの経緯は[実装記録](097_env_auto_reset_after_notification_21impl.md)の「最終受入集計」を参照。
- ローカル証拠: `.scratch/prd097/implementation-20261020/protocol-fixed/after-d7-v2/performance.json`、`.scratch/prd097/reimplementation-20261020/v2-performance-execution-audit.json`、`v2-before-performance-series-final.json`、`diagnosis-drop-20261020-213445/analysis.json`。後3件は同じreimplementation-20261020配下。

## 22implでの関連変更

Claudeレビュー指摘①に従い、DropMergeの結果をnative値所有・GetAuxData要求時Tensor化へ変更した。公開aux/scalar・保持結果の寿命を回帰し、別世代after-d7-v3で全7構成728534点の数値系列一致を確認した。性能再測定・追加プロファイル採取はユーザー指定で実施しない。上表は変更前のv2測定値であり、今回の高速化効果ではない。4構成の判定不能とserial追加費用の原因調査は未解決のまま残す。詳細は[22impl](097_env_auto_reset_after_notification_22impl.md)。

## 取り組むこと

1. 保存済みの正式条件と資材を保持し、4構成のばらつきを再現・調査する。資源時系列、外部負荷、実行順等を追い、原因が分かる前に測定値の選別や除外を増やさない。
2. 条件を安定させたうえで、同じbefore/after交互手順・指標・3組/不安定時5組・2%基準で再測定する。条件を変更する場合は内容と理由を記録し、両側へ同じ条件を適用する。
3. 22implのnative値所有・要求時Tensor化を含む別世代を使い、DropMerge serialの追加費用がどこまで解消したか再測定する。残る費用を切り分け、追加変更が必要なら結果不変性、GetScalar/aux/Viewの契約、統計式と浮動小数点演算順序を守る。

## 完了の目安と未決

- 4構成それぞれに、再現条件・生データ・資源観測・ばらつきの調査結果と再測定結果を残す。安定した比較ではbefore spreadとafter/before比を示す。
- 安定させられない構成は、原因と限界、次の調査を明示して扱いをユーザーへ返す。判定不能を合格へ置き換えない。
- native値所有・要求時Tensor化の性能効果、Dropの追加最適化の要否と方式、ばらつきを抑える具体策は未決。課題の記録は承認済みであり、これらの方式の採用決定ではない。
