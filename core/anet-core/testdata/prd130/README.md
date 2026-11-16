# PRD130 の検証資材

正本は [PRD130](../../../../docs/memo/130_rnd_estimate_statistics_perf_10prd.md)、実施結果は [実装メモ](../../../../docs/memo/130_rnd_estimate_statistics_perf_20impl.md) に記録する。

## 今回の範囲

ステージ 1・2 の実装、機能・性能受入 1〜15、比較基準の保存、日英の文書の追随をすべて終えた。最終の主・副・Taiga の性能比較は合格。測定は終了し、評価 Run を再開できることを開発者へ通知した。評価 Run の PAUSE・再開操作は行っていない。詳細は実装メモの末尾を参照する。

## 凍結した入力と実行体

`run_validation.py` の出力先は `.scratch/prd130/acceptance-cuda/`。`before/package/` に変更前の Debug・Release Runner と core テスト、設定とソース、ハッシュを保存する。更新しない外部 DLL はハードリンク、実行体と設定はコピーする。今回の変更前のパッケージは `acceptance/before/package/` から複製した。

初回に試した CPU 指定の未完了 Run は `.scratch/prd130/acceptance/before/` に保持し、比較へ混ぜない。

`prepare` は保存済みの LunarLander の既定連鎖を読み、無効・累積・EMA・variance_feedback・collect_errors・CUDA 再現用の 2 本の設定を作る。seed 1、決定的 backend、100k experience、CUDA、torch thread 1。評価と描画・画像出力を止め、終了時 checkpoint を保存する。Q の lane・batch・warm-up・ネットワークの出荷値は縮めない。variance_feedback は保存済み Atari の Taiga profile の 8 値を明示する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd130/run_validation.py prepare
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd130/run_validation.py run --label before
```

既存の入力・結果は上書きしない。`run --case` で選んだ条件だけを実行でき、独立した条件の失敗後も後続の条件は継続する。各 Run の process.log・終了値・設定ハッシュ・保存先を残す。

子プロセスには `ANET_PRD` で始まる検証用の一時スイッチを継承させない。実際に渡した環境のハッシュ、バックエンドに関係する環境変数の値、除外したキー名を `inputs.json` へ記録する。環境全体の値は、秘密の値を写さないため保存しない。

現用の採取器は `run_p130_` で始まる Run 名を使う。比較と checkpoint の参照は Run 名を推測せず、各条件の `inputs.json` が記録したパスを使う。初回の採取で既に作った `p130_` の Run は、過去の証拠として改名せず保持する。

## 単体・結合テスト

`[prd130]` は、Factory を通す統計更新専用の口、混合辞書、robust 更新、Load・Reset・device 切り替えの正規化、独立した倍精度の集計の参照値、CPU と CUDA の RND の観測統計と α を確認する。

`[prd116]`・`[prd123]`・`[prd127]`・`[scaler]`・`[shipped_config]` と core 既定 suite も実行する。詳細出力は `.scratch/prd130/acceptance/` のログと junit XML へ保存する。

## 前後比較と継続

変更後の Runner をビルドしてから `snapshot --label stage1` で別のパッケージを保存し、`run --label stage1` で同じ凍結入力を使う。無効と variance_feedback を `compare` で比較する。全 scalar 点を正規化し、時間・資源の 3 prefix だけを除く。変更前対ステージ 1 の variance_feedback だけは `56_agent_in/14_target_feature_var` も除く。

`repeat` は同じ label の CUDA の 2 本を比べ、除くのは時間と資源だけ。変更前から再現しない場合だけ、PRD §7 の 6 の扱いへ戻る。`load --label stage1 --reference before --case cumulative ema feedback` は変更前の 100k 終了 checkpoint を同じ構成で読み、続きの 100k の完走を確認する。ステージ 1 では続きの値の同一性を求めない。

## 性能測定

`run_performance.py` は PRD §7 の 1〜3 を扱う。`prepare` は機能検証で保存した before/stage1 の Release 実行体のハッシュを検査し、before の設定を新しい測定先へコピーする。コピーから可塑性メトリクスの定義を除き、実効設定と登録ログでも計算が無いことを確認する。出荷設定の原本は変えない。

`measure --set main` と `legacy` は、変更後の同じ実行体で off/on/on/off を 3 巡測る。`grid` は RND 有効・CPU・256 lane を before/stage1/stage1/before で 3 巡測る。各組の pilot で最速群が約 20 秒になる経験数を決め、全群で固定する。Q warm-up は 20,000、RND warm-up は 12,800、計測開始は 32,768 experience、torch thread は 1。評価・描画・画像と終了時 checkpoint を止める。

`plan.json` は実行体・設定のハッシュ、可塑性メトリクスの除去内容、休止済み PID、測定順序を保存する。`cases/` の各条件には設定・コマンド・実行体と環境のハッシュ・終了値・時間・Run のパスを保存する。`measurements_*.json` は順序付きの 12 本、`summary_*.json` は中央値・幅と合否である。時間比較へ Tracy 付きの Run を混ぜない。不正な順序・経験数・時間・採取付きの記録を受入へ使わない。

利用者が評価 Run の PAUSE 完了を明示してから測る。`--paused-pid` と対応する `--paused-run` を同じ順で指定し、休止済み Run だけを共存させる。Run 全体を通じて、そのログとメトリクスのサイズ・更新時刻が変わらないことを要求する。同じプロセスの HANDLE を保持し、`GetProcessTimes` の kernel＋user 時間を約 1 秒ごとに読む。終了・API 失敗・欠測・非有限値・CPU の逆行は拒否する。終了後に、測定開始前と証明できる最後の時刻のサンプルを選び、そのサンプルから終了後までの CPU 増加が 1 秒以内であることを要求する。CPU の大小でサンプルを選ばない。全サンプル・選択した始点・終端 CPU・全プロセス区間の CPU 増加を保存する。ほかの Runner は拒否する。build・test も重ねない。既存の測定先は上書きしない。設定を直した組は新しい測定先で採り直す。同条件の trace の再採取は、未使用のケース名を `execute` へ渡し、別のケースへ保存してもよい。

`profile` は本測定とは別の Run を起動し、warm-up のあとで Tracy を 8 秒採る。port 18086 を使い、休止中の利用者の Runner には接続しない。GridMaze は前後の 2 本、それ以外は変更後の RND 有効を採る。各 `capture.tracy`、`zones_all.csv`、`events_*.csv` と `breakdown.json` を保存する。集計は CPU host の完結した区間を包含関係で揃えたもので、GPU kernel 時間ではない。Nsight Systems の再採取は転送・同期の原因が不明な場合に行い、CLI と解析は `nsight-systems` の手順で確認する。

PRD §7 の 13 は、以下のステージ 2 の性能比較で扱う。過去の stage 0 の採取結果を今回の受入として使わない。

### 2026-11-16 のステージ 1 の結果

| 確認 | 結果 | 合格条件 |
| --- | --- | --- |
| 主の RND あり／なし | 21.499999／19.852997 秒 = 1.082960 倍 | ≤1.15 |
| 副の RND あり／なし | 22.048002／20.089498 秒 = 1.097489 倍 | ≤1.25 |
| GridMaze の前後 | 20.613500 → 20.414000 秒、差 -0.199501 秒 | 後退しない、または基準の幅 0.520001 秒以内 |
| GridMaze の target 出力の batch 集計 | 1 回平均 0.356280 → 0.021336 ms、各 196 回 | 変更後が短い |
| 主の統計の更新 | `UpdateStats` があり、統計区間内の `Normalize` は無い | 捨てる正規化が無い |

主・副は `.scratch/prd130/performance-20261116-0612/`、GridMaze は `.scratch/prd130/performance-grid-r2/`。採用した 36 本の本測定はすべて正常終了し、休止済み PID の CPU 時間の増加の最大は 0.875 秒だった。主の採用 trace は `cases/main_profile_on_r2/`、GridMaze は `cases/grid_profile_before/` と `cases/grid_profile_stage1/`。

最初の GridMaze pilot は採取器の Atari 専用の評価タグが原因で構築に失敗し、回帰テストを足して直した。最初の主の trace は休止済み PID の CPU 時間の増加が 1 秒を超えて不採用とし、上限を変えずに別ケース名で採り直した。いずれも失敗記録を残し、受入の集計には混ぜていない。

集約した結果は `.scratch/prd130/acceptance/performance-evidence.json`。次の PowerShell は保存済みの結果を読むだけで、採取を行わない（実行確認済み）。

```bash
Get-Content -Raw -Encoding UTF8 .scratch/prd130/acceptance/performance-evidence.json | ConvertFrom-Json | Select-Object -ExpandProperty summaries | ConvertTo-Json -Depth 5
```

入口と引数は `run_performance.py --help` で確認できる（実行確認済み）。今回の呼び出しは `prepare --out .scratch/prd130/performance-20261116-0612 --paused-pid 53272`、同出力先での `measure --set main`・`measure --set legacy`、新出力先での `measure --set grid`、`profile`。休止済み PID は今回の実測対象のもので、将来の利用時に固定の PID として使わない。

ステージ 1 の Debug・Release・core テスト・設定・本体ソースは `.scratch/prd130/acceptance-cuda/stage1-accepted/package/` に保存し、同階層の `acceptance.json` から機能検証と性能の証拠を参照する。機能検証の stage1 と Runner のハッシュは一致する。保存時点では英語版設計文書の追随とステージ 2 は未完了だった。

## ステージ 2 の機能検証

比較元は `stage1` の 5 構成の 100k Run。最終ビルドを `snapshot --label stage2` で保存し、同じ入力の無効・累積・EMA・variance_feedback・collect_errors を `run --label stage2 --case off cumulative ema feedback collect` で実行する。`compare --reference stage1 --label stage2 --case off cumulative ema feedback collect` は、target_feature_var も含め、時間・資源以外の全 scalar の一致を求める。

checkpoint の比較では、`stage1` の累積・EMA・variance_feedback の 100k 終了 checkpoint を両側で読む。基準側は `stage1-reference` の実行体で `load --reference stage1 --label stage1-reference`、変更後は `stage2` で `load --reference stage1 --label stage2` を実行する。各々の `--case` は `cumulative ema feedback`。続く 100k を `compare-load --reference stage1-reference --label stage2 --case cumulative ema feedback` で比べる。除外は時間・資源だけで、通常 Run と異なる checkpoint 用の Run パスを `inputs.json` から読む。

単体テストは、混合 byte・float・離散観測の Clone・checkpoint とその後の採点・補助学習・読むだけの採点・診断の一致を CPU/CUDA・4 条件で確認する。採取器のテストは checkpoint の比較対象と target_feature_var の比較を保護する。並行した機能 Run の実時間は性能の受入へ使わない。今回の結果は実装メモの「ステージ 2 の経過」に記録する。

### 2026-11-16 のステージ 2 の機能検証結果

| 比較 | 一致した scalar 点 |
| --- | ---: |
| 通常: 無効 / 累積 / EMA / variance_feedback / collect_errors | 38,922 / 43,465 / 43,371 / 43,507 / 43,533 |
| 同じ stage1 checkpoint の続き: 累積 / EMA / variance_feedback | 43,393 / 43,351 / 43,525 |

全 8 比較で時間・資源以外の全 scalar が一致した。継続の基準側も含む全 Run が 100k 経験を正常終了し、実行体・設定・元 checkpoint の一致も確認した。集約は `.scratch/prd130/acceptance/stage2-evidence.json`、固定した変更後の実行体・設定・本体ソースは `acceptance-cuda/stage2/package/` にある。

最終 Debug・Release 全ビルド、関連 9,886 assertions と Python 7 tests は成功。core 既定 suite は、初回に workspace の履歴ファイル置換の例外 1 件があり、該当テスト単独の 4 assertions と、同じ seed の既定 suite 196,027 assertions の再試験が失敗 0。断定と本体は変えていない。skip 1 は CUDA 非搭載時だけの検査である。初回例外の原因は未特定で、最初の記録も保持した。詳細は実装メモと `acceptance/stage2-default.xml`・`stage2-workspace-repro.xml`・`stage2-default-r2.xml` を参照する。

機能検証の終了時点では性能比較 13 は未実施だった。後述の性能比較で完了した。この機能検証の実時間から速度を判定しない。

## ステージ 2 の性能比較

`prepare --stage 2` は、受入済みの `stage1-accepted/package/` と機能検証済みの `stage2/package/` の Release 実行体を選ぶ。各 manifest と実行体のハッシュを照合し、stage2 の `functional-acceptance.json` の 11・12・14・15 と Release ハッシュも一致を求める。既存の before の設定のコピーから可塑性メトリクスを除去し、元の出荷設定と保存基準は変えない。

`measure --stage 2 --set main`・`legacy`・`taiga` の 3 条件を別々に測る。どちらの実行体にも同じ RND 有効の設定を渡し、main は価値分解を持つ主の構成、legacy は副の既定 RND、taiga は副に `run.@rnd_taiga` を重ねた構成とする。Taiga の RND warm-up だけは測定条件の 12,800 件へ揃える。実効設定でも mode・倍率・観測・学習の刻み・β・Adam eps を確認する。

各条件の pilot 後に経験区間を固定し、stage1/stage2/stage2/stage1 を 3 巡、各群 6 本ずつ採る。中央値の差の絶対値が、stage1 の最小〜最大の幅以内なら受入を満たす。順序や区間の違い、採取付き Run、不正な時間は拒否する。測定中は build と test を重ねない。

休止検査の初版はプロセスの全生存期間の CPU を数えたため、長い Taiga の採取で、学習停止が維持されていても 1 秒を超える例が出た。区間別の診断 Run は受入に使わず、CPU 検査を実測の全区間を含む範囲へ揃え、全生存期間ではログ・メトリクスの停止を追加検査するようにした。診断の 0.375 秒は 2 サンプル間の増加で、実測全体を覆う証拠ではない。上限 1 秒と性能判定式・計測経験数は変更していない。先に完了した主・副は、より広い全生存期間の 1 秒検査を通過した記録として保持する。初版で拒否した Taiga の記録を再解釈して採用せず、同じ区間で一式を新たに採る。

公開時計の出力遅延だけでは CPU の読み取りが測定開始前かを判定できないので、時間でも包含を要求する。各 CPU 読み取り完了までの単調時計の経過秒数を、子プロセス起動前を原点に記録する。Runner の `elapse_hour` の原点は子の起動後なので、読取完了の秒数 + 0.01 秒が測定始点の Runner 経過秒数より短ければ、実際の測定開始前だったことを保守的に保証できる。終了は子の終了後に CPU を読む。0.01 秒は float の時計の丸めに対する余裕である。終端の経験数と全区間のファイル停止も併用し、包含を示せない記録は拒否する。1 回だけ CPU を読む版の 3 回目の採取は、初期化が長いケースで保守的な時刻条件に届かず停止したため、一式を不採用とした。複数サンプル版でも、この不採用記録を採り直した結果へ混ぜない。

### 2026-11-16 のステージ 2 の性能結果

| 条件 | stage1 中央値（秒） | stage2 中央値（秒） | 差（秒） | stage1 の幅（秒） | 結果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 主 | 19.905501 | 20.014501 | +0.109000 | 0.581997 | 合格 |
| 副 | 20.068000 | 19.957000 | -0.110999 | 0.692996 | 合格 |
| Taiga | 32.709499 | 32.590000 | -0.119499 | 1.121995 | 合格 |

採用は各 12 本、計 36 本。主・副は `.scratch/prd130/performance-stage2-20261116-0910/`、Taiga は `.scratch/prd130/performance-stage2-taiga-r4-20261116-1030/`。主・副の全プロセス CPU 最大は 0.84375・0.875 秒、Taiga の時間包含を確認した区間の CPU 最大は 0.500 秒。実行体・設定・実効値・終了値・正確な経験区間・順序・休止検査と集計を原記録と照合した。不採用 3 組と診断を集計へ混ぜていない。

集約は `.scratch/prd130/acceptance/stage2-performance-evidence.json`、全受入 1〜15 の参照先は `.scratch/prd130/acceptance-cuda/stage2/acceptance.json`。過去の機能検証だけの受入記録とステージ 1 の受入記録は当時の範囲を保つ。Python の性能採取器の保護テストは 16 件が成功し、CPU 読取 API の生存・終了判定も実検証した。詳細な失敗・修正・レビューの経過は実装メモにある。

2 モデルレビュー後に、RND の意図コメントと前方宣言の配置・空行を修正した。現用ソースは保存基準とハッシュが異なるが、計算処理のコード行とデータメンバは不変であることを `acceptance/cc-r2-source-check.json` で確認した。固定した性能測定用の実行体と記録は保持し、現用コードの Debug・Release 全ビルドと関連 9,886 assertions を再確認した。今回の検証ログは `acceptance/cc-r2-*`、修正と再レビューの詳細は実装メモにある。
