# Run分析ユーザーガイド

> 主たる観点: 行程単位（成果物確認、可視化、比較、解釈）

## 1. はじめに

### 1.1 目的

本書は、Runが出力した`metrics.jsonl`をMetrics Viewerで可視化し、複数Runを同じ条件で比較し、必要に応じてOptunaのmulti-seed集計を確認する手順を説明する。

### 1.2 対象読者

- 学習曲線、評価値、loss、性能指標を比較する利用者
- Run間の差を過大評価せず、同じstep範囲で判断したい利用者
- DropMerge Optunaのseed runとsummary studyを使い分ける利用者

### 1.3 記載範囲

現行のJava/Spring Metrics Viewer、Plotly画面、Run artifact、DropMerge Optunaの閲覧方法を扱う。
新しいmetricの実装方法やOptuna探索空間の変更は設計文書と[DropMerge Optuna利用ガイド](../optuna.md)を参照する。

## 2. Metrics Viewerを起動する

本書はMetrics Viewerのjarがbuild済みであることを前提とする。Java、Maven、jarのbuild手順は[開発環境](040_development_environment.jp.md)を参照する。

通常Runを見る場合は`apps/runner`から次を起動する。

```powershell
22_metrics_viewer_java.bat
```

- URL: `http://localhost:8082`
- workspace directory: `apps/runner/workspaces`

画面上部の`Workspace` selectorで、`runs/`または`config/`を持つworkspaceを切り替える。
前回選択はbrowserへ保存され、次回起動時に同じworkspaceが存在すれば復元される。
保存済みworkspaceが存在しない場合はserverのcurrent workspaceへ戻る。

独自のworkspace群を読む場合はjarへ`--metricsviewer.workspaces-dir=<path>`を渡し、必要なら
`--metricsviewer.initial-workspace=<name>`で起動時workspaceを指定する。Viewerは選択workspaceの
`runs/`直下で、`metrics.jsonl`または`metrics.jsonl.gz`を持つdirectoryだけをRunとして認識する。
両方がある場合は`metrics.jsonl`を使う。

### 2.1 完了Runのmetricsを圧縮する

`apps/70_compress_workspace_metrics.bat`は、選択workspaceの`runs/`直下にある全Runの
`metrics.jsonl`を`metrics.jsonl.gz`へ移行する。引数なしで起動すると`runs/`を持つworkspaceを
名前順に列挙するので、番号を入力して選択する。`[0] EXIT`を選ぶとlauncherを終了する。
workspace名または絶対pathを第1引数で指定してもよい。

preflight後の`Execute compression? [YES/NO/DRY-RUN]:`には、次のいずれかを入力する。

- `YES`: gzipを作成・検証し、確定後に元の`metrics.jsonl`を削除する。
- `NO`: fileを変更せず、そのworkspaceの処理を取り消す。
- `DRY-RUN`: 対象、skip理由、必要容量だけを表示し、fileを変更しない。

非空fileの末尾が改行で完結していないRunや、Runnerが書き込み中のRunは変更せずskipする。
処理後もMetrics Viewer、TensorBoard bridge、MLflow bridge、Optuna集計はrawを優先し、
rawが無いRunではgzipを透過的に読む。
workspaceを引数指定せずに起動した場合は、`YES`、`NO`、`DRY-RUN`の結果表示後にpauseし、
キー入力後にworkspace選択へ戻る。`--no-pause`指定時はpauseを省略する。
複数workspaceを続けて処理し、最後に`[0] EXIT`で終了できる。workspaceを引数指定した場合は1回だけ処理する。

## 3. 画面の基本操作

![複数Runを選択したMetrics Viewer](assets/030_metrics_viewer_run_comparison.png)

| UI | 動作 |
|---|---|
| `Workspace` | serverが列挙したworkspaceへ切り替え、Run選択・色・viewport・凡例状態をリセットする |
| `Runs` | 行クリックで即時toggleする。同じ行を350ms以内にもう一度押すとそのRunだけを選ぶ。空選択も可能 |
| `Select All` / `Latest Only` | 全Run、または最新Runへ選択を切り替える |
| `Tags` | 表示するmetric tagを選ぶ |
| Tagsの`Filter` | 選択済みtagだけを一覧へ残す |
| Run行の背景と`%` | 選択したMetricsマスタをSQLiteへ取り込んだRun単位の進捗を示す |
| `Reload` | Run一覧、tag、現在viewportのmetric rangeを再取得して再描画する |
| `Auto Reload` | 30秒ごとにmetadataを更新し、最新を表示中のgraphだけを更新する |
| `LOD: MinMax / Mean / Band` | 右上のScroll Lock直前にある全graph共通のLOD表示mode。変更時に再取得しない |
| グラフの`Log` | 正負と0を扱えるsigned-log表示を切り替える |
| `Scroll Lock` | グラフ操作を抑え、drag/swipeを縦scrollへ使う |
| Screenshotボタン | side panelを隠し、比較画像向けの表示へ切り替える |

Plotlyのmodebarではzoom、pan、画像保存、`Reset axes`を利用できる。`Autoscale`ボタンは重複を避けるため非表示である。グラフ本体のdouble-clickはPlotlyのaxis resetを維持しつつ、ViewerのReloadも実行する。

初回表示だけ最新Runを自動選択する。以後は手動の空選択と、Run消失で空になった選択を維持する。
Reloadでは既知のOFF tagを保ち、新たに発見された可視tagだけを自動的にONへ加える。
選択workspace、選択tag、LOD mode、Scroll Lockはbrowserの`localStorage`へ保持される。

LODの`MinMax`は各bucketのmin、max、lastを元データの実step順に結ぶ。
`Mean`はbucket平均、`Band`はmin/max帯へ平均線を重ねる。点数が少なくL0を表示できる場合は、
どのmodeでもraw折れ線になる。グラフheaderの`Min / Max / Avg / Std`はviewportではなく、
選択Runのcommit済みraw全点に対する`TagStats`を合成した値である。

取り込み中でもRun一覧とcommit済みgraphは操作できる。Run-level errorまたはstep逆行で
隔離されたtagは`⚠`とtooltipで示し、commit済み部分は引き続き表示する。

## 4. Run比較の手順

### 4.1 比較条件を確認する

グラフを重ねる前に、各Runの次を確認する。

- `config/config_data.txt`: includeとCLI override解決後の全設定
- `config/*.txt`: コンポーネント別の注入済み設定。Envは`config/env.<Env name>.txt`
- seed、Agent/Env、Network構成、batch size、replay条件
- CPU/GPU、device index、決定論設定
- configured evalのinterval、RunMode、clone設定
- Runの停止理由と到達step

名前が似ていても設定が同じとは限らない。Run名ではなくRun artifactを正とする。

### 4.2 同じtagと同じstep軸を選ぶ

`metrics.jsonl`のscalar recordは`type`、`tag`、`step`、`value`を持つが、`step`が`train_step`、`learn_step`、`episode_step`、`exp_step`などのどれかはrecord自体に保存しない。軸はRun内のmetrics設定で決まる。

比較時は同じtag名だけでなく、`metrics.scalar.[tag]`の定義が同じstep軸を選んでいることを確認する。現行設定では、`@learn`と`@episode_end`は明示がなければ`exp_step`、`@train`は`train_step`が既定である。`$exp_step`などの明示指定があればそれを優先する。

### 4.3 matched `exp_step`範囲で比較する

到達stepが異なるRunの最終点同士をそのまま比較しない。両方にデータがある共通の`exp_step`範囲を決め、同じwindowで評価する。

例:

```text
Run A: 0 - 100M exp_step
Run B: 0 - 70M exp_step

比較window: 50M - 70M exp_step
```

同じwindowで次を分けて読む。

- 水準: window内のmean/medianや評価EMA
- 安定性: 振れ幅、急落、seed間のrange/std
- 傾向: window前半と後半の差、傾き
- 速度: 同一hardware・同一並列条件での`90_perf/90_elapse_hour`差分。`exp_step_per_sec`は補助として読む

1つの長いRunだけから、別設定より高速・高性能と断定しない。停止点が違う場合は、まずmatched windowへ揃える。

### 4.4 表示条件による誤読を避ける

- EMAの`ema_alpha`が異なる曲線は滑らかさが違うため、そのまま分散比較しない。
- `interval`やeval頻度が違うと点密度が変わる。線の滑らかさを性能差と解釈しない。
- signed-logは0付近を広げる表示変換である。線間の見た目の距離をlinear scaleと同じ比率として読まない。
- configured evalとEvalPanelは別Runnerである。tagのrunner scopeを確認する。
- `exp_step_per_sec`は他process、動画出力、profiling、Optunaの並列jobに影響される。
- `90_perf/12_exp_step_per_sec`はτ=10秒の時間重みEMAである。窓の長さで重み付けするためevalなどのstallも計上されるが、瞬間値ではないため区間の切り出しには使わない。真のthroughputは`90_perf/90_elapse_hour`の差分と`exp_step`の差分から算出する。`90_perf/22_exp_step_per_sec_ema`はそれをさらに`ema_alpha`で平滑した長期線である。
- 時間重みEMA化より前のRunでは同tagがstallを過小評価している（実測で真値478 steps/sに対し1,830 steps/sを表示した例がある）。過去Runと同tagの数値を直接比較しない。
- 1 seedの差はseedぶれを含む。候補選定後は複数seedで再評価する。

### 4.5 IQN探索P0診断を読む

IQN探索では、解決済み`config/config_data.txt`に`metrics.scalar.iqn_search_p0`が合成されていることと、Policy/Learnerそれぞれのtau配置方式・本数`K/N/M`を先に確認する。診断値だけで採用を決めず、DropMergeのDouble Suika生成数・達成率、報酬、PER健全性、throughputを同じmatched `exp_step` windowで分けて読む。

- `iqn_policy_margin_mc_ratio`はUQE上位2行動のgapをrisk quantileの有限本数scaleで正規化する。`random`ではMonte Carlo平均の安定度、`fixed`および`stratified`ではforward間の乱数分散ではなく積分解像度のproxyとして扱う。
- `iqn_current_mc_scale`と`iqn_target_mc_scale`は`N`と`M`を分けて読む。`iqn_priority_mc_ratio`は現行の平均TD priority信号が両側の有限tau scaleに対してどの程度大きいかを表す。
- `iqn_first_pair_abs_td`と`iqn_first_cancellation_ratio`は初回Learner priority更新行だけを対象とする。`per_sample_initial_count=0`の区間では`iqn_first_*`が`NaN`になるため、0への改善・悪化とは解釈しない。
- TBO有効時のLearner診断は実空間ではなく、現行priorityと同じh空間の値である。TBO有効/無効Runの絶対値を直接比較しない。
- `iqn_uqe_full_q_argmax_disagreement`と`action_full_q_margin.[i]`はfull-distribution queryがあるPolicyだけで成立する。欠落時の`NaN`を一致やmargin 0と解釈しない。
- P0 group OFF/ONの負荷比較は同一binary・seed・実行条件で直列に行い、matched windowの`90_perf/90_elapse_hour`差分から算出したthroughputを比較する。他processやparallel Optuna jobがある測定は採用しない。

### 4.6 分位tail探索診断を読む

分位tail診断はPolicyやpriorityを変更する信号ではなく、既存QR / IQNのreturn distributionを観測する6 scalarである。まず解決済み`config/config_data.txt`でPolicy 5本が`eval2`、Learner 1本が`@learn`へ登録されていることを確認し、Policy側はfixed full distributionの本数`K`、Learner側はPERとTBOの有効状態を併記する。

- `policy_upper_truncated_std`と`policy_lower_truncated_std`は最終実行actionについてmedianから上下へ広がる幅であり、Q値と同じ単位で読む。差からtail asymmetryは見られるが、単一networkの幅をparametric uncertaintyや探索bonusの有効性と断定しない。
- `lower_risk_full_q_argmax_disagreement`は、full Qのargmaxが係数1の仮想的なlower-tail penaltyで変わる割合である。既存`iqn_uqe_full_q_argmax_disagreement`とは目的が異なり、risk回避Policyが実際に有効という結果ではない。
- `quantile_crossing_ratio`はtau順の隣接quantileが降下した割合である。高い区間ではupper / lower tailを分位関数の領域として解釈する信頼度が低いため、tail幅の大小より先にorderingを確認する。同値はcrossingに含めない。
- `policy_selected_crossing_depth_p90_ratio`は、最終実行action内のpositive crossing深度を分布rangeで正規化し、action event内でlane別nearest-rank p90を求めてbatch平均した無次元量である。全actionの発生頻度を測る`quantile_crossing_ratio`と組み合わせ、頻度が横ばいでp90が下がる場合は浅い局所逆転へ寄った可能性、頻度が下がってp90が上がる場合は少数の深い逆転が残る可能性として読む。Run全期間のcrossing sampleをpoolしたp90ではない。
- `upper_tail_priority_spearman`は、PERで既に偏ってsamplingされたminibatch内に限った、upper-tail幅とclip後raw priorityの順位相関である。高い正相関は両者が似た経験を強調している可能性を示すがReplayBuffer全体の冗長性を証明せず、低相関や負相関も新しい信号の有用性を証明しない。
- PER無効、batch不足、定数順位列、Policy full distribution欠損、`K < 2`では該当値が`NaN`になる。0との一致や相関0へ読み替えない。ただしcrossing深度p90は、入力が成立していてpositive crossingがない場合、またはrangeが0の場合を正常値`0`とする。TBO有効時は実空間ではなく現行Policy score / priorityと同じh空間なので、TBO有効/無効Runの絶対値を直接比較しない。

## 5. Optuna結果を分析する

### 5.1 Metrics ViewerとDashboardの役割

| 対象 | 使用する画面 | 正本 |
|---|---|---|
| seed runの時系列 | Metrics Viewer | `<study>_<trial>_s<seed>/metrics.jsonl` |
| 1 trialのmulti-seed結果 | Optuna Dashboard / artifact | 代表folderの`multiseed_summary.json`、`seed_runs.json` |
| 同一params groupの再評価 | summary study | `group_summary.json`とmean/range/std objective |
| runner/config失敗調査 | Run artifact | `manifest.json`、`process.json`、`stdout.log`、`stderr.log` |

代表folder`<study_name>_<trial_name>`は`metrics.jsonl`を持たないためMetrics Viewerには出ない。`_s<seed>`付きのseed runだけを時系列比較する。summary studyはDashboard閲覧用であり、Metrics ViewerのRunではない。

Optuna Dashboardは`apps/runner`から次で起動する。

```powershell
23_optuna_dashboard.bat
```

URLは`http://127.0.0.1:8088`、storageは`apps/runner/runs_optuna/optuna.db`、artifact storeは`runs_optuna/artifacts`である。

### 5.2 scoreを読む

DropMergeのtrial valueはseed別scoreのaggregateである。`score_aggregate`を確認し、`mean`、`median`、`mean-minus-std`、`min`を混同しない。

現行のprimary scoreは指定window内の次の2 tagのmeanを平均する。

```text
21_eval/03_target_reward_ema
21_eval/04_policy_reward_ema
```

late windowの`score_60_80`、`score_80_100`、`late_slope`は伸びや頭打ちを見る補助指標であり、trial valueそのものではない。最終候補はaggregate scoreだけでなく、seed別score、range/std、matched `exp_step`時系列を併せて確認する。

`run-study --n-jobs > 1`は同一GPU上でrunnerを並列起動できるが、durationとstep/secは相互干渉を受ける。その条件のthroughputを単独Runと直接比較しない。

## 6. `inspect_run.py`でRunを検査・抽出する

Metrics Viewerは人間向けの可視化画面である。shellから構造化結果を取り出したい場合、特にAIエージェントへRun分析を依頼する場合は`inspect_run.py`を使う。Run artifactを一切変更しないread-onlyのCLIである。

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py RUN [RUN ...] [options]
```

### 6.1 Runの指定

`RUN`はRun名、または既存の相対・絶対directory pathを取る。Run名の探索範囲は`apps/runner/workspaces/*/runs/`直下だけである。同名Runが複数workspaceにある場合は候補pathを示して終了値2で止まり、どのworkspaceも暗黙選択しない。`apps/runner/runs_*`のlegacy配置はdirectory pathで明示すれば読める。

optionを付けない実行は軽量inspectionになり、Metricsマスタを開かない。artifactのpath・size・更新時刻、`config/config_data.txt`のSHA-256、Metricsマスタの選択結果、Metricsキャッシュの状態だけを返す。分析の入口としてまずこれを実行し、Run treeの再帰検索や巨大な`metrics.jsonl`の直接grepをしない。

### 6.2 metricとconfigを抽出する

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py run_A run_B --metric 42_env/81_double_suika_achieved_ema --window 80%:100% --window 0:5M
```

- `--metric`は完全一致のscalar tag。複数指定でき、Runごとに1 passでまとめて抽出する。
- `--window`は`START:END`（両端inclusive、`K`/`M`/`G`suffix可）と`START%:END%`を取る。複数指定すると各windowを独立に集約する。1つのwindow内で絶対値と百分率を混ぜることはできない。
- percentage windowはRun×step軸の最大到達stepを100%として解決する。Run長が違えば解決後の絶対範囲も違うので、結果には元の百分率表現と解決後の絶対範囲の両方が載る。ハイパラ比較の正式判断では、4.3のとおりmatchedな絶対windowも併記する。
- step軸は`config/config_data.txt`の解決済み`metrics.scalar.[tag]`定義から復元する。軸を決められないtagへpercentage windowを適用すると、対象tagとRunを示して失敗する。
- `--config-key`は実効configのキーまたはglobを取る。glob meta characterは`*`と`?`だけで、`[`と`]`はリテラルとして扱う。`train.eval.[eval1].run_mode`のような`[tag]`記法のキーをそのまま書ける。
- `--diff-config`は複数Run間で値または存在有無が異なるキーだけを返す。`--config-key`と併用するとその範囲に絞られる。

各Run×tag×windowについて`count`、`mean`、`population_std`、`min`/`max`、`first`/`last`、step範囲を返し、曲線形状の確認用に最大128点へ間引いた系列を添える。間引きは決定的で、cache経路とマスタ経路で同じ結果になる。

### 6.3 Run解析プロファイル

繰り返す抽出条件はJSONの[Run解析プロファイル](../../CONTEXT.md)へ切り出して`--profile`で渡す。

```json
{
  "version": 1,
  "name": "dropmerge-iqn-k-search",
  "metrics": ["42_env/81_double_suika_achieved_ema"],
  "config_keys": ["agent.*"],
  "windows": ["0:5M", "80%:100%"]
}
```

`--metric`と`--config-key`はprofileの配列末尾へ追加される。`--window`を1件でも指定するとprofileの`windows`は全置換される。profileは抽出条件だけを持ち、解釈・閾値・採否判断・Run pathは持たない。

### 6.4 Metricsキャッシュとの関係

`metrics_cache.db`は完全にcurrentなときだけread-onlyで使い、それ以外はMetricsマスタへ自動fallbackする。`--format md`または`json`の結果に判定結果と理由（`current` / `absent` / `invalid` / `partial` / `stale` / `error`）が載る。toolはキャッシュの作成・更新・修復・削除を一切行わない。

Runner実行中のRunも読める。rawは実行開始時のサイズまでを読み、未終端の末尾行を取り込まず、読み取り中にマスタが変化した場合は暫定結果として`provisional`と`source_changed_during_read`を立てる。

### 6.5 出力

既定はJSON（schema v1）で、`--format md`にするとMarkdownになる。どちらも同じresult modelから生成するので、Markdownだけに現れる分析判断はない。`--output`を付けるとstdoutの代わりにfileへ書き、一時file経由で原子的に置換する。警告と診断は常にstderrへ出るのでJSONを汚さない。

終了値は、Run未発見・曖昧性・引数やprofileの契約違反が`2`、source読み取りやquery失敗、および指定したmetric/config selectorが全Runで1件も成立しない場合が`1`、一部Runだけの欠損は`0`である。

## 7. 最小分析チェックリスト

1. 比較対象の`config/config_data.txt`を保存した。
2. 同じtag、同じstep軸、同じeval定義を確認した。
3. 共通の`exp_step`windowを決めた。
4. scoreとthroughputを分けて読んだ。
5. EMA、interval、signed-logなど表示条件を揃えた。
6. 1 seedの結果だけで最終判断していない。
7. Optunaではaggregate方式、seed別分布、trial stateを確認した。

## 8. 関連文書

- [Run実行ユーザーガイド](020_user_guide_run.jp.md)
- [開発環境](040_development_environment.jp.md)
- [可観測性](140_observability.jp.md)
- [アプリケーションとツール](160_applications_and_tools.jp.md)
- [DropMerge Optuna利用ガイド](../optuna.md)
