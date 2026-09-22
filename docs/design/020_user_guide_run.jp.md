# Run実行ユーザーガイド

> 主たる観点: 行程単位（設定、起動、操作、終了、成果物確認）

## 1. はじめに

### 1.1 目的

本書は、ANET RL Runnerで1つのRunを設定し、学習・評価画面を操作し、終了後の成果物を確認するまでの基本手順を説明する。

### 1.2 対象読者

- 既存のEnvとAgentを設定してRunを実行する利用者
- Train、Eval、可視化パネルの基本操作を知りたい利用者
- Run directoryに保存されるログ、メトリクス、checkpointを確認したい利用者

### 1.3 記載範囲

現行の`AnetRLRunner`、`apps/runner/config`とその設定記法、標準GUI操作、Run成果物を扱う。
新しいEnv、Agent、Observerの実装方法は対象外とし、各設計文書を参照する。

> [!NOTE]
> 本書の実行経路はWindows x64とNVIDIA CUDAを使う構成で検証済みである。CPU-only、Linux、macOS、他GPU backendは未検証であり、同じ操作結果を保証しない。

## 2. 実行前の準備

### 2.1 実行要件

本書は、`apps/runner/bin/Release/AnetRLRunner.exe`がbuild済みで、runnerが必要とするDLLとconfigが配置済みであることを前提とする。検証済み構成ではWindows x64、NVIDIA driver/CUDA runtime、CUDA対応libtorchを使用する。

開発環境の準備、依存関係、CMake preset、build手順は[開発環境](040_development_environment.jp.md)を参照する。

### 2.2 設定ファイルの選択

引数を省略したrunnerはworkspace選択ダイアログを表示し、選択したworkspaceの`config/_main.txt`からEnvを選ぶ。共通の`apps/runner/config/_main.txt`はAgent、Network、metric等だけを読み、Env選択はworkspace側へ分離される。新規workspace、および既存ディレクトリの初回選択時に不足しているworkspace configは、`apps/runner/config/_workspace_template.txt`を`config/_main.txt`へコピーして作成される。

```text
# apps/runner/workspaces/<workspace>/config/_main.txt
#$include <LunarLander.txt>
$include <DropMerge.txt>
```

1回のRunでは、意図したEnv設定だけを有効にする。各Env設定内の`app.$`、`DefaultDQNAgent.$`、`metrics.scalar.$`などの選択チェーンと、`=`・`?=`の使い分け、コマンドライン`key=value`の効き方は[3. 設定ファイルの書き方](#3-設定ファイルの書き方)を参照する。

### 2.3 最初に確認する設定

| キー | 役割 |
|---|---|
| `app.run_name` | Run名。`{t}`は起動時刻へ展開される |
| `app.runs_dir` | workspaceモードでは`<workspace>/runs`へRunnerが導出する。設定やCLIからの変更は禁止 |
| `app.train_auto_start` | `true`ならGUI初期化後に学習を開始する |
| `app.show_error_dialog` | error logに加えてモーダルダイアログを表示するか。未指定時は`true` |
| `app.save_agent_on_close` | 終了時に`agent_close.anet`を自動保存するか。未指定時は`true`。`false`でも手動のSave Checkpointは使える |
| `app.drain_timeout_sec` | 終了時にbackground Observerの完走を待つ全体上限秒。未指定時は3600。正整数だけを受け付ける |
| `app.eval_panel.auto_start` | 手動EvalPanelを起動直後から動かすか |
| `run.seed` | Runの基準seed |
| `run.train.num_envs` | Train用BatchEnvのlane数 |
| `run.train.actor` | 学習Actorのカタログ名。省略時は`train` |
| `run.eval.[tag].actor` | 評価Actorのカタログ名。省略時は評価タグ名 |
| `agent.class_id` | 使用するAgent実装 |
| `agent.device_type` / `agent.device_index` | AgentのCPU/CUDA device |
| `env.worker_type` / `env.worker_threads` | Env batchの実行方式とworker数 |
| `run.eval_device_type` / `run.eval_device_index` | configured evalのdevice |
| `backend.deterministic_algorithms` | 決定論的algorithmを要求するか |

`agent.device_type=1`はCUDA、`0`はCPUである。EnvをCPU、AgentとEvalをCUDAに置く構成では、device転送を含めて性能を判断する。

### 2.4 評価スロットの方策を選ぶ

評価スロットは`<Agent>.actor.[key]`を名前で参照する。方策の種類・epsilon・network・cloneはAgentのカタログに書く。例えばDefaultDQNでonline networkをepsilon 0.1で評価する場合:

```properties
DefaultDQNAgent.actor.[explore].$ = DefaultDQNAgent.actor.[eval]
DefaultDQNAgent.actor.[explore].policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.[explore].policy.eps_start = 0.1
DefaultDQNAgent.actor.[explore].policy.eps_end = 0.1
DefaultDQNAgent.actor.[explore].policy.eps_decay_steps = 0
DefaultDQNAgent.actor.[explore].network = online
DefaultDQNAgent.actor.[explore].clone_model = true
run.eval.[explore].actor = explore
run.eval.[explore].eval_batch_size = 2
run.eval.[explore].eval_episodes = 2
run.eval_schedule.[explore].interval = 100
run.eval_schedule.[explore].use_background = false
run.eval_schedule.[explore].wait_on_exit = true
metrics.scalar.[explore/epsilon] = $eval.[explore] $actor epsilon @session_end
```

共通スロットは`eval_target`と`eval`で、それぞれtarget / onlineを参照する。metricsの既存出力タグは維持する。未定義のActor参照は起動時に失敗するが、スケジュールのない定義や`interval=0`のスロットはActorを生成しない。sharedのActorはAgentと同じdeviceを使う。MuZeroはcloneに対応せず、`clone_model=false`を指定する。

## 3. 設定ファイルの書き方

本章は`apps/runner/config`の設定ファイルを読み書きするために必要な記法と優先順位だけを説明する。用語の正本は[CONTEXT.md](../../CONTEXT.md)、契約の詳細とresolverの内部処理は[実行基盤と設定](100_runtime_and_configuration.jp.md)の§2.1である。

### 3.1 基本ポリシー

1. 設定は「ベース + 差分」で組む。共通ファイルとプロファイルがベース、`>`でつないだ選択チェーンが差分、環境別ファイルに直接書いた値が最後の個別指定になる。
2. `X.$ = …`はその設定Xのベースを指定する。`X.key = v`はベースより強い個別指定(個別葉)、`X.key ?= v`はベースに負ける既定値(既定葉)である。
3. 異なるキーの行順は結果を変えない。同じキーを2回書いたら後勝ちだが、`=`と`?=`が混在したら順序に関係なく`=`が勝つ。同じ`X.$`を書き直すとチェーンは丸ごと置き換わる。
4. Runプロファイルとコマンドラインは通常の設定より強い。ただし効くのは指定したそのキーだけで、そのキーから継承される葉すべてに優先権が付くわけではない。
5. 演算子はファイルの役割で決める。共通ファイル(`common.txt`、`agent.txt`、`nn*.txt`、`metrics_*.txt`)のベース定義は`?=`、環境別ファイル(`<Env>.txt`)は「デフォルト設定」ブロックだけ`?=`、それ以外は`=`で書く。選択宣言`.$`、Runプロファイル、CLIは常に`=`である。

### 3.2 用語

| 用語 | 意味 | 例 |
|---|---|---|
| プロファイル | `@`で始まるセグメントを持つ名前付きの設定部品。選ばれるまで値に出ない | `backend.@deterministic`、`DefaultDQNAgent.@baseline` |
| 選択チェーン | `X.$ = A > B`の形で、各項の最終値を左から右へ差分合成してXのベースを作る指定 | `DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2` |
| 選択の最終値 | 選択元自身のベース・部分指定・個別指定・Run・CLIまで反映した値とキーの集合。チェーンの各項はこれを持ち込む | `@iqn`の最終値は、`@iqn`が持つ葉と`@iqn`自身の`.$`の結果 |
| カタログ | `[key]`をidentityとして名前で参照される部品定義群 | `net.block.[Linear_120]`、`run.eval.[test1]`、`metrics.scalar.@baseline.[21_eval/01_target_reward]` |
| 上書き層 | 環境別ファイルでチェーンの末尾に置く差分用の通常prefix。A1/A2/A3はAgent、E1はEnv、M1/M2はMetrics、P1はappで、番号が大きいほど一時的 | `A2.learner.per_alpha = 0.2` |
| Runプロファイル | 1つのRunを特徴づける選択と値の組を`run.@<name>`で命名したもの。`run.$`で選ぶ | `run.@repro` |
| 個別指定(個別葉) | `=`でその設定の葉に直接与える値。ベースより強い | `E1.obs_include_action = true` |
| 既定葉 | `?=`で書く既定値。その設定のベースが同じキーを供給するときだけ負ける | `AtariEnv.game ?= pong` |

### 3.3 記法一覧

| 記法 | 実例 | 意味 |
|---|---|---|
| `key = value` | `E1.obs_include_action = true` | 個別葉。ベースと既定葉より強い |
| `key ?= value` | `AtariEnv.game ?= pong` | 既定葉。その設定の`.$`が同じキーを供給しなければ残る |
| `Owner.@name.key = v` | `LunarLanderEnv.@trunk.wind_power = 3.0` | プロファイルの定義(ドット形) |
| `Owner.@name : key = v` | `DefaultDQNAgent.@qr  : quantile_mode = qr` | 同上(コロン形)。`:`は`.`と同じ意味で、プロファイルの境界にだけ使う |
| `@vars : key = v` | `@vars : max_exp_step  = 100,000,000` | 値スロット。`${@vars.max_exp_step}`で参照する |
| `Owner.$ = A > B > C` | `DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2` | 選択チェーン。短い`@name`は`Owner.@name`を指す |
| 完全修飾の項 | `LunarLanderEnv.$ = LunarLanderEnv.@trunk > E1` | 書いたとおりのprefixを指す。別のownerの定義も選べる |
| `Owner.sub.$ = …` | `run.eval.[test1].env.$ = LunarLanderEnv.@test1` | 部分選択。配下の一部だけのベース |
| `[key].leaf`、`[key].$` | `net.block.[Linear_120].type ?= Linear`、`net.block.[MLP_FC1].$ = net.block.[Linear_120] > net.block.FC1` | カタログ項目の定義と、その項目のベース |
| metrics定義と選択 | `metrics.scalar.@baseline.[21_eval/01_target_reward] ?= mean.episode_return $runner @session_end $eval.[eval_target]`、`metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@iqn_search_p0 > M1` | tagが`[key]`のカタログ。選択はチェーン |
| 上書き層 | `A2.learner.per_alpha = 0.2`、`app.$ = app.online > P1` | 環境別ファイルの差分。チェーンの末尾に置く |
| `run.@name : key = v` | `run.@iqn32_stratified : A2.learner.per_alpha = 0.2` | Runプロファイルの葉 |
| `run.@name : Owner.$ = …` | `run.@repro : backend.$ = backend.@deterministic` | Runプロファイルによるチェーンの置き換え |
| `run.$ = …` | `#run.$ = run.@repro`(ファイル)、`run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base`(bat・CLI) | Runの選択 |
| `$include <file>` | `$include <common.txt>` | 取り込み。`"path"`形も使える |
| CLI `key=value` | `E1.game=breakout`、`A3.auto_load_file=<path>` | 起動引数。指定したそのキーが最優先 |
| `${full.key}` | `app.run_name = run_{t}_atari_${E1.game}` | 解決済みの最終値を1段だけ参照する |
| `#`、`//` | `DropMergeEnv.grid_cols ?= 40              # グリッド列数（横方向）` | 以降はコメント |
| `{t}` | `app.run_name ?= run_{t}` | 起動時刻へ展開される |

### 3.4 各記法の詳細

#### 3.4.1 葉の代入`=`と`?=`

`key = value`は個別葉、`key ?= value`は既定葉である。

- 個別葉は、同じ設定の`.$`が供給する値より強い。プロファイルの中でも同じで、`X.@fast : $ = @baseline`と`X.@fast : lr = 0.01`を並べれば`@fast`は「`@baseline`の上に`lr`だけ変えたもの」になる。
- 既定葉は、その設定の`.$`(全体・部分)が同じキーを供給すればその値に負け、供給しなければ残る。チェーンを置き換えても既定葉は消えない。
- 同じキーを複数回書いたら、同じ演算子同士は後勝ちである。`$include`や追加ファイルをまたいでも同じ。`=`と`?=`が混在したら順序に関係なく`=`が勝ち、後から書いた既定葉が個別葉を取り消すことはない。
- `?=`は葉にだけ書ける。`.$`、Runプロファイル(`run.@x : …`)、CLIに書くと読み込み時にエラーになる。
- 落とし穴: `key ? = v`のように`?`と`=`を離すとエラーになる。プロファイルの中の`?=`はそのプロファイル自身の`.$`にだけ負け、外側の設定へは普通のベース値として届く。

```properties
backend.cudnn_benchmark ?= false                   # 既定葉
backend.torch_num_threads ?= 1                     # 既定葉
backend.@non-deterministic.cudnn_benchmark = true
backend.$ = backend.@non-deterministic             # ベースが cudnn_benchmark を供給する → true
                                                   # torch_num_threads はベースに無い → 1 のまま
```

#### 3.4.2 プロファイル`@name`

`Owner.@name.key = v`と`Owner.@name : key = v`は同じ意味で、`:`はプロファイルの境界にだけ使う。

- 定義しただけでは値に出ない。チェーンや部分選択で選ばれた分だけ最終値に入る。
- 短い`@name`は、書いた場所(定義元)の名前空間で解決する。`Env.$ = @a`は`Env.@a`、`Env.@a : $ = @b`は`Env.@b`を指す。`Other.$ = Env.@a`のように別の場所から使っても、`@a`の中の`@b`は`Env.@b`のままである。
- Runプロファイル内の`run.@x : Env.$ = @a`の`@a`は`Env.@a`を指す(展開先の名前空間)。
- プロファイルは自分の`.$`を持てる(`X.@fast : $ = @baseline`)。左辺の`@`は原則1個にする。
- `@vars`のような名前は値スロットとして使い、`${@vars.key}`で参照する。

```properties
DefaultDQNAgent.@qr  : quantile_mode = qr
DefaultDQNAgent.@qr  : net.$ = net.@qr
DefaultDQNAgent.@iqn : quantile_mode = iqn
DefaultDQNAgent.@iqn : net.$ = net.@iqn
```

#### 3.4.3 選択チェーン`.$`と`>`

`Owner.$ = A > B > C`と書く。項はプロファイル(`@name`、`Owner.@name`)、カタログ項目(`net.block.[X]`)、通常のprefix(`A2`、`app.online`)のどれでもよい。

- 各項は自分の最終値を持ち込み、左から右へ重ねる。右の項にあるキーは右が勝ち、右に無いキーは左から残る(差分合成)。
- 項の中の`.$`はその項のベースを作るために使われ、コピー先で再実行されない。
- 同じ`Owner.$`をもう一度書くとチェーンは丸ごと置き換わる。`Env.$ = A2`の後に`Env.$ = A3`と書けばA3だけになる。重ねたいなら1本のチェーンに書く。
- `.$`を持たない通常のprefix(`A3`に何も書いていなくても)は許容される。未定義の`@`プロファイルやカタログ項目を選ぶとエラーになる。
- 選択の入れ子は深さ10まで。

```properties
DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2
```

`@baseline`の上に`@iqn`の差分、`@heavy`の差分、A1(環境固有の恒久的な調整)、`@bf16`、A2(実験値)の順に重なる。A2に無いキーは左側から残る。

#### 3.4.4 部分選択

`Owner.sub.$ = …`は配下の一部だけのベースを指定する。配下の部分`.$`は全体の`.$`より強く、行順に依らない。プロファイル内に書いた部分`.$`はそのプロファイルの最終値の一部として外側へ届き、同じ場所にrootで書いた部分`.$`はそれより強い。

```properties
run.eval.[test1].env.$ = LunarLanderEnv.@test1
DefaultDQNAgent.@qr : net.$ = net.@qr
```

#### 3.4.5 カタログ`[key]`

`[key]`はカタログ項目のidentityである。NNブロック(`net.block.[Linear_120]`)、metrics tag(`metrics.scalar.@baseline.[21_eval/01_target_reward]`)、eval tag(`run.eval.[test1]`)が現行の例である。定義は`[key].leaf = v`、項目のベースは`[key].$ = …`で書き、項目同士の継承もチェーンで表す。未定義の項目を参照するとエラーになる。

```properties
net.block.[Linear_120].type ?= Linear
net.block.[Linear_120].linear.out_features ?= 120
net.block.[MLP_FC1].$ = net.block.[Linear_120] > net.block.FC1
```

#### 3.4.6 上書き層

環境別ファイルで実験値や環境固有の差分を書く場所である。A1/A2/A3はAgent、E1はEnv、M1/M2はMetrics、P1はappを対象にし、番号が大きいほど一時的という運用上の名前で、resolverにとっては通常のprefixである(名前による特別扱いは無い)。チェーンの末尾に置くことで「最後に重なる差分」になる。

```properties
A2.learner.iqn.current_taus.num_taus = 32
DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2
```

落とし穴: 上書き層に書いた値も、その設定にrootで`=`と書いた個別葉には負ける。`DefaultDQNAgent.learner.iqn.current_taus.num_taus = 8`をrootに書けば、A2の32よりそちらが勝つ。

#### 3.4.7 Runプロファイル`run.@name`と`run.$`

`run.@name : Owner.key = v`で葉を、`run.@name : Owner.$ = …`でチェーンの置き換えを書き、`run.$ = run.@a > run.@b`で選ぶ。

- `run.$`は通常の選択より先に展開される。項は左から右へ後勝ちで、同じキーを複数の項が書けば右の項の値になる。
- Runプロファイルの葉は通常設定の`=`より強い別の段にあり、CLIにだけ負ける。Runで`Owner.$`を書けばチェーンの置き換えになる。
- Runプロファイルの中に`?=`は書けない。別の`run.$`を供給する入れ子も禁止である。
- ファイルでは`#run.$ = run.@repro`のようにコメントで切り替え、batやCLIでは`run.$=run.@a>run.@b`で選ぶ。seedはRunプロファイルに含めず`run.seed`で別に指定し、同じプロファイルの複数seedを比較できるようにする。

```properties
run.@repro : backend.$ = backend.@deterministic
run.@repro : DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2
run.@repro : LunarLanderEnv.$ = LunarLanderEnv.@trunk > E1
#run.$ = run.@repro
```

#### 3.4.8 コマンドライン`key=value`

書式はファイルと同じで、`E1.game=breakout`のように空白を入れずに書くか引用符で囲む。効き方は指定したキーの種類で決まる。

| 指定 | 効く対象と結果 |
|---|---|
| `AtariEnv.game=breakout` | 実効葉そのもの。ファイル、継承、Runプロファイルの値に必ず勝つ |
| `E1.game=breakout` | 上書き層E1の葉。`AtariEnv.$ = … > E1`のチェーンを通ってbreakoutが届くが、rootに`AtariEnv.game = pong`と個別指定があればそちらが勝つ |
| `AtariEnv.$=AtariEnv.@classic>E1` | 選択キーだけ。チェーンを丸ごと置き換える。個別葉と既定葉は残る |
| `run.$=run.@a>run.@b` | Runの選択そのものを置き換える |

CLIに`?=`は書けない。

#### 3.4.9 `$include`、`${full.key}`、コメント

- `$include <name>`は、include元ファイルのディレクトリ、次にconfig search dirsの順に探す。`"path"`形も使える。深さは10まで。見つからない場合はWARNを出して続行するので、意図した設定が読まれているかは`config/config_data.txt`で確認する。
- 起動時の重ね順(共通main config、導出した`app.runs_dir`、workspaceの`config/_main.txt`、コマンドライン)は[4.1 標準起動](#41-標準起動)を参照する。
- `${full.key}`は、解決後の最終値を1段だけ参照する。参照先が無い、または参照先がさらに`${}`を含む場合はエラーになる。
- `#`と`//`以降はコメント。`=`の無い行は読み飛ばす。キー内の空白は除去され、`:`は1個だけ使える。`{t}`は起動時刻へ展開される。

### 3.5 優先順位のまとめ

```text
X.key ?= v              既定葉。X.$ に負ける
 < X.$ = A > B          全体のベース(チェーン。項同士は右勝ち)
 < X.sub.$ = C          部分のベース(全体より強い)
 < X.key = v            個別葉
 < run.@p : X.key = v   Run プロファイルの葉
 < CLI X.key=v          そのキーだけ最優先
```

- この順位は設定(owner)ごとに再帰する。チェーンの各項は自分の順位で最終値を確定してから合成されるので、A3の中に`=`で書いた葉も、外側の設定Xから見ればベースである。
- 同じキーの再指定は後勝ち、`=`と`?=`の混在は`=`、`X.$`の再指定はチェーン置換(3.1の3)。
- RunプロファイルとCLIの最優先は指定したそのキーだけに効く(3.4.8の表)。

### 3.6 どのファイルに何を書くか

| ファイル | 役割 | 演算子 |
|---|---|---|
| `_main.txt` | 共通ファイルの`$include`一覧 | - |
| `common.txt` | trainer、agent、env、gui、backend、appの共有既定値 | `?=`。切替用プロファイル(`backend.@deterministic`等)は`=` |
| `agent.txt` | 各Agentの`@baseline`、切替用プロファイル(`@qr`、`@iqn`、`@bf16`等)、`<Agent>.$ = @baseline` | `@baseline`は`?=`。切替用プロファイルとチェーンは`=` |
| `nn.txt`、`nn_cnx.txt` | 共有NN部品(`net.block.[…]`、`net.body.@…`) | `?=` |
| `metrics_scalar.txt`、`metrics_image.txt` | metricsカタログ(`metrics.scalar.@baseline.[tag]`等) | `?=` |
| `<Env>.txt` | Envとアルゴリズム構成: チェーン、プロファイル、上書き層、Runプロファイル、NN配線、metrics差分 | `=`。Envの「デフォルト設定」ブロックだけ`?=`。ブロック外で`?=`が必要な葉は理由をコメントに書く |
| workspaceの`config/_main.txt` | 有効にする`<Env>.txt`の`$include` | - |
| bat、CLI | `run.$`の選択と葉の上書き | `=` |

運用は、**共通ファイルのベース定義は`?=`、環境別ファイルはデフォルト設定だけ`?=`、それ以外は原則`=`**とする(正本は[AGENTS.md](../../AGENTS.md)「設定ファイルの代入演算子」)。「将来上書きするかもしれない」だけでは`?=`にしない。既存の実効値を保つために`?=`が必要な葉だけ、理由を添えてデフォルト設定へまとめる。ブロックの見出しやコメントの書き方は自由である(`DropMerge.txt`は`# --- DropMergeEnv デフォルト設定`、`LunarLander.txt`は見出し無し)。この運用は編集と監査の規約であり、resolverがファイル名や位置で扱いを変えることはない。設定を追加・変更したら静的検査で、共通ファイルやデフォルト設定に残った`=`と、環境別の個別指定に紛れた`?=`を確認する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

`rg`(ripgrep)がPATHに必要である。意図的に`=`で書く既定値(`DropMerge_optuna.txt`のseed等)は検査器の分類理由に登録する。

### 3.7 例

**backend の既定葉と切り替え**

```properties
# common.txt
backend.cudnn_benchmark ?= false
backend.cudnn_deterministic ?= false
backend.deterministic_algorithms ?= true
backend.deterministic_warn_only ?= false
backend.@non-deterministic.cudnn_benchmark = true
backend.@non-deterministic.cudnn_deterministic = false
backend.@non-deterministic.deterministic_algorithms = false

# LunarLander.txt
backend.$ = backend.@non-deterministic
run.@repro : backend.$ = backend.@deterministic
```

通常のRunでは`cudnn_benchmark`、`cudnn_deterministic`、`deterministic_algorithms`の3キーが`@non-deterministic`の値になり、`deterministic_warn_only`など残りの既定葉はそのまま残る。`run.$=run.@repro`で起動するとチェーンが`@deterministic`に置き換わるが、既定葉は影響を受けない。

**DropMergeEnv のデフォルト設定とチェーン**

```properties
# DropMerge.txt
DropMergeEnv.$ = DropMergeEnv.@baseline > DropMergeEnv.@G5846 > DropMergeEnv.@heavy > E1

# --- DropMergeEnv デフォルト設定
DropMergeEnv.grid_cols ?= 40              # グリッド列数（横方向）
DropMergeEnv.grid_rows ?= 64              # グリッド行数（縦方向）
DropMergeEnv.action_mode ?= move_fast     # move move_fast direct direct_noop
```

`grid_cols`は、`@baseline`、`@G5846`、`@heavy`、E1のうち右端で供給している項の値になる(現行では`@G5846`が58を供給する)。どの項も供給しないキーは既定葉の値のままである。

**デフォルト設定ブロックの外にある`?=`**

```properties
# Atari.txt
DefaultDQNAgent.net.branch.[value_stream].structure ?= AtariHeadFC512 > SiLU # NatureDQNのReLU選択とIQNの出力先を維持するために?=
DefaultDQNAgent.net.branch.[adv_stream].structure ?= AtariHeadFC512 > SiLU   # NatureDQNのReLU選択とIQNの出力先を維持するために?=
DefaultDQNAgent.net.body.output.[features] ?= main_feature # NatureDQNのReLU選択とIQNの出力先を維持するために?=
```

チェーンで`@nature`(ReLU)を選ぶ構成や、IQNの配線が`[features]`の出力先を変える構成でも既定を残すため、理由をコメントに書いて`?=`にしている。環境別ファイルで`?=`にするのはこのように理由がある葉だけである。

**この Run だけ値を変える**

```powershell
apps\runner\bin\Release\AnetRLRunner.exe --workspace atari-03 E1.game=breakout
apps\runner\bin\Release\AnetRLRunner.exe --workspace atari-03 AtariEnv.game=breakout
```

`E1.game=breakout`は上書き層E1の葉を変え、`AtariEnv.$ = … > E1`のチェーンを通って届く。通常はこれで足りる。`AtariEnv.game=breakout`は実効葉そのものを指定するので、ファイルに`AtariEnv.game = pong`と個別指定があっても勝つ。繰り返し使う組み合わせはRunプロファイルにまとめて`run.$=run.@breakout`で選ぶ。

```properties
run.@breakout : E1.game = breakout
run.@breakout : app.run_name = run_{t}_breakout
```

### 3.8 確認方法とよくあるエラー

- `config/config_data.txt`は解決後の実効設定である。上書き層(`A1`〜`A3`、`E1`、`M1`/`M2`、`P1`)や選択元の通常prefix(`app.online`等)の定義も同居するが、`@`プロファイルと`.$`は含まれない。どのプロファイルが選ばれたかはこのfileからは分からない。
- `json/config_resolution.json`は選択の記録(`selections`。`run.$`が先頭で、`key`は宣言した定義位置)、値参照(`references`)、Runプロファイルが変えた葉(`overrides`)を持つ。読み方は[分析ユーザーガイド](030_user_guide_analysis.jp.md)の§6.6と§6.7を参照する。
- 演算子の書き忘れは3.6の静的検査で確認する。

| メッセージの冒頭 | 原因 | 対処 |
|---|---|---|
| `Properties: invalid assignment operator` | `key ? = v`のように`?`と`=`が離れている、またはキーに`?`が残っている | `?=`を続けて書く |
| `Properties: default assignment is only allowed for ordinary leaves` | `.$`やRunプロファイルに`?=`を書いた | `=`にする |
| `ConfigManager: default assignment is not allowed on command line` | CLIに`?=`を書いた | `=`にする |
| `ConfigResolver: material selection target not found` | 未定義の`@`プロファイルを選んだ(`material`はプロファイルを指す) | 定義元の名前空間と綴りを確認する |
| `ConfigResolver: catalog selection target not found` | 未定義の`[key]`を選んだ | 同上 |
| `ConfigResolver: selection self-supply detected`、`selection cycle detected` | 自分自身や自分の配下を選んでいる、または選択が循環している | チェーンの参照先を見直す |
| `ConfigResolver: selection depth limit exceeded` | 選択の入れ子が10段を超えた | 中間のプロファイルを減らす |
| `ConfigResolver: value reference target not found`、`chained value reference is not supported` | `${}`の参照先が無い、または参照先がさらに`${}`を含む | 参照先を最終値のキーにする |
| `ConfigResolver: named trunk must not select another trunk` | Runプロファイルの中で`run.$`を書いた | `run.$`は起動側で1回だけ書く |
| `Workspace config changed app.runs_dir` | workspaceの設定やCLIで`app.runs_dir`を変えた | workspaceモードでは変更しない |
| WARN `Properties: Failed to open include file` | `$include`先が無い(エラーにならず続行する) | パスとconfig search dirsを確認する |

## 4. Runを開始する

### 4.1 標準起動

`apps/runner`から次を実行する。

```powershell
10_run.bat
```

初回は`_default`が新規名として入力済みの選択ダイアログが開く。履歴、`workspaces/`直下の全ディレクトリ一覧、任意パス参照、新規名から選択できる。過去Runだけを移動したフォルダなど`config/_main.txt`が無い既存ディレクトリも一覧に出て、選択時に不足configだけが補完される。新規名は入力中に検証され、不正理由が入力欄の下へ表示されている間はOKを選択できない。`--workspace dm_long`で相対workspaceを直接指定し、`--select-workspace`でスキップ設定に関係なくダイアログを表示できる。相対パスは`apps/runner/workspaces/`基準、絶対パスも使用できる。入力の外側空白は除去され、`#`、`//`、末尾`;`、UNC pathは拒否される。

または、リポジトリルートから実行ファイルとmain configを明示する。

```powershell
apps\runner\bin\Release\AnetRLRunner.exe `
  --config apps\runner\config\_main.txt `
  app.run_name=run_{t}_trial `
  run.seed=12345
```

`--config`はworkspace、履歴、`last_workspace.txt`を一切参照しない完全自己記述モードである。`--workspace`または`--select-workspace`との併用は起動エラーになる。

起動時は概ね次の順に初期化される。

1. workspaceを確定し、共通main config、導出`app.runs_dir`、workspace config、コマンドラインoverrideの順で解決する。
2. Run directory、`metrics.jsonl`、標準出力ログを準備する。
3. libtorch backendと登録済みEnvを初期化する。
4. `RunManager`がEnv、Agent、Train Runner、configured Evalを構築する。
5. Train、Eval、QValue、Logの各パネルを接続する。
6. `RunnerThread`を開始する。`app.train_auto_start=false`の場合はpause状態で待機する。

起動に失敗した場合はerror logを確認する。online構成ではログに加えてエラーダイアログを表示する。batchrun構成ではモーダル表示せず、Run directory成立前は親processのstderr、`StandardStreamLogger`開始後は対象Runの`stderr.log`、通常logger構築後は`<run_name>.log`へ記録する。fatalを処理したprocessは非ゼロで終了する。

### 4.2 自動停止・自動pause

`app.train_exit_step`、`app.exp_exit_step`は上限到達時にRunを終了する。`app.train_pause_step`、`app.exp_pause_step`は一度だけ自動pauseする。
batch実験では、`app.$=app.batchrun`で低FPS表示、`exp_exit_step`、`app.show_error_dialog=false`をまとめて選ぶbatchrun構成が用意されている。人が操作するonline構成は`app.$=app.online`で`app.show_error_dialog=true`を選ぶ。これらはTrain / Evalの`RunMode`とは別概念である。

`apps/11_batch_run.bat`、`apps/12_batch_run.bat`、`apps/18_batch_run_atari5.bat`は各Runの終了コードを記録する。失敗したRunでは`[ERROR] RUN FAILED exit_code=<code> args=<args>`を表示して後続Runを続け、全Run終了後の`pause`を経て1を返す。全Run成功時は0を返す。

## 5. AP画面

### 5.1 基本構成

Runner画面はwxAUIのpaneで構成される。

画面上端には対象別の4本のツールバーがある。各バーはgripperをドラッグしてdock、float、再dockでき、`View > Reset Layout`で上端1行の既定位置へ戻る。floatさせたバーのwindow titleにはバー名(`Run Control`、`Steps`、`Run Operations`、`Panels`)が出る。

- `Run制御`: Trainのpause/resumeと、separatorを挟んでEvalのpause/resume、Evalの1 step実行を提供する。走行中のtoolは押下表示になり、iconは次の操作を表す一時停止記号へ変わる。停止中は再生記号へ戻る。非表示のEvalをresumeするか1 step実行すると`Evaluation View`も表示する。右クリックによるresumeも同じ扱いで、keyboard経路(`Space` / `Ctrl`)だけが表示と独立している。
- `Step表示`: Trainの`exp_step`と`train_step`を表示する。
- `Run操作`: 任意pathへのcheckpoint保存とRun folder表示を提供する。
- `Panel表示`: `Logs`、`Eval View`、`Q-Values`の表示を切り替える。対応する`View` menuとpaneの状態に同期する。

- `Train View`: Train Runnerから受け取ったEnv固有Viewを表示する。
- `Evaluation View`: Trainとは別のEval Envを手動またはtimerで進める。初期状態では非表示である。
- `Evaluation Q-Values`: Eval Actorの出力を表示し、Actionを手動指定できる。
- `Logs`: 実行ログを表示し、Error、Warn、Info、Verboseを切り替える。
- `HeatMap` / `Conv2d`: `View`メニューから追加する補助pane。

![DropMergeのTrain ViewとEvaluation View](assets/020_runner_dropmerge_train_eval.png)

![LunarLanderの実行画面と可視化pane](assets/020_runner_lunarlander_visualization.png)

paneを閉じたり初期化前のViewを表示した場合は、次のように描画対象が少ない状態になることがある。`View > Reset Layout`で既定配置へ戻せる。

![描画対象がない状態のRunner画面](assets/020_runner_empty_view.png)

## 6. 操作方法

### 6.1 学習のpauseと再開

Run制御ツールバーの`Train`を選ぶとTrainをpause/resumeする。Train/Eval View上の左クリックと`Shift`も同じ操作として併存する。`app.train_auto_start=false`で起動した場合も同じ操作で開始できる。pause時はmetrics、stdout/stderr、text logを明示的にflushする。Trainが停止して再開不能になった後はツールが無効になる。

### 6.2 評価と画面操作

| 操作 | 動作 |
|---|---|
| Run制御ツールバーの`Eval` | EvalPanelをpause/resumeする。resume時にpaneが非表示なら表示する |
| Run制御ツールバーの`Step` | Evalを1 step自動実行する。paneが非表示なら表示する |
| 右クリック | EvalPanelをpause/resumeする。resume時にpaneが非表示なら表示する |
| `Space` | EvalPanelをpause/resumeする。paneの表示は変えない |
| `Ctrl` | Evalを1 step自動実行する |
| 矢印キー | LunarLander向けに`0`から`3`のActionを指定してEvalを1 step進める |
| テンキー`0`から`9` | 同じ番号のActionを指定してEvalを1 step進める |
| `View > Evaluation View` | Eval paneの表示を切り替える |
| `View > Evaluation QValue View` | Q値paneの表示を切り替える |
| `View > Log Level` | GUIへ表示するログlevelを切り替える |
| `View > Reset Layout` | pane配置とframe sizeを既定へ戻す |
| Run操作ツールバーの`Save Checkpoint` | Trainが走行中ならまずpauseし、Run directoryと`agent_<exp_step>.anet`を既定に、任意pathへAgentを保存する |
| Run操作ツールバーの`Open Run Folder` | 現在のRun directoryをExplorerで開く |

Action数はEnvごとに異なる。範囲外Actionを前提にせず、QValue paneまたはEnvのActionSpecを確認する。

EvalPanelの共通既定は`app.eval_panel.eval_config_tag = eval_panel`で、`run.eval.[eval_panel].actor = eval_panel`から専用Actorを参照する。DQN系はtarget netのGreedy（Rainbowはε=0）、MuZeroは温度0・探索noiseなし、ImageClsは最大スコアのActionを選ぶ。定期評価のεやUQE設定はそのまま使える。表示用の方策を変える場合は`<Agent>.actor.[eval_panel].*`を上書きし、別の評価設定を表示したい場合は`app.eval_panel.eval_config_tag`を指定する。

`app.eval_panel.model_sync.mode`は`frame`、`time`、`episode`の周期で同期する。modelを複製するかは参照先の`<Agent>.actor.[key].clone_model`で選ぶ。sharedでも同期時に学習側countsを更新し、resume時にも同期する。表示中のEvalが常にTrainの最新parameterと一致するとは限らないため、比較時はActor名、clone設定、modeとintervalを記録する。

### 6.3 表示FPSと進行状況

`View > Train View FPS`はTrain Viewの描画頻度だけを変更する。`0 (Off)`では描画timerを止めるが、学習は継続する。`View > Eval View FPS`はEvalPanelのtimer周期を変更するため、描画だけでなくEvalの進行速度も変わる。どちらの`Config (N)`も起動時config値へ戻す項目である。これらは実行時UI操作であり、選択結果をRunのconfig dumpへ書き戻さない。

Step表示ツールバーは`exp`と`train`のstep数を別々のread-only text欄へ表示する。値は選択してコピーでき、exp/train間は標準separatorで区切られる。status bar右側は`exp <N> steps/s    train <N> steps/s`と経過時間を表示する。SPSのEMAがまだ初期化されていない間は`-`、最初のTrain snapshotが無い間は両step欄が`-`、経過時間が`--:--:--`になる。経過時間はpause中もwall-clockとして進む。

### 6.4 停止、保存、checkpointからの再開

WindowのCloseまたは`File > Exit`でRunを停止する。終了処理はTrain停止・join、`agent_close.anet`保存、background Observerの排水、Run出力のflush、GUI破棄の順に進む。`app.save_agent_on_close=false`のRunではこの保存だけを省き、他の順序は変わらない。保存中または排水中にprocessを強制終了するとcheckpoint、metrics、動画の末尾が不完全になる可能性があるため、windowが閉じるまで待つ。

`run.eval_schedule.[tag].wait_on_exit`は、終了時に進行中のbackground評価を完走待ちするかを指定する。既定は`true`で、全スロットが`app.drain_timeout_sec`の単一deadlineを共有する。`false`、手動終了のCancel選択、またはdeadline超過では次のEnv Step境界で協調キャンセルする。キャンセルしたセッションはsession scalarを出さず、完了済みepisodeのtraceと`eval.[<tag>].session_cancelled` JSONを残す。

手動のCloseと`File > Exit`では、Train停止・join後に待ちが必要なbackground評価が残っていれば、`Wait and close`、`Cancel and close`、`Keep running`の3択を表示する。`Keep running`は同じTrain threadを閉じる前のpause状態で再開する。`app.train_exit_step`または`app.exp_exit_step`による予算到達では質問せず、設定とdeadlineに従う。

`Save Checkpoint`は押下時にTrainが走行中なら先にpauseする。これはdialog操作中にstepが進み、既定ファイル名と保存内容がずれるのを防ぐためで、保存やcancelの後もTrainは自動再開しない。再開はRun制御ツールバーの`Train`か`Shift`で行う。保存処理自体はTrain走行中でも安全である。`DefaultDQNAgent`はserialization全体をAgentのshared lockで保護し、Learner更新と排他する。保存先の権限、空き容量、file lockなどで失敗した場合は対象pathと理由をerror logへ記録し、online構成ではダイアログも表示する。これはnon-fatalで、Runとprocess終了コードには影響しない。失敗したfileは不完全な可能性があるが自動削除されないため、内容を確認してから処理する。有効なpathを選べば再度Saveできる。

close時の`agent_close.anet`保存に失敗した場合も同じ表示方針で通知し、その後のlog shutdownとGUI cleanupを続行してwindowを閉じる。この場合、`agent_close.anet`は有効なcheckpointとは限らない。AgentがSaveを実装していない場合は0 byteのfileが残り、対象path付きのWARNが出る。保存できたcheckpointもAgent、Network、archive contractが一致することを確認してから再開に使う。

checkpointから再開する場合は、新しいRunの互換設定にAgent固有の`auto_load_file`を指定する。現行例は`DefaultDQNAgent.auto_load_file`のaliasである`R.auto_load_file`、または`ImageClsAgent.auto_load_file`である。Network構成やarchive contractが異なるcheckpointは読み込めない。保存対象はAgentごとに異なり、現行DQN系ではReplayBufferの内容やsampling状態を復元しない。再開後は新しいRun directoryとstep系列を持つため、旧Runの`metrics.jsonl`へ追記する操作ではない。DQN系の保存対象は[DQN系Agent](200_dqn_agents.jp.md)を参照する。

## 7. 成果物

workspaceが`dm_long`、`app.run_name=run_{t}`の場合、成果物は`apps/runner/workspaces/dm_long/runs/<run_name>/`へ保存される。絶対パスworkspaceでも同様に、そのworkspaceの`runs/`配下へ保存される。

| 成果物 | 内容 |
|---|---|
| `metrics.jsonl` | scalar、JSON metadata、動画metadataを追記する主メトリクス |
| `config/config_data.txt` | 解決後の実効設定。`@`プロファイルと`.$`は含まない([3.8](#38-確認方法とよくあるエラー)) |
| `config/*.txt`、`json/*.json` | コンポーネント別の注入済み設定・metadata dump。Envは`config/env.<Env name>.txt` |
| `<run_name>.log` | timestampとlevelを含むrunner text log |
| `stdout.log` / `stderr.log` | process標準出力・標準エラー |
| `agent_close.anet` | 正常なwindow close時に保存されるAgent checkpoint。`app.save_agent_on_close=false`では作られない |
| `videos/*.mkv` | image系Observerが生成した動画 |
| `images/<tag>/*.png` | `app.metrics_logger.use_png_dump=true`時の個別frame |
| `dot/**/*.dot` | GraphViz Observerの出力 |

比較や再現では、手元の編集前ConfigではなくRun directory内の`config/config_data.txt`を正とする。グラフ分析は[分析ユーザーガイド](030_user_guide_analysis.jp.md)を参照する。

## 8. よくある確認事項

- 起動直後にTrainが進まない: `app.train_auto_start`を確認し、左クリックまたは`Shift`でresumeする。
- Evalが進まない: `Evaluation View`を表示し、`app.eval_panel.auto_start`または`Space`を確認する。
- toolbarのcheck状態が操作と合わない: 最大200ms待って実状態への同期を確認する。配置が崩れた場合は`View > Reset Layout`を使う。
- Train Viewだけ更新されない: `View > Train View FPS`が`0 (Off)`になっていないか確認する。
- Saveに失敗する: error log（online構成ではダイアログも表示）の対象pathと失敗段階を確認する。権限、空き容量、file lockを解消するか別pathを選んで再実行する。Runは継続しているが、失敗した出力fileは不完全な可能性がある。
- Save結果が0 byteになる: `<run_name>.log`の対象path付きWARNを確認する。利用中AgentがSaveを実装していない可能性がある。
- Run folderが開かない: error log（online構成ではダイアログも表示）の対象pathとOS側のfolder関連付けを確認する。失敗後もRunは継続する。
- CUDA初期化に失敗する: libtorch/CUDA/driverの組み合わせ、`agent.device_type`、eval deviceを確認する。
- 期待したEnvでない: 選択workspaceの`config/_main.txt`で有効なEnv includeと、Run内`config/config_data.txt`を確認する。
- workspaceを選び直したい: `--select-workspace`で起動する。履歴は`GetAppDataDir()/history.txt`、ダイアログ選好は`prefs.txt`を削除すると個別にリセットできる。
- Viewが空: Log paneのEnv class ID、View factory、初期化errorを確認し、`Reset Layout`も試す。

## 9. 関連文書

- [CONTEXT.md](../../CONTEXT.md)(設定用語の正本)
- [ADR 0042](../adr/0042-config-inheritance-as-differential-base.md)(設定契約の判断理由)
- [AGENTS.md](../../AGENTS.md)「設定ファイルの代入演算子」(演算子の運用規約)
- [分析ユーザーガイド](030_user_guide_analysis.jp.md)
- [開発環境](040_development_environment.jp.md)
- [実行基盤と設定](100_runtime_and_configuration.jp.md)
- [Agentと学習](110_agents_and_learning.jp.md)
- [環境](120_environments.jp.md)
- [可観測性](140_observability.jp.md)
- [ReplayBuffer](150_replay_buffer.jp.md)
- [DQN系Agent](200_dqn_agents.jp.md)
- [アプリケーションとツール](160_applications_and_tools.jp.md)
