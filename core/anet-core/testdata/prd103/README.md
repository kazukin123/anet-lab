# PRD103 検証資材

正本は[PRD103](../../../../docs/memo/103_temporl_reference_alignment_10prd.md)、実施記録は[実装メモ](../../../../docs/memo/103_temporl_reference_alignment_20impl.md)。実行はrepo rootのPowerShell。短縮受入から長時間の学習効果を判断しない。

## 入力と等価性

`capture_baseline.py`で変更前の実行体・DLL・設定・解決結果84 filesを保存済み。基準HEADは`e0fc65f401844a3e6aeca3d1b92c3c2bd0a3f386`。各SHA256は`baseline-manifest.json`へ保存。baseline-bin、baseline-config、resolved-beforeはローカル資材としてGit管理しない。既存保存先の上書きは拒否する。

`run_validation.py`は8 lane、batch16、warmup256、capacity4096、RR1、seed1、8704 experience、FP32、決定論的backend、評価停止でbefore/afterを実行する。Atariはqbert+BTR、DropMergeはnzrep。`90_perf/`・`91_resource/`・`92_resource/`だけを除外し、Noisyを含む全scalarのtag・step・値を比較する。skip ONもenv変更前の入力を両実行体へ渡す。

2026-10-28、6構成すべてexit 0・全点完全一致。

| 構成 | 点数 | 正規化SHA256（before=after） |
|---|---:|---|
| GridMaze OFF | 26946 | `2278b59c28b93e85d1adbfe203a1548a60589715fbdf576b029b1c9b3d800f98` |
| LunarLander OFF | 24399 | `941fa28b97b6a5833504c7a7b314bbc9966e77d6f66ff5edb8e4487e2a5a5a6c` |
| Atari BTR OFF | 23185 | `c7653fe5692ce1aaea960f9c3b4ccfe3833793e698e0b7c2d1ab2011deb8bd65` |
| DropMerge nzrep OFF | 31152 | `6cd959f2df2406a3c32d8371eaed17fddef5e50bcd8e19dec976fa8f2da672c9` |
| GridMaze ON | 37977 | `30eac6139fc315b200f6c341c8afbddc04255aa274c7b5c3479080dc0a41bd36` |
| LunarLander ON | 34628 | `7749bcfc67280271a3fbd2871995921bcba9ef9857a08ab85d35828cc5a4f70f` |

引数・実行体hash・Run名・正規化結果は`.scratch/prd103/validation/{before,after}_*/`。再採取は新しいlabelを使う。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd103/run_validation.py compare --label after
```

`verify_config_migration.py`は`.scratch/prd103/config-{before,after}/`の4 env×default/skipを全キー・文字列値で比較する。8構成とも意図しない差分0。許容差はsample_weightの既定、新しいgreedy設定と指標、GridMazeのskip方策、Atariのsession名とtag添字の移行だけ。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd103/verify_config_migration.py
```

`check_default_leaves.py`には変更前からAtariの`app.online.eval_panel.fps ?= 14.5`と`app.online.train_panel.fps ?= 10`の2指摘がある。

## 数値・公開経路

`[prd103]`は独立Adam更新の和、3集合のclip、decay一回、fused/標準Adam、checkpoint継続・役割移行・options/lr、kmax×ISとTD不変、ActorのSyncと複数lane、列挙値、TD/QR/IQN×Munchausen×Noisy/BF16、QだけのGNSを確認する。`[prd098]`・`[gradnoise]`・observer系を関連回帰に使う。

`[prd103-config]`はANET_PRD103_OUTPUTへ4 envのdefault/skipを保存するhidden test。`[prd103-resolve]`はANET_PRD103_INPUTを解決しANET_PRD103_OUTPUTへJSONを保存し、Python driverが呼ぶ。`[prd103-arms]`はT3/T4を構築しIQN Actorのforward・shapeを確認する。

## 現行設定の新基準

`run_current.py smoke`は実際の入力をresolverへ通して評価sessionを列挙する。smokeだけgreedy interval256・batch1・episode1・同期実行とし、Atariは評価上限400 emulator frames。通常の53/1009/503・backgroundは設定テストで確認する。`metrics.scalar.defs`のrunner=train・step_axis=exp_step、実際のscalar点、config_dataのactor/sessionを照合する。

| 保存先（`.scratch/prd103/current/`） | 点数 | 正規化SHA256 |
|---|---:|---|
| smoke1_0_GridMaze | 38388 | `be84fe789cacf90f9a914225d030db9610afb5c786387197db09d579402d35fe` |
| smoke1_1_LunarLander | 34658 | `c1200a2ee238b464f30cecd5f4a4ca4437c9f887f793b5d6287d4e8472931a85` |
| smoke1_2_Atari | 39750 | `9ddd7817eca0e692e76d23457a80ab3f59112ec67aea49fc16ae961ec9dd13e4` |

新基準は旧設定との一致条件ではない。各execution.jsonに実際のRun名と指標の先頭・末尾・件数を残す。scalar点そのものにはrunner欄がなく、定義メタデータで確認する。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd103/run_current.py smoke --label smoke1 --audit
```

## 性能

Release、Atari qbert+BTR、skip ON、N128/B256、BF16、prefetch、warmup256、RR1、capacity4096、82176 experience、評価停止。merge→split→split→mergeの4 Runで、`90_perf/98_prd098_time`のlearn step64～256の差を192で割る。比はsplit2本の合計／merge2本の合計。閾値は設けない。

`performance-onoff`は同じ条件で、skip有効（merge）→無効→無効→有効の4 Runを測り、有効2本の合計／無効2本の合計を`on_off_ratio`、1.3以下かを`on_off_within_1_3`に残す。PRD098の受入4（有効/無効が1.3倍以内）をこの条件で測るためのもの。PRD098の検証資材の`performance`は、評価を止めるsessionの一覧を2026-10-24の解決済み設定（`greedy_dist`を含む）から取るので、今の設定では使えない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd103/run_current.py performance-onoff --label <新しいlabel>
.\.venv\Scripts\python.exe core/anet-core/testdata/prd103/run_current.py performance --label <新しいlabel>
```

2026-10-31 の実測（今のソースの Release、`AnetRLRunner.exe` の SHA-256 先頭 `a18f17d3cb68c523`。ほかの Runner・ビルドが無い状態で、label ごとに 4 Run）:

| label | 順 | 秒／更新 | 比 |
|---|---|---|---|
| `perf1031-split` | merge→split→split→merge | 0.05106・0.05711・0.05721・0.05132 | split/merge 1.117 |
| `perf1031-onoff` | 有効→無効→無効→有効 | 0.06257・0.04135・0.04063・0.05054 | on/off 1.380 |
| `perf1031-onoff2` | 同上 | 0.05900・0.04520・0.04610・0.05950 | on/off 1.298 |
| `perf1031-onoff3` | 同上 | 0.05890・0.04680・0.04680・0.05930 | on/off 1.263 |

Run は `.scratch/prd103/current/runs/` の `run_20261031-18*_t103_perf1031-*`。`perf1031-onoff` の最初の Run はビルド直後の最初の Run で、同じ構成のほかの 5 本より遅い。on/off の 3 巡の構成ごとの中央値は有効 0.0591・無効 0.0456 で、比は 1.296。

## 構造の腕と長時間実験

`arms.txt`をAtari.txtの後でincludeし、run.@va_base > run.@skipの後にrun.@skip_conv_shared（T3）かrun.@skip_conv_separate（T4）を置く。T3はprimaryの畳み込みtrunkだけを公開し、skip側に独立Linear512/LN/ReLUとtau融合を持つ（merge）。T4はgridから同じ畳み込みと射影・tau融合を作る（stop）。既定の構造は変えない。branch名・構造が異なるため既定checkpointとは互換にしない。

`prepare_experiments.py`は段1～4の24入力を`.scratch/prd103/experiments/`へ生成・解決済み（errors=0、学習Run開始0）。recipes.jsonに実行コマンドと前提条件を保存する。

最終Debugの関連回帰は87 test cases / 36239 assertions成功、T3/T4は1 case / 15 assertions成功。証跡は`.scratch/prd103/final-tests.json`と`final-*.log`。性能4条件も`performance --label perf-ready --prepare`で解決成功し、`current/perf-ready-summary.json`へ保存済み。

初回レビューで実験レシピのmetrics選択による診断群欠落を修正した。現在の24入力は、診断追加前のRunをreference.jsonへ解決し、追加後もその全scalarキー・文字列値を保持する検査をすべて通過した。AtariのMunchausen/replay-fitとLunarLanderのMunchausen群を保持する。修正前のローカル証跡はexperiments-review1に保存。

| 段 | 腕 | seed・予算 | 読み方 |
|---|---|---|---|
| 1 | stop、merge+loss_weight0.3、merge+clip30（T2） | seed1・50M | 既存／実行中Runを使い段3の腕を選ぶ |
| 1' | GridMaze 6×10 | 実施済み3 seed・500k | PRDの結果を保持。再実行しない |
| 2 | merge、split、merge+kmax、skip epsilon-greedy（T10、spatial 0.3→0） | 3 seed・2M | policy報酬AUCとgreedy判断率・長さ別率 |
| 3 | split、条件付きsplit+kmax、条件付きmerge seed2 | 各50M | kmaxは段2で基準以上、seed2は段1でばらつきが候補の場合だけ |
| 4 | merge、T4非共有、T3畳み込み共有 | 2 seed・10M | 評価全体・評価7.5～10M・学習7.5～10Mと実時間 |

既定採用の条件、ビルド差の扱い、機序の観察はPRD §7・§9に従う。split/kmaxの既定採用と性能閾値は未決のまま保持する。
