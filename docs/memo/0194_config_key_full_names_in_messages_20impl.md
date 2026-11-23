# PRD194 完全キーを使う診断の実装メモ

## 概要

[PRD194](0194_config_key_full_names_in_messages_10rfd.md) の3段階を実施する。エラー・WARN・INFO が設定を名指すときに、読む時点で記録した完全キーを使う。device 診断、指標名と用語を揃え、PRD §7 A1〜A14 を検証する。

## 主な変更

1. 不正な device 指定ではキーと値を含む fatal を1件出す。明示 CUDA が使えない場合は、全対象のキーと指定値を一括で報告する。ParseDevice は4か所すべてからキーを受け取る。和文の用語を「実効 device」に揃え、指標の期待を出荷定義の `19_skip_primary_clip_ratio_ema` に合わせる。
2. ConfigData に各項目の完全キーと置き場を持たせ、切り出し・コピー・スコープ・合成・Set で §6.2 の規則を適用する。照会は既読の追跡に影響させない。Config は dump とは独立して読み取った欄の完全キーを記録し、値を写す際には対応する記録も写す。空の上書き接頭辞では上書きの読み取り、存在確認、スコープ作成を省く。ConfigData 自身、MetricsLogger、DQN 共通部品へ適用する。D24 の3文は記録に無い場合だけ購読解除を案内する。和文設計と実装品質確認観点を更新する。
3. §8.3・§8.5・§8.6 と D20 で拾った診断を移行する。D21 の7箇所は、不正なユーザー指定から届く経路に対応する早期検査を置く。NN の内側の診断が完全キーを出してから投げ直し2箇所を外す。PRD193 の初期化診断も §6.7 に従って移行する。
4. §8.2 の集計と期待一覧、照合手順を `src/core/base/testdata/prd194/` に置く。起草時の基準 a6e920ed と §8.7 の152文・48行の期待を使い、更新した期待は行と判定根拠を記録する。

## テスト

観測の入口は InitRL、ConfigData と Config の読み取り、Runner の動画記録、Agent と Env の構築、NN の構築。次の順序で1挙動ずつ RED から GREEN へ進める。

1. 不正な device 指定の fatal 1件を最初に再現し、3つのキーを確かめる。明示 CUDA の指定値、ParseDevice の空キーを続ける。指標の4環境を再検証する。
2. 完全キーを引いても未読が維持されることから始め、切り出し・コピー・合成・スコープ・Set、記録の既定・上書き・欠落・別欄へのコピー、空の上書き接頭辞を順に確かめる。Runner の読み口を通る動画記録と手組み設定、DefaultDQNAgent と RainbowAgent の同じ診断を追加する。
3. 具象 Agent、評価 Env、共有・Agent 配下の NN block の不正設定を検証する。NN は型違い・未読キー・混在時の既定置き場も確かめる。Linear の不正設定で完全キー付き fatal が1件になることを確かめる。

## 検証

- 段階の切れ目で VsDevCmd.bat 経由の Debug ビルドと既定9テストを実行する。各テスト実行体の `[prd194]` を実行し、Runner のテストには AnetRLRunner-gui-test を使う。A2 は CUDA_VISIBLE_DEVICES=-1 を設定した別プロセスで実行し、skip を合格に数えない。
- `[shipped_config]`、`[prd072]`、prepare.py の compare による `[prd072-baseline]`、ANET_PRD103_OUTPUT を指定した `[prd103-config]`、S1・S2、旧表記の検索、集計の照合を実施する。結果・未実施の理由を受入項目に対応づける。

リポジトリルートの PowerShell で実行した主要なコマンド。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 12 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 12'
```

```powershell
cmd /s /c 'set "CUDA_VISIBLE_DEVICES=-1" && out\test\Debug\anet-core-base-test.exe [prd194]'
```

```powershell
.\out\test\Debug\anet-algo-dqn-test.exe '[prd194]'
```

```powershell
.\out\test\Debug\anet-env-gridmaze-test.exe '[prd194]'
```

```powershell
& .\apps\runner\bin\Debug\AnetRLRunner-gui-test.exe '[prd194]' --reporter 'JUnit::out=.scratch/prd194/a3-runner.xml'
```

```powershell
& .\.scratch\prd194\verify-suites.ps1 -VerificationId full -IncludePrdSuites
```

この実行記録のスクリプトは A11 に挙がる9実行体を引数なしで起動し、その後に `[shipped_config]`、`[prd072]`、固定入力の比較を実行する。個別ログは `.scratch/prd194/full-*.log`。

```powershell
.\.venv\Scripts\python.exe src/core/base/testdata/prd072/prepare.py compare
```

```powershell
$env:ANET_PRD103_OUTPUT='C:\dev\anet-lab\.scratch\prd194\prd103-after'; .\out\test\Debug\anet-algo-dqn-test.exe '[prd103-config]'
```

```powershell
.\.venv\Scripts\python.exe src/core/base/testdata/prd194/audit_config_messages.py --expected src/core/base/testdata/prd194/expected.jsonl
```

## 前提

- 着手時 HEAD は 3805a114。並行作業の PRD193 が 8c507264 にコミットされたため、実装レビューの差分は 8c507264 を基準にする。診断集計の起草時の基準は a6e920ed とする。
- PRD の課題7件を扱う。計画保存時に対応中とし、レビュー前に対応内容と結果を記録する。
- 実装・自己点検後、cc-review で Codex と Claude に同じ依頼を出す。修正と再検証をまとめ、同じ両モデルへ再レビューを依頼する。原文・差分・依頼は `.scratch/prd194/` に保存し、結果を本書と課題に記録する。

## 経過

- 起草時の基準 a6e920ed に集計を当てると、旧規則では117文、欄名を全対象ファイルから集めると295文だった。入れ子のあるログ呼び出し8文を拾うと303文・64呼び出しとなり、§8.3 のファイルと行がすべて一致した。§8.2 の記述を補正し、全対象の共通の集合を使うことを明記する指摘を修正した。PRD の修正レビューは OK。原文は `.scratch/prd194/review-20261121-1411-prd-1.md` と `.scratch/prd194/review-20261121-1411-prd-2.md`。
- A1 の RED では3キーすべてで例外ログが2件。修正後の `[prd194]` は3キーすべてで完全キー付きの例外ログが1件。9 assertions が通過した。
- `[prd103-config]` の RED は旧指標名の断定だけで4環境すべて失敗。現行の出荷定義に合わせて断定を修正した。
- A2 の RED は CUDA を隠した状態で指定値の断定1件が失敗。修正後は CUDA を隠した `[device]` の5テスト・46 assertions が通過し、指定値を含む一括エラーを確認した。
- カタログの型エラーは RED で `key=width`、GREEN で `key=net.block.[sample].width`。空の上書き接頭辞の RED では root の相対キーが読まれ、既定値・存在確認・読み取り範囲・未読キーの4断定が失敗した。完全キーの記録を足し、空の上書き接頭辞の読み取りを止めたあとは、`[prd194_config],[prd194_empty_override]` の4テスト・33 assertions が通過した。
- A4 の実装中に、RainbowAgent が Agent の既定の購読口を使用し、共通 Learner に購読を渡さないことを確認した。開発者は既存挙動の維持を選択したため、A4 と D24 の説明を補正した。テストで DefaultDQNAgent、RainbowAgent の既定の購読口、RainbowAgentConfig を受け取った共通 Learner をそれぞれ通る。
- Runner の A3 は既存の GUI テスト実行体で3 assertions が通過した。MetricsLogger に手で組んだ設定の診断と、実際に読み込んだ完全キーを使う診断も追加した。
- A6 は `[prd103-config],[prd194_dqn_config]` の 10,194 assertions が通過した。GridMaze の RED では固定ルートと相対キーが同じエラーに混在した。DQN の PER の RED は両 Agent で `learner.per_eps` を表示し、移行後に両方の完全キーを確認した。Linear の RED では同じ例外の fatal が二重に出た。
- A4 と §8.2 の PRD レビューでは、冒頭の A4 変更記録と、集計の第1段階でも `GetFullKey` の引数を除く条件を追記した。修正レビューは OK。原文は `.scratch/prd194/review-prd-a4-1.md` と `.scratch/prd194/review-prd-a4-2.md`。
- A4 の S1 は DQN の Agent 登録2行だけ。A5 の S2 は和文6行に一致した。A9・A10 は期待の156文と49呼び出しに一致した。

### 実装レビュー前の検証

- MSVC の `VsDevCmd.bat` 経由の Debug 全体ビルド、および `AnetRLRunner-gui-test` のビルドが成功した。PRD194 の Core は11テスト・62 assertions、DQN は3テスト・8 assertions、GridMaze は1テスト・1 assertion、Runner GUI は1テスト・3 assertions が通過した。Core は `CUDA_VISIBLE_DEVICES=-1` で実行し、CUDA を隠した A2 を通過した。
- `[prd103-config]` の単独実行で10,191 assertions が通過した。a6e920ed の集計はエラー249文・INFO31文・WARN23文と64呼び出し。Git の差分検査、改行、BOM の検査に不一致は無かった。
- 既定の Core テストで、評価ログ、ProfiledValue、重み初期化、ResBlock の診断を旧表記で比較していた断定を完全キーの期待へ更新した。並行作業の `apps/runner/config/common.txt` では `app.main_view ?= train` が有効で、コミット済みの出荷設定は `eval`。作業ツリーの設定を使う `Main View common default is Eval` はこの差分で失敗する。修正した既存テストと A11 の再検証を実施中。

### D21 の呼び出し元の確認

- Noisy の `num_draws` は DefaultDQNAgent の設定読み込みで正数を検査する。RainbowAgent はこの欄を読まない。
- `GenerateTaus` の `batch_size` は観測やサンプルの行数から導く。設定できる `num_taus` と `sample_mode` は DQN の方策・Learner の設定読み込みで検査する。
- `IntervalGate` は、画像・評価 Observer が間隔を正数に限って構築し、指標の設定読み込みでも範囲を検査する。
- `SampleUniqueUniform` の `batch_size` は、DefaultDQNAgent が診断ごとの設定読み込みで範囲を検査する。
- `ImageDataSource` の `batch_size` は `run.eval.[tag].eval_batch_size` の既存の正数検査と、新たに加えた `run.train.num_envs` の正数検査で、構築前に不正値を止める。

### 集計の期待を変更した行

| ファイルと移行後の目印 | 集計に残る語 | 判定 |
|---|---|---|
| `default_dqn_agent.cpp` の checkpoint の価値分解と勾配の不一致 | `value_mode`、`feature_gradient`（別の2文） | R2。checkpoint に保存された値のラベル。現在の設定値は、それぞれ完全キー付きで表示する |
| `dqn_learner.cpp` の Noisy の sample forward が無い場合 | `config.{}` | R6。文末の `config.` と後続の文面をつなぐ区切り |
| `intrinsic_rnd.cpp` の `WeightInitConfig::ReadFields` | `head.init` | PRD193 が足した呼び出し。D20。設定を読むための相対キーで、診断には個々の欄の完全キーを表示する |
| `intrinsic_rnd.cpp` の checkpoint の統計方式の不一致 | `statistics_mode` | R2。checkpoint に保存された値のラベル。現在の設定値は完全キーを使う |

### 固定ルートの数え直しと一覧外の診断

§8.6 の Env の文字列を基準コミットで数え直した。C++ の include 行も起草時の件数に入っていた。診断か診断へ渡すキーを数えると、AtariEnv.cpp は5行、GridMazeEnv.cpp は12行、LunarLanderEnv.cpp は4行、DropMergeEnv.cpp・ImageClsEnv.cpp・CartPoleEnv.cpp は0行だった。ヘッダでは ImageClsEnv.hpp が1行、LunarLanderEnv.hpp が3行。CartPole は固定ルートを案内する診断が無かった。NN の5行はプロファイルの type と end、block 欄の診断2行、未定義グループの案内だった。

| D20 で追加確認した場所 | 完全キーへ直した診断 |
|---|---|
| DQN の具象 Agent と共通部品 | Munchausen の下限と組合せ、skip の方式と参照、intrinsic の休眠・境界、Actor の選択、tau の値域と方式、QR の前提、PER の優先度、optimizer の checkpoint 設定差、初期割引率、重み初期化。Rainbow の Actor の network と policy_type、QR の条件も読み口に合わせた |
| NN と組み立てる側 | 枝内プロファイルの未定義、必須欄、未読の各診断。bind の空要素、Linear の forward 再利用時のモード、Projection・Spectral Norm・初期化で扱う値 |
| 可視化と指標 | Conv2d の余白・チャンネル行・最小寸法、Sweep の観測と座標、出力キー、指標とグラフの間隔、評価の間隔と待機条件 |
| 正規化と報酬 | ObservationNormalizer の範囲と後処理、RewardScaler のモードと自動スケールの組み合わせ、RND の頭部、warmup、統計、optimizer 設定 |
| ImageCls・MuZero と Env | ImageClsAgent の mixup・interval・可塑性、MuZero の Actor 温度と hidden state、画像データセットの項目・path・cache・回転窓・worker の診断 |

D19 では、DQN の `intrinsic.reward_scaler kind` を checkpoint の状態種別を示す R2 と判定した。ProfiledValue の `min_value` と `max_value` は API で渡す境界のメタデータなので、設定ファイルの所属を組み立てずに案内する。分割 optimizer の `primary_skip` に適用する weight decay はコードが固定する0であり、設定値として扱わない。

### 公開ヘッダの追随

- `WeightInitConfig`、`ObservationNormalizerConfig`、`Conv2dVisualizerConfig`、`ProfiledValuePhaseConfig` は、別の部品に値を渡すため、欄単位の完全キーを運ぶ。各 `Validate*Config` から重複する接頭辞の引数を外し、値の出所を記録から引く。ProfiledValue の範囲検証と WeightInit の `kDefaultOwner` も同じ理由で見直した。
- `RewardScalerFactory::CreateRewardScaler` は呼ぶ側で分かる割引率のキーを受け取る。 `StateSweepProcessor` は観測・軸、`GraphVizObserver` は宣言の完全キーを受け取る。 `MetricsLogObserverBase::ValidateMetricsInterval` は宣言の読み口と直接構築で共有できるソース内関数に移した。
- `AgentBase::FindActorConfig` は具象側の接頭辞を受け取り、Actor の項目を識別する。 `NetworkConfig` は、共有・ネットワーク固有・branch のプロファイルの置き場を案内するためネットワークの接頭辞を持つ。
- `ConfigData::UseLocationOf` は、合成したカタログ項目で `type` が選ぶ置き場を移すために使う。 `Config::ReadInheritedSubConfig` は明示上書きが無い値で、DTO の継承元を維持する。

### 初回 cc-review の対応

- 両モデルは設計文書のフラグの表記を指摘した。Codex は枝内プロファイルの置き場、初期割引率、固定の optimizer 値、検証手順を指摘した。Claude は一覧外・固定ルートと公開ヘッダの記録、完全キーへの移行漏れとテスト、命名・コメントを指摘した。原文は `.scratch/prd194/review-20261121-1526-impl.md` と `.scratch/prd194/review-20261121-1526-impl-claude.md`。
- A8 の (1) では、`DefaultDQNAgentConfig` が不正な `learner.munchausen.clip_value_min` の設定を読んだ際に、値を与えた `Trial.learner.munchausen.clip_value_min` を報告するテストを追加した。共通部品を通す A8 の (2)・(3) は PER の検査で確認する。
- 初回の A11 で ImageCls Agent、MuZero、Atari、GridMaze、ImageCls Env、LunarLander と `[prd072]`、固定17入力の比較が通過した。評価・NN・DQN・DropMerge で診断の旧表記を比較した既存テストを更新した。Core は作業ツリー固有の Main View テスト1件を除き660件通過、1件スキップ、33,806 assertions が通過した。`[shipped_config]` は同じ1件が失敗し、残る8件は通過した。
- Main View の出荷既定は、8c507264 の `common.txt` を別のファイルへ取り出し、現行の `ConfigManager` で同じ欄を読んで検証した。`app.main_view=eval` で終了コード0。コマンドは `& .\.scratch\prd194\proof-main-view.ps1`、ログは `.scratch/prd194/proof-main-view.log`。

### 修正レビューの対応

- Codex は計画を OK とした。実装では、checkpoint 読み込み時の学習率と通常の weight decay が、スケジュールの開始値でも裸のキーで通知される点を指摘した。初期割引率、学習率、通常の weight decay で、定数の値、時間プロファイルの `start`、先頭 phase の値の位置を同じ内部関数から得る。sigma の `noisy_weight_decay` と `primary_skip` の固定0は個別に表示する。レビュー原文は `.scratch/prd194/review-20261121-1549-impl.md`。
- checkpoint テストでは、実際の設定読み込みを経た linear の学習率と phased の weight decay で、checkpoint との差分と適用後の各通知を、moment と step の保存と合わせて確認した。DQN の `[prd194]`、optimizer 復元、skip 移行、`[prd103-config]` をまとめた実行は9テスト・13,328 assertions が通過した。最終 Debug 全体と Runner GUI テストのビルドも成功した。
- 再実行した既定スイートでは ImageCls Agent、MuZero、Atari、DropMerge、GridMaze、ImageCls Env、LunarLander が通過した。`[prd072]` と固定17入力の比較も通過した。
- `Env bases bind immutable names to protected loggers` は seed `881592696` の Core 既定スイートで、3件を期待する capture に7件が入った。単独および直前の RND テストとの組み合わせでは通過した。同じ seed の再実行では workspace 履歴のファイル置換も一度 `Access is denied` で失敗し、単独再実行で通過した。Core のログは `.scratch/prd194/rereview-anet-core-base-test.log`、`.scratch/prd194/final-core-fixed-order.log`、`.scratch/prd194/final-env-logger-neighbor.log`、`.scratch/prd194/final-workspace-mru.log`。
- 8c507264 の出荷設定と検証入力を `.scratch/prd194/a11-shipped-fixture/` に置き、現在の Core テストのうち `__FILE__` から出荷設定を探す翻訳単位をそのディレクトリでコンパイルした。独立した実行体の既定は662件通過・1件スキップ、33,810 assertions。`[shipped_config]` は9件・954 assertions すべて通過した。作業ツリーの設定はそのまま。ログは `.scratch/prd194/fixture-core-default.log` と `.scratch/prd194/fixture-shipped-config.log`。
- 独立した実行体で seed `881592696` を指定し、capture の余分な4件が先行テストの背景評価の終了・取消・例外通知であることを特定した。Env のログテストは capture 開始前に保留通知を流す。同じ seed で、作業設定を参照する Main View テスト1件を外した既定スイートは661件通過・1件スキップ、33,809 assertions。再現時と修正後のログは `.scratch/prd194/fixture-core-env-seed.log` と `.scratch/prd194/final-core-isolated-capture.log`。
- DQN 既定の再実行で、Rainbow Actor の直接構築でポリシーや tau の不正値を渡した場合と、dedicated skip の複数の必須キーを欠いた場合の旧表記を修正した。設定を経由する場合の完全キーと直接構築する場合の相対キーと注記を、両方のテスト経路で比較する。
- 最終 Debug 全体と Runner GUI テストを再ビルドした。DQN の追補を含む対象テスト11件・3,163 assertions、GridMaze の対象1件・1 assertion、Runner の対象1件・3 assertions が通過。集計は205行一致、Git の空白88ファイルと UTF-8・LF・BOM 93ファイルに不一致は無い。
- Codex の修正レビューは計画・実装ともに OK。既定テストで追加した期待の更新と Env ログの捕捉前のフラッシュも OK。原文は `.scratch/prd194/review-20261121-1600-impl.md`、`.scratch/prd194/review-20261121-1606-impl.md`、`.scratch/prd194/review-20261121-1611-impl.md`。
- 最終構成の Core をコミット済み設定で再実行した。既定の乱数 seed と再現用 seed `881592696` はどちらも662件通過、1件スキップ、33,810 assertions。`[shipped_config]` は9件・954 assertions。コマンドは `& .\.scratch\prd194\check-core-shipped-fixture.ps1 -VerificationId final-fixture`。ログは `.scratch/prd194/final-fixture-core-default.log`、`.scratch/prd194/final-fixture-core-env-seed.log`、`.scratch/prd194/final-fixture-shipped-config.log`。
- 最終構成の DQN 既定スイートは引数なしで370件・161,188 assertions が通過。既定9実行体は、Core をコミット済みの出荷設定で検証した結果を含めすべて終了コード0。作業ツリーを直接参照する Main View の検査は、並行作業で指定された `train` により失敗する。DQN のログは `.scratch/prd194/verified-dqn-default.log`。

### Claude の修正レビューと追補

- 開発者の明示承認を受け、Claude Fable 5.2 の同じセッションで修正レビューを実施した。計画は OK。実装では、Actor の `action_beta_mode`、skip の `max_length_mode` の2文、IQN の `quantile_mode`、Atari の `resize_mode` に、まだ相対キーが出ると指摘された。レビュー原文は `.scratch/prd194/review-20261121-1600-impl-claude.md`。
- 条件も値を読んだ完全キーで示す。値が未指定だと optional の読み口は記録を持たないため、必須の Actor β は `ConfigData` から宣言先を案内する。skip の未使用通知では範囲を導いたプロファイルの値または種類の位置を示す。D20 に従い、固定ルートが残った `intrinsic.use_done_mask` と、Factory の `quantile_mode`・`bind` の診断も、値の読み取り元を使うよう直した。DQN、Atari、Runner、基盤、Env の C++ ソースで `when`・`with`・`expected` などに続く裸の `名前=` を確認した。残る箇所は実行時の形状、件数、モード、導出した範囲の表示だった。
- 修正前の診断では、DQN の4テストで7 assertions、Atari のサイズテストで2 assertions が完全キーの不足で失敗した。Actor の β を調べるテストでは先に名前付き forward の選択を整えた。IQN の κ の検査では、値の読み取りで止まる `nan`・`inf` と、組み合わせの検査で止まる有限値を分けた。テストは `CopyValueFrom` を経た設定の位置と、個別に設定した画面サイズと共通の resize mode の双方を確認する。RED のログは `.scratch/prd194/claude-followup-red-dqn.log`、`.scratch/prd194/claude-followup-red-atari.log`。
- 追補後も Debug 全体と Runner GUI テストのビルドが成功した。Atari の既定は56件・1,606 assertions が通過。A9/A10 の集計は205行一致し、Git の空白88ファイルと UTF-8・LF・BOM 93ファイルを確認した。ビルドと Atari のログは `.scratch/prd194/claude-followup-green-build.log`、`.scratch/prd194/claude-followup-green-atari.log`。
- 追補後の DQN の対象15件・195 assertions が通過した。未指定の Actor β を書く完全キーと、条件の読み取り元の完全キーの両方を検証した。ログは `.scratch/prd194/claude-followup-final-build.log` と `.scratch/prd194/claude-followup-final-dqn-targeted.log`。
- 同じ依頼 `.scratch/prd194/request-20261121-2005-impl.md` で追補を再レビューし、Codex と Claude はともに実装計画・実装を OK とした。原文は `.scratch/prd194/review-20261121-2005-impl.md` と `.scratch/prd194/review-20261121-2005-impl-claude.md`。
- 最終実行体で `[prd103-config]` を改めて実行し、4環境で10,191 assertions が通過した。ログは `.scratch/prd194/claude-followup-prd103-config.log`。
- 最終差分で DQN の既定スイートを引数なしで実行し、372件・161,192 assertions が通過した。ログは `.scratch/prd194/claude-followup-full-dqn.log`。Codex・Claude の再レビュー後に残る修正要求は無い。作業ツリーでは並行設定の Main View が `train` のため、Core の出荷既定値はコミット済み設定を使った独立実行で検証した。

### 追加の確認事項

- 和文の設計文書 140 の評価セッションの接頭辞、開始・終了の行、起動時の `scheduled` 行を実装の完全キー表記に揃えた。
- ImageCls のデータセット参照の診断では、設定に書いた値と存在しない `ImageDataset` 項目を分け、選択できる値をそのまま示す。評価 Env の上書きキーから診断を出し、候補を同じキーに設定すると構築時の検証を通ることを確かめるテストを追加した。
- ` (not read from config)` の生成を `ConfigFullKeyTable` に集約した。キーを受け取る API では既定引数を外し、設定を読まない呼び出しでは明示する。spectral normalization は ResBlock の `conv2` を初期化するときだけ `init2.mode` の案内を渡す。
- CUDA を隠した A2 のテストに `env.device` を戻し、3キーの指定値をまとめて確認する。
- Debug 全体と Runner GUI のビルド、Core の対象77件（2件スキップ）、DQN の対象56件、ImageCls Env の既定20件、Runner GUI の対象1件が通過した。A2 のログに `agent.device='cuda:1', env.device='cuda:2', run.eval_device='cuda'` が出た。集計は205行で一致した。本番コードに注記のリテラルがあるのは `ConfigFullKeyTable` の1か所だけ。ログは `.scratch/prd194/post-stage-review-*`。
