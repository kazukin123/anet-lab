# 設定体系再設計: 概念ツリー整合・素材宣言・ConfigResolver PRD

> **用語**: 本書の「素材」は現在の呼称で **プロファイル(設定プロファイル)**、「幹 / named 幹」は **Run プロファイル** に相当する。正本は `CONTEXT.md`。本書は設計時の記録として旧称のまま残す。

> 番号 059(旧 931。実装移行に伴い暫定番号を解除、2026-08-23)。
> 状態: **v3 = grill 完了・最終化済み**(2026-08-23)。goal anchor 合意 → scope screen → 主質疑 → final simplification pass の全裁定(§複雑性監査)を反映。**残裁定なし**(遅延ゲートのみ=§10)。次アクション: PH0 の実装依頼(本 PRD は分割しない。実装計画は依頼時の PH 指定に応じて `059_*_2ximpl_phN.md` として impl 側で分割する)。
> v2 = 3者レビュー(設計スレッド / Atari 側 / DropMerge・Optuna 担当)合意版。
> v1(問題の一般化・選定なし)の分析は付録 A〜D に保全した。決定の根拠として本文から参照する。
> 起点: QR/IQN の切替が `net.$` と `R.quantile_mode` の 2 行セット操作になる件。一般化すると「1 つの概念の切替が複数 namespace の同時編集を要求し、揃え忘れると起動時エラーか静かな不整合になる」箇所が 12 件ある(付録 B)。
> 関連: [adr/0018-iqn-via-bind-product-dag.md](../../adr/0018-iqn-via-bind-product-dag.md)、[adr/0027-eval-definition-schedule-separation.md](../../adr/0027-eval-definition-schedule-separation.md)、[adr/0021-run-classification-by-workspace-folder.md](../../adr/0021-run-classification-by-workspace-folder.md)、[029_config_profile_param_interp_10prd.md](029_config_profile_param_interp_10prd.md)、[033_imagecls_bf16_head_10prd.md](033_imagecls_bf16_head_10prd.md)、[052_eval_schedule_separation_10prd.md](052_eval_schedule_separation_10prd.md)、`CONTEXT.md`「Module Config」「Property」「configured eval tag」「eval schedule」。
> 概念図(2 ビュー: 構造ツリー / 合成フロー): https://claude.ai/code/artifact/695e8d24-d520-4096-b33d-0779ed1d2494
> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。本書は self-contained。実装時は行番号ではなく、近傍のシンボル名・キー名で再検索する。

---

## 0. 目標と非目標

### 0.1 目標

1. **標準の素材選択経路では、概念を構成する一部だけを選べないようにする。** 例: IQN を選んだら agent 側スイッチと NN 配線は 1 つの選択で同時に決まり、片方だけを選ぶ書き方が標準経路に存在しない。
2. **直接 override による契約違反は、発生点に近い早い段階で fail-fast する。** 原因の config キーを名指しするエラーにする(現状は NetworkModel 構築まで潜伏し、shape エラーで間接検出される。付録 A)。
3. **Run artifact から「実効値」と「選択経路」を両方読めるようにする。** 実効値は純化した `config_data.txt`、選択経路は新設の `json/config_resolution.json` と Metrics master の `config_resolution` record が持つ。
4. **Run を「幹」(= Run 署名)の単位で定義・切替・記録できるようにする。** 幹はスロット選択の束であり、同一幹×複数 seed = 比較母集団という運用を構造で表現する。

### 0.2 非目標

- flat string config と任意 leaf override の廃止(これらを残す限り「不正な組み合わせを完全に表現不能にする」ことは原理的に不可能。目標 1+2 が達成可能な上限である)。
- 付録 B の 12 件すべてを単一機構で解くこと(値同期型は変数、選択型は素材、導出型は人間+コメントに役割分担する)。
- プリセットの見た目が綺麗になること自体(得たい価値は誤構成の排除と Run 分析の明確化)。
- 過去 Run artifact の書き換え(当時の記録として保持。読み側ツールが新旧両形式を読む)。

### 0.3 受け入れ指標(成功の測定)

1. **アルゴ切替の編集行数**: 現状 2〜4 行×複数章の同期編集 → チェーン 1 項の変更のみ。
2. **RUN_BUDGET 型の「静かに間違う」事故の再発ゼロ**: bat の予算値と設定キーの二重指定齟齬が構造的に起きない。
3. **inspect_run.py の module dump 突合の削除**: 未選択素材が dump から消えることで「module dump と突合して実効らしい key を判定する」処理が不要になる(削除行数で測定)。**達成済み(2026-08-25)**: env 素材に加え backend / metrics 素材の `@` 化(D26)で③無印素材は完全消滅した。dump に残る非実効行は②無印上書き層(`<対象文字><番号>` パターン、D20 改)と mode(選択の源、D25)のみで、いずれも既知 prefix として機械識別でき突合を要しない。突合コードの実削除は inspect_run 側の後続作業。

---

## 1. 決定の要約

| # | 決定 | 概要 | 節 |
|---|---|---|---|
| D1 | 3 層モデル | カタログ(参照される部品)/ 素材(named 選択肢+nameless 上書き層)/ 幹(Run 署名) | §2 |
| D2 | 幹の頂点は `run.`、workspace は設定の外 | workspace は Run フォルダ操作で管理する箱。設定体系に含めない | §2.2 |
| D3 | `train.seed` は幹の外 | 同一幹×複数 seed=比較母集団。反復軸を署名から除外 | §2.2 |
| D4 | named 幹は「使う構成にだけ命名」 | 全組み合わせの事前定義(直積)は禁止。LightZero 型のファイル爆発を避ける | §2.3 |
| D5 | `@` セグメント=素材宣言(機構非強制) | dump 除外+参照 0 件 fail-fast の特典と、読み口に使わない義務のセット。Resolver は無印 prefix 合成のまま | §4.1 |
| D6 | チェーン項の解決規則 | `@` 始まり単独項=LHS 配下の相対参照 / それ以外=従来どおり root 絶対。後方互換完全 | §4.2 |
| D7 | `${フルキー}` 値参照 | 任意キーの解決値を参照(同期のみ・導出なし・1 段制限)。定数の置き場は root 素材 `@vars.*`(命名慣習であり予約 namespace ではない) | §4.3 |
| D8 | `:` 糖衣 | Key 部の Key/SubKey 区切り。`.` と完全等価・意味ゼロ。正規形はドット連結 | §4.4 |
| D9 | 競合は全順序で決定的に解決 | デフォルト直書き < 解決の産物(チェーン等)< CLI 第 2 相。同種内は読み込み順後勝ち。**fail-fast も WARN もなし**(「独立 writer」検出は廃案) | §4.5 |
| D10 | 軸選定の 3 条件基準 | ①値 2 つ以上 ②1 選択が 2 箇所以上を同時に変える ③揃え忘れると壊れる or 静かに間違う | §3.1 |
| D11 | RUN_BUDGET / LEARNING_SCHEDULE 分離 | 予算と学習 schedule は連動しない(DropMerge 実測根拠)。update_warmup は両方から除外 | §3.3 |
| D12 | ALGO 軸=`@dqn/@qr/@iqn`(配線とセット、BODY と分離) | 分布表現の 3 値。quantile_mode+NN 配線+tau 系を 1 素材に。**「Rainbow」素材は作らない**(baseline+`@qr` の俗称。文書・named 幹名でのみ使用)。BODY は独立軸のまま素材化しない | §3.2 |
| D13 | ConfigResolver 化+依存グラフ DFS | 不動点反復ではなく selection 参照グラフを DFS 解決。5 フェーズ。既存 `GetConfigData()` は不変、resolution 記録の取得 API を新設(§5.1) | §5 |
| D14 改 | CLI override の 2 相化(両相適用) | **CLI はすべての設定ファイルより優先**。全 CLI override を第 1 相(source map 先出し=源プレフィクス形も選択に乗る)+第 2 相(selection・素材キーを除く実効 leaf 上書き=最強)の両方に適用。v3 の「種別で分ける」は源プレフィクス形の静かな無効化(25impl 申し送り 7)で改訂 | §5.3 |
| D15 | dump 純化+`config_resolution.json` | `config_data.txt`=実効値+無印上書き層(D20 の意図的残置)。`@` 素材は除外。選択・変数の記録は構造化 JSON へ。コメント行方式は不採用。純化の完成は Phase 1 の素材 `@` 化完了にゲート | §6 |
| D16 | optuna 運用 3 規則 | trial の `.$` 振りは HEAD/BODY=可、ALGO=別 study、BUDGET=同一 study 禁止。preset 名を trial parameter に保存 | §7 |
| D17 | golden comparison で移行検証 | 代表 config の旧/新 resolver 実効 leaf map を突合。初回ゴールは「既存 config 無変更で完全一致」 | §8 |
| D18 | resume パスは素材化しない | checkpoint path は Run instance 固有値=run-local leaf override。再利用素材(@)に入れない | §7.3 |
| D19 | net 読み口は `<agent>.net` | 各 Agent が自分の config prefix 配下の net を読む(旧案 1 vs 案 2 の決着=案 2)。カタログ `net.block` はグローバル共有のまま。rename 実測: 直書き 107 行+bat 5 行(機械置換可)、素材・カタログ・テンプレ定義は **D19 rename の対象外**(素材の `@` 化は Phase 1 の別軸=§3.5、agent 配下への移設のみ任意)、optuna 影響ゼロ。ADR 0030 | §3.5 |
| D20 改 | 上書き層は無印のまま+**直交 2 軸の番号付き多段層**(grill 2026-08-25) | `@` は再利用素材専用(不変)。層の名前は**文字=対象(A=Agent / E=Env / M=Metrics / P=app)× 番号=恒久度**(大番号ほど揮発的・チェーン右=強い)の全層番号付きへ統一。**旧 R→A2、旧 X→A3、旧 O→M2 へ吸収**(用途文字の廃止)。段数は現行写像で必要時追加。昇格ライフサイクル(揮発番号→小番号→`@` 素材/幹)と接続 | §4.1 |
| D25 | mode(`app.online / app.batchrun`)は「選択の源」として無印を正式仕様化 | 素材(選択肢の在庫)ではなく Run 動作モードの定義。既知 prefix として dump 残留は仕様(②と同格)。`@` 化しない(bat / CLI / optuna の `app.$=app.batchrun` 波及を負わない) | §6.1 |
| D26 | backend / metrics 素材の `@` 化(Phase 1.5 は作らない) | `backend.@deterministic / @non-deterministic`、`metrics.scalar.@baseline / @min / @full / @iqn_search_p0`、`image.*.@<name>` 系。これで dump の ③ 無印素材は完全消滅し、指標 3 が名実ともに達成される | §6.1 |
| D21 | Key 部空白は除去のみ | 検証なし。誤記は未知キー化するが現行のタイポ NoCare と同水準で悪化なし | §4.4 |
| D22 | RainbowAgent(C++)は保持 | Agent 柔軟性(複数 Agent 実装の受け入れ)実証としての意図した保持。`<agent>.net` 規則の適用対象(現用 env 構成に使用なし=rename 実務への影響なし) | §3.5 |
| D23 | 用語 5 件を CONTEXT.md へ追加 | 素材 / カタログ / 幹 / 上書き層 / デフォルト直書き。_Avoid_ 欄に「プロファイル/profile」は使わない(`net.config_profile` の正式用語と予約衝突するため「プリセット」を充てる) | CONTEXT.md |
| D24 | ドメイン検証は Factory 境界(目標 2 の実装) | `quantile_mode` ↔ net 契約を `DefaultDQNAgentFactory::CreateAgent` 内の named validator(両 Config 構築直後・NetworkModel 構築前)で検証し原因キー名指しで fail-fast。D19 により「自分の net サブツリーとの整合」が局所契約になったため ADR 0018 の責任境界と両立。Phase 1 で実装 | §5.5 |

---

## 2. 概念モデル: 3 層と幹

### 2.1 3 層

| 層 | 役割 | 例 | dump |
|---|---|---|---|
| **カタログ** | 参照される部品。「選ばれる」ものではなく、コードや structure 文字列が名前で引く | `net.block.[*]`、`train.eval.[tag]` 定義、metrics 定義 | 残る(読み口がある) |
| **素材** | 合成の入力。named 選択肢(プリセット)と nameless 上書き層の総称 | `AtariEnv.@v5`、`DefaultDQNAgent.@iqn`、`@vars`、上書き層 A1/A2/A3/E1/M1/M2/P1(D20 改) | `@` 宣言分は除外 |
| **幹** | Run 署名。スロット選択の束。実際に動く最終設定へ合成される | 選択行の束、named 幹 `run.@<name>` | 実効値として残る |

カタログと素材の判定基準: **Resolver(`.$`)や `${}` の素材としてのみ意味を持つキー=素材。コードの読み口(Config / Builder)が直接読むキー=カタログまたは実効値。**

昇格ライフサイクル: nameless 上書き(X)→ 定着したら named 素材(@)→ 素材の組が定着したら named 幹(`run.@`)。

### 2.2 幹の規約

- 頂点は `run.`。**workspace は設定の外**(Run フォルダの移動・削除で管理。ADR 0021 の分類責務はフォルダ側に残る)。
- 幹は**選択のみ**を書く(スロット選択行と、宣言済み値スロットのみ)。値スロットは `E.game` 相当・`train.num_envs` のような「選択肢に構造がなく、値自体が署名の一部」のもの。
- `train.seed` は幹に含めない(反復軸)。
- 幹スロット(概念図①): mode(online/batchrun)/ env / agent / RUN_BUDGET / LEARNING_SCHEDULE / precision / backend / eval_schedule / metrics 選択 / num_envs。
- env 枠・agent 枠は「class_id(実効 leaf)+具象 namespace のチェーン」の 2 行構造とし、畳む機構は作らない。「class_id の値=具象 config prefix」を正式契約に格上げする(現状は暗黙の規約)。
- **幹のスタイルは 2 種を許容し用途で使い分ける**(grill 2026-08-25): **差分幹**=既定との差分だけを書く(短い・増殖圧力が低い・既定の変更に追随する。日常の切替・デモ向き。例: Atari の `run.@v5_iqn_impala_x2`)/**フル署名幹**=Run 署名を完全固定する(対照実験・文献再現向き。既定が変わっても幹の意味が不変。例: DropMerge の `run.@iqn32_stratified`)。
- **幹は上書き層キー(`A2.foo` / `A3.foo` 等)を供給してよい**(grill 2026-08-25)。機構的理由: 幹が供給する素の実効 leaf は「デフォルト直書き」順位に入り、チェーン解決の産物(A 系層の複製)に負ける。上書き層より強い値を幹から固定するには、チェーン優先順位を持つ層キーを書くのが唯一の形。どの番号を書くかは幹の用途で選ぶ(対照幹=大番号で手元試行にも負けない強固定/日常幹=小番号で手元の余地を残す)。幹専用番号は予約しない。

### 2.3 named 幹

```text
run.@atari100k_verify : env.class_id = AtariEnv
run.@atari100k_verify : AtariEnv.$ = @100k > @E
run.@atari100k_verify : AtariEnv.game = breakout
run.@atari100k_verify : backend.$ = @deterministic
run.$ = run.@atari100k_verify        # 1 行切替
```

- `run.$` の選択結果は root へ展開される(§5.2 の幹前段。「ファイル末尾に幹の中身を追記したのと同一」の意味論)。
- named 幹は「検収構成」「再現構成」など**使う構成にだけ**作る。素材直積の事前定義は禁止(D4)。
- 現行 config に実在する「3 箇所を手で同時に編集する契約」(NatureDQN 再現のコメント運用)が、named 幹 1 行に置き換わるのが代表ユースケース。
- `run.` prefix で特別なのは `run.$`(幹選択)と `run.@*`(幹素材)だけ。`run.` 配下の通常キー(`run.foo = x` 等)は特別扱いせず普通の実効 leaf として振る舞う(現行 config に `run.` prefix の使用は無く、予約化に衝突しない)。
- **幹のネストは fail-fast**: 幹素材の中に `run.$` を書くこと(幹から別の幹を選ぶこと)は禁止。幹は深さ 1 の規律とし、D4(直積の事前定義禁止)と整合させる。
- CLI から `run.$=run.@<name>` を与える幹切替は第 1 相(§5.3)で受け付ける(bat からの幹指定)。

---

## 3. 軸カタログと所有権

### 3.1 軸選定の 3 条件基準(D10)

プリセット化の価値は「値が複数ある」ことではなく、**1 つの選択が複数箇所を同時に変えるか**で判定する。

1. 値が 2 つ以上ある
2. 1 つの選択が 2 箇所以上を同時に変える(=手で揃える契約がある)
3. 揃え忘れると壊れる、または静かに間違う

### 3.2 軸表

| 軸 | 置き場所 | 中身 | 優先度 |
|---|---|---|---|
| RUN_BUDGET | `app.@budget50m.*` 等(**app 配下素材**。D11 分離の帰結で root 横断性は消滅=S1) | `app.online.exp_pause_step` / `app.batchrun.exp_exit_step`(optuna score window はハーネス側の値) | **最高**(bat の `BUDGET=` 二重指定事故が実痛) |
| LEARNING_SCHEDULE | agent 配下素材(env ファイル所有) | `learner.per_beta_step` / eps decay / uqe tau decay | **最高**(RUN_BUDGET とセットで) |
| ALGO | `DefaultDQNAgent.@dqn / @qr / @iqn`(分布表現の 3 値) | `quantile_mode`+tau 系+**NN 配線**(§3.4)。「Rainbow」素材は作らない(D12) | **最高** |
| HEAD | `DefaultDQNAgent.@dueling` 等 | `use_dueling_net`+V/A ストリーム宣言 | **遅延ゲート**(最初に事故したときに作る。§10) |
| exploration | `DefaultDQNAgent.@uqe` 等 | `train_policy.policy_type`+uqe 系+eps 系 | **遅延ゲート**(同上) |
| TARGET | `DefaultDQNAgent.@munchausen` 等 | target 計算の変種。NN 配線を持たず ALGO と直交(M-IQN = `@iqn`+`@munchausen`) | **遅延ゲート**(M-IQN 着手時) |
| env プロトコル | `AtariEnv.@v5/@classic/@100k` | 既存プリセットの `@` 化 | 済(宣言のみ) |
| BOARD(DropMerge) | `DropMergeEnv.@G5846` 等+`${}` | 盤面選択(素材)+width/height 数値の一点化(値参照)。**素材と値参照の役割分担のモデルケース** | 中 |
| ENV_PROTOCOL(DropMerge) | `DropMergeEnv.@…` | action mode / prev-action obs / DROP marker / NoLegal 裁定 / timeout 契約 | 中 |
| REWARD_CONTRACT(DropMerge) | env×agent 跨ぎ(所有で切れない数少ない軸。設計は着手時) | fruit score・penalty と Agent reward scale | 中 |
| IQN_RESOLUTION | ALGO 素材の既定値+trial layer 上書き | K/N/M と tau sampling | 中 |
| **BODY** | **プリセット化しない** | `net.branch.[main_feature].$` の現行 1 行が既に理想形(切替 1 箇所・揃える相手なし) | — |

分離の根拠(D11): DropMerge の実測では 100M Run でも PER beta 20M・UQE eps 5M・tau 20M であり、**Run 予算と学習 schedule は連動していない**。Atari の「per_beta_step=予算/2」は Atari ローカル慣行として Atari の LEARNING_SCHEDULE 素材側に持ち、共通仕様にしない。`update_warmup_steps` は BatchSize・Replay 設計に従属するため両軸から除外する。

### 3.3 ALGO と BODY の分離(D12)

ALGO が触る NN 側は 2 種類あり、扱いを分ける:

- **(a) 配線**(tau branch、fusion、stream の bind、features の出所)= **ALGO が所有**。IQN を選んだら fusion 配線は必然で選択肢がない。
- **(b) 特徴抽出器本体**(Nature / Impala / ViT)= **独立軸(BODY)のまま**。セットにすると algo×body の直積素材(3×5=15)に膨れる。

BODY↔ALGO を繋ぐ契約は「main_feature 最終次元」1 本のみ。暫定は BODY セクション見出しへの規約明記(「どの BODY も同一次元で終わること」)、恒久は `${}` による一点化(TauProj の out_features が BODY 最終 Linear の実キーを直接参照するか、双方が `@vars.feature_dim` を参照する。§4.3)。

### 3.4 所有権表

軸が完全直交しない交差キーは、両方に書かせず**所有者を 1 つに決める**:

| キー | 所有者 |
|---|---|
| `net.body.output.[features]`(features の実体: qr→main_feature / iqn→iqn_fusion) | ALGO |
| tau 系 branch(`[tau_embedding]` / `[iqn_fusion]`)の bind・structure トポロジ | ALGO(ブロック実体・寸法は env 供給) |
| `net.branch.[main_feature].$` | BODY(=利用者の直接選択。素材化しない) |
| main_feature 最終次元 | 契約(`${}` 一点化、§3.3) |

**遅延ゲート先の設計メモ(HEAD 軸着手時に検証)**: V/A ストリームの存在宣言は HEAD 所有とし、stream の `bind` は `features` 固定インターフェース(stream は常に `features` へ bind、ALGO は features の実体差し替えのみを所有)にすると ALGO×HEAD の交差キーが消える。IQN の rank-3 が features 経由で V/A stream へ流れる shape 整合は HEAD 軸を作るときに裏取りする。それまで現行配線(dueling 前提の qr/iqn プロファイル)の**内容**は変更しない(Phase 1 の `@` 化でキー名が `net.@qr/@iqn` になるのは別軸)。

### 3.5 net 読み口の所有(D19 / D22)

**各 Agent は自分の config prefix 配下の net を読む**(`DefaultDQNAgent.net.*` / `ImageClsAgent.net.*`)。旧 v1 の案 1(root overlay)vs 案 2(所有の移動)は、ALGO 軸において案 2 で決着した(ADR 0030)。

- カタログ `net.block.[*]` は全 Agent 共有のグローバル残留。素材(`net.qr/iqn` 等)とテンプレ定義は **D19 rename(112 行)の対象外** — AutoMerge は RHS 素材を LHS prefix へ複製するため、素材はどこに置かれていてもよい。ただし独立の 2 軸を区別する: **①素材の `@` 化**(キーを `net.@qr/@iqn` へ。**配置は root のまま**)は Phase 1 の完了条件(§8.2、指標 3 のゲート)、**②agent 配下への移設**(`DefaultDQNAgent.net.@iqn` へ。相対参照 `@iqn` で書けるようになる)は任意。Phase 1 後の参照は `DefaultDQNAgent.@iqn : net.$ = net.@iqn`(絶対)が標準形。
- rename 対象は「最終ツリーへの直書き行」のみ。実測: `net.$` 7 行+`net.branch.[slot]` 63 行+`net.body.output/$` 37 行+bat の CLI override 5 行=**112 行**(コメントラダー込み・10 ファイル・機械置換可)。optuna 生成 config は net を触っておらず影響ゼロ。
- コード変更: `NetworkConfig` の構築 prefix(既定 "net" → Agent config prefix 連結)と関連テスト。
- MuZeroAgent の**実際の最終ツリーは root の `net.rep` / `net.dyn` / `net.pred`** にある(agent.txt の `model.structure.*` は構造文字列の供給元。v3 時点の「agent 配下に構造を持つ流儀」という認識は PH1a 実装確認で不正確と判明)。MuZero は保留中で ALGO 素材の対象外のため **PH1a では rename しない** — 再着手時に `<agent>.net` へ寄せる(§10 遅延ゲート)。それまで root `net.*` に MuZero 最終ツリーが残るが、該当直書きは GridMaze_muzero 系設定に閉じており DQN 系 Run の dump へ混入しない。
- RainbowAgent(C++)は Agent 柔軟性実証として**保持**し(D22)、本規則の適用対象とする(現用 env 構成に使用がないため rename 実務への影響なし)。ALGO 素材化の対象は DefaultDQNAgent。

---

## 4. 構文契約

### 4.1 `@` セグメント=素材宣言(D5)

`@` で始まるセグメントを含むキーは**素材**である。宣言には特典と義務がセットで付く:

| | `@` あり(素材宣言) | `@` なし(従来の自由 prefix) |
|---|---|---|
| dump | 除外(`config_data.txt` に出ない) | 残る |
| 未定義参照 | fail-fast(参照先が存在しない) | 従来どおり黙って no-op(空層文化を維持) |
| 規約 | コードの読み口に使わない。再利用前提 | 自由 |

fail-fast の対象は**未定義素材への参照**(チェーン項・`${}` の参照先キーが 1 つも存在しない)である。逆方向の**未参照の素材定義は正常**であり、エラーにも WARN にもしない — 素材は選択肢のカタログであって、いつか使うための定義だけを置いておけることが要件である(dump 除外により、未選択の定義が dump を汚す現状問題もなくなる)。

- **Resolver は `@` を要求しない**(機構非強制)。`> TEST` のような無印 prefix 項は従来どおり動き、dump に残る。「dump に痕跡を残したい一時上書きは無印、消したい再利用素材は `@`」は書き手の意図表現である。
- 置き場所 = 所有者 prefix 配下(`AtariEnv.@v5`、`DefaultDQNAgent.@iqn`、`app.@budget50m`)。所有者が root にしかない横断素材(現時点で該当なし。precision が将来候補)のみ root 直下。「どの幹向けの素材か」は置き場所が語り、「素材である」ことは `@` が語る。
- **上書き層は無印のまま維持する**(D20 改)。`@` は再利用素材専用であり、nameless なファイルローカル差分を `@` 化しない。これにより空許容の宣言機構は不要となる(未定義の無印 prefix は従来どおり no-op)。
- **上書き層の命名は直交 2 軸**(D20 改、grill 2026-08-25): **文字=対象**(A=Agent / E=Env / M=Metrics / P=app)、**番号=恒久度**(大番号ほど揮発的。チェーンでは右に置く=後勝ちで強い)。全層番号付き(`A1 / A2 / A3`、`E1`、`M1 / M2`、`P1`)で無印単文字は使わない。旧 R(実験)→A2、旧 X(A/B 軸)→A3、旧 O(metrics 2 段目)→M2 へ吸収し、用途文字の暗記を廃した。段数は必要になった env が右に足す(空層は no-op)。値の昇格は「大番号(その場の試行)→小番号(定着中)→`@` 素材/幹(恒久)」の一方向で、§2.1 の昇格ライフサイクルの前段に接続する。

### 4.2 チェーン項の解決規則(D6)

```text
TEST.repeat_action_probability = 0.25
AtariEnv .$ = @baseline > AtariEnv.@v5 > TEST
#              │           │              └ 絶対: root の TEST.*(従来動作)
#              │           └ 絶対: AtariEnv.@v5(下の @v5 と同じ場所。冗長表記だが合法)
#              └ 相対: AtariEnv.@baseline
```

- **`@` 始まりの単独項** = LHS の所有者 prefix 配下の相対参照。解決 0 件は fail-fast(fallback しない)。
- **それ以外の項**(`.` を含む、または無印)= 従来どおり root からの絶対 prefix。
- 既存 config のチェーンは全項「それ以外」に該当するため、**後方互換は完全**。

### 4.3 値参照 `${フルキー}`(D7)

```text
@vars.max_exp_step = 50,000,000
app.online.exp_pause_step = ${@vars.max_exp_step}
app.batchrun.exp_exit_step = ${@vars.max_exp_step}

# 実キー同士の直接同期も書ける(どちらが源かが参照方向で明示される)
net.block.[TauProj].linear.out_features = ${net.block.[AtariLinear512].linear.out_features}
```

- `${K}` は**任意キー K の解決値**による置換。K はドット正規形のフルキー(絶対参照のみ。チェーン項の相対 `@name` とは別物で、相対形式は持たない)。
- 予約 namespace は設けない。定数の置き場は root 素材 `@vars.*` を**命名慣習**として推奨する(機構的特殊性はゼロ)。所有者配下の素材(`DropMergeEnv.@dims.width` 等)に置いてもよい。`@` 付きなら dump から消え、無印キーへの参照なら参照先は実効値として dump に残る — §4.1 の宣言原則がそのまま適用される。
- 用途は**同値の一点化(同期)のみ**。導出(式評価。例: 予算/2)は対象外 — 導出値は素材内に具体値で書き、導出根拠はコメントに書く(設定言語を計算言語化しない)。
- 1 段制限: 参照先の値がさらに `${}` を含む場合は fail-fast(連鎖・循環を最初から排除)。
- 展開は leaf override 適用の**後**(§5.2 フェーズ④)。CLI・optuna が参照先キーを上書きした場合も参照元へ波及する。未定義キーへの参照、実効値に残る未解決 `${}` は fail-fast。
- 参照の解決記録(参照元→参照先→解決値)は resolution.json に残す(§6.2)。
- 既存の `{t}`(run_name)とは構文衝突しない。
- 位置づけ(G8): §0.1 のゴールに直結しない補助機構(付録 B の値同期型 #3/#4/#10/#12 に効く)だが、明示要望・実装極小のため Phase 0 に含める。

### 4.4 `:` 糖衣と Key 部の空白(D8)

```text
AtariEnv.@v5    : repeat_action_probability = 0.25
AtariEnv.@100k  : noop_max = 30
DefaultDQNAgent.@iqn      : quantile_mode = iqn
DefaultDQNAgent.@iqn      : net.$ = net.@iqn
DefaultDQNAgent.@dueling  : use_dueling_net = true
```

- 行の Key 部(最初の `=` より左)にある `:` は、Key(どの素材か)と SubKey(その中のどこか)の**視覚上の区切り**。パース時に空白除去のうえ `.` へ置換してフラットキーに落とす。**`.` と完全等価・意味ゼロの糖衣**。
- 正規形はドット連結。resolution.json・エラーメッセージ・grep 対象は常にドット形。
- Key 部の `:` は最大 1 個。超過・空片は fail-fast(書式エラー)。
- Key 部の空白は**除去のみで検証しない**(D21。`AtariEnv .$` = `AtariEnv.$`)。空白絡みの誤記は未知キーになるが、現行のタイポキー NoCare と同水準で悪化はない(`=` 忘れ行が黙って無視されるのも現行どおり)。値側の `:`(metrics DSL の `ema_alpha:0.001`)には影響しない。
- 位置づけ(G8): `:` 糖衣は §0.1 のゴールに直結しない表記改善だが、明示要望・実装極小のため Phase 0 に含める。
- 機構としては任意の行で使える(例: `train.eval.[eval1] : run_mode = eval1`)。素材定義での使用を推奨、は規約。
- 上例の `net.$ = net.@iqn` が**絶対参照**なのは意図的: 素材は `@` 化されても(Phase 1 完了条件)**配置は root のまま**であり、相対 `@iqn`(= `DefaultDQNAgent.net.@iqn` を指す)では解決 0 件で fail-fast する。素材を agent 配下へ**移設**(任意=§3.5)した後にのみ相対 `@iqn` へ書き換えられる。
- 1 行完結(位置独立)は維持する。セクションヘッダ構文は不採用(行が上方ヘッダに依存し、コメントアウトラダーでの行トグル・並べ替えの自由を壊すため)。

### 4.5 競合規則(D9)

同一の実効キーへの複数書き込みは、すべて**種別の全順序**で決定的に解決する。fail-fast も WARN も発しない。

1. **デフォルト直書き**(素の設定行)— 最弱。役割は「選択が無い場合に採用される値の宣言」= 実装側デフォルトの設定ファイルへの可視化であり、チェーン結果に上書きされることを前提とした共存が設計上の正常形
2. **解決の産物**(チェーン展開。チェーン内は右勝ち)
3. **CLI 第 2 相**(実効 leaf override)— 最強

同種内はファイル読み込み順の後勝ち(現行どおり)。

**named 幹の順位(Phase 2 で確定)**: 幹の展開は新カテゴリを作らない。幹が供給する行(leaf 行・チェーン行とも)は、selection 解決の**前段**(§5.2)で working map を後書き上書きする — 意味論は「**幹の中身をファイル末尾に追記したのと同一**」。したがって幹供給の leaf は上記 1(デフォルト直書き)として、幹供給のチェーンは上記 2(解決の産物)として、既存の全順序にそのまま乗る。root 直書きのチェーン行と幹供給のチェーン行が同一キーで衝突した場合は幹が勝つ(末尾追記=後勝ち)。CLI 第 2 相は幹の産物にも勝つ(順位 3 のまま)。

「どの writer の値が採用されたか」の診断は resolution.json の writers(詳細モード)が担う。

v2 で検討した「独立 writer の fail-fast」は廃案(G5): 全順序を定義しきれば「順序未定義の衝突」は原理的に存在せず、デフォルト直書きとチェーンの共存は両方とも意図して書かれる正常形なので、警告もノイズになる。

---

## 5. ConfigResolver アーキテクチャ

### 5.1 モジュール構成(D13)

`ConfigManager` の既存 interface(構築 → `GetConfigData()`)は不変のまま、内部を deep module `ConfigResolver` として再構成する。現行 `AutoMerge()`(約 60 行、付録 A)の拡張・肥大化はしない。

Resolver は **{実効 `ConfigData`, resolution 記録} の対**を返し、`ConfigManager` が両方を保持する。resolution 記録の取得は新設 **`ConfigManager::GetResolutionJson()`**(`anet::json`を値で返す)で行う。公開record型は作らず、既存JSON出力interfaceへ構造化データのまま渡す(§6.2 に受け渡し契約とJSON具体形)。

### 5.2 解決フェーズ

1. **source layer と provenance の収集** — main / `$include` / workspace overlay / injected / CLI の各層を、出所(base / workspace / extra / generated / CLI)と共に読み込む。`:` 糖衣と Key 部空白はこの段階で正規形へ落とす。
2. **selection 参照の解決(DFS)** — `.$` チェーンを参照グラフとして DFS で解決する。相対項は LHS 配下で解決。**不動点反復ではなく依存グラフ**を採る理由: 循環経路・未知素材・深さ超過を、経路情報付きで正確に報告できるため。
3. **leaf override の適用** — CLI 第 2 相(§5.3)。
4. **値参照の展開** — `${フルキー}` を参照先キーの解決値で置換。leaf override の後に置くことで、CLI・optuna が参照先キーを上書きした場合も参照元へ正しく波及する。
5. **effective config と resolution 記録の返却** — 実効マップ(素材除外済み)と resolution 記録(§6.2)を出力。

**幹前段(Phase 2 で追加)**: named 幹の root 展開は、フェーズ 2 の**直後ではなく前段**(CLI 第 1 相適用後・selection スナップショット取得前)に行う。動作: working map に `run.$` があれば(無ければ no-op=既存 Run 完全互換)チェーンを解決し、各項(絶対 `run.@<name>`、相対 `@<name>` は owner=`run` で解決)配下のキーを **prefix 剥がしで root へ**後書き Set する。前段自身は effective map に触れず nested 再帰もしない — 幹が root へ落としたチェーン行は、直後のスナップショットに入り通常のフェーズ 2 で解決される。処理済み `run.$` はフェーズ 2 の対象から除外する。幹素材内の `run.$`(幹のネスト)は fail-fast(§2.3)。

前段に置く理由: フェーズ 2 のトップレベル selection は解決開始前にキー集合をスナップショットし、値を解決時に遅延取得する実装のため、「解決の直後に持ち上げ」だと ①持ち上げで root に落ちたチェーン行の再解決にもう 1 パス必要 ②root 直書きチェーンが先に解決され、幹側チェーンが書かないキーの産物が残留 ③同一チェーンの二重解決と selections の重複記録 — の 3 問題が生じる。前段なら root 直書きチェーンは幹値で上書きされた状態でスナップショットされ 1 回だけ解決される(「末尾追記と同一」が厳密に成立)。root 持ち上げは S1 により Phase 0 には含めない(現時点の素材はすべて所有者配下で root 持ち上げを必要としない)。

### 5.3 CLI override の 2 相化(D14 改)

原則: **CLI の指定はすべての設定ファイルより優先される**。CLI override は**全キーを両相に適用**する:

- **第 1 相(フェーズ 2 の前)**: **全 CLI override** を source map へ先出しする。selection(`.$`)・素材キー(`@` を含むキー)に加え、選択の源プレフィクス形(`app.batchrun.exp_exit_step` 等)もここで選択の複製に乗って効く。bat の `app.$=app.batchrun` はここ。
- **第 2 相(フェーズ 3)**: selection・素材キーを**除く**キーへの実効 leaf 上書き。選択・幹の産物すべてに最後に勝つ — 「CLI 最強」の保証はこの相が担う(第 1 相で source map に入れた実効 leaf キーが選択の複製に上書きされても、第 2 相で CLI 値が最終適用される)。
- CLI で与えた `.$` が解決後の dump に残る旧挙動の廃止(選択の記録は resolution.json が担う)は**維持**する — 第 2 相の篩は不変。
- 注記: 源プレフィクス形は「選択を経由して効く」間接キーであり、選択自体が別の源へ切り替わると効かなくなる(優先度でなくキーの意味論)。絶対優先が必要な値は実効 leaf 形か `@vars` 素材キー経由で渡す。

改訂経緯: v3 の「種別で分ける(第 1 相=selection と素材キーのみ)」では、源プレフィクス+leaf 形の CLI(bat の `app.batchrun.exp_exit_step=...`)が第 2 相へ回って選択の複製に乗らず、誰も読まないキーとして**静かに無効化**される取りこぼしが Atari 素材化(25impl 申し送り 7)で発覚した。旧 AutoMerge の「前後 2 回適用」は前段適用が源キーへ書かれ merge で効いていた — 両相適用はその安全な復元であり、dump 純化(D14 の当初目的)は第 2 相の篩で維持される。

### 5.4 fail-fast 一覧

| 条件 | 挙動 |
|---|---|
| **未定義素材・未定義キーへの参照**(チェーン項の `@name` / `${フルキー}`) | fail-fast(参照元・参照名・探索スコープを含める)。※逆方向の「未参照の素材定義」は正常(選択肢の在庫。エラー・WARN にしない) |
| selection グラフの循環 | fail-fast(循環経路を列挙) |
| 深さ上限超過 | fail-fast(上限値と経路) |
| 実効値に未解決 `${}` が残存 | fail-fast |
| Key 部の `:` 複数・空片 | fail-fast |
| 無印 prefix 項の解決 0 件 | 従来どおり no-op(空層文化。D20 により恒久確定 — 上書き層は無印のままなので宣言機構は不要) |

型変換・値域・enum・組み合わせの検証は現行どおり各 Config / 再利用設定型の責務(AGENTS.md「汎用機構と利用側の責任境界」)。本 PRD が足すのは、解決レイヤの構造検証(本節)と、目標 2 を実装するドメイン検証(§5.5)の 2 つである。

### 5.5 ドメイン検証(目標 2 の実装)

素材で束ねても防げない経路(CLI 第 2 相・上書き層による部分 override)の契約違反を、**Factory の構成検証境界(named validator)**で設定境界のうちに捕まえる。D19 により net が Agent config prefix 配下へ来たため、「Agent が自分の net サブツリーとの整合を検証する」ことは**局所契約**であり、ADR 0018 の責任境界(DAG の推移的到達性検証はしない)と衝突しない — v1 §4.3(案 3)で未決だった線引きは D19 が解消した。

Phase 1(D19 rename 後)に実装する最小セット:

| 検証 | 所有 | 時点 | 内容 |
|---|---|---|---|
| `quantile_mode` ↔ net 契約 | **`DefaultDQNAgentFactory::CreateAgent` 内の named validator**(`DefaultDQNAgentConfig` と `NetworkConfig` の両構築直後。両オブジェクトが揃う唯一の自然な境界で、`NetworkConfig` の解析済み bind 情報を照会する — bind parser の重複も `NetworkConfig` の二重構築も生じない) | 設定境界(NetworkModel 構築前) | `quantile_mode=iqn` なのに `<agent>.net` に `taus` を bind する branch が無い、または `quantile_mode=qr/none` なのに taus bind がある → **`quantile_mode` と該当 net キーを名指しして fail-fast** |

受け入れテスト: 付録 A の不整合 3 パターン(iqn×qr 配線 / qr×iqn 配線 / `net.$` 書き忘れ)が、NN 構築時の shape / bind エラーではなく、原因キーを含む設定境界エラーになること。到達性・意味的寄与の保証はしない(ADR 0018 の設定者責任は不変)。他 Agent・他概念への検証追加は、同型の事故が実際に起きたときに同じ枠で足す(先回りしない)。

---

## 6. Run artifact 契約

### 6.1 `config_data.txt` の純化(D15)

- 内容 = **実効値+無印上書き層の定義行**。`@` セグメントを含むキーは出力しない(定数置き場 `@vars.*` もこの規則でカバーされる)。無印の上書き層(A/E/R/X/M/O/P)は **D20 により意図的に残る** — 「この Run で何を上書きしたか」の痕跡であり、既知の固定 prefix なので読み手・ツールは機械的に識別できる(v2 の「実効値のみ」という表現は D20 と両立しないため本版で正確化した)。
- dump に載る行は最終的に 3 分類になる: **①実効値**(本体)、**②無印上書き層**(D20 改の恒久残置。`<対象文字><番号>` パターンで識別)、**③`@` 化前の無印素材**(過渡期のみ。AS / BF16 / `net.qr/iqn` / Agent baseline / env プリセット等)。③が残る間は module dump 突合の完全削除はできないため、**§0.3 指標 3 の達成は Phase 1 の素材 `@` 化完了にゲートされる**(§8.2)。①②のみになった時点で突合は不要になり、チェーンへの適用有無は resolution.json の `selections` で判別する。
- **mode(`app.online / app.batchrun`)は「選択の源」であり素材ではない**(D25)。dump への残留は②と同格の仕様(既知 prefix)。backend / metrics 素材の `@` 化(D26)完了により③は完全消滅し、指標 3 は名実ともに達成される。`--config` 完全自己記述モード(実効値の再読込)は従来どおり成立する — チェーン行は dump に含まれないため上書き層定義が再適用されることはなく、flatten 済み実効値がそのまま使われる。

### 6.2 `json/config_resolution.json`とMetrics master record(新設)

選択経路の構造化記録。最低限のスキーマ:

**必須(Phase 0 で実装)**:

| フィールド | 内容 |
|---|---|
| `schema_version` | 契約バージョン |
| `selections` | 選択された素材の完全修飾名(ドット正規形)と適用順 |
| `references` | `${}` 参照の解決記録(参照元キー → 参照先キー → 解決値) |

**詳細モード(分析ニーズが実在してから追加。S3)**:

| フィールド | 内容 |
|---|---|
| `sources` | 読み込んだ config ファイルの一覧と SHA-256、各層の出所(base / workspace / extra / generated / CLI) |
| `writers` | 各実効キーの最終 writer(素材/override の完全修飾名)。全 write trace は持たない。競合診断(§4.5)の担い手 |

**受け渡し契約**: App は `ConfigManager::GetResolutionJson()`(§5.1)で取得した構造化JSONを、`config_data.txt` と同じ初期化タイミングで既存の `MetricsLogger::Log("config_resolution", json)` へ渡す。MetricsLoggerは `json/config_resolution.json` に `type` / `tag` / `data` envelope付きで保存し、timestampを加えた同じrecordをMetrics masterへ記録する。書き出しの所有者は既存の構造化Run metadataと同じApp / MetricsLoggerであり、Config固有のartifact APIやAppの直接fs書き込みは追加しない。既存JSON経路の数値丸め、pretty-print、timestamp付与を共通契約として受け入れる。

**JSON 具体形**(`schema_version = 1`):

```json
{
  "type": "json",
  "tag": "config_resolution",
  "data": {
    "schema_version": 1,
    "selections": [
      {
        "key": "DefaultDQNAgent.$",
        "chain": [
          {"term": "@baseline", "resolved": "DefaultDQNAgent.@baseline"},
          {"term": "@iqn", "resolved": "DefaultDQNAgent.@iqn"}
        ]
      },
      {
        "key": "DefaultDQNAgent.net.$",
        "chain": [
          {"term": "net.@iqn", "resolved": "net.@iqn"}
        ]
      }
    ],
    "references": [
      {"source": "app.online.exp_pause_step", "target": "@vars.max_exp_step", "value": "50000000"}
    ]
  }
}
```

- resolution payloadは`data`配下に置く。`json/config_resolution.json`にはtimestampを含めず、Metrics master recordには既存JSON経路がtimestampを追加する。
- `selections` は**selection適用1回につき1 entry**。同一の実効`.$` keyがチェーン内の複数素材から生成された場合は、右勝ち履歴を表すため同じ`key`のentryが複数並ぶ。`key` はドット正規形の完全修飾。`chain` は左から右への適用順で、`term` = 記述どおりの項、`resolved` = 解決先のドット正規形絶対 prefix(相対 `@name` の解決結果がここで読める)。
- **素材の複製で生成された `.$` も独立 entry になる**。上例の 2 つ目は、`@iqn` 素材内の `net.$ : net.@iqn` が LHS(`DefaultDQNAgent.net`)へ複製されて生成されたもの — 外側 chain の項に参照先素材の中身を直接埋め込まない(埋め込むと展開先が `DefaultDQNAgent.branch.*` になり D19 の `.net.*` を外れる)。
- `references` は `${}` 1 参照ごとに 1 entry。`value` は置換後の文字列。
- 配列順は解決順(決定的)。詳細モードのフィールド追加時は `schema_version` を上げる。
- **named 幹(Phase 2)**: 幹前段(§5.2)の `run.$` 解決も通常の selection として 1 entry を積む(`key = "run.$"`、chain に幹素材名)。前段はフェーズ 2 より先に走るため、この entry は `selections` の**先頭**に来る — 「この Run はどの幹か」は先頭 entry を見れば判る。**schema_version は 1 のまま**(スキーマ変更なし)。幹越しに生成された下流 selection が「どの幹由来か」のトレース(declaration_key の JSON 化)は詳細モードと同じ扱いで遅延(§10)。

- コメント行方式(dump へのコメント埋め込み)は不採用。
- `--config` での再読込対象は従来どおり `config_data.txt` のみ。resolution.json は分析・診断専用で、読み戻しには使わない。

---

## 7. optuna / batch 運用契約

### 7.1 現行との整合

現行 optuna ハーネスは Runner へ `key=value` を渡さず、base → workspace → extra → trial 固有値の順で**自己完結 config を生成し `--config` 起動**する(`apps/runner/tools/dropmerge_optuna.py`)。本再設計はこの経路を変えない。include 順の責任をハーネスが持つ契約(ADR 0021)も維持される。

### 7.2 trial での選択の扱い(D16)

- trial が `.$` 選択を振る用途は限定する: **HEAD / BODY 候補=同一 objective で可。ALGO 切替=metric・loss・score の意味が変わるため原則別 study。RUN_BUDGET 切替=score window と計算予算が変わるため同一 study 内は禁止**。
- 「選択された素材名」を通常の trial parameter として保存し、実効 leaf 値は `config_data.txt` から読む。
- `@vars.*` を trial パラメータにすると、複数実効キーへ波及する探索パラメータを 1 個で定義できる(例: `@vars.max_exp_step`)。無印の実効キーを直接振っても、それを参照する `${}` へ波及する(§5.2 フェーズ④)。

### 7.3 resume(D18)

- resume 可能な**構成・方針**は素材にしてよい。**checkpoint path** は Run instance 固有値であり、run-local の leaf override(無印)で与える。素材(@)に具体パスを入れて恒久カタログ化しない。
- 実効パスは `config_data.txt`、元 checkpoint の由来情報は resolution/provenance 側が持つ。将来の Run 全体 save/load における step・scheduler・seed 状態との整合は別契約(本 PRD のスコープ外)。

---

## 8. 移行計画

### 8.1 golden comparison(D17)

- 代表 config(全 env ファイル×代表チェーン)について、旧 resolver と新 resolver の**実効 leaf map を生成・突合**し、意図した差分以外がゼロであることを移行の受け入れ基準とする。
- **初回ゴール: 既存 config を 1 文字も変えずに新 Resolver で完全一致**。機構が `@` 非強制(D5)・チェーン後方互換(D6)であるため、これが可能である。

### 8.2 段階宣言方式

rename は「一斉」ではなく「宣言を貼っていく」漸進作業になる:

1. Phase 0 完了時点: 既存 config 無変更で新 Resolver 稼働(dump は従来相当。純化は素材宣言が付いた分だけ進む)。
2. D19 の rename(net 直書き 112 行の `<agent>.net` 化+`NetworkConfig` prefix 変更)は Phase 1 冒頭に**コードと設定を同一変更で**実施し、golden comparison を再実行する。
3. 素材宣言(`@` 化)・`:` 化をファイル単位・章単位で実施(**上書き層 A/E/R/X/M/O/P は対象外**=D20)。実効キーは不変のため、**過去 Run との実効値 diff は断絶しない**(素材定義行が dump から消えるのはノイズ減)。**Phase 1 の完了条件 = 選択肢として使う全ての無印素材(AS / BF16 / `net.qr/iqn` / Agent baseline / env プリセット / backend / app モード等)の `@` 化**。これが §0.3 指標 3(module dump 突合の削除)の達成ゲートである。
4. 機械置換で済まない箇所(要個別対応): batch launcher の `.$` override、optuna 生成 config の include 後 override、metrics 素材と実効 tag、resume パス、BOARD と Embed の跨ぎ選択、CLI で選択キーを指定した場合の 2 相適用。

### 8.3 読み側の両対応

解析ツール(`inspect_run.py` 等)は旧「カタログ入り dump」と新「純化 dump+resolution.json」の両方を読めるようにする。過去 Run artifact は書き換えない(AGENTS.md「過去の Run artifact は当時の記録として保持」の読み側対応であり、クリーンブレーク方針の明示的例外として本行を根拠にする)。

---

## 9. 実装フェーズ分割

| Phase | 内容 | 担当 |
|---|---|---|
| **Phase 0**(機構) | `ConfigResolver`(5 フェーズ、DFS、相対参照、`${}` 値参照、`:` 糖衣、CLI 2 相、dump 純化、resolution.json 必須スキーマ)+golden comparison ハーネス+単体テスト。**root 持ち上げは含めない**(S1) | Codex |
| **Phase 1**(コード+設定) | 冒頭で D19 rename(net 直書き 112 行+`NetworkConfig` prefix、golden 再確認)+**ドメイン検証の実装**(§5.5: quantile_mode↔net 契約、受け入れテスト込み)。素材宣言の漸進(`@`/`:` 化。上書き層は対象外=D20)、軸素材の整備(**RUN_BUDGET / LEARNING_SCHEDULE / ALGO のみ**=S2)、BOARD の `${}` 化(※baseline からのアルゴ的キー抽出は grill 2026-08-25 で遅延ゲート化=§10) | Claude 設計+ユーザー/Codex |
| **Phase 2**(設定+小機構) | 幹前段の追加(§5.2。`run.$` の root 展開+幹ネスト fail-fast)、named 幹 `run.@` の導入と幹順位の確定(§4.5 で確定済み: 末尾追記と同一)、既存「再現コメント運用」の幹化、`inspect_run.py` の resolution 対応(新 subcommand `resolution`。mirror `json/config_resolution.json` 優先+過渡期の `config/` 直下・envelope なし形の両対応。`runs` への幹列追加は遅延=§10) | 同上 |

**Phase 2 実装(コード)の前倒し**: Phase 2 のコード部(幹前段+inspect_run)は、Phase 1 の env 素材化を各担当枠へ引き継ぐ**前**に実装する。理由: 素材化と幹化を 1 回の引き継ぎ依頼にまとめるため(PH1/PH2 で 2 回に分けると担当枠との調整コストが倍になる)。幹前段は素材の書き方に依存せず(既存複製機構の変種 1 点)、順序入替に技術的支障はない。

---

## 10. 遅延ゲート一覧

v2 の残裁定 8 項目は grill(2026-08-23)で全て決着した(§11 複雑性監査)。本節は「今は作らない」と裁定したものの**発動条件**を記録する。ゲートが開くまで設計・実装を先回りしない。

| 項目 | 発動条件 | 参照 |
|---|---|---|
| HEAD 軸(`@dueling/@plain`) | dueling 切替で最初に事故したとき | §3.2 / §3.4 設計メモ |
| exploration 軸(`@uqe` 等) | 探索構成の切替で最初に事故したとき | §3.2 |
| TARGET 軸(`@munchausen`) | M-IQN 等へ着手するとき | §3.2 / 999_MunchausenRL_10prd.md |
| features 固定インターフェースの shape 裏取り | HEAD 軸の着手時(連動) | §3.4 設計メモ |
| precision 軸(BF16) | ImageCls 系 bf16 語彙(033)との統一検討時。root 横断素材の初実例となり、root 持ち上げの一般化(run 以外)を要する | 付録 B #6 |
| MuZero 最終ツリー(`net.rep/dyn/pred`)の `<agent>.net` 移設 | MuZero への再着手時(D19 の一般規則へ寄せる。保留中は root 残置で実害なし) | §3.5 |
| resolution.json 詳細モード(sources / writers) | 分析で「値がどこ由来か」の確認が実際に必要になったとき | §6.2 |
| 幹由来トレース(selections への declaration_key 追加) | 幹越しに生成された下流 selection の由来分析が実際に必要になったとき。`schema_version` を上げる | §6.2 |
| inspect_run `runs` への幹列追加 | 多 Run 横断の幹一覧が実際に必要になったとき(単 Run は `resolution` subcommand で足りる) | §9 Phase 2 |
| baseline からのアルゴ的キー抽出 | baseline の quantile 系キー起因の混乱・事故が 1 件目に起きたとき、または HEAD / exploration 軸の導入時に同時実施(前準備: GridMaze / CartPole チェーンへの `@qr` 明示挿入) | 28impl 提案 2 |
| `AtariIQNTauProj512` の改名 | `AtariLinear512` の次元を実際に変えるとき(値参照で追随はするが名前の `512` が実態とずれる) | 25impl 申し送り 5 |
| `@nature` の糖衣/直書きスタイル統一 | Atari 枠が次に `@nature` を触るとき、ついでに統一 | 28impl 提案 5 |
| `MuZeroAgent.baseline` の `@` 化 | MuZero 再着手時に net 移設(§3.5)と同時実施 | 28impl 提案 6 |
| REWARD_CONTRACT 軸の設計 | DropMerge の報酬契約を次に変更するとき | §3.2 |

---

## 11. 複雑性監査(grill 裁定記録)

### 11.1 grill 2026-08-23(v3 最終化)

grill スキルの final simplification pass(done/058_grill_simplification_pass_10prd.md)に従い、scope screen と簡素化パスの全裁定を keep / shrink / defer-behind-gate / cut の 4 値で記録する。**「なぜ作らなかったか」の将来参照用**。

| 対象 | 裁定 | 根拠 |
|---|---|---|
| root 持ち上げ機構(Phase 0) | **defer-behind-gate**(Phase 2) | S1: RUN_BUDGET の app 配下化で Phase 0 での必要性が消滅 — D11 分離の帰結に設計が追随していなかった「決定の残骸」の解消 |
| HEAD / exploration 軸 | **defer-behind-gate** | S2: 揃え忘れで実際に事故した実績なし(3 条件基準の②③を実痛で満たさない) |
| resolution.json の sources / writers | **shrink**(詳細モード) | S3: 実痛に pin できるのは selections / references のみ |
| 成功指標 | **keep**(§0.3 新設) | S4: 測定可能性の欠落を是正 |
| net 読み口 `<agent>.net`(D19) | **keep** | G1: 実測 112 行=機械置換可・素材/カタログ無変更・optuna 影響ゼロで裁定条件クリア |
| RainbowAgent(C++) | **keep**(D22) | G2: Agent 柔軟性実証としての意図した保持 |
| 「Rainbow」素材 | **cut**(D12) | G3: 軸の値でない俗称。作らなければ命名の誠実性問題(C51 でない・NoisyNet なし)が消滅し素材も減る |
| 上書き層の `@` 化+空許容宣言機構 | **cut**(D20) | G4: 無印維持で宣言機構ごと不要。上書きの痕跡が dump に残るのは意図表現として正しい |
| 独立 writer 検出(fail-fast → WARN → 廃止) | **cut**(D9) | G5: 全順序を定義すれば「順序未定義の衝突」は原理的に不存在。デフォルト直書きとチェーンの共存は両方意図して書かれる正常形で、WARN もノイズ |
| Key 部空白の誤記検証 | **cut**(D21) | G6: 現行のタイポ NoCare・`=` 忘れ黙殺と同水準で悪化なし |
| `${}` 値参照・`:` 糖衣 | **keep**(位置づけ明記) | G8: §0.1 のゴール直結ではないが明示要望・実装極小として Phase 0 に含める |

### 11.2 grill 2026-08-25(横断整合チェック後の裁定待ち 7 件+追加構想)

| 機構 | 裁定 | 理由 |
|---|---|---|
| 幹スタイル(差分幹 vs フル署名幹) | **keep(両方許容)** | 用途が異なる(日常切替 vs 対照・再現)。片方への強制統一はもう片方の用途を損なう。§2.2 に明文化 |
| 幹の上書き層キー供給 | **keep(正式許容)** | 機構的必然(素の leaf 供給はデフォルト直書き順位で層に負ける)。番号で固定強度を設計。§2.2 に明文化 |
| 上書き層の直交 2 軸化(番号付き多段層。D20 改) | **keep(今回 rename 実施)** | 実在の痛み=実験値の恒久/揮発混在(LL の旧 R 章約 100 行)、用途文字の消費実例(`M > O`)。機構変更ゼロ(命名規約のみ) |
| baseline からのアルゴ的キー抽出 | **defer-behind-gate** | 実害ゼロのうちは抽出しない。ゲート=起因事故 1 件目 or HEAD/exploration 軸導入時(§10) |
| backend / metrics 素材の `@` 化(D26) | **keep(今回実施)** | ③無印素材の完全消滅=指標 3 の名実達成。「Phase 1.5」という遅延枠は作らない |
| mode の `@` 化 | **cut(無印を正式仕様化=D25)** | mode は素材でなく「選択の源」。`@` 化の呼び出し側波及(bat/CLI/optuna)に見合う利得なし |
| `@bf16` / `@random` の agent.txt 共通化 | **keep(今回実施)** | env 重複の削減。`@random` は差分を全部入りで統一(学習しない構成のため挙動不変) |
| TauProj 名 / `@nature` 糖衣 / MuZero `@` 化 / flaky | **defer-behind-gate** | ゲートは §10 参照(flaky は 28impl 記録のみ: 1 回観測・再現せず、再発時に診断) |

- 実装、コード変更、config ファイルの書き換え(grill は完了。実装は PH 指定の依頼で開始し、実装計画は `059_*_2ximpl_phN.md` として impl 側で分割する)。
- Munchausen 等の新アルゴリズム自体の実装(TARGET 軸は器だけ定義)。
- Run 全体 save/load の checkpoint 契約(§7.3 で分界)。
- v1 §5 に記録した個別の設定 drift(`net.body.$` デッド、`net.branch.AtariNature` の 2 用法同居、taus K の spec 反映、`$include` 解決失敗の warn 継続)。`$include` の fail-fast 化は Phase 0 で同時に扱ってよいが、本 PRD の必須要件にはしない。

---

## 付録 A. 現行機構の事実(v1 §1.3 / §1.4、コード確認済み)

合成機構は `ConfigManager::AutoMerge`([config.cpp](../../../core/anet-core/src/config.cpp) の `MERGE_KEYWORD = ".$"` 以下、約 60 行)が全て。

| 事実 | 根拠 |
|---|---|
| マージ先 = LHS prefix 固定 | `RemoveSuffix(merge_key, MERGE_KEYWORD)` |
| merge キー判定は `.$` サフィックスのみ(裸 `$` は認識されない) | `EndsWith(key, MERGE_KEYWORD)` |
| `>` 区切り・左から右へ適用・右勝ち | `Split(merge_val, { ">" }, true)` の順次適用 |
| 子孫のみ対象(`backend.deterministic` が `backend.deterministic_algorithms` を巻き込まない) | `merge_target_key + "."` の `StartsWith` |
| **単一パス。`.$` の連鎖は解決されない** | 内側ループが展開前 `map` を参照 |
| マージ結果は行位置と無関係に素の同名キーを常に上書き | 非マージキーを先に `new_map` へコピー後、マージ結果を `Set` |
| マージ元キーは最終 ConfigData に残る(dump 汚染の源) | `//new_map.Erase(key2);` のコメントアウト |
| `R./A./X./E./M./P.` はスキーマも登録もない自由記述 namespace | C++ 側に該当文字列なし |
| 未定義・誤字レイヤは無警告 no-op | 該当キー 0 件なら何も起きない(`P.` は定義 0 件のままチェーンに実在) |
| CLI override は AutoMerge 前後に同一適用が 2 回 | `ConfigManager` コンストラクタ |

不整合時の実挙動(QR/IQN の例): `quantile_mode=iqn`×`net.$=net.qr` は WARN 1 行の後 Head の rank 検証で落ちる。逆は bind 解決エラー。`net.$` 書き忘れは**静かに別グラフが構築される**。3 つとも原因キー(`quantile_mode` / `net.$`)に言及しない。これは ADR 0018 の責任境界の帰結であり、同 ADR は 2 行切替を利点として記述していた(D19 の採用に伴い ADR 0030 が supersede し、ADR 0018 の Consequences は改訂済み)。

## 付録 B. 同時変更セットの棚卸し(v1 §2、12 件)

| # | 概念 | またがる namespace | 本 PRD での受け皿 |
|---|---|---|---|
| 1 | 分布表現(QR/IQN) | `net.*`×`DefaultDQNAgent.*`×`metrics.*` | ALGO 素材(D12)。metrics 素材のみ残置(§0.2) |
| 2 | dueling V/A | `net.body.output.*`×`net.branch.*`×`use_dueling_net` | HEAD 素材(遅延ゲート=§10)+features 接続点(§3.4 設計メモ) |
| 3 | frame stack 段数 | `stucker.*`×`net.block.*.reshape`×RB history margin | `${}`(数値同期)。RB 側は ADR 0024 で決定済み |
| 4 | env グリッド寸法 | `DropMergeEnv.*`×`net.branch.*`×`net.block.*` | BOARD 素材+`${}`(§3.2) |
| 5 | eval タグ | 4 namespace | 解決済みの先例(ADR 0027)。変更なし |
| 6 | BF16/AMP | `BF16.agent.*`×4 箇所の `use_amp`×`force_fp32` | precision 横断素材(語彙は ImageCls 系と要統一、§10-8 に含めず別件) |
| 7 | TBO | `learner.*`×`reward_scaler.*`×`target_policy.*` | 検証追加(目標 2)。素材化は必要時 |
| 8 | optimistic target | 1 bool→別ブロック全行 | 解決済みの先例(導出) |
| 9 | 更新頻度 | `update_interval`×`replay_ratio`×`warmup` | AS 後継素材(@fast 等)+warmup は独立(D11) |
| 10 | 学習予算 | `app.*`×`eps_decay`×`per_beta`×`warmup` | RUN_BUDGET / LEARNING_SCHEDULE 分離(D11)+`${}` |
| 11 | 並列度 | `num_envs`×`use_spatial_exploration` | 幹の値スロット+検証(目標 2)。素材化しない |
| 12 | 分位数と可視化 | `qr.num_quantiles`×`QValuePanel.smooth_radius` | `${}`(必要なら)。優先度低 |

詳細な各件の記述(該当キー・行の実例)は v1 に依拠する。v1 全文は本ファイルの git 履歴(初コミット時点の前版)または設計スレッドを参照。

## 付録 C. repo が既に持つ解法 4 例(v1 §3、確定済み前提)

| 先例 | 手口 | 本 PRD への継承 |
|---|---|---|
| ADR 0027 / PRD 052(eval 定義/schedule 分離) | namespace 分離+名前参照。定義は純粋、駆動が名前で参照 | カタログ(定義)と幹(schedule)の分離。`interval=0`=明示 OFF が「オーバーレイ構造下の無効化は値で表現」の制約として全案に効く |
| PRD 029(`net.config_profile`) | スカラー 1 個+構築時自動展開 | `${}` 値参照の先例 |
| PRD 033(ImageCls Head 化) | 構造で保証し設定を絞る | 「所有=構造」の先例(D19 の論拠) |
| `AS.fast\|balance\|heavy` | 束ねる名前(単一 namespace 内) | 素材の原型。agent 配下 `@fast` 等へ継承 |

「カタログ+束ね」の 2 例(DatasetKey/ImageDataSource、configured eval tag/eval schedule)も同型。

## 付録 D. 検討経緯(v1 §4 の 3 案からの収束)

- v1 案 1(concept overlay / root スコープ合成)→ **run 限定の root 持ち上げ**(§5.2 幹前段)として採用。「root へ書けるのは `run.*` と宣言された横断素材だけ」の規律で、案 1 の懸念(自由すぎて散らかる)を抑えた。
- v1 案 2(所有の移動)→ **ALGO の net 配線所有**として部分採用。net 読み口の移動は grill で決着し **D19 として採用済み**(実測 112 行で条件クリア。ADR 0030)。
- v1 案 3(契約検証)→ 目標 2 として採用。素材宣言(`@`)が検証の足場(参照 0 件 fail-fast)を提供する。
- 検討過程で棄却した表記案: キー先頭ドット(素材が root の別領域に集まり所有が切れる)、連続ドット(視認性)、bracket 流用(`[名前付きインスタンス]` と意味衝突)、セクションヘッダ(1 行完結・位置独立を壊す)、`::`(仰々しい)。
- 外部調査(LightZero): env×algo×実行形態の全組み合わせファイル方式は atari だけで 28 ファイル、「差分だけ書く」原則は実態として崩壊し全キー再掲の複製が広範、deep merge は新キー許可でサイレント typo。採ったのは「アルゴ既定を名前付きで 1 箇所に集約」「total_config ダンプ(=config_data.txt で既存)」のみ。全組み合わせファイル(AtariRainbow.txt 型)は不採用。比較: rl-zoo3 は 1 アルゴ 1 YAML+anchor、CleanRL は複製許容の対極。
