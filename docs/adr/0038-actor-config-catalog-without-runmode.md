# Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す

configured eval tag は env・並列度・本数・clone をタグ単位で持てるのに、行動方策(ε 等)は `DefaultDQNAgent.eval_policy` 1 本で全タグ共通、評価対象 network(target / online)は `run_mode`(Eval1 か否か)への固定配線だった。Agent / Actor 内部で RunMode が意味を持つ箇所は 7 つあり(policy 選択、network 選択、clone 既定、定期 snapshot、Actor Q ヒント送出、MuZero の温度 / noise、RunMode 別共有 RNG)、いずれも「用途ラベルから Agent が Actor の中身を推測する」構造で、Runner 側の都合(train か eval か)を Agent 実装へ固定的に持ち込んでいた。

**Actor 設定(`CreateActor` が消費する設定の総体)を Agent 所有スキーマの名前付きカタログ `<AgentPrefix>.actor.[key].*` として宣言し、Runner は `ActorRequest { batch_env_spec, env_spec, device, seed, actor_key }` で名前参照だけを渡す**ことを決定する。Agent インタフェースは RunMode / `clone_model_override` / device 引数を持たない。既知キー `train` / `eval` は Runner の既定値(train runner は `train`、eval スロットは省略時にタグ名)であって Agent 実装は名前を解釈しない。差分はカタログ側で `$` 継承と `@` プロファイルで書き、スロット内に Actor 設定の上書き層は置かない。policy は Actor が専有し、スケジュールは MakeAction に渡される学習側 counts(直近の Sync 時点の train runner の StepCounts)で進める。

## Considered Options

- **スロット内上書き層(`train.eval.[tag].agent.eval_policy.*`、ENV と対称)**: 実害は満たすが Agent / Actor の定義に遡らない場当たり。request に override prefix が要り、Agent 側に未消費キー検査が要る。却下
- **2 ベース(`train_actor` / `eval_actor`)+ スロット上書き**: Agent 実装に train / eval の固定分岐が残る。却下
- **RunMode を request に残す**: 用途ラベルで分岐する構造が温存される。却下
- **policy カタログ + `policy_key`**: `[eval]` と `[eval_target]` の共有部分を宣言順コピーの制約下で書くための回避策。似た概念の二重化になるため却下し、リゾルバの依存順解決(PRD 072)で解く
- **env もカタログ参照へ(記法の完全対称化)**: env は「スロットが所有する上書き層」、actor は「スロットが参照するカタログ」という関係の違いがあり、`common.txt` のスロットが Agent 非依存のまま残る利点がある。P3 として方向だけ残す

## Consequences

- Eval1 / Eval2 の Agent 側依存は消える。RunMode の enum と Env 側利用(Train / Eval の Sampler 選択等)は残し、Eval1 / Eval2 の整理は PRD 061 P3 で行う
- `train.` root は `run.` へ改名する(P1、機械的)。Run プロファイル(`run.@<name>`)と同じ root だが、リゾルバが特別扱いするのは `run.$` / `run.@` だけなので衝突しない
- eval タグ名を eval1 → `eval_target`、eval2 → `eval` に改名する。metrics の tag 名(LHS)は不変で `$eval.[…]` の RHS だけ再指定するため、過去 Run との tag の意味は一致する
- RunMode 別共有 RNG を Actor 別 seed(`actor/<Runner 名>`)に置き換えるため、新旧バージョン間の同 seed bit 一致は不成立になる(新バージョン内の再現は成立)。EvalPanel と configured eval が同じ RNG stream を消費する結合は解消する
- metrics の参照先に `$actor` を追加し、`$agent epsilon` 等の Agent 経由の policy 値は削除する
- `clone_model` は Actor 設定の事項になり、EvalPanel の `model_sync.mode = shared` は廃止する
- カタログ項目間の `$` 継承が後段 overlay を取りこぼさないよう、設定リゾルバを依存順解決へ変える(PRD 072、先行)
- 詳細契約と受入条件は `docs/memo/061_eval_slot_policy_override_10prd.md`
