# 鏡写し eval インスタンスは dormant タグの trace を後から購読し、trace 行は時刻を持つ

EvalPanel は configured eval tag の内容（run_mode / env overlay / actor）を鏡写し参照する別インスタンスの EvalRunner で、
`RunManager` 構築後に `CreateEvalRunner("EvalPanel", tag)` で作られる。metrics observer の結び付け（`resolve_runner`）は
構築時に 1 回だけ行われ、後から作る Runner には何も付かない。参照先の `eval_panel` は definition-only（dormant）なので、
`$eval.[eval_panel]` を参照する trace 宣言は WARN で捨てられ、EvalPanel のゲームは trace に 1 本も残らなかった。
定義レコード（`metrics.trace.defs`）は構築時に 1 レコードだけ出し、解析側はそれを正本として読む（ADR 0029）。

さらに 2 点ある。EvalPanel が駆動する `DoStep()` の `EpisodeEndEvent` は EvalRunner 自身の `step_counts_` を載せる一方、
定義レコードの座標系所有者は `OwningRunner(EVAL, EPISODE_END)` = `train` 固定で、そのまま trace を付けると定義と行が食い違う
（configured eval は `RunSession` 冒頭で `Sync(event_counts)` した同じ counts を載せるので既に同値）。
また trace 行は時刻を持たず（PRD 069 D6「scalar と同じ最小形」）、録画の時刻から該当ゲームを引くにはログと trace の序数を数える必要があった。
`type:"json"` レコードは既に `timestamp` を持ち、Metrics Viewer の cache も `json_lines.timestamp` 列を持っている。

**`RunManager` は dormant タグを参照する trace observer を捨てずに保持し、同じタグで `CreateEvalRunner` された Runner へ
結び付けて手放す。保持した trace の定義は起動時の定義レコードに載せる。EvalRunner の `@episode_end` / `@session_end` は
常に学習側 counts に載せ、trace 行は固定属性として `timestamp` を持つ**ことを決定する。

- 保持するのは trace だけ。scalar は現行どおり dormant タグでは WARN して捨てる（鏡写しは人が開始・停止・手動操作するので統計の母集団ではない。trace は 1 ゲーム 1 行で母集団の前提を持たない）。
- 結び付けは最初の `CreateEvalRunner` で行い、保持から外す。同じタグの 2 つ目の Runner、および scheduled タグの鏡写し（trace は configured インスタンスに付いている）には付かず、INFO を 1 行出す。fail-fast にはしない。
- 定義レコードは構築時 1 回のまま。dormant タグの trace 定義は `eval_episodes=null` / `num_envs=null` で載せ、`runner` は `train`。誰も Runner を作らなければ行の無い定義が残る。
- counts 無しの `EvalRunner::DoStep()` / `DoStep(action)` は `source_counts_` を event に渡す。`RunSession` は同値なので configured eval の挙動は変わらない。
- `timestamp` は `type:"json"` レコードと同じ `GetCurrentTimeStr()`（`%Y-%m-%dT%H:%M:%S`、ローカル時刻、秒精度）。全 trace 行に付く。identity は従来どおり序数が持ち、`timestamp` は座標でも identity でもない。読み手の 3 制約（`type` は文字列、`step` は整数、top-level に数値 `value` を置かない）は変えない。

理由は 3 つある。**生成 API を変えない**ので、`CreateEvalRunner` を使う既存の app とテストはそのままで、変更は結び付けの 1 箇所に閉じる。
定義に dormant trace を載せる代償は「行の無い定義が残りうる」だけで、読み手は空として扱える。
**イベント counts を学習側 counts に揃える**ので、鏡写しの trace `step` が定義レコードの `runner=train` と一致し、
CONTEXT.md「step座標系」の「同じ eval tag でも `@episode_end` は train 側のカウンタに載る」が全ての EvalRunner で成立する。
**時刻を行に持つ**ので、trace 1 行で「いつ・何が」が閉じ、ログとの突き合わせ手順も購読先の Env name も要らない。
書式と欄名を `type:"json"` と揃えるので、Viewer の ingest と cache は無変更で受ける。

## Considered Options

- **起動時宣言（`RunManager(config, {name, config_tag}...)` で構築時に鏡写しを生成し、`CreateEvalRunner` を撤去）**: 「定義 = 実購読先・1 回」を
  厳密に守れるが、ctor API 変更・RunnerApp/RunnerFrame 変更・既存テスト 6 件の移行を伴い、結び付け 1 箇所の変更に対して大きすぎる。
  2026-09-24 に一度採用し、2026-09-26 に取り下げた。却下。
- **eval 定義に `on_demand` 印を置いて載せる定義を選ぶ**: 用途ラベル。アプリが Runner を作るかどうかは設定の関心ではない。却下。
- **`CreateEvalRunner` 時に定義レコードを追記し、読み手を複数レコードのマージへ変える**: 読み手契約の変更。起動時に載せれば不要。却下。
- **定義レコードに購読先の `env_name` 欄を足す**: 序数でログの lane 名と対応させるために要った。時刻を行に持てば不要。却下。
- **序数で結合する（k 番目の trace 行 ↔ k 番目のゲーム完了ログ行）**: 手順としては成立するが、録画の時刻から引くたびにログと trace を数える。却下。
- **env の `game_index` / trace 行の `episode_id` を突き合わせキーにする**: 時刻で足りる。`episode_id` は ADR 0037 のゲートのまま。却下。
- **`timestamp` をミリ秒・UTC にする**: `type:"json"` と書式が割れる。EvalPanel は 1 lane・15 fps で 1 秒に 2 ゲーム終わらず、identity は序数が持つ。却下。
- **鏡写しの `step` を自身の counts に載せ、定義の `runner` をインスタンス名にする**: 定義の座標規則がインスタンス依存になり、
  「`@episode_end` は train 側」の規則と矛盾する。却下。
- **全鏡写しにも trace を付け、行に Runner 名欄を足して区別する**: scheduled tag の鏡写しでは eval の分布にパネルのゲームが混ざる。却下。
- **scalar も鏡写しに付ける**: 統計の母集団ではない。却下。
- **2 つ目の `CreateEvalRunner` を fail-fast**: 保持分を手放す実装で自然に 1 つに閉じる。INFO で足りる。却下。
- **ADR を 2 本に分ける（結び付け / EvalRunner の counts）**: counts の統一は configured eval の挙動を変えず、鏡写しの step を
  定義と一致させるための帰結。一つの契約として本 ADR に置く。却下。

## Consequences

- ADR 0037 の行契約を改訂する: 固定属性は `tag / step / lane / timestamp`。PRD 069 D6 の「`timestamp` 無し」は本 ADR で置き換える。
  `42_env` / `51`〜`53` の既存 trace 行にも `timestamp` が付く。
- 同 seed 比較（ADR 0037 の受入方式）では trace 行の `timestamp` を除いて比べる。`type:"json"` レコードの `timestamp` と同じ扱い。
- `inspect_run.py trace-csv` は固定列に `timestamp` を足し、欠落した旧 Run は空欄にする。Metrics Viewer は無変更。
- 鏡写しの trace `step` は終局直前の Sync 時点の学習側 `exp_step`。学習停止中は全ゲームが同じ step になる。
  開始時のスナップショットが要るときは ADR 0037 の `model_version` ゲートで欄を足す。
- batchrun でも鏡写しは生成され定義に載るが、EvalPanel を動かさなければ行は出ない。
- scheduled tag を鏡写しにする設定（`app.eval_panel.eval_config_tag = eval`）は正当なままで、結果も現行と同じ（パネルに metrics は付かない）。
- 用語は CONTEXT.md に「鏡写しインスタンス」を追加し、「dormant」と「学習側 counts」に購読規則と counts の規則を追記する。
- 詳細設計は `docs/memo/081_evalpanel_episode_trace_10prd.md`。
