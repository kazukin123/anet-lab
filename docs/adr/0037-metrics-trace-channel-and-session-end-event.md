# 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する

scalar 行 `{step, tag, value}` には主体欄が無く、`mean.` 等の集約は「その step で完了した lane」を分母にするため、集約した時点で個体が失われる。PRD 060 の評価セッションで eval の N 本が 1 点へ畳まれた結果、eval 側の分位点・閾値越え率・score×len の同時分布が Run artifact から復元できなくなった。一方、`EpisodeEndEvent` は train 側では lane ごとに `env_index` 付きで既に届いており、metrics observer がそれを捨てているだけだった。eval 側は ADR 0034 が per-group event を抑制し、個体値は `EvalSessionEnv::CaptureScalars` が累積器へ畳む瞬間にしか存在しなかった。

**個体メトリクスは、scalar とは別チャネル trace（`type:"trace"`）として同じ Metrics マスタへ出す**ことを決定する。行は固定属性（`tag` / `step` / `lane`）を直書きし、宣言した個別値は `data` 下に置く。scalar と同じ設定 DSL トークン表記で `metrics.trace.[tag]` として宣言する。裸トークンは全てキーで1個以上・重複不可、イベント（`@episode_end` または属性形）と target は明示必須とする。イベント・target・scope・step軸の重複指定は同値・異値・別表記を問わず拒否し、未知・禁止トークンも設定読み込み時に fail-fast する。集約 prefix と EMA / interval / clip / `key:` は受理せず、`interval:1` のように既定値と同じ指定も拒否する。共通パーサーは指定の出現情報を保持し、scalar の既存パース挙動は変えない。宣言が無ければ何も生成されない（level は持たない）。定義レコードは scalar を **`metrics.scalar.defs` へ改名**し、trace は `metrics.trace.defs` を新設する。読み取りは **`metrics.scalar.defs` を優先し、存在しない場合だけ旧 `metrics.defs`** を読む。現用の過去 Run artifact への影響を避けるための互換例外とし、writer は新名だけを出力する。

系列は **`(type, tag)`** で区別し、scalar / trace の同名 tag を許可する。取得順と trace 定義の `keys` 配列は宣言順にするが、JSONL の `data` オブジェクトのキー順序は保証しない。既存の JSON 型と backend を使い、順序保持専用の書き込み経路やチャネル横断の tag 重複検査は追加しない。

**評価セッションのイベントは型で分ける**ことを決定する。採用 episode が完了するたびに `EpisodeEndEvent`（`env` = `EvalSessionEnv`、`env_index` = PER_LANE は group、SHARED は −1）を出し、セッション完了は新設の `SessionEndEvent` で 1 回通知する。scalar の eval metrics は `@session_end` に束縛し、trace は `@episode_end` に束縛する。`LastAdoptedGroups()` の内部 group index（SHARED は 0）を EvalRunner がイベントの index へ変換する。これにより trace observer は train と eval で同じコード（`GetScalar(key, env_index)` を完了 Step の直後に読む）になり、`EvalSessionEnv` は trace 用の個体値列を溜め込まない（既存 scalar 累積器と return 列は維持する）。eval scalar の旧書式 `$eval.[x] @episode_end` は途中経過の部分平均を点として出してしまうため fail-fast とし、alias や WARN は残さない。

## Considered Options

- **step の小数部へ env index を埋める**: cache の `step INTEGER`、`--range`、LOD bucket、`(runner, step_axis)` 座標系が整数 step 前提で、どこかで int 化されると個体が無言で潰れる。時間座標に個体識別を載せる設計であり却下。
- **scalar 行に識別子カラムを足す**: 問題の半分は解けるが、1 episode が複数行に分かれたまま原子性が無く、provenance（912 / 932）ではカラムが識別子の種類ごとに増える。契約を `json_lines` 側に持てないので却下。
- **既存 `type:"json"` 行で ad-hoc に出す**: 既存書き口は step を行に書かず side file を毎回作る。契約なしの行が 2 種類目になるので却下。
- **`EvalSessionEnv` が N 本を溜めて最後の 1 event で N 行出す**: N 本を取り出す口が「index = lane」の既存契約と同じ引数で衝突し、eval 専用経路になるので却下。
- **`EpisodeEndEvent` に kind（EPISODE / SESSION）を足す**: 型を分ければ scalar 側に実行時フィルタが要らないので却下。
- **eval scalar のトークン `@episode_end` を残して内部で SessionEnd に束縛**: 設定は無傷だが同じトークンがチャネルと scope で別イベントを指すので却下。

## Consequences

- 2026-09-05 の追加合意: 両チャネルの定義へ `scope` / `eval_name` / `eval_episodes` / `num_envs`、scalar 定義へ `clip` を追加する。`runner` は座標系所有者として維持し、購読先とは区別する。eval の採用予定数と構築済み lane 数は各 metric 定義へ重複保存する。train scope の eval 情報と未指定 clip は `null`。過去定義の欠落も不明として扱い、tag から推測しない。詳細は PRD §4.7 と現行 observability 設計に置く。

- eval scalar の設定行は `@episode_end` → `@session_end` へ機械置換する（リポジトリ内の現用設定・テスト・現行ドキュメントを同じ変更で移行。過去 Run の `config_data.txt` は記録として触らない）。ADR 0034 の「configured eval 中の per-group event を抑制する」は本 ADR で置き換える。
- scalar 定義の writer・ミラー・現用 fixture・文書は `metrics.scalar.defs` へ移行する。writer は空でない定義を新名で1回だけ出力する。`inspect_run.py` の master / cache 読み取りは基本として新名を読み、なければ旧 `metrics.defs` を読む。新 Run・過去 Run のどちらでも `def_source=metrics_defs` とし、旧名を読むことだけを理由とする WARN は出さない。
- 互換対象は旧名の定義を持つ現用の過去 Run artifact。移行方法は新しい Run の実行で、現用 Run 作業セットが全て `metrics.scalar.defs` を持つまで旧名を読み、その時点を削除条件とする。過去 artifact は書き換えず、旧名を二重出力しない。ADR 0029 の過去 Run 向け設定導出 fallback も削除条件を新名へ読み替える。
- 定義不在時の既存設定導出は維持し、`EVENT_NAMES` に `session_end` を追加する。cache 未構築の新 Run でも `metrics` の selector 展開と `tags --no-observed` がこの経路を通るためであり、過去 Run 向け互換の削除条件だけを理由に除去しない。詳細な到達経路は [PRD069 §4.7](../memo/069_metrics_trace_channel_10prd.md#47-書き口と定義レコード)に記す。
- trace 行を書く側は読み手の 3 制約を守る: `type` は文字列、`step` は整数（Metrics Viewer の ingest が Run 全体を ERROR にする）、top-level に数値 `value` を置かない（MLflow bridge が metric として送る）。既存の読み手は trace 行を捨てるだけで壊れない。
- trace のトリガは `@episode_end` のみ。`@train` / `@learn` / `@session_end` は fail-fast で閉じ、932（forensic）や 912（network version）が要るときに開ける。`episode_id` / `model_version` の欄も同じゲートで足す。
- `$runner` は index を無視してカウンタ等を返せるため、lane 指定なら必ず `nullopt` になるとは保証しない。未知キーの `nullopt` は fail-fast、既知値の NaN / ±Inf はキーを残して `null` とし、値の意味の選択は設定者が担う。
- 実装の受入では編集前 baseline と決定的 backend・foreground eval の同 seed Run を比較する。時間依存値を除く scalar の件数・step・値、eval1 / eval2 各3セッション以上、N=G=10 の各10行と scalar 平均の整合、分布の復元を要求する。N>G・N<G・SHARED・非採用完了・通知時の値取得・DSL 拒否条件・background の配送と例外伝播・既存 reader は別途検証する。未検証を合格扱いしない。
- train は共通 observer とコメントアウトした宣言例までとし、既定 OFF。残る機能は PRD §8 の需要発生まで保留し、将来用の拡張口だけも実装しない。
- 用語「trace」は CONTEXT.md で NN activation タップ（`TraceCallback`）と区別する。
- 詳細契約・受入条件・6項目の簡素化監査は [PRD069](../memo/069_metrics_trace_channel_10prd.md) に置く。現行設計文書・設定・AGENTS.md はコードと同じ変更で移行する。
