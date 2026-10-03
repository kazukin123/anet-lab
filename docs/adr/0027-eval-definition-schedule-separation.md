# configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する

`train.eval.[tag].*` はキーが 1 つでも存在するとタグが実 eval として列挙され、`interval` 省略時は既定 100 で背景 EvalRunner が起動していた。このため EvalPanel 用の設定キャリアとしてだけ `[eval_panel]` タグを書いていた LunarLander で全 Run に意図しない背景 eval が寄生し、Atari では SDL 音声デバイスの取り合い（無音化）として顕在化した。schema 上の根本原因は、`interval` / `use_background` の消費者が EpisodeEvalObserver（スケジューラ）ただ一つで Env / Runner の定義と無関係なのに、定義ブロックに間借りしていることである。他の消費者（metrics `$eval.[name]`、EvalPanel `eval_config_tag`）は全員「自分の namespace で宣言し、名前で参照」しており、EpisodeEvalObserver だけが例外だった。

**定義 `train.eval.[tag].*`（run_mode / eval_batch_size / clone_model / env.*。純粋で、書いただけでは何もインスタンス化しない）と、定期駆動 `train.eval_schedule.[tag].*`（interval / use_background。タグを名前参照）を別 namespace に分離し、Env + Runner + Observer の生成は有効な schedule エントリが駆動する**ことを決定する。`interval` は必須（既定なし。発端バグの原因＝暗黙の既定 100 を新 namespace へ持ち込まない）、`interval = 0` は明示 OFF として許可する（エントリ無しと同一挙動）。dormant は「interval=0 で宣言する状態」から「定義済み ∧ 有効 schedule 無し」の導出状態になる。schedule の未定義タグ参照は fail-fast。旧位置キーの検出は NoCare（クリーンブレーク方針どおり検出 tripwire を残さない）とし、取り残しは起動時のタグごとロールログ（scheduled / definition-only）が可視化する。

`interval = 0` を残す決め手は、トグルイディオム（コメント候補群 + 有効な 0 の 1 行切替）の温存に加え、**common 層の schedule を環境別 config で打ち消すにはエントリ削除では表現できない**ことである（ImageCls eval2 / GridMaze_muzero eval1 が現用）。config がオーバーレイ構造を持つ限り、「無効化を値で表現する」手段は必須になる。

## Considered Options

- **案 A: `interval` を現行位置のまま必須化（fail-fast）**: 症状は消えるが、EvalPanel 消費では `interval` は一切読まれないのに、trainer 非関与のタグへ trainer の語彙（interval=0 番兵）を強制する。run_mode / eval_batch_size / clone_model が省略可のまま interval だけ必須になる schema 歪みも生む。却下。
- **案 B: `interval` 既定を 100 → 0（出現 = opt-in）**: 症状は消えるが、スケジューラ設定が定義ブロックに寄生する構造は残り、「宣言＝起動」もつれの温床が温存される。却下。
- **案 C: schedule を別名エントリ + `tag =` フィールドで間接参照**: 同一タグへの多重 schedule が可能になり禁止検証が必要。多重周期駆動のユースケースが現状なく過剰設計。タグ名キーなら config キー一意性が構造的に多重を排除する。却下。
- **案 D（採用）: 兄弟 namespace `train.eval_schedule.[tag]`（タグ名キー）**: 既存慣例（自 namespace 宣言 + 名前参照)に揃い、定義は純粋化、dormant は導出状態になり魔法値宣言が消える。既存 config のオーバーレイ運用（common 打ち消し・トグル）も interval=0 で表現を維持できる。採用。

## Consequences

- LunarLander の `[eval_panel]`（`env.limit_step` + `eval_config_tag` のみ）は無変更で正しい形になり、寄生 eval だけが消える。暫定止血タスク「LunarLander に interval=0 追加」は不要化。
- リポジトリ管理下の全 config の `interval` / `use_background` 行は同一変更内で `train.eval_schedule` へ機械的に移動する（挙動等価。seed domain `"eval_env/" + tag` は不変）。Atari の `[eval_panel]` は interval 行削除で唯一の宣言キーが消えるため `run_mode = eval1` を明示して宣言を維持する。
- dead key `train.eval_interval`（RunManager::Config で読むだけで未使用）と CartPole.txt の現用記述を同時削除する。
- metrics の dormant タグ参照 WARN は旧語彙（interval=0）を排し、「定義済みだが有効な train.eval_schedule エントリが無い」ことと活性化手段を案内する文言に置き換える。
- EpisodeEvalObserver の実行機構（LEARN 軸判定・背景プール・`WaitBackgroundEval`）は変更しない。eval 同位相による Train 定期停止は別件。
- 詳細設計は `docs/memo/052_eval_schedule_separation_10prd.md`。
