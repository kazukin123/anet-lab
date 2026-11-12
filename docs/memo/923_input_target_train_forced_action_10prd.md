# 操作入力の入力先切替と Train 側の強制行動（学習への反映）暫定 PRD

> 2026-11-03 番号を 999 から 923 へ振り直した（PRD114）。

> 暫定 PRD（グリル前のドラフト）。番号 999。§6 の未決を裁定してから正式採番する。本書は実装着手を意味しない。
> 起点: 2026-09-26。行動上書きは EvalPanel（鏡写しインスタンス）にしか届かず、学習中の lane を人が操作してその遷移を学習に使う経路が無い。
> 人間デモ Actor（人のプレイを ReplayBuffer へ入れる路線）の最初の一歩として、既存の Train ループに人の行動を通す。
> 関連: [922_input_mapping_joystick_10prd.md](922_input_mapping_joystick_10prd.md)（対 PRD。装置層とマッピング層はそちら。本 PRD は入力先の切替と Train 側の適用点だけを扱う）、
> [083](done/083_action_context_retire_10prd.md) → 観測パイプライン PRD（forced action を `MakeAction` 入力にする）→ [921](921_agent_prev_action_obs_10prd.md)（順序は 921 の 2026-09-26 方針）、
> [035](done/035_approx_actor_priority_per_10prd.md)（Actor Q ヒント）、[067](done/067_MunchausenRL_10prd.md)（log π 項）、[036](done/036_train_actor_periodic_snapshot_10prd.md)、
> [100 §Runner](../design/100_runtime_and_configuration.jp.md)、[160](../design/160_applications_and_tools.jp.md)、[CONTEXT.md](../../CONTEXT.md)（「学習側 counts」「鏡写しインスタンス」「実行時 UI 操作」）。
> 担当: 未定（グリル時に決める。既定の担当は定めない）。
> 改訂: 2026-10-23。098のReplay窓との依存を追記。trainの強制行動はskip状態のリセットと同stepの判断記録を前提にする。

## 1. Context / Problem Statement

### 1.1 Eval 側にしか強制行動が無い（2026-09-26 確認）

- 行動上書きの経路は `RunnerFrame::OnKey` / `QValuePanel` → `EvalPanel::DoStep(action)` → `EvalRunner::DoStep(action)` の 1 本で、
  EvalRunner が `MakeAction` の結果を `WithAction` で差し替える（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の `EvalRunner::DoStepInternal`）。
  EvalRunner は Learner を呼ばないので、強制行動の遷移は学習に一切入らない（[100 §Runner 表](../design/100_runtime_and_configuration.jp.md)「Learner を呼ばず、Actor と Env で評価または手動操作を進める」）。
- Train 側には強制行動の口が無い。`SerialTrainRunner::DoStep` / `PipelineTrainRunner::DoStep` は
  `actor_->MakeAction(step_counts_, state_)` → `env_->Step(action_info)` → `BatchExperience{state, action_info, reward, next_state}` → `learner_->UpdateFromBatch`
  （Pipeline は 1 step 遅れで非同期投入）の直線で、`action_info` がそのまま経験になる。
- Train ループは `RunnerThread`（学習スレッド）が `DoUpdateFrame(1, pre_step_func, post_step_func)` を回し続ける（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の `RunnerThread::ProcessStep`、
  [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) の `pre_train_step_func`）。GUI スレッドとの既存の連絡路は `pre_step_func` の atomic 参照と `wxQueueEvent` だけ。
- Train の BatchEnv は多 lane（Atari 128、LunarLander 256。`run.train.num_envs`）。TrainPanel の View は全 lane のモザイクで、
  スコア等の代表値は lane 0 の aux から取っている（[AtariView.cpp](../../core/envs/atari1/src/AtariView.cpp) `CreateViewData` の `aux_list[0]`、LunarLanderView も同様）。
- Train ループの速さは制御されていない。人が操作できる上限は実機速度（Atari は 60 fps / frame_skip 4 = 15 step/s）で、
  Train の step/s はそれを上回りうる（LunarLander 256 lane で 2M exp step ≈ 5 分 ≈ 26 train step/s）。

### 1.2 決まっている方針（921 D1、2026-09-26）

- forced action は事後の hook や `WithAction` ではなく **`MakeAction` の入力（lane mask + action）** にする。決定点を 1 つに畳むと
  「Actor 出力 = 実行 action」が不変条件になり、lane メモリ（prev_action）、Actor Q ヒント、Munchausen 項、将来の log π(a) が自動で正しくなる。
  この入力は 083 後続の観測パイプライン PRD が導入し、EvalRunner の `WithAction` 経路もそこで消える。
- 本 PRD はその入力の **Train 側の最初の利用者**。入力化が先に入っていることを前提に書く（順序は §6 Q1）。

## 2. ゴール / 非ゴール

**ゴール**

- **G1**: 操作入力の**入力先**を、Run の実行中に UI 操作（キー / ツールバー）で Eval（既存）と Train のあいだで何度でも切り替えられる。
  設定で固定する方式ではない。Eval 側の挙動は対 PRD のまま変えない。
- **G2**: 入力先が Train のとき、操作対象 lane の行動が操作入力で決まり、**その行動で Env が進み、生じた遷移が通常どおり ReplayBuffer に入って学習に使われる**。
- **G3**: 強制行動が Actor の出力として扱われる（不変条件）。経験・PER 初期優先度・Munchausen 項・prev_action の整合に特別扱いを足さない。
- **G4**: 人が追従できる速さで Train ループを回せる（入力先が Train の間だけ）。

**非ゴール**

- **NG1**: デモ専用バッファ（R2D3 / DQfD 型の固定比率混合）、模倣損失（margin loss 等）、人間遷移の優先度ブースト。共有 ReplayBuffer は FIFO で
  人間の遷移が薄まって消えるが、それは次の PRD の関心。本 PRD は「通常の経験として流す」まで。
- **NG2**: 複数 lane の同時操作、複数の操作者。
- **NG3**: 人のプレイの記録・再生（ファイル化）。
- **NG4**: 装置・マッピング（対 PRD）。
- **NG5**: batchrun 構成。人がいないので入力先は常に無効。
- **NG6**: 同 seed 再現性の維持。操作入力がある Run は再現の対象外（実効設定にも記録されない「実行時 UI 操作」）。

## 3. 提案（暫定）

### 3.1 入力先（input target）は UI 操作で切り替える

- 入力先は Run の実行中に**UI 操作で切り替える**。「実行時 UI 操作」（CONTEXT.md）として扱い、キー 1 つ（候補 `Tab`、§6 Q9）と
  Run 制御ツールバーのトグルの両方から、何度でも `eval` ⇄ `train` を行き来できる。Train / Eval の pause トグルと同じ位置づけで、
  実効設定（config dump）には記録されない。
- 現在の入力先はステータスバーに常時表示する（どちらに向いているか分からない事故を避ける）。切替はログ 1 行（`Input target: train lane=0` の形）。
- `eval` のときは対 PRD の経路そのまま。`train` のときは Eval 側へは何も流れない（Eval は Actor で回り続ける）。
- Train が pause 中に `train` へ向けても何も起きない（step が無い）。pause 解除を自動でしない（§6 Q4）。
- 設定は**起動時の初期値だけ**を持つ（持つかどうか自体は §6 Q10）。既定は現行挙動と同じ `eval`。

```
app.input.target ?= eval      # eval | train。起動時の初期値のみ。以後は UI 操作が上書きする
```

### 3.2 Train 側の適用点: `MakeAction` 入力

- `TrainRunner::DoStep` は `MakeAction` の直前に「操作対象 lane に対する強制行動の有無」を取り出し、lane mask + action として `MakeAction` へ渡す。
  Serial / Pipeline とも `MakeAction` は 1 箇所なので、差分は各 1 行 + 取り出し。
- Actor は mask の立った lane についてその行動を出力し、他 lane は通常どおり方策で選ぶ。Actor Q ヒント（035）は実行行動で計算され、
  Munchausen の bonus・log π 項（067）も実行行動に対して出る。**`WithAction` のような事後再 gather は要らない**（921 D1 の帰結）。
- 経験は現行どおり `action_info` から作る。ReplayBuffer / PER / N-step / frame stack は強制かどうかを知らない。
- 探索（ε-greedy 等）は強制 lane では効かない。Actor 内で mask 優先。

### 3.3 スレッド境界: 最新状態の 1 スロット

- 操作入力の解決結果（行動番号 or 無し）は GUI スレッドで作られ、学習スレッドが `DoStep` ごとに読む。**キューではなく「最新値 1 スロット」**
  （mutex か atomic）にする。保持型（対 PRD §3.3）なので「いま押されているもの」だけに意味があり、押下の取りこぼしを積む必要が無い。
- 学習スレッド側は step ごとに 1 回読む。Pipeline の 1 step 遅れの学習投入には影響しない（経験の中身が変わるだけ）。
- Runner への渡し方（`pre_step_func` のような関数注入か、Runner が参照する小さな共有オブジェクトか）は §6 Q2。

### 3.4 操作対象 lane

```
app.input.train_lane ?= 0
```

- 1 lane 固定。既定は View が代表値を取っている lane 0 に合わせる。範囲外は起動時に fail-fast。
- モザイク上でどのタイルが自分かは分かる必要がある。View 側で操作対象 lane を枠で示すか、EvalPanel のように 1 lane の拡大表示を出すかは §6 Q3。

### 3.5 ペース（G4）

```
app.input.train_min_step_interval_ms ?= 0    # 0 = 制限なし。入力先が train の間だけ効く
```

- 入力先が `train` の間、`DoStep` の間隔に下限を入れる（`pre_step_func` で sleep するのが最小）。Atari の実機速度は約 67 ms/step（15 step/s）。
- 入力先を `eval` に戻せば制限は消える。throughput への影響は操作中だけで、実効設定には残るが実行時 UI 操作の一部として扱う。
- Train は 128 lane を同時に進めるので、人が操作している 1 lane 以外の 127 lane も同じ速さに落ちる。人間操作中の学習は「遅い」を受け入れる。

### 3.6 学習への影響（把握しておくこと。本 PRD で対策しない）

| 事項 | 内容 |
|---|---|
| 分布のずれ | 人間の遷移は方策外。DQN 系は off-policy なので学習自体は成立する。Munchausen の log π 項は方策外の行動を罰する向きだが、clip −1 で 1 step あたり上限は小さい |
| 薄まり | 共有 ReplayBuffer では 1 lane × 操作時間ぶんの遷移が 128 lane の遷移に埋もれ、FIFO で押し出される。効果を出すにはデモバッファ（NG1）が要る |
| PER | 初期優先度は実行行動の Q から近似される（035）。人間の行動が方策と大きく違えば TD 誤差が大きく出て、しばらく優先的に引かれる |
| 再現性 | NG6。操作入力がある Run は同 seed 比較の母集団に入れない |

### 3.7 検証と fail-fast

| 状態 | 扱い |
|---|---|
| `app.input.target`（Q10 で残す場合）が `eval` / `train` 以外、`train_lane` が `num_envs` の範囲外、`train_min_step_interval_ms` が負 | 起動時に `ANET_SYSTEM_ERROR` |
| batchrun 構成で初期値 `target = train`（Q10 で残す場合。batchrun には UI が無いので設定でしか起きない） | 意図と矛盾するので fail-fast か、1 回 WARN して無効化か（§6 Q5） |
| Train が pause 中に操作入力がある | 何も起きない。ログも出さない |
| Learner 無効（`learner.enabled = false`）の Run で `target = train` | Env は進むが学習には入らない。契約違反ではないので 1 回 WARN |

## 4. 用語（CONTEXT.md へ足す候補。対 PRD §4 と合わせて確定）

| 語 | 意味 | _Avoid_ 候補 |
|---|---|---|
| **入力先**（input target） | 操作入力が届く Runner の選択。`eval`（鏡写しインスタンス）か `train`（学習 Runner の操作対象 lane） | モード（用途ラベル）, 操作モード, フォーカス（GUI のフォーカスと紛れる） |
| **操作対象 lane**（operator lane） | Train BatchEnv のうち操作入力を受ける 1 lane。View の代表 lane と揃える | player lane, human lane |

「強制行動」「操作入力」「操作マッピング」は対 PRD §4。

## 5. 受入条件（暫定）

1. 入力先を `train` に切り替え、操作対象 lane を Joystick / キーで操作すると、その lane の `action_info` が操作どおりになり、
   他 lane は方策で動く（テストは Actor に渡る lane mask + action と、`TrainEvent` の `action_info` で確認）。
2. その step の経験が ReplayBuffer に入り、`UpdateFromBatch` に含まれる（強制 lane の遷移が sample されうる。Learner 側に特別扱いが無いこと）。
3. 強制 lane の Actor Q ヒント・Munchausen 項・（921 後は）prev_action が実行行動と一致する。`WithAction` 相当の再 gather コードが Train 側に存在しない。
4. 入力先が `eval` のときは既存挙動（対 PRD）と一致し、Train 側の `DoStep` に差分が観測されない（mask が全 false）。
5. `train_min_step_interval_ms` を設定すると、入力先が `train` の間だけ step 間隔が下限以上になり、`eval` へ戻すと制限が消える。
6. 入力先の切替がステータスバーとログに出る。
7. [020 §6](../design/020_user_guide_run.jp.md)、[100 §Runner](../design/100_runtime_and_configuration.jp.md)、[160](../design/160_applications_and_tools.jp.md) が新契約に同期している。

## 6. 未決（グリル対象）

| # | 論点 | 選択肢 / 備考 |
|---|---|---|
| Q1 | 順序 | (a) 観測パイプライン PRD（`MakeAction` 入力化）を待つ (b) 本 PRD が `MakeAction` 入力を先に導入し、EvalRunner の `WithAction` も同時に置き換える。921 の順序は (a)。(b) は 083 後続 PRD のスコープを本 PRD へ寄せる |
| Q2 | Runner への注入の形 | (a) `DoUpdateFrame` の `pre_step_func` 相当の関数を Runner が呼んで「次 step の強制指定」を受け取る (b) Runner が共有の小オブジェクト（最新値 1 スロット）を参照する (c) `TrainRunner` に `SetForcedAction(lane, action)` を生やす。GUI 依存を core へ持ち込まないのは全案共通 |
| Q3 | 操作対象 lane の見せ方 | (a) モザイクのタイルに枠 (b) EvalPanel 相当の 1 lane 拡大 pane を Train 用にも出す (c) 何もしない（lane 0 は左上）。Atari の 128 タイルで (c) は厳しい |
| Q4 | pause 中に `train` へ向けたとき | (a) 何もしない（提案） (b) 自動 resume。(b) は「Train を動かした覚えが無い」事故を作る |
| Q5 | batchrun で初期値 `target = train`（Q10 で設定を残す場合だけ成立） | (a) fail-fast (b) WARN 1 回で無効化。batchrun は人がいないので (a) で困らない |
| Q6 | 何も押していない step の扱い | 対 PRD Q4 と同じ。Train へ向けるときは「人が離した瞬間に方策へ戻る」(actor) と「離している間は NOOP」(noop) で学習に入る遷移が変わる。Env ごとに既定が違ってよい |
| Q7 | 可観測性 | 強制 step 数や強制率を metrics に出すか。最小はログ（切替時と、操作の有無が変わったとき）。scalar を出すなら `$runner` 系の key 1 つ |
| Q8 | ペースの入れ方 | (a) `pre_step_func` で sleep（最小） (b) Env step 側で待つ。(a) は Pipeline の学習投入も一緒に遅くなるが、操作中は許容 |
| Q9 | 入力先の切替キー | `Tab` 候補。`SHIFT` / `SPACE` / `CTRL` の既存割当と衝突しないもの |
| Q10 | 初期値を設定に持つか | (a) 持たない。常に `eval` で起動し、切替は UI 操作だけ（設定キーが 1 つ減り、§3.7 / Q5 の `target` 行も消える） (b) `app.input.target` で起動時の初期値だけ選べる。(b) が要るのは「最初から Train 向けでデモを始めたい」場合だけで、実行時 UI 操作で足りるなら (a) |

## 7. スコープ外（再掲）

NG1〜NG6。特にデモバッファ・模倣損失・人間遷移の別扱いは、本 PRD で人の遷移が普通に流れることを確認してから次の PRD で扱う。

## 8. 実装に触れる見込みの箇所

- [trainer.hpp](../../core/anet-core/include/anet/trainer.hpp) / [trainer.cpp](../../core/anet-core/src/trainer.cpp) — `SerialTrainRunner::DoStep` / `PipelineTrainRunner::DoStep` の `MakeAction` 直前（lane mask + action の取り出し）。Q2 の形で注入口。
- Actor 側の `MakeAction` 入力（観測パイプライン PRD の成果物。Q1 次第で本 PRD）。
- [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) — `train_lane` / `train_min_step_interval_ms`（Q10 で残すなら `app.input.target` も）の読み込みと検証、`pre_train_step_func` のペース制御。
- [RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) — 入力先トグル（キー + ツールバー）、ステータスバー表示、対 PRD の入力層の出力を入力先へ振り分け。
- View（[AtariView.cpp](../../core/envs/atari1/src/AtariView.cpp) 等） — Q3 の見せ方。
- [common.txt](../../apps/runner/config/common.txt) の既定、[020](../design/020_user_guide_run.jp.md) / [100](../design/100_runtime_and_configuration.jp.md) / [160](../design/160_applications_and_tools.jp.md)、CONTEXT.md（§4）。

## 9. 関連

- [098 TempoRL](098_temporl_skip_policy_10prd.md) / [観測パイプライン単位A](944_observation_pipeline_10prd.md): skip有効時は強制したlaneの継続状態を捨て、強制したstep自体を `decision=true` で記録する。Replay窓は判断間の行動一致を前提とし、不一致はSample時にfail-fastするため、trainへの入力注入はこの2条件を満たすActorを前提にする。

- [922_input_mapping_joystick_10prd.md](922_input_mapping_joystick_10prd.md) — 対 PRD。装置・マッピング・押下の意味論。
- [083](done/083_action_context_retire_10prd.md) / [921](921_agent_prev_action_obs_10prd.md) — `MakeAction` 入力化の方針と順序（921 D1、2026-09-26）。
- [035](done/035_approx_actor_priority_per_10prd.md)（Actor Q ヒント）、[067](done/067_MunchausenRL_10prd.md)（log π 項）— 強制行動が Actor 出力になることで整合が自動になる対象。
- [036](done/036_train_actor_periodic_snapshot_10prd.md) — Train Actor の snapshot 同期。強制行動は同期と無関係だが、Actor 内の mask 処理は snapshot 経路の外に置く。
- [ADR 0013](../adr/0013-actor-network-resource-policy.md) — Actor の Resource 所有。強制指定は Resource ではなく `MakeAction` の入力。
