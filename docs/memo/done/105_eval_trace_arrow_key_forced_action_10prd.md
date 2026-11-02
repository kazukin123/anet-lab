# Evaluation Trace の矢印キーと強制した step の見分け、表の並べ替えと行のダンプ PRD

> 2026-10-29 起票（暫定。質疑は省き、決めきれない点は §9 の未決表に暫定の既定を置いた）。
> 2026-10-29 再グリル（Claude Code、`anet-create-prd`。この PRD の最初のグリル）: 暫定の D1「pane が開いている間」を「Trace モードの間」に改め、G1 をそれに合わせた（ユーザーの裁定）。Trace モードの間の ←→ は何もしないことにした。`forced` の印は `WithAction` の結果に付けると決めた。ユーザーの求めで、表の並べ替え（G4、D6〜D8）と、行のダブルクリックでのダンプ（G5、D9・D10）を足した。用語「強制行動」を [CONTEXT.md](../../CONTEXT.md) に足した（暫定の §9 の 5 を改めた）。§9 の未決はすべて閉じた。
> 2026-10-29 別モデル（Fable）の検討のあと（同じ枠）: D1 の代償の文を正した（Live モードの矢印は、行を選んで Trace モードに入るまで押すたびに強制行動になる）。ユーザーは D1・D7 を理由つきで維持した。強制の値を `$action_info forced`（`WithAction` の結果に印）から `$runner forced`（EvalRunner が答える）へ改め（D3）、ダンプの VERBOSE を改行を含めない 1 行から既存の文字列化の複数行へ改めた（D10）。受入 3 を新しいテストに直し（既定列を比べる既存のテストは無かった）、D5 の連続行動の文を削り、D9 に合計のメモリを書いた。G5 の「2 行」は「INFO の 1 行と VERBOSE の 1 件」に改めた。
> 起点: qbert の EvalPanel を Evaluation Trace で調べていたとき、行をたどるつもりで押した矢印キーが EvalPanel の強制行動になった。Agent は頂上から UP して最後の 1 機を失い、Trace には「判断の行なのに Q_sa が V_s より低い」行が並んで、Agent の挙動と読み違えかけた（§1.3）。
> 関連: [094](094_evalpanel_step_history_10prd.md)（Evaluation Trace。D11 の上下キーの文を本書の D1 が、D12 の再生の向きを本書の D7 が置き換える）、[999_input_mapping_joystick](999_input_mapping_joystick_10prd.md)（操作マッピングの設定化。Env ごとの割り当てはそちら）、[999_input_target_train_forced_action](999_input_target_train_forced_action_10prd.md)（入力先 train の強制行動）、[999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md)（強制行動と Q の表示の時点のずれ）、[999_eval_session_step_trace](999_eval_session_step_trace_10prd.md)（評価 session の step trace を Run の記録に残す）、[081](081_evalpanel_episode_trace_10prd.md)（D6 手動操作のログ）、[CONTEXT.md](../../CONTEXT.md) の「実行時 UI 操作」「強制行動」「Live モード／Trace モード」「step trace」。

## 1. 問題

### 1.1 今の経路（2026-10-29 にコードで確かめた）

- anet の Panel（Env の View の Panel、QValuePanel、TrainPanel、Conv2dPanel など）は、`wxEVT_CHAR_HOOK` を受けて、キーを `ForwardedKeyEvent` で Frame へ転送する（[gui.cpp:66-126](../../core/anet-core/src/gui.cpp) の `Panel::Panel`・`Panel::OnKey`、[AtariView.hpp:29](../../core/envs/atari1/src/AtariView.hpp)、[QValuePanel.hpp:59](../../apps/runner/src/QValuePanel.hpp)、[TrainPanel.hpp:17](../../apps/runner/src/TrainPanel.hpp)）。Panel をクリックすると、その Panel にフォーカスが移る（[gui.cpp:75-77](../../core/anet-core/src/gui.cpp)）。
- Evaluation Trace の pane（EvalTracePanel）、LogPanel、EvalPanel 本体は anet の Panel ではない（[EvalTracePanel.hpp:18](../../apps/runner/src/EvalTracePanel.hpp)、[LogPanel.hpp:10](../../apps/runner/src/LogPanel.hpp)、[EvalPanel.hpp:40](../../apps/runner/src/EvalPanel.hpp)）。EvalTracePanel は、表にフォーカスがあるときだけ、表の `wxEVT_CHAR_HOOK` で ↑↓ を行の移動に使い、キーを先へ渡さない（[EvalTracePanel.cpp:180-192](../../apps/runner/src/EvalTracePanel.cpp)）。
- `RunnerFrame::OnKey` は、↑=0、←=1、↓=2、→=3、テンキー 0〜9 をそのまま行動番号にして、`EvalPanel::DoStep(action)` を 1 回呼ぶ（[RunnerFrame.cpp:1038-1083](../../apps/runner/src/RunnerFrame.cpp)）。QValuePanel の行動の行のクリックも、同じ `EvalPanel::DoStep(action)` を呼ぶ（[RunnerFrame.cpp:502-505](../../apps/runner/src/RunnerFrame.cpp)、[QValuePanel.cpp:896-908](../../apps/runner/src/QValuePanel.cpp)）。矢印の割り当ては LunarLander 向けで、コードのコメントは NOOP・LEFT ENGINE・MAIN ENGINE・RIGHT ENGINE、ユーザーガイドの表も「LunarLander向け」と書く（[020 §6.2](../design/020_user_guide_run.jp.md)）。
- qbert の行動は 0 NOOP、1 FIRE、2 UP、3 RIGHT、4 LEFT、5 DOWN である（Trace の action 列の表示名）。そのため、↓ キーは UP、↑ キーは NOOP になる。
- `EvalPanel::DoStep(action)` は、Live モードへ戻してから（094 の D11。[RunnerFrame.cpp:455-457](../../apps/runner/src/RunnerFrame.cpp) で `EvalTracePanel::ReturnToLive` を呼ぶ）、EvalRunner に 1 step を強制させる（[EvalPanel.cpp:184-199](../../apps/runner/src/EvalPanel.cpp)）。強制行動の入口は離散の行動番号だけである（`EvalPanel::DoStep(int64_t)`・`EvalRunner::DoStep(int64_t)`）。EvalRunner は `MakeAction` のあと、指定された行動（`action >= 0`）があれば `WithAction` で行動を差し替える（[trainer.cpp:274-283](../../core/anet-core/src/trainer.cpp)）。本番のコードで `WithAction` を呼ぶのはここだけで、テストは行動を差し替えるために使う（例: [replay_buffer_test.cpp:938](../../core/anet-core/src/replay_buffer_test.cpp)）。
- step trace の Q_sa と V_s は、差し替えた後の行動と、aux の `q_values` から作り直す（[dqn_based_agent.cpp:552-578](../../core/anet-core/src/dqn_based_agent.cpp)）。一方、TempoRL の判断の旗（`skip_decision`）とスキップ長（`skip_length`）は、Agent の内部の値のまま残る（[dqn_based_agent.cpp:2505-2556](../../core/anet-core/src/dqn_based_agent.cpp) の `Actor::ApplySkip`）。そのため、強制した行は「判断の行なのに Q_sa が V_s より低い」形になりうる。
- step trace には、その行の行動が強制かどうかを示す値が無い。`$action_info` で全 Agent に共通に読めるのは `action` と `action.[<次元>]` だけである（[rl.cpp:419-444](../../core/anet-core/src/rl.cpp) の `BatchActionInfo::GetScalar`）。081 の D6（手動操作のログ行）は実装されておらず、`EvalPanel::DoStep(action)` はログを出さない（[EvalPanel.cpp:184-199](../../apps/runner/src/EvalPanel.cpp)）。
- 094 の D11 は「上下キーは、フォーカスのある窓が受ける。表にフォーカスがあれば行を動かし、EvalPanel にフォーカスがあれば今までどおり行動になる」と決めた（[094 §D11](094_evalpanel_step_history_10prd.md)）。

### 1.2 何が困るか

- Trace を見るつもりで ↑↓ を押しても、フォーカスが表以外（Env の View、QValuePanel など）にあれば、行は動かず、EvalPanel が強制行動で 1 step 進む。強制すると Live モードへ戻るので、見ていた行の選択も外れる。キーを押し続けると、キーの自動の繰り返しの数だけ step が進む。
- Env によっては、矢印の向きと行動の向きが合わない。qbert では ↓ で上へ跳ぶ。
- 強制した行と Agent が選んだ行を、Trace の上で見分けられない。強制した行を Agent の挙動として読んでしまう。
- 表は新しい順に固定で、値の大きい行（prio、Q_sa など）や強制した行を探すには、目で追うしかない（§1.4）。
- 行の列に出していない値（全行動の Q 値、行動の情報の info、観測）は、Trace の上で見られない。step trace を Run の記録に残す機構もまだ無い（[999_eval_session_step_trace](999_eval_session_step_trace_10prd.md)、[081](081_evalpanel_episode_trace_10prd.md)）。

### 1.3 起きたこと（2026-10-29）

PRD103 の段 3 の分割の Run（`run_20261028-215839_a5_qb_aln_t103spl_s1`。学習は 46M 前後）の EvalPanel を Evaluation Trace で調べた。ユーザーが録画した画面で確かめた事実は次のとおりである。

- ep_step 2880〜2946 の行の行動は、`2:UP` と `0:NOOP` だけだった。判断の行（decision = 1）でも Q_sa が V_s より低かった（例: ep_step 2932 の判断の行で Q_sa 14.17、V_s 15.05。ep_step 2939 の判断の行で Q_sa 5.82、V_s 7.37）。最後は頂上から UP して最後の 1 機を失い、ep_step 2946 でエピソードが終わった。
- 同じ Trace の前の部分（ep_step 2155〜2643）の行動は `3:RIGHT`・`4:LEFT`・`5:DOWN` を含み、判断の行は Q_sa と V_s が等しかった。矢印キーでは 4 と 5 を出せない。
- ユーザーは、Trace を見るために矢印キーを押していたと述べた。どの窓にフォーカスがあったかの記録は無い。表にフォーカスがあれば行の移動になる（§1.1）ので、表以外にあったと見られる。

### 1.4 並べ替えとダンプに関わる今の事実（2026-10-29 にコードで確かめた）

- 表は新しい行を先頭へ足し（[EvalTracePanel.cpp:264-266](../../apps/runner/src/EvalTracePanel.cpp)）、列見出しのクリックには何もしない（表の購読は [EvalTracePanel.cpp:144-192](../../apps/runner/src/EvalTracePanel.cpp) のセルのクリック・選択・キーだけ）。上限 `max_rows` を超えると、いちばん古い行を消す（[EvalTracePanel.cpp:267-271](../../apps/runner/src/EvalTracePanel.cpp)）。
- 再生の部品 `gui::TracePlayer` は、古い→新しいの順の行 id の列を受け、再生・コマ送り・コマ戻し・端・Live の起点を、その列の前後だけで決める。Live からの再生は列の先頭、Live の起点と止まる端は列の末尾である。選んでいる行が列から消えると、列の先頭（いちばん古い行）へ移る（[gui.hpp:417-440](../../core/anet-core/include/anet/gui.hpp)、[gui.cpp:439-562](../../core/anet-core/src/gui.cpp)）。EvalTracePanel は、行を足すたびに全行の id を渡し直している（[EvalTracePanel.cpp:274-277](../../apps/runner/src/EvalTracePanel.cpp)）。
- 行を足すのは Live モードの間だけである（Trace モードに入ると EvalPanel が止まる。強制行動、行動を指定しない step〔ツールバーの Step、`Ctrl`〕、Resume は、どれも Live モードへ戻してから進める。094 の D11、[EvalPanel.cpp:174-199](../../apps/runner/src/EvalPanel.cpp)）。そのため今は、行を選んでいる間に上限で行が消える経路は無い。
- 列の target `$runner` は、通知の中で同期に読まれ、その遷移を数えた counts などを答える（[metrics.cpp:225-228](../../core/anet-core/src/metrics.cpp)、[trainer.cpp:176-205](../../core/anet-core/src/trainer.cpp) の `RunnerBase::GetScalar`）。既定の列の step・ep_step も `$runner` である（[common.txt:163-164](../../apps/runner/config/common.txt)）。
- 表の行が持つのは、列の値（`StepTrace`）と追従 pane のスナップショットだけである（[metrics.hpp:43-53](../../core/anet-core/include/anet/metrics.hpp)、[EvalTracePanel.cpp:248-258](../../apps/runner/src/EvalTracePanel.cpp)）。Atari の EvalPanel の View のスナップショットは 1 行あたり約 120〜220KB で、1,000 行のメモリは追従 pane を合わせて約 130〜300MB と見積もられている（[094](094_evalpanel_step_history_10prd.md) §4 のメモリの見積もり）。列の値はすべて数値で、表示の書式は整数なら整数、それ以外は有効数字 4 桁である（[160 §3.4](../design/160_applications_and_tools.jp.md)）。
- 行を作る通知（`TrainEvent`）のうち、遷移の経験 `experience` は参照で渡され、通知の間しか生きない。行動の情報 `action_info` と Step の結果 `step_result` は共有の参照である（[rl.hpp:1001-1013](../../core/anet-core/include/anet/rl.hpp)）。
- Atari の Step の結果は、観測（1×84×84 の uint8、7,056 バイト）と、aux の `rgb_frame`（3×210×160 の uint8、100,800 バイト）を持つ（[AtariEnv.cpp:60-70](../../core/envs/atari1/src/AtariEnv.cpp)、[AtariEnv.cpp:631-641](../../core/envs/atari1/src/AtariEnv.cpp)）。Step の結果の Tensor は、次の step で書き換わらない（[AtariEnv_test.cpp:1572-1577](../../core/envs/atari1/src/AtariEnv_test.cpp)）。
- `anet::ToString` は Tensor の全要素を、入れ子の括弧と改行で書く。精度は有効数字 4 桁である（[tensor_util.cpp:41-53](../../core/anet-core/src/tensor_util.cpp)）。既存の文字列化は、状態（`BatchState::ToString`。観測・done・truncated・episode_start）、行動の情報（`BatchActionInfo::ToString`。行動と aux。info と Replay初期優先度ヒントは書かない）、Step の結果（`BatchStepResult::ToString`。報酬・遷移後の状態・継続の状態・aux）にある（[rl.cpp:375-385](../../core/anet-core/src/rl.cpp)、[rl.cpp:487-520](../../core/anet-core/src/rl.cpp)）。EvalRunner の DEBUG ログも、これらを複数行で出している（[trainer.cpp:275-290](../../core/anet-core/src/trainer.cpp)）。
- `LOG::verbose()` は wxWidgets の Info の level で出る。ログ欄は View メニューの Log Level で絞り、verbose を選んだときだけ Info を出す（[log.hpp:214-215](../../core/anet-core/include/anet/log.hpp)、[LogPanel.cpp:44-53](../../apps/runner/src/LogPanel.cpp)）。
- 094 は「Evaluation Trace で Agent の入力を表示する」を先送りにし、再開条件を「必要と分かったとき、選択行の入力を文字列でログへ出す形から起こす」とした（[094 §6](094_evalpanel_step_history_10prd.md)）。本書の G5 は、ユーザーが必要と述べたので、この形で再開する。ただし、通知が運ぶのは Env の観測（Atari では 1 フレーム）で、Actor が内部で積み重ねた入力（frame stack）は入っていない。

## 2. ゴールと非ゴール

**解く問題:** Evaluation Trace で行を調べている人が、意図せず EvalPanel を強制行動で進めてしまうこと。強制した行を Agent の行と見分けられず、挙動を読み違えること。表の行を値で並べて探せないこと。行の列に出していない値を、その場で見られないこと。

**ゴール**

- G1: Trace モード（表で行を選んで見ている間）は、どの窓にフォーカスがあっても、矢印キーで強制行動が起きない。今は強制行動になる窓（anet の Panel。Env の View、QValuePanel など）で押した ↑↓ は、表の行を動かす。←→ は何もしない。表で押した矢印は今と同じく表が使う。矢印を自分で使う窓（ログ欄、Trace の操作バーのスライダーなど）は、今と同じくその窓が使う（今もこれらのキーは Frame へ転送されず、強制行動にならない）。Live モードの間の矢印は、今と同じく強制行動になる。
- G2: 強制した step を、Trace の上で Agent の step と見分けられる。
- G3: 強制行動が起きたことに、Trace を開いていなくても、その場で気付ける。Live モードの間の矢印の押し間違いにも、これで気付く。
- G4: Evaluation Trace の表を、列見出しのクリックで並べ替えて読める。クリックのたびに降順と昇順が入れ替わる。既定の並びは step の降順で、今と変えない。
- G5: Evaluation Trace の任意の行をダブルクリックすると、その step の通知の内容を、できるだけ多くログへ出せる。INFO の 1 行と VERBOSE の 1 件（複数行でよい）に分け、Tensor の値のような重い内容は VERBOSE にだけ出す。step trace を Run の記録に残せるようになるまでの、つなぎの機能とする。

**非ゴール**

- NG1: Env ごとのキーの割り当てと、操作マッピングの設定化（↓ が qbert の UP になる件の根本の直し）。装置層とマッピング層の設計という別の関心で、[999_input_mapping_joystick](999_input_mapping_joystick_10prd.md) が扱う。本当の制約ではなく、関心を分けるために外す。
- NG2: 入力先 train の強制行動（[999_input_target_train_forced_action](999_input_target_train_forced_action_10prd.md)）。別の PRD の関心で、制約ではない。
- NG3: QValuePanel の Q の表示と、強制行動が当たる状態の時点のずれ（[999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md)）。別の PRD の関心で、制約ではない。
- NG4: 強制した step での Agent の内部状態（スキップの継続、判断の旗）の扱いを変えること。G2 の列で強制と分かれば、判断の旗とスキップ長は「Agent の内部の値」として読める。強制行動を `MakeAction` の入力にする観測パイプラインの PRD（[999_observation_pipeline](999_observation_pipeline_10prd.md)）と、入力先 train の PRD の §9 の関心である。本当の制約ではなく（クリーンブレークで移せる）、強制行動の入れ方を決める PRD と一緒に変えるほうが一度で済むので外す。
- NG5: Trace モードでないとき（Live モードの間と、Evaluation Trace の pane を閉じているとき）の矢印キーと、モードによらないテンキー・QValuePanel のクリック・`Ctrl`・`Space` の、行動の割り当てと step の進み方（1 回の押下かクリックで 1 step）。Live モードのまま矢印で操縦する使い方（LunarLander の手動操作など）を残すためで、本当の制約ではない（2026-10-29 のユーザーの裁定）。これらの入力で強制した step にも、D5 の通知と D3・D4 の `forced` 列は出る（G2・G3 のため）。
- NG6: 081 の D6（手動操作のログ行）。081 の範囲で、制約ではない。
- NG7: 複数の列を組み合わせた並べ替え、行の絞り込み、並べ替えの状態の設定への保存。G4 の求めに無く、制約ではない（あとから足せる）。並べ替えの状態は実行時 UI 操作として扱う（D6）。
- NG8: step trace やダンプを、Run の記録やファイルへ保存すること。[999_eval_session_step_trace](999_eval_session_step_trace_10prd.md) と [081](081_evalpanel_episode_trace_10prd.md) の関心で、制約ではない。G5 はそれまでのつなぎである。
- NG9: Actor が内部で積み重ねた入力（frame stack）をダンプに出すこと。通知に無く、Actor の内部を読む口が要る。本当の制約ではない（口は足せる）が、つなぎの機能には要らない。

## 3. 成功の測り方

- G1: EvalPanel を一時停止し、Trace の表で行を 1 つ選んで Trace モードに入る。フォーカスを表以外の anet の Panel（Env の View、Train の View）へ移してから ↓ を 20 回押す。EvalPanel の step が進まず、Trace に新しい行が出ず、表の選択だけが 20 行動く。←→ では何も変わらない（前提と手順は §7 の受入 4）。
- G2: 強制した step の行の `forced` 列が 1、Agent が選んだ step の行が 0 になる（§7 の受入 1・2・4）。
- G3: 強制した step のたびに、ステータス欄に行動の表示名が出る。Live モードの間に矢印で強制した場合も出る（§7 の受入 4）。
- G4: 列見出しのクリックで表が数値の大小で並び、もう一度のクリックで向きが変わり、再生・コマ送りが表の並びに沿って進む（§7 の受入 7・8・10）。
- G5: 行のダブルクリックで、ログに `Trace dump` で始まる INFO の 1 行と VERBOSE の 1 件が出て、要素が 2 個以上の Tensor の値は VERBOSE にだけ並ぶ（§7 の受入 9・10）。

## 4. 確定事項

| # | 決定 | 理由 |
|---|---|---|
| D1 | Trace モードの間、`RunnerFrame` は anet の Panel から転送された矢印キーを強制行動にしない。↑↓ は表の行の移動に回し、動きは表にフォーカスがあるときの ↑↓ と同じにする（表の並びで 1 行、端で止まる。表の並びは D7）。Trace モードかどうかは、表で行を選んでいるか（再生の部品の選択）で決める。pane を閉じると Live モードへ戻るので、閉じている間は Trace モードにならない。ほかの pane の最大化で隠れている間は、Trace モードのまま続く（[EvalTracePanel.cpp:212-221](../../apps/runner/src/EvalTracePanel.cpp) の `SetActive`、[gui.cpp:190-195](../../core/anet-core/src/gui.cpp) の `IsPaneOpen`）。Live モードの間の矢印、テンキー、QValuePanel のクリック、`Ctrl`、`Space` は変えない（NG5）。転送されないキー（表、ログ欄、スライダーなど、その窓が自分で使う矢印）は今のままにする | Trace モードで調べている間に、表以外をクリックしたあとで気付かないまま強制行動になる経路（§1.3）を断つ。条件を「pane が開いている間」にすると Live モードの事故も防げるが、ユーザーは Trace モードの間だけにした（2026-10-29 の裁定。別モデルの反対を受け、代償を正して聞き直しても維持した）。理由は、明示的に Trace モードに入る操作をさせて、今のモードを意識させる考え方にある（操作バーの赤枠〔094 の D11〕も同じ考えで付けた）。Live モードのまま矢印で操縦する使い方も残る。失うもの: Live モードで矢印を押すと、強制行動のたびに Live モードへ戻るので（§1.1）、行をクリックして Trace モードに入るまで、押すたびに強制行動になる。D5 の通知で気付ける。強制行動になるのは転送されたキーだけなので（§1.1）、扱う範囲もそこに絞れば、矢印を自分で使う窓のキーを奪わない。Trace モードの判定を再生の部品の選択に置くのは、Live モードへ戻るきっかけ（094 の D11）がすべてそこへ集まっているためである |
| D2 | Trace モードの間、転送された ←→ は何もしない。表にフォーカスがあるときの ←→ は、今のまま表が使う（横の移動） | 行の移動は ↑↓ で足りる。←→ をコマ送り・コマ戻しにすると、↑↓ と同じ行へ動く入口が 2 つになる。強制行動のままにすると G1 に反する。表の ←→ は横に長い表のスクロールに使うので奪わない |
| D3 | EvalRunner が答える `$runner` の鍵 `forced` を足す。EvalRunner は、最後の step で行動を指定された（強制行動に差し替えた）かを lane ごとに覚え、lane を指定した問い合わせに 1、Actor の選んだ行動のままなら 0 を返す。lane を省いた問い合わせには値を返さない。lane が範囲外なら fail-fast にする（`$action_info action` と同じ扱い）。EvalRunner 以外の Runner はこの鍵を持たない。`WithAction` と行動の情報の型は変えない | 強制したかどうかは遷移の時点の事実である。`$runner` は通知の中で同期に読まれ、その遷移の値を答える（§1.4。[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) が列に許す target）。強制するのは Runner なので（§1.1）、Runner が答えると意味が直接で、変えるのも EvalRunner の 1 か所で済む。行動の情報（`$action_info`）で答える案は、基底・DQN 系・MuZero 系の `WithAction`・`To` とテストの型を変え、「`WithAction` の結果＝強制」という意味を公開の型に付けるので採らない（別モデルの指摘を受けて、2026-10-29 にユーザーが選んだ）。行動の情報だけを見る部品（ReplayBuffer など）には伝わらないが、今は使う部品が無く、train 側の方針も「ReplayBuffer は強制かどうかを知らない」である（[999_input_target_train_forced_action](999_input_target_train_forced_action_10prd.md) §3.2）。将来 train 側の強制行動を `MakeAction` の入力にしても、Runner は強制したかを知っているので、この鍵の意味は変わらない（§8）。GUI に Agent 固有の判定を置かない（同 ADR）。鍵の名前は D11 の用語に合わせる |
| D4 | 既定の列に `app.eval_trace_panel.columns.[260] ?= $runner forced label:forced` を足す（[common.txt](../../apps/runner/config/common.txt) の既定列）。番号は「行動と結果」（2xx）の、decision の 240 と skip の 250 の次である | 気付くには、既定で見えている必要がある。§1.3 は既定の列だけで読んでいた。Evaluation Trace の行を作るのはいつも EvalRunner なので、Env・Agent によらず読める。そこで、step・ep_step（`$runner`）と同じく common.txt に置く。Agent・Env に固有の列（decision など）は env の設定ファイルに置かれている（[Atari.txt:287-288](../../apps/runner/config/Atari.txt)）。番号は [020 §6.5](../design/020_user_guide_run.jp.md) の番号の決まり（十の位で分類の中の順、一の位は 0）による。D3 に依存する |
| D5 | 強制した step のたびに、`EvalPanel` がステータス欄 0 に `Forced action: <行動の表示名>` を出す（例: `Forced action: 2:UP`）。表示名は Trace の action 列の既定と同じ「番号:名前」の形にする。強制行動の入口は離散の行動番号だけなので（§1.1）、連続の行動の書き方は持たない。対象はテンキー、Live モードの間の矢印、QValuePanel のクリックである。行動を指定しない `Ctrl` とツールバーの Step では出さない。ログには出さない | Trace を開いていなくても、強制したことにその場で気付ける。D1 で Live モードの間の矢印が強制行動のまま残るので、その押し間違いに気付く手段にもなる。範囲外の行動番号の案内（[EvalPanel.cpp:187-191](../../apps/runner/src/EvalPanel.cpp)）と同じ欄を使い、新しい表示部品を足さない。ログ行は NG6 の範囲である。D1 に依存する |
| D6 | Evaluation Trace の表は、列見出しのクリックでその列の値の並びに並べ替える。並べていない列を初めてクリックすると降順、並べている列をもう一度クリックすると向きを入れ替える（降順と昇順の 2 状態。「並べ替えなし」の状態は持たない）。既定の並びは step の降順で、既定へは step 列のクリックで戻る。値は数値の大小で比べ、表示の文字列の辞書順にはしない。並べ替えの状態は実行時 UI 操作で、設定や config dump に残さない。細部は §6 の 3〜13 | ユーザーの求め（G4）。数値の大小で比べるのは、列の値がすべて数値で（§1.4）、表示の書式は見せ方だけだからである（2026-10-29 のユーザーの指摘）。実行時 UI 操作として扱うのは、再生の速度（094 の D12）と同じく Run の比較・再現の根拠にならないためである（[CONTEXT.md](../../CONTEXT.md) の「実行時 UI 操作」） |
| D7 | 並べ替えたとき、再生・コマ送り・コマ戻し・スライダー・↑↓ は、表の並びに沿って進む。再生は表の下の行から上の行へ送り、いちばん上の行で止まる。コマ送りは 1 行上、コマ戻しは 1 行下、スライダーの範囲と位置も表の並びである。094 の D12 の「いちばん古い行」「いちばん新しい行」は、それぞれ表の下端・上端と読み替える（Live モードからの再生は下端から、Live モードの起点と止まる端は上端、上端で一時停止しているときの再生は下端から）。既定の並び（step の降順）では、下端がいちばん古い行、上端がいちばん新しい行なので、今と同じである。再生の部品 `TracePlayer` には、表の下端から上端の順の行の列を渡す | ユーザーの裁定（2026-10-29。別モデルの「時系列に戻すほうが単純」という反対を受けて聞き直しても維持した）。並べ替えるのは何かの軸で分析しようとするときなので、時系列にこだわる理由がない。再生の部品は、受け取った列の前後だけで再生・端・Live の起点を決めているので（§1.4）、渡す列の順を変えるだけで規則と通知（`Trace: end`・`Trace: start`）がそのまま表の上端・下端の意味になる。D6 に依存する |
| D8 | 並べ替えている間（Live モードの間）に来た新しい行と、あとから値が決まった行（近似優先度など）は、並びの位置へ入れ直す。表に残す行は、並べ替えによらず step の新しい順に `max_rows` 件で、上限を超えたら並びによらず step のいちばん古い行を消す | D7 で再生が表の並びに沿うので、表が並んでいないと再生の順も意味を失う（ユーザーの選択、2026-10-29）。Trace モードの間は EvalPanel が止まっているので（094 の D11）、新しい行は来ず、並びも動かない。保持の範囲を並べ替えで変えないのは、表の上限と百分位の窓（094 の D10）が step の新しさで決まっているためである。溢れて消えるのが表示の順ではなく step のいちばん古い行であることは、ユーザーが確かめた（2026-10-29）。D7 に依存する |
| D9 | 行をダブルクリックすると、その行の通知の内容をログへ出す（ダンプ）。ダンプのために、表の各行に通知の部品を参照のまま持たせる: 行動の情報、Step の結果、遷移前の状態（通知の間しか生きないので、状態の Tensor の参照を写して持つ）、counts、Runner・Env・Agent の名前。文字列はダブルクリックしたときだけ作る。持つものは表の行と同じ寿命にし、上限で行が消えるとき、pane を閉じたときに一緒に放す。ダブルクリックは行を選ぶ操作も兼ね、Run の状態を変えない | 「できるだけ多く」（G5）に合い、Env・Agent ごとの処理が要らない（ユーザーの選択、2026-10-29）。Step の結果の Tensor は次の step で書き換わらないので（§1.4）、参照のまま持ってよい。増えるメモリは Atari・1,000 行で約 108MB で（§5.6）、今の追従 pane のスナップショット（1,000 行で約 130〜300MB。§1.4）と合わせて約 240〜410MB になる。同じ画面を View のスナップショットと Step の結果の `rgb_frame` の 2 か所に持つことになるが、表の上限で頭打ちになり、pane を閉じれば放す。行を作る時点で文字列まで作る案は、Atari の観測 1 つを数値の文字列にするだけで数十 KB・数十 ms かかり、step ごとには払えないので採らない（[AGENTS.md](../../AGENTS.md) の基本方針の性能の項） |
| D10 | ダンプは、`LOG::info()` の 1 件と `LOG::verbose()` の 1 件に分ける。INFO は改行を含めない 1 行にし、Tensor の値のうち要素が 2 個以上のものを出さず、形・型・device だけを書く。VERBOSE は、既存の文字列化（§1.4）をそのまま使う複数行の 1 件にし、既存の文字列化が書かない行動の情報の info と Replay初期優先度ヒントを、同じ書き方で足す。ステータス欄には出さない。中身は §5.5、細部は §6 の 15〜23 | 重い内容は VERBOSE にだけ出す（G5）。VERBOSE を見るのは AI か、コピーして整形する人で、grep で取り出す使い方は想定しない。エクスポート機能（NG8）ができるまでのつなぎとしては、既存の文字列化で足りる（2026-10-29 のユーザーの選択。別モデルの指摘を受けて、改行を含めない 1 行から改めた。Atari の VERBOSE は約 400〜500KB で、この大きさは行の分け方によらない）。INFO は 1 行のままで、grep で拾える。要素が 1 個の値はスカラーと同じ重さなので INFO に出す。ログの関数は既存のもの（[log.hpp:214-215](../../core/anet-core/include/anet/log.hpp)）を使う。D9 に依存する |
| D11 | 用語「強制行動」（forced action）を [CONTEXT.md](../../CONTEXT.md) に足す。定義は「人の操作（キー・クリックなど）によって、Actor が選んだ行動を Actor の外から置き換えた実行行動」で、`_Avoid_` に「手動 Action」「manual action」「上書き」「強制Action（表記ゆれ）」を置く。[999_input_mapping_joystick](999_input_mapping_joystick_10prd.md) §4 の候補のうち「強制行動」は、本書で確定する（999 の §4 の行に、確定した旨の注記を足した） | 用語集に無い概念に、鍵 `forced` という名前を付けるので、先に用語を決める（[AGENTS.md](../../AGENTS.md) §用語(CONTEXT.md)を引く条件）。既存の語では言い換えられない。「実行時 UI 操作」は pause・1 step の実行・FPS の変更など GUI の操作全般の区分で、Actor の選んだ行動を外から置き換えたことを指さない。「鏡写しインスタンス」も手動で操作される実行の単位を言うだけで、行動の出どころを言わない。設計文書が使う「強制Action」（[160](../design/160_applications_and_tools.jp.md) §3.1・§5.1、[100](../design/100_runtime_and_configuration.jp.md) §6.4・§8）と、999 の候補「強制行動」に合わせ、表記は「強制行動」にそろえる。160 が同じ概念を「手動Action」と呼んでいる箇所（§2.1 の `EvalPanel` の行、§3.1 の「手動の離散Action」）も、`_Avoid_` に入れたので「強制行動」へそろえる。999 の候補の定義は「操作入力によって」としていて QValuePanel のクリックを含まないが、`forced` はクリックで強制した step も 1 にするので、定義を「人の操作（キー・クリックなど）」に広げた |
| D12 | 094 の冒頭に、D11 の上下キーの文を本書の D1 が、D12 の再生の向きを本書の D7 が置き換えた旨の注記を 1 行足す（実装のときに、§5.7 の移行の対象として行う） | 094 は `docs/memo` 直下の現用の PRD なので、読む人が上下キーと再生の向きの今の正本にたどり着けるようにする。D1・D7 に依存する |

新しい ADR は起こさない。D1〜D12 はどれも後から戻せる UI の挙動、読み出しの鍵 1 つ、ログの書式、用語であり、ADR の 3 条件（戻しにくい、文脈なしでは意外、本当のトレードオフ）をすべては満たさない。

## 5. 実装契約

### 5.1 キーの振る舞い

表は anet の Panel（Env の View、QValuePanel など）から転送されたキーの扱いである。「今のまま」は、行動の割り当てと step の進み方（1 回の押下かクリックで 1 step）が今と同じことを指す。強制した step には、どの行でも D5 の通知と `forced` 列（D3・D4）が出る。

| 状態 | ↑↓ | ←→ | テンキー 0〜9 | QValuePanel のクリック | `Ctrl` / `Space` |
|---|---|---|---|---|---|
| Evaluation Trace の pane を閉じている | 今のまま（強制行動 0・2） | 今のまま（強制行動 1・3） | 今のまま（強制行動） | 今のまま | 今のまま |
| pane を開いていて Live モード | 今のまま（強制行動 0・2） | 今のまま（強制行動 1・3） | 今のまま（強制行動） | 今のまま | 今のまま |
| Trace モード（最大化で隠れている間を含む） | 表の行を表の並びで 1 行動かす。端で止まる。強制行動にしない。EvalPanel は進まない | 何もしない | 今のまま（強制行動。Live モードへ戻る） | 今のまま（Live モードへ戻る） | 今のまま |

- 転送されないキーは、モードによらず今のままである。表にフォーカスがあるときの ↑↓ は、今と同じく表が受けて行を動かす（[EvalTracePanel.cpp:180-192](../../apps/runner/src/EvalTracePanel.cpp)）。表の ←→ も今のまま表が使う。ログ欄やスライダーなど、矢印を自分で使う窓では、その窓が使う。
- Trace モードでは必ず行を選んでいるので、転送された ↑↓ は選んでいる行を起点に数える。
- 転送された ↑↓ で行を動かしても、通知欄には何も出さない（§6 の 12）。

### 5.2 `forced` の値

| EvalRunner の最後の step | `$runner forced`（lane 指定） | lane を省いた問い合わせ |
|---|---|---|
| 行動を指定しない step（`DoStep()`、自動の進行、`Ctrl`・ツールバーの Step、評価 session） | 0 | 値を返さない |
| 行動を指定した step（`DoStep(action)`。テンキー・矢印・QValuePanel のクリック） | 1 | 値を返さない |
| まだ 1 step も進めていない | 0 | 値を返さない |
| lane が範囲外 | fail-fast（`$action_info action` と同じ扱い） | — |

- EvalRunner 以外の Runner は `forced` を持たない。問い合わせると、今の未知の鍵の扱い（列では最初の読み出しで fail-fast、scalar では値が見つからない WARN）になる。
- 値は「最後の step」のものである。Evaluation Trace の列は通知の中で同期に読むので、その行の step の値になる（§1.4）。

### 5.3 並べ替えの契約

| 操作・場面 | 結果 |
|---|---|
| Evaluation Trace を開いた直後・開き直したとき | 既定の並び（step の降順）。step 列の見出しに降順の印。step 列を外した構成でも並びは同じで、印は出さない |
| 並べていない列の見出しをクリック | その列の降順に並べ、見出しに降順の印を出す。ほかの列の印は消す |
| 並べている列の見出しをもう一度クリック | 向きを入れ替える（降順⇔昇順） |
| 比べ方 | 数値の大小。`display:pct` の列は百分位の数値、行動の列（`2:UP` と表示する列）は行動番号の数値。空欄（NaN、まだ決まっていない値）は向きによらず下端。同じ値の行どうしは step の降順 |
| Live モードの間に新しい行が来た | 並びの位置へ入れる（D8） |
| あとから値が決まった（近似優先度など） | その行を並びの位置へ移す（D8） |
| 上限 `max_rows` を超えた | 並びによらず、step のいちばん古い行を消す（D8） |
| Trace モードで並べ替えを変えた | 選んでいる行を変えず、その行が見える位置までスクロールする。再生中なら、その行から新しい並びで再生を続ける |
| Live モードで並べ替えを変えた | Live モードのまま（並べ替えは Trace モードに入るきっかけにしない） |
| Live モードの間の表示の位置 | 並べ替えていても、今と同じく表の上端を見せる（[EvalTracePanel.cpp:286](../../apps/runner/src/EvalTracePanel.cpp) は、Live モードで行を足すと先頭へスクロールする） |
| 見出しの境目のドラッグ | 今のまま列幅を変える。並べ替えにはならない |
| 見出しのダブルクリック | 並べ替えのクリック 2 回として扱う。ダンプにはならない |

### 5.4 並べ替えたときの再生（094 の D12 の読み替え）

| 094 の D12 の規則 | 本書での読み替え |
|---|---|
| 再生は、Live のときはいちばん古い行から、Trace で一時停止しているときは選んでいる行から、新しい側へ 1 行ずつ | Live のときは表の下端から、Trace で一時停止しているときは選んでいる行から、表の上へ 1 行ずつ |
| いちばん新しい行に追いついたら一時停止する（`Trace: end`） | 表の上端に着いたら一時停止する（`Trace: end`） |
| いちばん新しい行で一時停止しているときの再生は、いちばん古い行から | 表の上端で一時停止しているときの再生は、表の下端から |
| コマ送り: 1 行新しい側へ。コマ戻し: 1 行古い側へ。端で止まり `Trace: end`・`Trace: start` | コマ送り: 1 行上へ。コマ戻し: 1 行下へ。上端で `Trace: end`、下端で `Trace: start` |
| Live でのコマ送り・コマ戻しは、いちばん新しい行を今の位置として扱う | 表の上端を今の位置として扱う |
| スライダー: 範囲は持っている行（いちばん古い行からいちばん新しい行まで） | 範囲は表の下端から上端まで、位置は選んでいる行の表の中の位置 |
| 表の上限で選んでいる行が消えたら、残っている中でいちばん古い行へ移る（094 の D11） | 表の下端の行へ移る |

既定の並びでは、左右の列は同じ挙動になる。最後の行（選んでいる行が上限で消えたとき）は、今は経路が無い（§1.4）が、094 と同じく規則としては持つ。

### 5.5 ダンプの中身

INFO の行と、VERBOSE の 1 件目の行は、どちらも `Trace dump step=<step> lane=<lane>:` で始める。`<step>` は step trace の step（counts の exp_step と同じ座標）である。

| 件 | 中身 |
|---|---|
| INFO（`LOG::info()`、改行を含めない 1 行） | 「名前=値」を空白で区切り、次の順に並べる。(1) 表の全列: `<見出し>=<値>`（表と同じ書式。まだ決まっていない値は `pending`、確定した非有限値は表と同じく空）。(2) 通知の項目: `runner=`・`env=`・`agent=`（名前）、`counts.<欄>=`（counts のすべての欄）、`action_info.action=`（番号:名前）、`runner.forced=`、`step_result.reward=`、`step_result.done=`、`step_result.truncated=`、`action_info.info.<鍵>=`・`action_info.aux.<鍵>=`・`step_result.aux.<鍵>=`（要素が 1 個なら値、2 個以上なら `[形] 型 device` だけ。未定義なら `undefined`、要素が 0 個なら `[形] 型 device` だけ）。(2) は、(1) に同じ値の列があっても、列の構成によらず必ず出す |
| VERBOSE（`LOG::verbose()`、複数行の 1 件） | 書き出しのあとに、遷移前の状態（`BatchState::ToString`）、行動の情報（`BatchActionInfo::ToString`）、行動の情報の info（`TensorDict` の文字列化）と Replay初期優先度ヒント（あれば、中身の Tensor を `anet::ToString` で書く。中身の意味は Agent が持ち、ここでは解釈しない）、Step の結果（`BatchStepResult::ToString`）を、見出しを付けて順に並べる。既存の文字列化の書き方（入れ子の括弧と改行、未定義は `undefined`）は変えない |

- INFO の (2) に取り出し元の名前を前に付けるのは、表の列の見出しとぶつけないためである。(2) を列の構成によらず出すので、列を外した構成でも同じ項目が読める。名前の付け方は、VERBOSE の見出し（`action_info`・`step_result`）と同じにする。
- 数値の精度は、既存の `anet::ToString` と同じ有効数字 4 桁にする（§6 の 18）。
- GPU の Tensor は、行が持つ間は参照のままにし、ダブルクリックしたときに CPU へ写してから書く（§6 の 19）。
- 追従 pane のスナップショット（View の画像、QValuePanel の値）は入れない（§6 の 20）。同じ値が、Step の結果の `rgb_frame` と行動の情報の Q 値にある。
- 同じ行を何度ダブルクリックしてもよく、1 回につき INFO 1 件と VERBOSE 1 件を出す。
- VERBOSE は今の level の扱いのまま出す。ログ欄には、Log Level が verbose のときだけ出る。Atari の VERBOSE は約 400〜500KB になる。

例（qbert。値は形だけを示す）:

```text
Trace dump step=184503 lane=0: step=184503 ep_step=2932 action=2:UP reward=0 decision=1 skip=1 forced=1 done=0 trunc=0 Q_sa=14.17 V_s=15.05 prio=pending prio_pct=pending runner=EvalPanel env=AtariEnv agent=DefaultDQNAgent counts.exp_step=184503 ... action_info.action=2:UP runner.forced=1 step_result.reward=0 step_result.done=0 step_result.truncated=0 action_info.info.decision=1 ... action_info.aux.q_values=[1,6] Float cuda:0 ... step_result.aux.rgb_frame=[3,210,160] Byte cpu
Trace dump step=184503 lane=0: verbose
state=BatchState{obs=..., done=..., truncated=..., episode_start=...}
action_info=BatchActionInfo{action=(cpu Long) [1] -> ..., aux={ q_values=(cpu Float) [1, 6] -> ... }}
action_info.info=...
action_info.replay_initial_priority_hint=...
step_result=BatchStepResult{ reward=..., next_state=..., continue_state=..., auxs={ rgb_frame=... } }
```

### 5.6 ホットパスの見積もり

- 並べ替え: 既定の並び（step の降順）では、今と同じく先頭へ足すだけで、追加の手間は 0 である。並べ替えている間は、1 行ごとに並びの中の位置を探して入れ（最大 `max_rows`＝既定 1,000 行の中で位置を探して入れる）、値が決まったら移す。今も 1 行ごとに再生の部品へ全行の id を渡し直していて（§1.4）、手間の桁は同じである（マイクロ秒の桁）。Atari の 1 step は約 69 ms（14.5 step/s）である。094 の受入 8（表示の速さ）の見積もり（1 step あたり上限 1.40 ms）に対して、加わる分は無視できる。
- ダンプ: step ごとの追加は、行に通知の部品の参照を数個写すことだけで、ナノ秒の桁である。文字列はダブルクリックしたときだけ作る。メモリは、Atari・1,000 行で約 108MB 増える（1 行あたり `rgb_frame` 100,800 バイト＋観測 7,056 バイト×約 1.5。遷移前の状態の観測は前の行の継続の状態と同じ Tensor を指すことが多い。行動の情報の Q 値などは小さい）。今の追従 pane のスナップショット（約 130〜300MB）と合わせて約 240〜410MB になる。
- `forced`: EvalRunner が step ごとに lane の数だけ値を書き換えるだけで、ナノ秒の桁である。
- ダブルクリックしたときのログ欄への追記（Log Level が verbose のときの約 400〜500KB）の重さは、試さないと分からない（④）。実装のときに Atari の行で、ダブルクリックしてからログ欄に出るまでの時間を測り、実装計画に記録する（§7 の受入 10）。重いときは Log Level を info のまま使えば、ログ欄には INFO だけが出る。
- どちらも、学習側の値と順番には触れない（§7 の受入 6）。

### 5.7 移行の対象

- コード: [RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp)（`OnKey`）、[EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp)・[EvalTracePanel.hpp](../../apps/runner/src/EvalTracePanel.hpp)（Trace モードかを答える口、行の移動を外から呼べる口、見出しのクリックでの並べ替え、行ごとの通知の部品の保持、ダブルクリックでのダンプ）、[EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp)（D5）、[gui.hpp](../../core/anet-core/include/anet/gui.hpp)・[gui.cpp](../../core/anet-core/src/gui.cpp)（`TracePlayer` が受け取る行の列の意味、並べ替えの順とダンプの文字列を GUI に依らず決める部品）、[trainer.hpp](../../core/anet-core/include/anet/trainer.hpp)・[trainer.cpp](../../core/anet-core/src/trainer.cpp)（EvalRunner の `forced`）。
- 設定: [common.txt](../../apps/runner/config/common.txt) の既定列に 260 を足す。
- テスト: EvalRunner の `forced` のテスト（§7 の受入 1・2）、common.txt を読んで既定列を比べる新しいテスト（受入 3。[config_test.cpp:1115-1150](../../core/anet-core/src/config_test.cpp) が common.txt を読む形を使う）、[gui_test.cpp](../../core/anet-core/src/gui_test.cpp) の `TracePlayer` のテスト（受け取る列の順を表の並びにした場合）、並べ替えの順とダンプの文字列の単体テスト（§7 の受入 7〜9）。2026-10-29 に確かめた範囲では、common.txt の既定列を期待値と比べる現用のテストは無い（[EvalTracePanel_test.cpp](../../apps/runner/src/EvalTracePanel_test.cpp) と [dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) は自前の列を宣言する）。
- 直さないもの: [testdata/prd094/compare_configs.py](../../core/anet-core/testdata/prd094/compare_configs.py) と `testdata/prd103/resolved-before/` は、それぞれの PRD の検証の記録で、当時の既定列を持つ。過去の記録なので書き換えない。
- 文書: [140](../design/140_observability.jp.md) の §6.1 の target 表の `$runner` の行（EvalRunner の鍵 `forced` を足す）。[020 §6.2](../design/020_user_guide_run.jp.md) の矢印キーの行（Trace モードの間の扱い）と、ステータス欄の通知。[020 §6.5](../design/020_user_guide_run.jp.md) の矢印キー・既定列（forced を足す）・並べ替え・ダブルクリックのダンプ・再生の向き。[160](../design/160_applications_and_tools.jp.md) の §3.1 の強制行動の段落と、§3.4 の既定列（forced を足す。今の「step、ep_step、action、reward、Q_sa、V_s、done、trunc…」の順の文は common.txt の番号順〔done・trunc が Q_sa の前〕と食い違っているので、あわせて直す）・並べ替え・再生の向き・ダンプ。[100](../design/100_runtime_and_configuration.jp.md) の §6.4・§8 と [160](../design/160_applications_and_tools.jp.md) の §3.1・§5.1 の「強制Action」と、[160](../design/160_applications_and_tools.jp.md) の §2.1 のコンポーネント表の `EvalPanel` の行の「手動Action」・§3.1 の「手動の離散Action」を、用語の「強制行動」にそろえる（D11）。[094](094_evalpanel_step_history_10prd.md) の冒頭の注記（D12）。[999_input_mapping_joystick](999_input_mapping_joystick_10prd.md) §4 の注記は、本書の起草のときに済ませた（D11）。英訳（`.en.md`）は翻訳の工程で追随する。
- 互換の扱い: 旧い挙動を選ぶ設定や別名は作らない。

### 5.8 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RunnerFrame` | 変更 | Trace モードの間、転送された矢印を強制行動にせず、↑↓ を表の行の移動へ回し、←→ を捨てる | 110 RunnerApp |
| `EvalTracePanel` | 変更 | Trace モードかを答え、表の ↑↓ と同じ行の移動をほかの窓から呼べるようにする。見出しのクリックで並べ替え、行ごとに通知の部品を持ち、行のダブルクリックでダンプを出す | 110 RunnerApp |
| `EvalPanel` | 変更 | 強制した step のたびに、ステータス欄へ行動の表示名を出す | 110 RunnerApp |
| `TraceRowSorter` | 新規 | 列の値・向き・step から、表の並び（上端から下端）を GUI に依らず決める。行の追加と値の確定で入れ直し、上限で消す行を step で選ぶ | 319 GUI 共通基盤 |
| `TraceDumpFormatter` | 新規 | 行が持つ通知の部品と列の値から、ダンプの INFO の 1 行を作り、VERBOSE を既存の文字列化をつないで作る。GUI に依らない | 319 GUI 共通基盤 |
| `TracePlayer` | 変更 | 受け取る行の列を「再生の順」（表の下端から上端）として扱う。規則は列の前後で決まるので、処理は変えず、受け取る列の名前と説明を改める | 319 GUI 共通基盤 |
| `EvalRunner` | 変更 | 最後の step で行動を指定されたかを lane ごとに覚え、`$runner forced` として答える | 318 強化学習 共通基盤 |

## 6. 細部

並べ替え・通知（2026-10-29 に推奨の既定で埋めた。7 はユーザーの指摘で直した）

1. ステータス欄の文言: `Forced action: <行動の表示名>`（例 `Forced action: 2:UP`）。対象と出さない場合は D5（理由: ステータス欄の既存の案内が英語である）。
2. `forced` 列の表示: 数値の value（0 / 1）、見出し `forced`（理由: decision 列と同じ形で、列の文法を足さない）。
3. 並べていない列を初めてクリックしたときの向き: 降順（理由: 既定の step の降順と同じ向き。prio・Q など、Trace では大きい値を上に見たい列が多い）。
4. 並べている列の再クリック: 降順と昇順を入れ替える。「並べ替えなし」は持たず、既定へは step 列のクリックで戻る（理由: 求めは「降順・昇順をトグル」の 2 状態）。
5. 空欄（NaN、まだ決まっていない値）の置き場: 向きによらず下端（理由: 一般的な表計算と同じ）。
6. 同じ値の行どうしの順: step の降順。D7 の再生では、同じ値の中は古い→新しいの順に送られる（理由: 既定の並びと同じ向き）。
7. 比べる値: 数値の大小。表示の文字列の辞書順にしない。`display:pct` の列は百分位、行動の列は行動番号の数値（理由: 列の値はすべて数値で、書式は見せ方だけ）。
8. 見出しの印: 並べている列の見出しに降順・昇順の印。既定では step 列に降順の印。step 列が無い構成では印なし（理由: 一般的な表と同じ）。
9. 開き直したとき: 既定の並びへ戻す（理由: 開き直したら表と列幅を空から始める 094 の D10 とそろえる）。
10. 並べ替えを変えたときの選択と、Live モードの間の表示の位置: §5.3 のとおり（理由: 調べている行を見失わない。並べ替えは見え方の操作で、モードを変えない。Live モードの表示の位置は今のコードと同じ）。
11. 選んでいる行が上限で消えたとき: 表の下端の行へ移る（理由: D7 の上端・下端の読み替えにそろえる。`TracePlayer` は消えたら列の先頭へ移るので、渡す列の先頭が下端になる）。
12. 転送された ↑↓ で行を動かしたときの通知: 出さない（理由: 表の ↑↓ と同じ）。
13. 見出しのクリックと列幅の変更: 両方できる。境目のドラッグは今のまま列幅を変える（理由: 手で変えた列幅を保つ今の機能を壊さない）。
14. 094 の扱い: D12 のとおり冒頭に注記を 1 行足す（理由: 094 は現用の PRD）。

ダンプ（2026-10-29 に推奨の既定で埋めた。15・17・22 は D10 の改めに合わせて直した）

15. 書き出し: INFO と、VERBOSE の 1 行目を `Trace dump step=<step> lane=<lane>:` で始める（理由: 同じダンプの 2 件を step で結べる）。
16. INFO の中身: §5.5 の表のとおり（理由: Tensor の値は VERBOSE だけ、という求め。要素が 1 個の値はスカラーと同じ重さ）。
17. VERBOSE の中身: §5.5 の表のとおり。既存の文字列化をそのまま使い、書かない info と Replay初期優先度ヒントだけを足す（理由: D10。新しい書式を作らない）。
18. 数値の精度: 既存の `anet::ToString` と同じ有効数字 4 桁（理由: 既存の書式にそろえる。§1.3 の Q_sa 14.17 と V_s 15.05 の差は読める）。
19. GPU の Tensor: ダブルクリックしたときに CPU へ写して書く（理由: D9 のとおり、step ごとには転送しない）。
20. 追従 pane のスナップショット: 入れない（理由: 同じ値が通知の部品にある）。
21. 見出しのダブルクリック: ダンプにしない（理由: 見出しのクリックは並べ替えに使う。§5.3）。
22. 同じ行の繰り返し: 何度でもダンプでき、1 回につき INFO 1 件と VERBOSE 1 件を出す。
23. 表示の先: 今の level の扱いのまま（理由: [LogPanel.cpp:44-53](../../apps/runner/src/LogPanel.cpp) の今の挙動を変えない）。

## 7. 受入基準

1. 単体テスト（EvalRunner）: 1 step も進めていない EvalRunner の `GetScalar("forced", 0)` が 0、`DoStep(action)` のあとが 1、続く `DoStep()` のあとが 0 になる。lane を省くと値を返さない。lane が範囲外なら fail-fast になる。TrainRunner の `GetScalar("forced", 0)` は値を返さない。
2. 単体テスト（step trace）: `$runner forced` を列に持つ step trace を EvalRunner の通知から作ると、`DoStep(action)` の行が 1、`DoStep()` の行が 0 になる（通知の中で読んだ値が、その行の step の値である）。
3. 単体テスト（既定列）: common.txt を読んで解いた設定の `app.eval_trace_panel.columns.[260]` が `$runner forced label:forced` で、ほかの既定列（200〜230、350、360、400、410、500、510）の値が今の common.txt と同じである。
4. 手動の確認（Runner、Atari の online、qbert）:
   - 前提: Evaluation Trace を開き、表に 21 行以上たまってから、Run 制御ツールバーの `Eval` で EvalPanel を一時停止する。表のいちばん上の行をクリックして選び、Trace モードに入る。
   - Env の View を左クリックしてフォーカスを移す（左クリックは学習の一時停止を切り替えるが、この確認では学習側を見ない。右クリックは EvalPanel の再開で Live モードに戻るので使わない）。最初の ↓ の直前の Evaluation View の Game step と、表のいちばん新しい行の step を控え、↓ を 20 回押す。表の選択が上から 21 行目になり、Game step といちばん新しい行の step が控えた値のままである（EvalPanel が進まず、新しい行が出ない）。
   - 同じ状態で ←→ を押しても、Game step と表の行と選択が変わらない。
   - Train の View を左クリックしてフォーカスを移し、↑ を 20 回押すと、選択が上から 1 行目に戻り、Game step は変わらない。QValuePanel の表（行動の行）はクリックしない。クリックはその行動の強制行動になる（[QValuePanel.cpp:896-908](../../apps/runner/src/QValuePanel.cpp)）。
   - Evaluation Trace の停止ボタン（四角。094 の D12）で Live モードへ戻し、Env の View を左クリックしてから ↓ を 1 回押すと、強制行動（qbert では `2:UP`）で 1 step 進み、ステータス欄に `Forced action: 2:UP` が出て、新しい行の `forced` が 1 になる。
   - テンキーの 2 を押すと 1 step 進み、新しい行の `forced` が 1、ステータス欄に `Forced action: 2:UP` が出る。Agent が選んだ行の `forced` は 0 である。
   - Evaluation Trace を閉じて ↓ を押すと、今までどおり強制行動（qbert では `2:UP`）で 1 step 進み、ステータス欄に `Forced action: 2:UP` が出る。
5. 手動の確認（Runner、LunarLander の online）: Evaluation Trace を閉じた状態と、開いて Live モードの状態で、矢印キーの強制行動が今までどおり 0〜3 になる。
6. 学習側の値を変えない: [PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md) の段 0 と同じ短縮条件（4 環境、seed 1、8,704 experience、評価停止、FP32・決定論的 backend）で、本書の前のビルドと後のビルドの metrics checksum が 4 環境とも一致する。比べ方も段 0 と同じ（`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar 点の完全一致）。`forced` は scalar の定義に入れない。
7. 単体テスト（並べ替えの順、`TraceRowSorter`）:
   - step が 9・10・100 の 3 行を step で並べると、降順で 100・10・9、昇順で 9・10・100 になる（辞書順の 9・100・10 にならない）。
   - NaN を含む列では、降順でも昇順でも NaN の行が下端になる。
   - 同じ値の行は、降順でも昇順でも step の降順に並ぶ。
   - 並べていない列の 1 回目は降順、同じ列の 2 回目は昇順、3 回目は降順になる。
   - 並べ替えている間に足した行は並びの位置へ入り、あとから値が決まった行は並びの位置へ移る。
   - `max_rows` を超えたら、並びによらず step のいちばん古い行が消える。
   - `display:pct` の列は百分位の数値、行動の列は行動番号の数値で並ぶ。
8. 単体テスト（再生、`TracePlayer`）: 表の下端から上端の順の列を渡したとき、§5.4 の右の列のとおりに、Live からの再生は下端から始まって上端で `Trace: end` で止まり、コマ送りは 1 行上、コマ戻しは 1 行下、上端で一時停止しているときの再生は下端から始まる。選んでいる行が列から消えると下端の行へ移る（今は経路が無いが、部品の規則として確かめる）。既定の並びの列では、今のテストがそのまま通る。
9. 単体テスト（ダンプ、`TraceDumpFormatter`）: 要素が 2 個以上の Tensor（観測、Q 値）と 1 個の Tensor（報酬）を持つ行の部品から作ったダンプが、次を満たす。
   - INFO は改行を含まず、`Trace dump step=<step> lane=<lane>:` で始まる。
   - INFO に、要素が 2 個以上の Tensor の値が無く、`[形] 型 device` だけがある。要素が 1 個の値はある。まだ決まっていない列は `pending` になる。未定義の Tensor は `名前=undefined`、要素が 0 個の Tensor は `[形] 型 device` だけになる。
   - INFO の通知の項目（`runner.forced`・`step_result.reward` など）は、同じ値の列がある構成でも、列が無い構成でも、同じ名前で 1 回ずつ出る。
   - VERBOSE の 1 行目は `Trace dump step=<step> lane=<lane>:` で始まる。続く行に、遷移前の状態・行動の情報・Step の結果の既存の文字列化と同じ文字列と、行動の情報の info と Replay初期優先度ヒントの値がある。
10. 手動の確認（Runner、Atari の online、qbert）:
    - 表の forced と prio の見出しをクリックすると、表がその列の降順に並び、もう一度のクリックで昇順になり、見出しの印が変わる。並べた状態で再生すると、表の下端から上へ 1 行ずつ選択が進む。pane を閉じて開き直すと、step の降順に戻る。
    - 表の任意の行をダブルクリックすると、Run のテキストログに `Trace dump step=<その行の step>` で始まる行がちょうど 2 行出る（INFO の 1 行と、VERBOSE の 1 件目の行）。VERBOSE の 1 件は、続く行に既存の文字列化を含む。Log Level が info のときログ欄には INFO だけが出て、verbose にすると両方出る。
    - Log Level を verbose にしてダブルクリックしてから、ログ欄に VERBOSE が出るまでの時間を測り、実装計画に記録する（§5.6 の④）。

## 8. 範囲外・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| Env ごとのキーの割り当て（qbert で ↓ を DOWN にする等） | 範囲外 | 操作マッピングの設定化の関心。[999_input_mapping_joystick](999_input_mapping_joystick_10prd.md) のグリルで扱う |
| 強制した行を色で目立たせる | 先送り | まず列と通知で足りるかを見る。再開条件: 列があっても強制した行を見落とす例が出たとき |
| 強制した step で、Agent のスキップの継続を捨てて判断として記録する | 範囲外 | 強制行動を `MakeAction` の入力にするときに決める。再開条件: [999_observation_pipeline](999_observation_pipeline_10prd.md) の単位 A か、[999_input_target_train_forced_action](999_input_target_train_forced_action_10prd.md) の §9 を実装するとき |
| train 側の Runner にも `forced` を答えさせる | 範囲外 | NG2 の関心。再開条件: [999_input_target_train_forced_action](999_input_target_train_forced_action_10prd.md) を実装するとき。そのとき TrainRunner が同じ鍵 `$runner forced` を同じ意味で答える（D3） |
| 強制した step の数や割合を scalar に出す | 範囲外 | 入力先 train の PRD の Q7（可観測性）の関心。本当の制約ではなく（scalar の定義は足せる）、強制行動の可観測性を入力先 train と一緒に決めるほうが一度で済むので外す |
| 手動操作のログ行 | 範囲外 | [081](081_evalpanel_episode_trace_10prd.md) の D6 |
| step trace やダンプを Run の記録・ファイルへ保存する | 範囲外 | NG8。[999_eval_session_step_trace](999_eval_session_step_trace_10prd.md) と [081](081_evalpanel_episode_trace_10prd.md) の関心 |
| Actor が積み重ねた入力をダンプに出す | 範囲外 | NG9。再開条件: 積み重ねた入力を見ないと判断を読めない例が出たとき。Actor の内部を読む口から起こす |

## 9. 未決

なし。暫定の起票で置いた未決は、2026-10-29 の再グリルで次のとおり閉じた。

| # | 論点 | 閉じ方 |
|---|---|---|
| 1 | 矢印を行の移動にする条件 | Trace モードの間だけ（D1。ユーザーの裁定で、暫定の既定「pane が開いている間」を改めた） |
| 2 | pane が開いている間の ←→ | Trace モードの間は何もしない（D2） |
| 3 | `forced` 列を既定に入れるか | 入れる（D4。scope の確認で合意） |
| 4 | ステータス欄の通知 | 出す（D5。scope の確認で合意） |
| 5 | `CONTEXT.md` に「強制行動」を足すか | 足す（D11。用語集に無い概念に名前を付けるので、[AGENTS.md](../../AGENTS.md) の規則から決まる） |
| 6 | 094 の扱い | 冒頭に注記を 1 行足す（D12） |

## 10. 関連

- [094 Evaluation Trace](094_evalpanel_step_history_10prd.md) の D10（表の上限と開き直し）、D11（Live モードと Trace モード、上下キー）、D12（プレイヤーの操作）、§6（Agent の入力の表示の先送り）。
- [999_input_mapping_joystick](999_input_mapping_joystick_10prd.md) §1.1（今のキーの経路）、§3.3（押下型と保持型）、§4（用語の候補）。
- [999_input_target_train_forced_action](999_input_target_train_forced_action_10prd.md) §1.1（強制行動の経路）、§1.2（強制行動を `MakeAction` の入力にする方針）、§9（スキップの継続と判断の記録）。
- [999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) §5 の 3（キーでの強制行動も同じ扱いにする）。
- [999_eval_session_step_trace](999_eval_session_step_trace_10prd.md)（評価 session の step trace を Run の記録に残す）。
- [ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)（Evaluation Trace の列の宣言）。
