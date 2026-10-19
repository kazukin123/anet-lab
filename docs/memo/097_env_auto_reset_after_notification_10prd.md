# BatchEnv の自動リセットを通知のあとへ — 通知の間は env を「遷移のあと・リセット前」に保つ 暫定 PRD

> 2026-10-19 起票（暫定。グリル前）。ユーザーの希望で、999 ではなく正式番号で起票した。質問はせず、暫定の既定を置いて §6 に決めることを残した。
> 起点: [PRD 094](094_evalpanel_step_history_10prd.md)（Evaluation Trace）のグリルで、step trace の列 `$env` を「遷移のあと・リセット前」の env と定義したところ、今の BatchEnv では終端の lane だけがリセット後になると分かったこと（094 の §3.3、D4、§7 の 1）。ユーザーの裁定で、この直しは別の PRD とし、094 より先に実施する。
> 関連: [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md)、[081_evalpanel_episode_trace_10prd.md](081_evalpanel_episode_trace_10prd.md)、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)（trace は `@episode_end` の通知の中で env の `GetScalar` を読む）、[140_observability.jp.md](../design/140_observability.jp.md)、[CONTEXT.md](../../CONTEXT.md) の「episode scope」「episode group」「trace」。

## 1. 目的と問題

BatchEnv の `Step` は、lane ごとに env を 1 step 進め、終端（done か truncated）になった lane をその場で `Reset` する。そのため、`Step` が戻って通知（`TrainEvent`・`EpisodeEndEvent`）が届いた時点では、終端の lane の env はもう次のエピソードになっている。通知の中で env へ問い合わせる読み手（`GetScalar`・`GetTensor`）は、終端の lane だけリセット後の値を読む。

これを埋め合わせるために、env は完了したゲームの値をリセットをまたいで持ち続け（Atari の `RecordGameCompletion`）、エピソード完了の読み手はそれに頼っている。一方、残機のような「今の値」は、終端の通知では次のゲームの値になる。EvalPanel の Atari の画面も、終端の step だけリセット後の画面になる。

### ゴール

- 通知（`TrainEvent`・`EpisodeEndEvent`）の間、どの lane の env も「遷移のあと・リセット前」の状態にある。通知の中の `GetScalar`・`GetTensor` は、終端の lane でもその遷移の結果を返す。
- そのうえで、次の判断に使う状態（今の `continue_state` に当たるもの）を、通知のあとに得る。
- 学習の値（metrics checksum）を変えない。

### 非ゴール

- replay と Agent が扱う遷移（`next_state` はリセット前の終端の観測）の意味を変えること。
- env の「完了時だけ値を返す」という `GetScalar` の仕様を変えること（エピソード完了の統計がこれに依っている）。
- 定期評価の採用 episode の規則（ADR 0034・0037）を変えること。

## 2. 背景: なぜ今の形か

ユーザーの説明（2026-10-19）: 自動リセットが要るようになったとき、`Step` の中でリセットまで行えば、既存の流れとも単一の env とも同じ形のまま、`Reset` と `Step` の 2 つだけという単純さを保てたので、この形にした。`ResetFinished` のような段を足すのは、その単純さを崩すので慎重に決めたい。

ユーザーの方向（同日）: 欲しいのは step の結果なので、リセットの前に Observer を呼ぶのが正しい。気がかりは、リセット後の情報をいつ・なぜ取るか（その必要性を含む）。

## 3. 現状（2026-10-19 にコードで確かめたこと）

- 単一の env をまとめる BatchEnv は 2 つある。逐次の `VectorizedDiscreteBatchEnv`（[env.cpp:712-742](../../core/anet-core/src/env.cpp)）と、スレッドの `ThreadPoolDiscreteEnv`（同 850-885 付近）。どちらも lane ごとに `envs_[i]->Step(a)` を呼び、done か truncated なら続けて `envs_[i]->Reset()` を呼び、`next_state` にはリセット前、`continue_state` にはリセット後の状態を詰める。aux（step の info）は `Step` の結果なのでリセット前。
- 自前でバッチを持つ `ImageClsEnv` は、エピソードの終わりで `continue_state` の `episode_start` を立てる（[ImageClsEnv.cpp:168-190](../../core/envs/imagecls1/src/ImageClsEnv.cpp)）。
- `EvalSessionEnv` は中の BatchEnv を包み、中の `Step` が戻ったあとで完了した lane の `GetScalar` を溜める（[env.cpp:336-369](../../core/anet-core/src/env.cpp)）。読むのはリセット後だが、読む鍵が完了値なので正しく読めている。セッションの継ぎ目では、前の `Step` の `continue_state` が全 lane とも新しいエピソードなら、`Reset` を呼ばずにそれを使い回す（同 317-327）。
- `continue_state` は、Runner の `state_ = result->continue_state`（評価 [trainer.cpp:295](../../core/anet-core/src/trainer.cpp)、逐次の学習 同 566、非同期の学習 同 700）と、エピソード構造の検査（`continue_state.episode_start` が done か truncated と一致すること。[env.cpp:142-176](../../core/anet-core/src/env.cpp)）で使われる。Agent・replay・View は使わない。
- 非同期の学習 Runner は、step t の `TrainEvent` を次の `DoStep` の頭で通知する（[trainer.cpp:634-656](../../core/anet-core/src/trainer.cpp)）。このとき env は step t の `Step`（とリセット）を終えている。逐次の学習 Runner と EvalRunner は、`Step` の直後に通知する（同 532-567、231-298）。
- Atari の env は、ゲームの完了時に `completed_*` を記録し、`GetScalar` は完了時だけそれを返す（[AtariEnv.cpp:696-717](../../core/envs/atari1/src/AtariEnv.cpp)、同 839-891）。Atari の View は終端の通知で、env の `rgb_frame`（リセット後）と、step の結果の観測・aux（リセット前）を一緒に出している（[AtariView.cpp:218-257](../../core/envs/atari1/src/AtariView.cpp)）。

## 4. 提案の骨子（暫定）

2 つの作り方がある。どちらも歪みは残らない。暫定の既定は、ユーザーの方向（リセットの前に Observer を呼ぶ）に沿う **M2** とする。

| | M1. 今の方式を完成させる（Reset/Step はそのまま） | M2. リセットを通知のあとの段にする（暫定の既定） |
|---|---|---|
| 考え方 | 遷移の時点の値は、すべて `Step` の結果から読む。宣言で購読された `GetScalar` の鍵（必要なら `GetTensor` も）を、BatchEnv が lane ごとにリセット前に取り、結果に添える。生きている env の `GetScalar` は「今」を返すもの、と位置付ける | `Step` はリセットしない。通知の間、env は遷移のあとのまま。読む人は生きている env を読めばよい。通知のあと、Runner が終わった lane だけをリセットする段（仮に `ResetFinished`）を呼び、次の判断の状態を得る |
| 契約の変更 | BatchEnv に購読の口を足す。読む側（step trace、episode_end の trace、セッションの集計）を結果から読むように変える | BatchEnv の契約（`continue_state` を `Step` の結果から外し、リセットの段を足す）、Runner 3 つ、エピソード構造の検査、EvalSessionEnv の継ぎ目 |
| env の埋め合わせ（完了値をリセットをまたいで持つこと） | 読む側が結果から読むようになれば要らない | 要らない |
| EvalPanel の終端の画面 | View が `rgb_frame` を購読すれば直る | そのまま直る |
| Reset/Step の単純さ | 保てる | 段が 1 つ増える |

M2 のときの Runner の順（暫定）:

```
逐次・評価:  MakeAction(s_t) → Step(a_t)（リセットしない） → 通知 → ResetFinished() → s_{t+1}'
非同期:      [前の step t の TrainEvent を通知] → ResetFinished() → MakeAction(t+1) → Step …
```

env ごとの乱数の消費順は「Step → Reset」のままで、Actor の乱数とは別なので、学習の値は変わらない見込みである（受入で確かめる）。

## 5. 作らないもの・先送り

なし（暫定の起票のため、グリルで決める）。

## 6. 決めること（グリルで詰める）

| # | 決めること | 暫定の既定 |
|---|---|---|
| 1 | M1 か M2 か | M2。ユーザーの方向に沿い、読む側を単純にできる |
| 2 | リセット後の情報（次のエピソードの最初の観測・画面）を、誰がいつ取るか。取る必要があるか | `ResetFinished` の戻り値（次の判断の状態）として Runner が受け取る。View がリセット直後の画面を出す必要があるかは、EvalPanel の表示の要件として確かめる（PRD 094 では、エピソードの最初の s_0 の画面はどの step trace にも入らないことがある） |
| 3 | リセットの段の名前と、`continue_state` を `Step` の結果から外すか | 名前は `ResetFinished`（仮）。`continue_state` は `Step` の結果から外し、リセットの段の戻り値にする（クリーンブレーク） |
| 4 | 非同期の学習 Runner の順 | 前の step の `TrainEvent` を通知してからリセットし、それから次の `MakeAction` |
| 5 | EvalSessionEnv の継ぎ目（全 lane が新しいエピソードのとき `Reset` を省く処理）と、採用 episode の規則 | 継ぎ目の判断をリセットの段の結果で行う。採用の規則は変えない |
| 6 | エピソード構造の検査（`continue_state.episode_start` と done・truncated の一致）の置き場 | リセットの段の戻り値で検査する |
| 7 | 自前でバッチを持つ env（ImageCls）の扱い | 同じ契約に合わせる（エピソードの終わりの切り替えをリセットの段へ移す） |
| 8 | env の完了値の保持（Atari の `completed_*`）を残すか | 読む側がすべてリセット前に読むようになるので、保持は要らなくなる。完了時だけ値を返す仕様は残し、保持の仕組みを外すかはグリルで決める |
| 9 | 受入の方法 | 同 seed・決定的モードの短い Run（Atari、DropMerge、LunarLander、GridMaze、ImageCls）で metrics checksum が一致する。終端の通知の中で `GetScalar("lives")` がリセット前の値を返す単体テスト。EvalPanel の Atari の終端の画面がリセット前になる手動の確認 |

## 7. 成功の測り方の目安

- 終端の lane の通知の中で、env の `GetScalar`・`GetTensor` がその遷移の結果（リセット前）を返す。
- 学習の値が変わらない（metrics checksum の一致）。
- PRD 094 の step trace の `$env` が、終端の行でも「遷移のあと・リセット前」になる。

## 8. 依存と順序

本 PRD → PRD 094（段階 1・2）→ PRD 081 の改訂。
