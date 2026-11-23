# 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 RFD

> 2026-11-17 番号を 944 から 161 へ振り直した（PRD131）。
> 2026-11-03 番号を 999 から 944 へ振り直した（PRD114）。

> 暫定 PRD（グリル前のドラフト）。番号 999。§9 の未決を裁定してから正式採番する。
> 起点: 2026-09-26 の [142](0142_agent_prev_action_obs_10rfd.md) 再検討。直前行動を Observation へ足す位置を Agent 内部に決めた結果、
> 前提として次の 2 つが残った。(1) frame stack と正規化の定義が 4 つの経路と spec 生成に散っていて、観測キーを 1 つ足すたびに
> 全部を手で揃える必要がある。(2) 強制行動が Actor の外で事後に差し替えられるので、Actor は自分の出力が実行されたかを知らない。
> [083](done/0083_action_context_retire_10rfd.md) は ActionContext の廃止だけを先行し、容器・用語・ADR を本 PRD へ送った。
> 改訂: 2026-10-23。098のReplay窓との接続条件を補足。強制したstep自体を判断として記録する。
> 改訂: 2026-10-26。PRD098から参照する既存ドラフトとして同梱対象を明示した。入力側の利用者に関する未同梱の検討メモは名称で区別し、その設計や本ドラフトの未決は確定事項として採用していない。§3のNG4・NG5・NG9へ今回扱わない理由を補った。
> 改訂: 2026-11-21。グリルを行って（2026-11-20〜21）暫定の印を外し、本文を全面的に書き直した。旧ドラフトの §4 の提案と §9 の Q1〜Q11 は、確定事項（§6 の D1〜D45）・未決表（§12）・クラス一覧（§7.1）へ振り分けた（振り分けは §14）。起草のあとに別モデル（Opus）の検討を受け（2026-11-21〜22）、事実の訂正と、決定から導ける補いを入れた（D8・D13・D18・D27・D32・D39、P122〜P130）。検討の問いへの答えで、共通の口を `Forward` だけにして spec の変換を拡張の口へ分け（D20・D22）、並びへの依頼を「何を渡して、何が欲しいか」で名付け、統計の更新を伴う Frame の依頼を足した（D26・D40）。並びと枚数を書かない設定はエラーにせず、1 枚の `FrameStack` だけの並びとして組む（D27・D30）。等価性の Run に DropMerge と Atari を足し（D35）、並びの図（dot）を足し（D41）、aux の `raw_obs`・`norm_obs` を消す（D42）。主な変更は次のとおり。観測加工の仕組みを、設定で書く「Forwarder の並び」にした。設定キー `stucker.*`・`obs_norm.*` を `perception.block.*` へ移す（旧 NG7 を外した）。stack の対象のキーは 1 枚でも stack 軸を持つ（旧 §4.4 の「1 枚は軸なし」を改めた）。素通しでない正規化は stack 軸へ広げて当てる（旧 §4.5 の構築時のエラーを外した）。`TensorSpec` に属性を 2 つ足し、Network も Forwarder の口を実装する。強制行動は `MakeActionOptions` の欄にする（旧 §4.6 の 3 つ目の引数を改めた）。レビュー（2026-11-22）の指摘で、次を直した。開発者が選んだものは、正規化の設定キーを `mode` とモードの名前空間に分けること（D43）と、新しいクラスを機能グループ `forwarder` の 1 つに置くこと（D45）である。そのほか、stack の対象のキーを `keys.mode` で選ぶ形にし（D44）、spec を通す口を `ForwardSpec` に改名し、`Scaled` の定義（D17）、Forwarder が持てる値（D20）、Actor から運ぶもの（D2、§7.4）、`ByMask` の検査（D10）、並びの読み方と検査の表（§7.6）、等価性の比べ方と受入 5・16・21・23・29・32（§9）、クラス一覧の置き場（§7.1）を書き直し、受入 34 と、前提 P131〜P139 を足した。再レビュー（同日）の指摘で、stack の対象のキーのスコープの名前を `keys` にし（開発者の選択。D44）、並びの入力に stack 軸を持つキーがある場合と、正規化の対象のキーが無い場合の扱い（D27・D40）、定数の要素数の照合（§7.6 の 8）、等価性の確かめの出力の取り方と Run を伸ばす上限（§9）を足した。3 回目のレビューの指摘で、並びの入力の spec にキーが無い構成の扱い（§7.6 の 6）、定数の一覧の空と要素数の扱い（§7.5、§7.6 の 8）、受入 34 の比べ方の手順と、前提 P140 を足した。レビューが OK になったあと、先送りしたものを課題に記録して、§15 の行を課題の ID に置き換えた。この行より前の引用行にある節番号は、旧ドラフトのものを指す。
> 改訂: 2026-11-23。実装中の確認を受け、学習 Run の等価性の比較条件を改めた。LunarLander・DropMerge・Atari では、比較する両方の Run に決定的なバックエンドを指定する（G2、P141、D35、§9）。Atari は Q*bert の学習 Run で比べるとした（D35、§9、U4）。
> 改訂: 2026-11-23。開発者の指示により、GridMaze を含めた全環境で、比較する両方の Run に決定的なバックエンドを明示する（D35、§9）。
> 改訂: 2026-11-23。§7.8 に旧キーを使う検証 2 箇所を記録し、受入 18 の検索で固定入力の移行と未読キーのテストに各 1 行、製品コードと出荷設定に 0 行となることを確かめるよう改めた。
> 基準コード: 2026-11-21 の main（`94182c99`）。実装の入口で、§1 と §4 の行番号をシンボル名で引き直す。
> 関連: [ADR 0083](../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md)、[ADR 0084](../adr/0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md)、[ADR 0085](../adr/0085-tensorspec-carries-value-scale-and-stack-axis-attributes.md)、[ADR 0086](../adr/0086-perception-stack-target-keys-always-carry-stack-axis.md)、[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)、[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)、[130 §2.2](../design/130_neural_networks.jp.md)、[150 §2.3](../design/150_replay_buffer.jp.md)、[200](../design/200_dqn_agents.jp.md)、[CONTEXT.md](../../CONTEXT.md)。

## 1. 問題

行番号は 2026-11-21 の main のもの。

### 1.1 観測の加工の定義が 4 つの経路と spec の生成に分かれている

DefaultDQN が「Env の出した Observation を Network の入力にする」処理は、次の 4 つの経路と、spec を作る関数 1 つに分かれて書かれている。

| 経路 | 場所 | していること |
|---|---|---|
| 行動選択 | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1534・1541 行 | stacker で重ねる（stacker が無ければ device へ移すだけ）。そのあと正規化する |
| 経験の受入れ | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 964・965 行 | 正規化の統計を、obs と next_obs で 1 回ずつ更新する。ReplayBuffer へは加工前の観測を渡す（968 行） |
| Sample | [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 929・952〜962・2348 行 | stack の定義を ReplayBuffer の設定へ写す。ReplayBuffer が重ね直し、Learner が正規化する。正規化するのは obs、next_obs、skip の学習の窓の終点の観測の 3 か所 |
| 可視化 | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 637〜678 行 | 合成した 1 Frame を stack の幅へ複製し、正規化する |
| spec | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 424〜447 行（`MakeNetworkInputSpec`） | Env の観測の spec を写し、stack の対象のキーの shape の先頭へ枚数を足す。taus の spec はそのあとに足す |

「経験の受入れ」は、Env を 1 step 進めたあと、Agent が経験（観測・行動・報酬・終了・次の観測）を受け取って、ReplayBuffer に入れるまでの処理である（`DefaultDQNAgent::UpdateFromBatch`）。本 RFD では push とも書く。

「どのキーを重ねるか」「形がどう変わるか」「何で正規化するか」は 1 つの定義なのに、経路ごとに書き直されている。観測キーを 1 つ足す（RFD0142 の直前の行動）には、この全部をそろえて直すことになる。

### 1.2 stack の対象の判定と、無いキー名

「`stack_keys` が空なら全キー、そうでなければ書いたキーだけ」の判定が、spec を作る関数（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 435 行）と stacker（[stacker.cpp](../../src/core/base/src/stacker.cpp) 56 行）に別々にある。観測に無いキー名を `stack_keys` に書いても、どこも検出しない。

### 1.3 形の食い違いと、表に出ていない不成立

| # | 事実 | 表に出ていない理由 |
|---|---|---|
| a | stack を使う設定で枚数が 1 のとき、行動選択と可視化は `[B, 1, *shape]`、Network の入力 spec と Sample は `[B, *shape]` になる（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 430・662・733 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 929 行） | この組み合わせの出荷設定は GridMaze（[GridMaze.txt](../../apps/runner/config/GridMaze.txt) 108・154 行）。Network の先頭が `Flatten`（213 行）なので、値は同じになる |
| b | 正規化の部品は、stack 軸の無い形（`[B, *shape]`）を前提に形を検査する。素通しでない設定に stack 軸つきの入力を渡すと、動的な統計では止まる。統計の更新は重ねる前の Frame に対して呼ぶので、形は合う | 出荷設定はどれも素通し（[agent.txt](../../apps/runner/config/agent.txt) 304 行） |
| c | Actor の正規化は、Agent の排他の外で統計を読む。統計の更新は排他の中で行う。背景評価と EvalPanel の Actor は、学習と並んで走る | 同上（統計を読まない） |
| d | 正規化の統計は checkpoint に入らない | 同上 |

a と b は本 RFD が直す。c と d は直さない（§2 の NG2）。

### 1.4 強制行動が Actor の外で、あとから差し替えられる

- 経路は `EvalPanel`・`QValuePanel` → `EvalRunner::DoStep(action, …)` → `EvalRunner::DoStepInternal` の 1 本で、`MakeAction` の結果を `WithAction` で差し替える（[trainer.cpp](../../src/core/base/src/trainer.cpp) 288〜297 行）。負数が「強制なし」の印である。Train Runner に強制の口は無い。
- `DQNActionInfo::WithAction`（[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 183〜226 行）は、Actor Q ヒントと Munchausen の項を、強制行動で作り直す。そのためだけに、Actor は全行動ぶんの `munchausen_terms` を aux に載せている（1633 行）。
- 行動を決める点が 2 つ（Actor の選択と Runner の差し替え）あるので、Actor は「自分の出力 = 実行された行動」を仮定できない。TempoRL の skip は Actor の中で行動を上書きして lane ごとの継続状態を持つが、強制はそのあとに差し替わるので、skip の状態は強制を知らない。RFD0142 の直前の行動も、実行された行動から作る値である。
- 強制したかどうかは `EvalRunner` が lane ごとに覚えるが、今は全 lane を同じ値で埋めている（[trainer.cpp](../../src/core/base/src/trainer.cpp) 314 行）。
- 旧ドラフトが所見に挙げた「GPU 推論で `WithAction` の device 検査に掛かって止まる」は、RFD0192 が直した（`94182c99`）。

### 1.5 本 RFD を前提にする RFD

| RFD | 要るもの |
|---|---|
| [142](0142_agent_prev_action_obs_10rfd.md)（直前の行動） | 観測キーを足す部品の置き場（単位 B）。Actor の出力 = 実行された行動（単位 A） |
| [159](0159_input_mapping_joystick_10rfd.md)（操作マッピング）・[160](0160_input_target_train_forced_action_10rfd.md)（入力先の切り替え） | timer の経路と Train Runner から、同じ欄で強制行動を渡せること（単位 A） |
| [174](0174_llm_actor_10rfd.md)・[183](0183_temporl_skip_followups_10rfd.md) | 単位 A |
| [173](0173_intrinsic_reward_novelty_10rfd.md)（NGU 型） | 方策番号・直前の行動・直前の報酬を Network の入力へ足す置き場（単位 B） |

## 2. ゴールと非ゴール

**ゴール**

- **G1**: 観測の加工の定義（どのキーを重ねるか、形がどう変わるか、何で正規化するか）を 1 つの仕組みが持ち、spec・行動選択・Sample・可視化がそこから導かれる。
- **G2**: 出荷設定（新しい設定キーへ移したもの）を使い、学習の挙動が変わらないことを確かめる（同 seed の Run の metrics が一致する。比較時のバックエンドの指定は §9）。
- **G3**: 行動選択・可視化・Sample・Network の入力 spec の形が、枚数 1 を含む全設定で一致する。
- **G4**: 成り立たない設定（観測に無いキー名、並びの誤り、知らない部品の種類）は構築時に止まる。
- **G5**: 強制行動が `MakeAction` の入力になり、「Actor の出力 = 実行された行動」が不変条件になる。
- **G6**: RFD0142 が「部品を 1 つ足す」局所の変更で済む置き場になっている。
- あわせて、題材の課題 2 件（§5）が片付く。

**非ゴール**

行の「本当の制約か」は、クリーンブレークで今回一緒に移せない理由があるかを書く。

| # | 入れないもの（消える案） | 理由と、本当の制約か |
|---|---|---|
| NG1 | 直前の行動（`prev_action`）の中身と、その lane の記憶 | 別の関心（RFD0142）。本 RFD は置き場と口までを決める。制約ではなく分担 |
| NG2 | 統計つきの正規化を成立させること（統計の読み書きの競合、checkpoint、評価だけの Run での統計の凍結）。正規化を NN の block にする案 | 別の関心。出荷設定はどれも素通しで、今回の等価性の確かめに乗らない。本 RFD は、形の不成立（§1.3 b）だけを直す（D28）。制約ではなく分担で、作る条件は「統計つきの正規化を使う構成（連続制御など）を足すとき」 |
| NG3 | RND の観測加工を同じ仕組みに載せること。RND の正規化の判定を尺度の属性に合わせること | 別の関心。RND は byte の画像を float に直して自前の統計で正規化していて、属性に合わせると Atari の動きが変わる（D33）。RFD0191 が RND を先に直す。制約ではなく分担で、課題に記録した（§15）。着手の条件は、RFD0191 の実装が済み、RND の入力に、Q の側の並びと同じ部品（重ねる、キーを落とす、正規化）を当てる構成が要るようになったとき |
| NG4 | 正規化の部品の内部の整理（定数がキー共通の 1 組であること、など） | 別の関心（部品の内部の作り）。観測の加工の定義を 1 か所にする今回の目的には要らない。制約ではなく分担で、課題に記録した（§15）。着手の条件は、統計つきか定数の正規化を、形の違う float のキーが複数ある構成で使うとき（§8 の 11）。NG2 の作業と一緒に行う |
| NG5 | ReplayBuffer の重ね直し（ADR 0024・0044）と `DictFrameStacker` の契約の変更 | 保存した Frame の重ね直しと、逐次の観測の重ねを担う部品で、観測の加工の定義をそろえる今回の目的と別の処理方式の変更になる。本 RFD は 2 つをそのまま使う（D27）。移せない制約ではなく、範囲の線である |
| NG6 | taus と、Env の中の前処理（Atari の縮小など）の統合。NN の入口の整形（byte→float、離散の one-hot）の置き換え | taus は forward のたびに引き直す Agent の入力、Env の中の前処理は Observation を作る側で、どちらも Perception を作る仕組みの外にある。NN の入口は、stack 軸の見分け方だけを直す（D32）。移せない制約ではなく、責務の線である |
| NG7 | Rainbow・MuZero・ImageCls の観測の加工を、この仕組みに載せること | Rainbow は stack も正規化も持たず、MuZero は device へ移すだけ、ImageCls は対象外で、消す重複が無い。Network は 3 つとも `Forwarder` を実装する形に変わり（D20）、Env の spec の宣言（D18）は Env の側で変わる。制約ではなく、載せても消える重複が無いための線である |
| NG8 | Train 側で、人の操作を強制行動として届ける配線 | 別の関心（RFD0160）。Runner と Actor の口は Train 側にも開いている（D15）。制約ではなく分担 |
| NG9 | PPO | 新しい Agent の実装で、今回の 2 つの単位に要らない。制約ではなく、別の機能の追加である |
| NG10 | 逆方向（入れるとき用）の変換の口 | ReplayBuffer には加工前の Frame を入れるので、戻す処理が要らない。制約ではなく、要るようになったら足せる（見直しの条件は D6） |

## 3. 範囲

| # | 入れる仕組み | 効くゴール | 単位 |
|---|---|---|---|
| S1 | 強制行動を `MakeActionOptions` の欄にして、`MakeAction` の入力にする | G5 | A |
| S2 | `WithAction`（あとからの差し替え）と、そのためだけの aux `munchausen_terms` を消す | G5 | A |
| S3 | 強制した lane は、TempoRL の skip の継続状態を捨て、その step を新しい判断として記録する | G5 | A |
| S4 | 観測の加工の定義を持つ 1 つの仕組み（Forwarder の並び）。Network の入力 spec、stack の対象の判定、可視化用の複製、正規化をここから出す | G1・G6 | B |
| S5 | 語（Frame・Perception・Forwarder・Forward の依頼・stack 軸・値の尺度） | G6 | 全体 |
| S6 | stack の対象のキーは、行動選択・Sample・可視化のどこでも、必ず stack 軸を持つ。1 枚でも長さ 1 の軸を持つ | G3 | B |
| S7 | 構築時の検査（観測に無いキー名、枚数が 1 未満、並びの誤り） | G4 | B |
| S8 | 素通しでない正規化は、統計を stack 軸へ広げて当てる | G4 | B |
| S9 | 値の尺度の属性を `TensorSpec` に足す。spec は部品が順に受け渡して作る | G1・G6 | C |
| S10 | 「TensorDict を受けて返す変換」の共通の口（Forwarder）と、spec の変換も持つ拡張（SpecForwarder）。Network は Forwarder を実装する | G1 | C |
| S11 | stack 軸を持つかの属性を `TensorSpec` に足す。stack の部品、そのあとの部品、NN の入口がこの属性を見る | G3・G4 | C |
| S12 | 並びを設定で書く口。今の設定キー `stucker.*`・`obs_norm.*` を、部品の設定へ移す | G1・G6 | B |
| S13 | 汎用の部品の最初の 1 つとして、キーを落とす部品 | G1 | B |
| S14 | forward の途中の値の持ち帰り（capture）を、共通の口でも同じ器で受ける | G1（Network の今の口を、そのまま共通の口にする） | C |
| S15 | 並びと実行器に seed を渡す口 | G6（乱数を使う部品を、口を変えずに足せる） | C |
| S16 | NN の側の、stack 軸の扱いの確かめ（先頭の module の WARN、`StackMerge` の入力の検査、離散の Grid の見分け方） | G3・G4 | B |
| S17 | 並びの図（dot）。NN の構造図と同じ仕組みで、部品・順番・各段の spec を Run の記録に出す | G1・G3（定義と形を、図で確かめられる） | B |

入れないものは §2 の非ゴールのとおり。

## 4. 前提

種類は、事実・用語・既存部品の働き・過去の判断・守る線。P1〜P14 はゴールの合意のときに示した共通の前提、P15 以降は問いに添えた前提、P112 以降は起草で足した前提、P122 以降は別モデルの検討（2026-11-21）で足した前提、P131〜P140 はレビューの指摘への直し（2026-11-22）で、P141 は実装中の確認（2026-11-23）で足した前提である。P66 は欠番。

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実 | 観測の加工（stack・device への転送・正規化）は、行動選択・統計の更新・Sample・可視化の 4 か所と、Network の入力 spec を作る関数に分かれている | §1.1 | 済み |
| P2 | 事実 | 枚数 1 で形が食い違う。出荷設定では GridMaze が当たる | §1.3 a | 済み |
| P3 | 事実 | `stucker.stack_keys` に観測に無いキー名を書いても、どこも検出しない | §1.2 | 済み |
| P4 | 事実 | 出荷設定の観測の正規化はすべて素通し。素通しでない正規化は 1 Frame の形を検査するので、stack ありだと最初の行動選択で掛かる | §1.3 b | 済み |
| P5 | 既存部品の働き | 強制行動は、EvalRunner が `MakeAction` の結果を `WithAction` で差し替えて作る。Train 側に口は無い | §1.4 | 済み |
| P6 | 過去の判断 | `MakeAction` の追加の入力は `MakeActionOptions` で、Runner の `DoStep` がそのまま渡す。強制行動はこの欄として足す、と RFD0170 が決めた | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 357 行、[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)、[0170_nn_trace_capture_unification_10rfd.md](0170_nn_trace_capture_unification_10rfd.md) §9 | 済み |
| P7 | 既存部品の働き | 強制したかどうかは EvalRunner が lane ごとに覚え、`forced` の鍵で答える | [trainer.cpp](../../src/core/base/src/trainer.cpp) 314 行 | 済み |
| P8 | 既存部品の働き | TempoRL の skip は Actor の中で行動を上書きし、lane ごとの継続状態を持つ。強制はそのあと差し替わるので、skip の状態は強制を知らない | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1571・1581 行、[0098_temporl_skip_policy_10rfd.md](0098_temporl_skip_policy_10rfd.md) §9 | 済み |
| P9 | 既存部品の働き | Actor Q ヒント（5 列）は `WithAction` が強制行動で作り直す。そのためだけに Actor は `munchausen_terms` を aux に載せている | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 183・1633 行 | 済み |
| P10 | 過去の判断 | RFD0083 は `ObservationPipeline` クラスを作らず、「要件がそろう後続の RFD で入れる」とした | [0083_action_context_retire_10rfd.md](done/0083_action_context_retire_10rfd.md) の複雑さの監査 | 済み |
| P11 | 守る線 | 重ねる → 転送 → 正規化の並びと、乱数の消費順を変えない。等価性は同 seed の metrics checksum で確かめる | [321_agent_core.md](../wiki/categories/321_agent_core.md) の「変えるときの条件」 | 済み |
| P12 | 事実 | 待っている RFD はすべて暫定。単位 A は 142・159・160・174・183、単位 B は 142・173（173 は、Q の入力に直前の行動・直前の報酬・方策番号を足す案を持ち、その置き場に単位 B を使う） | §1.5 | 済み |
| P13 | 過去の判断 | RND は自前の観測の正規化を持ち、RFD0130 は「観測の加工を 1 か所にまとめる話は RFD0161 の題材」と送った | [0130_rnd_estimate_statistics_perf_10rfd.md](0130_rnd_estimate_statistics_perf_10rfd.md) の D3 | 済み |
| P14 | 事実 | DQN のファイル名は `dqn_based_*` から `dqn_*` に変わった | コミット `e72bab34`（`git log` で 2026-11-20 に確認） | 済み |
| P15 | 既存部品の働き | ReplayBuffer は重ねる前の 1 Frame を保存し、Sample のときに重ね直す。加工の流れは保存の位置で切れる | [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) §2.3 | 対象外 |
| P16 | 既存部品の働き | `MakeAction` が受ける状態は観測・終了・episode の開始の印だけで、報酬は入っていない | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 430 行 | 対象外 |
| P17 | 既存部品の働き | 「TensorDict を受けて返す関数」の型は既にあり、可視化が使っている | [common.hpp](../../src/core/base/include/anet/common.hpp) 26 行 | 対象外 |
| P18 | 過去の判断 | 定義を共有し、用途ごとの実行器が状態を持つ作りは、NN の forward に前例がある | [ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) | 対象外 |
| P19 | 既存部品の働き | NN は block の種類を名前で登録し、`structure` で並べ、定義と用途ごとの実行器を分けている | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §2.3・§2.4・§7.7 | 対象外 |
| P20 | 既存部品の働き | NN の block は Tensor 1 つを受けて 1 つ返す。どのキーを通すかは `bind` が決める | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §3 | 対象外 |
| P21 | 既存部品の働き | Network の入口が、byte→float と離散値の one-hot を既に行っている | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1221〜1262 行 | 対象外 |
| P22 | 既存部品の働き | Network の入力 spec は stack 軸つきで、仮の forward も Actor も Learner も同じ形を入れる | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §2.2 | 対象外 |
| P23 | 既存部品の働き | 今の stacker は、Frame を先に Actor の device へ移し、その device の上に履歴を持つ | [stacker.cpp](../../src/core/base/src/stacker.cpp) 62 行 | 対象外 |
| P24 | 事実 | 「GPU/CPU の両対応」は、「部品は渡された Tensor のある device の上で動き、どこで移すかは経路が決める」の意味と読んだ（AI の読み） | 開発者の発言（2026-11-20）。§13 | 対象外 |
| P25 | 既存部品の働き | ReplayBuffer の slot は、その step の観測 1 つ・行動・info・履歴開始・報酬・終了・判断のフラグを持つ。打ち切りのときだけ終端の観測を専用の slot に置く | [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) §2.2・§7.2 | 対象外 |
| P26 | 既存部品の働き | Sample は、選んだ slot から過去へ枚数ぶんの観測をたどって重ね、履歴開始より前は開始の観測で埋める | [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) §2.3 | 対象外 |
| P27 | 用語 | 「フレーム」は用語集で、エミュレータのフレームと stack の 1 枚の両方の意味に使われていた | [CONTEXT.md](../../CONTEXT.md)（2026-11-20 に確認。Frame の項目を足して書き分けた） | 対象外 |
| P28 | 既存部品の働き | NN では、アルゴリズム設定から実装が Head と追加の入力 spec（taus）を作り、`net.*` は Body の配線だけを持つ | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §2.1・§6.1 | 対象外 |
| P29 | 守る線 | 境界の検査は局所で決まるものだけで、入力が出力に寄与するかは設定する側の責任 | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §7.1、[AGENTS.md](../../AGENTS.md) | 対象外 |
| P30 | 過去の判断 | RFD0142 のドラフトは、直前の行動を Agent の設定キーで有効にし、NN で使うには `bind` に書くとしている | [0142_agent_prev_action_obs_10rfd.md](0142_agent_prev_action_obs_10rfd.md) の「設定」 | 対象外 |
| P31 | 過去の判断 | RFD0173 の NGU の基準案は、Q の条件づけに方策番号・直前の行動・直前の報酬を使う | [0173_intrinsic_reward_novelty_10rfd.md](0173_intrinsic_reward_novelty_10rfd.md) §4.5 | 対象外 |
| P32 | 過去の判断 | RFD0173 は、ある方策番号で集めた遷移を別の方策番号として学ぶ形を、決める候補に挙げている | [0173_intrinsic_reward_novelty_10rfd.md](0173_intrinsic_reward_novelty_10rfd.md) §4.5 | 対象外 |
| P33 | 用語 | 用語集は `action_mask` を「学習入力には含めない」観測キーとしている | [CONTEXT.md](../../CONTEXT.md) の「観測キー」 | 対象外 |
| P34 | 事実 | RND の補助学習を、保存した標本から引きたいという課題がある | [issue-20261119-4976bf.md](../management/issues/issue-20261119-4976bf.md) | 対象外 |
| P35 | 既存部品の働き | skip の Network と RND は、Q の Network とは別の入力を実装が組んでいる | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1687 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 242 行 | 対象外 |
| P36 | 事実 | 行動選択側と Sample 側の Perception が、episode の境目・打ち切り・n-step 先まで往復で一致すること。Frame にキーを足す部品については確かめられていない | 欠け（分かれ方で決めた。未決表 U1） | 対象外 |
| P37 | 既存部品の働き | Actor が作った値を行動の情報に載せ、ReplayBuffer が Push で使う道が既にある（Actor Q ヒント、判断のフラグ） | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1615〜1635 行、[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 1505・1565 行 | 対象外 |
| P38 | 既存部品の働き | 学習の 1 step は、行動選択 → Env の step → 受入れの順で、行動選択の時点ではその観測は ReplayBuffer に無い | [trainer.cpp](../../src/core/base/src/trainer.cpp) 574 行 | 対象外 |
| P39 | 既存部品の働き | 今の stacker は、観測のキーごとに履歴を持つ（観測の dict 全体の履歴） | [stacker.cpp](../../src/core/base/src/stacker.cpp) 71 行 | 対象外 |
| P40 | 既存部品の働き | N-step のキューは ReplayBuffer の中にあり、観測を持たない軽い記録である | [replay_experience.hpp](../../src/core/base/include/anet/replay_experience.hpp) 15 行、[150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) §2 | 対象外 |
| P41 | 既存部品の働き | next_obs は t + 実際の n の slot を最新として重ねる。stack の境は N-step の状態を使わない | [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) §2.3 | 対象外 |
| P42 | 用語 | 用語集とコードは、taus を「Observation ではなく Agent が注入する入力」とし、spec へは stack の調整のあとに足している | [CONTEXT.md](../../CONTEXT.md)、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 447 行 | 対象外 |
| P43 | 既存部品の働き | taus は、行動選択と学習でそれぞれ引く。学習の中でも複数回引く | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 958 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 2325 行 | 対象外 |
| P44 | 既存部品の働き | stack の対象は、設定のキーの一覧で決まる。設定の一覧が空のとき、Agent は stacker に一覧を渡さず（指定なしにして）、stacker は、一覧の指定が無ければ全キーを重ねる | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 735 行（`CreateFrameStacker`。空でないときだけ一覧を渡す）、[stacker.cpp](../../src/core/base/src/stacker.cpp) 56 行 | 対象外 |
| P45 | 既存部品の働き | 正規化の対象は、キーの性質で決まる。float の連続値のキーにだけ作り、ほかは素通し | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 633〜649・760 行 | 対象外 |
| P46 | 過去の判断 | RFD0142 のドラフトは、直前の行動を float の one-hot の別キーにし、「足したキーは正規化しない」「stack の既定は全キー」としている | [0142_agent_prev_action_obs_10rfd.md](0142_agent_prev_action_obs_10rfd.md) の決定 2・7・8 | 対象外 |
| P47 | 既存部品の働き | 定数の正規化の平均と標準偏差は、全キー共通の 1 組 | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 769 行 | 対象外 |
| P48 | 既存部品の働き | `TensorSpec` は、形の種類・shape・dtype・クラス数（0 なら連続値）・値の範囲を持つ。離散値は Env が宣言している | [common.hpp](../../src/core/base/include/anet/common.hpp) 94 行、[DropMergeEnv.cpp](../../src/env/dropmerge/src/DropMergeEnv.cpp) 294 行 | 対象外 |
| P49 | 既存部品の働き | spec の検査は、離散値なのに float の指定を止める | [rl.cpp](../../src/core/base/src/rl.cpp) 82 行 | 対象外 |
| P50 | 事実 | Network の入力 spec は、関数 1 つが stack の分を手で足して作っている | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 424 行 | 対象外 |
| P51 | 事実 | 「float かつ連続値」の判定が、観測の正規化と RND の 2 か所に同じ式で書かれている | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 760 行、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 394 行 | 対象外 |
| P52 | 事実 | `IsDiscrete()` を呼ぶのは、テストを除いて 6 か所。`num_classes` は 33 ファイルに 70 か所 | `git grep`（`src` と `apps`、2026-11-20 に確認） | 対象外 |
| P53 | 既存部品の働き | spec は JSON に書き出され、Env どうしの spec の一致の確かめに使われる | [common.cpp](../../src/core/base/src/common.cpp) 19 行、[rl.cpp](../../src/core/base/src/rl.cpp) 245 行 | 対象外 |
| P54 | 既存部品の働き | `Network::Forward` は TensorDict を受けて TensorDict を返す。状態つきの実行は別の `NetworkForwardExecutor` が持つ | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 210・263 行 | 対象外 |
| P55 | 既存部品の働き | Network は入力の spec を構築時に受け取る。出力の spec を返す口は無い | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 201 行。NN の公開ヘッダを `output_spec`・`OutputSpec` で検索して 0 件（2026-11-20） | 対象外 |
| P56 | 既存部品の働き | 可視化の関数は、Agent が手で組んでいる | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 637 行 | 対象外 |
| P57 | 既存部品の働き | Network を組むとき、仮の入力で Body を 1 回通し、その出力を Head の生成に渡す。Head は、この仮の入力では実行されない（skip が有効なときだけ、別の箇所で Head まで通す。P124） | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 2522〜2550 行 | 対象外 |
| P58 | 既存部品の働き | Head は、構造図のために出力の名前と shape を手で書いている。使うのは図だけ | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 128 行 | 対象外 |
| P59 | 既存部品の働き | skip の補助 Network の入力 spec は、primary を仮の入力で 1 回 forward し、出力の shape と dtype を読んで作っている | [dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) 62 行 | 対象外 |
| P60 | 過去の判断 | Network の出力には、Head の出力に加えて、Body が公開する特徴が入る | [ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) | 対象外 |
| P61 | 事実 | Head のクラスは、テストを除いて 12（DQN 8、MuZero 2、基盤 2） | `git grep`（`src`、2026-11-20 に確認） | 対象外 |
| P62 | 事実 | GridMaze は、座標を Env の中で 0〜1 にして出している。spec は float の連続値 | [GridMazeEnv.cpp](../../src/env/gridmaze/src/GridMazeEnv.cpp) 158・190 行 | 対象外 |
| P63 | 事実 | Atari の `grid` は byte で、値の範囲は 0〜255 | [AtariEnv.cpp](../../src/env/atari/src/AtariEnv.cpp) 689 行 | 対象外 |
| P64 | 事実 | 重ね済みかを spec に書く欄は無い。stack の対象のキーの spec は、元の shape の先頭に軸を足した形になる | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §2.2 | 対象外 |
| P65 | 既存部品の働き | NN の入口は、離散値の Grid を one-hot にするとき、次元の数で stack の有無を見分けている | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1228 行 | 対象外 |
| P67 | 事実 | `stucker.` の設定キーは、出荷設定の 4 ファイルに 17 行。`obs_norm.` の設定キーは `agent.txt` の 13 行。bat と ps1 には無い | `git grep`（`apps`、2026-11-20・21 に確認） | 対象外 |
| P68 | 既存部品の働き | RND は、byte の画像を float に直して、自前の統計で正規化している | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 394・683 行 | 対象外 |
| P69 | 事実 | `action_mask` を出す Env は今は無い（鍵の定義と、古い形式への変換の除外だけがある） | `git grep`（`src`、テストを除く。2026-11-20 に確認） | 対象外 |
| P70 | 事実 | TorchRL の逆方向の変換は、行動を env に渡す前と、replay buffer へ入れるときに呼ばれる。逆方向のキーが空なら何もしない。並びは逆順に当たる | TorchRL の公開のソース `torchrl/envs/transforms/_base.py` と公式文書の Transforms の解説（main、2026-11-20 に確認） | 対象外 |
| P71 | 既存部品の働き | anet は ReplayBuffer に加工前の観測を入れる。入れるときにするのは、統計の更新と、報酬の scale | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 950・964・968 行 | 対象外 |
| P72 | 事実 | Sample の側の正規化は 3 か所（obs、next_obs、skip の学習の窓の終点の観測） | [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 952〜962・2348 行 | 対象外 |
| P73 | 事実 | 観測を元の値へ戻す処理は、いまの `src` に無い | `src` の hpp・cpp を、戻す処理の名前（Denormalize・Unnormalize・Inverse・Unscale）で検索。当たった 10 行はどれも勾配の GradScaler（2026-11-20 に確認） | 対象外 |
| P74 | 事実 | ワークスペースの設定（`data/`）に、`stucker.`・`obs_norm.` のキーは無い | `git grep`（`data`、0 行。2026-11-21 に確認） | 対象外 |
| P75 | 既存部品の働き | どの部品にも読まれなかった設定キーは、RunnerApp の起動時に WARN で一覧される。NN の block の定義の中は、読まれないキーがあるとエラーで止まる | [config.cpp](../../src/core/base/src/config.cpp) 353〜385 行、[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 548 行、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1699 行 | 対象外 |
| P76 | 事実 | 単位 A を前提や利用先に挙げる RFD は 5 本（142・159・160・174・183）。単位 B を前提にするのは 2 本（142・173） | §1.5 | 対象外 |
| P77 | 過去の判断 | RFD0192 は、単位 A が先なら自分の段階 3 を行わず、RFD0192 が先なら単位 A が `WithAction` ごと消す、と書いている | [0192_concurrency_and_boundary_defect_fixes_10rfd.md](0192_concurrency_and_boundary_defect_fixes_10rfd.md) §6.9 | 対象外 |
| P78 | 事実 | 課題 aa5758 は単位 A と B の両方を 1 件に含む。課題 b754ff は B だけ | §5 | 対象外 |
| P79 | 過去の判断 | 旧ドラフトの起票のときの推奨は「単位 A を独立の RFD にして先行」だった | 旧ドラフトの §9 の Q1（`git show 94182c99:docs/memo/0161_observation_pipeline_10rfd.md` で、2026-11-21 に確認） | 対象外 |
| P80 | 守る線 | 次に進めるのは RFD0191 で、RFD0161 を待つ後続は急ぎではない。グリルの作業効率を優先する | 開発者の発言（2026-11-20）。§13 | 対象外 |
| P81 | 事実 | RFD0191 は RND の内的報酬の式と predictor の学習の RFD | [0191_rnd_raw_error_and_recent_learning_10rfd.md](0191_rnd_raw_error_and_recent_learning_10rfd.md) の題と §1.1 | 対象外 |
| P82 | 既存部品の働き | Runner の強制の口は、行動の引数を持つ `EvalRunner::DoStep` で、負数が強制なし。中で `WithAction` に 1 要素の行動を渡している | [trainer.hpp](../../src/core/base/include/anet/trainer.hpp) 137〜144 行、[trainer.cpp](../../src/core/base/src/trainer.cpp) 293 行 | 対象外 |
| P83 | 事実 | 行動の引数つきの `EvalRunner::DoStep` を呼ぶのは、アプリの 2 か所（`EvalPanel.cpp` 219 行、`QValuePanel.cpp` 975 行）と、テストの 2 ファイル（`trainer_test.cpp`、`episode_end_test.cpp`） | `git grep`（`src`、2026-11-21 に確認） | 対象外 |
| P84 | 過去の判断 | RFD0159 は、押し続ける操作のために、timer の経路へ強制行動を渡す口が要るとしている。今の口は 1 step 単位で、timer の経路に乗らない | [0159_input_mapping_joystick_10rfd.md](0159_input_mapping_joystick_10rfd.md) 221 行 | 対象外 |
| P85 | 既存部品の働き | `nn_trace` は `MakeActionOptions` の欄で、Runner の `DoStep` と `DoUpdateFrame` が同じ構造体を受ける | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 357 行、[trainer.hpp](../../src/core/base/include/anet/trainer.hpp) 32 行 | 対象外 |
| P86 | 事実 | 今の強制の口は 1 要素の行動を渡す。Runner は lane ごとの「強制したか」の記録を、全 lane 同じ値で埋めている | [trainer.cpp](../../src/core/base/src/trainer.cpp) 293・314 行 | 対象外 |
| P87 | 過去の判断 | RFD0160 の Train の強制は、多 lane のうち 1 lane 固定。複数 lane の同時操作は非ゴール | [0160_input_target_train_forced_action_10rfd.md](0160_input_target_train_forced_action_10rfd.md) 54・98 行 | 対象外 |
| P88 | 既存部品の働き | DQN の Actor は、行動を選んだあと、その行動から初期優先度のヒントを作る。強制した行動は、この間に書けばヒントに反映される | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1565・1615 行 | 対象外 |
| P89 | 事実 | Stable-Baselines3 は、env を指す引数が無指定・番号 1 つ・番号の列を受け、中で番号の列に直す | 公開のソース `stable_baselines3/common/vec_env/base_vec_env.py`（master、2026-11-20 に確認） | 対象外 |
| P90 | 事実 | Isaac Lab は、番号で指す書き込みの口と、印で指す口の対を持つ。番号は重複不可 | 公開のソース `isaaclab/assets/articulation/base_articulation.py`（main、2026-11-20 に確認。後半は要約で読んだ） | 対象外 |
| P91 | 事実 | EnvPool は番号の列と同じ長さの行動の列、Gymnasium は env の数ぶんの印（reset だけ）、imitation の DAgger は印で方策の外から差し替える | EnvPool の Python interface の文書、Gymnasium `sync_vector_env.py`、imitation `dagger.py`（2026-11-20 に確認） | 対象外 |
| P92 | 既存部品の働き | NN は、状態なしの `Network::Forward` と、状態つきの実行器の二段になっている | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 210・221・263 行、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) | 対象外 |
| P93 | 事実 | stack の動きは、使われる所で 3 通りある（行動選択は履歴から重ねる、Sample は ReplayBuffer が重ね済み、可視化は複製する）。正規化の当て方は、使われる所によらない | §1.1 | 対象外 |
| P94 | 事実 | DQN の観測に増強を足す RFD は、今は無い | `docs/memo` の RFD を、増強の語（DrQ・random shift・データ拡張・augment）で検索。当たった 7 本はどれも `done/` の ImageCls の RFD（2026-11-20 に確認） | 対象外 |
| P95 | 事実 | 今は、stack を使う設定で 1 枚のとき、Network の入力 spec と学習側には軸が無く、行動選択と可視化には長さ 1 の軸がある | §1.3 a | 対象外 |
| P96 | 過去の判断 | stack 軸を Network で扱う module（`StackMerge` など）は、設定に明示して書く | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §2.2 | 対象外 |
| P97 | 事実 | Gymnasium の frame stack は 1 枚でも先頭に長さ 1 の軸を足す。Stable-Baselines3 は既存の軸に沿って重ね、軸は増えない | 公開のソース Gymnasium `wrappers/stateful_observation.py`、Stable-Baselines3 `vec_env/stacked_observations.py`（2026-11-20 に確認） | 対象外 |
| P98 | 事実 | 今 1 枚で動いている出荷設定は GridMaze（A1・A2）。Network の先頭は `Flatten` | [GridMaze.txt](../../apps/runner/config/GridMaze.txt) 108・154・213 行 | 対象外 |
| P99 | 事実 | 出荷設定で、stack を使わない設定は今は無い（基準が「使う、4 枚」） | [agent.txt](../../apps/runner/config/agent.txt) 178〜180 行、`git grep`（`apps`、2026-11-20 に確認） | 対象外 |
| P100 | 事実 | 設定の構造体の既定は「stack を使わない」。テストの一部は、この既定のまま Network を組んでいる | [dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 190 行、`src` を `use_stacker` で検索（テスト 5 ファイルに 12 行） | 対象外 |
| P101 | 事実 | PyTorch の LSTM は時系列の軸が必ずあり、1 step は長さ 1。RLlib は状態を持つモデルのときだけ軸を足す。rlpyt は軸が任意で、次元の数から推し量る | PyTorch の LSTM の文書、RLlib の ConnectorV2 の文書、rlpyt `utils/tensor.py`（2026-11-20 に確認） | 対象外 |
| P102 | 過去の判断 | RFD0057 は「先頭の block で畳み方を書く」と決めた。spec に印を持たせて組む側が見る案は、`TensorSpec` の直しが広いとして見送った。見直しの条件は「stack に依る block が増えたら」 | [0057_native_stack_dim_handling_10rfd.md](done/0057_native_stack_dim_handling_10rfd.md) 70・118〜127 行 | 対象外 |
| P103 | 既存部品の働き | `Flatten` は軸があっても無くても動く。`StackMerge` は 5 次元なら畳み、4 次元ならそのまま通す | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 873〜880・934〜947 行 | 対象外 |
| P104 | 事実 | 出荷設定には、軸を畳む定義（Atari の 29 個が `StackMerge` 始まり、MLP は `Flatten` 始まり）と、軸を時間方向に使う定義（LunarLander）の両方がある | [Atari.txt](../../apps/runner/config/Atari.txt) 1278 行ほか、[LunarLander.txt](../../apps/runner/config/LunarLander.txt) 272 行、[nn.txt](../../apps/runner/config/nn.txt) 290・309 行 | 対象外 |
| P105 | 既存部品の働き | NN の入力には、入口の変換を通さない印（raw）がある。離散の Grid は、入口が軸とクラスをチャネルへ畳んでいる | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 945〜999・1228 行 | 対象外 |
| P106 | 既存部品の働き | 正規化の部品は、軸なしの形を前提に形を検査する | §1.3 b | 対象外 |
| P107 | 事実 | 出荷設定は、どれも素通し（基準の設定だけが `obs_norm` を書いている） | [agent.txt](../../apps/runner/config/agent.txt) 304〜316 行 | 対象外 |
| P108 | 事実 | 設定の構造体の既定は「素通しでない」。テストの一部は、この既定のまま Agent を組んでいる | [scaler.hpp](../../src/core/base/include/anet/scaler.hpp) 48 行 | 対象外 |
| P109 | 過去の判断 | 旧ドラフトは、統計つきの正規化を成立させる作業を別の RFD へ持ち越し、その中で「先頭の stack 軸へ統計を広げる」方向を挙げていた | 旧ドラフトの §10（`git show 94182c99:docs/memo/0161_observation_pipeline_10rfd.md` で、2026-11-21 に確認） | 対象外 |
| P110 | 事実 | 既存の型に `Forwarder`・`Perception` で始まるものは無い。`Pipeline` を含むのは `PipelineTrainRunner` だけ。`ForwardExecutor` という名前の型は無い（近いのは `NetworkForwardExecutor` と `DqnForwardExecutor`） | `src` の hpp・cpp を型の名前で検索（2026-11-20・21 に確認） | 対象外 |
| P111 | 既存部品の働き | NN の設定は、部品を `net.block`、並びを `structure` と書く。設定の種類が `Flatten` なら、クラスは `FlattenModule` | [nn.txt](../../apps/runner/config/nn.txt) 65・240 行、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 873 行 | 対象外 |
| P112 | 既存部品の働き | forward の途中の値の持ち帰りは、鍵で要求する `NetworkForwardCapture` の 1 つに統一されている。`Network::Forward`・`ForwardUpTo`・実行器の `Forward` が任意で受け、渡さないときは「渡されたか」の判定だけで済む。要求した鍵が書かれなければ、Network の出口で止まる | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 162・210〜215・270 行、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 135〜149 行、[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) | 対象外 |
| P113 | 既存部品の働き | Network は、構築と、実行器を作る口の 2 か所で seed を受ける。既定の設定では、実行器の seed は使われない。名前から seed を作る計算は、乱数を消費しない | [nn.hpp](../../src/core/base/include/anet/nn.hpp) 207・219〜222 行、[random.cpp](../../src/core/base/src/random.cpp) 190〜193 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 337〜340 行 | 対象外 |
| P114 | 過去の判断 | WARN は「利用者が確かめて、設定・コード・実行環境を直すべき状態」に限る。読まれない設定キーは、実体化したカタログ項目では構築時に止め、それ以外は Run の構築の終わりに WARN で一覧する | [ADR 0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) | 対象外 |
| P115 | 既存部品の働き | DQN の Head の多くは、構築のとき、特徴の最後の次元だけを読む。IQN の Head は 3 次元の特徴を前提にする。Body の出力に stack 軸が残っていても、Head の構築では止まらない場合がある（動かして確かめてはいない） | [dqn_heads.cpp](../../src/algo/dqn/src/dqn_heads.cpp) 785・975・1016 行 | 対象外 |
| P116 | 既存部品の働き | 設定の解決結果を、保存した基準（17 構成）と比べるテストがある。基準は、解決後の設定キーの名前を含む | [config_baseline_test.cpp](../../src/core/base/src/config_baseline_test.cpp) 177〜187 行、`src/core/base/testdata/rfd0072/baseline/` | 対象外 |
| P117 | 事実 | RFD0192 と RFD0190 は実装済み（RFD0192 は `94182c99`）。RFD0192 は `WithAction` の device の扱いを直し、そのテストを足した | [0192_concurrency_and_boundary_defect_fixes_10rfd.md](0192_concurrency_and_boundary_defect_fixes_10rfd.md) §6.4、[0190_dqn_design_doc_alignment_10rfd.md](0190_dqn_design_doc_alignment_10rfd.md) 115〜120 行、[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 205 行 | 対象外 |
| P118 | 用語 | 設計文書は、重ねた枚数の軸を「stack 軸」と呼んでいる | [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) §2.2（41 行） | 対象外 |
| P119 | 事実 | メトリクスの鍵 `obs_norm.*`（`$agent obs_norm.count` など）は、設定キーとは別にあり、メトリクスの定義が使っている。RND は自前の設定キー `intrinsic.rnd.obs_norm.*` を持つ | [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) 301〜305 行、[agent.txt](../../apps/runner/config/agent.txt) 53 行 | 対象外 |
| P120 | 事実 | 同じ箇所に触れる、まだ実装していない RFD は 13 本（081・134・141・142・143・159・160・173・174・176・180・183・191） | 課題の一覧（[INDEX.md](../management/issues/INDEX.md)）を、§7.11 の RFD の番号で引いた（2026-11-21 に確認） | 対象外 |
| P121 | 事実 | 出荷設定の RND の Network の定義に、`StackMerge` で始まるものは無い | [Atari.txt](../../apps/runner/config/Atari.txt) 13〜17 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 536〜565 行、[GridMaze.txt](../../apps/runner/config/GridMaze.txt) 33 行、[LunarLander.txt](../../apps/runner/config/LunarLander.txt) 50 行 | 対象外 |
| P122 | 既存部品の働き | Learner が、stack の設定（枚数とキー）を ReplayBuffer の設定へ写している | [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 927〜933 行 | 対象外 |
| P123 | 既存部品の働き | ReplayBuffer の保存領域は、Env の spec から作る | [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 1475 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 945 行 | 対象外 |
| P124 | 既存部品の働き | Network を組むときの Body の仮の forward は、train の状態で行い、組んだあとに eval にする。DropPath は、生成器を指定しない乱数を引く。skip が有効なときは、eval にしたあとで、Head まで仮の forward を通している | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 2540〜2546 行、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 173 行、[dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) 40〜42・70 行 | 対象外 |
| P125 | 既存部品の働き | module は Tensor だけを受け、spec の属性を見られない。Embedder の module は、次元の数で stack 軸のあるなしを見分けている | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 2177 行 | 対象外 |
| P126 | 事実 | DropMerge の DefaultDQN は 4 枚重ねで、branch の先頭は `SpatialEmbedder`・`HybridSpatialEmbedder` である。DropMerge の float の vector は番号を運び、module が one-hot にする | [DropMerge.txt](../../apps/runner/config/DropMerge.txt) 59・725・726・810・1081 行、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 2209〜2214 行 | 対象外 |
| P127 | 既存部品の働き | Actor は、重ねたあと・正規化の前の観測を aux の `raw_obs` に、正規化のあとを `norm_obs` に載せている。読むのはテストだけである | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1640〜1643 行、`git grep`（`src`・`apps`・`data`、2026-11-21 に確認） | 対象外 |
| P128 | 事実 | テストは、設定を文字のキーで組んでいる（`DefaultDQNAgent.obs_norm.pass_through` など、5 ファイルに 12 行）。`DefaultDQNAgentConfig` は `src` の 11 ファイルに 248 行現れ、テストが DefaultDQN の設定や Agent を組む箇所は、テスト 7 ファイルと共通の補助 1 ファイルに約 145 行ある。正規化の設定キーの読み取りは 13 行ある | `git grep`（`src`、2026-11-21 に確認）、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 532〜544 行 | 対象外 |
| P129 | 既存部品の働き | `EvalRunner` の行動の引数を持たない口は、step 数だけを受けるものと、`MakeActionOptions` だけを受けるものの 2 つである。`TensorSpec` の JSON は書き出しだけで、読み戻す関数は無い。メトリクスの鍵 `obs_norm.*` は、Agent が自分の持つ正規化の部品から読んでいる | [trainer.hpp](../../src/core/base/include/anet/trainer.hpp) 137〜140 行、[common.hpp](../../src/core/base/include/anet/common.hpp) 156 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 695〜698・717〜720 行 | 対象外 |
| P130 | 既存部品の働き | NN の構造図は、Network が dot の図を作り、Agent が `net.structure`（構造だけ）と `net.detail`（設定 `nn_viz.*` に従った詳しい図）の 2 つを記録する。図は Run の `dot/` の下に出る | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の `RecordNetworkStructure`（2026-11-22 の main で 627〜643 行）、[nn.hpp](../../src/core/base/include/anet/nn.hpp) の `MakeGraphViz`、[graphviz.hpp](../../src/core/base/include/anet/graphviz.hpp)、[140_observability.jp.md](../design/140_observability.jp.md) の成果物の表 | 対象外 |
| P131 | 既存部品の働き | NN の `structure` は、`>` を空白に置き換えてから空白で区切って読む。空の要素は残らないので、末尾の `>` は無いものとして読まれる | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) の `SplitPipelineString`（2026-11-22 の main で 553〜572 行） | 対象外 |
| P132 | 既存部品の働き | 等価性の比較の道具は、`scalar` の値を 16 進の文字列にし、tag・step・値の組の集まりとして前後を比べる。値が NaN どうし、値なしどうしは一致になり、片方にだけあるレコードは不一致になる。両方に無い tag は不一致にならない。出力に、tag ごとの件数（`counts`）と、全体の点数（`points`）を出す | [compare_metrics.py](../../src/core/base/testdata/rfd0097/compare_metrics.py)（2026-11-22 の main で 24〜47・70〜77 行） | 対象外 |
| P133 | 事実 | 出荷の設定ファイルの `run.$` の有効な行は、LunarLander が `run.@munch > run.@skip`、Atari が 164 行の並びで、GridMaze と DropMerge には有効な行が無い（コメントの行だけ）。4 つとも、メトリクスの定義に `metrics.scalar.@baseline` を使い、`37_agent_qtd/01_td_mean` と `20_eps/10_train_reward` はその中にある | [LunarLander.txt](../../apps/runner/config/LunarLander.txt) 28・441 行、[Atari.txt](../../apps/runner/config/Atari.txt) 164・624 行、[GridMaze.txt](../../apps/runner/config/GridMaze.txt) 22・23・222 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 104〜108・1356 行、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) 16・101 行（どれも 2026-11-22 の main の行番号） | 対象外 |
| P134 | 既存部品の働き | 今の観測の正規化は、`pass_through` が真なら素通しで、どの設定キーも使わない。偽で `use_dynamic_scaling` が偽なら、定数の部品が `constant_mean`・`constant_std`・`use_clipping`・`clip_range` を使う。偽で `use_dynamic_scaling` が真なら、統計つきの部品が `use_centering`・`epsilon`・`use_robust_update`・`robust_warmup_count`・`robust_std_threshold`・`post_process_type`・`post_process_threshold`・`use_clipping`・`clip_range` を使う。値域の検査は 1 つの関数にある。構造体の既定は、`pass_through` が偽、`use_dynamic_scaling` が真である | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 262〜290・721〜742・762〜777 行、[scaler.hpp](../../src/core/base/include/anet/scaler.hpp) 47〜67 行（2026-11-22 の main の行番号） | 対象外 |
| P135 | 守る線 | モードで使う設定キーが変わるまとまりは、スコープに `mode` を置き、設定キーをモードの名前空間（`<スコープ>.<モード>.<キー>`）に持つ。沿わないまとまりの設定キーを足す・変えるときは、同じ変更で移す。特定の値に別の意味を持たせず、意図の分岐はモードの値で書く。選ばれていないモードの設定キーは、型と値域を検査し、知らせは出さない | [ADR 0092](../adr/0092-mode-dependent-config-keys-live-in-per-mode-namespaces-under-concern-scopes.md)、[ADR 0091](../adr/0091-config-keys-idle-under-chosen-mode-or-intended-pause-are-range-checked-not-notified.md)、[impl.md](../quality/impl.md) の「名前の付け方」、[AGENTS.md](../../AGENTS.md) §設定値の扱い | 対象外 |
| P136 | 事実 | 同じ形の設定が RND にある（`intrinsic.rnd.error_norm.mode` と、`error_norm.standardize.*`・`error_norm.none.*`）。メトリクスの鍵 `obs_norm.*` を記録する定義は `metrics.scalar.@full` だけにあり、出荷設定に `@full` を選ぶ有効な行は無い | [agent.txt](../../apps/runner/config/agent.txt) 47〜56 行、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) 152〜154・301〜305 行（2026-11-22 の main の行番号）、`apps` を `scalar.@full` で検索（2026-11-22 に確認。有効な行は 0） | 対象外 |
| P137 | 事実 | `nn.hpp` を読むファイルは `src` に 16 ある。NN は、公開の口の `nn`、組み立ての `nn_impl`、module の `nn_modules` などの組に分かれている。機能カテゴリは、ファイル名の規則で決まる（314 の行は `tensor_util`・`tensor_check`・`tensor_store`・`transfer`・`stacker`・`scaler`・`scaler_impl`・`image`） | `src` を `anet/nn.hpp` の `#include` で検索（2026-11-22 に確認）、[categories.json](../../reports/tools/categories.json) | 対象外 |
| P138 | 事実 | 公開のプロジェクトは、基底と部品のファイルを分けている。PyTorch の `torch/nn/modules/` は `module.py`・`container.py` と個々の module、TorchRL の `torchrl/envs/transforms/` は `_base.py` と種類ごとのファイル、Gymnasium の `gymnasium/wrappers/` は種類ごとのファイルである | GitHub のファイルの一覧（3 つとも 2026-11-22 に確認。見たのはファイル名で、中身の分担は見ていない） | 対象外 |
| P139 | 既存部品の働き | 今の観測の正規化は、float の dtype で、離散でないキーにだけ部品を作る。部品が 1 つも無いと、メトリクスの鍵は値を返さない。素通しの部品は `count` などに NaN を返し、全部が NaN のとき、集計は 0 を返す。ReplayBuffer は、重ねるキーの一覧が空のとき、全キーを重ねる | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 311〜319・687〜710・756〜763 行、[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 706〜711 行（2026-11-22 の main の行番号） | 対象外 |
| P140 | 既存部品の働き | 観測の spec の検査（`StateSpec::AssertSanity`）は、キーが 1 つも無い spec を拒否しない。設定に書いた空の一覧は、空の一覧として読まれる（基準の設定が `stucker.stack_keys ?=` と書き、Agent は、それを空の一覧として扱っている）。定数の正規化の部品は、空の一覧を、平均 0・標準偏差 1 として組み、空でない一覧は、キーの形に並べ直す（要素数が合わないと、構築で失敗する）。統計つきの部品は、入力の次元の数が「batch ＋ 1 Frame 分の形」と違うと止まる | [rl.cpp](../../src/core/base/src/rl.cpp) 74〜95 行、[agent.txt](../../apps/runner/config/agent.txt) 195 行、[scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 236〜254・361〜372 行（2026-11-22 の main の行番号） | 対象外 |
| P141 | 事実 | LunarLander の出荷時のバックエンドでは、同じ実行体を同じ seed で繰り返しても学習リターンが一致しなかった。決定的なバックエンドを変更前後に指定した比較では、§9 の比較対象のメトリクスがすべて一致した | 段階 A の [実装記録](0161_observation_pipeline_20impl.md#経過)（2026-11-23） | 済み |

## 5. issue の対応表

| ID | 題 | area | 今のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-aa5758](../management/issues/issue-20261103-aa5758.md) | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 | 321_agent_core、318_rl_core、324_dqn、314_tensor、325_muzero | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1534・1541 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 424・964 行、[trainer.cpp](../../src/core/base/src/trainer.cpp) 293 行 | D7〜D16（強制行動）、D17〜D24・D45（基盤）、D25〜D34・D39〜D44（Perception の仕組み） | 1〜34 |
| [issue-20261103-b754ff](../management/issues/issue-20261103-b754ff.md) | `stucker.stack_keys` に存在しないキー名を書いても検出されない | 324_dqn | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 435 行、[stacker.cpp](../../src/core/base/src/stacker.cpp) 56 行 | D30 | 21 |

課題 aa5758 の 2026-11-11 の追記（正規化の部品の内部の 3 点）と、RND の観測加工は、本 RFD の範囲に入れない（NG3・NG4）。§15 のとおり、課題に記録した。

## 6. 確定事項

「依存」は、理由がよりどころにする前提（P）と、ほかの決定（D）を挙げる。発言の日付と原文は §13 にある。

### 6.1 全体

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D1 | 語を 3 つに分ける。**Observation** は Env が Agent へ渡すもの。**Frame** は ReplayBuffer に保存する、観測側の 1 step 分。**Perception（知覚）** は、Agent が Observation と自分の持つ関連情報から作る、Network へ渡す入力で、taus を含めない。Perception は、渡す先の Network ごとにある（Q の Network、skip の補助 Network、RND の Network は、それぞれ別の入力を受ける）。本 RFD が並びで作るのは、DefaultDQN の Q の Network（primary）へ渡す Perception で、以下、Network の名を添えない Perception はこれを指す。Frame と Perception の定義は互いを参照しない。「Frame をもとに Perception を作る」という依存は、フレームワークの前提にしない | 「生か加工済みか」では線が引けない（Atari の Observation は Env の中で縮小を済ませている）ので、誰が形を決めるかで分ける。直線の依存を前提にすると、保存するが Network に入れない情報や、別の方策番号として学ぶ形を塞ぐ | P27、P32、P33、P42 |
| D2 | Agent が足す情報（今回は無い。RFD0142 の直前の行動など）は、行動選択のときに、Actor の実行器が 1 回だけ作る。作った値は、Perception の一部としてその場で Network へ渡し、あわせて行動の情報（`BatchActionInfo`）に載せて経験の受入れへ運び、Frame の一部として ReplayBuffer に入れる。受入れの側で、同じ値をもう一度作ることはしない。Perception そのものは運ばない。Learner の側の Perception は、Sample のときに、保存した Frame から作る（行動選択のほかに Perception を作るのは、Sample と可視化だけである）。今回は足す情報が無いので、Actor から受入れへ運ぶ観測側の値は無く、Frame は経験の Observation と同じ中身である。lane の履歴は、stack の部品ではなく、Actor ごとの実行器が持つ | Actor が作った値を行動の情報に載せて ReplayBuffer が使う道が既にある。行動選択の時点でその観測は ReplayBuffer に無く、評価用の Actor は ReplayBuffer に入れないので、ReplayBuffer の末尾を履歴に使えない。D7 で経験がすべて Actor を通るので、「Actor を通らない経験でも受入れ側だけで Frame が完成する」利点は効かない | P37、P38、P39、D7 |
| D3 | Perception に「何の情報が入るか」はアルゴリズム設定が決めて実装が作り、「どういう形で入るか」だけを設定の並びで組む | NN が Head と taus を実装で作り、`net.*` が Body の配線だけを持つのと同じ線。全部を並びで書く形は、機能は有効なのに並びにその入力が無い、という食い違いが書ける | P28、P29、P31 |
| D4 | 行動選択側と Sample 側の Perception が往復で一致するかは、Frame にキーを足す部品については RFD0142 で確かめる。本 RFD の受入には、stack と正規化だけの往復のテストを入れる | Frame にキーを足す部品は本 RFD に無い。今は、Actor 側の stacker の出力と ReplayBuffer の Sample の出力を直接突き合わせるテストが無い | P36、D34 |
| D5 | RFD は本 RFD の 1 本のままとし、段階を A（強制行動）、C（基盤）、B（Perception の仕組み）の順に分ける。課題の結び付けと、ほかの RFD からのリンクは変えない。題とファイル名も変えない | 後続は急ぎではなく、グリルの作業を 1 回で済ませる。共通の口（C）を、使う側（B）と一緒に決められる。実装の順は、B の運び方（D2）が A をよりどころにするので、A、C、B になる。題とファイル名を変えないのは、待っている RFD からのリンクを保つためである | P76、P78、P80、D2 |
| D6 | 共通の口は順方向だけとし、逆方向（入れるとき用）の変換の口は持たない。ReplayBuffer には加工前の Frame を入れ、読む側がどれも同じ並びを順に当てる。見直しの条件は、ReplayBuffer に入れる形を加工する要求か、行動の側を並びで加工する要求が出たとき | 入れるときに加工しないので、戻す処理が要らない。今回の部品（重ねる、統計つきの正規化、キーを落とす）は、どれも元に戻せない加工で、逆方向の口を足しても何もしない口になる。あとから既定つきの口を足せば、既存の部品は変えずに済む | P70、P71、P73 |

### 6.2 単位 A（強制行動）

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D7 | 強制行動は、`MakeActionOptions` の欄 `forced_actions` として `MakeAction` へ渡す。3 つ目の引数にはしない。値なしが「強制なし」を表す | `MakeAction` の追加の入力は、呼び出しごとに構造体で渡すと決まっている | P6、P85 |
| D8 | Runner に強制行動を渡す口は 1 つにする。行動の引数を持つ `EvalRunner::DoStep` の 2 つを消す。今は、step 数だけを受ける口と、`MakeActionOptions` だけを受ける口が別々にあるので、両方を受ける口を 1 つ足す（`EvalPanel` の強制の経路が step 数も渡しているため）。呼ぶ側が欄に強制行動を入れて渡す。負数が「強制なし」を表す決まりは無くなる | 引数と欄の両方に強制があると食い違いうる。timer の経路（`DoUpdateFrame`）も同じ構造体を受けるので、同じ欄で渡せる | P82、P83、P84、D7 |
| D9 | 欄の型は `ForcedActions` の 1 つで、作り方を 2 つ持つ。`ForcedActions::ByLane` は lane の番号と行動で 1 lane を指す。`ForcedActions::ByMask` は lane 数ぶんの印と行動でまとめて指す。差し替えと検査は型の中の 1 か所で行い、Actor は作り方の違いを見ない | 指定のほとんどは 1 lane なので、lane 数を知らずに書ける作り方が要る。複数 lane をまとめて指す形も塞がない。番号で指す口と印で指す口の対は、Isaac Lab と Stable-Baselines3 に事例がある | P87、P89、P90、P91 |
| D10 | 検査は `ForcedActions` の中の 1 か所で行う。`ByLane` は、lane の番号が 0 以上 lane 数未満、行動が 0 以上 行動数未満であること。`ByMask` は、印が bool、行動が int64 の Tensor で、どちらも形が `[lane 数]` であること、印が真の lane の行動が 0 以上 行動数未満であること。印が偽の lane の行動の値は読まず、検査もしない（どんな値が入っていてもよい）。違反はエラーで止め、文言に lane・指定した値・行動数を入れる。連続行動の Env に強制行動を渡したときもエラーで止める。`EvalRunner` は、Reset より前にこの検査を呼ぶ。Actor も、差し替える直前に同じ検査を呼ぶ | 今の「Reset・Actor・Env より前に拒否する」を保つ。今は、連続行動の Env では範囲の検査を飛ばして通している。Train Runner を通る強制は Actor の検査で止まる。強制するかを表すのは印だけなので、印が偽の lane の値には意味を持たせない（値で「強制なし」を表す決まりを無くした D8 と同じ線） | P82、D8、D9、D15 |
| D11 | 方策の処理（forward、探索の乱数、診断の集計）は、全 lane で今のとおり行い、そのあと強制した lane の行動を差し替える。強制していない lane の結果と乱数は、強制の有無で変わらない。行動から作る値（Actor Q ヒントの行動の Q、Munchausen の項、QR・IQN の tail の診断）は、差し替えたあとの行動から作る。値は今の `WithAction` と同じになる | 乱数の消費順を変えない線。DQN の Actor は、行動を選んだあとにその行動からヒントを作るので、その間で差し替えれば、ヒントはそのまま最終の行動を読む | P11、P88 |
| D12 | `BatchActionInfo::WithAction` と、それを上書きする 2 つ（DQN、MuZero）、テスト用の差し替え、aux の `munchausen_terms` を消す。RFD0192 が足した `WithAction` のテストも消し、`ForcedActions` 用に書き直す | 行動を決める点を Actor の中の 1 つにする。`munchausen_terms` を読むのは `WithAction` だけである | P9、P77、P117、D11 |
| D13 | TempoRL の skip が有効な Actor では、強制した lane の継続状態（続けている行動と残りの step）を捨て、強制した step を新しい判断として記録する。強制した lane は、判断のフラグを真、skip 長を 1、残りの step を 0、続ける行動を強制行動にする。次の step は、方策が判断し直す。skip の診断（判断した lane の数と skip 長の分布）は、強制を反映したあとの値で数える。Train につなぐときではなく、今回入れる | Train Runner も同じ構造体を Actor へ渡すので、口は Train 側にも開く。区間の途中で行動が変わると、ReplayBuffer の窓が Sample のときに止まる | P8、D15 |
| D14 | 強制行動に対応できない Actor は、強制した lane があるときにエラーで止める。今の 3 つの Actor はすべて対応する。MuZero と ImageCls は行動の差し替えだけにし、MuZero の探索木と info は探索の結果のままにする（今と同じ）。MuZero を Train で強制する形は RFD0160 で決める | 黙って無視しない（Fail-Fast）。MuZero の今の `WithAction` が行動だけを差し替えている | [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則、[muzero_based_agent.cpp](../../src/algo/muzero/src/muzero_based_agent.cpp) の `MuZeroActionInfo::WithAction` |
| D15 | Train Runner は、受けた `MakeActionOptions` をそのまま Actor へ渡す（今と同じ）。人の操作を Train へ届ける配線は入れない | 口は、今も Train Runner が構造体をそのまま渡す形で開いている。人の操作を届ける配線は別の関心で、RFD0160 が決める（NG8） | P85 |
| D16 | Runner の「強制したか」の記録は、`ForcedActions` の lane の印から作る。EvalPanel の範囲外の入力は、今のまま、ステータスバーに出して何もしない | 今は全 lane を同じ値で埋めていて、一部の lane だけの強制を表せない。範囲外の入力の見せ方は、強制行動を渡す口とは別の関心で、変える理由が無い | P7、P86、D9 |

### 6.3 単位 C（基盤）

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D17 | `TensorSpec` に値の尺度の属性 `scale` を足す。値は `ValueScale::Scaled`（値の尺度が決まっている）と `ValueScale::Unscaled`（決まっていない）の 2 つで、既定は `Scaled`。`Scaled` は、キーを作った側か正規化が尺度を決めていて、観測の正規化でそろえる対象にしないことだけを表す。値に上下限があることも、0〜1 に直してあることも表さない。byte の画像と番号のキーも `Scaled` に入る（dtype と番号の定義が尺度を決めている）。整数の dtype か、離散のキーが `Unscaled` なら、spec の検査でエラーにする。正規化の対象は `Unscaled` のキーで、素通しでない正規化は、クリップや後処理の設定によらず、通したキーを `Scaled` に変える（その正規化が尺度を決めたため）。キーの性質は、注入する側（Env、情報を足す部品）が spec に宣言する | 方策番号や直前の行動のように、性質として正規化の対象でないキーを、設定に書かせずに外す。連続値のうち正規化しないもの（Env が 0〜1 にしている値）も表せる。既定を `Scaled` にすると、宣言の無いキーが正規化の対象にならない。値の種類を 4 つにする案は、その区別を使う箇所が無い。加工ごとのフラグは、spec が加工の名前を知ることになる | P45、P46、P48、P49、P62、D3 |
| D18 | Env の宣言は、キーの性質で決める。LunarLander・CartPole の float の vector は `Unscaled`。GridMaze（Env が 0〜1 に直している）、DropMerge（float の vector が番号を運び、NN の module が one-hot にする）、Atari（byte）は `Scaled`。テスト用の spec で正規化を確かめているものは、宣言を足す | 属性は性質で宣言する（D17）。素通しでない正規化を使うと、今は GridMaze と DropMerge の vector も対象になるが、性質としては対象でない。出荷設定は素通しなので、動きは変わらない | P62、P63、P107、P126、D17 |
| D19 | `TensorSpec` に、stack 軸を持つかの属性 `has_stack_axis`（真偽。既定は偽）を足す。stack の対象は軸を持たないキーだけで、stack の部品が「持つ」に変える。spec の JSON に `scale` と `stack_axis` の欄を足す | 重ねるかどうかは形の選択で設定が決め、重ね済みかはデータの状態で spec が持つ。`TensorSpec` は Perception だけの型ではない。重ねる前の spec（Observation、Frame）と、Perception を通らない Network の入力（ImageCls、MuZero、RND、テスト）には軸が無く、NN の入口と正規化が、次元の数から推し量らずに見分ける。語は、設計文書の「stack 軸」を使う | P53、P64、P65、P118、D27 |
| D20 | 「TensorDict を受けて返す変換」の共通の口 `Forwarder` を、基盤に置く。Forwarder は、lane の履歴のような、呼ぶ側ごとの状態を持たない。重みや統計のような、どの呼ぶ側にも共通の値は持ってよい（Network の重み、`Normalize` の統計。D40）。口は `Forward`（TensorDict を受けて返す。capture を任意で受ける）の 1 つだけにする。通したあとの spec も答える口 `ForwardSpec` は、`Forwarder` を広げた `SpecForwarder` が持つ。Network は今回 `Forwarder` を実装し、今の `Network::Forward` がそのまま共通の口になる。観測の加工の部品は `SpecForwarder` を実装し、並びに置けるのは `SpecForwarder` だけである。呼ぶ側ごとの状態を使う実行は、今と同じく実行器が持つ。可視化が NN を呼ぶ型（`TensorDictFunction`）は変えず、共通の口からこの型の関数を作って渡す（可視化の口を今回の範囲に入れないための線で、移せない制約ではない） | 可視化や Learner が、Perception の変換と Network を同じ型でつなげる。名前のとおり `Forward` だけを持つ口にすると、Network は今の形のまま実装できる。spec が要るのは、並びを組むときの部品だけである。spec を通す口の名前は、データを通す `Forward` と対にする（Transform は、用語集が避ける語である）。capture なしの口を別に足すと、capture を渡さない呼び出しが、今の `Forward` と見分けられなくなる | P17、P54、P92、P112、[ADR 0002](../adr/0002-tensordict-function-unify.md)、D6 |
| D21 | `NetworkForwardCapture` を `ForwardCapture` に改名し、`Forwarder` と同じ基盤のヘッダへ移す。鍵で要求する決まりと、満たされない要求を Network の出口で止める検査は変えない。今回の 3 つの部品は capture に何も書かない。部品が書く鍵は、使う側が出てきたときに決める | forward の持ち帰りは 1 つの器に保つ、と決まっている。Forwarder にあとから別の器を足すと、これに反する。今回の目的は、名前と置き場を共通の口に合わせることだけなので、鍵の決まりと検査は変えない。渡さないときは判定だけで、速度とメモリは変わらない見立て（測ってはいない） | P112、D20 |
| D22 | Network は `SpecForwarder` を実装せず、出力の spec を持たない。skip の補助 Network の入力 spec の作り方は、今のままにする。見直しの条件は、Network の出力の spec を使う部品が出たとき、または、並びと Network を 1 本の `SpecForwarder` の並びとして組む要求が出たとき。そのときは、出力の spec を、eval の状態・勾配なしの仮の forward から導く案を起点にする | ゴールのどれも、Network の出力の spec を必要としない。出力の spec を導くには Head まで仮の forward を通すことになり、乱数を引く module が Head にあると、同 seed の等価性が崩れうる。今、Head まで通しているのは、skip が有効な構成だけである | P11、P57、P59、P124、D20 |
| D23 | seed を、NN と同じ 2 か所で受ける。並びを組む口（block の名前から block ごとの seed を作る）と、行動選択の実行器を作る口。今回の 3 つの部品は seed を使わない。乱数を使う部品は、渡された seed から作った自前の乱数だけを使う | あとで乱数を使う部品（増強など）を足すとき、口を変えずに済む。Network も、使わない既定の設定で seed を受けている。名前から seed を作る計算は乱数を消費しないので、今の乱数の並びは変わらない | P11、P94、P113、D26 |
| D24 | `Forwarder`・`SpecForwarder`、並びの仕組み、3 つの部品（重ねる、正規化、キーを落とす）は、基盤（`src/core/base`）に置く。DefaultDQN がそれを使う。Rainbow・MuZero・ImageCls は使わない | 今の stacker と正規化の部品が基盤にある。NG7 | D20 |
| D45 | 置き場の無い新しいクラス 10 個（口、器、並びの仕組み、3 つの部品）は、新しく立てる機能グループ `forwarder` の 1 つに置く（公開ヘッダ `anet/forwarder.hpp`）。機能カテゴリは 314 テンソル処理に入れる。受け持つ関心、入るもの、入れないものは、§7.1 の「機能グループの見込み」のとおり | 開発者の選択（2026-11-22）。Forwarder を使うときも作るときも `anet/forwarder.hpp` を読めば足りる、がフレームワークの方針である。口・並びの仕組み・部品を 2 つか 3 つの組に分ける案（NN の `nn`・`nn_impl`・`nn_modules` と同じ分け方）は、この方針に合わない | P137、P138、D20、D24 |

### 6.4 単位 B（Perception の仕組み）

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D25 | Forwarder の並びを、NN と同じ書き方で設定に書く。持ち主が渡した接頭辞の下の `block.[名前].type` が部品、`structure` が並びである。DefaultDQN の接頭辞は `perception` とする。今の設定キー `stucker.*`・`obs_norm.*` は、部品の設定へ移す（§7.5）。モードで使うキーが変わるまとまりは、`mode` とモードの名前空間に分ける（D43・D44）。旧いキーに別名や互換の読みは残さない。旧いキーを書くと、読まれないキーとして起動時の WARN の一覧に出る。`stucker.use_stacker` は無くなる。メトリクスの鍵 `obs_norm.*` は変えない | 汎用の部品は、置く位置を設定で書けないと使えない。書き方を NN と 1 つにそろえる。`perception` は「この並びが作るもの」の名前で、並びの仕組みのコードには出ない。メトリクスの鍵は設定キーではなく、Run の比較とメトリクスの定義が使っている | P75、P111、P114、P119、[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針、D3 |
| D26 | 部品は `SpecForwarder` を実装する。依頼によって動きが違う部品だけ、足して `ForwarderProvider` を実装し、依頼ごとの Forwarder と、行動選択用の実行器 `ForwardExecutor` を返す。並びへの依頼 `ForwardRequest` は、何を渡して何が欲しいかで名付けた 3 つとする。`FrameWithStatsUpdate`（観測から、統計を更新しつつ、保存する Frame を作る）、`PerceptionFromStack`（重ねた束から Perception を作る）、`PerceptionFromFrame`（1 枚の Frame から、履歴なしで Perception を作る）。1 枚の Frame と lane の履歴から Perception を作る依頼は、値にせず、実行器を作る口で受ける。呼ぶ側は、依頼を選んでその依頼用の Forwarder を受け取り、どの依頼でも `Forward` を呼ぶ。並びを組む仕組みは `ForwarderChain`。部品の種類（設定の `type`）は `FrameStack`・`Normalize`・`DropKeys` で、クラス名は `FrameStackForwarder` のように付ける。部品に専用の基底の型は置かない。`ForwarderProvider` を実装しない部品は、Perception を作る依頼では自分の `Forward` を当て、`FrameWithStatsUpdate` の依頼では何もしない（`DropKeys` がこれに当たる）。今回 `ForwarderProvider` を実装するのは `FrameStack` と `Normalize` | NN の `Network` と実行器と同じ二段。重ねる動きは依頼によって違い、状態なしの変換 1 つに決まらない。選ぶ値は、呼ぶ側の状況（行動選択・push・Sample・可視化）ではなく、頼む中身で名付ける。push するのは呼ぶ側で、並びに頼むのは「Frame を作る」である。Forwarder の集まりで作るのは Perception だけではないので、仕組みの側の名前に Perception を付けない | P18、P92、P93、P110、D2、D20 |
| D27 | stack の対象のキーは、必ず stack 軸を持つ（`[B, S, *shape]`、S は 1 以上）。枚数の既定は 1 で、重ねないときは 1 枚になる。stack 軸の無い構成は作れない。並びに `FrameStack` を書かなければ（並びそのものを書かない場合を含む）、エラーにせず、先頭に 1 枚の `FrameStack` を置いて組む。`FrameStack` を書くなら、並びの先頭に限る。対象のキーは、部品の設定の `keys.mode` で決める。`all`（既定）は全キー、`listed` は一覧に書いたキーで、構築時に spec と照らす（D44）。並びの入力の spec に、既に stack 軸を持つキーがあれば、構築時にエラーにする（Env が自分で重ねた観測は、今回は扱わない。扱うのは、そういう Env を足すとき）。並びが保存する側へ答えるキーの一覧は、決めたキーの名前を並べたもので、空にはならない（ReplayBuffer は、空の一覧を全キーと読む。並びの入力の spec にキーが 1 つも無い構成も、構築時にエラーにする）。ReplayBuffer と `DictFrameStacker` の契約は変えない（NG5）。並びは、保存する側に求める重ね方（枚数と、決めたキー）を答え、Learner がそれを ReplayBuffer の設定へ渡す。行動選択では今の stacker を実行器の中で使い、Sample では ReplayBuffer が重ねた束を受けて、1 枚のときだけ長さ 1 の軸を足し、可視化では 1 Frame を枚数ぶん複製する | stack の有無で形の意味が変わらない。stack 軸の扱いを曖昧にしない。枚数だけ変えて比べる Run が、設定の 1 行で済む。PyTorch の LSTM と Gymnasium の frame stack が、同じ流儀である。今は Learner が stack の設定を ReplayBuffer の設定へ写していて、その設定の型を消すので、並びが答える | P44、P95、P97、P101、P122、D19 |
| D28 | 素通しでない正規化は、1 Frame 分の統計を stack 軸へ広げて当てる。形の検査は、末尾の次元の一致で行う。旧ドラフトの「stack ありで素通しでない正規化を構築時に止める」は外す。正規化の部品の中身（統計の持ち方、読み書きの競合、checkpoint）は変えない | D27 で、正規化は必ず軸つきの入力を受ける。止める案は、「stack なしなら使える」が前提だった | P106、P109、D27、NG2 |
| D29 | `DropKeys` は、指定したキーを Perception から落とす。`keys` は、落とすキーの一覧で、必須である（モードを持たないので、スコープにしない）。spec に無いキーはエラー。全部のキーを落とす指定もエラー。Frame には残し、落とすのは Perception の側だけである | 汎用の部品の最初の 1 つ。黙って何もしない設定を作らない | D25 |
| D30 | 構築時にエラーで止めるものは、§7.6 の表の 1〜9 を正本とする。並びを書かない設定と、`count` を書かない `FrameStack` は、エラーにしない（D27） | 成り立たない設定を、最初の行動選択まで潜ませない。block の中の読まれないキーは、NN の block と同じ扱いにする | P3、P75、D27 |
| D31 | 4 つの口の役割を §7.4 のとおりにする。行動選択: Actor が実行器を 1 つ持ち、Observation と episode の開始の印を渡して Perception を受ける。経験の受入れ（push）: `FrameWithStatsUpdate` の Forwarder に経験の obs と next_obs を通し（今回、起きるのは統計の更新だけ。D40）、返った Frame（重ねも正規化もしていない）を ReplayBuffer へ渡す。Sample: Sample した観測のすべて（obs、next_obs、skip の学習の窓の終点の観測）に、`PerceptionFromStack` の Forwarder を当てる。可視化: `PerceptionFromFrame` の Forwarder と Network をつないで、`TensorDictFunction` を作る | D2・D6・D26 の帰結。Sample の側の正規化は今 3 か所ある | P72、D2、D6、D26 |
| D32 | NN の側は 3 点を直す。(1) stack 軸を持つ入力を受ける branch の先頭が、「stack 軸つきの入力を扱う」と宣言した module でなければ、構築時に WARN を 1 回出して続ける。宣言は module の型ごとに持ち（今回は `Flatten`・`StackMerge`・`Permute`・`Reshape` と、軸のあるなしを自分で見分ける `SpatialEmbedder`・`HybridSpatialEmbedder`）、検査の側に一覧を持たない。(2) `StackMerge` は、5 次元以外の入力をエラーにする。(3) 入口の離散の Grid は、次元の数ではなく、属性で stack 軸を見分ける。畳む動きは今のままにする（そろえるかは、§15 の課題）。NN の入口で既定で畳む仕掛けは入れず、先頭の module で軸の扱いを書く決まり（RFD0057）は変えない | 軸の扱いを曖昧にしないが、書き方は強制しない。一律のエラーは、軸を自分で扱う新しい module を塞ぐ。WARN は利用者が確かめて直す状態に限るので、正しい設計は module の宣言で WARN を消せるようにする。先頭が `Linear` のときは、書き忘れが構築で止まらない見込みである | P96、P102、P103、P114、P115、D19、D27 |
| D33 | RND の正規化の判定は変えない | RND は byte の画像を float に直して自前の統計で正規化していて、尺度の属性に合わせると Atari の動きが変わる | P68、NG3 |
| D34 | 往復のテストを入れる。行動選択側の Perception と、同じ経験を ReplayBuffer に入れて Sample で作り直した Perception が一致することを、4 枚と 1 枚、episode の先頭、打ち切りの終端、n より手前で episode が終わる next_obs、skip の窓の終点で確かめる | D4。今は、2 つの出力を直接突き合わせるテストが無い | P26、P41、D4 |
| D39 | Frame の spec は、Agent が 1 つ作る（Observation の spec に、実装が足すキーを合わせたもの。今回は足すキーが無い）。ReplayBuffer の生成と、並びの入力の両方に、この 1 つを渡す | RFD0142 が Frame にキーを足すとき、直す spec が 1 か所で済む（G6）。今は、ReplayBuffer の保存領域を Env の spec から作り、Network の入力 spec は別の関数が作っている | P50、P123、D1 |
| D40 | 観測の正規化の統計（ObservationNormalizer の平均と分散）は、`Normalize` の部品が 1 組持つ。更新するのは `FrameWithStatsUpdate` の依頼だけで、そのとき観測の値は変えずに返す。Perception を作る依頼（実行器を含む）は、統計を読んで当てるだけにする。Agent は、経験を受け取ったとき、`FrameWithStatsUpdate` の Forwarder に obs と next_obs を 1 回ずつ通す（今と同じ 2 回）。`Normalize` は並びに 1 つまでとする。メトリクスの鍵 `obs_norm.*` は、並びが部品へ問い合わせを回して答える。報酬の RewardScaler と、RND の観測の正規化は、今のまま Agent と RND が持つ。正規化の対象のキー（`Unscaled`）が 1 つも無い構成では、`Normalize` は並びに居ても何もせず、メトリクスの鍵 `obs_norm.*` は値を返さない（今の、float の連続値のキーが無い構成と同じ動き）。D18 の宣言で、GridMaze と DropMerge はこの構成になる（今は、素通しの部品が作られ、`obs_norm.count` が 0 を返す） | 依頼ごとの動きを部品が持つ形（D26）に収まる。統計の更新は、Frame を作ることに付随する処理なので、依頼の名前に `With` で表す（`And` だと、作るものが 2 つあるように読める）。Agent が部品の種類を知らずに済み、設定キーを部品へ移す決定（D25）も保てる。動きは今と同じである。RewardScaler と RND は観測の加工ではないので、並びに入れない。対象のキーが無い構成で `obs_norm.*` の返す値が変わる点は、出荷設定に、この鍵を記録する有効な定義が無いので、等価性の Run に現れない | P129、P136、P139、D1、D18、D25、D26、NG3 |
| D41 | 並びの図（dot）を、NN の構造図と同じ仕組みで出す。`ForwarderChain` が図を作る口を持ち、`DefaultDQNAgent` が、Network の構造図を記録するのと同じ時機に、`perception.structure`（部品と順番だけ）と `perception.detail`（各段の spec と、部品の設定を足したもの）の 2 つを記録する。図は Run の `dot/` の下に出る。設定キーは足さない（§7.12） | 並びの中身（部品、順番、各段の形と属性）を、Run の記録で確かめられる。NN は、構造だけの図と詳しい図の 2 つを、同じ形で出している | P130、D25、D26 |
| D42 | Actor が aux に載せている `raw_obs`（重ねたあと・正規化の前の観測）と `norm_obs`（正規化のあとの観測）を消す。途中の値が要るようになったら、capture の鍵として足す | 読むのはテストだけである。行動選択用の実行器が返すのは、最後の Perception だけになる。形の確かめは、Perception を直接見るテスト（受入 23）に移す | P127、D21、D31 |
| D43 | `Normalize` の設定キーは、`mode`（`none`・`constant`・`running`）と、モードの名前空間（`constant.*`・`running.*`）に分ける。`pass_through` と `use_dynamic_scaling` は無くなる。`mode` の既定は `running`（今の構造体の既定と同じ動き）で、基準の設定が `none` を置く。モードの名前空間のキーは、今の同じ名前のキーの型・値域・既定を引き継ぐ。選ばれていないモードの設定キーも、読んで型と値域を検査する（§7.5） | 開発者の選択（2026-11-22）。今のキーは、2 つのフラグで 3 通りに動き、通りごとに読むキーが違う。キーを全部移すので、モードで使うキーが変わるまとまりを移すときの決まりに当たる。RND の `error_norm` と同じ形になる。素通しでも部品が並びに居るので、正規化の対象のキーがある構成では、メトリクスの鍵 `obs_norm.*` の動きが変わらない（対象のキーが無い構成は D40）。部品の種類で分ける案は、統計を持つ部品が 2 種類になり、「正規化は並びに 1 つまで」が種類をまたぐ | P134、P135、P136、D25、D40 |
| D44 | `FrameStack` の対象のキーは、`keys.mode` で選ぶ。`all`（既定）は全キー、`listed` は `keys.listed.names` に書いたキーだけである。空の一覧に「全キー」の意味は持たせない。`listed` で一覧が空ならエラーにする。`all` のとき、`keys.listed.names` は型だけを検査し、spec とは突き合わせない | 特定の値（空の一覧）に別の意味を持たせる形は、設定値の扱いの決まりに反する。意図の分岐は、モードの値で書く。形は D43 とそろえる。スコープの名前 `keys` と、一覧の名前 `names` は、開発者が選んだ（2026-11-22。AI が初めに置いた `target` は、target network の階層の名前と紛れる）。`DropKeys` の `keys` は一覧で、`FrameStack` の `keys` は下に `mode` を持つスコープである。どちらも、その部品が扱う観測キーを指す。モードの値 `all`・`listed` は、AI が置いた（§13） | P44、P135、D27、D43 |

### 6.5 確かめ方・文書・順序

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D35 | 等価性は、段階 A・C・B のそれぞれの前後で、LunarLander と GridMaze の同 seed の短い Run の metrics が一致することで確かめる。段階 C と B では、DropMerge の学習 Run と、Atari の Q*bert の学習 Run も 1 本ずつ足す。全環境の比較で前後とも決定的なバックエンドを指定する（§9）。Atari は学習 Run で確かめる（U4）。checkpoint の raw checksum は合否に使わない | 重ねる → 転送 → 正規化の並びと、乱数の消費順を変えない線。LunarLander は 4 枚、GridMaze は 1 枚で、形が変わる側を含む。DropMerge と Atari は、離散の Grid、Embedder、`StackMerge`、byte の画像の経路を通る。段階 A は強制行動で、題材によらない。LunarLander の出荷時のバックエンドでは、同じ実行体を同じ seed で繰り返しても学習リターンが変わった。2026-11-23 の開発者の指示（§13）により、比較時の条件を全環境でそろえる | P11、P98、P141 |
| D36 | テストのタグは `[forced_action]`・`[forwarder]`・`[perception]` とする | 今ある `[forced_action]` に合わせる | — |
| D37 | ADR を 4 本起こす（ADR 0083〜0086）。設計文書を 1 本新設し（Forwarder と Perception の仕組み）、130 の §2.2 と 200 は書き直す | 決定の単位ごとに 1 本。新しい仕組みには設計文書を新設する | — |
| D38 | ほかの RFD との順序は §7.11 のとおりにする。RFD0192・RFD0190 は実装済みで、本 RFD はその上に書く。RFD0191 が先に入り、本 RFD は RND の観測加工に触れない | RFD0192 と RFD0190 は実装が済んでいる。開発者が、次に進めるのは RFD0191 だと述べた。同じ箇所に触れる未実装の RFD ごとに、どちらが先でも成り立つ扱いを決めておく | P80、P117、P120 |

### 6.6 退けた案

| 案 | 退けた理由 |
|---|---|
| 定義の持ち主のクラス 1 つ（旧ドラフトの `ObservationPipeline`）。部品を取り出して並べるのは各経路 | 部品が増えるたびに、取り出す口と呼ぶ側の両方が変わる |
| 全体を「TensorDict を入れたら加工した TensorDict が出る」1 つの形にする | ReplayBuffer が重ねる前の 1 Frame を保存する位置で流れが切れ、重ねる処理には lane の履歴が要る。その形が成り立つ部分（正規化、キーを落とす、可視化）では、そのまま使う |
| 全部を設定の並びで書く（情報を足す部品も並びに書く） | D3 の理由 |
| 単位 A を別の RFD にして先行する（旧ドラフトの推奨） | D5 の理由 |
| 強制行動を lane 数ぶんの印と行動だけで表す（旧ドラフトの形） | 1 lane の指定で、呼ぶ側が lane 数を調べて組み立てることになる（D9） |
| 1 枚のときは stack 軸を足さない（旧ドラフトの §4.4） | stack の有無で形の意味が変わり、見分ける道具が要る（D27） |
| stack 軸を必ず持ち、NN の入口で既定で畳む | 軸の扱いが module の定義から読めなくなる（D32） |
| stack ありで素通しでない正規化を構築時に止める（旧ドラフトの §4.5） | D28 の理由 |
| 逆方向の口を、何もしない既定つきで足す | D6 の理由 |
| 正規化を NN の block にする | 統計を更新する口と checkpoint の書式が要る。NG2 の RFD で決める |
| 設定キーをそのままにして、並びだけ新しいキーにする | 書き方が 2 通りになる（D25） |
| 正規化を、部品の種類で分ける（定数の部品と統計つきの部品。素通しは並びに書かない） | 統計を持つ部品が 2 種類になり、「正規化は並びに 1 つまで」が種類をまたぐ。素通しのとき、メトリクスの鍵 `obs_norm.*` が未知の鍵になる（D43） |
| 正規化の設定キーを、今の名前のまま移す | モードで使うキーが変わるまとまりを移すときの決まりに、例外を残すことになる（D43） |
| stack の対象のキーを、空の一覧で「全キー」と表す（今の `stack_keys` の決まり） | 特定の値に別の意味を持たせる形になる（D44） |
| stack の対象のキーを選ぶスコープを、`target`（AI が初めに置いた名前）か、`key_selection`（AI の推奨）と名付ける | `target` は、target network の階層の名前と紛れる。`key_selection` と `keys` を比べて、開発者が `keys` を選んだ（D44） |
| 新しいクラスを、口・並びの仕組み・部品の 3 つ（または、口と、仕組みと部品の 2 つ）の機能グループに分ける（AI の推奨は 3 つ） | 使う側も作る側も、ヘッダを 1 つ読めば足りる、というフレームワークの方針に合わない（D45） |
| 共通の口に spec の変換も持たせ、Network が出力の spec を持つ（質疑では一度これを選び、出力の spec は仮の forward から導くとした） | 先送り。理由と見直しの条件は D22。共通の口は `Forward` だけにし、spec の変換は拡張の `SpecForwarder` に分けた（D20） |
| 観測の正規化の統計を、今のまま Agent が持つ（別モデルの推奨） | 設定キーを部品へ移す決定（D25）の一部を戻すことになり、1 つの正規化を、並びの位置と Agent の持つ統計の 2 か所で表す（D40） |
| Agent が、並びの中から `Normalize` の部品を探して、統計の更新を直接呼ぶ | Agent が部品の種類を知ることになる（D40） |
| 依頼ごとに別の名前の口（push 用の口など）を作る | 依頼を選ぶと、その依頼用の Forwarder が返る形にそろえる。呼ぶのは、どの依頼でも `Forward` である（D26） |
| 選ぶ値を、呼ぶ側の状況で名付ける（`ForwardScene` の `Act`・`Push`・`Sample`・`View`。質疑では一度これを選んだ） | 呼ぶ側の都合の名前で、頼む中身を表さない（D26） |
| 統計の更新を伴う依頼を、`Frame` とだけ名付ける。または `FrameAndUpdateStats` と名付ける | 前者は、名前に無い動き（統計の更新）が隠れる。後者は、作るものと付随の処理が同列に読める（D40） |
| device への転送を部品にして、並びに書く | 転送の位置が口ごとに違う（行動選択は重ねる前、Sample は ReplayBuffer が組み直したあと）。部品は、渡された Tensor のある device の上で動く（P23、P24） |
| N-step のキューを、stack の履歴とまとめる | 持つもの（観測と、報酬・終了）も、見る向き（過去と未来）も違う。接点は、Sample が obs と next_obs に同じ並びを当てることだけである（P40、P41） |

### 6.7 用語

用語集（[CONTEXT.md](../../CONTEXT.md)）に足した語と、定義を直した語である。

| 語 | 意味 | 既存の語で言い換えられない理由 |
|---|---|---|
| Frame | ReplayBuffer に保存する、観測側の 1 step 分 | 「フレーム」は ALE のエミュレータのフレームにも使われていた。slot は行動・報酬・終了も含む |
| Perception（知覚） | Agent が作る、Network へ渡す入力（taus を含めない）。渡す先の Network ごとにあり、本 RFD が並びで作るのは、Q の Network への Perception である（D1） | Observation は Env が形を決めるもので、重ねたり情報を足したりしたあとの入力を指す語が無かった |
| Forwarder | TensorDict を受けて返す変換。lane の履歴のような、呼ぶ側ごとの状態は持たない（重みや統計のような、共通の値は持つ。D20） | Network と観測の加工の部品を、1 つの語で指す語が無かった。Transform は Transformer と、パイプラインは `PipelineTrainRunner` と紛れる |
| Forwarder の並び（ForwarderChain） | 設定に書いた順に Forwarder をつないだもの | NN の `structure` は module の並びを指す。部品の種類が違うので語を分ける |
| Forward の依頼（ForwardRequest） | 並びに頼む変換の種類。何を渡して、何が欲しいかで区別する（統計を更新しつつ Frame を作る、重ねた束から Perception を作る、1 枚の Frame から Perception を作る） | 「経路」や「場面」は、呼ぶ側の状況を指す。頼む中身を指す語が無かった |
| stack 軸 | 観測キーを時刻の順に重ねた枚数の軸 | 設計文書 130 が使っている語で、用語集に無かった |
| 値の尺度（ValueScale） | 値の尺度が決まっているか（Scaled）、いないか（Unscaled）の区分。決まっているキーは、観測の正規化の対象にしない。上下限があるかは表さない（D17） | クラス数（離散か連続か）では、連続値のうち正規化しないものを表せない |
| 強制行動（定義を直した） | 人の操作などによって、方策の選択に代えて実行させる行動。lane ごとに指定する | 旧い定義は「Actor の外から置き換えた」で、D7 と合わない |

## 7. 実装契約

### 7.1 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。ただし、一覧に無い機能グループを新しく立てるときは、作る前に開発者に確かめる。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ForcedActions` | 新規 | lane ごとの強制行動を表し、検査と行動の差し替えを持つ | `rl`（318_rl_core。`src/core/base`） |
| `MakeActionOptions` | 変更 | 欄 `forced_actions` を足す | `rl` |
| `BatchActionInfo` | 変更 | `WithAction` を消す | `rl` |
| `EvalRunner` | 変更 | 行動の引数つきの `DoStep` を消す。検査を Reset の前に呼ぶ。強制の記録を lane ごとに作る | `trainer`（318_rl_core） |
| `dqn::Actor` | 変更 | 強制行動を選択のあと・ヒントの前に反映する。強制した lane の skip 状態を捨てる。`munchausen_terms` を消す。Perception の実行器を持つ | `dqn_actor`（321_agent_core。`src/algo/dqn`） |
| `DQNActionInfo` | 変更 | `WithAction` を消す | `dqn_actor` |
| `MuZeroActor`・`MuZeroActionInfo` | 変更 | 強制行動を受けて行動を差し替える。`WithAction` を消す | `muzero_based_agent`（321_agent_core。`src/algo/muzero`） |
| `ImageClsActor` | 変更 | 強制行動を受けて行動を差し替える | `image_cls_agent`（324_dqn。`src/algo/imagecls`） |
| `EvalPanel`・`QValuePanel` | 変更 | 強制行動を欄に入れて Runner へ渡す | `EvalPanel`・`QValuePanel`（110_runner_app。`src/apps/runner`） |
| `TensorSpec` | 変更 | 属性 `scale`・`has_stack_axis` と、JSON の欄を足す | `common`（312_util。`src/core/base`） |
| `ValueScale` | 新規 | 値の尺度（`Scaled`・`Unscaled`） | `common` |
| `Forwarder` | 新規 | 呼ぶ側ごとの状態を持たない変換の共通の口（`Forward` だけ） | `forwarder`（新設。314_tensor。D45） |
| `SpecForwarder` | 新規 | 通したあとの spec も答える `Forwarder`（`ForwardSpec` を足す）。並びに置ける部品の口 | `forwarder`（新設。314_tensor。D45） |
| `ForwardCapture` | 変更（`NetworkForwardCapture` から改名して移す） | forward の途中の値を、鍵で要求して持ち帰る器 | `forwarder`（新設。314_tensor。D45） |
| `ForwarderChain` | 新規 | 設定から Forwarder の並びを組み、依頼ごとの Forwarder と実行器を出す。並びの図（dot）を作る | `forwarder`（新設。314_tensor。D45） |
| `ForwardRequest` | 新規 | 並びへの依頼（`FrameWithStatsUpdate`・`PerceptionFromStack`・`PerceptionFromFrame`） | `forwarder`（新設。314_tensor。D45） |
| `ForwarderProvider` | 新規 | 依頼によって動きが違う部品が足して実装する口。依頼ごとの Forwarder と、行動選択用の実行器を出す | `forwarder`（新設。314_tensor。D45） |
| `ForwardExecutor` | 新規 | 行動選択用の実行器。lane の履歴を持つ | `forwarder`（新設。314_tensor。D45） |
| `FrameStackForwarder` | 新規 | 重ねる部品。依頼ごとの動きを持つ | `forwarder`（新設。314_tensor。D45） |
| `NormalizeForwarder` | 新規 | 正規化の部品。統計を 1 組持ち、`FrameWithStatsUpdate` の依頼で更新する。今の正規化の部品を中で使う | `forwarder`（新設。314_tensor。D45） |
| `DropKeysForwarder` | 新規 | キーを落とす部品 | `forwarder`（新設。314_tensor。D45） |
| `Network` | 変更 | `Forwarder` を実装する（出力の spec は持たない）。入口の離散の Grid は属性で stack 軸を見分ける | `nn`・`nn_impl`（315_nn） |
| `StackMergeModule` | 変更 | 5 次元以外の入力をエラーにする | `nn_modules`（315_nn） |
| `Flatten`・`StackMerge`・`Permute`・`Reshape`・`SpatialEmbedder`・`HybridSpatialEmbedder` の module | 変更 | 「stack 軸つきの入力を扱う」と宣言する | `nn_modules` |
| 正規化の部品（`DictObservationNormalizer` ほか） | 変更 | 形の検査を末尾の一致にし、統計を stack 軸へ広げて当てる | `scaler`・`scaler_impl`（314_tensor） |
| `DefaultDQNAgent` | 変更 | Perception の並びを 1 つ持ち、4 つの口をつなぐ。`MakeNetworkInputSpec`・`CreateFrameStacker`・可視化の手組みを消す | `default_dqn_agent`（324_dqn） |
| `dqn::Learner` | 変更 | Sample 用の並びを当てる。`NormalizeSampleObservations` を消す | `dqn_learner`（321_agent_core） |
| `StuckerConfig` | 削除 | — | `dqn_config`（321_agent_core） |
| `LunarLanderEnv`・`CartPoleEnv` | 変更 | float の vector の spec に `Unscaled` を宣言する（GridMaze・DropMerge・Atari の Env は、既定の `Scaled` のままで、宣言を足さない。D18） | `LunarLanderEnv`・`CartPoleEnv`（322_env_individual。`src/env`） |

`DictFrameStacker` と `ReplayBuffer` は変えない（NG5）。

機能グループの見込み（新しく立てる `forwarder`。D45）:

| 項目 | 中身 |
|---|---|
| ファイル名と置くライブラリ | `forwarder`。公開ヘッダ `src/core/base/include/anet/forwarder.hpp` と、`src/core/base/src/forwarder.cpp`。使い手の DQN Agent は別のライブラリ（`src/algo/dqn`）にあり、Network（`nn.hpp`）も `Forwarder` を実装するので、公開ヘッダに置く。Forwarder を使う側も、新しい Forwarder を作る側も、このヘッダ 1 つを読めば足りる |
| 受け持つ関心 | 「TensorDict を受けて返す変換」の共通の口と、それを設定の順に並べて、依頼ごとの変換と行動選択用の実行器を出す仕組みと、並びに置く汎用の部品。特定の Agent・Env・アルゴリズムの名前や鍵に依らない |
| 今回入るもの | 口の `Forwarder`・`SpecForwarder`、forward の途中の値を持ち帰る器の `ForwardCapture`、並びの仕組みの `ForwarderChain`・`ForwardRequest`・`ForwarderProvider`・`ForwardExecutor`、部品の `FrameStackForwarder`・`NormalizeForwarder`・`DropKeysForwarder` |
| あとで入る見込みのもの | 題材やアルゴリズムに依らない汎用の部品（キーの改名、型の変換、学習側だけの増強など）。足す要求が出たときに入る |
| 入れないもの | Agent が足す情報を作る部品（RFD0142 の直前の行動、RFD0173 の方策番号など。アルゴリズムの実装の側に置く。D3）。重ねる本体（`DictFrameStacker`。`stacker`）と、正規化の本体（`scaler`）。Network と module（`nn`） |
| 近い既存の機能グループと、入れない理由 | `tensor_util`（`TensorDict` と、tensor を扱う関数の集まり。口と並びの仕組みを入れると、関心が混ざる）。`stacker`（重ねる本体。契約を変えない。NG5）と、`scaler`（正規化の本体。stack 軸つきの入力を受けるようにする変更は、`scaler` の側で行う。D28）。`forwarder` の部品は、この 2 つを使う側である。`nn`（Network の側が `Forwarder` を実装する向きで、`nn` に置くと、観測の加工が NN のヘッダに依る） |
| 機能カテゴリ | 314 テンソル処理（Topic #10）。[categories.json](../../reports/tools/categories.json) の 314 の行の `match` に `forwarder` を足す。`stacker`・`scaler`・`tensor_store` と同じ行である |
| 設計文書 | 新設する 1 本（Forwarder と Perception の仕組み。D37）に書く。[130](../design/130_neural_networks.jp.md) の §2.2 と [200](../design/200_dqn_agents.jp.md) から案内する |
| 1 つにまとめることで起きること | `nn.hpp` がこのヘッダを読むので、並びの仕組みの宣言を変えると、`nn.hpp` を読むファイル（`src` に 16。P137）も再コンパイルされる。開発者が、読むヘッダを 1 つにする方針を優先した（D45） |

### 7.2 強制行動

呼ぶ側の書き方（例）。

```cpp
anet::rl::MakeActionOptions options;
options.forced_actions = anet::rl::ForcedActions::ByLane(0, action);   // 1 lane
// options.forced_actions = anet::rl::ForcedActions::ByMask(mask, actions);  // まとめて
runner->DoStep(counts, options);
```

- DQN の Actor は、方策の選択、skip の適用、強制行動の反映、ヒントの計算の順に行う。
- `ByLane(lane, action)` は、lane の番号と行動を整数で受ける。`ByMask(mask, actions)` は、bool の印と int64 の行動を、形 `[lane 数]` の Tensor で受ける。印が偽の lane の行動は読まない（D10）。
- `ByMask` の印と行動は、呼ぶ側の device のままでよい。型が Actor の device へ移す。
- 強制なしの `MakeAction` は、行動・aux・ヒント・乱数の消費のどれも今と同じである。

### 7.3 `TensorSpec` の属性と `Forwarder`

- 属性の組み合わせの検査は、行を上から順に当て、最初に当たった行で扱う。

| # | 条件 | 扱い |
|---|---|---|
| 1 | dtype が整数で、`scale` が `Unscaled` | エラー |
| 2 | 離散（クラス数が 1 以上）で、`scale` が `Unscaled` | エラー |
| 3 | 上のどれでもない | 通す |

- `SpecForwarder::ForwardSpec` は、受けた spec から、その変換を通したあとの spec を返す。構築時に、Frame の spec から部品が順に受け渡して、Perception の spec を作る。データは流さない。
- Frame の spec は、`DefaultDQNAgent` が 1 つ作り、ReplayBuffer の生成と、並びの入力の両方に渡す（D39）。今回は足すキーが無いので、Observation の spec と同じ中身である。
- Network は `Forwarder` だけを実装し、出力の spec は持たない（D22）。
- `ForwardCapture` を渡した呼び出しでも、今回の 3 つの部品は何も書かない。並びは、受けた capture を部品へそのまま渡す。

### 7.4 4 つの口

| 口 | 持ち主 | 受けるもの（渡す側） | 作るもの | 受け手 | すること |
|---|---|---|---|---|---|
| 行動選択 | Actor ごとの `ForwardExecutor` | Observation と、episode の開始の印（Runner から Actor へ） | Perception | Q の Network。その場で使い、運ばない | stack の実行器が、今と同じく Observation を Actor の device へ移してから履歴に積み、そのあとの部品はその device の上で動く |
| 経験の受入れ（push） | `DefaultDQNAgent` | 経験の obs と next_obs（Observation。Runner から Agent へ） | Frame（今回は Observation と同じ中身） | ReplayBuffer | `FrameWithStatsUpdate` の Forwarder に、obs と next_obs を 1 回ずつ通す。`Normalize` が統計を更新する。値は変えず、複製もしない（同じ Tensor を返す）。返った Frame を ReplayBuffer へ渡す |
| Sample | `dqn::Learner` | ReplayBuffer が重ねて返した Frame の束 | Perception | Q の Network（学習の forward） | obs、next_obs、skip の学習の窓の終点の観測に、同じ `PerceptionFromStack` の Forwarder を当てる |
| 可視化 | `DefaultDQNAgent` | 合成した 1 Frame（可視化の側から） | Perception | Q の Network | `PerceptionFromFrame` の Forwarder と Network をつないだ `TensorDictFunction` を作って渡す |

今回、Actor から経験の受入れへ運ぶ観測側の値は無い。Agent が足す情報を、行動の情報に載せて受入れへ運ぶ道（D2）を使うのは RFD0142 からで、運んだ値を `FrameWithStatsUpdate` の依頼の入力にどう合わせるかは、RFD0142 で決める（§15）。

依頼ごとの部品の動きは次のとおり。1 列目の「実行器」は、1 枚の Frame と lane の履歴から Perception を作る依頼で、値ではなく、実行器を作る口で受ける。

| 部品 | 実行器 | `FrameWithStatsUpdate` | `PerceptionFromStack` | `PerceptionFromFrame` |
|---|---|---|---|---|
| `FrameStack` | lane の履歴から重ねる | 何もしない | ReplayBuffer が重ね済み。1 枚のときだけ長さ 1 の軸を足す | 1 Frame を枚数ぶん複製する |
| `Normalize` | 統計を読んで当てる | 統計を更新する。値は変えない | 統計を読んで当てる | 統計を読んで当てる |
| `DropKeys` | 落とす | 何もしない（Frame には残す） | 落とす | 落とす |

呼ぶ側は、依頼を選んで、その依頼用の Forwarder を並びから受け取り、`Forward` を呼ぶ。行動選択だけは、履歴を持つ実行器の `Forward` を呼ぶ。依頼ごとに別の名前の口は作らない。`FrameWithStatsUpdate` の出力は Frame で、今回は入力の観測と同じ中身である。この依頼は、ReplayBuffer に入れる Frame を作るときだけ使う。

Perception の spec に taus を足したものが、Network の入力 spec になる（今と同じ）。

### 7.5 設定キー

`<Agent>` は Agent の設定の接頭辞（例: `DefaultDQNAgent.@baseline`）。

```text
DefaultDQNAgent.@baseline : perception.block.[stack].type ?= FrameStack
DefaultDQNAgent.@baseline : perception.block.[stack].count ?= 4
DefaultDQNAgent.@baseline : perception.block.[stack].keys.mode ?= all
DefaultDQNAgent.@baseline : perception.block.[norm].type ?= Normalize
DefaultDQNAgent.@baseline : perception.block.[norm].mode ?= none
DefaultDQNAgent.@baseline : perception.block.[norm].constant.use_clipping ?= false
DefaultDQNAgent.@baseline : perception.block.[norm].constant.clip_range ?= 10.0
DefaultDQNAgent.@baseline : perception.block.[norm].running.use_clipping ?= false
DefaultDQNAgent.@baseline : perception.block.[norm].running.clip_range ?= 10.0
DefaultDQNAgent.@baseline : perception.block.[norm].running.use_centering ?= false
DefaultDQNAgent.@baseline : perception.block.[norm].running.epsilon ?= 1
DefaultDQNAgent.@baseline : perception.block.[norm].running.use_robust_update ?= true
DefaultDQNAgent.@baseline : perception.block.[norm].running.robust_warmup_count ?= 200
DefaultDQNAgent.@baseline : perception.block.[norm].running.robust_std_threshold ?= 5.0
DefaultDQNAgent.@baseline : perception.block.[norm].running.post_process_type ?= 1
DefaultDQNAgent.@baseline : perception.block.[norm].running.post_process_threshold ?= 10.0
DefaultDQNAgent.@baseline : perception.structure ?= stack > norm

A1 : perception.block.[stack].count = 1                       # GridMaze。1 枚（既定も 1）
A9 : perception.block.[stack].keys.mode = listed            # 一部のキーだけを重ねる例
A9 : perception.block.[stack].keys.listed.names = vector
A9 : perception.block.[norm].mode = running                   # 統計つきの正規化を使う例
```

| 新しいキー | 型・値域 | 既定 | 旧いキー |
|---|---|---|---|
| `perception.structure` | block の名前を `>` でつないだ並び | 空（1 枚の `FrameStack` だけの並びになる。D27） | —（今は固定） |
| `perception.block.[名前].type` | `FrameStack`・`Normalize`・`DropKeys` | なし（必須） | — |
| `FrameStack` の `count` | 1 以上の整数 | 1 | `stucker.stack_count` |
| `FrameStack` の `keys.mode` | `all`（全キーを重ねる）・`listed`（`keys.listed.names` のキーだけを重ねる） | `all` | `stucker.stack_keys` が空（`all`）か、空でない（`listed`）か |
| `FrameStack` の `keys.listed.names` | 観測キーの一覧。`listed` のときは 1 つ以上 | 空 | `stucker.stack_keys` |
| `Normalize` の `mode` | `none`（素通し）・`constant`（定数で引いて割る）・`running`（走らせながら取る統計で標準化する） | `running`（基準の設定は `none` を置く） | `obs_norm.pass_through`・`obs_norm.use_dynamic_scaling`（組み合わせは下の表） |
| `Normalize` の `constant.mean`・`constant.std` | float の一覧。`mean` は有限、`std` は正。一覧が空のときは、今と同じく、`mean` は全要素 0、`std` は全要素 1 の定数になる。省略したときと、空の一覧を書いたとき（`constant.mean =`）は、同じ空の一覧として読む。空でない一覧は、要素数を、正規化の対象のキーの 1 Frame 分の要素数に合わせる（`constant` のときに、対象のキーごとに照合する。§7.6 の 8） | 空 | `obs_norm.constant_mean`・`obs_norm.constant_std` |
| `Normalize` の `constant.use_clipping`・`constant.clip_range` | 真偽。正の有限 | `true`・10 | `obs_norm.use_clipping`・`obs_norm.clip_range` |
| `Normalize` の `running.use_clipping`・`running.clip_range` | 真偽。正の有限 | `true`・10 | 同上 |
| `Normalize` の `running.use_centering`・`running.epsilon`・`running.use_robust_update`・`running.robust_warmup_count`・`running.robust_std_threshold`・`running.post_process_type`・`running.post_process_threshold` | 今の同じ名前のキーと同じ。`epsilon`・`robust_std_threshold`・`post_process_threshold` は正の有限、`robust_warmup_count` は 0 以上、`post_process_type` は 0〜3 | 今の同じ名前のキーと同じ（順に `false`・1・`true`・200・5・1・10） | `obs_norm.` の下の同じ名前 |
| `DropKeys` の `keys` | 観測キーの一覧。1 つ以上 | なし（必須） | — |
| — | — | — | `stucker.use_stacker`（無くなる。重ねないときは `count = 1`） |

`Normalize` の `mode` と、旧い 2 つのフラグの対応（D43）。

| 旧い指定 | 新しい `mode` | 読む設定キー |
|---|---|---|
| `pass_through = true`（`use_dynamic_scaling` はどちらでも） | `none` | なし |
| `pass_through = false`、`use_dynamic_scaling = false` | `constant` | `constant.*` の 4 つ |
| `pass_through = false`、`use_dynamic_scaling = true` | `running` | `running.*` の 9 つ |

- 選ばれていないモードの設定キー（`Normalize` の `constant.*`・`running.*`、`FrameStack` の `keys.listed.names`）も、読んで、そのキーだけで決まる型と値域を検査し、外れたらエラーにする。WARN や INFO は出さない。ほかの設定や spec と突き合わせる検査は、そのモードが選ばれているときだけ当てる。`FrameStack` の `keys.listed.names` を spec と照らす検査（§7.6 の 5 の前半）は `listed` のとき、`constant.mean`・`constant.std` の要素数の照合（§7.6 の 8 の後半）は `constant` のときである。`DropKeys` の `keys` を spec と照らす検査（§7.6 の 5 の後半）は、モードが無いので、常に当てる。
- 出荷設定の基準（`agent.txt`）が、並び `stack > norm`、4 枚、`keys.mode` の `all`、`mode` の `none` と、`constant.*`・`running.*` の今の値を置く。動きは今と同じである。
- 並びに現れない block の定義は検査しない（NN の使わないカタログ項目と同じ）。
- Agent の外に共通のカタログ（NN の `net.block` に当たるもの）は置かない。部品が 3 種類で、共有する定義がまだ無いためである。要るようになったら足す。

### 7.6 構築時の検査

`structure` は、NN の `structure` と同じ関数で読む。`>` と空白で区切り、空の要素は無いものとして読む（`stack > norm >` は `stack > norm` と同じ並びになる。P131）。分岐の書き方は無く、名前を順に並べるだけである。区切ったあとに名前が 1 つも無い設定は、並びを書かない設定と同じに扱う。

並びに `FrameStack` が書かれていなければ（並びを書かない場合を含む）、先頭に 1 枚の `FrameStack` を置いたものとして、表を当てる。行を上から順に当て、最初に当たった行で扱う。1〜9 のどれにも当たらず 10 に当たった設定は、WARN の一覧に出たうえで、11 と同じに構築へ進む。

| # | 条件 | 扱い |
|---|---|---|
| 1 | 並びの名前に当たる `block.[名前]` が無い、または `type` が無いか知らない値 | エラー（名前と `type` を文言に入れる） |
| 2 | 並びに `FrameStack` が書かれていて、先頭でない | エラー |
| 3 | 並びに `FrameStack` が 2 つ以上ある、または `Normalize` が 2 つ以上ある | エラー |
| 4 | `FrameStack` の `count` が 1 未満 | エラー（指定した値を文言に入れる） |
| 5 | `FrameStack` の `keys.listed.names`（`listed` のとき）か、`DropKeys` の `keys` に、その位置の spec に無いキーがある | エラー（キーと、spec にあるキーの一覧を文言に入れる） |
| 6 | 並びの入力の spec に、キーが 1 つも無い。または、既に stack 軸を持つキーがある（Env が自分で重ねた観測は、今回は扱わない） | エラー（stack 軸を持つキーは、キーを文言に入れる） |
| 7 | `DropKeys` の `keys` が無いか空、または全部のキーを落とす。`FrameStack` の `keys.mode` が `listed` で、`keys.listed.names` が空 | エラー |
| 8 | 並びに現れる block の設定キーの値が、§7.5 の型・値域の外にある（`mode`・`keys.mode` の知らない値と、選ばれていないモードの設定キーの値を含む）。または、`Normalize` の `mode` が `constant` で、空でない `constant.mean`・`constant.std` の要素数が、正規化の対象のキーのどれかの 1 Frame 分の要素数と合わない（空の一覧は照合しない。対象のキーが 1 つも無いときも照合しない） | エラー（キー・指定した値・期待する範囲か要素数を文言に入れる） |
| 9 | 並びに現れる block の定義の中に、読まれない設定キーがある | エラー |
| 10 | 旧いキー（`stucker.*`、Agent の直下の `obs_norm.*`）がある | 止めない。読まれないキーとして、起動時の WARN の一覧に出る（今ある仕組み）。旧いキーの値は使われないので、新しいキーが無ければ、既定の並び（1 枚の `FrameStack` だけ）になる |
| 11 | 上のどれでもない | 構築する |

NN の側（D32）。2 つの行は別の対象（branch の先頭の module、`StackMerge` の入力）を見るので、それぞれに当てる。

| # | 条件 | 扱い |
|---|---|---|
| 1 | stack 軸を持つ入力を受ける branch の先頭の module が、「stack 軸つきの入力を扱う」と宣言していない | WARN を 1 回出して続ける（branch・module・入力のキーと形を文言に入れる） |
| 2 | `StackMerge` の入力が 5 次元でない | エラー |

### 7.7 性能の見立て

性能の受入は置かない。行動選択と Sample で増えるのは、並びの部品の数（今回は 2〜3）ぶんの呼び出しと、TensorDict の受け渡しである。step ごとに 1 回（行動選択）と、学習の batch ごとに 3 回まで（Sample）で、Network の forward に比べて小さい見立てである。測ってはいない。等価性の Run（D35）で、実時間が大きく変わらないことを目で確かめる。

### 7.8 移行対象

追跡しているファイルの移動・改名・削除は無い（置き場を変える扱いには当たらない）。検索は 2026-11-21〜22 の `git grep`（大文字小文字を区別する固定文字列）による。

**直すもの**

| 種類 | 対象 | 直し方 |
|---|---|---|
| コード | `WithAction(` が当たる `src` の 11 ファイル（24 行。`rl.hpp`、`trainer.cpp`、`dqn_actor.cpp`・`.hpp`、`muzero_based_agent.cpp` と、テスト 6 ファイル） | D12 |
| コード | `munchausen_terms` が当たる `src` の 5 ファイル | D12 |
| コード | aux の `raw_obs`・`norm_obs`（`dqn_actor.cpp` の 2 行と、`dqn_concrete_test.cpp` の 10 行） | D42。テストは、Perception を直接見る形に書き直す |
| コード | 行動の引数つきの `EvalRunner::DoStep` を呼ぶ `EvalPanel.cpp`・`QValuePanel.cpp`・`trainer_test.cpp`・`episode_end_test.cpp` | D8 |
| コード | `NetworkForwardCapture` が当たる `src` の 18 ファイル（74 行） | D21 |
| コード | `stucker` が当たる `src` の 9 ファイル（資材を除く。52 行）、`StuckerConfig` が当たる `src` の 6 ファイル（15 行。`dqn_assembler.cpp`・`.hpp` を含む）、`CreateFrameStacker`・`MakeNetworkInputSpec`・`NormalizeSampleObservations` | D25・§7.1 |
| 設定 | `stucker` が当たる出荷設定の 4 ファイル（`agent.txt`・`GridMaze.txt`・`LunarLander.txt`・`DropMerge.txt`、17 行）と、`agent.txt` の `@baseline.obs_norm.` の 13 行 | §7.5。`stack_keys` と、正規化の 2 つのフラグは、§7.5 の対応のとおりにモードへ直す（D43・D44） |
| 設定 | stack の対象のキーを受ける branch の先頭に、軸を扱う module が無い Network の定義 | 先頭に module を足す。軸を自分で扱う module（DropMerge の Embedder）は、定義を変えずに、module に宣言を足す。実装の入口で、出荷設定を組んで WARN（§7.6）が出るものを数える |
| テスト | 並びを書かずに Network や Agent を組んでいるテスト（DefaultDQN の設定を組む箇所は、テスト 7 ファイルに約 145 行。P128） | 並びは書かなくてよい（1 枚の `FrameStack` だけになる）。stack 軸を扱う module を、Network の定義の先頭に足す。今の既定の正規化（素通しでない）に頼っているテストは、`Normalize` を並びに書く。件数は実装の入口で数える |
| テスト | 旧い設定キーを文字で書いているテスト（`Agent.obs_norm.` が当たる 5 ファイル 12 行と、`stucker` が当たるテスト）。設定を組む箇所は多い（`DefaultDQNAgentConfig` が 11 ファイルに 248 行。P128） | 新しいキーへ書き換える。移し漏れると、読まれないキーはテストでは止まらず、既定の値に戻る。受入 18・19 の検索で確かめる |
| テスト | `[native_stack]` のテストと、`StackMerge` に 4 次元を入れているテスト | D27・D32 に合わせる。件数は実装の入口で数える |
| 検証資材 | `src/core/base/testdata/rfd0072/baseline/` の 17 ファイル | 設定キーを移したあとの解決結果で作り直す |
| 設計文書（jp） | `110_agents_and_learning`、`130_neural_networks`（§2.2）、`140_observability`、`150_replay_buffer`、`200_dqn_agents`、`220_atari_env`、新設 1 本（Forwarder と Perception の仕組み） | 新しい契約に合わせる |
| wiki | `110_runner_app`、`312_util`、`314_tensor`、`315_nn`、`318_rl_core`、`321_agent_core`、`322_env_individual`、`324_dqn` | 実装のあとの取り込みで直す |
| 道具の設定 | [categories.json](../../reports/tools/categories.json) の 314 の行 | `match` に `forwarder` を足す（D45） |
| ADR | 0010・0012・0036（`WithAction`）、0026（`stucker.*`）、0031・0055（`NetworkForwardCapture`） | 本文は変えず、先頭に後発の決定の注記を足す |

**直さないもの**

| 種類 | 対象 | 理由 |
|---|---|---|
| 過去の記録 | `docs/experiments/`（560 ファイル）、`docs/memo/` のほかの RFD と実装記録、`reports/`、`CHANGELOG.md`、`docs/management/issues/` | 当時の記録 |
| 検証資材 | `src/core/base/testdata/rfd0092/config/`（11 ファイル）、`src/core/base/testdata/rfd0121/README.md` | 過去の Run の設定の写しと、資材を作ったときの記録。テストは設定として読まない（`src` を `rfd0092` で検索して、資材の外に 0 行） |
| 旧キーを扱う検証の入力 | [固定入力の移行](../../src/core/base/testdata/rfd0072/perception_migration.py)の旧スタック設定の接頭辞と、[受入 29 のテスト](../../src/core/base/src/forwarder_test.cpp)の旧設定キー。各 1 行 | 固定した過去の入力を現行の 17 構成と比較するため、および旧キーが未読になることを確かめるため。両方とも旧名をそのまま書き、受入 18 の検索結果に含めて照合する |
| 英訳 | `docs/design/*.en.md` | 翻訳の道具が作り直す |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) の、`ObservationNormalizer` に触れる 1 項目 | 設定キーの名前を書いていない。語は起草のときに足してある（§6.7） |
| 待っている RFD | 142・159・160・173・174・183 の本文 | 再グリルで直す。材料は §15 |
| 本 RFD | 本 RFD が旧い名前を書いている箇所 | 改訂の記録、移行対象、検索の文字列としての引用 |

検索の当たりの割り振り（2026-11-21。ファイル数）は次のとおりで、どの行にも入らないファイルは無い。

- S5（`stucker`）: 621 ＝ 直す 33（`src` 9、`rfd0072/baseline` 17、`apps` 4、設計文書 jp 3）＋ wiki 5（取り込みで直る）＋ 直さない 583（`docs/experiments` 560、`rfd0092` 11、`docs/memo` 7、課題 4、ADR 1）。
- S8（`WithAction`）: 51 ＝ 直す 14（`src` 11、設計文書 jp 3）＋ wiki 2 ＋ 直さない 35（`docs/memo` 25、設計文書 en 3、ADR 3、`reports` 2、課題 1、`rfd0121` の README 1）。
- S4（`NetworkForwardCapture`）: 31 ＝ 直す 19（`src` 18、設計文書 jp 1）＋ wiki 1 ＋ 直さない 11（`docs/memo` 6、課題 2、ADR 2、設計文書 en 1）。
- S1（`WithAction(`）: 23 ＝ 直す 13（`src` 11、設計文書 jp 2）＋ 直さない 10（`docs/memo` 8、設計文書 en 2）。S2: 8 ＝ 直す 5 ＋ 直さない 3（`docs/memo`）。S3: 10 ＝ 直す 2（`trainer.hpp`・`trainer.cpp`）＋ 直さない 8（`EvalPanel` の 2、`docs/memo` 5、課題 1）。S6: 2 ＝ 直す 1（`agent.txt`）＋ 直さない 1（本 RFD）。S7: 7 ＝ 直す 6 ＋ 直さない 1（本 RFD）。
- S9（`Agent.obs_norm.`）: 598 ＝ 直す 22（`src` 5、`rfd0072/baseline` 17）＋ 直さない 576（`docs/experiments` 564、`rfd0092` 11、本 RFD 1）。S10: 2 ＝ 直す 1（`default_dqn_agent.hpp`）＋ 直さない 1（本 RFD）。S11・S12: それぞれ 4 ＝ 直す 2（`dqn_actor.cpp`・`dqn_concrete_test.cpp`）＋ 直さない 2（`docs/memo`）。

### 7.9 実装の入口で数えること

- 宣言（D18）を足すテスト用の spec の件数。
- 軸を扱う module を先頭に足す Network の定義の件数（出荷設定とテスト）。
- `StackMerge` に 4 次元を入れているテストの有無。
- 先頭が `Linear` の branch に stack 軸つきの入力を入れたとき、今は構築で止まるか（P115）。

### 7.10 段階

| 段階 | 中身 | 終わりの条件 |
|---|---|---|
| A | D7〜D16 | 受入 1〜10 |
| C | D17〜D22・D24・D45。今の入力 spec を作る関数が、軸を足すときに `has_stack_axis` も立てる | 受入 11〜15・17 |
| B | D23・D25〜D34・D39〜D44 | 受入 16・18〜30・33・34 |

各段階のあとで、全テストが通り、等価性の Run が一致すること。段階の途中で止めない。

### 7.11 ほかの RFD との順序

| RFD | 触る同じ箇所 | 扱い |
|---|---|---|
| [192](0192_concurrency_and_boundary_defect_fixes_10rfd.md)・[190](0190_dqn_design_doc_alignment_10rfd.md) | `WithAction`、設計文書 200 | 実装済み。本 RFD はその上に書く。RFD0190 の本文は直さない |
| [191](0191_rnd_raw_error_and_recent_learning_10rfd.md) | `default_dqn_agent.cpp` の受入れ、RND | RFD0191 が先に入る。本 RFD は RND の観測加工に触れない。RFD0191 が段階の切れ目で入るなら、次の段階の前に等価性の基準を取り直す。段階の途中では入れない |
| [142](0142_agent_prev_action_obs_10rfd.md) | Actor、受入れ、spec | 本 RFD の A・B のあと。RFD0142 のドラフトは、同じ部品を Actor と受入れ側に 1 つずつ持つ形で、D2 と違う。再グリルの材料は §15 |
| [159](0159_input_mapping_joystick_10rfd.md)・[160](0160_input_target_train_forced_action_10rfd.md) | Runner の強制の口、EvalPanel | 本 RFD の A のあと。RFD0160 の「lane の印と行動」の書き方は `ForcedActions` に合わせて直す（再グリル） |
| [174](0174_llm_actor_10rfd.md)・[183](0183_temporl_skip_followups_10rfd.md) | Actor の強制、skip の状態 | 本 RFD の A のあと |
| [176](0176_qvaluepanel_manual_action_state_alignment_10rfd.md) | `QValuePanel` の手動の行動、`WithAction` の経路 | どちらが先でもよい。本 RFD が先なら、RFD0176 は「強制は `MakeAction` の入力」を前提に書き直す。RFD0176 が先なら、本 RFD の A が `QValuePanel` の呼び出しを欄に直す |
| [081](0081_evalpanel_episode_trace_10rfd.md) | `EvalRunner::DoStep` の口 | どちらが先でもよい。あとの側が、行動の引数つきの口が無い形に合わせる |
| [173](0173_intrinsic_reward_novelty_10rfd.md) | Network の入力へ足す情報 | 本 RFD の B のあと。材料は §15 |
| [180](0180_core_perf_candidates_202607_10rfd.md) | `NetworkForwardCapture` | どちらが先でもよい。あとの側が `ForwardCapture` の名前に合わせる |
| [134](0134_network_lock_audit_10rfd.md)・[141](0141_background_eval_snapshot_ordering_10rfd.md)・[143](0143_serialize_10rfd.md) | 排他、評価の順序、checkpoint | 本 RFD は排他と checkpoint を変えない（NG2）。順序の制約は無い |

### 7.12 並びの図（dot）

NN の構造図と同じ仕組みで出す（D41）。

| 図の名前 | 中身 |
|---|---|
| `perception.structure` | 入力（Frame の spec のキー）から、部品（名前と種類）を並びの順につないで、出力（Perception）まで |
| `perception.detail` | 上に加えて、各段の出口の spec（キーごとの shape・dtype・値の尺度・stack 軸の有無）と、部品の設定（`count`・`keys.mode`・`mode` など、読んだ設定キーと値） |

- 記録する時機と出る場所は、Network の構造図（`net.structure`・`net.detail`）と同じである。Run の `dot/` の下に出る。
- 図は、並びを組んだときの情報から作る。部品を足すときに、図のための実装は要らない。
- 並びを書かない設定でも出す（1 枚の `FrameStack` だけの図になる）。
- 向きは左から右（NN の構造図の既定と同じ）。設定キーは足さない。図の出し分けが要るようになったら、そのときに足す。
- `metrics.jsonl` の `scalar` と `trace` には何も足さないので、等価性の比べ方（§9）には影響しない。

## 8. 机上の確かめ

確定事項の形に、想定されるパターンを 1 つずつ流した結果である（開発者の指示による。§13）。2026-11-20 に 36 件を流し、質疑の終わりと、別モデルの検討の決定で流し直して、11 件を足した。レビューの指摘への直し（2026-11-22）で、6 件を足した。

| # | パターン | 流した結果 | 判定 |
|---|---|---|---|
| 1 | LunarLander（`vector[8]`、4 枚、素通し） | `[8]`・`Unscaled`・軸なし → 重ねて `[4,8]`・軸あり → 素通しで変わらず。今と同じ形と値 | 整合 |
| 2 | Atari（byte の `grid`、4 枚）と RND | Q の側は `Scaled` の宣言で今と同じ。RND の判定は変えない（D33） | 整合 |
| 3 | DropMerge（float の `vector` と、番号の `grid`） | `vector` も `grid` も `Scaled`（D18）。NN の入口は、属性で stack 軸を見分けて one-hot にする。branch の先頭の Embedder は、軸を自分で扱うと宣言する（D32）。結果は同じ | 整合 |
| 4 | GridMaze（1 枚） | 行動選択・Sample・可視化のどこでも、`[1,2]`・軸ありにそろう。Network の入力 spec と Sample の形が変わる。先頭が `Flatten` なので値は同じになる見込み | 受入 30 で確かめる（未決表 U2） |
| 5 | ImageCls（`grid` とラベルの番号） | Perception を通らない。入力に stack 軸は無い。Network は `Forwarder` を実装する | 整合 |
| 6 | Rainbow・MuZero | 観測の加工は無いまま。Network が `Forwarder` を実装する | 整合 |
| 7 | `keys.mode = listed`、`keys.listed.names = vector`（一部だけ重ねる） | 書いたキーだけ軸あり。ReplayBuffer にも同じ一覧を渡す | 整合 |
| 8 | `keys.listed.names` に無いキー名、重ね済みのキー | 構築時にエラー（§7.6 の 5・6） | 整合 |
| 9 | 重ねない | 枚数 1（既定）。並びに `FrameStack` を書かなくても、1 枚の `FrameStack` が先頭に入り、長さ 1 の軸を持つ（D27） | 整合 |
| 10 | 統計つきの正規化と stack | 重ねたあとのキーは軸あり・`Unscaled` で、正規化の対象になる。統計を軸へ広げて当てる（D28） | 整合 |
| 11 | 統計つきの正規化と、float のキーが複数 | 定数がキー共通の 1 組なので、形が合わないキーがあると今も成り立たない | 範囲外（NG4） |
| 12 | 並びの順 | 出荷設定は `stack > norm` の 1 通り。先頭は `FrameStack` に限る | 整合 |
| 13 | Actor は GPU、ReplayBuffer は CPU、prefetch あり | 素通しなら整合。統計つきは、統計の置き場と並んだ読み書きが決まっていない | 範囲外（NG2） |
| 14 | episode の先頭 | 行動選択も ReplayBuffer も、開始の Frame で埋める。同じ規則 | 整合（受入 22） |
| 15 | 打ち切り | 終端の Frame は専用の slot。next_obs は終端を最新として読む | 整合（受入 22） |
| 16 | n-step の途中で episode が終わる | 実際の n が短くなり、next_obs は終端になる | 整合（受入 22） |
| 17 | 評価用の Actor | 実行器は履歴を持ち、ReplayBuffer には入れない | 整合 |
| 18 | Actor が複数 | 定義は 1 つ、実行器は Actor ごと、統計は共有 | 整合 |
| 19 | Eval で強制行動 | Actor の出力が強制行動になり、Actor Q ヒントも Actor が作る | 整合 |
| 20 | Train で強制行動（RFD0160 のあと） | skip の継続状態を捨て、その step を判断として記録する | 整合 |
| 21 | 強制行動と、将来の「直前の行動」 | 次の step の直前の行動は、Actor の出力（強制行動）になる | 整合 |
| 22 | primary の Network（IQN） | 入力の spec は Perception の spec に taus を足したもの。Network は出力の spec を持たない（D22） | 整合 |
| 23 | skip の補助 Network | 入力の spec の作り方は今のまま（primary を仮の入力で 1 回通して、出力の形を読む。D22） | 整合 |
| 24 | Network の複製（target、Actor 用） | 複製も `Forwarder` として同じに使える | 整合 |
| 25 | 状態つきの実行（NoisyNet など） | 今の実行器のまま。共通の口の `Forward` は既定の動き | 整合 |
| 26 | 可視化（状態スイープ） | 並びから関数を作る。1 枚でも長さ 1 の軸を足す | 整合 |
| 27 | 直前の行動（RFD0142。float の one-hot） | `Scaled` と宣言する。重ねると行動の履歴になる | 整合 |
| 28 | 方策番号（RFD0173。番号） | `keys.mode` が `all` だと重なる（意味は無い） | RFD0173 で決める |
| 29 | 直前の報酬（RFD0173） | 行動選択の時点で、報酬が Actor に届かない | RFD0173 で決める |
| 30 | 別の方策番号として学ぶ | Sample の口が Sample の結果を受けるので、口を変えずに足せる | 整合 |
| 31 | 学習側だけの増強 | `ForwarderProvider` として、`PerceptionFromStack` の依頼のときだけ動く。往復の確かめの対象外と宣言する | 整合 |
| 32 | キーを落とす部品 | Frame には残し、Perception を作る依頼だけで落とす | 整合 |
| 33 | 共通の口と、依頼によって動きが違う `FrameStack` | `ForwarderProvider` が、依頼ごとの Forwarder と実行器を返す（D26） | 整合 |
| 34 | 素通しの正規化と尺度 | 素通しは `Scaled` に変えない。Perception の spec は `Unscaled` のまま Network に入る | 整合 |
| 35 | spec の JSON に属性が増える | Env どうしの一致の確かめは、同じ宣言どうしなので通る。Run の記録に欄が増える | 整合 |
| 36 | 非ゴールの文面 | NG6・NG7 を、spec の宣言と Network の口は変わる、と書き直した | 整合 |
| 37 | 並びと実行器に seed を渡す | 名前から作るので、今の乱数の並びは変わらない（D23） | 整合 |
| 38 | capture を渡した forward | 並びは部品へそのまま渡す。今回の部品は何も書かない。要求が満たされなければ Network の出口で止まる | 整合 |
| 39 | stack 軸つきの入力を受ける branch の先頭が `Linear` | WARN を 1 回出して続ける（D32） | 整合 |
| 40 | `StackMerge` に 4 次元 | エラー。出荷設定の RND の定義に当たるものは無い | 整合（テストは §7.9） |
| 41 | 旧いキー `stucker.stack_count` を書いた設定 | 読まれないキーとして WARN の一覧に出る。新しいキーが無ければ、1 枚の `FrameStack` だけの並びになり、旧いキーの枚数は使われない。出荷設定は、受入 18・19 の検索で、旧いキーが残っていないことを確かめる | 整合 |
| 42 | 設定の解決結果の基準（17 構成） | 設定キーが変わるので、基準を作り直す（§7.8） | 整合 |
| 43 | 並びに現れない block の定義 | 検査しない | 整合 |
| 44 | 経験の受入れ（push） | Agent が `FrameWithStatsUpdate` の Forwarder に obs と next_obs を通す。`Normalize` が統計を更新し、ほかの部品は何もしない。観測はそのまま ReplayBuffer に入る（D40） | 整合 |
| 45 | 並びに `Normalize` を書かない | `FrameWithStatsUpdate` の依頼は、観測をそのまま返すだけになる。メトリクスの鍵 `obs_norm.*` に答える部品が無いので、その鍵は未知の鍵になる | 整合 |
| 46 | 並びの図 | `perception.structure` と `perception.detail` が、Run の `dot/` の下に出る。並びを書かない設定でも出る。metrics の比較（scalar と trace）には入らない（D41） | 整合 |
| 47 | 並びは書いたが、`FrameStack` を書かない（`structure = norm`） | 先頭に 1 枚の `FrameStack` を置いて組む。図にも出る（D27） | 整合 |
| 48 | `Normalize` の `mode` を切り替える | `none` は素通しで、どのキーも使わない。`constant` は `constant.*`、`running` は `running.*` だけを使う。選ばれていないモードのキーは、値域だけ検査する。今の 2 つのフラグの 3 通りと、同じ動きになる（D43） | 整合 |
| 49 | キーの一覧を空にして「全キー」を表す（今の `stack_keys` の書き方） | その書き方は無い。全キーは `keys.mode = all`（既定）で書く。`listed` で一覧が空ならエラー（D44） | 整合 |
| 50 | `Normalize` が並びにあり、正規化の対象のキーが無い（GridMaze・DropMerge・Atari） | 構築は通り、何もしない。メトリクスの鍵 `obs_norm.*` は値を返さない。GridMaze と DropMerge は、今は 0 を返す（D40）。出荷設定に、この鍵を記録する有効な定義は無い | 整合（返す値は変わる。等価性の Run には現れない） |
| 51 | 並びの入力の spec に、既に stack 軸を持つキーがある | 構築時にエラー（§7.6 の 6）。今の Env に、そういう観測を出すものは無い | 整合 |
| 52 | 並びの入力の spec に、キーが 1 つも無い | 構築時にエラー（§7.6 の 6）。`keys.mode = all` で決まるキーが 0 件になり、ReplayBuffer が空の一覧を全キーと読むのを防ぐ | 整合 |
| 53 | `mode = constant` の定数の書き方（8 要素の `Unscaled` のキー） | 省略と、空の一覧（`constant.mean =`）は、同じ空の一覧で、`mean` は全要素 0、`std` は全要素 1 になる。8 要素の一覧は、そのまま使う。2 要素の一覧は、構築時にエラー（§7.6 の 8）。`mean` と `std` は、別々に判定する | 整合 |

## 9. 受入

検索は `git grep -n -F`（大文字小文字を区別する固定文字列）で、範囲は項目に書く。検索の文字列は表の下に置く。

**単位 A**

| # | 受入 |
|---|---|
| 1 | 検索 S1 が、`src` と `apps` で 0 行 |
| 2 | 検索 S2 が、`src` と `apps` で 0 行 |
| 3 | `[forced_action]` のテスト: 2 lane で lane 0 だけ強制すると、出力の行動は lane 0 が強制行動、lane 1 が方策の選択になる。同じ seed の Actor を強制なしで動かした結果と、lane 1 の行動・Q・ヒントが一致する |
| 4 | `[forced_action]` のテスト: 強制した lane の Actor Q ヒントの行動の Q と Munchausen の項が、強制行動から計算される。期待値は、消す `WithAction` のテストの期待値を移したものと一致する |
| 5 | `[forced_action]` のテスト: 範囲外の lane（`ByLane`）、範囲外の行動（`ByLane` と、`ByMask` の印が真の lane）、長さの違う `ByMask`、dtype の違う `ByMask`、連続行動の Env への強制が、それぞれエラーになる。`ByMask` で、印が偽の lane に範囲外の値（-1）を入れた指定は通り、その lane の行動は方策の選択になる。Reset していない `EvalRunner` に範囲外の強制を渡すと、エラーのあとも Env は Reset されていない |
| 6 | 検索 S3 が、`src/core/base/include/anet/trainer.hpp` と `src/core/base/src/trainer.cpp` で 0 行（`EvalPanel` 自身の同じ名前の関数は数えない） |
| 7 | `[forced_action]` のテスト: skip が有効な Actor で lane を強制すると、その lane の継続状態が消え、その step が判断として記録される（判断のフラグが真、skip 長が 1）。継続の途中だった lane を強制しても同じになる。skip の診断の数は、強制を反映したあとの値と一致する |
| 8 | `[forced_action]` のテスト: 2 lane で lane 0 だけ強制すると、`EvalRunner` の `forced` の鍵が lane 0 だけ真になる |
| 9 | `[forced_action]` のテスト: MuZero と ImageCls の Actor で、強制した lane の行動が差し替わる |
| 10 | 等価性（段階 A）: 表の下の「等価性の比べ方」で、段階の前後が一致する |

**単位 C**

| # | 受入 |
|---|---|
| 11 | `[forwarder]` のテスト: `TensorSpec` の JSON の出力に、`scale` と `stack_axis` の欄が、宣言した値で出る |
| 12 | `[forwarder]` のテスト: 整数の dtype で `Unscaled`、離散で `Unscaled` の spec が、それぞれエラーになる |
| 13 | `[forwarder]` のテスト: LunarLander・CartPole の `vector` の spec が `Unscaled`、GridMaze・DropMerge・Atari の観測の spec が `Scaled` である |
| 14 | `[forwarder]` のテスト: Network を `Forwarder` として受ける関数に渡したときの出力が、`Network::Forward` を直接呼んだ出力と一致する。capture を渡した呼び出しでも、要求した鍵の値が同じである |
| 15 | 検索 S4 が、`src` と `apps` で 0 行 |
| 16 | （段階 B で確かめる）`[perception]` のテスト: 並びの `ForwardSpec` が返す Perception の spec の各キーの shape と dtype が、同じ並びの `PerceptionFromFrame` の Forwarder に、Frame の spec どおりの形の仮の Frame（batch は 2）を通した出力の、batch の軸を除いた形と dtype に一致する（4 枚と 1 枚、キーを落とす部品を含む並び）。属性は spec の側だけにあるので、返った spec で、stack の対象のキーが「stack 軸を持つ」になり、素通しでない正規化を通したキーが `Scaled` になることを確かめる。並びを組む口に渡した seed から、block ごとの seed が block の名前で決まり、並びの順を入れ替えても変わらない |
| 17 | 等価性（段階 C）: 表の下の「等価性の比べ方」で、段階の前後が一致する |

**単位 B**

| # | 受入 |
|---|---|
| 18 | 検索 S5 を `src`・`apps`・`data` にかける。過去の Run の設定の写し `src/core/base/testdata/rfd0092/` を除いた全結果を表示し、§7.8 の旧キーを扱う検証の 2 行（固定入力の移行と受入 29 に各 1 行）だけが当たることを確かめる。製品コードと出荷設定は 0 行 |
| 19 | 検索 S6 が、`apps` と `data` で 0 行。検索 S9 と S10 が、`src` で 0 行（テストが文字で書く旧い設定キーと、旧いキーの読み取り）。`src/core/base/testdata/rfd0092/` は数えない（受入 18 と同じ理由） |
| 20 | 検索 S7 が、`src` で 0 行。検索 S11 と S12 が、`src` と `apps` で 0 行（aux の `raw_obs`・`norm_obs`） |
| 21 | `[perception]` のテスト: §7.6 の表の 1〜9 の各行で、構築がエラーになる。`structure` が `stack > norm >` の設定は、`stack > norm` と同じ並びとして組まれる |
| 22 | `[perception]` の往復のテスト: 行動選択側の Perception と、同じ経験を ReplayBuffer に入れて Sample で作り直した Perception が一致する。条件は、4 枚と 1 枚、episode の先頭、打ち切りの終端、n より手前で episode が終わる next_obs、skip の窓の終点。あわせて、ReplayBuffer の設定に渡った枚数とキーが、行動選択の側と同じである |
| 23 | `[perception]` のテスト: 枚数 1 と 4 のそれぞれで、行動選択・Sample・可視化の Perception と、Network の入力 spec の stack の対象のキーが、`[B, S, *shape]` になる。並びを書かない設定と、`count` を書かない `FrameStack` では、1 枚（`[B, 1, *shape]`）になる。並びに `Normalize` を書かない設定（並びを書かない設定を含む）では、Perception の値が、Frame を重ねただけの値と一致する |
| 24 | `[perception]` のテスト: 4 枚で素通しでない正規化を使うと、行動選択が止まらず、値が 1 Frame ずつ正規化して重ねたものと一致する。`FrameWithStatsUpdate` の Forwarder に観測を通すと、統計が更新され、返る値は入力と同じである。Perception を作る依頼（実行器、`PerceptionFromStack`、`PerceptionFromFrame`）では、統計が変わらない |
| 25 | `[perception]` のテスト: `DropKeys` に書いたキーが、Perception を作る依頼（実行器、`PerceptionFromStack`、`PerceptionFromFrame`）の出力と、Perception の spec から消える。ReplayBuffer の Frame には残る |
| 26 | `[perception]` のテスト: stack 軸つきの入力を受ける branch の先頭が `Linear` の Network を組むと、WARN が 1 回出て、構築は続く。先頭が `Flatten` なら出ない |
| 27 | `[shipped_config]` のテスト: 設定の基準のテスト（P116）が使う 17 構成のうち、DefaultDQN の構成のそれぞれで Agent を組んだとき、§7.6 の NN の側の 1 行目の WARN が 0 件である |
| 28 | `[perception]` のテスト: `StackMerge` に 4 次元を入れるとエラーになる。離散の Grid は、属性のあるなしで、stack 軸のあるなしが決まる |
| 29 | `[perception]` のテスト: 旧いキー `stucker.stack_count` を書いた設定で、読まれないキーの一覧にそのキーが出る。統計つき（素通しでも定数でもない）の `Normalize` を並びに書いた設定で、`FrameWithStatsUpdate` の Forwarder に観測を通したあと、Agent のメトリクスの鍵 `obs_norm.count` の値が、同じ観測を今の正規化の部品（`DictObservationNormalizer`）へ同じ回数だけ直接通したときの値と一致し、有限である。`Normalize` を並びに書かない設定では、同じ鍵は値を返さない（未知の鍵になる）。正規化の対象のキーが 1 つも無い構成（全キーが `Scaled`）で `Normalize` を並びに書くと、構築は通り、Perception の値は変わらず、同じ鍵は値を返さない |
| 30 | 等価性（段階 B）: 表の下の「等価性の比べ方」で、段階の前後が一致する。段階のあとの Run は、新しい設定キーへ移した出荷設定を使う。GridMaze が一致しないときは、未決表 U2 による |
| 33 | `[perception]` のテスト: 並び `stack > norm` の図の dot の文字列に、2 つの部品の名前と種類、並びの順の edge、入力と出力のキーが入っている。詳しい図には、各段の shape と、`count` の値も入っている。等価性の Run（受入 30）の Run フォルダの `dot/` の下に、`perception.structure` と `perception.detail` の dot ファイルがある |
| 34 | `[perception]` のテスト: `FrameStack` の `keys.mode` が `listed` の設定で、`keys.listed.names` に書いたキーだけが stack 軸を持つ。`all` では、全キーが持つ。`Unscaled` と宣言した float のキー（形は LunarLander の `vector` の `[8]`）について、`Normalize` の `mode` が `none`・`constant`・`running` のそれぞれで、Perception の値が、正規化の本体（`scaler` の `DictObservationNormalizer`）を、順に素通し・定数・統計つきの同じ設定の値で直接組んだものと、次の手順で一致する。(1) 両方に、同じ Frame の列を同じ順に通して、統計を更新する（新しい側は `FrameWithStatsUpdate` の Forwarder、本体の側は統計を更新する口。`none` と `constant` では、何も変わらない）。(2) 本体の側は、4 枚の Frame を、stack 軸の無い形（`[B, 8]`）で 1 枚ずつ正規化して、stack 軸で重ねる。新しい側は、同じ 4 枚を重ねた束（`[B, 4, 8]`）を、`PerceptionFromStack` の Forwarder に通す。(3) 2 つの値を比べる（受入 24 と同じ比べ方）。`Scaled` と宣言した float のキーは、どの `mode` でも、Perception の値が、Frame を重ねただけの値と一致する（今の部品は、このキーも正規化する。D18 による違いである）。`mode` が `none` の設定で `running.epsilon` に 0 を書くと、エラーになる |

**共通**

| # | 受入 |
|---|---|
| 31 | Debug のビルドが通り、ビルドが作るテスト実行体（`out\test\Debug\` の `*-test.exe`）を引数なしで走らせて、どれも失敗が 0 件である。`git diff --check` の出力が空である |
| 32 | 検索 S8・S5・S4 が、`docs/design` の `*.jp.md` と [CONTEXT.md](../../CONTEXT.md) で 0 行（今は S8 が 3 ファイル 5 行、S5 が 3 ファイル 4 行、S4 が 1 ファイル 1 行）。Forwarder と Perception の仕組みを書いた新しい `*.jp.md` が `docs/design` に 1 本あり、`docs/design/README.jp.md` の一覧に載っている。`reports/tools/categories.json` の 314 の行の `match` に、`forwarder` が入っている |

```text
S1: WithAction(
S2: munchausen_terms
S3: DoStep(int64_t
S4: NetworkForwardCapture
S5: stucker
S6: @baseline.obs_norm.
S7: StuckerConfig
S8: WithAction
S9: Agent.obs_norm.
S10: config_data, obs_norm.
S11: "raw_obs"
S12: "norm_obs"
```

**等価性の比べ方**（受入 10・17・30）: 段階の直前の HEAD の exe と、段階のあとの exe で、同じ設定・同じ seed の Run を 1 本ずつ流す。対象は、出荷の設定ファイルを、`run.$` の行もほかの行も書き換えずに流した Run である。LunarLander（[LunarLander.txt](../../apps/runner/config/LunarLander.txt)。4 枚）と GridMaze（[GridMaze.txt](../../apps/runner/config/GridMaze.txt)。1 枚）の 2 本を流し、段階 C と B（受入 17・30）では、DropMerge（[DropMerge.txt](../../apps/runner/config/DropMerge.txt)）と Atari（[Atari.txt](../../apps/runner/config/Atari.txt)）も 1 本ずつ足す。どの Run プロファイルで動くかは、設定ファイルが決める（2026-11-22 の時点で、LunarLander は `run.@munch > run.@skip`、Atari は 164 行の並びで、GridMaze と DropMerge は有効な `run.$` の行が無く、ファイルの本体の値で動く。P133）。段階 B のあとの Run は、新しい設定キーへ移した同じファイルを使う。実装計画に、Run ごとに解決された `run.$` と Atari のゲームを、Run の設定の写しから書く。長さは、learn の step が 1,000 に達するまでとする（そのための exp の step 数は実装計画に書く）。

全環境で、変更前後の両方の起動時に `backend.$=backend.@deterministic` を指定する。Atari では前後の起動時に `E1.game=qbert` も指定し、実効設定の `AtariEnv.game` が `qbert` であることを確かめる。出荷ファイルと Run プロファイルを使い、比較のための指定は起動時に渡す。比較に使う前後両方の Run で、`backend.deterministic_algorithms` と `backend.cudnn_deterministic` の実効値がともに `true` であることを確認し、実装計画に記録する。

合否は、道具（`src/core/base/testdata/rfd0097/compare_metrics.py`）の出力で決める。次の 2 つをどちらも満たせば一致、どちらかを満たさなければ不一致とする。

1. 段階の前の Run が、比べる中身を持つ。道具に `--output <ファイル>` を付けて書き出した JSON の `old.counts`（標準出力には出ない）で、`scalar:37_agent_qtd/01_td_mean`（学習の TD 誤差）と `scalar:20_eps/10_train_reward`（学習側の episode の return）の件数が、どちらも 1 以上である。満たさなければ、Run の長さを learn の step 10,000 まで伸ばして、前後の両方を取り直す。それでも満たさなければ、受入は不成立とし、実装を止めて開発者に示す。
2. 道具の `exact_match` が真である。道具は、`metrics.jsonl` の `scalar` と `trace` の全レコード（tag・step・値）を比べ、片方にだけあるレコード（tag が片方に無い、点数が違う、値が違う）が 1 つでもあれば偽にする。値が NaN どうし、値なしどうしのレコードは、同じ tag・同じ step なら一致とする（道具が、値を 16 進の文字列にして比べるため。P132）。

除くのは、実時間と資源に依る `90_perf/`・`91_resource/`・`92_resource/` で始まる tag である（RFD0097 の比べ方と同じ）。道具の `--allow-lives-values` と `--allow-migrated-edges` は付けない。checkpoint の raw checksum は使わない。

S1 は括弧までを含める（`replay_buffer_test.cpp` の `PushTimeWithActionInfo` を数えないため）。今の当たりは、S1 が `src` の 11 ファイル 24 行、S2 が 5 ファイル 10 行、S3 が `trainer.hpp` の 2 行と `trainer.cpp` の 2 行（ほかに、数えない `EvalPanel` の 2 行）、S4 が `src` の 18 ファイル 74 行、S5 が `src` の 9 ファイル 52 行（資材を除く）と `rfd0072/baseline` の 17 ファイルと `apps` の 4 ファイル 17 行、S6 が `agent.txt` の 13 行、S7 が `src` の 6 ファイル 15 行、S9 が `src` のテスト 5 ファイル 12 行、S10 が `default_dqn_agent.hpp` の 13 行、S11 と S12 が合わせて `dqn_actor.cpp` の 2 行と `dqn_concrete_test.cpp` の 10 行である。新しい名前（`ForcedActions`、`ForwardCapture`、`perception.block`）は、S1〜S10 のどれにも当たらない。RND の設定キー `intrinsic.rnd.obs_norm.*` は、S6・S9・S10 のどれにも当たらない。

## 10. テスト項目

受入 3〜9、11〜14、16、21〜29、33、34 のテストを足す。消す `WithAction` のテスト（`dqn_actor_test.cpp`・`dqn_concrete_test.cpp`・`dqn_munchausen_test.cpp`・`replay_buffer_test.cpp` の呼び出しと、RFD0192 が足した device のテスト）は、期待値を変えずに、強制行動つきの `MakeAction` へ移す。既存の `[native_stack]` のテストは、D27・D32 に合わせて直す。等価性の Run（受入 10・17・30）は、段階の直前の HEAD で基準を取る。build と Run の起動は、実装する側が段取りを示して、開発者の承認を受けてから行う。

## 11. 成功の測り方

| 指標 | 測り方 |
|---|---|
| 観測キーを 1 つ足すときに触る箇所 | RFD0142 の実装で、4 つの経路と spec の関数を直さずに、部品 1 つと設定で済む |
| 形の食い違い | 受入 23 が通り、枚数 1 の設定でも、行動選択・Sample・可視化の Perception と、Network の入力 spec の形が同じである |
| 強制行動のための専用の仕組み | 受入 1・2 が通り、`WithAction` と `munchausen_terms` が無い |
| 待っている RFD | 142・159・160・173・174・183 が、本 RFD の口を前提に再グリルできる |

先送りしたものは、統計つきの正規化の成立（NG2）、RND の観測加工を同じ仕組みに載せること（NG3）、正規化の部品の内部の整理（NG4）、逆方向の口（D6）、Network の出力の spec（D22）、図の出し分けの設定（§7.12）である。どれも別の関心によるもので、作る条件は、それぞれの箇所に書いた。

## 12. 未決

| # | 論点 | 確かめ方 | 結果ごとの分かれ方 |
|---|---|---|---|
| U1 | Frame にキーを足す部品で、行動選択側と Sample 側の Perception が往復で一致するか（P36） | RFD0142 が、受入 22 と同じテストに自分の部品を載せる | 一致すれば、そのまま。一致しなければ、部品の依頼ごとの作り方だけを直し、4 つの口（§7.4）は変えない |
| U2 | 段階 B で、GridMaze の checksum が一致するか（Network の入力 spec の形が `[2]` から `[1,2]` に変わる） | 受入 30 | 一致すれば、そのまま。一致しなければ、原因を調べて、実装を止めて開発者に示す（形の変更に伴う初期化や乱数の順の変化なら、受入の扱いを開発者が決める） |
| U3 | 出荷設定に、stack 軸つきのまま、宣言の無い module へ通す Network の定義が、DropMerge の Embedder のほかにあるか（Embedder の 2 種は、軸を自分で扱うので宣言に入れた。D32、P126） | 実装の入口で、出荷設定を組んで WARN を数える（§7.9） | 無ければ、そのまま。あれば、その module が軸を自分で扱っているなら宣言を足し、扱っていないなら、定義の先頭に module を足すかを開発者に聞く |
| U4 | Atari の等価性の Run（learn の step が 1,000 に達するまで）に、どれだけ時間がかかるか | 実装の入口で Q*bert の基準 Run を流した。1,000 学習 step で 163.94 秒だった（[実装記録](0161_observation_pipeline_20impl.md#経過)） | 2026-11-23 に開発者は Q*bert の学習 Run を選んだ（§13）。起動時の指定は §9 による |

## 13. グリルの材料

決定ではなく、決定に至る材料である。

**開発者の発言**（日付を書いていない行は 2026-11-20。`ValueScale::Fixed` の行とその次の行は 2026-11-21）

- 2026-11-23「同一性の確認テストは決定論設定で行う必要があります」（D35、§9）。
- 2026-11-23「1」── Atari の比較方法を選ぶ問いへの回答。Q*bert の学習 Run を使う案を選択した（U4、[基準 Run の結果と選択の記録](0161_observation_pipeline_20impl.md#経過)）。

- 「パイプライン」をどのようにどの範囲でインタフェース化するか、慎重に調整したい。TensorDict を入れたら加工した TensorDict が出てくるインタフェースのように広く考えることもできるし、Observation を対象にする前提でインタフェース化することも考えられる。
- イメージは NN みたいに、ブロック定義的に処理ステージを定義して、それらの組み合わせ順番で組み立てる。NN との統合はやり過ぎかもだが、backward しない以外、類似点が多い。GPU/CPU の両対応にしたい。
- Observation は低レベルの（生の）観測情報で、それや関連情報を加工集約してから NN に入れる。この統合したものに名前を付けたい。Perception（知覚）でどうか。
- RB の登録単位を Frame と呼ぶとしたら、関連要素との関係性やデータフローはどうなるか。
- アルゴリズム設定と構造定義設定の境界が気になる。NN では Head を実装側の生成にすることで最低限の境界を維持している。NGU の有効無効のような設定は、Perception の構築に影響する。これも構造定義に任せると、柔軟だが統制が取れなくなる懸念がある。
- Observation 等 → Frame → Perception というデータフローの関係を、フレームワーク全体の前提に組み込むのは危険ではないか。
- （往復の一致の確かめは）RFD0142 で確認。
- Stacking する以上キューができるが、それが Stacking のモジュール単体の持ち物になるのは勿体ない。N-STEP は今回の論点にしなくてよいか。
- Frame をもとに Perception を作るというような依存関係は想定しない（結果そうなるかもしれないけど）、でよいか。
- 方策インデックスなどは、キーやデータ型というより、それ自体の性質として正規化の対象外で、注入する時点で決まっている。`TensorSpecMap` で TensorDict 自体の定義として注入するのが素直。`TensorSpecMap` も初期化時にチェインして生成する感じか。
- 値の種類の属性も今回足す。後回しにする理由がない。
- Network クラスと共通する上位インターフェイスとして、Forward(TensorDict)→TensorDict を含んだ「変換できる」インターフェイスを切りたい。
- 連続値を正規化しても量は量のままなので、正規化の対象外であることを（種類の属性では）表現できないのではないか。
- 似た話が Stacking にもある。重ねるべき属性かどうか、というより、その処理を通過済みか。
- （設定の境界は）案 Y でよい（案 Y は、何の情報を入れるかはアルゴリズム設定が決めて実装が作り、どういう形で入れるかだけを設定の並びで組む形で、D3 になった）。ただし実装前に、想定される限りのパターンを机上でシミュレーションして、矛盾が起きないか整理しておく。特定の Key を Drop するような、汎用的な Forwarder（仮称）も用意するとよい。
- TorchRL の Transform は順方向と逆方向で双方向なのか。概念が複雑になりすぎる懸念の一方で、入れるときと出すときで整合を保つ場合はよい方法だと思う。
- 次は RFD0191 を予定していて、後続はそんなに急ぎではない。グリルの作業効率優先で考える。
- `forced_action` は、指定するほとんどのケースでシングルレーンなので、その場合の実装のシンプルさも確保したい。Tensor でまとめて指定と、特定の Lane だけ指定は両極端だが、どちらかだけというわけにもいかない。`ForLane`/`ForMask` は違和感があるので `ByLane`/`ByMask` で。
- Stack 無しの場合でも (B, S, *) の形に統一するのはやりすぎか。スタック有無で形状の意味が変わるのが面倒。
- （先頭の module で扱いを書く形を選ぶ理由）スタック次元の扱いを曖昧にしない方が良い。
- Forwarder の集まりで作るのは Perception だけではないので、`PerceptionBlock` は変ではないか。`ForwarderExecutor` だけ変で、`ForwardExecutor` か。
- `ValueScale::Fixed` は表現として大丈夫か。`Scaled` でなくてよいか。`has_time_axis` はスタック次元のことか。構築時に seed を渡すようにしておいたほうが良いかも。`NetworkForwardCapture` には対応しないか。速度やメモリに影響しなければ、インタフェース統合の面で最初から入れておきたい。
- NN の先頭の module の書き忘れは、曖昧にしない、は同時に強制しない、でもあるので、一律エラーはちょっと。WARN ログにするか。
- （2026-11-22。別モデルの検討の「Network にも共通の口を実装させるか」への答え）Forwarder インターフェイスとしては、その名の通り Forward だけで、出力の Spec を取るメソッドは付けなくてよい。その拡張版として、Spec も取れる版を作ればよい。
- （2026-11-22。「正規化の統計の更新を、新しい作りのどこに書くか」への答え）Push でよい。`ForwardScene` の種類に Push を追加して、Push の場合だけ統計を更新する。Act でも Push でも同じ Forwarder の口を使うので、メソッドは `Forward` のままにする。Forward にモードを付ける話（部品の並びは同じで、Perception を作る用・Frame を作る用で挙動だけを切り替える）のとおりである。
- （2026-11-22。名前について）`ForwardScene` の命名は良くない。Scene は呼び出し側のコンテキストの都合で、メソッドに渡すべきなのは、何が欲しいか・何をして欲しいかという依頼の中身である。Push するのは呼び元で、依頼としては「Frame を作ってください」、もしくは「統計を更新しつつ Frame を作る」である。`FrameAndUpdateStats` は変で、And だと Frame と UpdateStats が同列に見える。前者は生成物で、後者は付随の処理である。`FrameWithStatsUpdate` でよい。
- （2026-11-22。「設定に並びを書かなかったときの扱い」への答え）structure が無い場合は、NN と同様に、エラーにしない。count は、デフォルトを 1 にするので、通常は気にしなくてよいはず。並びを書かない構成でも、必ず stack 軸を持つ（「何もしない並び」にすると stack 軸が付かない、と示されたうえでの選択）。
- （2026-11-22）残りの細部（等価性の Run に DropMerge と Atari を足す、aux の `raw_obs`・`norm_obs` を消す、並びに `FrameStack` を書かなかったときも先頭に 1 枚を置く）は、推奨でよい。あわせて、NN と同じ仕組みで、dot によるグラフの可視化を入れておきたい。
- （2026-11-22。レビューの指摘「正規化の設定キーが、モードで使うキーの分け方の決まりに合わない」への答え。案は、A が 1 つの部品に `mode` を置く、B が部品の種類で分けて素通しは並びに書かない、C が今の名前のまま移す、の 3 つ）A にする。
- （2026-11-22。レビューの指摘「クラス一覧の置き場が、機能グループになっていない」を受けた問いへの答え。案は、Y が 3 つに分ける（口の `forwarder`、並びの仕組みの `forwarder_chain`、部品の `forwarder_parts`）、X が 2 つに分ける、Z が `forwarder` の 1 つに全部を入れる、の 3 つで、AI は Y を推した）Z。forwarder を使いたい、もしくは作りたいときは、とりあえず `anet/forwarder.hpp` をインクルードする、がフレームワーク方針である。
- （2026-11-22。再レビューの指摘「stack の対象のキーを選ぶスコープの名前 `target` が、target network の `target` と紛れる」を受けた問いへの答え。案は、A が `key_selection.mode` と `key_selection.listed.keys`（AI の推奨）、B が `keys.mode` と `keys.listed.names` の 2 つ）B: keys。

**AI の案と、確かめきれていない点**

- 口の形を 5 案で比べた（関数だけ、定義の持ち主のクラス、経路ごとの口、汎用の変換の列、外側は経路ごとの口で内側は NN と同じ作法の並び）。5 つ目が今の形のもとになった。
- 事例として、TorchRL の Transform（spec を変換する口を持つ。重ねる処理は収集側と sample 側の 2 通りに分かれる）、RLlib の ConnectorV2、Stable-Baselines3 の VecNormalize を公式の文書とソースで確かめた（2026-11-20）。RLlib の `ExternalEnv` は、ファイルが見つからず確かめられていない。
- `Scaled`・`Unscaled` が機械学習の一般の言い方（feature scaling、unscaled features）に合う、というのは記憶による。
- D21 の「速度とメモリは変わらない」と §7.7 の見立ては、今の Network の作りからの見立てで、測っていない。
- D32 の「先頭が `Linear` のとき、書き忘れが構築で止まらない」は、Head の作り（P115）からの見込みで、動かして確かめていない。
- 起草では、`FrameStack` の `count` と `perception.structure` を、AI が問いにせず必須（無ければエラー）と置いた。別モデルの検討の問いで、開発者が「エラーにしない。`count` の既定は 1」と決め、並びを書かないときも必ず stack 軸を持つ形にした（D27）。
- 並びを書いたが `FrameStack` を書いていないとき（`structure = norm` など）も、先頭に 1 枚の `FrameStack` を置いて組む、としたのは、上の決定から AI が置いた。
- Agent の外に共通のカタログを置かないこと、メトリクスの鍵 `obs_norm.*` を変えないこと（D25）は、問いにせず、D25 から AI が置いた。
- D43 のモードの名前（`none`・`constant`・`running`）と、`mode` の既定を `running` にすることは、問いの中で AI が示した案である。
- `FrameStack` の対象のキーを `keys.mode`（`all`・`listed`）で選ぶ形（D44）と、モードの値の名前は、レビューの指摘（空の一覧に「全キー」の意味を持たせている）を受けて、設定値の扱いの決まりと D43 の形から AI が置いた。問いにはしていない。スコープの名前は、AI が初めに `target` と置き、再レビューで、target network の `target` と紛れると指摘されて、開発者に聞いた（上の発言）。
- レビューの指摘への直しのうち、`ByMask` で印が偽の lane の行動を検査しないこと（D10）、`structure` を NN と同じ関数で読むこと（§7.6）、spec を通す口の名前 `ForwardSpec`（D20）、`Scaled` の定義の言い直し（D17）は、問いにせず AI が置いた。
- 再レビューの指摘への直しのうち、並びの入力の spec に stack 軸を持つキーがあればエラーにすること（D27）、正規化の対象のキーが無い構成で `obs_norm.*` が値を返さないこと（D40）、`constant` の定数の要素数を構築時に照合すること（§7.6 の 8）、等価性の Run を伸ばす上限（learn の step 10,000）は、問いにせず AI が置いた。3 回目のレビューの指摘への直しのうち、並びの入力の spec にキーが 1 つも無い構成をエラーにすること（§7.6 の 6）と、定数の一覧の、省略・空・要素数の不一致の扱い（§7.5、§7.6 の 8）も、同じく AI が置いた。

## 14. 旧ドラフトの未決の行き先

| 旧 | 内容 | 行き先 |
|---|---|---|
| Q1 | 単位 A を別の RFD に切り出すか | D5 |
| Q2 | `ObservationPipeline` をクラスにするか | D20・D26（§6.6 の 1 行目） |
| Q3 | Actor と Learner が直接参照するか | D31（Actor が実行器を持ち、Learner が Sample 用の並びを当てる） |
| Q4 | stack ありの素通しでない正規化を構築時に止めるか | D28（止めない） |
| Q5 | 部品へ渡す `stack_keys` | D27（構築時に spec と照らして決める） |
| Q6 | ファイル配置 | §7.1（クラス一覧の置き場と、機能グループの見込み。D45）。ファイルの中での並べ方は、RFD の契約に効かない実装の詳細として、実装計画で決める |
| Q7 | `MakeAction` の引数の形 | D7 |
| Q8 | `EvalRunner::DoStep` の負数 | D8 |
| Q9 | 設定キー `stucker.*` の綴り | D25（キーごと移すので、綴りの問題は消える） |
| Q10 | ADR の本数 | D37 |
| Q11 | MuZero の強制 lane | D14 |
| §4.1〜§4.5 | 3 つの置き場、`ObservationPipeline`、経路ごとの置換、スタック段の規則、バッチ段の歯止め | D20・D25〜D31。「1 枚は軸なし」と「構築時に止める」は改めた（D27・D28） |
| §4.6 | `ForcedAction`（印と行動）と契約 1〜8 | D7〜D16 |
| §10 | 統計つきの正規化の持ち越し | NG2。「stack 軸へ統計を広げる」だけ今回入れた（D28） |

## 15. 持ち越し（ほかの RFD・課題）

| 事項 | 行き先 | 本 RFD に残したもの |
|---|---|---|
| 直前の行動のキーと設定キー、部品の依頼ごとの作り方、Frame にキーを足す部品の往復の確かめ | RFD0142 | 置き場と口。D2（Agent が足す情報は、行動選択で 1 回作り、行動の情報に載せて受入れへ運ぶ）と RFD0142 のドラフト（Actor と受入れ側に 1 つずつ）の違いが、再グリルの材料になる。stack の既定に足したキーを含めるか、も RFD0142 で決める |
| 直前の報酬を行動選択側へ届ける口、別の方策番号として学ぶ形、方策番号を重ねるか | RFD0173 | Sample の口が Sample の結果を受けること |
| Train 側で、人の操作を強制行動として届ける配線。MuZero の Train での強制 | RFD0160 | Runner と Actor の口（Train 側にも開く）。RFD0160 の「lane の印と行動」の書き方は `ForcedActions` と違う |
| 統計つきの正規化を成立させる（競合、checkpoint、統計の凍結）。正規化を NN の block にする案 | [issue-20261122-258ef8](../management/issues/issue-20261122-258ef8.md)（統計の読み書きの競合）と、[issue-20261113-b96b2c](../management/issues/issue-20261113-b96b2c.md)（checkpoint への保存。追記した）。NG2。着手は、統計つきの正規化を使う構成を足すとき | 形の不成立だけを直した（D28） |
| RND の観測加工を同じ仕組みに載せる | [issue-20261122-ba42c9](../management/issues/issue-20261122-ba42c9.md)。NG3。着手は、RFD0191 の実装が済み、RND の入力に Q の側と同じ部品を当てる構成が要るとき | — |
| 正規化の部品の内部の整理（課題 aa5758 の 2026-11-11 の追記の 3 点） | [issue-20261111-5f695d](../management/issues/issue-20261111-5f695d.md)（追記した）。NG4。着手は、形の違う float のキーが複数ある構成で、統計つきか定数の正規化を使うとき | — |
| NN の入口が、離散の Grid の stack 軸とクラスを暗黙にチャネルへ畳むこと。「軸の扱いを曖昧にしない」（D27・D32）に照らしてそろえるか | [issue-20261122-4b9e2d](../management/issues/issue-20261122-4b9e2d.md)。着手は、離散の Grid を、stack 軸を残したまま扱う module を足すとき | 見分け方だけを属性に直した（D32） |

## 16. 関連

- [0142_agent_prev_action_obs_10rfd.md](0142_agent_prev_action_obs_10rfd.md)、[0159_input_mapping_joystick_10rfd.md](0159_input_mapping_joystick_10rfd.md)、[0160_input_target_train_forced_action_10rfd.md](0160_input_target_train_forced_action_10rfd.md)、[0173_intrinsic_reward_novelty_10rfd.md](0173_intrinsic_reward_novelty_10rfd.md)、[0174_llm_actor_10rfd.md](0174_llm_actor_10rfd.md)、[0183_temporl_skip_followups_10rfd.md](0183_temporl_skip_followups_10rfd.md) — 本 RFD を前提にする RFD。
- [0083_action_context_retire_10rfd.md](done/0083_action_context_retire_10rfd.md) — 前の RFD。容器・用語・ADR を本 RFD へ送った。
- [0057_native_stack_dim_handling_10rfd.md](done/0057_native_stack_dim_handling_10rfd.md) — stack 軸を Network の入力 spec に持ち、先頭の module で扱いを書く決まりの出所。
- [0170_nn_trace_capture_unification_10rfd.md](0170_nn_trace_capture_unification_10rfd.md) — `MakeActionOptions` と capture の出所。
- [0098_temporl_skip_policy_10rfd.md](0098_temporl_skip_policy_10rfd.md) — skip の継続状態と、判断のフラグ。
- [ADR 0013](../adr/0013-actor-network-resource-policy.md)、[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)。
