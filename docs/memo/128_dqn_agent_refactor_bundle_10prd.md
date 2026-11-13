# PRD128: DQN 系 Agent まわりのリファクタリング 5 件をまとめて片づける（価値の分解の前に） 暫定 PRD

> 2026-11-12 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §8 の未決表に置いた。§5〜§7 は §8 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: 2026-11-12 の [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)（価値の分解）のグリル。開発者は、PRD121 と同じ場所を触るリファクタリングの課題 5 件を、PRD121 の実装より先に片づけると選び（PRD121 の D18）、5 件を 1 本の暫定 PRD にまとめるよう指示した。直し方の中身について、会話で出た方向は無い。
> 題材: §9 の課題 5 件。
> 関連: [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)・[PRD924](924_intrinsic_reward_novelty_10prd.md)（同じ場所へ変更を足す PRD）、[PRD125](125_dqn_learner_update_skeleton_10prd.md)・[PRD122](122_small_refactor_and_test_gap_fixes_10prd.md)（挙動を変えない直しと、短い Run の前後比較の前例）、[PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md)（同じファイルを触る）、[PRD937](937_nn_head_projection_separation_10prd.md)（Head の射影の分離）、[PRD938](938_nn_trace_capture_unification_10prd.md)（Learner の購読の見直し案）、[リファクタリング観点](../quality/refactor.md)。

## 1. 目的と問題

### 1.1 問題

2026-11-10〜11 のリファクタリング候補の抽出で、DQN 系の Agent まわりに、重複と散らばりが 5 件見つかった（§2）。どれも、価値の分解（PRD121）が変更を足す場所にある。

| 場所 | PRD121 が足すもの | 今のままだと |
|---|---|---|
| 取り込みの内的報酬の扱い | 内的報酬用の報酬スケーラ、成分での受け渡し、保存する状態 | 5 つの関数と 6 つのメンバに散った扱いが、さらに膨らむ |
| 標本のフィールドの並び | 成分ごとの収益 | フィールドを 1 つ足すたびに 4〜5 か所を直す |
| Head の選択と Agent の構築 | 合成 Head | DefaultDQNAgent と RainbowAgent に写した分岐の、食い違いが増える |
| 設定の読み取り | 設定キー 10 個ほど | 534 行の 1 つの関数に、読む行と検証する行が離れて足される |
| 診断指標の群 | Head ごとの損失などのメトリクス | 群を足すたびに 12 か所前後を直し、添字の数値の直書きが増える |

### 1.2 ゴール

- **G1**: §9 の課題 5 件が、挙動を変えずにすべて片付く。
- **G2**: PRD121 の実装が、上の表の場所ごとに 1 か所へ足せば済む形になる。
- **G3**: 挙動を変えていないことを、実装前に保存した実行体との前後比較で確かめられる（§7）。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 理由 | 本当の制約か |
|---|---|---|
| PRD121 の機能（価値の分解、報酬の成分、報酬スケーラの種類）を足すこと | 機能の追加と、挙動を変えない直しを 1 本に混ぜると、前後の比較で差が出たときに原因を分けにくい（PRD121 の D18） | 制約ではない（PRD121 が本 PRD のあとに足す） |
| RainbowAgent を消すこと | Agent の削除は挙動の変更で、別の判断が要る（§8 の 2） | 制約ではない |
| ボーナスの診断の `.item<float>()` 6 回を 1 回の転送にまとめること | 構造の直しとは別の変更で、速さを測ってから決める（§8 の 7） | 制約ではない |
| Learner の購読を「開始前の宣言」と「呼び出しごとの要求」に分ける見直し | 別の関心（PRD938 §8 の 1） | 制約ではない |
| Head の射影の分離 | 別の関心で、PRD937 が持つ | 制約ではない |
| ソースの配置（ファイルの分け方）を変えること | 配置は別の課題が扱う（[リファクタリング観点](../quality/refactor.md) の「候補にしないもの」） | 制約ではない |

### 1.4 機構とゴールの対応

新しい機構は足さない。足す部品は、散った処理を寄せる先の 3 つである（§6.2）。

| 部品 | ゴールへの寄与 |
|---|---|
| 内的報酬の扱いをまとめるクラス | G1（08d557）、G2（取り込み） |
| DQN 系の Agent が共有する構築の関数 | G1（72d640）、G2（Head の選択） |
| 診断指標の群の共通の型 | G1（24d885）、G2（診断） |

### 1.5 成功の測り方

- §7 の前後比較が、5 つのステージすべてで一致する。
- PRD121 の実装計画が、§1.1 の表の場所ごとに、足す先を 1 か所ずつ挙げられる。

## 2. 題材の課題と現状（2026-11-12 にコードで確認）

行番号は 2026-11-12 の作業ツリーの位置である。同じファイルを触る作業が並行しているので、実装時は関数の名前と近くの記述で探し直す。

### 2.1 内的報酬の扱いが散っている（08d557）

- 内的報酬だけのメンバが、`DefaultDQNAgent` に並んでいる（[default_dqn_agent.hpp:728-734](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
- `DefaultDQNAgent::UpdateFromBatch`（[default_dqn_agent.cpp:831](../../core/anet-core/src/default_dqn_agent.cpp)）の中に、ボーナスの計算、診断値の公開、credit の加算、lane の抽選、補助学習のループが並ぶ。
- 推定器の診断値を写すループが、構築・読み込み・更新の 3 か所にある。
- 保存（`Save`、[同 333](../../core/anet-core/src/default_dqn_agent.cpp)）と読み込み（`LoadNetwork`、[同 384](../../core/anet-core/src/default_dqn_agent.cpp)）で、書く順と読む順を手で合わせている。
- lane の抽選の乱数だけ `std::mt19937_64` で、同じクラスのほかの乱数は `anet::RandomGenerator` である。

### 2.2 標本のフィールドの並びが分かれている（288879）

- `ExperienceSamples::To`（[rl.cpp:773](../../core/anet-core/src/rl.cpp)）は、CUDA へ送る経路と CPU の経路で、全フィールドの並びを 2 回書いている。
- 2 つの並びは、未定義の tensor の扱い、CPU に置くフィールドの書き方、stream guard の有無の 3 点で食い違う。
- `ForEachTensor`（[rl.hpp:889](../../core/anet-core/include/anet/rl.hpp)）も全フィールドを並べる。`ToString`（[rl.cpp:843](../../core/anet-core/src/rl.cpp)）は一部だけを並べる。
- 学習の毎回の転送の経路である。

### 2.3 Agent の構築と、Rainbow に写した分岐（72d640）

- `DefaultDQNAgent` のコンストラクタ（[default_dqn_agent.cpp:52](../../core/anet-core/src/default_dqn_agent.cpp)）の 1 つの関数に、seed、内的報酬、スケーラ、Head の Factory の選択、forward 設定、spec、skip の定義、モデル、構造図、target の方策、Learner、読み込みが並ぶ。
- Head の選択（[同 148-170](../../core/anet-core/src/default_dqn_agent.cpp)）は、分布の表し方 3 値と Dueling の有無の組を、入れ子の分岐で書く。Learner の構築 3 つ（[同 262-281](../../core/anet-core/src/default_dqn_agent.cpp)）は、型を除いて引数が同じである。
- `RainbowAgent` が、Head の選択（[rainbow_agent.cpp:66-80](../../core/anet-core/src/rainbow_agent.cpp)）、Actor の生成、値の振り分け、更新を写している。課題は、すでに 5 点が食い違っていると書いている。(a) 方策と τ の検証を呼ばない。(b) 方策の種類を見ずに常に ε-greedy を作る。(c) 設定の記録の仕方が違う。(d) `net.body` と構造図の記録が無い。(e) `UpdateFromBatch` の公開範囲が違う。
- 出荷の設定で RainbowAgent を選ぶ行は、コメントになっている（[common.txt:29](../../apps/runner/config/common.txt)）。

### 2.4 設定の読み取りが 1 つの関数に並ぶ（e33797）

- `DefaultDQNAgentConfig` のコンストラクタ（[default_dqn_agent.hpp:116](../../core/anet-core/include/anet/default_dqn_agent.hpp) から）に、読み取り・検証・既定値の導出・記録が並ぶ。
- 「値が決まった語のどれかであること」の検証が同じ形で繰り返され、エラー文の候補の並べ方が 4 通りある。
- 「接頭辞の下の設定キーをそのまま写して記録する」処理が 4 か所に手書きされている（[同 147・162・201・481](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。写す先のキーの形が場所ごとに違う。

### 2.5 診断指標の群が同じ形を繰り返す（24d885）

- 「指標の番号と間隔」の組を表す型が、群ごとに 6 つある（[dqn_based_agent.hpp:961・974・987・996・1042・1081](../../core/anet-core/src/dqn_based_agent.hpp)）。
- キー配列から添字を引く関数、発火した指標だけ要求の旗を立てるループ、読み出しの形も、群ごとに繰り返す。
- `Learner::ConfigureScalarMetricSubscriptions`（[dqn_based_agent.cpp:2750](../../core/anet-core/src/dqn_based_agent.cpp)）は、購読の振り分け・設定値の検証・WARN・Resource の確保を 1 つの関数に並べる。
- キー配列や pack の添字を、数値で直書きしている。配列へ 1 つ挿すと、離れた場所の番号が黙って別の指標を指す。

## 3. 用語

新しい語は足さない。[CONTEXT.md](../../CONTEXT.md) を変えない。

## 4. 規約と過去の判断

- 挙動を変えない直しでは、hot path の速さが変わらないことを Run のブレ幅で確かめる。乱数の呼び出し順に触るなら、同 seed の再現性を確かめる。テストの無い箇所は、特性テストを先に足す（[リファクタリング観点](../quality/refactor.md) の「ガードレール」）。
- 「挙動を変えない」には、ログの文言と WARN の出る回数を含める。変わる直しは、変わることを書く（同「運用ルール」）。
- 学習器や Agent を変えるときは、乱数の派生と消費の順を変えず、同 seed・決定的モードの短い Run のメトリクスを、実装の前に保存した実行体と比べる（[324 DQN](../wiki/categories/324_dqn.md) の「変えるときの条件」）。
- 短い Run の前後比較は、PRD122 と PRD125 が同じファイルで行った。変更前の実行体と設定を保存し、seed 1・決定的モードで流して、時間と資源の tag を除く全 scalar の点を比べる（[PRD122 の検証資材](../../core/anet-core/testdata/prd122/README.md)）。
- もとからある死にコードは、消さずに報告する（[AGENTS.md](../../AGENTS.md) の「LLM コーディング規律」）。

性能の区分（リファクタリング観点の「ガードレール」）は次のとおり。

| 課題 | 区分 | 確かめ方 |
|---|---|---|
| 08d557 | 中立（step ごとの取り込みの経路）。lane の抽選の乱数に触る | 速さをブレ幅で確かめる。同 seed の前後比較 |
| 288879 | 中立（学習の毎回の転送） | 速さをブレ幅で確かめる |
| 72d640 | 関わらない（構築時） | 前後比較 |
| e33797 | 関わらない（設定の読み込み） | config dump の比較 |
| 24d885 | 中立（更新ごとの要求の組み立て） | 速さをブレ幅で確かめる |

## 5. 直し方（§8 の暫定の既定を前提）

### 5.1 ステージ

1 本の PRD で扱い、課題ごとの 5 つのステージに分ける（§8 の 1）。ステージ 0 で、実行体の保存と、比べる組の確かめを行う。各ステージの終わりに §7 の前後比較を通し、ステージごとにコミットできる。

| ステージ | 課題 | 順にする理由 |
|---|---|---|
| 0 | — | 実装前の実行体と設定を保存する。前後比較の組（§7 の 1）が、直す経路を通ることを確かめる |
| 1 | 288879 | 小さく、ほかと独立している |
| 2 | e33797 | 設定の読み取りだけで、ほかと独立している |
| 3 | 08d557 | Agent の構築（ステージ 4）から、内的報酬の部分を先に抜く |
| 4 | 72d640 | ステージ 3 のあとの、短くなったコンストラクタを分ける |
| 5 | 24d885 | Learner の診断だけで、ほかと独立している |

### 5.2 ステージごとの直し方

- **ステージ 1（標本の並び）。** `To` の並びを 1 つにし、「device へ送る」「CPU に置く」の 2 つの小さな関数だけを使う。`ForEachTensor` と `ToString` も同じ並びから作る。CPU の経路の未定義の tensor の扱いは、CUDA の経路と同じ（未定義は素通し）にそろえる（§8 の 5）。stream guard は今のまま残し、CUDA のときだけ同じ範囲で作る（§8 の 6）。
- **ステージ 2（設定の読み取り）。** コンストラクタを、関心ごとの private のメンバ関数に分ける。語の検証と下限の検証は、キーと候補を受ける小さな関数に寄せる。エラー文の文言は変えない（§8 の 4）。「接頭辞の下を写して記録する」関数を 1 つにし、写す先のキーの形の違いは引数で保つ（§8 の 15）。
- **ステージ 3（内的報酬）。** 内的報酬だけのメンバと、ボーナスの適用・補助学習・診断値・保存・読み込みを、1 つのクラスへ移す。Agent の側は「有効なら報酬へ適用する」「保存と読み込みを委ねる」「`intrinsic.` で始まる値の読み出しを委ねる」だけにする（§8 の 9）。乱数の型、保存の形式、乱数の消費の順は変えない（§8 の 8）。
- **ステージ 4（Agent の構築）。** RainbowAgent は残す（§8 の 2）。Head の Factory の選択、Actor の生成の共通部分、値の振り分けを、2 つの Agent が共有する関数へ出す。食い違い (a)〜(e) は、今の差を保つ（§8 の 3）。DefaultDQNAgent のコンストラクタは、「Head の Factory を選ぶ」「入力の spec を作る」「Learner を作る」「構造図を記録する」に分け、Learner の構築は引数の列を 1 回だけ書く。
- **ステージ 5（診断の群）。** 「キー配列・添字の解析・購読の間隔・発火した要求」を 1 つの型にまとめ、6 つの群がそれを 1 つずつ持つ。読み出しは、共通の関数 1 つにする。群ごとの添字は、キー配列と同じ場所に置いた名前で引く（§8 の 10）。probe 用の乱数の渡し口は変えない（§8 の 11）。

### 5.3 特性テスト

直す前に、次の特性テストがあるかを実装計画で確かめ、無ければ先に足す（§8 の 14）。

- `ExperienceSamples::To` の、CPU の経路と CUDA の経路それぞれでの、定義済み・未定義の tensor の扱い。
- 内的報酬の状態（credit、乱数の状態、回数、診断値）の、保存と読み込みの往復。
- RainbowAgent の構築と、食い違い (a)・(b) の今の挙動。

## 6. 実装契約（§8 の暫定の既定を前提）

### 6.1 変えないもの

設定キー、メトリクスの名前、TensorDict の鍵、保存の形式、ログと WARN と ERROR の文言・順・回数、seed の作り方、乱数の消費の順、forward の回数と順、公開ヘッダの口（`ExperienceSamples` の利用側から見た口、Agent の公開の関数）。

意図して変える点は 1 つだけである。`ExperienceSamples::To` の CPU の経路で、未定義の `actions`・`target_returns`・`next_state.terminals` を、止めずに素通しする（§8 の 5）。今の挙動は、ステージ 1 の特性テストで確かめてから実装計画に書く。

### 6.2 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| ExperienceSamples | 変更 | フィールドの並びを 1 つにし、転送・走査・文字列化が同じ並びを使う | RL 共通基盤（rl） |
| DefaultDQNAgentConfig | 変更 | 読み取りを関心ごとの関数に分ける。語と下限の検証、接頭辞の下の記録を、共通の関数に寄せる | DefaultDQNAgent（default_dqn_agent） |
| IntrinsicRewardCoordinator | 新規 | 内的報酬の採点結果の報酬への適用、補助学習の credit と lane の抽選、診断値、保存と読み込みを持つ | DefaultDQNAgent（default_dqn_agent） |
| DefaultDQNAgent | 変更 | 内的報酬の扱いを委ねる。構築を、役割ごとの private の関数に分ける | DefaultDQNAgent（default_dqn_agent） |
| RainbowAgent | 変更 | Head の Factory の選択、Actor の生成の共通部分、値の振り分けを、共有の関数から呼ぶ | RainbowAgent（rainbow_agent） |
| DqnComponentAssembler | 新規 | DQN 系の 2 つの Agent が共有する構築の関数（Head の Factory の選択、Actor の生成の共通部分、値の振り分け）を持つ | DQN 系（dqn_based_agent） |
| DiagnosticDemandTable | 新規 | 診断指標の群 1 つぶんの、キー配列・添字の解析・購読の間隔・発火した要求を持つ | DQN 系（dqn_based_agent） |
| dqn::Learner | 変更 | 6 つの群が共通の型を持つ。購読の設定を、振り分け・検証・Resource の確保に分ける | DQN 系（dqn_based_agent） |
| dqn::BatchUpdateResult | 変更 | 群の値の読み出しを、共通の関数 1 つにする | DQN 系（dqn_based_agent） |

### 6.3 移行対象

経路の検索（2026-11-12、`git grep -l -F`、範囲は追跡している `core`・`apps`・`docs`）で、直す関数と型を名指す箇所を確かめた。検索語と、当たったファイルの数は、`ExperienceSamples::To` 11、`ForEachTensor` 11、`DefaultDQNAgentConfig` 42、`intrinsic_learn_` 3、`lifelong_novelty_` 34、`RainbowAgent` 646、`ConfigureScalarMetricSubscriptions` 26、`DemandEntry` 3 である。過去の PRD・実装計画・ADR・課題・wiki・実験記録（書き換えない）を除くと、現用のものは次のとおり。

- **変えるコード**: [rl.hpp](../../core/anet-core/include/anet/rl.hpp)・[rl.cpp](../../core/anet-core/src/rl.cpp)・[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp)・[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp)・[rainbow_agent.hpp](../../core/anet-core/include/anet/rainbow_agent.hpp)・[rainbow_agent.cpp](../../core/anet-core/src/rainbow_agent.cpp)・[dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp)・[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp)。
- **検索に当たるが変えないコード**: 公開の口を変えないので、呼ぶ側は変えない。[init.cpp](../../core/anet-core/src/init.cpp)（RainbowAgent の登録）、[muzero_rb.cpp](../../core/anet-core/src/muzero_rb.cpp)（`ExperienceSamples::To` を呼ぶ）、[trainer.cpp](../../core/anet-core/src/trainer.cpp)（購読の設定を呼ぶ）。[tensor_util.hpp](../../core/anet-core/include/anet/tensor_util.hpp)・[transfer.hpp](../../core/anet-core/include/anet/transfer.hpp) の `ForEachTensor` は別の型の関数である。[image_cls_agent.hpp](../../core/anet-core/include/anet/image_cls_agent.hpp)・[image_cls_agent.cpp](../../core/anet-core/src/image_cls_agent.cpp) は、ImageClsAgent が持つ同じ形の診断の要求で、課題の範囲の外である（§8 の 21）。
- **テスト**: [dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp)・[intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp)・[replay_buffer_test.cpp](../../core/anet-core/src/replay_buffer_test.cpp) に、§5.3 の特性テストを足す。既存の断定の期待値は変えない（§6.1 の 1 点を除く）。[config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp)・[dqn_munchausen_test.cpp](../../core/anet-core/src/dqn_munchausen_test.cpp)・[nn_test.cpp](../../core/anet-core/src/nn_test.cpp)・[image_cls_agent_test.cpp](../../core/anet-core/src/image_cls_agent_test.cpp) は、検索に当たるが変えない（設定の型と購読の設定の、公開の口を使う側）。
- **設定**: 変えない。[agent.txt](../../apps/runner/config/agent.txt)・[common.txt](../../apps/runner/config/common.txt) と、`core/anet-core/testdata/prd072/` の解決済みの設定の基準は、検索に当たるが変わらない。
- **設計文書**: クラスの構成と、メンバや関数を名指している節を追随させる。[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md)（§3〜§5、§8.1）、[150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md)（`ForEachTensor` を名指す箇所）、[212_rnd_lifelong_novelty.jp.md](../design/212_rnd_lifelong_novelty.jp.md)（内的報酬のメンバを名指す箇所。§3.8 ほか）、[110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md)（購読の設定を名指す箇所）。[010_framework_overview.jp.md](../design/010_framework_overview.jp.md)・[100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) は、RainbowAgent の名前を挙げるだけなので変えない。新しい仕組みではないので、設計文書は新設しない（§8 の 18）。en は `anet-translate-docs` で作る。
- **検証資材**: `core/anet-core/testdata/prd128/README.md` を新設する（実装前の実行体と設定の保存、前後比較の手順）。200 の §10 から参照する。[prd116 の README](../../core/anet-core/testdata/prd116/README.md) は、内的報酬のメンバを名指す箇所を追随させる。`prd084`・`prd092`・`prd093` の資材は、過去の検証の記録なので変えない。
- **互換層**: 作らない。過去の Run・ADR・実験記録は書き換えない。

## 7. 受入の候補（§8 の暫定の既定を前提）

再グリルで契約を確定したあとに固める。前後比較は、各ステージの終わりに通す。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | メトリクスの前後比較 | 実装前に保存した実行体と、seed 1・決定的モードの短い Run で、metrics checksum が一致する。対象の構成は、Atari の `run.@rnd`、LunarLander の既定（skip 有効）、LunarLander の RND あり、GridMaze の既定、DropMerge の既定の 5 つで、区間は検証資材の README に書く。比べるのは全 scalar の点で、時間と資源の tag は除く（実時間で変わるため） |
| 2 | 設定の前後比較 | 同じ 5 つの構成で、Run の `config/config_data.txt` が一致する。Run 名と時刻を含む行は除く（Run ごとに変わるため） |
| 3 | ログの前後比較 | 同じ 5 つの構成で、ログの WARN と ERROR の行の文言と順が一致する。行の先頭の時刻は除く |
| 4 | checkpoint | 実装前の実行体が保存した、RND 有効の checkpoint を新しい実行体が読める。続きの短い Run の metrics checksum が、実装前の実行体で続けた場合と一致する |
| 5 | 特性テスト | §5.3 の特性テストが、直す前と直したあとの両方で通る。§6.1 の意図して変える 1 点だけは、期待値の変更が実装計画に書かれている |
| 6 | 標本の並び | 全フィールドに値を入れた標本で、`To`（CPU）、`To`（CUDA。使える環境で）、`ForEachTensor`、`ToString` が、同じフィールドの集合を扱う |
| 7 | RainbowAgent | RainbowAgent を構築するテストが通り、食い違い (a)・(b) の特性テストが、直す前と同じ結果になる |
| 8 | 速さ | ステージ 1・3・5 のそれぞれのあと、Atari の同じ構成の実時間を、前後の実行体で交互に 3 巡以上測る。群ごとの中央値の差が、同じ実行体どうしのブレ幅の内である |
| 9 | 添字の直書き | [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp)・[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp)・[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) を正規表現 `request\[[0-9]+\]` で引いて 0 件である |
| 10 | 既存のテスト | `anet-core-test` の既定のスイートがすべて通る |

## 8. 決めること（未決表）

暫定の起票なので、どの行も決まっていない。「暫定の既定」は AI が置いた既定で、§5〜§7 はこれを前提に書いた。再グリルですべて開き直す。

| # | 事項 | 暫定の既定 | 理由と依存 | ほかの候補 |
|---|---|---|---|---|
| 1 | 束ね方とステージ | 1 本の PRD で扱い、課題ごとの 5 つのステージに分ける。順は §5.1 | 5 件は同じ 8 つのファイルにあり、確かめ方が同じなので、保存した実行体と Run の組を使い回せる。ステージを分けるのは、前後比較で原因を絞るため（PRD125 の D1 と同じ考え方）。依存: なし | 課題ごとに PRD を分ける。規模の小さい 288879 と e33797 だけを先に出す |
| 2 | RainbowAgent を残すか | 残す。共通の部分を、2 つの Agent が共有する関数へ出す | Agent を消すのは挙動の変更で、挙動を変えない直しに混ぜない。出荷の設定で選ぶ行はコメントだが（§2.3）、テストと Factory の登録が構築している。依存: なし | 消す（DefaultDQN の設定で表せるなら）。消すなら別の PRD にする |
| 3 | Rainbow の食い違い 5 点の扱い | 5 点とも今の差を保つ。共有する関数に、差を引数で残す | 意図した差か直し漏れかが、コードから読めない。そろえると Rainbow の検証・方策・記録が変わる。依存: 2 | 直し漏れと決めたものだけそろえ、変わることを書く |
| 4 | 設定のエラー文の書き方（4 通り） | 文言を変えない。書き方ごとに関数を分けて保つ | 「挙動を変えない」にログの文言を含める（§4）。依存: なし | 1 つの書き方にそろえ、変わることを PRD に書く |
| 5 | `To` の CPU の経路の、未定義の tensor の扱い | CUDA の経路と同じ（未定義は素通し）にそろえる。変わる点として §6.1 に書く | 並びを 1 つにすると、扱いも 1 つになる。食い違いは片方だけを直した跡と読める（課題の記述）。今の CPU の経路の挙動は確かめていないので、特性テストで先に固定する。依存: 14 | 今の食い違いを保つ（並びは 1 つにし、経路ごとの扱いを引数で分ける） |
| 6 | `To` の stream guard | 今のまま残し、CUDA のときだけ同じ範囲で作る | 何を防ぐためかが、コードとコメントから読めない。消すのは挙動の変更になりうる。依存: なし | 要否を調べて、要らなければ消す（変わることを書く） |
| 7 | ボーナスの診断の `.item<float>()` 6 回 | 本 PRD では変えない | 構造の直しとは別の変更で、速さを測ってから決める（課題の対応方針）。依存: なし | 1 回の転送にまとめる（別の課題にする） |
| 8 | lane の抽選の乱数の種類（§2.1 の最後の項目） | 変えない | 種類を変えると、保存する乱数の状態の形式と、乱数の列が変わる。依存: なし | 同じクラスのほかの乱数と同じ種類にそろえる（保存の形式と再現性が変わることを書く） |
| 9 | 内的報酬のクラスの口 | 「報酬へ適用する」「保存と読み込み」「`intrinsic.` で始まる値の読み出し」の 3 つ。Agent が持つ | 課題の対応方針のとおり。PRD121 は、ここに内的報酬用の報酬スケーラと、成分での受け渡しを足す。依存: なし | 推定器の側（一生の新しさの口）へ寄せる |
| 10 | 診断の群の共通の型 | 6 つの群を 1 つの型に載せる。可塑性は 3 つのチャネルを持つので、載せられるかを実装計画で確かめ、載らなければ可塑性だけ今の形のまま、添字に名前を付ける | 課題の対応方針のとおり。依存: なし | 添字に名前を付けるだけにし、型はまとめない |
| 11 | probe 用の乱数の渡し口（コンストラクタと、購読の設定の引数の 2 つ） | 変えない | 渡し口を変えると、Agent の構築の口と、seed の派生の順に触る。依存: なし | 購読の設定の引数にそろえる |
| 12 | Learner の購読の見直し（開始前の宣言と、呼び出しごとの要求に分ける案） | 入れない | 別の関心（PRD938 §8 の 1）。ステージ 5 の型は、この見直しの土台になりうる。依存: 10 | 同時に入れる |
| 13 | 確かめ方 | §7 の 1〜4 と 8。構成は 5 つ | PRD122・PRD125 と同じ進め方（§4）。08d557 は lane の抽選に触るので、同 seed の比較が必須。依存: なし | 構成を増やす（Atari の skip 有効、QR、TD） |
| 14 | 特性テスト | §5.3 の 3 つを、直す前に足す | テストの無い箇所は、特性テストを先に足す（§4）。依存: なし | 前後比較だけで済ませる |
| 15 | 「接頭辞の下を写して記録する」関数の、写す先のキーの形 | 場所ごとの違い（完全なキーか、相対のキーか）を引数で保つ | config dump が変わらないことを優先する。違いの理由はコードから読めない。依存: なし | 1 つの形にそろえる（config dump が変わることを書く） |
| 16 | PRD937（Head の射影の分離）との順序 | 待たない | PRD937 が入ると Head の選択が簡単になるが、PRD121 を待たせる期間が延びる。依存: なし | PRD937 を先に入れる |
| 17 | PRD127 との順序 | PRD127 の実装が終わってから始める | PRD127 は、本 PRD と同じファイル（default_dqn_agent・dqn_based_agent）を触る。依存: なし | 並行して進める |
| 18 | 設計文書 | 新設しない。クラスの構成を書いている節だけを追随させる（§6.3） | 新しい仕組みではない。依存: なし | — |
| 19 | ADR | 起こさない | 戻しにくい判断が無い。RainbowAgent を消すと決めた場合は、別に考える。依存: 2 | — |
| 20 | 新しいクラスの名前（§6.2） | `IntrinsicRewardCoordinator`・`DqnComponentAssembler`・`DiagnosticDemandTable` | 役割を表す接尾辞を付ける（[実装 品質確認観点](../quality/impl.md) の「名前」）。再グリルと実装で見直してよい。依存: 9、10 | — |
| 21 | ImageClsAgent が持つ同じ形の診断の要求 | 本 PRD では触らない。共通の型は DQN 系に置く | 題材の課題の範囲は DQN 系の Learner である。ImageClsAgent は別の Agent で、前後比較の組（§7 の 1）が通らない。依存: 10 | 共通の型を Agent の共通基盤に置き、ImageClsAgent も載せる（前後比較に ImageCls の構成を足す） |

## 9. issue の対応表

| 課題 | 題 | area | 根拠の位置（2026-11-12） | 未決表 | 受入 |
|---|---|---|---|---|---|
| [issue-20261111-08d557](../management/issues/issue-20261111-08d557.md) | 内的報酬の扱いが DefaultDQNAgent の 5 つの関数と 6 つのメンバに散っている | 324_dqn | [default_dqn_agent.hpp:728-734](../../core/anet-core/include/anet/default_dqn_agent.hpp)、[default_dqn_agent.cpp:831](../../core/anet-core/src/default_dqn_agent.cpp) | 7、8、9、14 | 1、4、5、8 |
| [issue-20261111-288879](../management/issues/issue-20261111-288879.md) | ExperienceSamples のフィールドの並びが分かれ、未定義の tensor の扱いが食い違っている | 318_rl_core | [rl.cpp:773](../../core/anet-core/src/rl.cpp)、[rl.hpp:889](../../core/anet-core/include/anet/rl.hpp) | 5、6、14 | 1、5、6、8 |
| [issue-20261111-72d640](../management/issues/issue-20261111-72d640.md) | DefaultDQNAgent のコンストラクタが長く、RainbowAgent が写して 5 点で食い違っている | 324_dqn | [default_dqn_agent.cpp:52](../../core/anet-core/src/default_dqn_agent.cpp)、[rainbow_agent.cpp:66-80](../../core/anet-core/src/rainbow_agent.cpp) | 2、3、14、16 | 1、3、7 |
| [issue-20261111-e33797](../management/issues/issue-20261111-e33797.md) | DefaultDQNAgentConfig のコンストラクタが 534 行あり、検証と記録が手書きで繰り返されている | 324_dqn | [default_dqn_agent.hpp:116](../../core/anet-core/include/anet/default_dqn_agent.hpp) | 4、15 | 2、3 |
| [issue-20261111-24d885](../management/issues/issue-20261111-24d885.md) | 診断指標の群を 1 つ足すたびに 12 か所前後へ足し、添字も数値で直書きしている | 321_agent_core | [dqn_based_agent.hpp:961](../../core/anet-core/src/dqn_based_agent.hpp)、[dqn_based_agent.cpp:2750](../../core/anet-core/src/dqn_based_agent.cpp) | 10、11、12 | 1、8、9 |

## 10. ほかの PRD との関係

- [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)（価値の分解）は、本 PRD のあとに実装する（PRD121 の D18）。PRD121 の課題は、§9 の 5 件を待つ。
- [PRD924](924_intrinsic_reward_novelty_10prd.md)（内的報酬と方策の組）も、内的報酬の扱いと設定を広げる。本 PRD の直しのあとに足す。
- [PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md) は同じファイルを触る（§8 の 17）。
- [PRD938](938_nn_trace_capture_unification_10prd.md) の再グリルで出た、Learner の購読の見直し案は入れない（§8 の 12）。
