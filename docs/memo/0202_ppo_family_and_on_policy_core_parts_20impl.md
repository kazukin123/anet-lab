# PPO 系と方策ベースの学習の共通部品 実装メモ

## 概要

[RFD0202](0202_ppo_family_and_on_policy_core_parts_10rfd.md) に従い、方策ベースの Agent の系統「PPO 系」（`src/algo/ppo`、`anet-algo-ppo`）と、離散行動の `PPOAgent` を足す。あわせて、PPO が使う共通部品を core（`src/core/base`）に足す。DQN 系（`src/algo/dqn/`）と、Runner・Observer・`rl.hpp` の今ある口は変えない。

RFD は暫定の起票で、確定事項の節を持たない。実装は、RFD の実装契約（§7）、移行対象（§8）、受入（§9）と、未決表（§12）の暫定の既定に従う。既定値は CleanRL の `ppo.py` の既定（RFD §2.7）である。

## 主な変更

### core（`src/core/base`）

- `anet/distribution.hpp`・`distribution.cpp`（新設）: `anet::rl::CategoricalDistribution`。行ごとの logits から作り、抽選（`Sample`）、確率が最大の行動（`GetMode`）、log 確率（`ComputeLogProb`）、entropy（`ComputeEntropy`）、全行動の確率と log 確率（`GetProbs`・`GetLogProbs`）を、logits と同じ device で計算する。
- `anet/advantage.hpp`・`advantage.cpp`（新設）: `anet::rl::ComputeGae` と `GaeResult`。時刻 × lane の報酬・value・次の観測の value・終端の印・episode の終わりの印から、advantage と value の目標を作る。
- `anet/nn_head.hpp`・`nn_head.cpp`: `anet::nn::CompositeHeadEntry` と `CompositeHeadFactory`。名前・Head が受け取るキー・factory の組から、1 つの Head を作る。中の Head は組の名前で登録する。
- `nn_modules.cpp`: 活性化の module の種類 `Tanh` を足す。
- `anet/tensor_store.hpp`・`tensor_store.cpp`: `LaneTimeTensorStore::GetAll`。たまっている範囲の全体を、時刻 × lane の形で複製せずに返す。

### PPO 系（`src/algo/ppo`）

- `CMakeLists.txt`、`include/anet/algo/ppo.hpp`、`src/ppo.cpp`: ライブラリ `anet-algo-ppo`、テスト実行体 `anet-algo-ppo-test`、初期化関数 `anet::rl::algo::InitPpo()`。
- `src/ppo_config.hpp`・`.cpp`: `PPOAgentConfig`・`ActorConfig`・`LearnerConfig`。RFD §7.4 の設定キーの読み取りと検証。
- `src/ppo_network_model.hpp`・`.cpp`: `NetworkModel`。`CompositeHeadFactory` と `LinearHeadFactory` で方策と価値の Head を持つ Network を組む。観測の各キーに長さ 1 の stack 軸を足して Network の入力にする。写し、保存と読込、方策の世代、勾配の内訳に使う parameter の組を持つ。
- `src/ppo_math.hpp`・`.cpp`: 方策の損失、value の損失、approx KL、advantage の標準化、explained variance の純粋な計算。
- `src/ppo_actor.hpp`・`.cpp`: `Actor`。行動選択（抽選・Greedy）と、行動の情報への値の載せ方。
- `src/ppo_learner.hpp`・`.cpp`: `Learner` と `UpdateResult`。rollout の蓄積、GAE、複数 epoch の更新、統計。
- `src/ppo_agent.hpp`・`.cpp`: `PPOAgent` と `PPOAgentFactory`。
- テスト: `src/ppo_fixture_test.hpp`・`.cpp`（設定と小さな課題の補助、何も書かない MetricsLogger）と、機能グループごとの `*_test.cpp`（`ppo_math_test`・`ppo_network_model_test`・`ppo_actor_test`・`ppo_learner_test`・`ppo_config_test`・`ppo_agent_test`）、出荷の設定を読む `ppo_config_baseline_test.cpp`（DQN 系・MuZero 系の `*_config_baseline_test.cpp` と同じ置き方）。

### 配線・設定・文書

- ルートの `CMakeLists.txt`、`src/apps/runner/CMakeLists.txt`、`src/apps/runner/src/RunnerApp.cpp`: ライブラリの追加、リンク、`InitPpo()` の呼び出し。
- `src/core/base/testdata/rfd0182/check_stage3_dependencies.py`: 系統の一覧に `anet-algo-ppo` を足す。
- `src/core/base/testdata/rfd0072/migrate.py`: env の設定ファイルの一覧に `CartPole_ppo.txt` を足す（設定の既定の検査が、新しい env の設定の行を分類できるようにする。実装で分かった移行対象で、RFD §8 に足した）。
- `src/core/base/testdata/rfd0194/expected.jsonl`: `nn_head.cpp` の 2 行を足す（根拠は「経過」）。
- `apps/runner/config/agent.txt`（`PPOAgent.@baseline` と Actor カタログ）、`nn.txt`（`Tanh` と PPO の MLP の block）、`metrics_scalar.txt`（`metrics.scalar.@ppo`）、`CartPole_ppo.txt`（新設）。
- `docs/design/115_on_policy_learning.jp.md`・`230_ppo_agents.jp.md`（新設）、`README.jp.md`・`010`・`011`・`100`・`110`・`130`・`150`（追随）。英語版は作らず、直さない。
- `CONTEXT.md`（「rollout」を広げ、「PPO系エージェント」を足す）、`reports/tools/categories.json`（326 の行、321・318 の `match`）。
- `docs/wiki/`: `categories/326_ppo.md`（新設）、`314_tensor`・`315_nn`・`318_rl_core`・`321_agent_core` の「成立の経緯」と「概要・責務」、目録、`log.md`。機械的な節は `reports/tools/wiki_support.py maps` で作り直した（`apps/110_runner_app.md` のファイルの一覧、`docs/management/issues/INDEX.md` の「課題を持たない RFD」に RFD0202 の行。`311_dev_tools.md` の課題の表の 1 行は、今回と関係のない、基準コミットの時点の遅れを道具が直したものである）。

## テスト

- Public interface / surface: core は `anet/distribution.hpp`・`anet/advantage.hpp`・`anet/nn_head.hpp`・`anet/tensor_store.hpp` の公開の口と、設定の `net.block.[名前].type = Tanh`。PPO 系は、`PPOAgentFactory::CreateAgent`（設定から）、`Agent::CreateActor`・`CreateLearner`、`Actor::MakeAction`・`Sync`、`Learner::UpdateFromBatch`、更新結果の `GetScalar`、`Agent::Save` と `auto_load_file`。
- 優先 behavior と TDD の順序: 次の表の行ごとに、先にテストを書いて失敗を確かめ、その挙動を成立させてから次の行へ進む。行 1 は、設定から Agent を作り、行動選択から rollout の更新までを 1 つのテストで通す。

| 順 | 入口 | 確かめる挙動 | 受入 |
|---|---|---|---|
| 1 | `[ppo]`。設定 → `PPOAgentFactory` → `MakeAction` → `UpdateFromBatch` | lane 2、rollout 2、1 epoch、1 ミニバッチで、1 回目の intake は空、2 回目の intake に更新結果が 1 件返り、`loss_total` が有限である | A6・A8 |
| 2 | `[distribution]` | log 確率と entropy が手計算と合う。`GetMode` が確率が最大の行動を返す。抽選の頻度が確率と合う（行動 3 つ、10 万回、差 0.01 以内）。logits の rank、行動数、行動の型と行数の不正で止まる。CPU と CUDA | A2 |
| 3 | `[advantage]` | 手計算の 3 step の例、終端、episode の終わり（次の観測の value を使う）、λ = 0、λ = 1 で値が合う。形と γ・λ の範囲の不正で止まる。手計算の 3 step の例は CPU と CUDA | A3 |
| 4 | `[composite_head]` | 2 つの Head の出力が 1 つにまとまり、受け取るキーごとに別の branch の値が届く。parameter の名前が `head.<名前>.` で始まる。組が 0 個、名前の重複、キーの不足、出力のキーの重複で止まる | A4 |
| 5 | `[nn]` | `Tanh` の module が、設定から組んだ Network で入力に tanh を当てる | A19 |
| 6 | `[tensor_store]` | `GetAll` が、ためた時刻の数 × lane 数の形でためた値を返し、空のときは止まる | A5 |
| 7 | `[ppo]` Network | 共有の設定では、方策と価値の両方の Head へ届く parameter の組が空でなく、別の設定では空である。別の設定で、`value_feature` に向けた branch だけが読む観測キーを変えても logits が変わらない。Head の出力の形が違う Network の設定は構築で止まる | A7 |
| 8 | `[ppo]` Actor | `sample` は抽選、`greedy` は確率が最大の行動を返す。行動の情報に、選んだ行動の log 確率、value、Network の入力、方策の世代が載る。強制行動を反映し、その行動の log 確率を載せる。写しを持つ Actor は `Sync()` で重みを写す | A6 |
| 9 | `[ppo]` rollout | rollout の長さに届くまで空で、`kl_stop.mode = none` では届いた intake で `epochs × minibatch_count` 件が返る。次の rollout も同じに進む。PPO 系の Actor が載せる値の無い経験は止まる | A8 |
| 10 | `[ppo]` 比 | 共有の Network で集めた rollout で、`first_ratio_deviation` が 1e-5 以下、`policy_lag` が 0 | A9 |
| 11 | `[ppo]` 打ち切り | `bootstrap` では打ち切りの step の value の目標に次の観測の value が入り、`terminal` では入らない。`done` と `truncated` が両方真なら終端として扱う | A10 |
| 12 | `[ppo]` 方策の古さ | 方策の古さの最大が 2 以上の rollout で止まる。1 では止まらず、`policy_lag` に出る | A11 |
| 13 | `[ppo]` 更新 | 学習率が schedule どおりに変わる。`kl_stop.mode = epoch` で、最初の epoch の終わりに閾値を超える設定では、`minibatch_count` 件が返り、`epochs_run` が 1 になる。勾配の clip の有無、value の clip の有無が統計に出る | A8 |
| 14 | `[ppo]` メトリクス | RFD §7.5 の表の鍵のすべてで `GetScalar` が値か NaN を返す。行動ごとの鍵は、行動数 2 で、n = 0・1 が値、n = 2 が NaN。どちらにも当たらない鍵（n が数字でないものを含む）は値なし。購読が無いと統計を計算せず、既知の鍵は NaN を返す | A16 |
| 15 | `[ppo]` 設定 | RFD §7.4 の値域の外の値、未知の列挙値、未定義の Actor キー、離散でない行動、割り切れない組み合わせ、1 件のミニバッチと標準化の組み合わせで、完全キーと指定値を含むメッセージで止まる。使わない側の設定キーも値域を検証する | A17 |
| 16 | `[ppo]` device | 行 1・8・9・10 の経路が CUDA でも通る（CUDA が使えない環境では飛ばす） | A6 |
| 17 | `[ppo]` 保存 | `Save` で書いたファイルを `auto_load_file` で読むと、同じ観測に、同じ方策の log 確率（行動を強制して読む）と value を返す。同じ経験でもう 1 回更新すると同じ重みになる（optimizer の状態も読めている）。形式の名前が違うファイルと、無いファイルは止まる | A6 |
| 18 | `[ppo]`（可視）・`[.][ppo_learn]`（hidden） | テストの中の小さな課題で、学習の終わりの episode return の平均が、学習の初めより高い（可視。rollout 40 個）。hidden は、rollout 200 個の学習のあと、seed 3 つのすべてで、最後の 20 個の rollout の中で終わった episode の return の平均が、最適な方策の return の 9 割以上 | A12 |
| 19 | `[shipped_config]` | 出荷の `agent.txt` の `PPOAgent.@baseline` が、RFD §7.4 の表の出荷の既定を持つ。`metrics.scalar.@ppo` の、更新結果を読む行の鍵が、どれも更新結果の既知の鍵である | A16・A18 |

- 計画を保存したあとに足したテスト:
  - `[.][ppo_shipped_cartpole]`（hidden。CPU と CUDA）: 出荷の `CartPole_ppo.txt` を `_main.txt` に重ねて読み、`PPOAgent` と 4 つの Actor を組み、出荷の metrics の行の購読を渡して rollout を 1 つ更新し、更新結果を読む行のすべてが値を返すことを確かめる。Run を起動できない（A13）ので、出荷の設定で Agent が組めることを Runner なしで確かめるために足した。env の設定の値は等値比較しない（`impl.md` の項番 73）。
  - `[ppo_learner]`: lane 数が Agent を作ったときと違う経験が、形を示して止まること。Agent を CPU に置き、写しを持つ Actor を CUDA に置くと、その Actor の値を受けた intake で止まること（どちらも RFD §7.4 に足した検査）。
  - `[ppo_runner]`: 製品の `SerialTrainRunner` と `PipelineTrainRunner` に、テストの補助の `CorridorBatchEnv`（一本道の課題を BatchEnv の口で出す）を渡して rollout を 4 つ回す。rollout ごとに LearnEvent が 1 回出て、learn step が返した件数だけ進み、方策の古さが Serial で 0、Pipeline で 2 つ目からの rollout で 1 step ぶん（1 / rollout の長さ）になることを確かめる。RFD の U5（Pipeline でも動く）を、Runner を通して確かめるために足した。
  - `[composite_head]`: Body の出力が CUDA にあるときの組み立て（下の「経過」の不具合の再発を防ぐ）。
- テストの中の小さな課題（行 18）は、テストの補助（`ppo_fixture_test`）に置く。観測は位置の one-hot（長さ 8）、行動は左・右、step ごとの報酬は −0.01、右端に着くと報酬 1 で終端、20 step で打ち切り、lane は 8 である。最適な方策の return は 0.94（右へ 7 回）。Env のライブラリには依存しない。学習の設定は、rollout 32、4 epoch、ミニバッチ 4、学習率 3e-3 の定数である。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 5'
out\test\Debug\anet-core-base-test.exe "[distribution],[advantage],[composite_head],[tensor_store]"
out\test\Debug\anet-algo-ppo-test.exe "[ppo]"
out\test\Debug\anet-algo-ppo-test.exe "[ppo_learn]"
out\test\Debug\anet-core-base-test.exe
out\test\Debug\anet-algo-dqn-test.exe
out\test\Debug\anet-algo-muzero-test.exe
out\test\Debug\anet-algo-imagecls-test.exe
out\test\Debug\anet-algo-ppo-test.exe
out\test\Debug\anet-core-base-test.exe "[shipped_config]"
out\test\Debug\anet-algo-ppo-test.exe "[shipped_config]"
out\test\Debug\anet-algo-ppo-test.exe "[ppo_shipped_cartpole]"
git diff --stat 9310b087 -- src/algo/dqn
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner --parallel 5'
```

- 最初の全ビルドは、セッションから切り離して起動し、出力と終了値をファイル（`.scratch/rfd0202/build/`）に書かせる。メイン機で開発者の Run が走っているので、優先度を下げ、並列を 5 に絞る。
- テストの出力は `.scratch/rfd0202/test/` に保存し、応答には要約と終了値だけを返す。
- 基準コミット（`9310b087`）の既定スイートを、変更の前のビルドで 1 回走らせ、既に落ちるテストの名前を「経過」に書く（A15 の除く対象）。
- 設定ファイルの検査: `check_default_leaves.py` と `[shipped_config]`。
- ライブラリの依存: `src/core/base/testdata/rfd0182/check_stage3_dependencies.py`（README の手順）。
- 設定キーを示す診断: `src/core/base/testdata/rfd0194/audit_config_messages.py --expected …`。PPO 系の診断で期待一覧に差が出たら、RFD0194 §8.7 の規則で期待一覧を直し、根拠を「経過」に書く。
- A13（CartPole の Run）は、開発者の了承が出るまで起動しない。出なければ未実施として、段取りを「結果」に書く。

## 前提

- RFD は暫定の起票で、実装の入口でレビューを受けた（判定と往復は「経過」）。開発者の判断が要るブロッカーは 0 件である。指示（質問せず、既定を置いて進める）により、未決表の暫定の既定を実装の前提に固定する。
- 了承の無い 3 つ（Runner で Run を起動する、DQN 系を変える、git の操作）は行わない。RFD の課題（`rfd` でこの RFD を指す課題）は起票していないので、課題の状態と対応内容の更新は行わない。
- 既定値: rollout 128、4 epoch、ミニバッチ 4、γ 0.99、λ 0.95、学習率 2.5e-4（`@baseline` は定数。CartPole の設定は exp step 500,000 で 0 へ線形）、比の clip 0.2、value の clip あり（0.2）、entropy 0.01、value 0.5、勾配 clip 0.5、Adam の eps 1e-5。CartPole の出荷の設定は lane 8。
- 行動の情報の鍵は `log_prob`・`value`・`policy_version` と、Network の入力のキーに接頭辞 `input.` を付けたものにする。PPO 系の中だけの取り決めで、名前つきの定数にする。
- 分布の抽選は `torch::multinomial` に device ごとの生成器（`RandomGenerator::GetTorchGenerator`）を渡す。並べ替えは `torch::randperm` を device の上で作る。
- 排他は、RFD §7.10 のとおりにする。Learner は、ミニバッチ 1 つの forward から optimizer の step までの区間ごとに排他のロックを取り、`TrainingModeGuard` で train にする。ロックを rollout の更新の全体では持たない（評価の Actor の `Sync()` を長く待たせないため）。
- device 間の同期の転送は、RFD §7.6 の表のとおりである。足す同期の転送の、頻度 × 1 回あたりの時間の見積もり（agent が CUDA のとき。時間はどれも見込みで、測っていない。出荷の CartPole の設定＝lane 8・rollout 128・4 epoch・ミニバッチ 4 で数える）:
  - 行動選択の観測（CPU → device）と行動（device → CPU）: 毎 step に 1 回ずつ × 数十マイクロ秒（lane 8・観測 4 要素）。1 step あたり 0.1 ミリ秒未満で、exp step 500,000（62,500 step）で合計数秒。Env が CPU にある構成では、DQN 系・ImageCls と同じく避けられない。
  - rollout の終わりの報酬と印（CPU → device。1,024 × 3 個の float を 1 本）と、最後の次の観測（8 × 4 個）: rollout に 1 回ずつ × 数十マイクロ秒。rollout あたり 0.1 ミリ秒未満（同じ rollout の更新は optimizer の step が 16 回で、数十ミリ秒の見込み）。
  - 更新結果の統計（device → CPU。16 行 × 22 個と rollout の値 15 個を 1 本）: rollout に 1 回 × 数十マイクロ秒と、device に残っている計算の完了待ち。購読が無ければ無い。読むのは更新の直後の同じスレッドで、あいだに進める処理が無いので、`HostReadback` に載せても同期の転送と変わらない（RFD の U22）。
  - 打ち切りの value（CPU → device。Reset の前の観測と印を送り、forward を 1 回）: 打ち切りの lane がある step に 1 回 × 数百マイクロ秒。CartPole（上限 500 step、lane 8）で、多くても 500 step に 8 回なので、62,500 step で多くて 1,000 回、合計 1 秒未満。
  - KL で止める判定（device → CPU。値 1 個）: `kl_stop.mode = epoch` のときだけ、epoch に 1 回（rollout に多くて 4 回）× 数十マイクロ秒と、完了待ち。既定の `none` では無い。
  - lane や行を回すループの中には置かない。
- 別のスレッドのロック待ちの、頻度 × 1 回あたりの時間の見積もり（時間は見込みで、測っていない）:
  - 学習の Actor の共有ロック: 毎 step に 1 回取る。待ちは 0 である。Serial の Runner は行動選択と学習を交互に進め、Pipeline の Runner は、前の学習の完了を待ったあと、次の学習を投入する前に行動を選ぶ（`PipelineTrainRunner::DoStep`）ので、Learner の排他のロックと重ならない。
  - Learner の排他のロック: ミニバッチごとに 1 回（rollout に 16 回）取る。待つ相手は、評価の Actor の `Sync()`（共有のロックを持って重みを写す。小さな MLP で数十マイクロ秒）だけである。評価の発火は rollout の更新 1 回につき多くても 1 回なので、rollout あたり多くて数十マイクロ秒。
  - 評価の Actor の `Sync()` の側の待ち: 発火ごとに、多くてミニバッチ 1 つの更新の時間（CartPole の MLP で 1〜数ミリ秒）。
- `dynamic_cast` を使わない。Learner は、行動の情報の `info` の鍵で、PPO 系の Actor の値を受け取る。
- 動詞の使い方は既存に合わせる: `Compute…`（`ComputePlasticityMetrics`）、`Make…`（`MakePassThroughForwarder`）、`Get…`、`Create…`（`CreateHead`・`CreateActor`）。クラスの接尾辞は `Factory`（`LinearHeadFactory`）、`Config`、`Result`（`BatchUpdateResult`）、`Model`（`dqn::NetworkModel`）、`Distribution`。
- 英語版の設計文書、正式な Run、実験記録、ADR は範囲外（RFD の NG6、U26、§8）。

## 経過

- 2026-11-25: RFD0202 を暫定で起票した。実装の入口で RFD のレビューを受けた。1 往復目は差し戻し（8 件。評価の間隔と ADR 0028、行動ごとの鍵の受入、KL でやめた rollout の件数、AMP の差し込み位置、勾配の clip の上限の名前、「方策の世代」の言い換え、学習率の既定と時間の値を読む step と KL の閾値の根拠、材料の使い先）。どれも開発者の判断は要らず、直して出し直した。あわせて、活性化の module `Tanh`（core に無いことをコードで確かめた）と、移行対象の漏れ（依存の検査の系統の一覧、設定キーの診断の期待一覧、110 のコードマップ）を RFD に足した。2 往復目で OK。原文は `.scratch/rfd0202/review-20261125-1506-rfd.md`・`review-20261125-1514-rfd.md`。
- 2026-11-25: 2 往復目の再確認ポイントのうち、文言の明確化だけを RFD に反映した（§6.6 の等間隔の条件、冒頭の改訂行、U20 の言い回し、§7.1 の float32 の正規化）。RFD の実装のあとのレビューで、あわせて確かめる。
- 2026-11-25: 未決事項の監査。開発者の判断が要るブロッカーは 0 件。
- 2026-11-25: 基準コミット（`9310b087`）の Debug の全ビルドと既定スイートを、変更の前に実行した。10 本のテスト実行体のうち、`anet-core-base-test` の 1 件（`Logger prefixes an info message exactly once`。`log_test.cpp:163`）だけが落ち、1 件が飛ばされた（`init_test.cpp:222`）。この 1 件は、A15 の「基準コミットで既に落ちるテスト」として除く。結果は `.scratch/rfd0202/test/baseline/summary.txt`。
- 2026-11-25: TDD の進め方。ビルドが重い（PPO 系のテスト実行体の作り直しに数分、core を変えると全体の再リンク）ので、行ごとに RED → GREEN を 1 回ずつは回さず、次のまとまりで進めた。
  - 行 1（経路の全体）: テストと配線だけを置いてビルドし、RED を確かめた（`ppo.cpp(4): fatal error C1083: Cannot open include file: 'ppo_agent.hpp'`。`.scratch/rfd0202/build/red1.log`）。本体を置いて GREEN にした。
  - 行 2〜6（core）: テストと実装を同じビルドに入れた。RED は確かめていない。
  - 行 7〜17: テストを先に全部書き、1 回のビルドで走らせた。35 件のうち 11 件が落ちた（`.scratch/rfd0202/test/green3/ppo.log`）。10 件は下の CUDA の不具合で、実装の欠けを示す RED になった。1 件はテストの側の誤り（連続行動の仕様に次元を書いていなかった）だった。直して全部通した。
  - 行 18・19: 実装のあとにテストを足した。RED は確かめていない。
- 2026-11-25: テストで見つけて直した不具合。
  - `CompositeHeadFactory::CreateHead` が、出力のキーの重なりを確かめる仮の forward を、CUDA の仮の特徴量と、CPU にある作ったばかりの中の Head で行って止まった。CPU のテストだけでは通る。中の Head を、仮の forward の前に特徴量の device へ移すようにし、`[composite_head]` のテストを CUDA でも走らせるようにした。
  - テストの設定で、時間の値の定数を `…learning_rate.value` に書いていた。定数は設定キーそのもの（`…learning_rate`）に書く。テストを直した。
  - テストの補助に、何も書かない MetricsLogger が無く、Agent の構築で落ちた（DQN 系・ImageCls のテストと同じ補助を足した）。
- 2026-11-25: 設定キーを示す診断の監査（`audit_config_messages.py --expected …`）。基準コミットの時点で、期待一覧と 19 件の差がある（終了値 1。`forwarder.cpp`・`tensor_store.cpp`・`default_dqn_agent.cpp` ほか。今回の変更とは関係が無い）。今回の変更で増えたのは `src/core/base/src/nn_head.cpp` の 2 件（名前 `index` と `key`）である。どちらも設定キーではない（`index` は factory へ渡した組の番号、`key` は Head の出力の TensorDict のキー）ので、RFD0194 §8.7 の「直したあとに拾われてよいもの」に当たる名前として、期待一覧の末尾に 2 行を足した。足したあとの差は、基準コミットと同じ 19 件である（`.scratch/rfd0202/test/audit_base.txt` と `audit_now.txt` が一致）。PPO 系（`src/algo/ppo`）・`distribution.cpp`・`advantage.cpp` の診断は、1 件も拾われていない。
- 2026-11-25: 設定の既定の検査（`check_default_leaves.py`）。はじめは `CartPole_ppo.txt` の Network の配線 6 行が「分類できない個別指定」として出た。検査が持つ env の設定ファイルの一覧（`rfd0072/migrate.py` の `ENV_FILES`）に `CartPole_ppo.txt` が無いためで、一覧に足して解いた（RFD §8 に足した）。残る 4 件（`apps/tools/metrics_viewer.py` の `app.title` ほか）は、今回変えていないファイルの指摘で、基準コミットでも出る。検査器の回帰テスト（`check_default_leaves_test.py`）は 16 件が通った。
- 2026-11-25: ライブラリの依存の検査。Debug の構成に `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` と `--graphviz` を付けて出し直し、`check_stage3_dependencies.py` を実行して、違反 0 件（系統の一覧に `anet-algo-ppo` を足した版）。
- 2026-11-25: 作業の誤りと戻し。依存の検査の構成を出すスクリプトを `printf` で書いたら、パスの `\` が崩れ、Visual Studio の環境が無いまま `cmake --preset x64-Debug` が走って途中で失敗した。ビルドの定義（`build.ninja`）は変わっていなかった。スクリプトを書き直して構成をやり直し、そのあとの Debug の全ビルドが通ることを確かめた。
- 2026-11-25: 行末。変更した既存ファイルのうち 9 つ（`CMakeLists.txt`・`src/apps/runner/CMakeLists.txt`・`apps/runner/config/` の 3 つ・`reports/tools/categories.json`・`testdata` の 3 つ）は、ワークツリーの checkout の時点で作業ツリーの側が CRLF だった（`.gitattributes` の指定が無く、`core.autocrlf=true`）。index は LF である。`impl.md` の項番 7 に合わせ、作業ツリーの側を LF にそろえた（BOM は保った）。
- 2026-11-25: クラス一覧と機能グループは RFD §7.7・§7.8 のとおりで、変えていない。RFD に無いファイルは、テスト（`ppo_fixture_test`・`ppo_config_baseline_test`）だけである。
- 2026-11-25: 自己点検（`impl.md` の全項目）で直したこと。クラスの宣言の並び（`LearnerConfig`・`PPOAgentConfig`・`UpdateResult::RolloutStatistics` を、型 → 定数 → 関数 → データメンバの順にした）、宣言が 3 個以下のところの空行（`Actor`・テストの `CorridorEnv`）、2 か所で使う値の定数化（value の損失の係数、印の閾値、テストの鍵と名前）、テストの集成体の指示付き初期化、経験の形の検査（lane 数が違う経験を、器に入れる前に止める）。機械検査（項番 1・2・3・62・71・76）は 0 件・OK。項番 72 の候補のうち、core の `nn_test.cpp` に足したテストの TensorDict キー（`"logits"`・`"value"` ほか）は、この時点では、同じファイルの既存のテストの書き方に合わせて直書きのままにした（実装のレビューの指摘を受けて、あとで定数にした）。
- 2026-11-25: 足した INFO と、Fail-Fast の表の対応。`ppo_agent.cpp` の INFO 3 つ（構築時の設定の記録、`auto_load_file` の読込の通知、保存の完了の通知）は、表の「上記以外（知らせる必要があれば INFO）」の行に当たる。DQN 系・ImageCls の Agent が、同じ時機に同じ形の INFO を出している。構築と読込は Run ごとに 1 回、保存は操作ごとである。WARN は足していない。
- 2026-11-25: 実装のレビュー（1 往復目）。判定は差し戻し（実装計画 1 件、実装 15 件）。原文は `.scratch/rfd0202/review-20261125-1747-impl.md`。どれも開発者の判断は要らず、次のとおり直した。
  - 計画 1・実装 14: 検証の記録の時点。直しを全部入れたあとで、監査・設定の既定の検査・固定 17 入力の比較（`rfd0072/prepare.py compare`）を実行し直し、「検証の結果」を書き直した。
  - 実装 1（定数化）: `nn_test.cpp` に足したテストの TensorDict キー・入力名・出力のキーと、`ppo_agent_test.cpp` の無いファイルの名前を、名前つきの定数にした。
  - 実装 2（指示付き初期化）: `MakeActionOptions`（テストの補助 `MakeForcedActionOptions` にまとめた）、`MetricsLoggerConfig`、`WeightInitConfig`、`Learner::Minibatch` を、指示付き初期化にした。
  - 実装 3・4: `LearnerConfig::KlStop` の空行を外した。テストの `MeanReturn` を `ComputeMeanReturn` に改めた。
  - 実装 5（メッセージ）: `advantage.cpp`・`distribution.cpp`・`ppo_math.cpp` のエラーに、渡された tensor の device・型・形を入れた。
  - 実装 6（設定キー）: `nn_head.cpp` と `ppo_learner.cpp` の文面から、相対キーの名指し（`net.body.output.[…]`、`clone_model=false`）を外した。
  - 実装 7: 上の行（INFO と表の対応）を書いた。
  - 実装 8: 統計の読み戻しに計測範囲（`statistics_readback`）を置いた。
  - 実装 9: 「前提」の見積もりを、頻度 × 1 回あたりの時間の形にし、ロック待ちの見積もりを足した。
  - 実装 10: Learner の device の検査を、`AgentBase::ValidateActorDevice` と同じ比べ方（番号まで比べる）の写しから、device の種類（CPU と CUDA）だけを比べる検査に変えた。Learner が受け取るのは tensor で、Agent の基底の検査（Actor の生成時）とは時機も対象も違う。写しを持つ Actor を別の種類の device に置いて学習に使うと、あとの計算で device の違う tensor が混ざるので、ためる前に止める。契約（RFD §7.4、設計文書 230 §6.1）とテスト（`[ppo_learner]`）に載せた。
  - 実装 11: 時間の値の 3 キーのテストで、メッセージに指定値が入ることも確かめるようにした。
  - 実装 12: 保存のテストで、方策の log 確率（行動を強制して読む）も比べるようにし、計画の行 17 の文をそれに合わせた。
  - 実装 13: 設計文書 150 §2.5 の表と、100 の Agent Factory の列挙に `PPOAgent` を足した。
  - 実装 15: 測っていない速さの比較（設計文書 230 §11、`CartPole_ppo.txt` のコメント）を、見込みと分かる文にした。
  - 範囲外の検出 2 件（`nn_head.cpp` の `LinearHead` の文面、`agent.cpp` の `clone_model` の文面が、設定を相対の名前で名指す）は、今回の変更の前からある文面で、直していない。起票の候補に足した。
- 2026-11-25: 実装のあとの RFD のレビュー（3 往復目）。判定は差し戻し（1 件。用語集の「rollout」の定義が 3 文になった）。定義を 2 文にした。あわせて、レビューが「判定に迷った点」に挙げた、RFD の列挙に無い検査 3 つ（浮動小数でない logits、受け取るキーが空・factory が無い組、別の種類の device にある Actor の値）を RFD に足し、§8 の用語集の項目の表記を題に合わせた。原文は `.scratch/rfd0202/review-20261125-1728-rfd.md`。
- 2026-11-25: レビューを待つ間に、Runner を通すテスト（`[ppo_runner]`）を足した。Pipeline の Runner の 1 step の流れ（`PipelineTrainRunner::DoStep`）を読み直すと、行動選択は、前の学習の完了を待ったあと・次の学習の投入の前に走る。rollout を埋めた経験の更新は、次の step の Env の step と並んで走るので、その step の行動だけが更新の前の方策で選ばれ、次の rollout の最初の経験になる。方策の古さは、2 つ目からの rollout でちょうど 1 step ぶんになる。
- 2026-11-25: 実装のあとの RFD のレビュー（4 往復目）。判定は OK。原文は `.scratch/rfd0202/review-20261125-1825-rfd.md`。
- 2026-11-25: 実装のレビュー（2 往復目）。実装計画は OK、実装は差し戻し（1 件）。原文は `.scratch/rfd0202/review-20261125-1831-impl.md`。
  - 実装 1（項番 41）: 統計の読み戻しの計測範囲を `std::call_once` に渡すラムダの中に置いていて、計測名が `クラス名::関数名.段` の形にならなかった。読み戻しを、名前を持つ関数（`UpdateResult::RolloutStatistics::ReadBackDeviceValues`）に分け、`ANET_PROFILE_FUNC` で計測する形にした。
- 2026-11-25: 実装のレビュー（3 往復目）。実装計画・実装とも OK。原文は `.scratch/rfd0202/review-20261125-1850-impl.md`。
- 2026-11-25: 完了報告のあと、開発者から「試運転起動」の指示を受けた。CartPole の試運転を、段取りの表のとおりに流した（19:22〜19:24。seed 3 つ）。その前に、設定の一覧を確かめる短い試しを 1 本、あとに、EvalPanel を自動で動かす短い試しを 2 本流した。結果は「CartPole の試運転の結果」に書いた。この追記は、レビューを受けていない。

## 検証の結果

2026-11-25 に、実装のレビューの直しを入れた状態で実行した。製品のコードの最後の変更（実装のレビューの 2 往復目の直し。PPO 系の `ppo_learner.hpp`・`.cpp` だけ）は 18:31 である。そのあとに、Debug の全ビルド（18:32〜18:35）、PPO 系のテスト（hidden の 2 つを含む。18:36〜18:40）、Release の Runner（18:40〜18:43）、静的な検査の 4 つ（18:43〜18:46）を実行した。既定スイートのうち PPO 系のほかの 10 本は、その前（1 往復目の直しのあと。17:59〜18:10）の実行である。この 10 本の実行体は、18:31 の変更（PPO 系のライブラリの中だけ）を含まず、作り直されていない。ログは `.scratch/rfd0202/build/` と、`.scratch/rfd0202/test/final4/`（18:31 のあとの実行）・`test/final3/`（その前の実行）にある。

| 確かめたこと | 結果 | 記録 |
|---|---|---|
| Debug の全ビルド（`cmake --build --preset x64-Debug --parallel 5`） | 終了値 0 | `build/build_debug.log` |
| Release の Runner（`--preset x64-Release --target AnetRLRunner`） | 終了値 0。`apps/runner/bin/Release/AnetRLRunner.exe` ができた | `build/release_runner.log` |
| `anet-algo-ppo-test`（引数なし） | 39 件・1,185 アサーションが通過。CUDA の分も走った（飛ばしは 0 件） | `test/final4/anet-algo-ppo-test.log` |
| `anet-algo-ppo-test "[ppo_learn]"`（hidden） | seed 1・2・3 のすべてで、最後の 20 個の rollout の return の平均が 0.94（最適な方策の return と同じ）。1 seed あたり約 57 秒 | `test/final4/ppo_learn.log` |
| `anet-algo-ppo-test "[ppo_shipped_cartpole]"`（hidden） | CPU・CUDA とも通過（176 アサーション） | `test/final4/ppo_shipped_cartpole.log` |
| `anet-algo-ppo-test "[shipped_config]"` | 2 件・302 アサーションが通過 | `test/final4/ppo_shipped_config.log` |
| `anet-core-base-test "[distribution],[advantage],[composite_head],[tensor_store],[tanh],[shipped_config]"` | 23 件・1,094 アサーションが通過 | `test/final3/core_new_and_shipped.log` |
| 既定スイート（`out/test/Debug/*-test.exe` の 11 本を、引数なしで。17:59 に開始） | 11 本とも終了値 0。`anet-algo-dqn-test` 384 件、`anet-algo-imagecls-test` 29 件、`anet-algo-muzero-test` 7 件、`anet-algo-ppo-test` 39 件、`anet-core-base-test` 708 件（707 件通過・1 件飛ばし）、env の 6 本（atari 56 件・cartpole 1 件・dropmerge 33 件・gridmaze 12 件・imagecls 20 件・lunarlander 8 件） | `test/final3/summary.txt` |
| `git diff --stat 9310b087 -- src/algo/dqn` | 出力なし（`src/algo/imagecls`・`src/algo/muzero`・`src/env` も同じ） | — |
| 静的な検査: ライブラリの依存（`rfd0182/check_stage3_dependencies.py`） | 違反 0 件 | `test/final4/deps_check.txt` |
| 静的な検査: 設定の既定（`rfd0072/check_default_leaves.py`） | 指摘 4 件（`apps/tools/metrics_viewer.py` の 4 行。基準コミットと同じ。今回足した設定の指摘は 0 件）。検査器の回帰テストは 16 件が通過 | `test/final4/default_leaves.txt` |
| 静的な検査: 固定 17 入力の全キー・文字列値の比較（`rfd0072/prepare.py prepare`・`compare`） | 終了値 0（`[rfd0072-baseline]` の 157 アサーションが通過） | `test/final4/rfd0072_compare.txt` |
| 静的な検査: 設定キーを示す診断の監査（`rfd0194/audit_config_messages.py --expected …`） | 差 19 件（基準コミットと同じ 19 行。終了値は基準コミットでも 1） | `test/final4/audit_now.txt`・`test/audit_base.txt` |

注（どちらも、今回の変更の前からある不安定さである。直しのあとの最後の実行では、どちらも出ていない）:

- `anet-core-base-test` は、テストの実行順を乱数で決める。別のテストの背景スレッドが出すログ（`EpisodeEvalObserver[eval1] RunEvaluationSession failed: forced shutdown failure` など）が、ログの件数や末尾を確かめるテストに紛れ込んで、実行順しだいで 1 件落ちる。落ちたのは、基準コミットの実行（seed 4233399730）で `Logger prefixes an info message exactly once`、実装のあとの実行（seed 3247615252）で `Logger level methods preserve wx levels and prefix`、同じく（seed 1597833589）で `Config reads from a catalog preserve original unread keys` である。実装のあとの実行のうち 2 回（seed 223745117、seed 265906785）は 0 件だった。Logger の 3 件は、単独で走らせると通る。今回足したテストは、背景スレッドもログの取り込みも使わない。A15 の「基準コミットで既に落ちるテスト」は、この一群として除く。
- `anet-env-imagecls-test` は、実装のあとの実行の 1 回（seed 3387089429）で、最初のテストが wxWidgets のメッセージボックス（`No image handler for type 15 defined.`）を出して、入力待ちで止まった。スタックで確かめ、プロセスを終了した。ほかの実行（seed 1973187538 など）は 20 件が通過した。`src/env` は変えていない。

未実施の検証:

- Runner の画面での表示のうち、Eval View・Q-Values・Trace の各ペイン。試運転の Run では、既定の配置（Train View と Logs）だけが開いていて、ほかのペインは開いていない。`app.eval_panel.auto_start=true` を渡した短い試し 2 本は、警告もエラーもなく終わったが、EvalPanel が進んだ跡はログに残らないので、動いたことまでは確かめられていない。`PPOAgent` は、step ごとの診断の口と、Q 値の表示の値を持たない。共通の設定（`common.txt`）の Trace の列 `$agent_step q_sa`・`state_value`・`approx_priority` は、DQN 系の値である。Train Runner（Serial・Pipeline）を通す経路は、テスト `[ppo_runner]` で確かめた。

A13（CartPole の Run）は、完了報告のあとに開発者の指示を受けて流した。結果は「CartPole の試運転の結果」にある。


## 受入との対応

| # | 結果 | よりどころ |
|---|---|---|
| A1 | 満たす | Debug の全ビルドが終了値 0 |
| A2〜A5・A19 | 満たす | `[distribution]`・`[advantage]`・`[composite_head]`・`[tensor_store]`・`[tanh]` が通過（CPU と CUDA） |
| A6 | 満たす | `anet-algo-ppo-test "[ppo]"` が通過。device を受けるテストは CPU と CUDA の両方で走った。製品の Train Runner を通す `[ppo_runner]` も通過 |
| A7 | 満たす | `[ppo_network]` の 2 件 |
| A8 | 満たす | `[ppo_learner]` の「rollout が埋まるまで空」「KL で止める」、`[ppo_config]` の割り切れない組み合わせ |
| A9 | 満たす | `[ppo_learner]`（`first_ratio_deviation` が 1e-5 以下、`policy_lag` が 0） |
| A10 | 満たす | `[ppo_learner]` の打ち切りの 4 つの場合 |
| A11 | 満たす | `[ppo_learner]` の方策の古さ（1 では止まらず、2 で止まる） |
| A12 | 満たす | `[ppo_learn]`（rollout 200 個、seed 3 つ、最後の 20 個の rollout の平均が 0.94） |
| A13 | 満たす | CartPole の試運転（「CartPole の試運転の結果」）。3 seed のすべてで、Greedy の評価が 500 に届き、exp step 400,000 から 500,000 の評価の点が 20 点とも 500 |
| A14 | 満たす | `git diff --stat` が空、`anet-algo-dqn-test` が通過 |
| A15 | 満たす（上の注の一群を除く） | 既定スイート |
| A16 | 満たす | `[ppo_learner]` の鍵のテスト、`[shipped_config]` の `metrics.scalar.@ppo` のテスト |
| A17 | 満たす | `[ppo_config]`・`[ppo_actor]`（未定義の Actor キー） |
| A18 | 満たす | `[shipped_config]`（core と PPO 系）、設定の既定の検査の指摘が基準コミットから増えていない |

## CartPole の試運転の段取り（A13。起動は開発者の了承のあと）

| 項目 | 案 |
|---|---|
| env | CartPole（`apps/runner/config/CartPole_ppo.txt`）。`CartPoleEnv.limit_step = 500` |
| 群 | 1 群（出荷の設定のまま。方策と価値の Network は別、lane 8、rollout 128、4 epoch、ミニバッチ 4、学習率は 2.5e-4 から 0 へ線形） |
| seed | 3 つ（`run.seed=1`・`2`・`3`） |
| 予算 | 1 本あたり exp step 500,000（rollout 約 488 個、learn step 約 7,800） |
| 実行体 | このワークツリーの Release（`apps/runner/bin/Release/AnetRLRunner.exe`。コミットしていない変更から作ったもの） |
| device | `agent.device=cpu`・`run.eval_device=cpu`（小さな MLP と lane 8 では CPU のほうが速い見込み。メイン機の GPU は、走行中の Run が使っている） |
| 並列 | 3 本を順に 1 本ずつ |
| 置き場 | `app.runs_dir` をこのワークツリーの `.scratch/rfd0202/runs` にする（Git が追跡しない） |
| Run 名 | `run_{t}_tmp_ppo_cp_s1`・`_s2`・`_s3`（`tmp` を含み、40 文字以内） |
| 起動 | スクラッチに、`_main.txt` と `CartPole_ppo.txt` を `$include` する設定を書き、`--config` で渡す（workspace の履歴を触らない）。`app.$=app.batchrun` を CLI で渡して、予算で終わらせる |
| 見る指標 | `21_eval/21_ep_steps_greedy`（500 に届くか。exp step 400,000 から 500,000 の評価の点の 8 割以上が 500 か）、`21_eval/22_ep_steps_sample`、`78_agent_ppo/11_approx_kl`・`15_ratio_clip_fraction`・`31_explained_variance`、`79_agent_policy/02_entropy_ratio`、`78_agent_ppo/20_first_ratio_deviation`・`79_agent_policy/34_policy_lag`（どちらも 0 のはず）、`90_perf/12_exp_step_per_sec` |
| あわせて見る | 画面の表示（EvalPanel、QValuePanel、Evaluation Trace）が、`PPOAgent` で止まらずに動くか。読まれない設定キーの一覧と WARN が出ないか |

## CartPole の試運転の結果（A13。2026-11-25）

段取りの表のとおりに流した。実行体はこのワークツリーの Release（18:42 にできた `apps/runner/bin/Release/AnetRLRunner.exe`）、設定はスクラッチの `.scratch/rfd0202/trial/ppo_cp_trial.txt`（`_main.txt` と `CartPole_ppo.txt` を読み、`app.runs_dir` を `.scratch/rfd0202/runs` にする）を `--config` で渡した。CLI では `app.$=app.batchrun`・`agent.device=cpu`・`run.eval_device=cpu`・`run.seed`・`app.run_name` だけを渡した。3 本を順に 1 本ずつ流し、どれも終了値 0 で、1 本あたり 33〜35 秒だった。メイン機では、12 bat の Run（GPU を使う）が並んで走っていた。

設定の一覧（`config/config_data.txt`）と `json/` で確かめたこと: `agent.device = cpu`（`effective_device` も cpu）、`run.train.runner_type = serial`、`run.train.num_envs = 8`、`CartPoleEnv.limit_step = 500`、`app.exp_exit_step = 500,000`。学習率は、exp step 1,016 の最初の更新で 2.4949e-4（2.5e-4 × (1 − 1,016 / 500,000)）、最後の更新で 1.48e-7 で、線形に下がっている。

| seed | Run 名 | Greedy が初めて 500 に届いた exp step | exp step 400,000〜500,000 の Greedy（500 の点 / 点の数） | 同じ窓の sample | 判定 |
|---|---|---|---|---|---|
| 1 | `run_20261125-192220_tmp_ppo_cp_s1` | 26,616 | 20 / 20 | 20 / 20 | 満たす |
| 2 | `run_20261125-192254_tmp_ppo_cp_s2` | 185,336 | 20 / 20 | 20 / 20 | 満たす |
| 3 | `run_20261125-192327_tmp_ppo_cp_s3` | 21,496 | 20 / 20 | 19 / 20（436,216 で 123） | 満たす |

評価は exp step 5,120 ごとで、1 回の評価は 1 episode である。A13（3 seed のすべてで、Greedy の評価が 500 に届き、exp step 400,000 から 500,000 の評価の点の 8 割以上が 500）を満たす。

100,000 step ごとの Greedy の評価（500 の点 / 点の数）:

| seed | 0〜100k | 100k〜200k | 200k〜300k | 300k〜400k | 400k〜500k |
|---|---|---|---|---|---|
| 1 | 1 / 19 | 12 / 19 | 20 / 20 | 19 / 19 | 20 / 20 |
| 2 | 0 / 19 | 3 / 19 | 19 / 20 | 19 / 19 | 20 / 20 |
| 3 | 10 / 19 | 19 / 19 | 20 / 20 | 19 / 19 | 20 / 20 |

500 に定着するまでの長さは、seed で 100,000 step から 300,000 step まで開いた。

あわせて見た指標（seed 1・2・3 の順）:

| 指標 | 値 |
|---|---|
| `78_agent_ppo/11_approx_kl` の最大 | 0.00094・0.0017・0.0022。KL でやめた rollout は無い（`14_epochs_run` は常に 4） |
| `78_agent_ppo/15_ratio_clip_fraction` | 中央値は 0。最大は 0.0017・0.011・0.012 |
| `78_agent_ppo/31_explained_variance` | 中央値は 0.019・0.007・−0.0007。最大は 0.61・0.77・0.18。終わりは 0 の近く |
| `79_agent_policy/02_entropy_ratio` | 最小は 0.81・0.80・0.76。終わりは 0.86・0.86・0.86 |
| `78_agent_ppo/20_first_ratio_deviation` | 最大 4.8e-7（float32 の丸めの大きさ）。中央値 1.2e-7 |
| `79_agent_policy/34_policy_lag` | 常に 0 |
| `78_agent_ppo/55_grad_clipped` | 中央値 1（最小 0.69・0.81・0.69）。ほぼ毎回、勾配を clip している |
| `90_perf/12_exp_step_per_sec` の中央値 | 16,455・16,456・16,147（CPU。GPU では測っていない） |
| ログ | WARN 0 件、ERROR 0 件、読まれない設定キーの一覧なし、`stderr.log` は空、`agent_close.anet` を保存 |

seed 1 では、勾配のノルムが 8〜38（clip の上限は 0.5）で、その大半が価値の側（8〜38）、方策の側は 0.05〜0.25 だった。価値の損失は 8.6〜92、価値の教師の平均は終わりで 75 である。報酬の正規化は入れていない（RFD の NG5）。

短い試し（どれも終了値 0、WARN 0 件。`.scratch/rfd0202/runs` に残っている）:

| Run 名 | 中身 |
|---|---|
| `run_20261125-192120_tmp_smoke_ppo_cp` | 3 本の前に流した。exp step 12,000、6 秒。設定の一覧、評価 2 回、終了時の保存を確かめた |
| `run_20261125-192430_tmp_smoke_ppo_panel` | exp step 200,000、`app.eval_panel.auto_start=true`、14 秒 |
| `run_20261125-192529_tmp_smoke_ppo_shot` | exp step 400,000、`app.eval_panel.auto_start=true`。Runner のウィンドウを 2 回撮った（`.scratch/rfd0202/trial/runner_window_1.png`・`_2.png`）。Train View に CartPole の絵と状態の値、ツールバーに exp と train の step の数、Logs に進みの行が出ていた |

Run フォルダは 6 つで、どれも名前に `tmp` を含む（`.scratch/rfd0202/runs`、計 16 MB）。消すのは開発者である。起動の記録（`launch.log`）と集計のスクリプト（`summarize_trial.py`・`greedy_curve.py`・`read_run.py`）は `.scratch/rfd0202/trial/` にある。

## 未決と、起票の候補

- RFD の未決表（U1〜U30）は、暫定の既定のまま実装した。再グリルで見直す。
- RFD の課題（`rfd` でこの RFD を指す課題）は起票していない。課題の一覧（`docs/management/issues/INDEX.md`）の「どの課題の rfd からも指されない RFD」に、RFD0202 の行が出ている。
- 起票の候補（今回は起票していない）:
  - RFD0202 の仕事の課題（PPO 系と方策ベースの共通部品。再グリルが残る）。
  - `anet-core-base-test` の、背景スレッドのログが紛れ込んで実行順しだいで落ちるテスト（上の注）。
  - `anet-env-imagecls-test` の、実行順しだいで wxWidgets のメッセージボックスが出て止まる問題（上の注）。既にある課題 [issue-20261123-9a0e78](../management/issues/issue-20261123-9a0e78.md)（imagecls の Env のテストが画像の口を初期化せず、乱数の並びで先に回ると止まる）と同じ事象で、今回の seed（3387089429）とメッセージの文面を追記する候補である。
  - 設定キーを示す診断の監査の期待一覧が、基準コミットの時点で 19 件ずれている。
  - 設定の既定の検査が、`apps/tools/metrics_viewer.py` の 4 行を設定の行として拾う。
  - `PPOAgent` の Evaluation Trace の列（value、log 確率、行動の確率）と、画面の表示。
  - 設定を相対の名前で名指す、今回の変更の前からある文面 2 つ（`nn_head.cpp` の `LinearHead` の `'net.body.output.[features]'`、`agent.cpp` の `clone_model=false`・`clone_model=true`）。実装のレビューの「範囲外の検出」による。

## 完了報告（2026-11-25）

実施内容:

- RFD0202（暫定の起票）に従って、core の共通部品（行動の分布、GAE、Head の合成、活性化の `Tanh`、ためた経験の全体の読み出し）と、PPO 系（`src/algo/ppo`。離散行動の `PPOAgent`）を実装した。Runner への配線、出荷の設定（`PPOAgent.@baseline`、`metrics.scalar.@ppo`、`CartPole_ppo.txt`）、設計文書（115・230 の新設と 7 本の追随）、用語集、機能カテゴリ、wiki を同じ変更に入れた。
- 検証は「検証の結果」のとおりである。Debug の全ビルド、Release の Runner、既定スイート、PPO 系のテスト（CPU と CUDA）、hidden の学習テスト（seed 3 つとも最適値）が通った。CartPole の Run（A13）は、完了報告のあとに開発者の指示を受けて流し、3 seed とも基準を満たした（「CartPole の試運転の結果」）。

変更ファイル:

- 新規: `src/algo/ppo/`（24 ファイル）、`src/core/base` の `distribution`・`advantage`（ヘッダ・実装・テストの 6 ファイル）、`apps/runner/config/CartPole_ppo.txt`、`docs/design/115_on_policy_learning.jp.md`・`230_ppo_agents.jp.md`、`docs/wiki/categories/326_ppo.md`、RFD とこの実装計画。
- 変更: ルートと Runner の `CMakeLists.txt`、`RunnerApp.cpp`、`anet/nn_head.hpp`・`anet/tensor_store.hpp` とその実装・テスト、`nn_modules.cpp`、`agent.txt`・`nn.txt`・`metrics_scalar.txt`、検証資材 3 つ、設計文書 7 本（README・010・011・100・110・130・150）、`CONTEXT.md`、`reports/tools/categories.json`、wiki（目録・記録・5 ページ）、`docs/management/issues/INDEX.md`（道具が作り直した表）。

レビュー結果:

- RFD: 4 往復で OK（1 往復目は実装の入口で 8 件、3 往復目は実装のあとの改訂に 1 件）。実装計画: 3 往復で OK（1 往復目に 1 件）。実装: 3 往復で OK（1 往復目に 15 件、2 往復目に 1 件）。指摘は全部直し、根拠が無いとして見送ったものは無い。
- レビューが残した再確認の点:
  - 評価の Actor の `Sync()` を呼ぶスレッドまでは、レビューがコードで追っていない（UI のスレッドが学習のロックを長く待たないことの確かめ）。
  - 設計文書の 010・110・130・150・README・100 は、差分と関係する節だけが読まれている。
  - CartPole の Run（A13）と、Runner の画面での表示は、レビューの時点では未実施だった。Run は、そのあとに流した（「CartPole の試運転の結果」）。Eval View・Q-Values・Trace のペインの表示は、未確認のままである。
  - RFD は暫定の起票のままである。再グリルで暫定の印を外すときは、未決表の全行、機能グループの見込み、前提の表の確かめの列、ADR を起こすかが、改めて見られる。

決まらなかったこと:

- 起票の候補が 7 件ある（候補の段階で、起票していない。下の表）。

判断してほしいこと:

- RFD0202 の未決表（U1〜U30）の暫定の既定を、再グリルでいつ見直すか。
- コミットの対象名（scope）に `PPO` を足すか（未決表の U18）。

次にできること:

記録する課題の候補 7 件

| # | 区分 | 内容 | 根拠 |
|---|---|---|---|
| 1 | 新規 | RFD0202 の仕事の課題（PPO 系と方策ベースの共通部品。再グリルが残る） | RFD に結び付いた課題が無く、課題の一覧に「どの課題からも指されない RFD」として出ている。質問していない（課題の起票は今回の指示の外） |
| 2 | 新規 | `anet-core-base-test` が、別のテストの背景スレッドのログの紛れ込みで、実行順しだいで 1 件落ちる | 基準コミットを含む 5 回の実行のうち 3 回で、別々のテストが落ちた。質問していない（検証で見つかった） |
| 3 | 既存 issue-20261123-9a0e78 へ追記 | `anet-env-imagecls-test` が、実行順しだいで wxWidgets のメッセージボックスを出して止まる（seed 3387089429、`No image handler for type 15 defined.`） | 同じ事象の課題が既にある。今回の seed と文面を足す。質問していない |
| 4 | 新規 | 設定キーを示す診断の監査の期待一覧が、基準コミットの時点で 19 件ずれている | 監査の終了値が、変更の前から 1 である。質問していない |
| 5 | 新規 | 設定の既定の検査が、`apps/tools/metrics_viewer.py` の 4 行（`app.title` ほか）を設定の行として拾う | 検査の終了値が、変更の前から 1 である。質問していない |
| 6 | 新規 | `PPOAgent` の Evaluation Trace の列（value、log 確率、行動の確率）と、画面の表示 | RFD の非ゴール（NG5）。試運転では Train View と Logs だけを見ていて、Eval View・Q-Values・Trace のペインは未確認。質問していない |
| 7 | 見送り推奨 | 設定を相対の名前で名指す、前からある文面 2 つ（`nn_head.cpp` の `LinearHead`、`agent.cpp` の `clone_model`） | 実装のレビューの範囲外の検出。影響が小さく、候補 4 の監査の見直しと一緒に扱える |

採用するものを番号で指定してください。すべて見送る場合はその旨を。
