# PRD 061: eval スロットごとの policy / network 指定

- 起票日: 2026-08-24
- 状態: **draft**（本文の「決めるべきこと」は未確定。別枠のグリルで詰める）
- 対象: `core/anet-core`（`default_dqn_agent.cpp` の policy 生成と `CreateActor`、`trainer.cpp` の eval スロット構築、`rl.hpp` の `Agent` IF）
- 関連: `060_eval_batch_episodes_10prd.md`（本数。§「本 PRD では解けない隣接論点」で本件を名指ししている）、`done/052_eval_schedule_separation_10prd.md`（定義とスケジュールの分離。実装済み）、`912_background_eval_snapshot_ordering_10prd.md`（同じ eval 経路の別論点）
- 発見経緯: Atari Breakout の探索（`docs/experiments/default-dqn/atari/2026-08-17_baseline.md` 探索ブロック 14 / 16 / 17 / 18 / 19）で、eval の ε と評価対象ネットワークが Run 単位でしか選べないために測れなかった項目が積み上がった

## Context（背景・目的）

eval スロット（`train.eval.[tag]`）は 052 で定義とスケジュールが分離され、**タグ単位で env 設定・`run_mode`・`eval_batch_size`・`clone_model` を持てる**ようになった。しかし **行動方策（ε 等）だけは agent 単位で 1 個**であり、スロットごとに変えられない。加えて **評価対象ネットワーク（target / online）の選択は `run_mode` に固定配線**されている。

この 2 つの縛りにより、「同じ Run の中で、違う ε・違うネットワークの eval を並べる」ことができない。文献比較・回帰追跡・分布測定はそれぞれ別の eval 設定を要求するため、1 Run では最大 1 つしか成立しない。

## 現行コードで確定している事実

### 1. eval policy は agent 構築時に 1 個だけ

```cpp
this->eval_policy_ = CreateActionPolicy(config_.eval_policy, false, num_envs_, device_);
```
`core/anet-core/src/default_dqn_agent.cpp:194`

設定側も 1 個。`ActionPolicyConfig train_policy / eval_policy / target_policy` の 3 本しかない（`core/anet-core/include/anet/default_dqn_agent.hpp:26-28`）。

### 2. eval 時は無条件にその 1 個が選ばれる

```cpp
if (anet::rl::IsEval(run_mode)) {
    policy = eval_policy_;
    src_network = IsForTarget(run_mode) ? model_->GetTargetNetwork() : model_->GetOnlineNetwork();
}
```
`core/anet-core/src/default_dqn_agent.cpp:496-497`

### 3. target / online の選択は `RunMode::Eval1` かどうかだけで決まる

```cpp
static bool IsForTarget(anet::rl::RunMode run_mode)
{
    if (!anet::rl::IsEval(run_mode)) return false;
    return run_mode == anet::rl::RunMode::Eval1;
}
```
`core/anet-core/src/default_dqn_agent.cpp:458-466`

`RunMode` は `Train / Eval / Eval1 / Eval2` の 4 値（`core/anet-core/include/anet/rl.hpp:147-153`）。**eval のバリエーションは実質 2 通りが上限**である。

### 4. eval スロットが持てるキーは 3 つ

`trainer.cpp` の eval スロット構築が読むのは `run_mode`（:875）、`eval_batch_size`（:880）、`clone_model`（:888）だけ。env 設定は `config_prefix = "train.eval.[" + tag + "].env"`（:868）で env factory 側へ渡る（:920）。**policy に相当するキーは存在しない。**

`EvalRunner` の生成は `trainer.cpp:923` で、`run_mode` と `clone_model` と device を渡すのみ。

### 5. `Agent::CreateActor` に policy を渡す口が無い

```cpp
virtual std::shared_ptr<Actor> CreateActor(
    const BatchEnvSpec&, const EnvSpec&, RunMode run_mode,
    std::optional<bool> clone_model_override = std::nullopt,
    std::optional<torch::Device> device = std::nullopt) const = 0;
```
`core/anet-core/include/anet/rl.hpp:727-732`

実装は 4 つ（`default_dqn_agent.cpp` / `rainbow_agent.cpp` / `muzero_proto_agent.cpp` / `image_cls_agent.cpp`）。`eval_policy` を持つのは `default_dqn_agent.cpp` のみ。

### 6. metrics の束ね直しはスロットタグ基準

```
# PRD 060 P2 適用後。左辺の metrics tag は維持し、Runner source key だけを移行する。
metrics.scalar.full.[21_eval/01_target_reward] = mean.episode_return $runner @episode_end $eval.[eval1] clip:3000
```
`apps/runner/config/metrics_scalar.txt:120`

`$eval.[eval1]` は**スロットタグで束ねており run_mode では束ねていない**。したがって run_mode を付け替えても metrics の配線は切れない。ただしタグ名（`01_target_reward` / `02_policy_reward` / `51_eval1` / `52_eval2`）は「eval1 = target net」という現在の固定配線を名前に埋め込んでいる。

### 7. 現行の運用値

```
train.eval.[eval1].run_mode = eval1     # → target net
train.eval.[eval2].run_mode = eval2     # → online net
train.eval.[eval1].eval_batch_size = 1
train.eval.[eval2].eval_batch_size = 1
```
`apps/runner/config/common.txt:5-21`

ε は `DefaultDQNAgent.eval_policy.eps_start / eps_end` の 1 組で、両スロットに共通で効く。

## 問題: 縛りが実際に潰した測定

探索記録の該当項目を挙げる。いずれも「1 Run では 1 つしか成立しない」ことが原因である。

### A. 文献比較の土俵が 1 Run で組めない

Nature DQN 2015 は **ε=0.05 / 30 本**、BTR は **ε=0.01 / 100 本**（060 §「文献側が要求する本数と ε」）。ε が agent 共通なので、**どちらか一方の土俵しか同時に置けない**。Atari の 2 プロファイル体制（Classic / v5）は env 側をスロットごとに切り替えられるのに、ε だけは Run 単位という非対称が残っている。

### B. 決定論 eval と文献 eval が排他になった

探索ブロック 16 で v5 の FIRE デッドロックを回避するため eval を ε=0.01 にした。その結果、メタデータ「eval の決定論性」に記録した性質（**ある重みに対する eval は毎回同一の 1 軌道**）が失われ、探索ブロック 01〜14 の eval 値と土俵が変わった。**ε=0 の決定論 eval（回帰追跡用）と ε>0 の文献 eval を並べて持てない。**

### C. eval1 / eval2 の健全性指標と ε 変更が両立しない

まとめ 17 の「eval1（target net）と eval2（online net）の比が 1 に近いほど online net が健全」という指標は、**両者が同一 policy であること**が前提。ε を文献値に合わせたい局面でこの指標を残すには 3 つ目のスロットが要るが、3 つ目も同じ ε になるため意味を持たない。現行 Atari が eval2 を停止しているのはこの帰結でもある。

### D. online net を 2 通りの ε で評価できない

`IsForTarget` が `Eval1` にだけ true を返すため、**(network, policy) の 4 通りのうち実現できるのは 2 通り**（target×共通ε、online×共通ε）。文献比較は online net（まとめ 17）を要求するので、「online net を ε=0.05 と ε=0.01 の両方で」が本来ほしい形だが表現できない。

### E. 分布の測定が train 側からしか取れない

探索ブロック 18・19 の分布比較（`<100` 率、`≥432` 率、p25 など）は**すべて train エピソードで測っている**。eval が 1 本・決定論なので分布が作れないためである。本数（060）と ε（本 PRD）が揃わないと eval 側で分布を測れない。

### F. NN 構成 A/B の判定手段が無い（まとめ 19）

決定論の 1 軌道では 5% の差（331.4 対 314.0）が判定不能だった。ε>0 と本数増で標本を作るしかない。

## 目標契約（案。要確定）

1 つの Run の中で、**スロットごとに独立に「どのネットワークを」「どの方策で」評価するか**を指定できる。例:

```
train.eval.[greedy] : network = target ; policy = greedy      # 回帰追跡（決定論 1 軌道）
train.eval.[btr]    : network = online ; policy = btr         # ε=0.01、文献比較
train.eval.[nature] : network = online ; policy = nature      # ε=0.05、文献比較
```

上の記法は説明用であり、キー名・階層は未確定。

## 解決候補（最終選択は別枠のグリルで）

### 候補 A: eval スロットが policy 設定を直接持つ

```
train.eval.[eval1].policy.policy_type = EpsilonGreedy
train.eval.[eval1].policy.eps_start   = 0.05
```

- 変更は trainer 側だけで閉じるように見える
- **難点**: `ActionPolicyConfig` は agent 固有（`tau_rule` / `uqe_*` / `full_distribution_query` を含む）。trainer が agent の設定スキーマを知ることになり、抽象が漏れる。`Agent::CreateActor` に `ConfigData` を流し込む形になりやすい

### 候補 B: agent が named eval policy のカタログを持ち、スロットは名前で参照

```
DefaultDQNAgent.eval_policy.[btr].policy_type = EpsilonGreedy
DefaultDQNAgent.eval_policy.[btr].eps_start   = 0.01
DefaultDQNAgent.eval_policy.[nature].eps_start = 0.05

train.eval.[eval1].policy = btr
```

- agent 固有の設定は agent 側に残り、スロットが持つのは `run_mode` と同じ「名前 1 個」だけ
- `[tag]` カタログは PRD 059 のカタログ層と同形で、config 体系に既に存在する形
- agent 構築時に `map<string, shared_ptr<ActionPolicy>>` を作り、`CreateActor` が名前で引く
- 未定義名は fail-fast（059 の素材未定義参照と同じ扱い）
- 後方互換: `policy` 未指定なら現行の無名 `eval_policy` を使う

### 候補 C: `run_mode` を分解し、network 軸をスロットが持つ

```
train.eval.[eval1].network = target      # target | online
```

- `IsForTarget` の run_mode 依存を廃止する
- `RunMode::Eval1 / Eval2` は metrics 束ね・seed 領域・Env factory の識別子としてのみ残す（あるいは `Eval` に一本化）
- **難点**: `RunMode` は `Agent` IF・seed 領域（`"eval_env/" + tag`）・`EnvFactory::ValidateConfig`・EvalPanel まで波及する。影響範囲は B より広い

**B と C は直交する。** B が policy 軸、C が network 軸で、問題 D（online net を 2 通りの ε で）を解くには両方が要る。問題 A / B / E / F は B だけで解ける。

## 決めるべきこと（未確定）

| # | 論点 | 備考 |
|---|---|---|
| D1 | policy 軸だけ解くか、network 軸（候補 C）も同時に外すか | A/B/E/F は B のみで解ける。D は B+C が要る |
| D2 | named policy の置き場と記法 | `DefaultDQNAgent.eval_policy.[name]` か。無名 `eval_policy` を `[default]` へ寄せるか、無名と併存させるか（059 の shrink 議論と同型） |
| D3 | 既定値と後方互換 | `policy` 未指定時は現行の無名 `eval_policy`。既存 config を 1 行も変えずに現行挙動になること |
| D4 | `Agent::CreateActor` の変更形 | 引数追加（`std::optional<std::string> eval_policy_name`）か、`EvalActorSpec` 構造体化か。実装 4 つのうち 3 つは無視するだけになる |
| D5 | 未定義 policy 名の扱い | fail-fast（参照元スロット名・参照名・在庫一覧をメッセージに含める）で良いか |
| D6 | metrics タグ名の扱い | `21_eval/01_target_reward` / `02_policy_reward` / `51_eval1` / `52_eval2` は「eval1=target」を名前に埋め込んでいる。C を採ると名前と実体がずれる。改名するか、名前は単なるスロット識別子と割り切るか |
| D7 | スロット数の上限と命名 | 3 本以上（`greedy` / `btr` / `nature`）を常用するのか。`common.txt` の既定は 2 本のままで良いか |
| D8 | wall-clock コスト | 探索ブロック 03 で eval が throughput を 2.5 倍動かした実績がある。本数（060）と掛け算になるため、`interval` とセットで決める必要がある |
| D9 | 060 との着手順 | 060 側は「本数が先でよい」と書いている。逆順・同時のいずれが良いか |
| D10 | policy 差し替えで `OnLearn` がどうなるか | `eval_policy_->OnLearn(counts)`（`default_dqn_agent.cpp:564`）は 1 個前提。カタログ化したら全数に配るのか、eval policy は decay を持たない前提で据え置くのか |

## スコープ外

- **eval のエピソード本数**（`060_eval_batch_episodes_10prd.md`）。本 PRD は ε とネットワークの軸だけを扱う
- **eval のスケジュール**（052。実装済み）
- **background eval の snapshot 順序**（912）
- **best checkpoint の選択**（913）
- **train policy のスロット化**。train は 1 本しか無いので該当しない
- **eval1 / eval2 の停止・復活の運用判断**。設定で表現できるようにするところまでが本 PRD

## Further Notes

- 問題 C の健全性指標（eval1/eval2 比）は、B を入れた後は「**同一 policy を指定した 2 スロット**」として明示的に組めるようになる。現在は暗黙に同一 policy であることに依存している
- 探索ブロック 16 の FIRE デッドロックは v5（`fire_reset=false`）固有で、Classic では発生しない。したがって「ε=0 の決定論 eval」は Classic では今も成立する。本 PRD が要るのは v5 側である
