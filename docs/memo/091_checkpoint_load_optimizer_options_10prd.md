# checkpoint からの継続で optimizer の設定（学習率ほか）を効かせる 暫定 PRD

> 起点: 2026-10-14 の PRD 整理。DropMerge の次の候補（後半で学習率を下げる、B256 で学習率を半分にする）を、
> 既存 Run の checkpoint から分岐して試すと、設定ファイルで変えた学習率が効かない。
> 事実は [930](930_serialize_10prd.md) の起点そのもので、本 PRD はそのうち Optimizer の retune だけを先に切り出す。
> 関連: [930_serialize_10prd.md](930_serialize_10prd.md)（Run 全体の Save / Load。本 PRD の上位）、
> [done/011_fused_adamw_10prd.md](done/011_fused_adamw_10prd.md)（FusedAdamW）、
> [084_noisynet_10prd.md](084_noisynet_10prd.md)（Noisy の sigma 用 param group）。
> 行番号は 2026-10-14 時点。実装時は近傍のシンボル名で再検索する。

## 1. 問題

`DefaultDQNAgent.auto_load_file` で checkpoint から継続すると、optimizer の設定値が checkpoint 側の値に戻る。
設定ファイルや CLI で `learner.alpha` を変えても、実効の学習率は checkpoint を作った Run の値のままになる。
警告も出ないので、気づけない。

同じことが学習率以外の optimizer の設定にも起きる。

| 設定キー（`DefaultDQNAgent.` 配下） | 既定（`@baseline`） | 例 |
|---|---|---|
| `learner.alpha` | `5e-4` | 継続時に学習率を下げても効かない |
| `learner.weight_decay` | `1e-2` | 継続時に重み減衰を変えても効かない |
| `learner.adam_eps` | `1e-5` | `@bf16` は `1e-4` にする（`agent.txt:98`）。bf16 を切り替えて継続しても eps は元のまま |
| `learner.noisy_weight_decay` | `0` | NoisyNet の sigma 用 param group の重み減衰。同上 |

影響しないもの: batch size、replay ratio、soft τ、PER の設定、評価間隔など、optimizer の外にある設定は、
継続時も設定ファイルの値が効いている。

## 2. 現状（コードで確認したこと）

- optimizer は Learner の構築時に設定から作る。`Learner::SetupOptimizer`
  （`core/anet-core/src/dqn_based_agent.cpp:2511`）が `AdamWOptions(alpha).weight_decay(weight_decay).eps(adam_eps)` を作り、
  NoisyNet が無ければ param group 1 つ、あれば 2 つ（通常と sigma。sigma の重み減衰は `noisy_weight_decay`）で
  `AdamW` か `FusedAdamW`（`AdamW` の派生、`include/anet/nn_util.hpp:105`）を構築する。
- 継続は構築の後に起きる。`DefaultDQNAgent` の構築の最後で `auto_load_file` があれば `LoadNetwork` を呼ぶ
  （`core/anet-core/src/default_dqn_agent.cpp:227-229`）。`LoadNetwork`（同 `:306`）は header、保存時の Config 文字列
  （verbose ログに出すだけ）、Network、Learner の順に読む。
- `Learner::Load`（`dqn_based_agent.cpp:3195`）は `archive.ReadTorchObject(*optimizer_)` だけで、
  LibTorch の optimizer の読み込みは moment と step に加えて各 param group の options（lr、weight_decay、eps、betas）も
  checkpoint の値で置き換える。読み込みの後に設定を当て直す処理は無い。
- 構築時のログ `Learner: lr=... weight_decay=... eps=...`（`dqn_based_agent.cpp:2515`、verbose）は読み込みの**前**に出るので、
  実効値ではなく設定値を示している。ログを見ても気づけない理由はここにある。
- ImageCls は学習率を更新ごとに設定し直している（`core/anet-core/src/image_cls_agent.cpp:409` の
  `SetAdamWLearningRate`）ので、学習率については同じ問題が起きない。Rainbow と MuZero には継続の経路が無い。

## 3. 提案の骨子（暫定の既定）

1. **継続時は、optimizer の設定値を現在の設定に合わせる。** moment と optimizer の step は checkpoint から引き継ぐ。
   930 の分類でいう Optimizer group の `retune` を、唯一の挙動にする。
2. **設定値を作る場所を 1 つにする。** param group ごとの options を設定から作る処理を `SetupOptimizer` から切り出し、
   構築時と `Learner::Load` の直後の両方で同じものを使う。読み込みの後に別の処理で値を書き戻す形にはしない。
3. **差があったことをログに残す。** 読み込みの後、param group ごとに実効値を `LOG::info` で出す。
   checkpoint の値と現在の設定値が違う options は、キー名・checkpoint の値・設定値を添えて `LOG::warn` で出す。
4. 設定キーは足さない。checkpoint の値をそのまま使いたい場合は、設定を checkpoint と同じ値にすればよい
   （checkpoint の Config 文字列は verbose ログで確認できる）。

設定を変えずに継続する場合は、checkpoint の値と設定値が同じなので、挙動は今と変わらない。

## 4. User Stories

1. 実験者として、checkpoint から継続した Run で学習率を下げ、その効果を確かめたい。
2. 実験者として、継続した Run の実効の学習率を、ログで確かめたい。
3. 実験者として、設定と checkpoint の値が違っていたときに、それを見落としたくない。

## 5. 決めること（暫定の既定）

| # | 決めること | 暫定の既定 | ほかの案 |
|---|---|---|---|
| 1 | 値が違うときにどちらを使うか | 現在の設定（retune）。差は `LOG::warn` に出す | 設定キーで `continue` / `retune` を選ぶ（930 の policy 名の enum。boolean にはしない）。違ったら fail-fast |
| 2 | 対象の options | `lr`、`weight_decay`、`eps`。sigma の group は `noisy_weight_decay` | betas も対象にする（設定キーが無く、常に LibTorch の既定なので既定では対象外） |
| 3 | 対象の Agent | DefaultDQN（`dqn::Learner`） | ImageCls の `weight_decay` と `eps` も揃える |
| 4 | param group の数が checkpoint と違う場合（NoisyNet の有無が違う継続など） | 今の LibTorch の挙動を実装時に確かめる。黙って通るなら、読み込みの前に fail-fast を足す | — |
| 5 | 実効値の見せ方 | ログだけ | 学習率を GetScalar で出し、指標で見られるようにする |

## 6. 受入基準

1. 単体テスト（`dqn_based_agent_test.cpp` の checkpoint 往復のテストと同じ組み立て）: `learner.alpha = 1e-4` で作った
   checkpoint を `learner.alpha = 5e-5` の設定で読むと、全 param group の `lr` が `5e-5` になる。
   moment（`exp_avg`、`exp_avg_sq`）と step は checkpoint と一致する。
2. 同じテストを `weight_decay` と `adam_eps` でも行う。NoisyNet ON では、sigma の group の `weight_decay` が
   設定の `noisy_weight_decay` になる。
3. 設定が checkpoint と同じなら、読み込み後の options と optimizer の State は今の実装と一致し、`LOG::warn` は出ない。
4. 値が違うと `LOG::warn` が出て、メッセージにキー名、checkpoint の値、設定値が入る。
5. smoke: 既存 Run の `agent_close.anet` を CLI の `auto_load_file` で読み、`learner.alpha` を変えて短く回すと、
   ログに実効値と警告が出る。

## 7. 対象外

- Run 全体の継続（ReplayBuffer、正規化の統計、`StepCounts` と schedule、Env、RNG）。930 が扱う。
- checkpoint の値を残す `continue` や、違ったら止める `require_match` の選択肢。930 の policy として入れる。
- manifest、実効値の Run artifact への記録。930 が扱う。
- 学習率の schedule。

## 8. 930 との関係

930 は「現行 `auto_load_file` へ個別 boolean を足すだけの暫定拡張」を対象外にしている。本 PRD は設定キーを足さず、
Optimizer group の `retune` を唯一の挙動にするだけなので、これに当たらない。930 が入るときは、本 PRD の挙動が
Optimizer group の `retune` policy になり、`continue` と `require_match` が選択肢として加わる。

過去の Run の読み方は 930 の記述のとおり。継続時に optimizer の設定を変えた Run は、checkpoint の値で学習していた。
