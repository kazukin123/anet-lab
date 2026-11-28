# Survey: TorchRL の設計と現状 ― Transforms、実装の参考、実験結果の比較元

Date: 2026-11-20
Scope: TorchRL（GitHub の pytorch/rl。PyTorch 公式の強化学習ライブラリ）と、その土台の tensordict パッケージを調べる。部品の構成（TensorDict・TensorSpec、env、Transforms、Collector、ReplayBuffer、損失、Trainer）、とくに Transforms の spec の変換・フレームの重ね方・正規化・Replay 側での扱いを原典のコードと文書で確かめる。あわせて、収録している手法（DQN 系と内的報酬）、公開している実験結果とその条件、実時間の性能、C++ から使えるか、採用状況と批判を調べる。最後に、anet の部品と PRD161 の観測パイプラインとの対応を筆者の整理として示す。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。原文の逐語は著作権への配慮から 1 か所に絞り、ほかは要約と位置（ファイルと行、節、表）で示す。コードの行番号は、§2〜§4 が v0.15.0 の tag（commit `a711ef99`、2026-11-10 公開）、§5〜§9 が main の commit `40407917`（2026-11-19）のものである。tensordict は v0.15.6（2026-11-11 公開）を見た。出典の層は「公式文書」「公式コード」「公式リリース・PR・Issue」「論文」「第三者」に分け、各項に書く。§11 だけは筆者が anet の文書と並べた整理で、TorchRL の出典が anet について述べているわけではない。調査は観点ごとに 3 本のサブエージェントで行い、本冊で統合した。主張のうち 3 点（§3.3 の Replay 側の CatFrames、§3.5 の Replay の transform が働く時点、§3.6 の LastAction）は本冊で v0.15.0 のコードを取得して照合した。

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. 設計の全体像](#2-設計の全体像)
   - [2.1 論文が掲げる設計の柱](#21-論文が掲げる設計の柱)
   - [2.2 TensorDict と TensorDictModule](#22-tensordict-と-tensordictmodule)
   - [2.3 TensorSpec と env の spec](#23-tensorspec-と-env-の-spec)
   - [2.4 env の step と reset の契約](#24-env-の-step-と-reset-の契約)
3. [3. Transforms](#3-transforms)
   - [3.1 Transform と TransformedEnv と Compose](#31-transform-と-transformedenv-と-compose)
   - [3.2 spec の変換とメソッドの使い分け](#32-spec-の変換とメソッドの使い分け)
   - [3.3 CatFrames の環境側と Replay 側](#33-catframes-の環境側と-replay-側)
   - [3.4 正規化と報酬の Transform](#34-正規化と報酬の-transform)
   - [3.5 Replay に付けた Transform が働く時点](#35-replay-に付けた-transform-が働く時点)
   - [3.6 観測へ情報を足す Transform](#36-観測へ情報を足す-transform)
   - [3.7 Atari の DQN 参照実装の組](#37-atari-の-dqn-参照実装の組)
4. [4. Collector と ReplayBuffer と損失](#4-collector-と-replaybuffer-と損失)
   - [4.1 Collector](#41-collector)
   - [4.2 ReplayBuffer](#42-replaybuffer)
   - [4.3 損失と target の更新](#43-損失と-target-の更新)
   - [4.4 Trainer と版の変化](#44-trainer-と版の変化)
5. [5. 収録している手法](#5-収録している手法)
   - [5.1 参照実装の置き場と一覧](#51-参照実装の置き場と一覧)
   - [5.2 DQN 系の対応状況](#52-dqn-系の対応状況)
   - [5.3 探索と内的報酬](#53-探索と内的報酬)
6. [6. 公開している実験結果](#6-公開している実験結果)
   - [6.1 論文の結果と条件](#61-論文の結果と条件)
   - [6.2 参照実装の README と継続検証](#62-参照実装の-readme-と継続検証)
   - [6.3 ほかの実装と比べられる形か](#63-ほかの実装と比べられる形か)
   - [6.4 ほかの公開実装の数字の条件と置き場](#64-ほかの公開実装の数字の条件と置き場)
7. [7. 実時間の性能](#7-実時間の性能)
   - [7.1 論文の速度比較](#71-論文の速度比較)
   - [7.2 compile と CUDA graph](#72-compile-と-cuda-graph)
   - [7.3 第三者の速度の報告](#73-第三者の速度の報告)
   - [7.4 TensorDict の受け渡しの重さ](#74-tensordict-の受け渡しの重さ)
8. [8. C++ から使えるか](#8-c-から使えるか)
9. [9. 採用状況と方向](#9-採用状況と方向)
10. [10. 批判と懸念](#10-批判と懸念)
    - [10.1 批判](#101-批判)
    - [10.2 支持と保守者の応答](#102-支持と保守者の応答)
11. [11. anet との対応（筆者の整理）](#11-anet-との対応筆者の整理)
    - [11.1 部品の対応](#111-部品の対応)
    - [11.2 TensorDict と TensorSpec の同じ点と違う点](#112-tensordict-と-tensorspec-の同じ点と違う点)
    - [11.3 Transforms と PRD161 の観測パイプライン](#113-transforms-と-prd161-の観測パイプライン)
    - [11.4 実装の参考にできる範囲](#114-実装の参考にできる範囲)
    - [11.5 実験結果の比較元にできる範囲](#115-実験結果の比較元にできる範囲)
12. [12. 総合評価](#12-総合評価)
13. [13. 調査の限界](#13-調査の限界)
14. [14. 出典一覧](#14-出典一覧)

## 1. 対象と読み方

TorchRL の論文は Bou, Bettini, Dittert, Kumar, Sodhani, Yang, De Fabritiis, Moens による arXiv:2306.00577 で、v1 が 2023-06-01、v2 が 2023-11-27 に出た。ICLR 2024 に Spotlight Poster として採られている。著者の所属は、Universitat Pompeu Fabra と Acellera（Bou、Dittert、De Fabritiis）、University of Cambridge（Bettini）、Meta AI（Kumar、Sodhani、Yang）、PyTorch Team, Meta（Moens）である。被引用数は Semantic Scholar で 102（うち influential 7）、OpenAlex で 122（どちらも 2026-11-20 取得）。【論文】[TorchRL論文 (2023/11), 表紙]、[ICLR 2024 (2024/05), ページ見出し]、[Semantic Scholar (2026/11)]、[OpenAlex (2026/11)]

調べた範囲の部品を、論文と公式文書の区分で並べると次のとおり。

| 区分 | 主な部品 | 本冊の節 |
|---|---|---|
| データの受け渡し | TensorDict、TensorDictModule、TensorSpec | §2 |
| 環境 | EnvBase、ParallelEnv・SerialEnv、GymEnv などのラッパー | §2.4 |
| 変換 | Transform、TransformedEnv、Compose | §3 |
| 収集 | Collector、MultiSyncCollector、MultiAsyncCollector、分散 collector | §4.1 |
| 経験の保存 | ReplayBuffer（storage・sampler・writer・transform） | §4.2 |
| 学習 | LossModule、value estimator、target の更新、Trainer | §4.3、§4.4 |

## 2. 設計の全体像

### 2.1 論文が掲げる設計の柱

論文は、汎用の意思決定ライブラリの柱として、モジュールの可搬性・効率的な通信・コードの再利用を挙げ、結論では、TensorDict でつながる単一責任の部品の集まりとして TorchRL を要約している。【論文】[TorchRL論文 (2023/11), §1 終盤・§5]

部品どうしは多くを TensorDict でやり取りするので、collector は方策の構造を、ReplayBuffer は格納するデータの構造を知らずに作れる、と論文は説明している。【論文】[TorchRL論文 (2023/11), §2 冒頭]

依存は PyTorch・NumPy・tensordict に絞り、環境・分散・ロガーは任意の依存にしている。アーキテクチャやデータ収集の設定で、偏った選択はしない方針である。番号を振った「設計原則」の一覧は論文に無い。【論文】[TorchRL論文 (2023/11), 付録 I]

### 2.2 TensorDict と TensorDictModule

TensorDict は、先頭の形（batch size）を共有するテンソルの辞書である。インデックスや reshape は全エントリーの batch の次元に一度にかかる。値に別の TensorDict を入れられ、キーは `td["a","b"]` のように入れ子にできる。【公式コード】[tensordict@v0.15.6 (2026/11), `tensordict/_td.py` L251–289]

制約は 3 つある。batch size は推論せず既定は `[]`、`device` を指定すると全エントリーをその device へ移す、キーは文字列か文字列のタプルに限る。【公式コード】[tensordict@v0.15.6 (2026/11), `_td.py` L291–301]

ほかに次の仕組みがある。【公式コード】【公式リリース】

- **lazy stack**: `LazyStackedTensorDict` は入力を別々に持ち、読み出したときに積む。キーや形が違う入力も受けられる。[tensordict@v0.15.6 (2026/11), `_lazy.py` L1038–1066]
  - 論文の付録 B は「`torch.stack` の結果が lazy stack になる」と書いている。これは tensordict v0.4.0 で既定が変わり、いまは密な TensorDict を返す。[TorchRL論文 (2023/11), 付録 B.1]、[tensordict v0.4.0リリース (2024/04), Refactor 欄]
- **memmap**: `MemoryMappedTensor` は記憶域がディスク上のファイル（名前が無ければ無名の共有メモリ）の CPU テンソルで、プロセス間では参照だけを送る。[tensordict@v0.15.6 (2026/11), `memmap.py` L178–187]

TensorDictModule は関数や module を包み、`in_keys` から読んで `out_keys` へ書く。TensorDictSequential はそれを順に実行し、全体の `in_keys` は「どの先行 module も書かないキー」になる。論文は、両者が `torch.compile`・functorch・vmap と両立すると書いている。【公式コード】【論文】[tensordict@v0.15.6 (2026/11), `nn/common.py` L862–887・`nn/sequence.py` L39–51]、[TorchRL論文 (2023/11), §2]

### 2.3 TensorSpec と env の spec

v0.15 の spec のクラスは Binary、Bounded、Categorical、Composite、MultiCategorical、MultiOneHot、NonTensor、OneHot、Stacked、StackedComposite、Unbounded（continuous・discrete）などである。【公式文書】[TorchRL文書 (2026/11), `data_specs.rst`]

env の spec は全体を Composite で持ち、入力側（`full_action_spec`、`full_state_spec`）と出力側（`full_observation_spec`、`full_reward_spec`、`full_done_spec`）に分かれる。【公式文書】[TorchRL文書 (2026/11), `envs_api.rst` L19–45]

spec のクラス名は 2 段で変わった。【公式リリース・PR】【公式コード】

1. v0.6.0（2024-10）の PR #2368 で、`CompositeSpec` → `Composite`、`BoundedTensorSpec` → `Bounded`、`DiscreteTensorSpec` → `Categorical`、`OneHotDiscreteTensorSpec` → `OneHot` のように短い名前へ変えた。旧名は非推奨の別名として残った。[pytorch/rl PR #2368 (2024/08)]、[TorchRL v0.6.0リリース (2024/10)]、[rl@v0.10.0 (2025/09), `tensor_specs.py` L6900–7023]
2. v0.11.0（2026-01）で旧名のクラスを削除した。論文 v2 の例は旧名を使っている。[TorchRL v0.11.0リリース (2026/01), Refactors 欄]、[TorchRL論文 (2023/11), 付録 C]

### 2.4 env の step と reset の契約

`reset` は rollout の最初の TensorDict を返す。`step` は action を含む TensorDict を受け取り、観測・done・報酬を `"next"` の下へ書く。`step_mdp` が `"next"` を根へ移して次の入力を作る。`step_mdp` の既定は、現在の action を除き、報酬を除き、done を残す。【公式文書】【公式コード】[TorchRL文書 (2026/11), `envs_api.rst` L178–247]、[rl@v0.15.0 (2026/11), `torchrl/envs/utils.py` L328–371]

終わりの合図は 3 つのキーで表す。【公式文書】[TorchRL文書 (2026/11), `envs_api.rst` L248–260]

- すべての env が `"done"` と `"terminated"` を出し、片方が無ければ補われる。
- `"done"` は軌跡の終わりの合図すべての和で、reset が要る合図である。
- truncation を出す env では `"truncated"` も書かれる。collector と rollout は既定で `"done"` を見て reset する。

`step_and_maybe_reset` は、`"next"` を持つ更新済みの入力と、次の step 用の入力（終わった要素だけ reset した結果）の 2 つを返す。束ねた env では、部分 step・部分 reset を一時キー `"_step"`・`"_reset"` の真偽マスクで指示する。【公式文書】[TorchRL文書 (2026/11), `envs_api.rst` L208–224・`envs_vectorized.rst` L132–205]

束ね方は次のとおり。【公式文書】[TorchRL文書 (2026/11), `envs_vectorized.rst` L6–33・L110–119・L316–336]

- `ParallelEnv` は `SerialEnv` を継承し、env の数を batch size とする。spec から通信用バッファを作るため、`check_env_specs` が通ることが前提である。
- `SerialEnv` は直列で、主にテスト用である。
- `MultiThreadedEnv` は EnvPool を使い、速いが EnvPool にある環境に限られる。
- `AsyncEnvPool` は、一部の env を動かしている間に別の処理を進められる。

v0.15.0 で `reset` と `rollout` の既定が `set_state=False` になり、入力に状態が入っていても既定では新しく reset する。【公式リリース】[TorchRL v0.15.0リリース (2026/11), Breaking changes]

## 3. Transforms

### 3.1 Transform と TransformedEnv と Compose

`TransformedEnv(base_env, transform)` で env に Transform を重ね、複数の Transform は `Compose` でまとめる。Transform は `Transform` のサブクラスでも、TensorDict を受けて返す関数でもよい。既定では base env の device を引き継ぐ。【公式文書】[TorchRL文書 (2026/11), `envs_transforms.rst` L14–30]

Transform は 1 つの親にしか登録できず、`transform.parent` は「base env と、その Transform までの Transform」からなる TransformedEnv を返す。ほかの env へ移すときは `clone()` を使う。TransformedEnv を入れ子にすると、既定（`auto_unwrap=True`）で内側の Transform を外側へ展開する。【公式文書】【公式コード】[TorchRL文書 (2026/11), `envs_transforms.rst` L37–46・L202–242]、[rl@v0.15.0 (2026/11), `torchrl/envs/transforms/_base.py` L989–1037]

論文は、Transform を普通の `nn.Module` と位置づけ、env だけでなく ReplayBuffer・collector・モデルでも使えると書いている。【論文】[TorchRL論文 (2023/11), §2 Environment API]

### 3.2 spec の変換とメソッドの使い分け

キーの向きは、`in_keys`・`in_keys_inv` が base env 側、`out_keys`・`out_keys_inv` が外側（方策）側を指す。逆変換（`inv`）は合成の逆順で走り、action の spec は内側から外側へ変換される。たとえば ActionDiscretizer を付けると、外には `"action_discrete"` が見え、`"action"` は見えなくなる。【公式文書】[TorchRL文書 (2026/11), `envs_transforms.rst` L68–131]

spec の変換は 2 層のメソッドで行う。【公式コード】[rl@v0.15.0 (2026/11), `_base.py` L702–818]

- 上位: `transform_output_spec` が観測・報酬・done の spec を、`transform_input_spec` が状態・action の spec を、下位のメソッドへ振り分ける。
- 下位: `transform_observation_spec`・`transform_reward_spec`・`transform_done_spec`・`transform_action_spec`・`transform_state_spec`。既定は恒等写像である。
- `out_keys` にあって `in_keys` に無いキー（Transform が新しく足すキー）が出力 spec に登録されていないと、KeyError になる。

データの変換のメソッドは、呼ばれる経路で分かれている。【公式コード】[rl@v0.15.0 (2026/11), `_base.py` L370–374・L401–457・L530–639・L1373–1422・L1812–1853]

| メソッド | 呼ばれる経路 | 用途 |
|---|---|---|
| `_apply_transform` | `_call` と `forward` から `in_keys` ごとに | 1 キー分のテンソルの変換 |
| `_call` | env の step と reset | 前の step に依らない出力の変換 |
| `_step` | env の step だけ | 時刻 t の入力と親の env が要る変換（FrameSkip など）。既定は `_call` を呼ぶ |
| `_reset` | env の reset | 状態を持つ Transform の初期化 |
| `forward` | ReplayBuffer の sample など env の外 | env の step では呼ばれない |
| `_inv_call`・`inv` | env の step の前、ReplayBuffer の書き込み | 行動側の逆変換。書き込み時の除外にも使う |

TransformedEnv の step は、`inv` で入力を逆変換し、base env の step を実行し、`_step` で出力を変換する、の順に進む。Compose は順変換を前から、逆変換を後ろから回す。v0.15.0 では `inv` の docstring が「`_inv_call` を呼ばない」と書く一方、コードは呼んでいる。【公式コード】[rl@v0.15.0 (2026/11), `_base.py` L671–691・L1373–1422]

公式の文書は、Transform を書くときの指針を 3 つ挙げている。各テンソルに同じ処理をかけるなら `_apply_transform` を書く。step の入出力の両方が要るなら `_step`、そうでなければ `_call` を書く。ReplayBuffer で使うなら `forward`・`inv` も書く。【公式文書】[TorchRL文書 (2026/11), `envs_transforms.rst` L150–171]

### 3.3 CatFrames の環境側と Replay 側

CatFrames はフレームを重ねる Transform で、環境側と ReplayBuffer 側の 2 通りの使い方を公式に持つ。【公式コード】[rl@v0.15.0 (2026/11), `torchrl/envs/transforms/_observation.py` L873–1236]

- **環境側**: 状態を持ち、step ごとにバッファを `torch.roll` でずらして新しいフレームを書く。reset された要素は、`padding="same"`（既定）なら最初のフレームの繰り返しで、`"constant"` なら定数で埋め直す。根にある `"done"` しか扱わず、入れ子の done（マルチエージェント）には対応していない。
- **Replay 側**: `as_inverse=True` は ReplayBuffer 専用で、env に付けると例外になる。オフライン用の `future=True` と `mask_key` が v0.14 で加わった。

Replay にフレームを重ねたまま保存するとメモリが N 倍になる、と CatFrames の docstring は注意している。対策として、軌跡のまま保存し、抽出時に CatFrames をかける使い方を示している。

```
Storing stacks of frames in the replay buffer can significantly increase memory consumption

フレームを重ねたものを replay buffer に保存すると、メモリの消費が大きく増えうる
```

[rl@v0.15.0 (2026/11), `_observation.py` L981]

その組み立てを `make_rb_transform_and_sampler(batch_size)` がまとめて返す。【公式コード】[rl@v0.15.0 (2026/11), `_observation.py` L1051–1120]

1. `SliceSampler(slice_len=N)` で、連続する N 個の遷移を切り出す（`_batch_size_multiplier = N`）。
2. Transform の Compose は 4 段。`reshape(-1, N)` → `("next", key)` も対象にした CatFrames → 最後の時刻だけ取る → 重ねたキーを書き込み時に落とす `ExcludeTransform(..., inverse=True)`。

この使い方には、docstring に 3 つの注意がある。画像では、前段の ToTensorImage の入出力のキーを別名にする。画像以外では CatFrames の前に RenameTransform を入れる。CatFrames より前の Transform も Replay 側に渡し、収集時と同じ形のデータを CatFrames に見せる。【公式コード】[rl@v0.15.0 (2026/11), `_observation.py` L1077–1089]

公式チュートリアルは、環境側の Compose と、`("next", ...)` のキーを足した同じ Compose を ReplayBuffer に付け、重ねたキーを除いて保存し、抽出した結果がオンラインの結果と一致することを確かめている。理由には、メモリの消費と「学習時と推論時で窓の長さが違う場合」を挙げる。窓の切り出しは関数 `functional.cat_frames` としても公開されている。【公式文書】[rl@v0.15.0 (2026/11), `tutorials/sphinx-tutorials/rb_tutorial.py` L719–791]、[TorchRL文書 (2026/11), `envs_transforms.rst` L350–366]

### 3.4 正規化と報酬の Transform

正規化の Transform は 3 つあり、統計をいつ更新するかが違う。【公式コード】[rl@v0.15.0 (2026/11), `_normalization.py` L52–80・L179–275・L452–490・L640–652]、[rl@v0.15.0 (2026/11), `vecnorm.py` L33–75・L320–333・L408–457]

| Transform | 変換 | 統計の更新 |
|---|---|---|
| ObservationNorm | `obs*scale+loc`（`standard_normal=True` なら `(obs-loc)/scale`） | `init_stats` が親の env を方策なしで rollout し、平均と標準偏差から一度だけ決める。初期化済みで呼ぶとエラー |
| VecNorm（旧） | 移動平均による正規化 | `_call` の中、つまり env の step の経路で更新する。eval モードでは更新しない。共有メモリで複数プロセスが統計を共有でき、推論用に `to_observation_norm()`・`frozen_copy()` で固定できる |
| VecNormV2 | 同上 | `_step` と `_reset` の中で、凍結されていなければ更新する。`freeze()`・`unfreeze()` を持つ。統計を内部に持つモードと、TensorDict で運ぶモードがある。`decay` の既定は 0.9999 |

v0.15.0 でも、`new_api` を指定しなければ旧 VecNorm が返り、FutureWarning が出る。警告の文言は「v0.10 で VecNormV2 に置き換える」のまま残っている。v0.12.0 で両者に逆変換が加わった。【公式コード】【公式リリース】[rl@v0.15.0 (2026/11), `_normalization.py` L452–466]、[TorchRL v0.12.0リリース (2026/04)]

報酬側には RewardScaling（アフィン変換）、RewardSum（エピソードの累積を `episode_reward` に書く）、RewardClipping、BinarizeReward、SignTransform、Reward2GoTransform などがある。【公式コード】【公式文書】[rl@v0.15.0 (2026/11), `_normalization.py` L370–386・`_reward.py` L368–390]、[TorchRL文書 (2026/11), `envs_transforms.rst` L244–348]

### 3.5 Replay に付けた Transform が働く時点

ReplayBuffer に付けた Transform は、読み出し（`sample`・`__getitem__`）で順変換が、書き込み（`extend`・`add`・`__setitem__`）で逆変換（`inv`）が呼ばれる。`transform` 引数の説明も「`sample` のときに実行する」と書いている。`add` は Transform が時間の次元を期待するため、一時的に次元を足す。【公式コード】[rl@v0.15.0 (2026/11), `torchrl/data/replay_buffers/replay_buffers/base.py` L216–224・L1772–1794・L2265–2273・L2413–2455]

論文は、ReplayBuffer の Transform は env の Transform と同じもので、収集時の処理をオフラインのデータセットにも使い回せると書く。例として、画像を uint8 で保存して取り出し時に device 上で変換し、少しの計算と引き換えにメモリを節約する使い方を挙げる。【論文】[TorchRL論文 (2023/11), 付録 D]

環境側と Replay 側のどちらに置くかについて、公式の文書は一般原則を書いておらず、事例だけがある。【公式文書】【公式コード】

- 重すぎてそのまま保存できないデータ（uint8 の画像を float にする、フレームを重ねる）は、Replay 側に置く。collector の `postproc` で変換後のキーを捨てて生の画像だけを保存し、Replay 側で同じ変換をかけ直す。静的なデータでは `"next"` を自動で扱わないので、`("next", key)` を明示する。[rl@v0.15.0 (2026/11), `rb_tutorial.py` L610–716]
- n-step は、collector の `postproc` の `MultiStep` だとバッチの末尾が早期の打ち切りで偏るので、Replay 側の `MultiStepTransform` を勧める。[rl@v0.15.0 (2026/11), `torchrl/data/postprocs/postprocs.py` L99–105・`rb_transforms.py` L25–42]
- 並列 env では、各プロセスの中で変換してもメインプロセスで変換しても結果は同じとされる。ただし ActionMask は並列 env 自体に付ける。[rl@v0.15.0 (2026/11), `tutorials/sphinx-tutorials/torchrl_envs.py` L651–683]、[TorchRL文書 (2026/11), `envs_transforms.rst` L410–412]
- `envs_transforms.rst` の「Using a Transform with a Replay Buffer」の節は本文 1 行だけで、例が無い。[TorchRL文書 (2026/11), `envs_transforms.rst` L197–200]

### 3.6 観測へ情報を足す Transform

観測やその周りへ値を足す Transform は次のとおり。【公式コード】【公式文書】【公式リリース】

| Transform | 足すもの | 出典 |
|---|---|---|
| TensorDictPrimer | reset 時に spec に従った初期値（再帰方策の隠れ状態など）。env の外で `forward` しても値を入れる | [rl@v0.15.0 (2026/11), `_env.py` L211–250] |
| InitTracker | reset の step で `is_init=True`、それ以外で False | [rl@v0.15.0 (2026/11), `_env.py` L1764–1785] |
| StepCounter | reset からの step 数。`max_steps` で truncated を立て、既定で done も更新する | [rl@v0.15.0 (2026/11), `_env.py` L830–868] |
| ActionMask | action spec のマスクを更新し、既定の方策が使える行動を知るのに使う。自作の方策では MaskedCategorical に `"action_mask"` を渡す | [TorchRL文書 (2026/11), `envs_transforms.rst` L368–412] |
| LastAction | 時刻 t の action を `"next"` の `last_action` に書く。reset 時は既定値（0、nan、スカラー、テンソル）で埋め、spec は action と同じ形と dtype の Unbounded で登録する | [rl@v0.15.0 (2026/11), `_action.py` L2691–2748] |

LastAction は v0.15.0（PR #4360）で加わった。v0.14 以前に同じことをする Transform があったかは確かめていない。env のコンストラクタに方策を渡すと、再帰の module に要る InitTracker と TensorDictPrimer を自動で付ける仕組みもある。【公式リリース】【公式文書】[TorchRL v0.15.0リリース (2026/11), Features]、[TorchRL文書 (2026/11), `envs_api.rst` L67–123]

### 3.7 Atari の DQN 参照実装の組

公式の DQN の Atari 実装は、`GymEnv(..., frame_skip=4, from_pixels=True, categorical_action_encoding=True)` に次の順で Transform を重ねる。フレームの間引きは Transform ではなく GymEnv の引数で行う。【公式コード】[rl@v0.15.0 (2026/11), `sota-implementations/dqn/utils_atari.py` L36–59]

1. NoopResetEnv(noops=30, random=True)
2. EndOfLifeTransform（学習用だけ）
3. SignTransform（報酬の符号化。学習用だけ）
4. ToTensorImage → GrayScale → Resize(84, 84) → CatFrames(N=4)
5. RewardSum → StepCounter(max_steps=4500) → DoubleToFloat → VecNorm(画素)

EndOfLifeTransform は ALE の残機の減少を `"end-of-life"` に書くが、done の spec には登録しない（env を reset させない）。損失の側で done・terminated の代わりに使うのが典型で、DQN 実装も損失に `set_keys(done="end-of-life", terminated="end-of-life")` を設定している。【公式コード】[rl@v0.15.0 (2026/11), `gym_transforms.py` L20–33・L91–101・`dqn_atari.py` L64–96]

## 4. Collector と ReplayBuffer と損失

### 4.1 Collector

collector の名前は v0.11.0 で変わり、`Collector`・`AsyncCollector`・`MultiSyncCollector`・`MultiAsyncCollector` になった。旧名（`SyncDataCollector`、`MultiaSyncDataCollector` など）の別名は v0.13.0 で削除された。【公式リリース・PR】[pytorch/rl commit 8779f2a3 (2025/12)]、[rl@v0.12.0 (2026/04), `torchrl/collectors/__init__.py`]、[pytorch/rl PR #3805 (2026/06)]

`Collector(num_collectors=N, sync=True)` は複数プロセスの結果をまとめて渡し、`sync=False` は最初に集まった worker の結果を渡す。非同期では方策の重みが遅れるため off-policy に向く、と文書は書く。分散用に DistributedCollector、RPCCollector、RayCollector などがある。【公式文書】[TorchRL文書 (2026/11), `collectors_basics.rst` L17–36・`collectors_distributed.rst` L42–118]

`frames_per_batch` は 1 回に返す要素数、`total_frames` は全期間のフレーム数で、割り切れないと例外になる。worker 側の方策の重みは、`update_policy_weights_()` か `update_at_each_batch=True` で更新する。重みの同期は v0.11.0 で `torchrl.weight_update`（全体の流れ、転送、形式の 3 層）に作り直された。【公式コード】【公式文書】[rl@v0.15.0 (2026/11), `torchrl/collectors/_single.py` L361–368]、[TorchRL文書 (2026/11), `collectors_basics.rst` L30–35・`collectors_weightsync.rst` L26–65]、[TorchRL v0.11.0リリース (2026/01)]

### 4.2 ReplayBuffer

ReplayBuffer は storage・sampler・writer・collate_fn・transform・prefetch を差し替える構成で、既定は `ListStorage(1000)`・`RandomSampler`・`RoundRobinWriter` である。【公式コード】[rl@v0.15.0 (2026/11), `replay_buffers/base.py` L184–226]

| 部品 | 主な選択肢 |
|---|---|
| storage | ListStorage（何でも入るが遅い）、LazyTensorStorage、LazyMemmapStorage（分散で共有するなら強く推奨） |
| sampler | RandomSampler、PrioritizedSampler、SliceSampler（終端キーか軌跡 ID で部分軌跡を切り出す）、SamplerWithoutReplacement、PrioritizedSliceSampler、StalenessAwareSampler |
| writer | RoundRobinWriter、TensorDictMaxValueWriter |

【公式文書】【公式コード】[TorchRL文書 (2026/11), `data_replaybuffers.rst` L391–400・`data_storage.rst`・`data_samplers.rst`]、[rl@v0.15.0 (2026/11), `samplers/slice.py` L28–68]

PER は、C++ 拡張 `torchrl._torchrl` の Sum・Min segment tree（Fp32・Fp64）を使う。C++ 側は pybind11 で Python へ公開している。v0.13.0 から CUDA 版のカーネルを含む wheel が任意で配られ、標準の PyPI の wheel は CPU 版のままである。【公式コード】【公式リリース】[rl@v0.15.0 (2026/11), `samplers/prioritized.py` L19–121・`torchrl/csrc/segment_tree.h`・`pybind.cpp` L21–33]、[TorchRL v0.13.0リリース (2026/06)]

n-step は ReplayBuffer に `MultiStepTransform(n_steps, gamma)` を付けて作る（v0.4.0 で追加）。`prefetch` はマルチスレッドで先読みするバッチ数で、既定は無し。抽出ごとに batch size を渡す使い方や、消費型の抽出とは両立しない。【公式リリース】【公式コード】[TorchRL v0.4.0リリース (2024/04)]、[rl@v0.15.0 (2026/11), `rb_transforms.py` L25–95・`replay_buffers/base.py` L214–238]

### 4.3 損失と target の更新

LossModule は学習するパラメータを持つ部品で、入出力とも TensorDict、損失は `"loss_<名前>"` のキーで返す。【公式文書】[TorchRL文書 (2026/11), `objectives.rst`]

DQNLoss の主な既定は `loss_function="l2"`、`delay_value=True`（target ネットを持つ）、`double_dqn=False` で、既定の value estimator は TD0 である。DistributionalDQNLoss は離散の支持点上の分布を出すネットに使う。value estimator には TD0、TD1、TDLambda、GAE、VTrace、MultiAgentGAE がある。【公式コード】[rl@v0.15.0 (2026/11), `torchrl/objectives/dqn.py` L35–63・L166–193・L376–400・`objectives/utils.py` L53–69]

target の更新は `SoftUpdate`（Polyak 平均）と `HardUpdate`（`step()` を N 回呼ぶごとにコピー）である。v0.15.0 のリリースノートは、HardUpdate が N+1 回ごとにコピーしていた誤りを直したと書いている。【公式コード】【公式リリース】[rl@v0.15.0 (2026/11), `objectives/utils.py` L589–679]、[TorchRL v0.15.0リリース (2026/11), Training correctness]

### 4.4 Trainer と版の変化

Trainer は、外側で収集し内側で最適化する再利用できる学習ループで、フックを差し込める箇所が 18 ある。PPO、SAC、DQN、DDPG、IQL、CQL、TD3 などの手法別の Trainer と Hydra の設定がある（v0.10〜v0.12 で追加）。非推奨という記述は無い。README は、参照実装を読める出発点と位置づけ、Trainer の抽象が合わない研究コードには写して使えると書く。【公式文書】【公式リリース】[TorchRL文書 (2026/11), `trainers.rst`・`trainers_basics.rst`]、[rl@v0.15.0 (2026/11), `README.md` L535–537]

最新版は v0.15.0（2026-11-10）で、PyTorch 2.11 以上と tensordict 0.15 系を要する。主な破壊的変更は次のとおり。【公式リリース】[TorchRL v0.7.0〜v0.15.0リリース]

| 版 | 主な破壊的変更 |
|---|---|
| v0.6.0（2024-10） | spec の改名（旧名は別名で残す）。Gymnasium 1.0 以降を支えない判断 |
| v0.7.0（2025-02） | env の spec を既定でロック。MLP の既定の深さを 3 から 0 へ。log-prob のキー名 |
| v0.11.0（2026-01） | v0.10 の非推奨をすべてエラーに。旧 spec 名の削除。collector の再編と改名 |
| v0.12.0（2026-04） | Collector から `local_init_rb` を削除 |
| v0.13.0（2026-06） | collector の旧名の削除。NoisyLinear の既定の変更 |
| v0.15.0（2026-11） | PyTorch 2.11 以上。`reset`・`rollout` の既定を `set_state=False` に |

## 5. 収録している手法

### 5.1 参照実装の置き場と一覧

完全な学習スクリプトは `sota-implementations/`、機能ごとの小さな例は `examples/`、チュートリアルは `tutorials/sphinx-tutorials/` にある。README は参照実装を、ブラックボックスのベンチマークではなく読める出発点と位置づけている。【公式文書】[rl@40407917 (2026/11), `README.md` L51・L526–536]

`sota-implementations/` の主な手法と既定の環境は次のとおり（「検証」は `sota-check/run_*.sh` があるもの）。【公式コード】[rl@40407917 (2026/11), 各 config の環境名と `sota-check/`]

| 置き場 | 手法 | 既定の環境 | 検証 |
|---|---|---|---|
| `dqn/` | DQN | ALE/Pong-v5、CartPole-v1 | ○ |
| `a2c/`、`ppo/` | A2C、PPO | ALE/Pong-v5、HalfCheetah-v4 | ○ |
| `impala/` | IMPALA | ALE/Pong-v5 | ○（単一ノード） |
| `rnd/` | PPO＋RND | HalfCheetah-v4 | ○ |
| `ddpg/` `td3/` `sac/` `redq/` `crossq/` `tqc/` | 連続行動の off-policy | HalfCheetah-v4 | redq 以外は ○ |
| `discrete_sac/` | Discrete SAC | CartPole-v1 | ○ |
| `cql/` `iql/` `td3_bc/` `decision_transformer/` ほか | offline・模倣 | Hopper-v5、HalfCheetah-v4 など | 一部 ○ |
| `dreamer/` `dreamer_v3/` `tdmpc2_trainer/` ほか | モデルベース | DMC、Isaac Lab など | dreamer v1 以外は ○ |
| `multiagent/` | IQL、MADDPG、MAPPO、IPPO、QMIX、VDN、MASAC | VMAS | ○ |
| `grpo/` `expert-iteration/` ほか | LLM・VLA の後学習 | GSM8K、LIBERO など | 一部 ○ |

DQN の Atari 実装の学習設定は次のとおり。【公式コード】[rl@40407917 (2026/11), `dqn/config_atari.yaml`・`dqn_atari.py` L39–134・L249–256]

- 環境は ALE/Pong-v5（gymnasium）。スクリプトは `repeat_action_probability` を指定していない。ALE の文書では v5 の既定は 0.25 である。【第三者（ALE 文書）】[ALE文書 (2026/11), Environments の表]
- 総量 40,000,100 フレーム。ε は 1.0 → 0.01 を 4M フレームで下げ、最初の 200k フレームはランダム行動。
- バッファ 1M、batch 32、Adam lr 2.5e-4、target のハード更新は 10k ごと、収集 1 回あたりの更新は 100 回。
- バッファは一様抽出、損失は二乗誤差で、Double DQN は使わない。スクリプトの中に seed の設定は無い。
- 評価は 1M フレームごとに 3 エピソード。

### 5.2 DQN 系の対応状況

DQN 系の部品は一部しか揃っておらず、Atari の DQN 系の参照実装は素の DQN だけである。【公式コード】【公式 Issue】

| 手法 | 部品 | 参照実装 | 出典 |
|---|---|---|---|
| Double DQN | 有り。`DQNLoss(double_dqn=True)`（既定 False） | DQN の参照実装は使わない | [rl@40407917 (2026/11), `objectives/dqn.py` L47–48・L179–193] |
| Dueling | 有り。`DuelingCnnDQNet`、`DuelingMlpDQNet` | チュートリアル `coding_dqn.py` だけ（CartPole） | [rl@40407917 (2026/11), `modules/models/models.py` L847・L964] |
| 分布型（C51） | 有り。`DistributionalDQNLoss` ほか | Atari には無い。`bandits/` に選択肢がある | [rl@40407917 (2026/11), `objectives/dqn.py` L376・`bandits/dqn.py` L109] |
| QR-DQN、IQN | 無し | 無し | [rl@40407917 (2026/11)] |
| Rainbow | 組み合わせとしては無し | 無し。Issue #2269 が未解決 | [pytorch/rl Issue #2269 (2024/07)] |
| NoisyNet | 有り。`NoisyLinear`（既定 `std_init=0.5`） | DQN には無い。REDQ が選択肢として使う | [rl@40407917 (2026/11), `modules/models/exploration.py` L29–327] |
| PER | 有り。`PrioritizedReplayBuffer` ほか。DQNLoss は既定で `td_error` を書く | DQN は一様抽出。SAC・TD3 などは選択肢で使える | [rl@40407917 (2026/11), `data/__init__.py` L60–91] |
| n-step | 有り。`MultiStep`、`MultiStepTransform` | DQN には無い。チュートリアルと REDQ が使う | [rl@40407917 (2026/11), `redq/utils.py` L998] |

C51 と Rainbow を参照実装に入れる予定を問う Issue #2269（2024-07）に、保守者は、部品は有るが公開できる品質のスクリプトを作ったことが無い、と答えている。同じ Issue で DistributionalDQNLoss の形の誤りが報告され、その修正案は取り込まれずに閉じられた。2026-10 には、MultiStep が `("next","terminated")` をずらさない不具合と、MultiStep と DistributionalDQNLoss の組み合わせの形の不一致が報告され（Issue #4536）、修正の PR #4537 は 2026-11-20 時点で取り込まれていない。【公式 Issue・PR】[pytorch/rl Issue #2269 (2024/07)]、[pytorch/rl Issue #4536 (2026/10)]、[pytorch/rl PR #4537 (2026/10)]

NoisyLinear は、自作の Rainbow で評価時の成績が悪いという報告（Issue #3069、2025-07）を受け、2026-01 の修正でノイズの有無を exploration type で決めるように変わり、`std_init` の既定も 0.1 から 0.5 へ見直された。【公式 Issue・コード】[pytorch/rl Issue #3069 (2025/07)]、[pytorch/rl commit 870ee20a (2026/01)]

### 5.3 探索と内的報酬

内的報酬の部品は RND だけで、v0.14.0 で入った。参照実装は MuJoCo の PPO＋RND だけである。【公式コード】【公式 PR】[rl@40407917 (2026/11), `objectives/rnd.py` L19–60・`envs/transforms/rnd.py` L80]、[TorchRL v0.14.0リリース (2026/09)]

| 手法 | 部品 | 参照実装 |
|---|---|---|
| RND | `RNDTransform`（収集時に `intrinsic_reward` を書く）、`RNDLoss`（predictor の学習。既定 `update_fraction=0.25`）、`RunningMeanStd` | `rnd/rnd_mujoco.py`（PPO＋RND、HalfCheetah-v4、1M フレーム）。Atari には無い |
| ICM、NGU、回数ベース | 無し | 無し |

RND の参照実装は、内的報酬に係数（既定 1.0）を掛けて外的報酬に足してから GAE を計算し、critic は 1 本である。参照実装を足した PR #3905（2026-06）が検証に挙げるのは、コンパイル・import・lint の通過だけで、学習結果の数字は無い。ICM、NGU、回数ベースを題材にした Issue・PR・Discussion は、キーワード検索の範囲では見つからなかった。【公式コード・PR】[rl@40407917 (2026/11), `rnd/rnd_mujoco.py` L92–94・L299–305]、[pytorch/rl PR #3889 (2026/06)]、[pytorch/rl PR #3905 (2026/06), Notes]

## 6. 公開している実験結果

### 6.1 論文の結果と条件

論文の Table 1 は、オンラインの単一エージェントの最終報酬を 5 seed の平均±標準偏差で示し、元論文の結果とだけ比べている。Atari は A2C・PPO が 40M フレーム、IMPALA が 200M フレーム、MuJoCo（v3）は 1M ステップである。【論文】[TorchRL論文 (2023/11), §4・Table 1]

| 手法 | Pong | Freeway | Boxing | Breakout |
|---|---|---|---|---|
| A2C | 20.57±0.65 | 30.58±1.40 | 88.76±6.45 | 375.65±47.34 |
| PPO | 20.52±0.58 | 33.31±0.95 | 98.31±0.93 | 335.71±46.72 |
| IMPALA | 20.54±0.18 | 0.0±0.0 | 99.19±0.54 | 525.57±105.47 |

| 手法 | HalfCheetah | Hopper | Walker2D | Ant |
|---|---|---|---|---|
| A2C | 836±964 | 493±192 | 381±109 | 54±20 |
| PPO | 2770±821 | 1703±742 | 3336±928 | 2469±606 |
| DDPG | 10433±357 | 914±144 | 1683±761 | 1169±658 |
| TD3 | 10285±837 | 2809±618 | 4409±256 | 5373±165 |
| SAC | 11077±323 | 2963±644 | 4561±92 | 3467±654 |

Atari の条件のうち、frameskip 4 と DeepMind の前処理に従うことは付録に書かれている。一方、環境の版、sticky actions の有無、評価の手順（テストのエピソード数、「最終報酬」の定義）は本文と付録で確かめられなかった。DQN は実験の節に無く、付録の損失の使用例にだけ出る。【論文】[TorchRL論文 (2023/11), 付録 G.1・付録 E]

offline（Table 2）は IQL、DT、oDT を D4RL で 50,000 回更新した結果で、seed の数（5 と 3）とデータセット名（medium-replay と medium）が表の説明と付録の図で食い違う。マルチエージェントは VMAS の 3 課題で、RLlib との比較は図と文だけで数値の表は無い。【論文】[TorchRL論文 (2023/11), Table 2・Figure 2・Figure 13・Figure 15]

### 6.2 参照実装の README と継続検証

参照実装の README に成績の数字は無い。`dqn/`、`ppo/`、`a2c/`、`impala/` の README は「Reproducing ... Results」と題するが、表も数字も無く、`rnd/` には README が無い。`sota-implementations/README.md` の Results の節（SAC・REDQ の図）は Markdown のコメントで隠されている。【公式コード】[rl@40407917 (2026/11), `sota-implementations/*/README.md`]

数字や合否の条件を書くのは DQN 系以外の 2 つだけである。DreamerV3 の README は、DMC walker-walk で 3 seed の最終窓の中央値が 900 以上なら合格とする。FQL の README は、参照実装との比べ方（seed 0〜2、1M 回更新）を書く。【公式コード】[rl@40407917 (2026/11), `dreamer_v3/README.md` L37–44・`fql/README.md` L81–110]

`sota-check/` は、各リリースの前に全スクリプトを既定の設定で実行して W&B に記録する仕組みと説明されているが、その公開レポートは見つからなかった。CI はごく短い実行だけで、たとえば DQN の Atari は 48 フレーム、RND は 40 フレームである。【公式コード】[rl@40407917 (2026/11), `sota-check/README.md`・`.github/unittest/linux_sota/scripts/test_sota.py` L178–233]

### 6.3 ほかの実装と比べられる形か

学習曲線やスコアの数値ファイルは、リポジトリに無い（`.csv`、`.json`、`.npz` などを探して 0 件）。論文の Table 1・2 の最終値が、数値で残る唯一の結果である。【公式コード】【論文】[rl@40407917 (2026/11)]、[TorchRL論文 (2023/11), Table 1・2]

第三者からも、生データを含まないライブラリとして扱われている。Open RL Benchmark の論文は、要約や図だけを示し生データを含まないライブラリの 1 つに TorchRL を挙げる。ICSE 2025 の Hundal らは、TorchRL を DQN を持つライブラリとして表に載せるが、DQN の比較実験の 5 実装（RLlib、SB3、Dopamine、TF-Agents、自作）には入れていない。【第三者（論文）】[Open RL Benchmark論文 (2024/02), §4.2]、[Hundal ら (2025/10), Table I・Figure 2]

### 6.4 ほかの公開実装の数字の条件と置き場

TorchRL に無い比較元（Atari の DQN 系と内的報酬）について、ほかの公開実装が数字を出しているかを、条件と置き場だけ並べる。TorchRL とは別の層である（取得はいずれも 2026-11-20）。【公式文書・コード（各実装）】【論文】

| 実装・手法 | 環境 | sticky actions | 学習量 | seed | 数字の置き場 |
|---|---|---|---|---|---|
| CleanRL `dqn_atari.py` | Breakout、Pong、BeamRider（NoFrameskip-v4） | 無し（この ID の既定は 0.00） | 10M ステップ＝40M フレーム。学習中のリターン | 3 | [CleanRL文書 (2026/11), DQN]、W&B の公開レポート |
| CleanRL `c51_atari.py` | 同上 | 無し | 確認できなかった | 3 | [CleanRL文書 (2026/11), C51] |
| CleanRL `rainbow_atari.py` | Alien ほか 6 ゲーム（NoFrameskip-v4） | 無し | 既定 10M ステップ | 確認できなかった | [CleanRL文書 (2026/11), Rainbow] |
| CleanRL `ppo_rnd_envpool.py` | MontezumaRevenge-v5（EnvPool） | 0.25 | 既定 2,000M ステップ。表は 7100（1 seed）と Burda らの 8152（3 seed） | 1 | [CleanRL文書 (2026/11), PPO-RND]、[CleanRL@fe8d8a03 (2026/04), `ppo_rnd_envpool.py` L41–43] |
| Dopamine（DQN、C51、Rainbow、IQN、Quantile） | Atari の各ゲーム | 0.25（gin の既定） | 200 反復×250,000 エージェントステップ（frame skip 4）。評価は 125,000 ステップ、ε 0.001 | JSON に反復ごと 5 値 | [Dopamine@1cba1f9a (2026/11), `baselines/atari/data`] |
| RLeXplore（PPO＋8 種の内的報酬） | Gravitar、Montezuma、PrivateEye、Seaquest、Venture | 0.25 | 25M ステップ（Table 14）、100M フレーム（Table 15）。最後の 100 エピソード | 3（Table 14） | [RLeXplore論文 (2025/04), 付録 D.5] |

Dopamine の JSON の値が学習時と評価時のどちらのリターンかは確かめられなかった（plots のページは「training runs」とだけ書く）。Taïga らの Rainbow 上の内的報酬の比較は、既存の [内的報酬の全体像のサーベイ](intrinsic_reward_landscape_survey_2026-11-18.md#61-taïga-らの-rainbow-上の比較) が扱っている。

## 7. 実時間の性能

### 7.1 論文の速度比較

論文の Table 5 は、データの収集速度で TorchRL の collector がほかのライブラリを上回ると示す。条件は AWS の単一ノード（CPU 96 コア、Run ごとに A100 1 枚）、worker 32 で、単位は本文で「data collection speed」とだけ書かれる。【論文】[TorchRL論文 (2023/11), §4・Table 5]

| 環境 | Tianshou | RLlib | Gymnasium | TorchRL parallel env | TorchRL sync collector | TorchRL async collector |
|---|---:|---:|---:|---:|---:|---:|
| Breakout-v5 | 1,212 | 97 | 9,289 | 9,092 | 10,394 | 19,401 |
| HalfCheetah-v4 | 4,719 | 1,868 | 24,388 | 21,742 | 23,527 | 32,894 |
| Pendulum-v1 | 5,823 | 1,845 | 25,910 | 23,363 | 24,530 | 31,584 |

ほかの速度の表は次のとおり。ReplayBuffer の書き込み・抽出の速度をほかのライブラリと比べた表は無い。【論文】[TorchRL論文 (2023/11), Table 3・Table 4・付録 D・§4]

- **VMAS（Table 3）**: env 1 つでは TorchRL 127 fps、RLlib 93 fps。env 20,000 では TorchRL 538,930 fps、RLlib 1,581 fps（RLlib の最大は env 3,334 での 1,885 fps）。ハードウェアの記載は無い。
- **GAE（Table 4）**: 1000 step の軌跡 1000 本で、Tianshou 5.15 ms、SB3 14.1 ms、RLlib 9.38 ms、CleanRL 1.43 ms（JAX、GPU）、TorchRL compiled 2.67 ms、TorchRL vec 1.33 ms（GPU）。numpy 系の CPU と GPU の間の移し替えは計測に含まない。
- **分散 ReplayBuffer**: 単純な実装より転送が 3〜10 倍速い。条件の記載は無い。
- **学習時間**: 「modern desktop GPU」で、MuJoCo の on-policy が約 1 時間、off-policy が 5〜7 時間、Atari が約 8 時間。

### 7.2 compile と CUDA graph

公式は、torch.compile と CUDA graph で学習全体が 1〜4 倍速くなると書く。v0.6.0（2024-10）で、collector の方策への compile と CUDA graph、グラフ切れの無い損失の compile が入り、compile した損失は eager の 2 倍程度速いとされた。v0.7.0（2025-02）から参照実装はすべて `compile=1` を受け付け、その README は CPU で eager と比べた倍率を載せていた（DQN 1.9 倍、PPO 2.42 倍、SAC 2.62 倍、IQL 3.22 倍など）。【公式リリース】【公式文書】[TorchRL v0.6.0リリース (2024/10)]、[TorchRL v0.7.0リリース (2025/02), Compile compatibility]、[rl@v0.7.0 README (2025/02), アルゴリズムの表]

いまの参照実装の設定では、compile も CUDA graph も既定で切ってある。DQN の Atari 実装は CUDA graph を入れると `CudaGraphModule(update, warmup=50)` を使い、実験的で結果が黙って誤りうるという警告を出す。tensordict の `CudaGraphModule` は、入力の形と batch size が捕捉時と同じであること、データに依る制御フローを使わないことを求める。【公式コード】【公式文書】[rl@40407917 (2026/11), `dqn/config_atari.yaml`・`dqn_atari.py` L157–162]、[tensordict文書 (2026/11), CudaGraphModule]

公式側の派生の LeanRL（CleanRL を fork して torch.compile と CudaGraphModule を足したもの）は、H100 で CleanRL より 2.7〜6.8 倍速いと示す。リポジトリはアーカイブ済み（最終 push 2025-08-22）。【公式コード（派生）】[LeanRL README (2026/11)]

| アルゴリズム | CleanRL（fps） | compile＋CUDA graph（fps） | 倍率 |
|---|---:|---:|---:|
| PPO Atari（Breakout-v5、EnvPool） | 1,022 | 6,809 | 6.8 倍 |
| PPO 連続行動 | 652 | 1,774 | 2.7 倍 |
| SAC | 127 | 725 | 5.7 倍 |
| TD3 | 272 | 936 | 3.4 倍 |

DreamerV3（v0.14.0）では、GB200 で 1 更新の中央値が、本家 JAX 22.04 ms に対し、TorchRL の compiled scan だけでは 358.51 ms（16.3 倍遅い）、CUDA graph を足すと 17.83 ms（1.24 倍速い）である。replay の抽出と環境の収集は除く。【公式リリース】[TorchRL v0.14.0リリース (2026/09), Measured performance]

### 7.3 第三者の速度の報告

CPU で単純な環境を回すと、TorchRL は gym のベクトル化より遅いという報告が 2023 年に 2 件ある。【第三者（Issue）】

- **Issue #1325（2023-06）**: オクラホマ大の研究員が、PongNoFrameskip-v4・並列 8〜16・ランダム方策で、TorchRL 約 2,600 FPS、Gym AsyncVectorEnv 約 11,000 FPS と報告した。EPFL の博士課程の利用者の追試でも 2,866 対 11,588 だった。[pytorch/rl Issue #1325 (2023/06)]
- **Issue #1247（2023-06）**: 同じ EPFL の利用者が、CartPole・CPU で、単体の env と SerialEnv は gymnasium より最大 50 倍、ParallelEnv は最大 10 倍遅いと報告した。保守者の修正で、8 worker の並列版は gym の 2 倍の遅さまで縮んだ。[pytorch/rl Issue #1247 (2023/06)]

SB3・CleanRL・RLlib・EnvPool と同じ条件で Atari の速度を比べた第三者の研究は見つからなかった。PufferLib 2.0 は TorchRL を関連ライブラリとして挙げるだけ、RLtools は star 数を表に載せるだけである。【第三者（論文）】[PufferLib 2.0論文 (2025), §1]、[RLtools論文 (2024/02), Table 7]

### 7.4 TensorDict の受け渡しの重さ

TensorDict の費用は、要素数ではなく操作ごと・テンソルごとの Python の仕事に比例する、と tensordict の README は説明する。【公式文書】[tensordict README (2026/11), FAQ]

その費用は数字で測られている。【論文】【公式 PR】

- 論文（v0.2.1 時点）は、100 段の入れ子の辞書で PyTree と比べ、値の加算 0.916 対 1.0555 ms、split 1.015 対 4.254 ms などを示した。[TorchRL論文 (2023/11), 付録 B.4]
- PR #1801（2026-09）は eager の `set`・`update` を速くし、1 キーの `set` が 0.79 → 0.49 µs、256 個のテンソルの `update(td)` が 248.10 → 42.74 µs になった（Apple M5 Max）。[tensordict PR #1801 (2026/09)]
- PR #1916（2026-10）は、入れ子のキーの補助を Python で書き直し、`td.get("x")` が 580 → 432 ns、3 段の入れ子のキーで 1,660 → 1,110 ns になった。[tensordict PR #1916 (2026/10)]

Issue #1247 で保守者は、gym の env から TensorDict を組み立てる費用は常に残る、と答えている。【公式 Issue】[pytorch/rl Issue #1247 (2023/06)]

## 8. C++ から使えるか

TorchRL も tensordict も C++ の API を持たない。C++ は Python の拡張として使われているだけである。【公式コード】【公式文書】

- **TorchRL の C++ 拡張**: `torchrl/csrc` に CPU と CUDA の Sum・Min segment tree（PER 用）と、SafeTanh の autograd 関数がある。`pybind.cpp` がこれを `torchrl._torchrl` として Python へ公開し、`segment_tree.h` 自体が pybind11 と `torch/extension.h` を include している。ビルドは `torch.utils.cpp_extension` で行う。[rl@40407917 (2026/11), `torchrl/csrc/pybind.cpp`・`segment_tree.h` L8–22・`setup.py` L19–42]
- **CUDA 版 PER**: PR #3677（2026-04）で入った。CI の計測では batch 65,536 の平均抽出時間が、優先度 1M 件で CPU 9.99 ms → CUDA 0.45 ms、10M 件で 18.68 → 0.93 ms。ハードウェアの記載は無い。[pytorch/rl PR #3677 (2026/04)]、[TorchRL v0.13.0リリース (2026/06)]
- **tensordict の C++ 拡張は廃止**: PR #1916（2026-10）で `tensordict/csrc` を Python に置き換え、いまは純 Python の wheel である。互換用の `tensordict._C` は 0.17 で消える予定。[tensordict PR #1916 (2026/10)]、[tensordict README (2026/11), Install]
- **libtorch だけの C++ から使う道**: C++ 向けの API・文書・CMake のパッケージは見当たらず、Issue の検索でも、libtorch の話題は import エラーの報告と依存の更新だけだった。公式が示すのは、学習済みの方策を書き出す道（torch.export → AOTInductor の C++ バイナリ、ONNX Runtime、v0.15.0 の `export_onnx_policy`）である。[rl@40407917 (2026/11), `tutorials/sphinx-tutorials/export.py` L329–359]、[TorchRL v0.15.0リリース (2026/11), Highlights]

PER の C++ 拡張は PyTorch の版と合わないと undefined symbol のエラーになる、と README は注意している。【公式文書】[rl@40407917 (2026/11), `README.md` インストールの節]

## 9. 採用状況と方向

GitHub と PyPI の数値（2026-11-20 取得）は次のとおり。【第三者（集計）】[GitHub API (2026/11)]、[pypistats (2026/11)]

| 項目 | pytorch/rl | pytorch/tensordict |
|---|---:|---:|
| star | 3,639 | 1,046 |
| fork | 541 | 157 |
| 貢献者（アカウント） | 234 | 81 |
| 未解決の Issue＋PR | 357 | 231 |
| PyPI の直近 1 か月のダウンロード | 134,549 | 1,870,085 |

リリースは頻繁で、2025-11-20 からの 1 年で TorchRL は 10 本、main へのコミットは約 991 件だった。tensordict は 2026-10-15 から 2026-11-11 の間に v0.15.0〜v0.15.6 の 7 本を出した。【公式リリース】[GitHub API (2026/11)]

保守の中心が動いている。【公式文書】【SNS】【第三者】

- コミット数の最多は vmoens（TorchRL 1,277、tensordict 544）だが、取得時点の最終コミットは TorchRL で 2026-09-19、tensordict で 2026-09-18 だった。[GitHub API (2026/11)]
- tensordict では 2026-08-03 に初コミットした Peter Sharpe（GitHub の自己記載は NVIDIA、tensordict の保守者）が 1,034 コミットで最多である。[GitHub API (2026/11)]
- いまの README が挙げる TorchRL の保守者は vmoens、theap06（UC Berkeley）、bsprenger（Kodiak AI）である。tensordict の README は PyTorch organization が保守すると書く。[rl@40407917 (2026/11), `README.md`]、[tensordict README (2026/11), FAQ]
- Vincent Moens の所属は、論文の時点で PyTorch Team, Meta、2025-10-02 の Bluesky で Periodic Labs に入ったと告知、2026-11 の GitHub の会社欄は「Ineffable」である。最後のものが何の組織かは確かめられなかった。[TorchRL論文 (2023/11), 表紙]、[Bluesky vmoens (2025/10)]

TorchRL を使う派生プロジェクトとして、README は ACEGEN、BenchMARL、BricksRL、OmniDrones、RL4CO、RoboHive を挙げる。tensordict の README は、PhysicsNeMo、MuJoCo Playground、verl などで使われ、2,000 を超えるリポジトリが依存すると書く（README の主張で、個別には確かめていない）。【公式文書】[rl@40407917 (2026/11), `README.md` Ecosystem and publications]、[tensordict README (2026/11)]

2025〜2026 年の版は、LLM の後学習とロボットへ広がっている。【公式リリース】

- v0.9.0（2025-07）: LLM の API（Hugging Face・vLLM のラッパー、GRPO・SFT の損失）。
- v0.10.0（2025-09）: GRPO、PPOTrainer、Isaac Lab のラッパー。
- v0.11.0〜v0.12.0（2026-01〜04）: 重みの同期、設定ベースの Trainer、SGLang。
- v0.14.0（2026-09）: DreamerV3、VLA（LeRobot、OpenVLA、LIBERO）。
- v0.15.0（2026-11）: Hugging Face・LeRobot への書き出し、TD-MPC2、ONNX への書き出し。

[TorchRL v0.9.0〜v0.15.0リリース]

Meta は 2025-10-22 に TorchForge を発表した。TorchRL がそれに統合されるのか問われた Discussion で、vmoens は概念の重なりを認めつつ、TorchRL は開発を続け、両者は補い合うと答えた（2025-10-28）。【公式ブログ】【公式 Discussion】[TorchForgeブログ (2025/10)]、[pytorch/rl Discussion #3224 (2025/10)]

## 10. 批判と懸念

### 10.1 批判

**性能**【第三者（Issue）】【公式リリース】

- CPU で単純な環境を回すと gym より遅い（§7.3 の Issue #1247・#1325）。
- `ReplayBuffer.extend` と `sample` を続けて compile すると eager の約 5 倍遅い、という報告（Issue #2501、2024-10）がある。PR #2529 で試験の範囲では eager と同等以上になったが、Issue は開いたままである。[pytorch/rl Issue #2501 (2024/10)]、[pytorch/rl PR #2529 (2024/11)]
- LossModule・TensorDictSequential・ProbabilisticActor が compile できないという報告（Issue #2312、2024-07）は、応答が無いまま開いている。[pytorch/rl Issue #2312 (2024/07)]
- DreamerV3 は compiled scan だけでは JAX の 16.3 倍遅い（§7.2、公式の数字）。

**抽象化と学び始めの難しさ**【論文】【第三者】

- 論文自身が、明示的なループの設計は、すぐ使える 1 つの形を求める初学者の採用を妨げうると書いている。[TorchRL論文 (2023/11), 付録 F]
- vmoens はポッドキャストで、TensorDict に慣れるには頭の切り替えが要ると認めている。[TalkRL (2024/04)]
- LossModule の重みの更新が元の module に反映されるのか分からないとして、文書化を求める Issue（#3032、2025-07）が未解決である。[pytorch/rl Issue #3032 (2025/07)]
- TensorDict を PyTorch 本体へ入れる提案（PR #112441、2023-10）では、PyTorch のコア開発者が「なぜ別リポジトリでなくコアか」と問い、提案は取り込まれずに開いたままである。[PyTorch PR #112441 (2023/10)]

**API の変化**【公式リリース・文書】【第三者（Issue）】

- 破壊的変更が多い（§4.4 の表）。README は、TorchRL を PyTorch の beta の機能として出しており、破壊的変更は複数のリリースにわたって警告してから入れると書く。tensordict も 1.0 前で、警告を 2 つのマイナー版で出してから変えるとしている。[rl@40407917 (2026/11), `README.md` Status and license]、[tensordict README (2026/11), FAQ]
- v0.6.0 で Gymnasium 1.0 以降を支えないと決めたため、公式チュートリアルが gymnasium 1.0 で動かないという Issue（#2495）が出た。v0.8.0 で Gymnasium 1.1 に対応した。[pytorch/rl Discussion #2483 (2024/10)]、[pytorch/rl Issue #2495 (2024/10)]、[TorchRL v0.8.0リリース (2025/04)]

**インストールと版合わせ**【第三者（Issue）】

- C++ の拡張を import できないという報告が繰り返し出ている（#1888 は 2024-02、#3112 は 2025-08）。[pytorch/rl Issue #1888 (2024/02)]、[pytorch/rl Issue #3112 (2025/08)]

**DQN 系の手薄さ**【公式 Issue・PR】

- 分布型の損失に、形と dtype の不具合の修正 PR が取り込まれないまま残っている（§5.2）。

### 10.2 支持と保守者の応答

支持の声は次のとおり。【第三者】

- LeanRL の Show HN（2024-09）のコメントは、速さと分かりやすさを評価するものが中心だった。上流の CleanRL に戻さない理由を問うコメントが 1 件ある。[HN LeanRL (2024/09)]
- Issue #3032 の起票者は、複数のプロジェクトで使い、おおむね快適だったと書く。Issue #1247 の起票者は、保守者の対応の速さを評価した。[pytorch/rl Issue #3032 (2025/07)]、[pytorch/rl Issue #1247 (2023/06)]
- PR #112441 では、IBM の参加者が、TensorDict は大きな state dict の管理に魅力的だと述べた。[PyTorch PR #112441 (2023/11)]

保守者（vmoens）の応答は次のとおり。【公式 Issue・Discussion】

- 性能の報告に対し、CPU では改善の余地があると認め、非同期の collector や CUDA の利用を勧めた。[pytorch/rl Issue #1247 (2023/06)]、[pytorch/rl Issue #1325 (2023/08)]
- 版合わせの報告に対し、macOS 15 向けにビルドしたのが原因で、次のマイナー版で対象を下げると答えた。[pytorch/rl Issue #3112 (2025/08)]
- Gymnasium との非互換の理由を長文で説明した。[pytorch/rl Discussion #2483 (2024/10)]

## 11. anet との対応（筆者の整理）

この節は、筆者が TorchRL の出典と anet の文書を並べた整理である。TorchRL 側の事実は前の節の出典に、anet 側の事実は括弧内のリポジトリのファイルに拠る。

### 11.1 部品の対応

| TorchRL | anet | 主な違い |
|---|---|---|
| EnvBase、ParallelEnv・SerialEnv | Env、BatchEnv（[120](../../docs/design/120_environments.jp.md)） | TorchRL は終わった要素の reset を `"_reset"` のマスクで指示する。anet の BatchEnv は終端の lane を Step の中で自動で Reset し、Replay へは Reset 前の終端観測を渡す（[150 §2.1](../../docs/design/150_replay_buffer.jp.md)） |
| env 側の Transform | Env の中の前処理（[220](../../docs/design/220_atari_env.jp.md)、ADR 0026） | TorchRL は env の外に重ねる。anet の Atari は Env の中に前処理を持つ |
| 観測を Network の入力にする Transform | 観測パイプライン（[PRD161](../../docs/memo/0161_observation_pipeline_10rfd.md)、暫定）、`DictFrameStacker`、`ObservationNormalizer` | §11.3 |
| Collector | TrainRunner（SerialTrainRunner、PipelineTrainRunner）と Actor（[100](../../docs/design/100_runtime_and_configuration.jp.md)、[110](../../docs/design/110_agents_and_learning.jp.md)） | TorchRL の collector は方策を受け取って rollout し、`frames_per_batch` 単位の TensorDict を返す。anet は Runner が Actor・Env・Learner の呼び出しの順序を持つ |
| ReplayBuffer（storage・sampler・writer・transform） | ReplayBuffer（DefaultReplayBuffer、PrefetchingReplayBuffer。[150](../../docs/design/150_replay_buffer.jp.md)） | TorchRL は部品を差し替える構成。anet は N-step・stack の再構成・PER・prefetch を 1 つの契約にまとめる |
| LossModule、value estimator、target の更新 | `dqn::Learner`（[200](../../docs/design/200_dqn_agents.jp.md)） | TorchRL は損失を独立した部品にする。anet は Learner が損失と更新をまとめて持つ |
| Trainer | RunManager と Runner（[100](../../docs/design/100_runtime_and_configuration.jp.md)） | TorchRL の Trainer はフックで差し込む学習ループ |

### 11.2 TensorDict と TensorSpec の同じ点と違う点

名前と「文字列のキーでテンソルを束ね、まとめて device へ移す」という役割は同じである。違いは、anet の TensorDict が `std::map<std::string, torch::Tensor>` の平らな表であること（`src/core/base/include/anet/tensor_util.hpp`）にある。

| 観点 | TorchRL の TensorDict | anet の TensorDict |
|---|---|---|
| batch の形 | 全エントリーで共有する `batch_size` を持ち、インデックスや reshape が全体にかかる | 持たない。各テンソルの先頭の次元を、使う側が batch として扱う |
| キー | 文字列のタプルで入れ子にできる。step の出力は `"next"` の下に入る | 平らな文字列。観測キーは `vector`・`grid`・`action_mask`（[CONTEXT.md](../../CONTEXT.md)） |
| 補助の仕組み | lazy stack、memmap、TensorDictModule による `in_keys`・`out_keys` の配線 | `To`・`Cpu`・`Cuda`・`Detach`・`Clone` などの一括操作 |

spec も、TorchRL はクラスの階層、anet は 1 つの構造体である。anet の `TensorSpec`（`src/core/base/include/anet/common.hpp`）は、空間の配置（Vector・Grid・Sequence）、batch を除いた形、dtype、クラス数（0 なら連続値）、表示用のラベル、最小値・最大値を持つ。これはおおむね、TorchRL の Bounded・Unbounded（最小・最大）と Categorical（クラス数）を 1 つにまとめた形にあたる。TorchRL の spec の一覧（§2.3）には空間の配置を表す種類が無い。一方、anet の spec には、TorchRL の env の spec のような入力側（action・状態）と出力側（観測・報酬・done）の区分が無い。

### 11.3 Transforms と PRD161 の観測パイプライン

**同じ点**

- データの変換と spec の変換を対で持つ。TorchRL は各 Transform の `transform_observation_spec` で spec を変え、新しいキーを spec に登録しないと KeyError にする（§3.2）。PRD161 は段の出口ごとに spec（フレーム spec、Network の入力 spec）を持ち、検証を構築時に行う（PRD161 §4.1・§4.4）。
- フレームの重ねを、行動選択側と学習側の 2 か所で実現する。TorchRL の CatFrames は、env 側では roll するバッファ、Replay 側では SliceSampler で切り出した N 個の遷移から重ね直す（§3.3）。anet は行動選択側を `DictFrameStacker` の ring buffer、学習側を ReplayBuffer の再構成で行う（PRD161 §4.1、[150 §2.3](../../docs/design/150_replay_buffer.jp.md)）。reset のときに最初のフレームで埋める規則も同じである（TorchRL の `padding="same"` と、`DictFrameStacker` の初期 frame での埋め直し）。
- Replay には重ねる前のフレームを保存し、抽出時に重ねる。TorchRL はメモリの N 倍を避けるためと書き（§3.3）、anet も Push 時の 1 frame と履歴開始を保存して Sample 時に重ねる（[150 §2.3](../../docs/design/150_replay_buffer.jp.md)）。

**違う点**

| 観点 | TorchRL の Transforms | PRD161 の観測パイプライン |
|---|---|---|
| 形 | 同じ型の Transform を Compose で 1 列につなぐ。順序は自由で、同じ部品を env・Replay・モデルに付けられる | 3 つの置き場（フレーム段・スタック段・バッチ段）を固定の順で持つ。段は置き場の規約で、interface の階層にしない（§4.1） |
| フレームの重ねの質の違い | 同じ CatFrames クラスを 2 つの使い方（env 側、`as_inverse` と SliceSampler の Replay 側）で使い、組み立ての補助（`make_rb_transform_and_sampler`）を用意する | 行動選択側のストリーム処理と、Replay の格納境界の処理を、同じ「スタック段」の 2 つの実現として扱う（§4.1） |
| env 側と Replay 側の定義の一致 | 利用者が 2 つの Compose を作り、`("next", key)` を足し、前段の Transform も Replay 側へ渡して揃える（§3.3 の注意 3 つ） | 1 つの定義（stack の定義、normalizer）から各経路が部品を取り出す。PRD161 §1.1 は、定義が経路ごとに書き直されていることを問題にしている |
| 行動側 | `inv` で行動も変換する（ActionDiscretizer など） | 観測だけを扱う。強制行動は別の単位（§4.6） |

TorchRL の形は、自由な 1 列の鎖で、env 側と Replay 側の一致を利用者の組み立てに任せる。PRD161 は、段の数と順序を固定する代わりに、1 つの定義から両側の部品を作る。PRD161 が鎖にしない理由（frame stack だけが質的に違う）に対し、TorchRL は同じクラスに 2 つの使い方を持たせ、補助関数と注意書きでその違いを吸収している。

**正規化の統計の扱い**

TorchRL の VecNorm は env の step の経路で統計を更新し、eval モードと `freeze()` で更新を止め、推論用に `frozen_copy()`・`to_observation_norm()` で固定した写しを作る（§3.4）。VecNormV2 には統計を TensorDict で運ぶモードもある。PRD161 §10 は、評価専用の Run での統計の凍結、統計の写しを network の写しと対で持つこと、checkpoint を持ち越しの論点に挙げている。TorchRL の凍結と写しの作り方は、これらの論点の事例として読める。統計を更新する場所は、TorchRL（env の step）と anet（Experience の受け入れ。PRD161 §1.1）とで違う。

**直前の行動**

TorchRL の LastAction（v0.15.0）は、action を `"next"` の `last_action` へ書き、reset 時は既定値で埋め、spec は action と同じ形と dtype で登録する（§3.6）。直前の行動を観測へ足す PRD142 は、PRD161 のフレーム段の最初の利用者とされている（PRD161 冒頭）。LastAction は action をそのまま写すので、離散の行動の one-hot 化は Transform の外の仕事になる。

### 11.4 実装の参考にできる範囲

ライブラリとしては使えない。C++ の API が無く、C++ は Python の拡張にしか使われていない（§8）。参考になるのは設計と、版を固定したうえでのコードの読み方である。

参考になる部分は次のとおり。

- ReplayBuffer の storage・sampler・writer・transform の分け方、SliceSampler、Replay 側の n-step（`MultiStepTransform`）と、それを collector 側より勧める理由（§3.5、§4.2）。
- PER の segment tree の CPU と CUDA の実装と、その速度（§8）。
- Transform の spec の伝え方と、メソッドを呼ばれる経路で分ける規則（§3.2）。
- 部分 step・部分 reset のマスク、`AsyncEnvPool`、重みの同期の 3 層（§2.4、§4.1）。

参考になりにくい部分もある。

- DQN 系は手薄で、IQN・QR-DQN・Rainbow は無く、分布型の損失には未解決の不具合がある（§5.2）。DQN 系の参照には、IQN を含む Dopamine の方が近い（§6.4）。
- API の変化が速く、1 年で 10 リリース、破壊的変更も多い（§4.4、§9）。コードを読むときは commit を固定する必要がある。

### 11.5 実験結果の比較元にできる範囲

TorchRL は、RND と DQN 系の比較元にはならない。

- **RND**: 参照実装は PPO＋RND の HalfCheetah だけで、学習結果の数字が無い（§5.3）。
- **DQN 系**: Atari の数字が論文にも README にも無い。論文の Atari の数字は A2C・PPO・IMPALA の 4 ゲームの最終値だけで、環境の版・sticky actions・評価の手順が書かれていない（§6.1）。
- **生データ**: 学習曲線の数値ファイルが無い（§6.3）。

比較元の候補は §6.4 の表にある。どれも条件を個別に確かめる必要がある。anet の Atari の既定のプロトコルは `v5_noop0`（sticky 0.25、noop_max 0、episodic_life false。[220](../../docs/design/220_atari_env.jp.md)）である。

| 比べたいもの | 候補 | 条件で揃っている点 | 揃っていない点、確かめる点 |
|---|---|---|---|
| DQN 系（IQN を含む） | Dopamine の baselines（JSON） | sticky 0.25 | noop と残機の扱い。値が学習時か評価時か（§6.4） |
| RND（Montezuma） | CleanRL の PPO-RND、Burda らの値 | sticky 0.25 | 方策が PPO で、anet は IQN の上の RND（[212 §5.3](../../docs/design/212_rnd_lifelong_novelty.jp.md)）。CleanRL は 1 seed |
| 内的報酬の手法の横並び | RLeXplore | sticky 0.25、3 seed | 方策が PPO |
| Rainbow 上の RND | Taïga らの比較 | 既存の[サーベイ §6.1](intrinsic_reward_landscape_survey_2026-11-18.md#61-taïga-らの-rainbow-上の比較)を参照 | anet の `run.@rnd_taiga` も完全な再現ではない（[212 §5.3](../../docs/design/212_rnd_lifelong_novelty.jp.md)） |

## 12. 総合評価

- **設計**: TorchRL は TensorDict でつながる部品の集まりで、Transforms は env・Replay・モデルに共通の 1 列の鎖として作られ、データと spec を一緒に変える（§2、§3）。CatFrames の Replay 側の使い方は、保存は 1 frame、抽出時に重ねる、という PRD161 のスタック段と同じ問題に、同じ答えを出している。違いは、TorchRL が env 側と Replay 側の定義を揃える作業を利用者に任せ、PRD161 が 1 つの定義から両側を作る点にある（§11.3）。
- **実装の参考**: ReplayBuffer の分け方、PER の CUDA 実装、Transform の spec の規則、重みの同期は読む価値がある。C++ からは使えず、DQN 系は手薄で、API の変化も速い（§8、§5.2、§4.4）。
- **実験結果の比較元**: RND も DQN 系も、比較に使える数字を出していない（§5.3、§6）。比較元は Dopamine（DQN 系、IQN を含む）と CleanRL・RLeXplore（RND、PPO の上）で、どれも条件を個別に確かめる必要がある（§6.4、§11.5）。
- **性能**: 論文は、収集の速度でほかのライブラリを上回ると示す。第三者は、CPU の単純な環境では gym より遅いと報告している。compile と CUDA graph の効果は、公式で 1〜4 倍、派生の LeanRL で 2.7〜6.8 倍とされる（§7）。
- **持続性**: 開発は活発で、範囲は LLM の後学習とロボットへ広がっている。中心の保守者の最終コミットは 2026-09 で、tensordict の保守は NVIDIA の保守者が主に担っている（§9）。

## 13. 調査の限界

- 本文はサブエージェントの調べを統合したもので、原典のコードと照合したのは 3 点（§3.3、§3.5、§3.6）だけである。ほかの行番号と数値は照合していない。
- 被引用数は、Semantic Scholar（102）と OpenAlex（122）で違う。片方のサブエージェントは Semantic Scholar の API が回数制限で使えず、OpenAlex で代えた。
- 論文の Atari の条件（環境の版、sticky actions、評価の手順）は見つからなかった。ICLR の OpenReview の査読コメントは読んでいない。
- `sota-check` の W&B の公開レポート、公式のベンチマークのページ（docs.pytorch.org/rl/dev/bench/）の数値、`examples/replay-buffers/catframes-in-buffer.py` の中身は確かめていない。
- Reddit は検索できず、Hacker News に TorchRL 単独のスレッドは見つからなかった。
- SB3・CleanRL・RLlib・EnvPool と同じ条件で比べた第三者の速度の研究、ReplayBuffer の速度をほかのライブラリと比べた数字は見つからなかった。
- ICM・NGU・回数ベース・IQN・QR-DQN の提案は、キーワード検索の範囲で見つからなかった。表記の違う提案は拾えていない可能性がある。
- `Transform.inv` の docstring とコードの食い違い（§3.2）は、どちらが意図かを確かめていない。VecNorm の既定が VecNormV2 へいつ切り替わるかも確かめていない。
- LastAction と同じことをする Transform が v0.14 以前にあったかは確かめていない。
- Dopamine の JSON の値が学習時か評価時か、CleanRL の C51 の学習量と Rainbow の seed の数は確かめられなかった。
- vmoens の GitHub の会社欄の「Ineffable」が何を指すか、LeanRL をアーカイブした理由は確かめられなかった。
- GitHub・PyPI の数値は 2026-11-20 の取得値で、日々変わる。

## 14. 出典一覧

**論文・書誌**

- [TorchRL論文, 2023/11] Bou, A., Bettini, M., Dittert, S., Kumar, V., Sodhani, S., Yang, X., De Fabritiis, G., Moens, V. "TorchRL: A data-driven decision-making library for PyTorch." arXiv:2306.00577v2（ICLR 2024）. https://arxiv.org/abs/2306.00577
- [ICLR 2024, 2024/05] ICLR. "TorchRL: A data-driven decision-making library for PyTorch (Spotlight Poster)." ICLR 2024 Virtual. https://iclr.cc/virtual/2024/poster/18660
- [Semantic Scholar, 2026/11] Semantic Scholar. "TorchRL" citationCount=102（取得 2026-11-20）. Semantic Scholar API. https://api.semanticscholar.org/graph/v1/paper/arXiv:2306.00577
- [OpenAlex, 2026/11] OpenAlex. "W4379390289" cited_by_count=122（取得 2026-11-20）. OpenAlex API. https://api.openalex.org/works/doi:10.48550/arXiv.2306.00577
- [Open RL Benchmark論文, 2024/02] Huang, S. ほか. "Open RL Benchmark: Comprehensive Tracked Experiments for Reinforcement Learning." arXiv:2402.03046. https://arxiv.org/abs/2402.03046
- [Hundal ら, 2025/10] Hundal, R. S., Xiao, Y., Cao, X., Dong, J. S., Rigger, M. "On the Mistaken Assumption of Interchangeable Deep Reinforcement Learning Implementations." ICSE 2025（arXiv:2503.22575v2）. https://arxiv.org/abs/2503.22575
- [RLeXplore論文, 2025/04] Yuan, M., Castanyer, R. C., Li, B., Jin, X., Zeng, W., Berseth, G. "RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning." TMLR 2025（arXiv:2405.19548v2）. https://arxiv.org/abs/2405.19548
- [PufferLib 2.0論文, 2025] Suarez, J. "PufferLib 2.0: Reinforcement Learning at 1M steps/s." RLJ 2025. https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_151.pdf
- [RLtools論文, 2024/02] Eschmann, J., Albani, D., Loianno, G. "RLtools." arXiv:2306.03530v3. https://arxiv.org/abs/2306.03530

**TorchRL のコードと文書**

- [rl@v0.15.0, 2026/11] PyTorch. "pytorch/rl v0.15.0（commit a711ef99）." GitHub. https://github.com/pytorch/rl/tree/a711ef99d29f31ce6bc49ce5da499ddd87b4b1f8
- [rl@40407917, 2026/11] PyTorch. "pytorch/rl main（commit 40407917, 2026-11-19）." GitHub. https://github.com/pytorch/rl/tree/40407917aef3169e5e32c35d90bd58ba6cfb3626
- [rl@v0.12.0, 2026/04] PyTorch. "torchrl/collectors/__init__.py @ v0.12.0." GitHub. https://github.com/pytorch/rl/blob/v0.12.0/torchrl/collectors/__init__.py
- [rl@v0.10.0, 2025/09] PyTorch. "torchrl/data/tensor_specs.py @ v0.10.0." GitHub. https://github.com/pytorch/rl/blob/v0.10.0/torchrl/data/tensor_specs.py
- [rl@v0.7.0 README, 2025/02] PyTorch. "README.md @ v0.7.0." GitHub. https://github.com/pytorch/rl/blob/v0.7.0/README.md
- [TorchRL文書, 2026/11] PyTorch. "TorchRL documentation（stable, 0.15）." docs.pytorch.org. https://docs.pytorch.org/rl/stable/ （rst は rl@v0.15.0 の `docs/source/reference/`）

**TorchRL のリリース**

- [TorchRL v0.15.0リリース, 2026/11] PyTorch. "TorchRL 0.15." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.15.0
- [TorchRL v0.14.0リリース, 2026/09] PyTorch. "TorchRL 0.14." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.14.0
- [TorchRL v0.13.0リリース, 2026/06] PyTorch. "TorchRL v0.13.0." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.13.0
- [TorchRL v0.12.0リリース, 2026/04] PyTorch. "TorchRL v0.12.0." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.12.0
- [TorchRL v0.11.0リリース, 2026/01] PyTorch. "TorchRL v0.11.0." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.11.0
- [TorchRL v0.10.0リリース, 2025/09] PyTorch. "TorchRL 0.10.0." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.10.0
- [TorchRL v0.9.0リリース, 2025/07] PyTorch. "v0.9.0." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.9.0
- [TorchRL v0.8.0リリース, 2025/04] PyTorch. "v0.8.0: Async envs and better weight update API." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.8.0
- [TorchRL v0.7.0リリース, 2025/02] PyTorch. "0.7.0: Compile compatibility, Chess and better multi-head policies." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.7.0
- [TorchRL v0.6.0リリース, 2024/10] PyTorch. "v0.6.0: compiled losses and partial steps." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.6.0
- [TorchRL v0.4.0リリース, 2024/04] PyTorch. "v0.4.0." GitHub Release. https://github.com/pytorch/rl/releases/tag/v0.4.0

**TorchRL の PR・Issue・Discussion・commit**

- [pytorch/rl PR #2368, 2024/08] Moens, V. "[Refactor] Rename specs to simpler names." GitHub. https://github.com/pytorch/rl/pull/2368
- [pytorch/rl PR #2529, 2024/11] kurtamohler. "[Performance] Improve performance of compiled ReplayBuffer." GitHub. https://github.com/pytorch/rl/pull/2529
- [pytorch/rl PR #3677, 2026/04] vmoens. "[Feature] Add CUDA support for prioritized replay sampling." GitHub. https://github.com/pytorch/rl/pull/3677
- [pytorch/rl PR #3805, 2026/06] PyTorch. "[Deprecation] Enact v0.13 deprecations." GitHub. https://github.com/pytorch/rl/pull/3805
- [pytorch/rl PR #3889, 2026/06] theap06. "[Feature] RND Implementation." GitHub. https://github.com/pytorch/rl/pull/3889
- [pytorch/rl PR #3905, 2026/06] theap06. "[Feature] PPO + RND sota-implementation for MuJoCo." GitHub. https://github.com/pytorch/rl/pull/3905
- [pytorch/rl PR #4537, 2026/10] romanbranovets. "[BugFix] Shift terminated in MultiStep and align multi-step discount in DistributionalDQNLoss." GitHub. https://github.com/pytorch/rl/pull/4537
- [pytorch/rl Issue #1247, 2023/06] skandermoalla. "[BUG] Too much overhead in the environment classes?" GitHub. https://github.com/pytorch/rl/issues/1247
- [pytorch/rl Issue #1325, 2023/06] ShaneFlandermeyer. "[BUG] Throughput vs Gym AsyncVectorEnv." GitHub. https://github.com/pytorch/rl/issues/1325
- [pytorch/rl Issue #1888, 2024/02] wbinventor. "Failed to import torchrl C++ binaries." GitHub. https://github.com/pytorch/rl/issues/1888
- [pytorch/rl Issue #2269, 2024/07] roger-creus ほか. "[Feature Request] Support for distributional-DQN algorithms (C51, Rainbow)." GitHub. https://github.com/pytorch/rl/issues/2269
- [pytorch/rl Issue #2312, 2024/07] N00bcak. "LossModule/TensorDictSequential/ProbabilisticActor cannot be compiled." GitHub. https://github.com/pytorch/rl/issues/2312
- [pytorch/rl Issue #2495, 2024/10] maximilienleclei. "Tutorial #4, gym compat." GitHub. https://github.com/pytorch/rl/issues/2495
- [pytorch/rl Issue #2501, 2024/10] kurtamohler. "Support compiling ReplayBuffer.extend/sample without recompile." GitHub. https://github.com/pytorch/rl/issues/2501
- [pytorch/rl Issue #3032, 2025/07] valterschutz. "Documentation about weight update propagation." GitHub. https://github.com/pytorch/rl/issues/3032
- [pytorch/rl Issue #3069, 2025/07] sparisi ほか. "Bad performance with NoisyLinear." GitHub. https://github.com/pytorch/rl/issues/3069
- [pytorch/rl Issue #3112, 2025/08] anirjoshi. "Failed to import torchrl C++ binaries." GitHub. https://github.com/pytorch/rl/issues/3112
- [pytorch/rl Issue #4536, 2026/10] romanbranovets. "[BUG] MultiStep does not shift ("next","terminated")." GitHub. https://github.com/pytorch/rl/issues/4536
- [pytorch/rl Discussion #2483, 2024/10] vmoens. "Incompatibility between TorchRL and Gymnasium 1.0." GitHub. https://github.com/pytorch/rl/discussions/2483
- [pytorch/rl Discussion #3224, 2025/10] xiaotian、vmoens. "TorchForge と TorchRL の関係の質問." GitHub. https://github.com/pytorch/rl/discussions/3224
- [pytorch/rl commit 8779f2a3, 2025/12] Moens, V. "[Refactor] Rename collectors." GitHub. https://github.com/pytorch/rl/commit/8779f2a32b1fd10b0090a74f8bde174bacb3bd91
- [pytorch/rl commit 870ee20a, 2026/01] PyTorch. "[BugFix] Fix NoisyLinear to use exploration_type instead of self.training." GitHub. https://github.com/pytorch/rl/commit/870ee20a6cf5d972088b8cbea80e06bface77fb7

**tensordict**

- [tensordict@v0.15.6, 2026/11] PyTorch. "pytorch/tensordict v0.15.6." GitHub. https://github.com/pytorch/tensordict/tree/v0.15.6
- [tensordict README, 2026/11] PyTorch. "README.md @ 74a8cd2b15." GitHub. https://github.com/pytorch/tensordict/blob/74a8cd2b15/README.md
- [tensordict文書, 2026/11] PyTorch. "torch.compile, torch.func, torch.export, and CUDA graphs." docs.pytorch.org. https://docs.pytorch.org/tensordict/main/compile.html
- [tensordict v0.4.0リリース, 2024/04] PyTorch. "v0.4.0." GitHub Release. https://github.com/pytorch/tensordict/releases/tag/v0.4.0
- [tensordict PR #1801, 2026/09] vmoens. "[Performance] Accelerate eager TensorDict set and update." GitHub. https://github.com/pytorch/tensordict/pull/1801
- [tensordict PR #1916, 2026/10] peterdsharpe. "[Refactor] Write the nested-key helpers in Python and drop the C++ extension." GitHub. https://github.com/pytorch/tensordict/pull/1916

**ほかの公開実装と資料**

- [LeanRL README, 2026/11] meta-pytorch. "LeanRL." GitHub. https://github.com/meta-pytorch/LeanRL
- [HN LeanRL, 2024/09] vmoens ほか. "Show HN: LeanRL." Hacker News. https://news.ycombinator.com/item?id=41597676
- [PyTorch PR #112441, 2023/10] vmoens ほか. "[RFC] Tensordict integration." GitHub. https://github.com/pytorch/pytorch/pull/112441
- [TalkRL, 2024/04] TalkRL. "Vincent Moens on TorchRL (transcript)." TalkRL Podcast. https://www.talkrl.com/episodes/vincent-moens-on-torchrl/transcript
- [Bluesky vmoens, 2025/10] Moens, V. Bluesky の投稿（2024-11-19、2025-10-02）とプロフィール. Bluesky. https://bsky.app/profile/vmoens.bsky.social
- [TorchForgeブログ, 2025/10] PyTorch. "Introducing torchforge." PyTorch Blog. https://pytorch.org/blog/introducing-torchforge/
- [ALE文書, 2026/11] Farama Foundation. "Environments." ALE Documentation. https://ale.farama.org/environments/
- [CleanRL文書, 2026/11] CleanRL. "DQN / C51 / Rainbow / PPO-RND." CleanRL Docs. https://docs.cleanrl.dev/rl-algorithms/dqn/
- [CleanRL@fe8d8a03, 2026/04] vwxyzjn. "cleanrl（commit fe8d8a03）." GitHub. https://github.com/vwxyzjn/cleanrl/tree/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb
- [Dopamine@1cba1f9a, 2026/11] Google. "dopamine baselines/atari（commit 1cba1f9a）." GitHub. https://github.com/google/dopamine/tree/master/baselines/atari
- [GitHub API, 2026/11] GitHub. "REST API（repos、contributors、releases、commits、users）"（取得 2026-11-20）. https://api.github.com/repos/pytorch/rl
- [pypistats, 2026/11] pypistats.org. "torchrl・tensordict recent downloads"（取得 2026-11-20）. https://pypistats.org/api/packages/torchrl/recent
