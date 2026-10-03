# NoisyNet (Noisy Networks for Exploration) PRD

2026-09-30 のグリルと、以下のレビューで合意した仕様と受け入れ条件を記録する。PRD 番号は 084 とし、本書の改訂は実装完了を意味しない。

> 2026-10-02 レビュー指摘 A.1: ノイズ込み重みの SN を[独立した暫定 PRD](999_noisynet_effective_weight_spectral_norm_10prd.md)へ切り出した。084 の SN 併用は μ 正規化に限定し、切り出した機能を完了条件から除く。
>
> 2026-10-03 レビュー指摘 A.2: Noisy・AMP を名前付きの NN 実行設定へ集約する。用途別の tag、標準値、休眠、計算部品による局所検証と `online_reuse` の条件を確定した。Learner の現在値にも μ-only を許し、設定・AMP 移行を実装の第 1 段階とする。
>
> 2026-10-03 追加グリル: AMP の集約と移行を 084 から外し、[全 Agent を対象にした別 PRD](999_nn_forward_amp_consolidation_10prd.md)へ移した。NN 実行設定の項目はノイズだけとし、カタログの名前を `nn_runtime`、参照の語を「キー」、用途を Actor・Learner の現在値・target 構築の 3 つに確定した。実行状態の所有（§9）、ε の共有規則（§7.1）、μ-only の現在値での σ の凍結、`Linear.force_fp32`（§8.2）、群ごとの σ 統計（§10.1）、受け入れ条件の判定方法（§12）を追加し、[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) を起票した。
>
> 2026-10-03 改名: カタログの名前を `nn_runtime` から `nn_forward` に変えた。CONTEXT.md が runtime config を Avoid に挙げ、DQN の `RuntimeVars` が可変の内部変数の語だからである。参照キーは `<thing>_key` の前例に合わせて `nn_forward_key` とし、Learner の 2 用途は役割の階層 `learner.current` / `learner.target` の下に置いた。

関連: [IQN](done/001_iqn_10prd.md)、[τ サンプリング](done/044_iqn_tau_stratified_sampling_10prd.md)、[Munchausen RL](done/067_MunchausenRL_10prd.md)、[Head 分離の暫定 PRD](999_nn_head_projection_separation_10prd.md)。

## 1. 目的と問題

Atari を主軸に、学習可能なパラメータ摂動による探索を、既存の ε-greedy・UQE と比較・併用できるようにする。BTR は数式、初期化、利用箇所を確認する参考実装であり、スコア再現を本 PRD の目的にはしない。

現在の DefaultDQN は ε-greedy に加え、IQN の分位点配置や UQE による行動選択を持つ。「IQN は探索しない」という前提は置かない。一方、学習可能なノイズスケール、ノイズの保持期間、環境・遷移間の共有、target へのノイズ適用を比較する契約はまだない。これらを明示して、探索と学習のどちらの変更が結果へ寄与したかを調べられるようにする。

本書では、確認できた現状・外部資料の事実、今回の設計合意、実験で調べる仮説を分ける。スコア改善、σ の減少、長い行動列での探索改善は保証しない。

## 2. 現状と参考資料から確認したこと

- Body の `Linear` は NN module の設定・構築経路を使う。DefaultDQN の Head は別の経路で最終射影の Linear を所有し、Dueling 合成、QR の reshape、IQN の軸変換と分位平均、TensorDict 出力を担う。Body の `HeadFC` 等と Head 内の最終射影は同一ではない。
- `NetworkHead` 自体は重み所有を要求せず、既存の `PassThroughHead` は重みを持たない。最終射影を Head に残す構造は [ADR 0018](../adr/0018-iqn-via-bind-product-dag.md) の意図的な判断である。
- 現行 Network は Head の実行全体を FP32 で保護する。Body の SN と Head の精度契約は [NN 設計](../design/130_neural_networks.jp.md)、DQN の target・診断・保存契約は [DQN 系 Agent 設計](../design/200_dqn_agents.jp.md) を参照する。
- 原論文は、学習可能な μ・σ と抽選する ε によるパラメータ摂動、および factorised Gaussian を定義している。本書はこの演算を採用する。σ を較正された不確実性として扱う契約は導入しない。[NoisyNet (2017/06), §3](https://arxiv.org/pdf/1706.10295)
- BTR の `FactorizedNoisyLinear` は μ の fan-in に基づく一様初期化と、weight・bias とも fan-in を分母に使う σ 初期化を持つ。IQN の全結合経路に利用される。この参照は、本書の Actor・Learner のノイズ管理全体が BTR と同じであることを意味しない。[BTR (2026/09 参照), networks.py](https://github.com/VIPTankz/BTR/blob/main/networks.py)
- BTR は ε を module の buffer に持ち、`choose_action` ごとに online net を、`learn_call` ごとに target net を引き直す。online net は学習時に引き直さない。hard update の `load_state_dict` は ε の buffer も複製するため、置換 step では target の ε が online の ε になる（2026-10-03 に `C:\dev\BTR` の `Agent.py` / `networks.py` で確認）。本書が ε を module に持たせない理由の一つである（§9、[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)）。

## 3. User Stories

- 実験者として、既存の Body Linear と Head 最終射影へ NoisyNet を適用し、TD・QR・IQN、Dueling の有無を同じ機構で比較したい。
- 実験者として、Actor の保持方式（`call` / `count` / `episode`）と共有範囲（`batch` / `sample`）を比較したい。
- 実験者として、Learner の現在値・target の sample / μ-only と、sample 時の `batch` / `sample`（遷移ごとに独立）を比較したい。
- 実験者として、Actor ごとの評価方式を指定し、ε-greedy・UQE との組み合わせを明示したい。
- 実験者として、μ を対象にした SN の併用、`force_fp32` による精度、σ の weight decay を独立に変えて数値・性能・学習結果を調べたい。
- 実験者として、σ の大きさに加えて、同じ状態でノイズだけを変えた Q・行動・反復エントロピーを観測したい。
- 実装者として、同期・保存・診断がノイズ保持や学習 RNG を意図せず変えない、検証可能な責任境界を持ちたい。
- 実装者として、NN の構造設定と駆動方法（NN 実行設定）を分け、Actor・Learner が Noisy の有無を調べずに、名前付きの設定をキーで選んで使えるようにしたい。

## 4. 適用範囲と構造

対象 Agent は DefaultDQN の TD・QR・IQN とし、Dueling の value・advantage 両ストリームを含める。

### 4.1 NoisyNet を適用する計算部品

既存 `Linear` に NoisyNet の設定を追加し、既存 Head の最終射影にも対応する設定を設ける。独立した公開ブロック型 `NoisyLinear` を増やすことは採用しない。Body と Head はノイズ合成・射影の演算を共用し、Head は現在の最終射影所有を維持する。

適用する Body Linear は利用側が明示する。Conv 版は本 PRD の対象外とする。τ 埋め込みの `Linear` への適用は検証対象外であり、フレームワークは禁止しない（構成の意味は利用側の責任）。汎用 NN 層は、宣言・shape・参照など自身が所有する契約を検証し、設定された接続がアルゴリズムの意図を表すことは利用側が確認する。

本書では「構造設定で NoisyNet を適用した計算部品」と「その呼び出しで sample / μ-only のどちらを使うか」を区別する。NoisyNet の適用は μ・σ を持つ構造を選ぶことであり、μ-only の実行や σ の値が 0 であることは、その構造を通常の Linear へ変える指定ではない。曖昧な「NoisyNet 有効」を全体の判定条件にせず、各部品が使用する実行設定に基づいて検証する。

[Head 分離の暫定 PRD](999_nn_head_projection_separation_10prd.md) は独立した設計検討であり、NoisyNet の前提条件にしない。NoisyNet の演算共用後にも残る設定・拡張上の問題を評価してから、別途方式と対象範囲を決める。

### 4.2 NN 実行設定と責任境界

名前付きの NN 実行設定カタログ `nn_forward.[name]` に、NN の駆動方法を集約する。構造設定が「どの計算部品を持つか」を定めるのに対し、NN 実行設定は「その NN を今回どの条件で使うか」を定める。084 で持つ項目はノイズ方式・保持方式・共有範囲・N だけである。μ・σ の初期化、精度（§8.2）、optimizer の設定は、それぞれ既存の構築・学習側の責任に残す。AMP の集約は[別 PRD](999_nn_forward_amp_consolidation_10prd.md)で扱い、084 では動かさない。

用途は 3 つで、それぞれが NN 実行設定を参照キー `nn_forward_key` で参照する。参照キーの名前は `<thing>_key` の前例（`dataset_key`、`network_key`、`feature_key`）に揃え、Learner の 2 用途は役割の階層 `learner.current` / `learner.target` の下に置く。

| 用途 | 参照の置き場 |
|---|---|
| Actor（カタログ項目ごと） | `actor.[key].nn_forward_key`。`policy` の外に置く。`use_optimistic_target=true` が `actor.[train].policy` を `target_policy` へ複製するため、`policy` の中には置かない |
| Learner の現在値 | `learner.current.nn_forward_key` |
| target 構築（target 行動選択、target 価値評価、Munchausen の追加 forward） | `learner.target.nn_forward_key`。target 構築全体で 1 つ |

どの NN を使うか（`actor.[key].network`）という選択とは分ける。`target_policy`（Agent 直下にある target 行動選択の ActionPolicy 設定）は別の設定で、084 では置き場を動かさない。target 行動選択の forward も `learner.target.nn_forward_key` に従う。キーは不透明な参照名であり、コードは名前から train / eval 等の用途を推測しない。参照の必須性は Noisy の有無に依存させず、標準の選択は共通設定で与える（§11.1）。

Network は解決済みの実行設定と、呼び出し側の実行状態・呼び出しごとの入力を、各計算部品へ明示的に渡す（§9）。各 `NetworkModule` は自分が必要な項目だけを利用し、固有の解釈・計算・必須項目・制約の検証を担う。Head の Noisy 演算部品にも同じ実行情報を届ける。Head・SN 等が持つ局所的な精度保護は §8.2 に従う。

実行情報は全経路で明示する。μ-only は正当な明示値である。構築時の dummy forward、状態スイープ（`TensorDictFunction`）、policy_churn・replay_fit・plasticity の probe チャネル、および NN 実行設定を持たない Agent（ImageCls・MuZero・Rainbow）の呼び出しは、コード固定の μ-only（実行状態なし）を渡す。NoisyNet の probe だけは診断用 RNG で sample を渡す（§10.2）。sample を要求できるのは `nn_forward` を参照する 3 用途だけであり、対象外 Agent の構造設定に Noisy な `Linear` があっても μ-only で動き、エラーにしない。μ-only の実行は満たされない要求ではないからである（§4.1）。

Actor・Learner・共通実行処理が Noisy の型や有無を調べて、forward の挙動や設定の検証を切り替える構造にしない。Noisy の有無フラグ、機能一覧の登録、適用された実行条件を収集して汎用的に比較する仕組み、独立した実行基盤は追加しない。例外は Network が返す σ エントリの列挙（§8.3、§10）で、使ってよいのは optimizer の parameter 分類と診断だけである。既存 NN の構築・forward・検証の境界を拡張し、固有知識を計算部品へ閉じる。

ε・RNG・保持状態は呼び出し側の実行単位ごとに分離する。同じキーや同じ NN の利用だけでは状態を共有しない。ε の共有は §7.1 の規則に従う。通常・部分・診断の forward にも実行情報を届け、module 内の暗黙の現在値や共有の可変設定で呼び出し間の状態を伝えない。

`train()` / `eval()`、勾配制御、IQN の τ、AMP は今回のカタログへの集約対象に含めない。これらは現在の呼び出し側・アルゴリズムの責任に残し、Noisy の sample / μ-only とは独立に扱う。

## 5. 演算と初期化

### 5.1 factorised Gaussian

入力幅を \(p\)、出力幅を \(q\) とする。層ごとに独立な標準正規ベクトル \(z^{in}\in\mathbb{R}^{p}\)、\(z^{out}\in\mathbb{R}^{q}\) を抽選し、次を作る。

\[
f(z)=\operatorname{sgn}(z)\sqrt{|z|},\qquad
e^{in}=f(z^{in}),\quad e^{out}=f(z^{out})
\]

\[
\varepsilon^w=e^{out}(e^{in})^\top,\qquad
\varepsilon^b=e^{out}
\]

\[
y=(\mu^w+\sigma^w\odot\varepsilon^w)x+
  (\mu^b+\sigma^b\odot\varepsilon^b)
\]

μ・σ は学習パラメータであり、ε は勾配を持たない。bias 無しの Linear には bias の μ・σ を作らない。μ-only は ε を 0 とする計算であり、非線形ネットワークのノイズ期待値 \(\mathbb{E}_{\varepsilon}[Q]\) と同一視しない。

同じ層の同じ共有単位では、入力側・出力側のベクトルを保持・再利用する。`sample`（サンプルごとに独立）では、SN を含まない上式を次の等価な形で計算できる。

\[
y_b=\operatorname{Linear}(x_b,\mu^w,\mu^b)
+e^{out}_b\odot
 \operatorname{Linear}(x_b\odot e^{in}_b,\sigma^w,\sigma^b)
\]

不要な \([B,q,p]\) の重み実体化を避ける設計の基準とする。共有単位は dim 0（batch）のサンプルで、それ以外の軸へは同じベクトルを broadcast する。IQN の \([B,K,D]\) では、同じ状態の K 分位点へ同じベクトルを broadcast することになる。演算共用の API と最適化方法は実装計画で具体化し、参照計算との出力・勾配一致で検証する。

### 5.2 初期化

| 対象 | 合意 |
|---|---|
| μ | Body の既存 `init`、Head の既存 `head_init` に従う |
| σ weight | 全要素を \(\sigma_0/\sqrt{p}\) で初期化 |
| σ bias | bias がある場合、全要素を \(\sigma_0/\sqrt{p}\) で初期化 |
| \(\sigma_0\) | 初期値 0.5。有限な非負値を明示的に指定できる |
| 学習後の σ | 符号付き学習パラメータとして扱い、非負化や自動 clamp を加えない |

参照 profile は μ の一様初期化を明示する。現行の `init.mode=default` / `head_init.mode=default` が保持する torch Linear 初期化を使い、weight・bias を \(U[-1/\sqrt{p},1/\sqrt{p}]\) とする。比較する NoisyNet 無効側にも同じ μ 初期化を指定する。[PyTorch Linear (2.11), Parameters](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Linear.html)（リポジトリの libtorch は 2.12.0。初期化式は版に依らない）

この初期化と σ₀=0.5 では、初期の \(\|\sigma^w\|_F/\|\mu^w\|_F=\sigma_0\sqrt{3}\approx0.87\) となり、p・q に依らない。テストの期待値に使える。

層への NoisyNet 適用を理由に既存の Xavier 等を黙って置き換えない。小さい fan-in を理由に σ₀ を自動補正しない。別の初期値が適切かは実験で判断する。

## 6. Actor のノイズ契約

Actor は NN 実行設定のキー（`actor.[key].nn_forward_key`）を選び、その Actor 専用の実行状態と、呼び出しごとの入力（行動選択の時計の値、`episode_start` の lane マスク）を渡す。ノイズ方式・保持方式・共有範囲は参照先の設定が持ち、固有の解釈は計算部品へ閉じる。`train()` / `eval()` だけで sample / μ-only を切り替えず、forward に実行情報を明示して渡す。標準の学習用 Actor は sample・`call`・`batch` とし、共通設定の既定値で与える（§11.1）。

| 保持方式 | 抽選境界 | 共有範囲 |
|---|---|---|
| `call`（呼び出しごと） | 各行動選択の直前 | `batch` / `sample` を選択 |
| `count`（N 回） | 最初の行動選択前に抽選し、N 回使った後の次の行動選択前に再抽選 | `batch` / `sample` を選択 |
| `episode` | 最初の行動選択前、および各 lane の `episode_start` | `sample` だけ。`batch` との同時指定は設定エラー |

`sample` は dim 0 のサンプルごとに独立な ε で、Actor では env（lane）ごとに当たる。N は `count` で明示必須の正の整数で、Actor の行動選択呼び出し回数を数える。並列環境数を掛けた経験数や Learner update 数ではない。`count` の周期は episode 境界でリセットしない。N=1 は `call` と同じ抽選境界になる。

`episode` は、既存の `episode_start` に従って対象 lane だけを更新する。Atari の `episodic_life=true` では life loss による境界を含み、false では既存の game over / truncation の境界に従う。NoisyNet 専用の game 終了通知は追加しない。[Atari Env 設計](../design/220_atari_env.jp.md)

保持するのは ε であり、μ・σ や方策全体ではない。学習や Train Actor network snapshot の更新で μ・σ が変わっても、保持方式に定めた境界まで ε と抽選時の時計の値を維持する。

評価は Actor ごとにキーを通じて sample / μ-only（`mu_only`）を選択し、標準の評価用 Actor と参照評価 profile は μ-only とする。μ-only では保持・共有・N を必須にしない。sample の評価には同じ保持・共有の選択肢を適用する。configured eval tag（`eval` / `eval_target`）の役割を NoisyNet 側で固定しない。

ε-greedy・UQE との併用を許す。層への NoisyNet 適用や実行設定の選択だけでは、既存の探索設定を変更しない。NoisyNet 単独を比較する profile では、ε-greedy の確率 0、risk-neutral な行動選択など、比較に必要な値を利用側で明示する。

## 7. Learner と target のノイズ契約

### 7.1 更新内の共有と分離

Learner の現在値計算は `learner.current.nn_forward_key` で sample / μ-only を選択でき、標準は sample とする。sample 選択時は更新ごと（`call`）にノイズを抽選し、`batch` / `sample`（遷移ごとに独立）を選択できる。既定は `batch` とする。μ-only では σ を計算グラフへ入れない。σ の勾配は未定義のままで、optimizer は σ を更新せず weight decay も掛けないので、σ は凍結される（`torch::optim::AdamW` と `FusedAdamW` は grad 未定義の parameter を飛ばす）。現在値の方式は target の方式とは独立に選べるが、`online_reuse` は §7.2 の条件に従う。

Learner には `count` や `episode` を追加しない。実際にノイズを使う計算部品が、Learner の時計（`learn_step`）に対しては `call` だけを受け付けることを初期化時に検証する。ノイズ設定を使わない部品での休眠は §11.2 に従う。

`sample` でも、同じ状態の IQN 分位点間では ε を共有する。これはノイズの共有契約であり、IQN の分位単調性を保証するものではない。

ε の共有は次の 1 つの規則で定める。

> ε は「実行状態の箱（用途）× network の層（module 実体）」ごとに保持し、同じ箱・同じ層・同じ時計の値なら共有、それ以外は独立とする。

Learner は 1 回の更新内の forward すべてに同じ `learn_step` を渡す。以下は現在値と target がともに sample のときの帰結である。μ-only を選んだ経路では ε を使用しない。

| 用途 | 規則からの帰結 |
|---|---|
| 学習する現在値の online forward | 現在値の箱 × online net の層。現在値用 ε |
| target network の価値評価 | target の箱 × target net の層。現在値と独立 |
| Double-DQN の online 行動選択 | target の箱 × online net の層。現在値（別の箱）とも target net（別の層）とも独立 |
| Double-DQN 無効時の target 行動選択と価値評価 | 同じ箱・同じ層・同じ `learn_step` なので同じ ε を共有 |
| Munchausen `target` 方式の current / next | 1 回の forward なので同一遷移で共有。`sample` では \([B]\) の抽選を \([2B]\) へ複製する |
| Munchausen `online` 方式の追加 current forward | target の箱 × online net の層。現在値とも target net とも独立 |
| Munchausen `online_reuse` | forward しないので抽選なし。既存契約どおり学習時の現在値出力を detach して使う |

Munchausen の current / next を \([2B,\ldots]\) にまとめる場合も、元の同一遷移の対応を保つ。batch を結合したことを理由に current / next の ε を独立にしてはならない。

### 7.2 target と online_reuse

target 構築は `learner.target.nn_forward_key` で sample / μ-only（`mu_only`）を選択でき、標準は sample とする。μ-only は target network だけの指定ではなく、次状態の価値評価、Double-DQN の選択、Munchausen bonus を含む target 構築全体に適用する。学習する現在値の forward は自身の実行設定に従い、この指定では変更しない。

Munchausen 有効時に `online_reuse` を選ぶ場合は、現在値の出力を bonus が要求するノイズ方式で再利用できるかを、各計算部品が検証する。Body の Noisy Linear と Head の Noisy 最終射影のどちらも対象にする。

| 現在値 → bonus が要求する方式 | 計算部品ごとの判定 |
|---|---|
| sample → sample | 再利用可。現在値の ε をそのまま使う |
| μ-only → μ-only | 再利用可 |
| sample → μ-only / μ-only → sample | Noisy の計算部品では再利用不可とし、初期化時に設定エラーにする |
| ノイズ設定を使わない計算部品 | 方式の差は休眠し、この理由では拒否しない |

ノイズ設定を使う部品がない NN では、方式の差だけを理由に拒否しない。その実現のために Actor・Learner・共通実行処理へ Noisy の有無判定を追加せず、各部品の局所検証に委ねる。σ が偶然 0 であることやキー名の一致・不一致では再利用可否を判定しない。

`online_reuse` は既存契約どおり、現在値の出力を detach し、その出力を作った精度・τ・ε を引き継ぐ。追加 forward・追加抽選・黙った再計算は行わない。現在値と target の分離原則から、この明示された再利用まで禁止するものではない。

この検証はノイズ方式の両立を確認するものであり、任意の実行設定や出力の同値性を判定する仕組みに広げない。既存の `online_reuse` が許容している train / eval・精度の差を一律に禁止しない。`online` / `target` 方式は再利用せず、指定した target のノイズ方式で計算する。

## 8. SN・精度・optimizer

### 8.1 同一 Linear での SN（μ 正規化）

既存の SN 対応 Linear では NoisyNet と SN を併用できる。084 で正規化する対象は μ に限定する。\(\mathcal{S}\) は既存 `spectral` / `spectral_cap` の意味に従う正規化を表す。

\[
W_{\mathrm{forward}}=\mathcal{S}(\mu^w)+\sigma^w\odot\varepsilon^w
\]

bias は SN の対象外とする。ノイズを加える前の μ を正規化するため、ノイズ込み重みのスペクトルノルムを同じ範囲へ抑える保証はない。batch 共有・独立の両方式を扱い、SN 側の μ の計算と各共有単位の摂動を合成する。

永続 u/v は μ を追う。初期化時の warm-start、学習 forward（train mode かつ GradMode 有効）での 1 回の power iteration、毎 forward の μ と u/v による推定値の再計算、FP32・分母を経由する μ の勾配、buffer の同期・保存は既存 SN の契約を維持する。Actor・target・追加診断は永続 u/v を更新しない。[NN 設計](../design/130_neural_networks.jp.md)、[ADR 0032](../adr/0032-spectral-norm-self-impl-buffer-semantics.md)

SN を併用した μ-only は \(\mathcal{S}(\mu^w)\) を使う。同一の μ・u/v であれば、ε の有無や再抽選だけでは SN の分母は変わらない。ただし、既存 SN 自体の学習時と eval 時の u/v 更新条件の違いは維持する。

受け入れ条件の SN の「参照計算」は、同じ μ・初期 u/v・更新条件で既存の SN 推定を適用し、その結果へ指定した摂動を加える計算を指す。厳密な最大特異値で除した結果との一致や、ノイズ込み重みへの厳密な上限は要求しない。

\(\mathcal{S}(\mu^w+\sigma^w\odot\varepsilon^w)\) を使うノイズ込み重みの SN は、[別の暫定 PRD](999_noisynet_effective_weight_spectral_norm_10prd.md)で扱う。ε ごとの推定方法、永続 u/v が追う対象、参照計算と許容誤差、μ-only の意味を裁定する必要があり、084 の実装・検証・完了条件には含めない。正規化対象の切替設定や未対応の選択肢も 084 では追加しない。

本項は既存 SN 対応範囲の併用契約であり、現在 SN を持たない Head に新たな SN 設定を追加する要求ではない。

### 8.2 演算精度

AMP の有無と FP16 / BF16 の選択は、既存の Actor（ActionPolicy 設定）・Learner・target policy の設定に残す。現行では target 行動選択が target policy 側の設定、target 価値評価と Munchausen の追加 forward が Learner 側の設定に従い、`@bf16` では前者が FP32、後者が BF16 になる。084 はこの配置も実効精度も動かさない。NN 実行設定へ集約する案は[別 PRD](999_nn_forward_amp_consolidation_10prd.md)で扱う。

`Linear` block に `force_fp32` を構造設定として追加する。BN / LN の `force_fp32` と同じ棚で、既定は `false`（周囲の AMP を継承）とする。Noisy の有無とは独立であり、層への NoisyNet 適用を理由に精度を黙って変えない。参照 profile では両腕（NoisyNet 適用の有無）の該当層に `force_fp32 = true` を明示し、ON / OFF の比較に精度を混ぜない。

```properties
net.block.[AtariHeadFC512Def] : force_fp32 = true   # 参照 profile。Noisy の有無に依らず両腕で同じ
```

SN 計算と Head は既存の FP32 契約を維持する。Head は最終射影を含めて保護し、低精度で射影した出力を後から FP32 へ cast するだけの実装にしない。

AMP を継承する Noisy Linear は、§5.1 の等価式（μ の Linear と σ の Linear を分ける形）を標準にする。μ+σ⊙ε を 1 本の重みに合成してから低精度へ丸めると、σ が μ の BF16 分解能（相対 0.4%）を下回った段階で摂動が消える。分けた形では σ 側が自分のスケールで丸められる。初期値では σ⊙ε の要素は μ の要素の 0.7 倍ほどで、分解能より 2 桁大きい。低精度での丸めや学習差は検証対象であり、「BF16 では必ずノイズが消える」とは断定しない。`force_fp32` と AMP 継承の出力・勾配・速度・メモリを測定する。

### 8.3 σ の weight decay

σ の weight decay を μ と独立に数値指定でき、既定は 0 とする。設定は `learner` 配下の 1 キーとする。weight・bias の σ を漏れなく対象にする。σ の特定には Network が返す σ エントリの列挙（§4.2 の例外）を使い、parameter 名の規約に依存しない。μ の weight decay と分類は既存契約に従う。

通常 AdamW と FusedAdamW の双方で parameter group を構成し、σ の二重登録・登録漏れを防ぐ。weight decay が探索へ与える影響は比較対象であり、非ゼロなら必ず探索が失われるとは扱わない。

## 9. 所有権・同期・保存・再現性

実行に関わる情報を「不変の設定」「呼び出しごとの入力」「保持する状態」の 3 つに分ける。

| 対象 | 性質 | 所有と永続化 |
|---|---|---|
| NN 実行設定（`nn_forward.[name]`） | 不変の設定 | キーで参照する。解決後も Agent の設定内に保持するが、実行時に名前を引く機構は持たない（§13 のゲート）。設定の共用に ε・RNG・保持状態の共用を含めない |
| μ・σ | 学習 parameter | network が所有する。optimizer、hard / soft copy、snapshot、保存・読込の対象 |
| 入力 Tensor（batch size・device）、`episode_start` の lane マスク、時計の今の値 | 呼び出しごとの入力 | 呼び出し側が forward のたびに作って渡す。保持しない |
| 実行状態の箱 | 保持する状態 | 呼び出し側（Actor は 1 つ、Learner は現在値用と target 用に 1 つずつ）が所有する private Resource。`Network` が作り、呼び出し側は中身を読み書きしない。network の同期・保存対象に含めない |
| 箱の中身: 層ごとの slot（保持中の入力側・出力側ベクトルと、抽選したときの時計の値 `drawn_at`）とノイズ用 RNG | 保持する状態 | 計算部品が forward 中に更新する State。slot は `batch` なら \([p]\)・\([q]\)、`sample` なら \([B,p]\)・\([B,q]\)。RNG は呼び出し側の seed から派生し、抽選のたびに進む |
| 時計 | 保持する状態 | 呼び出し側の State。Learner は `learn_step`、Actor は自分の行動選択回数。境界ごとに 1 進める |
| 用途別 RNG | 保持する状態 | acting、Learner の各役割、診断の状態抽出・ノイズ、IQN τ の用途を分離する |

再抽選の判定は計算部品が「保持方式 × 時計の値 × 自分の slot」の純粋な関数として行う。`call` は時計が `drawn_at` と異なれば全行、`count` は時計と `drawn_at` の差が N 以上なら全行、`episode` は slot が無ければ全行、あれば `episode_start` が真の行だけを引き直す。`episode_start` は箱に残らず、引き直した結果のベクトルだけが残る。複数の Noisy 層は同じ呼び出しごとの入力を見るので足並みが揃う。この所有の形は [ownership_guideline.md](../ownership_guideline.md) に private Resource の例として記す。

module 内の暗黙の「現在の ε」やグローバル切替で呼び出し間の状態を共有しない。Actor snapshot へ ε を紛れ込ませず、同期を再抽選の境界にしない。shared network を複数の Actor が `shared_lock` で同時に forward する構成では、module に ε を持たせると読み手どうしで競合する。

固定した構成・seed・呼び出し順序で再現できること、および診断の購読・有効化が学習や行動選択の数値系列へ干渉しないことを保証範囲とする。並列環境数、他 lane の終了順、prefetch 構成等を変えても lane ごとの乱数列が一致する保証には広げない。

現行 checkpoint は network / optimizer を読み込んで新しい Run を開始する契約である。ε、保持カウンタ、ReplayBuffer、全 RNG を含む完全な Run 再開は本 PRD の対象外とする。

NoisyNet の設定・parameter 名・保存形式を具体化する際はクリーンブレーク方針に従う。現用設定・テスト・ドキュメントを同じ変更で移行し、旧 checkpoint のためだけの alias や自動変換を追加しない。過去の Run artifact は書き換えない。

## 10. 診断

### 10.1 群ごとの σ 統計

weight norm 61〜64 と同じ `feature_key` の依存閉包で feature / readout の 2 群に分け、群ごとに weight と bias を分けて \(\operatorname{mean}(|\sigma|)\) と \(\|\sigma\|_2/\|\mu\|_2\) を出す。ノルムは群内の全要素を並べた L2 ノルム（Frobenius ノルム）とし、分母の μ は 63/64 と同じく SN 併用層では実効重みへ換算する（`spectral` では生の μ のスケールが結果に影響しないため）。層ごとの内訳は [920](920_nn_block_metrics_10prd.md) の記録経路が決まってから扱い、084 では持たない。

既存の weight norm 61〜64 には σ を含めない。`ComputeParameterNormSplit` は σ エントリを除いて集計し、σ の大きさは本項の指標で見る。σ エントリが無い network、分母が 0 等、認識済み指標の値が成立しない場合は `NaN` とする。絶対値やノルムを使うため、学習後の σ の符号を消す処理を parameter 自体へ加える必要はない。

### 10.2 同じ状態でノイズだけを変える probe

ReplayBuffer から、専用 RNG を使って一様・非復元に状態を抽出する。PER の学習 batch 抽選を流用しない。対象は online network で、測定位置は replay_fit と同じく、plasticity / policy churn の probe の後、`UpdateFromSamples` の前（この update が適用される直前）とする。参照 profile の初期値は次のとおりとし、値は設定可能にする。

| 項目 | 初期値 |
|---|---:|
| 状態数 B | 128 |
| ノイズ抽選数 R | 32 |
| IQN の固定分位点数 K | 32 |
| 実行間隔 | 503 Learner updates |

1 回の probe では同じ状態集合を使い、μ-only を 1 回、sample を状態ごとに独立な ε 標本 R 個で評価する。R 回の forward で `batch` 共有にしても、R·B 行を 1 回の forward で `sample` にしても期待値は同じなので、まとめ方は実装計画で決める。IQN は \(\tau_k=(k+1/2)/K\)、\(k=0,\ldots,K-1\) の midpoint を固定し、すべての forward で共有する。同じ状態の分位点間では ε も共有する。FP32・eval・NoGrad で測り、通常の学習・Actor forward と独立した診断用 RNG だけを消費する。σ エントリが無い network では三指標とも `NaN` とし、0 を出さない。

probe の反復で変えるのは NoisyNet の ε だけとする。ε-greedy、UQE の risk distortion、τ の再抽選等を混ぜない。その他の確率的処理も eval 条件に従って固定する。Q はネットワークの分位平均後の出力を使い、TBO 有効時も逆変換せず同じ Q 空間で比較する。

状態 \(s_b\) の μ-only 出力を \(Q_0(s_b,a)\)、r 回目の noisy 出力を \(Q_r(s_b,a)\)、行動数を A とする。同じ出力を次の三指標で共用する。

| 指標 | 定義 |
|---|---|
| Q 差 | \(\frac{1}{RBA}\sum_{r,b,a}|Q_r(s_b,a)-Q_0(s_b,a)|\) |
| 行動不一致率 | \(\frac{1}{RB}\sum_{r,b}\mathbf{1}[\arg\max_a Q_r(s_b,a)\ne\arg\max_a Q_0(s_b,a)]\) |
| 反復エントロピー | 状態ごとに R 回の greedy 行動頻度 \(p_b(a)\) を求め、\(\frac{1}{B}\sum_b-\sum_a p_b(a)\ln p_b(a)\) |

エントロピーは自然対数を使い、単位は nats、\(0\ln0=0\) とする。argmax の同点処理は全比較で揃える。状態ごとの行動分布を先に作り、そのエントロピーを状態間で平均する。異なる状態の行動を一つにまとめたエントロピーではない。

これらは固定した状態でパラメータノイズが出力・行動をどの程度変えるかの観測であり、較正された不確実性や探索の良さを直接示す指標ではない。μ-only とノイズ平均 Q の違いも維持する。

### 10.3 既存診断との関係と非干渉

- `policy_churn`、`replay_fit`、plasticity の probe チャネルは μ-only とする。ノイズ条件が変わることで既存指標の解釈が変わらないようにする。精度は従来どおりで、plasticity の probe チャネルは Learner と同じ autocast、policy_churn と replay_fit は FP32 のままとする。
- この μ-only は、[ADR 0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md) と PRD 066 が NoisyNet の churn 対応に求めていた「before / after / target で同一ノイズ」の契約にあたる（ε=0）。実装時に [DQN 系 Agent 設計](../design/200_dqn_agents.jp.md) §9.5 の「NoisyNet は現行対象外」を更新する。
- actual 系の loss・feature 等、実際の学習 forward から取得する指標は、そのときに選んだ sample / μ-only の実際の出力を表す。Actor の ActionInfo に由来する指標（Q の margin 等）も同じで、Actor が sample ならその出力を表す。診断の都合で別方式の値へ置き換えない。
- 購読された指標に必要な計算だけを行う。σ 統計だけの購読で反復 forward を実行せず、Q 差・行動不一致率・反復エントロピーは probe を共用する。Actor の毎行動に追加推論を入れない。
- サンプル不足、未実行の測定、無効な機能等で既知の値が成立しないときは `NaN`、未知キーだけ `nullopt` とする。0・過去値・縮小した batch の値に偽装しない。
- 診断によって学習・Actor の RNG、ε の保持、optimizer、SN・BN 等の永続状態を変えない。既存の学習用 ReplayBuffer 抽選状態にも触れない。
- probe の精度設定や eval mode の切り替えを通常実行へ漏らさず、診断の処理時間と追加メモリを別に測定する。

## 11. 設定検証と公開契約

本書で確定するのは NN 実行設定と計算部品の責任境界、各比較軸の意味・既定値・組み合わせである。設定キーの具体的な綴り、forward へ渡す型、metric key は実装計画で既存の Reader・NN・購読契約へ合わせて具体化する。その際に本書で合意した比較軸を削らない。

### 11.1 キーと標準設定

§4.2 の 3 用途では、NN の構成によらず NN 実行設定のキー参照を必須にする。参照の欠落や未知のキーは設定解決時に fail-fast にし、Noisy がない場合だけ参照を省略したり、未指定時にコードで train / eval の既定を推測したりしない。独自に定義する Actor 項目は既存どおり `[eval]` 等から選択チェーンで作られるため、ベースに置く既定葉がそのまま届き、編集の連鎖は起きない。

共通設定のベース定義で、標準のキー選択と値を `?=` により与える。設定プロファイルの選択宣言 `.$`、実験の明示選択、Run・CLI の指定は既存の `=` の運用に従う。標準の Actor の用途は設定上の組み合わせで表し、Actor 名・キー名からコードが意味を推測しない。綴りは例で、最終的な綴りは実装計画で既存の Reader に合わせる。

```properties
DefaultDQNAgent.nn_forward.[act_sample] : noisy.mode  = sample
DefaultDQNAgent.nn_forward.[act_sample] : noisy.hold  = call
DefaultDQNAgent.nn_forward.[act_sample] : noisy.share = batch
DefaultDQNAgent.nn_forward.[mu_only]    : noisy.mode  = mu_only
DefaultDQNAgent.@baseline : actor.[train].nn_forward_key   ?= act_sample
DefaultDQNAgent.actor.@eval_base : nn_forward_key           ?= mu_only
DefaultDQNAgent.@baseline : learner.current.nn_forward_key ?= act_sample
DefaultDQNAgent.@baseline : learner.target.nn_forward_key  ?= act_sample
run.@nz_episode : A2.actor.[train].nn_forward_key = act_episode   # 項目 act_episode を別に定義しておく
```

| 用途 | 標準のノイズ方式 | sample 時の抽選・共有 |
|---|---|---|
| 学習用 Actor | sample | `call`・`batch` |
| 評価用 Actor | μ-only | 抽選しない |
| Learner の現在値 | sample | `call`・`batch` |
| target 構築 | sample | `call`・`batch` |

列挙値は、方式が `sample` / `mu_only`、保持が `call` / `count` / `episode`、共有が `batch` / `sample` とする。語は呼び出し側に依らず、`sample` の共有は Actor では env、Learner では遷移ごとに当たる。標準の選択をコードへ埋め込まず、設定側から方式と許可された保持・共有を選べるようにする。

sample を使用する部品には、必要な方式・保持・共有の実効値を渡す。`count` では N を明示必須にし、別方式のためのダミー値は要求しない。μ-only では保持・共有・N を必須にしない。Learner の sample で受け付ける保持は `call` だけとする。

### 11.2 休眠と検証の境界

キーの参照契約と、各計算部品が要求する固有項目を分ける。ノイズ設定を使わない計算部品は、その設定群が省略されていても実行できる。例えば、ノイズ設定だけの項目を Noisy な層を持たない NN の用途が参照しても休眠し、Noisy の計算部品が利用する場合は、その部品が必要なノイズ設定の不足を初期化時に検出する。

ノイズ設定を使わない計算部品では、その設定群は休眠し、そのための ε 抽選・保持処理を要求しない。NN 実行設定全体を無効扱いにはしない。μ-only の保持・共有等の未使用項目も、その利用を前提とする検証の対象にはしない。

記述された認識済み設定の型・列挙値・値域は、使用・休眠にかかわらず検証する。σ₀、weight decay、N、probe の件数・間隔などに不正値を潜伏させない。正の件数を要求する項目に 0 や負値を渡した際に、全件扱い・自動 clamp・別方式への fallback を加えない。意図的な診断無効化は既存の購読・有効化契約で表す。

宣言だけで判定できる型・値域・参照の違反は設定読込・解決時に検出する。`episode` と `batch` の同時指定は、互換性のない組み合わせとしてここで fail-fast にする。固有項目の不足、実際に利用する保持方式と呼び出し側の時計の種類の不整合、`online_reuse` の両立条件は、それらを使う計算部品が初期化時に検証する。shape や ε と batch / device の不整合は、情報が揃う初期化・forward 境界で fail-fast にする。エラーには対象のキー・NN 実行設定の名前・計算部品・指定値と期待条件を含める。

Actor・Learner・共通実行処理へ Noisy の型判定・有無判定を追加して、固有項目の検証を代行しない。再利用の条件は §7.2 に従い、全実行条件やキー名の同一性の検証へ広げない。

BN の前後、ε-greedy、UQE、他の探索方式との併用を一律に禁止しない。局所的な契約違反と、実験上の有効性・意味の選択を分ける。

## 12. Testing・受け入れ条件

以下は実装時の完了条件であり、今回の文書改訂で実行した検証結果ではない。

| 領域 | 受け入れ条件 |
|---|---|
| 基本演算 | 明示的に合成した重みでの参照計算と、出力・入力勾配・μ/σ 勾配が精度に応じた許容誤差内で一致する。shared / independent、bias 有無、σ₀=0、rank 2 / IQN rank 3 を含む |
| 初期化 | μ が既存初期化設定に従い、σ weight・bias が同じ fan-in 規則で初期化される。層への NoisyNet 適用による μ 初期化の暗黙変更がない |
| NN 実行設定・休眠 | Noisy なし・Body のみ・Head のみ・両方で、キー参照、必要項目の不足、休眠、不正値、再利用条件を検証する。μ-only での保持項目省略、sample 時の保持制約、`episode` と `batch` の拒否、共通設定の標準値を含む |
| 責任境界 | forward の挙動と設定の検証に Noisy の有無による分岐が無いことをレビューする。σ エントリの列挙を使う箇所が optimizer の parameter 分類と診断だけであることも同じレビューで見る。Body / Head の計算部品が固有の解釈・必須項目・制約の検証を担い、通常・部分・診断の forward へ実行情報が届くことを確認する |
| Actor | 初回抽選、`count` の境界、episode をまたぐ周期、対象 lane だけの episode 更新、`batch` / `sample`、snapshot 時の ε 維持を検証する |
| Learner / target | TD・QR・IQN、Dueling、Double-DQN、Munchausen の各経路で §7.1 の規則どおりの共有と独立を検証する。同一遷移の current / next と IQN の τ 間共有、現在値と target の sample / μ-only、μ-only の現在値で σ が更新も decay もされないことを含む |
| online_reuse | §7.2 の全組み合わせ、σ=0 でも方式の不一致を許容しないこと、ノイズ設定を使わない部品ではその差が休眠することを検証する。追加 forward・抽選がなく、現在値の出力・精度・τ・ε を引き継ぎ、既存の train / eval・精度差を一律に拒否しないことを確認する |
| SN・精度 | μ 正規化と `batch` / `sample` の参照計算（§8.1）、bias 非正規化、既存 `spectral` / `spectral_cap` の意味、`force_fp32` の有無、Head と SN の FP32 保護を検証する。同じ μ・u/v で ε だけを変えても SN 分母が変わらないこと、通常 forward と診断がそれぞれ定めた u/v 更新条件に従うこと、Noisy の適用で層の精度が変わらないことを確認する |
| parameter・optimizer | μ/σ が clone、hard / soft copy、snapshot、保存・読込、optimizer に漏れなく含まれる。ε・slot・時計は含まれない。σ エントリの列挙と parameter group が一致し、通常 AdamW / FusedAdamW の独立 decay が効くことを検証する |
| 診断 | 固定状態・固定 τ での Q 差・行動不一致率・既知の行動頻度によるエントロピー、群ごとの σ 統計を検証する。十分な母集団、一様・非復元抽選、購読 gating、σ エントリが無い network での NaN、未知 nullopt、weight norm 61〜64 から σ が除かれることを含む |
| 非干渉・再現性 | 固定構成で、診断の ON / OFF によって行動・学習の RNG 系列、更新結果、ε 保持、永続 buffer が変わらない。同じキー / NN を使う Actor 同士や役割間でも状態を分離し、一方の抽選・保持更新が他方へ干渉しない。ε の共有は §7.1 の規則に限る |
| 無効時の回帰 | NoisyNet を適用しない構成で、実装前後の同 seed Run の metrics checksum が一致する（ADR 0035 の OFF 保証と同じ水準）。対象は Atari の既定 Run プロファイル（`@bf16`）と CPU 実行の LunarLander。Noisy を適用しない `Linear` / Head の parameter 名と checkpoint の中身は変えず、実装前の checkpoint を `auto_load_file` で読めることを確認する |
| Atari 結合 | 再現可能な設定・seed・実行条件を保存し、1 ゲーム・1 seed・短い予算（例: `exp_exit_step` 500k）で学習、Actor 評価、target、診断を結合した Run が完走し、§10 の全キーが値を出す。成績は見ない。episode 保持は `episodic_life` の既存境界と整合する |
| 性能 | 参照 profile から 1 軸ずつ変え、ラウンドロビンで測る（実時間のドリフトがあるため総当たりはしない）。軸は無効時との差、`batch` / `sample`、保持方式、SN なし / μ の `spectral` / μ の `spectral_cap`、`force_fp32` / AMP 継承、診断の有無。条件と数値を残し、未測定の高速化や全体 2 倍等を約束しない |

判定は次の機械的な指標で行う。

- 駆動の切替は設定 1 行（`actor.[x].nn_forward_key = 名前`）で、C++ の変更がゼロである。
- `dqn_based_agent.cpp` と `default_dqn_agent.cpp` に出る `noisy` / `sigma` の語が、optimizer の parameter 分類と診断の関数の中だけにある。Actor・Policy・forward 経路・設定検証には 0 件である。
- 無効時の回帰は同 seed の metrics checksum の一致で判定する。
- smoke Run で §10 の全キーが値を出し、Noisy を適用しない Run では同じキーが NaN になる。
- 実行設定の項目追加が「NN 側の型の field 追加と計算部品の解釈」だけで済み、Agent 側の差分が 0 行である。

スコア改善、σ の一方向の減少、特定ゲームでの優位性は受け入れ条件に含めない。期待した改善が出なかったことも研究結果として残す。

## 13. 実装順と複雑度監査

実装計画は次の順で作る。段階を分けても、本書に残した比較軸と診断は 084 全体の完了条件に含める。ノイズ込み重みの SN は独立した暫定 PRD の対象であり、その裁定・実装を 084 の完了条件にしない。

1. 基本 NoisyNet: 実行情報の経路（forward 引数・箱・時計）、`nn_forward` と 3 用途の参照、既存 Linear / Head への μ・σ と共用演算、保持・共有・target・評価、局所検証と `online_reuse`、μ 正規化 SN、`Linear.force_fp32`、σ decay、同期・保存、無効時回帰。TDD の順序として、経路と箱を先に通し、Noisy を付けない状態で checksum 一致を確認してから μ・σ を足す。
2. 診断: 群ごとの σ 統計、固定 probe、Q 差・行動不一致率・反復エントロピー、既存診断の μ-only 化、非干渉検証。
3. 結合・性能: Atari 結合 Run、ラウンドロビンの性能測定、AMP 継承（等価式の形）の数値比較。

各段階は単独で価値を持ち、どこで止めても無効時は実装前と一致する。各段階で現用 API・設定・テスト・ドキュメントを揃え、旧契約との二重運用を残さない。084 の完了は段階 3 までである。ここでいう段階は将来の実装順であり、今回の文書改訂で実装・検証が済んだことを示すものではない。

| 裁定 | 内容と理由 |
|---|---|
| 維持 | Actor の保持・共有、Learner の方式・共有、target、評価、μ 正規化 SN の併用、`force_fp32`、σ decay。目的がアルゴリズム探求であり、BTR 再現に必要な最小構成へ縮小しない |
| 維持 | 既存 `episode_start` と Atari の episode 定義。独自の境界概念を増やさない |
| 維持 | 反復エントロピーを初回の診断へ含め、初期値を 32 抽選とする。Q 差・行動不一致率と推論を共用する |
| 維持（方針） | `nn_forward` カタログとキー参照。実害ではなく「実行時に名前の一覧を残す」拡張方針で持つ。`@` プロファイルと `.$` でも同じ設定共用は書ける（[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)） |
| 縮小 | 再現性は固定構成に限定し、環境数等を変えたときの lane ごとの乱数列一致まで要求しない |
| 縮小 | σ 統計は feature / readout 群の集約だけ。層ごとの内訳は 920 の記録経路が決まってから |
| 削除 | 小さい fan-in の自動 clamp、BN・他探索方式との一律排他、層への NoisyNet 適用だけでの既存探索設定・精度の変更、target 内の設定を揃える規則（target の設定は 1 つ） |
| 別 PRD | AMP の集約と移行は[全 Agent を対象にした別 PRD](999_nn_forward_amp_consolidation_10prd.md)。NoisyNet の目的に必要でなく、Noisy と束ねると target の設定が 2 つになり、ImageCls・MuZero・Rainbow と plasticity の probe チャネルの精度へ波及するため |
| 別途検討 | Head 分離は[独立した暫定 PRD](999_nn_head_projection_separation_10prd.md)で比較・裁定する |
| 別途検討 | ノイズ込み重みの SN は[独立した暫定 PRD](999_noisynet_effective_weight_spectral_norm_10prd.md)へ切り出す。ε ごとの推定、永続 u/v、参照計算と μ-only の契約が未裁定のため、084 の完了条件から除く |
| ゲート | 実行時に名前で NN 実行設定を引く機構（GUI からの切替など）は、その切替が要件になった時点で作る。それまでカタログは設定解決時にだけ使う |
| 段階化 | 基本 NoisyNet、診断、結合・性能の 3 段階で進め、各段階で対応する検証を行う |
| 完了基準 | 数値契約、状態・保存、診断の非干渉、Atari 結合と性能測定で判断し、学習結果の改善を必須にしない |

2026-10-03 の追加グリルでは、次の六観点で仕組み全体を監査した（10-03 前半の A.2 対応の監査は本表で置き換える）。

| 観点 | 裁定と理由 |
|---|---|
| 全体量 | **維持**: 実行情報の経路、呼び出し側所有の箱、局所検証、σ エントリの列挙、`force_fp32`、Learner の μ-only、集約 σ 統計と probe、既存診断の μ-only 化、μ 正規化 SN、σ decay。いずれも削ると実害（module 保持の競合、σ の decay 巻き込み、精度の混入、効果の帰属不能）が戻る。**維持（方針）**: `nn_forward` カタログ。削っても戻る実害は無く、拡張方針で持つ |
| 要求の必要性 | **実害 pin**: `force_fp32`（`@bf16` で Noisy 適用層だけ精度が変わる）、σ の凍結（AdamW が grad 未定義を飛ばす仕様で、決めないと実装で結果が割れる）、キー参照の必須化（学習用 Actor の sample の書き忘れに気づけない）。**将来需要**: カタログの実行時の名前解決はゲートの後ろ。**保留**: train / eval、勾配制御、τ、AMP の集約は、それらを実行設定として選ぶ具体的な用途が生じてから |
| 前提変更の残り | **置換**: AMP の除外に伴い、target の 2 つの参照と「揃える」規則、AMP の移行表・段階・受け入れ行を削除。所有は不変の設定 / 呼び出しごとの入力 / 保持する状態の 3 分類で §9 を書き換え。ε の共有は箱 × 層 × 時計の単一規則へ。旧「NoisyNet 有効」条件は §7.2 の部品ごとの再利用条件のまま |
| 最小構成との差 | 差分は `nn_forward` カタログだけ。方針により維持し、ゲートを明記する。それ以外は最小解に含まれる |
| 段階の独立性 | **段階化**: 基本 NoisyNet、診断、結合・性能の 3 段階。各段階単独で価値があり、どこで止めても無効時は実装前と一致する。経路先行は段階 1 内の TDD 順序 |
| 成功の測定 | §12 の判定方法: 設定 1 行での駆動切替、grep による Noisy 語の位置、checksum 一致、NaN 契約、拡張時の Agent 側差分 0 行 |

## 14. 実験で確かめる仮説と対象外

### 14.1 研究仮説

- target のノイズが、学習する分位幅や安定性にどう影響するか。分位幅を環境の確率性だけの測定値として解釈しない。
- TD・QR・IQN の loss、Huber の設定と σ の推移に関係があるか。σ が縮まるべき、あるいは単調に縮まるとは仮定しない。
- IQN の K/M、τ 配置、UQE と NoisyNet の組み合わせで、行動選択と学習効率がどう変わるか。従来の最良値をそのまま移植することも、必ず再最適化が必要だと断定することもしない。
- Actor と Learner のノイズ差が、近似初期優先度と実測 TD の関係や PER の挙動へどう影響するか。乖離の拡大を既定の結論にしない。
- Atari 100k を含む予算・ゲーム・保持方式の違いで効果がどう変わるか。効く環境や改善方向は測定で判断する。

### 14.2 対象外

Conv1d / Conv2d 版、τ 埋め込みへの適用、σ を較正された不確実性として使う拡張、完全な Run 再開、ε-greedy / UQE の機構削除、Rainbow・ImageCls・MuZero 全体への展開、Head の構造分離、ノイズ込み重みの SN は対象外とする。

NN 実行設定への集約は Noisy だけとする。AMP の集約は[別 PRD](999_nn_forward_amp_consolidation_10prd.md)で扱う。train / eval、勾配制御、IQN の τ の集約、Learner の `count` / `episode`、実行時に名前で実行設定を引く機構、独立した実行基盤や機能登録・実行条件の汎用比較は今回導入しない。

## 15. 参照

- [NoisyNet, 2017/06] Fortunato et al. “Noisy Networks for Exploration.” arXiv:1706.10295 / ICLR 2018. [論文](https://arxiv.org/abs/1706.10295)
- [BTR, 2026/09 参照] VIPTankz. “BTR.” GitHub. [repository](https://github.com/VIPTankz/BTR)、[networks.py](https://github.com/VIPTankz/BTR/blob/main/networks.py)。参照した実装の事実と、本書で独自に定めた共有・保持・診断契約は区別する。
- [PyTorch Linear, 2.11] PyTorch Contributors. “Linear.” PyTorch documentation. [初期化仕様](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Linear.html)
- 現行仕様: [NN 設計](../design/130_neural_networks.jp.md)、[DQN 系 Agent 設計](../design/200_dqn_agents.jp.md)、[Atari Env 設計](../design/220_atari_env.jp.md)。
- 判断の背景: [ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)（ε の所有とカタログの採用理由）、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)（Actor カタログと、policy カタログを却下した経緯）、[ADR 0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md)（NoisyNet の churn 対応のゲート）、[ADR 0032](../adr/0032-spectral-norm-self-impl-buffer-semantics.md)、[ADR 0018](../adr/0018-iqn-via-bind-product-dag.md)、[Train Actor snapshot](done/036_train_actor_periodic_snapshot_10prd.md)、[SN](done/065_nn_spectral_norm_10prd.md)、[Head FP32 保護](done/033_imagecls_bf16_head_10prd.md)、[Agent 実装の所有権ガイドライン](../ownership_guideline.md)。
- 別途検討: [NN Head の最終射影と出力変換の責務分離](999_nn_head_projection_separation_10prd.md)、[NoisyNet のノイズ込み重みに対する SN](999_noisynet_effective_weight_spectral_norm_10prd.md)、[NN 実行設定への AMP の集約](999_nn_forward_amp_consolidation_10prd.md)、[NN ブロック別メトリクス](920_nn_block_metrics_10prd.md)。
