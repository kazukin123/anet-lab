# PRD193: 重みの初期化に、分散を倍率÷fan_in にする一様分布の方式を足す（Dopamine の IQN・Rainbow と同じ初期値） 暫定 PRD

> 2026-11-21 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §6 の未決表に置いた。§7〜§9 は §6 の暫定の既定を前提に書き、前提にした行の番号を名指す。行番号は起票時点の作業ツリーの位置である。
> 起点: 2026-11-21 の会話。サブ機で Breakout の anet の IQN を Dopamine の IQN の公開値と比べた切り分けを整理したところ、効いたのが初期値だけだった（§5.3）。開発者が、Dopamine と同じ幅の初期化を先に実装し、流れで確かめたいとした。
> 題材: [issue-20261121-533820](../management/issues/issue-20261121-533820.md)（重みの初期化に、分散を倍率÷fan_in にする一様分布の方式が無い）。
> 関連: [PRD027](done/027_weight_init_mode_string_10prd.md)（初期化方式の文字列化）、[ADR 0008](../adr/0008-weight-init-mode-string.md)、[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)（初期化の設定に読み取りの口が無い）、[PRD150](150_residual_alpha_gate_10prd.md)・[PRD169](169_outside_core_extension_layout_10prd.md)（§10）。

## 1. 目的と問題

### 1.1 問題

1. **anet の初期化方式では、Dopamine の IQN・Rainbow の Network の初期値を作れない。** Dopamine は全層の重みを、分散が 0.577/fan_in の一様分布で初期化し、bias を 0 にする（P1・P3）。anet の `default`（torch の既定）は分散が 1/(3·fan_in) で、標準偏差は Dopamine の 0.76 倍になり、bias も一様乱数のまま残る（P4・P6）。`he` は正規分布で fan_out、`xavier` は fan_in と fan_out の両方を使う（P5）。
2. **Breakout で Dopamine の IQN に残る差のうち、どれだけが初期値の幅によるのかを確かめられない。** 切り分けで効いたのは初期値だけだった。he・xavier から torch の既定へ替えると、学習側の it 15〜19 の平均が同じ seed で 59.6 から 89.6 になった。それでも Dopamine の IQN の 119.7〜177.3 には届いていない（§5.3、P7）。

### 1.2 ゴール

- **G1**: 設定で、どの層にも「重みを、分散が倍率÷fan_in の一様分布で初期化し、bias を 0 にする」初期値を指定できる。
- **G2**: Dopamine と同じ初期値で Breakout を流し、torch の既定の Run（ステージ 59）と並べて、残る差に初期値の幅が効くかを確かめる。
- **G3**: 既存の初期化方式と、新しい方式を使わない設定の Run の挙動を変えない。

### 1.3 仮説・比べ方・終わり方

- **仮説**: Breakout で anet の IQN（B 群）と Dopamine の IQN に残る差（it 15〜19 の平均で 89.6 対 119.7〜177.3）の一部は、初期値の幅と bias の初期値の違いにある。
- **比べ方**: サブ機で、ステージ 57 と同じ構成・seed 2・5M exp step に、全層（conv 3 層、Linear 512、τ の射影、出力の Head）を Dopamine と同じ初期値（倍率 1/√3）にした Run を 1 本流す。学習側のゲームの点を 250,000 exp step ごとに平均し、it 15〜19 の平均をステージ 59（89.6）・ステージ 57（59.6）・Dopamine の IQN の 5 本と並べる（未決表の 11）。seed による差の目安は、seed だけが違うステージ 50 とステージ 57 の差 10.7 である（§5.3）。
- **終わり方**: 1 本の結果を次の表に当てる。上から順に当て、最初に当たった行で読む。

| it 15〜19 の平均 | 読み | 次 |
|---|---|---|
| 120 以上 | 初期値（幅と bias）で Dopamine の IQN の帯に入る | seed 1 でも確かめる。Dopamine に寄せた構成（`run.@rainbow_like`）の Montezuma を、この初期値で流し直す候補にする |
| 100 以上 | 差の一部が初期値で縮む | 初期値の確かめを終え、残る差（目標値で行動を選ぶ τ の本数など）の切り分けへ移る |
| 79 以上 | ステージ 59 と同じ帯（89.6 ± 10.7）。幅と bias の違いは効かない | 初期値の確かめを終え、ほかの差へ移る |
| 79 未満 | ステージ 59 より悪い | 幅と bias を分ける 2 本目を流す（未決表の 11） |

1 本だけなので、帯の境目に近い結果（± 10.7 の内）は、読みを決めずに seed を足すかを開発者が決める。

### 1.4 非ゴール（退けた案と、本当の制約か）

- **Atari.txt の既定の初期値（conv は `he`、Linear と Head は `xavier`）を変えること。** 範囲外。効果を確かめてから、Atari の枠で実験の段取りと一緒に決める（P12）。制約ではなく、実験の結果が要る先送りである。
- **既存の `he` の fan_out を fan_in に直すなど、既存の方式の中身を変えること。** 範囲外。クリーンブレークで移すことはできるが、既存の設定の Run の初期値が変わり、G3 に反する。直すなら別の判断にする。
- **初期化の設定の読み取りを 1 か所にまとめること（[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)）。** 範囲外。別の関心（リファクタリング）で、制約ではない。順序は §10。
- **Dopamine とのほかの差（padding、τ を混ぜる位置、目標値で行動を選ぶ τ の本数、打ち切りの扱い）。** 範囲外。別の関心で、制約ではない。padding と τ を混ぜる位置は、すでに効かなかった（§5.3）。
- **fan_out・fan_avg・正規分布の形。** 今回は作らない（未決表の 3）。制約ではなく、Dopamine の形に要らないので、2 つ目の形が要るときに足す（P11）。

### 1.5 機構とゴールの対応

| 機構 | ゴール | 寄与 |
|---|---|---|
| 初期化方式 `variance_scaling`（未決表の 1） | G1 | Dopamine の初期値を設定で作れる |
| 設定キー `init.variance_scale`（未決表の 2） | G1 | 分散の倍率を層ごとに指定できる |
| 14 か所の設定の読み手での読み取り（未決表の 7） | G1・G3 | どの層でも使え、書いたキーが黙って読まれないことが無い |
| 確かめの Run（§9.1） | G2 | 残る差に初期値が効くかを読む |

## 2. 前提

暫定の起票なので、確かめは行っていない（確かめの列は「対象外」）。

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実（コード） | Dopamine（e7d780d）の `RainbowNetwork` と `ImplicitQuantileNetwork` は、conv 3 層・τ の射影（`dense_quantile`）・Linear 512・出力の kernel を `VarianceScaling(scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')` で初期化する。bias の初期化は指定せず、Keras の既定（0）になる | [atari_lib.py](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py) 304〜364 行・367〜444 行 | 対象外 |
| P2 | 事実（コード） | 同じ版の `NatureDQNNetwork` は kernel の初期化を指定せず、Keras の既定（glorot uniform）になる。1/√3 の倍率は IQN と Rainbow の Network だけである | 同 251〜301 行 | 対象外 |
| P3 | 事実（文書） | Keras の `VarianceScaling` は、uniform のとき [−limit, limit]、limit = √(3·scale/n) から引く。mode が fan_in なら n は重みの入力の数である。重みの分散は scale/n になる | [TensorFlow の VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling) | 対象外 |
| P4 | 事実（コード） | anet の `WeightInitializer::Initialize` は、`default` では weight にも bias にも触らない（torch の既定のまま）。ほかの方式は weight を初期化し、bias を 0（`constant` は定数）にする | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 22〜79 行（26 行・68 行） | 対象外 |
| P5 | 事実（コード） | anet の `he` は `kaiming_normal_` の fan_out、`xavier` は `xavier_uniform_` である | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 36 行・42 行 | 対象外 |
| P6 | 事実（文書） | torch の Linear・Conv2d の既定は、weight も bias も U(−√k, √k)、k = 1/fan_in（Conv2d は groups ÷（入力 channel × kernel の要素数））から引く。重みの分散は 1/(3·fan_in) になる | [torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)、[torch.nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | 対象外 |
| P7 | 事実（測定） | Breakout の切り分けで、効いたのは初期値だけだった | §5.3 | 対象外 |
| P8 | 事実（測定） | 10 月のサンプル効率優先（SN 付きの残差 conv）の qbert でも、初期値を he から PyTorch の既定に戻すと序盤の評価が上がった | [atari/2026-10-23_btr-qbert-repro-init](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md) | 対象外 |
| P9 | 過去の判断 | 初期化方式は文字列で指定し、未知の値は即座に止める | [ADR 0008](../adr/0008-weight-init-mode-string.md) | 対象外 |
| P10 | 既存部品の働き | 初期化の設定は、読み手ごとに 1 キーずつ読む。読み手は 14 か所あり、足したキーを読み忘れた所では、そのキーが黙って読まれない | §5.2、[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md) の影響 | 対象外 |
| P11 | 守る線（規約） | 形が 1 つしか無いものに、列挙の値を持つキーを作らない。2 つ目が要るときに足す | [実装 品質確認観点](../quality/impl.md) の「名前」の規約 5 | 対象外 |
| P12 | 過去の判断 | 効果の確かめと Atari の構成の既定は、Atari の枠で実験の段取りと一緒に決める | [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md) §1.3 の終わり方（D19） | 対象外 |

## 3. 背景

サブ機では、Taiga ら（Rainbow＋RND、ICLR 2020）と anet の Montezuma の差を調べている。Dopamine の Q 側に寄せた構成に Taiga らの式の RND を載せた Run は、鍵より先へ進まなかった。そこで、まず Q 側（IQN）が易しいゲームで Dopamine の IQN の公開値と同じように学べるかを、Breakout と Qbert で確かめた。Qbert は Dopamine の帯に入り、Breakout は下回った。Breakout の切り分けの結果は §5.3 に書く。

## 4. 用語

新しい用語は足さない。暫定の起票なので `CONTEXT.md` を変えない（未決表の 13）。

## 5. 現状（2026-11-21 にコードと Run で確認）

### 5.1 初期化方式

- 方式は `default`・`xavier`・`he`・`orthogonal`・`constant`・`trunc_normal` の 6 つで、`ValidateWeightInitConfigMode` が列挙を検査する（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 735〜743 行）。`ValidateWeightInitConfig` は、方式によらず数値の引数の有限性と `trunc_std > 0`・`trunc_a < trunc_b` を検査する（同 746〜763 行）。
- 適用は `WeightInitializer::Initialize` の 1 か所で、Body の Linear・Conv・残差の block・CNBlock、DQN の Head、MuZero の Head が呼ぶ（P4）。
- Atari の Nature 型の block は、conv 3 層が `he`、`AtariLinear512` と `AtariIQNTauProj512` が `xavier` である（[Atari.txt](../../apps/runner/config/Atari.txt) 848・855・862・866・1209 行）。DQN の Head は既定の `xavier`（[nn.hpp](../../src/core/base/include/anet/nn.hpp) 65 行）。

ステージ 57 の構成（Nature 型、padding 0、IQN の cos の基底 64、Breakout の行動 4）で、各方式の重みの標準偏差を P3〜P6 の式から計算すると次のとおりである。Dopamine の列は倍率 1/√3、fan_in は anet の形（flatten 3,136）で計算した。

| 層 | fan_in | fan_out | 今（he・xavier） | torch の既定 | Dopamine と同じ |
|---|---|---|---|---|---|
| conv 1（4→32、8×8） | 256 | 2,048 | 0.0313（he） | 0.0361 | 0.0475 |
| conv 2（32→64、4×4） | 512 | 1,024 | 0.0442（he） | 0.0255 | 0.0336 |
| conv 3（64→64、3×3） | 576 | 576 | 0.0589（he） | 0.0241 | 0.0317 |
| Linear 512 | 3,136 | 512 | 0.0234（xavier） | 0.0103 | 0.0136 |
| τ の射影（64→512） | 64 | 512 | 0.0589（xavier） | 0.0722 | 0.0950 |
| 出力の Head（512→4） | 512 | 4 | 0.0623（xavier） | 0.0255 | 0.0336 |

bias は、今の方式と Dopamine が 0、torch の既定が ±1/√fan_in の一様分布である（P4・P6）。

### 5.2 設定の読み手

初期化の設定（`mode`・`manual_gain`・`nonlinearity`・`constant_val`・`trunc_std`・`trunc_a`・`trunc_b`）を読む所は 14 か所ある（`git grep -n -E "ANET_READ_CONFIG\([a-z_]+, *[a-z_.0-9]*init[a-z0-9_]*\.mode\)" -- src` の全件、14 件）。

| 読み手 | 設定の接頭辞 | 位置 |
|---|---|---|
| ResBlock | `init1`・`init2`・`init_ds` | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 1678〜1700 行 |
| CNBlock | `init_dw`・`init_pw1`・`init_pw2` | 同 1926〜1950 行 |
| Linear | `init` | 同 3036〜3042 行 |
| Conv1d | `init` | 同 3073〜3079 行 |
| Conv2d | `init` | 同 3110〜3116 行 |
| DefaultDQNAgent | `intrinsic.value.head.init`・`head.init`・`skip.head.init` | [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 216〜221 行・317〜322 行・610〜615 行 |
| RainbowAgent | `head.init` | [rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 28〜33 行 |
| RND | `head.init` | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 565〜571 行 |

DefaultDQNAgent は、価値を分解しない構成で `intrinsic.value.head.init.*` に値が書かれていると知らせる（同 286〜293 行の `report_init`）。設定ファイルでは、[agent.txt](../../apps/runner/config/agent.txt) 17〜22 行が `intrinsic.value.head.init.*` の既定を並べ、[nn.txt](../../apps/runner/config/nn.txt) 5〜21 行が方式ごとの書き方をコメントで示す。

### 5.3 Breakout の切り分けの結果（サブ機）

構成は、Dopamine の IQN に寄せたB 群（`run.@rainbow_like` の Q 側から PER を外し、Adam 5e-5・eps 3.125e-4、τ は学習 64／64・行動 32、FP32、4 env、seed は Run 名の末尾）である。値は、学習側のゲームの点（clip の前、全 lane）を 250,000 exp step ごとに平均したもので、it は Dopamine の iteration（it 19 で 5M exp step）に当たる。Dopamine の値は、e7d780d の [baselines/data/breakout.json](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/baselines/data/breakout.json) の 5 本である。

| Run | ステージ | 変えたもの | it 4 | it 9 | it 19 | it 15〜19 の平均 |
|---|---|---|---|---|---|---|
| `run_20261117-223826_bo_dqiqn_s1` | 50 | （B 群、seed 1） | 26.7 | 54.4 | 71.0 | 70.3 |
| `run_20261120-150755_bo_dqiqn_s2` | 57 | （B 群、seed 2） | 25.1 | 54.3 | 53.7 | 59.6 |
| `run_20261120-023904_bo_dqiqn_pad_s1` | 55 | conv の padding を Dopamine と同じ出力の大きさに | 26.8 | 55.6 | 65.5 | 65.0 |
| `run_20261120-171809_bo_dqiqn_pad_s2` | 58 | 55 の seed 2 | 26.7 | 56.1 | 57.0 | 61.2 |
| `run_20261120-050934_bo_dqiqn_pad_tf_s1` | 56 | 55 に、τ を flatten の直後で混ぜる形（τ の射影の活性化も ReLU に） | 17.9 | 40.5 | 66.0 | 63.5 |
| `run_20261120-223324_bo_dqiqn_initdef_s2` | 59 | 57 の全層（conv 3 層、Linear 512、τ の射影、Head）を `init.mode=default` に | 22.0 | 61.2 | 91.1 | 89.6 |
| Dopamine の IQN（5 本） | — | — | 35.9〜45.6 | 82.3〜121.8 | 124.9〜167.5 | 119.7〜177.3 |
| Dopamine の Rainbow（5 本） | — | — | 18.9〜20.4 | 38.2〜45.2 | 63.4〜74.8 | 62.7〜69.4 |

- seed だけが違うステージ 50 とステージ 57 の差は 10.7 である。padding（ステージ 55・58）と τ を混ぜる位置（ステージ 56）は、その幅を超えて変わらなかった。
- ステージ 59 はステージ 57 より 30.0 高く、it 14 に 95.2、it 16 に 96.6 だった。
- Dopamine の json に、学習の phase の値か評価の phase の値かの説明は無い。

## 6. 決めること（未決表）

暫定の起票なので、どの行も決まっていない。「暫定の既定」は AI が置いた既定で、§7〜§9 はこれを前提に書いた。再グリルですべて開き直す。

| # | 事項 | 暫定の既定 | 理由と依存 | ほかの候補 |
|---|---|---|---|---|
| 1 | 方式の名前 | `variance_scaling` | Keras・TensorFlow・JAX の同じ初期化と同じ名前で、Dopamine の設定（P1）と 1 対 1 に読める | `fan_in_uniform`（形をそのまま言う）、`lecun_uniform`（倍率 1 の名前で、倍率を変える方式に合わない） |
| 2 | 倍率の設定キー | `init.variance_scale`（有限で 0 より大きい実数、既定 1.0）。重みの分散は variance_scale ÷ fan_in になる | 方式ごとの値を平らな名前で持つ今の形（`trunc_std` など、§5.2）に揃える。値の意味を名前で言う | `init.vs_scale`（略語）、`init.variance_scaling.scale`（入れ子。今の形と揃わない） |
| 3 | fan と分布の選び方 | fan_in・一様分布だけにし、選ぶキーを作らない | Dopamine の形はこれだけで（P1）、形が 1 つしか無いものに列挙のキーを作らない（P11） | Keras と同じ `init.variance_fan`（`fan_in`｜`fan_out`｜`fan_avg`）と `init.variance_distribution`（`uniform`｜`normal`｜`truncated_normal`）を同時に作る |
| 4 | bias | 0 にする | Dopamine の bias は 0（P1）。`default` 以外の今の方式と同じ扱い（P4） | torch の既定と同じ一様乱数のまま残す |
| 5 | 既定値 1.0 | Keras の既定と同じ 1.0。Dopamine に合わせるときは 0.5773502691896258 を書く | 設定の文法では 1/√3 を式で書けない。1.0 は Keras の既定（P3）で、特定の手法に結び付かない | 既定を置かず、`variance_scaling` のときは必ず書かせる |
| 6 | 検証の範囲 | `variance_scale` は、方式によらず、有限で 0 より大きいことを検査する | `trunc_std` と同じ扱い（§5.1）。方式で検査を分けない | `variance_scaling` のときだけ検査する |
| 7 | 読み手 | 14 か所すべて（§5.2）で `variance_scale` を読み、`report_init` にも足す | 読み忘れた所では黙って読まれない（P10） | 先に [issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md) で読み取りの口を 1 つにし、そこで読む（§10） |
| 8 | 設定ファイル | nn.txt のコメントに書き方の例を足し、agent.txt の `intrinsic.value.head.init.*` の並びに `variance_scale ?= 1` を足す | 今の並べ方（§5.2）に揃える | 例を足さない |
| 9 | Atari.txt と bat | 本 PRD では変えない。確かめの Run は、実装のあとに 13 bat にステージを足して組む | P12。確かめの Run は bat の引数だけで組める（§9.1） | 本 PRD で Atari.txt に Dopamine の初期値のプロファイルを足す |
| 10 | サブ機への届け方 | 開発者が commit と push を行い、サブ機で pull と build をする | サブ機の 13 bat は、サブ機の checkout を build した実行体を写して使う | メイン機の実行体をサブ機へ写す |
| 11 | 確かめの Run の本数 | まず 1 本（ステージ 57 と同じ構成・seed 2、全層を倍率 1/√3 に）。§1.3 の表で「79 未満」なら、幅と bias を分ける 2 本目（倍率 1/3、bias 0＝torch の既定と同じ幅）を流す | ステージ 59 との差は、幅（標準偏差 1.32 倍）と bias（一様乱数か 0 か）の 2 つが重なる。分けるのは、効き方が読めないときだけにする | 最初から 2 本流す |
| 12 | 設計文書と ADR | 設計文書 [130](../design/130_neural_networks.jp.md) の §7.1 の初期化の行（jp・en）に方式と検査を足す。ADR は起こさない | 既定を変えない方式の追加で、戻しやすい | 方式の名前の選び方を ADR にする |
| 13 | 用語 | `CONTEXT.md` を変えない | 初期化方式の名前は設定の値で、用語集の語ではない | — |

## 7. 仕様（暫定の既定を前提）

前提: 未決表の 1〜6。

- `init.mode = variance_scaling` のとき、weight を [−√(3·s/fan_in), √(3·s/fan_in)] の一様分布から引き（s は `init.variance_scale`）、bias を 0 にする。重みの分散は s/fan_in になる。
- fan_in は torch の fan の計算による。Linear は入力の数、Conv は入力 channel × kernel の要素数である（P6 と同じ数え方）。
- 乱数は、`xavier` と同じく、parameter の初期化が使う乱数を消費する。
- 例（Breakout の確かめの Run で bat に渡す引数。Dopamine と同じ倍率）:

```text
net.block.[AtariConv32].init.mode=variance_scaling net.block.[AtariConv32].init.variance_scale=0.5773502691896258
net.block.[AtariConv64].init.mode=variance_scaling net.block.[AtariConv64].init.variance_scale=0.5773502691896258
net.block.[AtariConv64Final].init.mode=variance_scaling net.block.[AtariConv64Final].init.variance_scale=0.5773502691896258
net.block.[AtariLinear512].init.mode=variance_scaling net.block.[AtariLinear512].init.variance_scale=0.5773502691896258
net.block.[AtariIQNTauProj512].init.mode=variance_scaling net.block.[AtariIQNTauProj512].init.variance_scale=0.5773502691896258
DefaultDQNAgent.head.init.mode=variance_scaling DefaultDQNAgent.head.init.variance_scale=0.5773502691896258
```

- プロファイルで書くときの例: `DefaultDQNAgent.@baseline : head.init.mode ?= variance_scaling`、`DefaultDQNAgent.@baseline : head.init.variance_scale ?= 0.5773502691896258`。

## 8. 実装契約（暫定の既定を前提）

### 8.1 設定と失敗の扱い

前提: 未決表の 1・2・5・6。

| 設定キー | 型 | 値の範囲 | 既定 |
|---|---|---|---|
| `<接頭辞>.mode` | 文字列 | `default`・`xavier`・`he`・`orthogonal`・`constant`・`trunc_normal`・`variance_scaling` | 読み手ごとの今の既定のまま |
| `<接頭辞>.variance_scale` | 実数 | 有限で 0 より大きい | 1.0 |

`<接頭辞>` は §5.2 の 14 か所の接頭辞である。判定は次の表で、上から順に当て、最初に当たった行で扱う。

| # | 条件 | 扱い |
|---|---|---|
| 1 | `mode` が上の 7 つのどれでもない | ERROR で止める。文に設定キー、指定した値、受け付ける 7 つの値を入れる |
| 2 | `variance_scale` が有限でない（NaN・無限大） | ERROR で止める。文に設定キーと値を入れる |
| 3 | `variance_scale` が 0 以下 | ERROR で止める。文に設定キーと値を入れる |
| 4 | それ以外 | `mode` に従って初期化する |

行 2・3 は、`trunc_std` と同じく `mode` によらず当てる（未決表の 6）。止める時機は、今の検査と同じく、設定を読んだときか、初期化の前である（§5.1）。

### 8.2 変えないもの

- 既存の 6 つの方式の初期化の結果と、乱数の消費の順（G3）。
- 新しい方式を使わない設定の Run の学習の結果。config dump には、各読み手の `variance_scale` の行（既定 1.0）が増える。

### 8.3 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| WeightInitConfig | 変更 | `variance_scale` を持つ | NN の公開の口（nn） |
| WeightInitializer | 変更 | `variance_scaling` の初期化を行う | NN の Head・初期化（nn_head） |
| ValidateWeightInitConfigMode・ValidateWeightInitConfig（関数） | 変更 | 新しい方式を列挙に入れ、`variance_scale` を検査する | NN の組み立て（nn_impl） |
| ResBlock・CNBlock・Linear・Conv1d・Conv2d の各 Factory の設定 | 変更 | `variance_scale` を読む | NN の標準 module（nn_modules） |
| DefaultDQNAgent の設定 | 変更 | 3 つの Head の初期化の設定で `variance_scale` を読み、`report_init` に足す | DefaultDQNAgent |
| RainbowAgent の設定 | 変更 | `head.init.variance_scale` を読む | RainbowAgent |
| RndNoveltyEstimatorConfig | 変更 | `head.init.variance_scale` を読む | 内的報酬（RND） |

### 8.4 移行対象

経路の検索（2026-11-21、`git grep -i -l -F`、追跡しているファイルの全体）: `trunc_std` 27 件、`expected default, xavier` 1 件。`trunc_std` の 27 件の割り振りは次のとおりである。

- コード（直す、8 件）: [nn.hpp](../../src/core/base/include/anet/nn.hpp)、[nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp)、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp)（列挙の検査と ERROR の文。`expected default, xavier` の 1 件もここ）、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp)、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp)、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp)、[nn_test.cpp](../../src/core/base/src/nn_test.cpp)（初期化のテストを足す）。
- テスト（直す、1 件）: [dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp)（Rainbow の Head の初期化の読み取りと、skip の Head の不正値の断定の並びに、`variance_scale` を足す）。
- 設定（直す、2 件）: [agent.txt](../../apps/runner/config/agent.txt)・[nn.txt](../../apps/runner/config/nn.txt)（未決表の 8）。
- 設計文書（直す、2 件）: [130 jp](../design/130_neural_networks.jp.md) 299 行・[130 en](../design/130_neural_networks.en.md) 300 行（未決表の 12）。
- 検証資材（直さない、1 件）: `src/core/base/testdata/prd121/run_validation.py`。PRD121 の確かめの道具で、当時の既定キーを並べる。今の受入に使わない。
- 課題記録（直さない、1 件）: 初期化の設定の読み取りの口の課題（§10）。読み手の数を当時のまま記録している。
- 過去の PRD・実装計画（直さない、4 件）: `098_temporl_skip_policy_22impl.md`、`150_residual_alpha_gate_10prd.md`、`done/028_cnblock_convnext_10prd.md`・`done/028_cnblock_convnext_20impl.md`。
- 実験記録の config の写し（直さない、8 件）: `docs/experiments/default-dqn/dropmerge/config/` の Run の設定の写し。過去の Run の記録である。

合計は 8＋1＋2＋2＋1＋1＋4＋8＝27 件。

方式の一覧を書く箇所は `trunc_normal` で引いた（2026-11-21、`git grep -i -l -F`、追跡しているファイルの全体、151 件）。割り振りは次のとおりである。

- 直す（8 件）: `src/core/base/` の include と src の 5 件（上のコードの 8 件のうち nn.hpp・nn_head.hpp・nn_impl.cpp・nn_modules.cpp・nn_test.cpp）、設計文書 130 の jp・en の 2 件、`apps/runner/config/nn.txt` の 1 件。
- 直さない（143 件）: `apps/runner/config/DropMerge.txt`・`nn_cnx.txt` の 2 件（`trunc_normal` を使う block の定義で、方式の一覧ではない）、`docs/memo/` の過去の PRD・実装計画 7 件、課題記録 2 件、ADR 0008 の 1 件、`src/core/base/testdata/` の 14 件（PRD072 の固定 commit の golden 6 件と PRD092 の Run の設定の写し 8 件）、`docs/experiments/` の 117 件（過去の Run の設定の写しと実験記録）。

合計は 8＋143＝151 件。互換層（旧い形の別名や分岐）は作らない。過去の Run・実験記録・ADR は書き換えない。

## 9. 成功の測り方と受入

### 9.1 効果（サブ機で確かめる）

§1.3 の仮説・比べ方・終わり方で確かめる。ステージの組み立ては実装のあとに 13 bat で行い（未決表の 9）、次はそのときの材料で、決定ではない。

- ステージ 57 の引数に、§7 の例の 6 か所の引数を足す。Run 名の例は `run_{t}_bo_dqiqn_initvs_s2`。
- 読む指標は、学習側のゲームの点を 250,000 exp step ごとに平均した it 15〜19 の平均と、it 4・9・19 の値である（§5.3 と同じ集計）。

### 9.2 正しさの受入

暫定の既定を前提にした候補で、再グリルで固める。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 既存の方式の同一性 | 新しい方式を使わない設定で、同 seed・決定的モードの短い Run の metrics checksum が、実装前に保存した実行体と一致する。対象は LunarLander の既定と、Atari Breakout のステージ 57 の構成（30,000 exp step） |
| 2 | 初期値の分布 | テストで、`variance_scaling` の Linear（入力 256・出力 512）と Conv2d（入力 4・出力 32・kernel 8×8）について、倍率 1/√3 と 1.0 のそれぞれで、重みがすべて ±√(3·s/fan_in) の内にあり、標本の分散が s/fan_in の ±5% の内にあり、bias がすべて 0 である |
| 3 | 判定 | §8.1 の判定の表の各行が、表のとおりに扱われる。行 1 の ERROR の文に `variance_scaling` を含む 7 つの値が入る。行 2・3 は `variance_scale` に NaN・無限大・0・−1 を書いたときに当たり、ERROR の文に設定キーが入る |
| 4 | 読み手 | §5.2 の 14 か所のそれぞれで、`<接頭辞>.variance_scale` に 0 を書くと、その設定キーを含む ERROR で止まる。数える単位は読み手の接頭辞で、14 件すべて |
| 5 | 設定ファイル | `apps/runner/config/nn.txt` に `#init.mode ?= variance_scaling` と `#init.variance_scale` の行がある。`apps/runner/config/agent.txt` に `intrinsic.value.head.init.variance_scale ?= 1` の行がある |

## 10. ほかの PRD・課題との順序

- [issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)（初期化の設定の読み取りの口。PRD は無い）が先なら、本 PRD は口の 1 か所で `variance_scale` を読めばよい。本 PRD が先なら、14 か所で読み、cdc102 がまとめるときに `variance_scale` も含める。どちらが先でもよい（未決表の 7）。
- [PRD150](150_residual_alpha_gate_10prd.md)（未実装）は ResBlock を触る。ResBlock に新しい初期化の設定を足すなら、`variance_scale` も読む。どちらが先でもよい。
- [PRD169](169_outside_core_extension_layout_10prd.md)（未実装）は `WeightInitializer` の置き場を扱う。中身は変えないので、どちらが先でもよい。

## 11. issue の対応表

| ID | 題 | area | 根拠の位置 | 直す（未決表） | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261121-533820](../management/issues/issue-20261121-533820.md) | 重みの初期化に、分散を倍率÷fan_in にする一様分布の方式が無い | 315_nn, 324_dqn | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 22〜79 行、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 735〜763 行 | 1〜8 | 2・3・4・5 |

## 12. 参照

- Dopamine e7d780d（2019-12-19）: [atari_lib.py](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py)、[baselines/data/breakout.json](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/baselines/data/breakout.json)。
- [TensorFlow の VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling)、[torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)、[torch.nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。
