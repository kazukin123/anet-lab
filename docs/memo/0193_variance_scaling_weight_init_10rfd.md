# PRD193: 重みの初期化に、分散を倍率÷fan_in にする一様分布の方式を足し、初期化の設定の読み取りを 1 つの口にまとめる（Dopamine の IQN・Rainbow と同じ初期値） PRD

> 2026-11-21 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §6 の未決表に置いた。§7〜§9 は §6 の暫定の既定を前提に書き、前提にした行の番号を名指す。行番号は起票時点の作業ツリーの位置である。
> 2026-11-21 改訂: グリルを行って暫定の印を外した。起票時の未決表の 13 行をすべて決め直し、確定事項（§6）へ移した（対応は §6.3）。節の番号は付け直した（起票の行にある節の番号は当時のもの）。行番号は改訂の時点の作業ツリーの位置である。変えた点は次のとおり。(1) G2 を「流せる状態にする」に改め、5M の Run の段取りと読みを Atari の枠へ渡した（D1）。(2) 初期化の設定の読み取りを 1 つの口にまとめる G4 を足し、読み取りの口の課題のうち初期化の分を範囲に入れた（D5・D6）。(3) DQN と Rainbow の Head も `nonlinearity` を読む形にし（D7）、未知の `nonlinearity` を ERROR にした（D8）。(4) 前提の P10 を今のコードに合わせて直し、P13〜P26 を足した。
> 2026-11-21 改訂: 起草のあと、別モデル（Opus）の検討を受けた。開発者に聞く論点は無く、矛盾 1 件と事実の誤り 2 件を直し、理由と前提を補った（§10）。変えた点は次のとおり。(1) `nonlinearity` の列挙の検査を、方式の列挙と同じ時点にした（D8、§8.1）。(2) 読み取りの口を、読んで検査する入口と読むだけの入口の 2 つにした（D5）。(3) 流せることの確認の確かめ先を、入力の写しと Agent の設定の記録に分けた（D19、§9.2 の 9）。(4) 読んだ値が今と同じことの受入を足した（§9.2 の 10・11）。(5) P18・P23 を直し、P27〜P30 を足した。(6) 並んで起票された PRD194 との順序を足した（§11）。
> 2026-11-21 改訂: レビューの指摘を受けて直した。(1) `skip.head.init` は、新しい `variance_scale` も含めて `head.init` の値を既定にすると明記した（D9・D11、§8.1、§9.2 の 6）。(2) 既存の挙動の確かめに、比べる道具・範囲・除くものと、LunarLander の Run の長さを書いた（D21、§9.2 の 1）。(3) block の既定の受入に ResBlock の shortcut の conv を足した（§9.2 の 11）。(4) §8.5 の実験記録の所在と、P8 の内容を直した。
> 起点: 2026-11-21 の会話。サブ機で Breakout の anet の IQN を Dopamine の IQN の公開値と比べた切り分けを整理したところ、効いたのが初期値だけだった（§5.3）。開発者が、Dopamine と同じ幅の初期化を先に実装し、流れで確かめたいとした。
> 題材: [issue-20261121-533820](../management/issues/issue-20261121-533820.md)（重みの初期化に、分散を倍率÷fan_in にする一様分布の方式が無い）。読み取りの口の分は、グリルのあとに起こした [issue-20261121-d4feb6](../management/issues/issue-20261121-d4feb6.md)（初期化の設定を 14 か所が 1 キーずつ読む）が持つ（§12）。
> 関連: [PRD027](done/0027_weight_init_mode_string_10rfd.md)（初期化方式の文字列化）、[ADR 0008](../adr/0008-weight-init-mode-string.md)、[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)（設定の部品に読み取りの口が無い。本 PRD はこのうち初期化の分だけを扱う）、[PRD150](0150_residual_alpha_gate_10rfd.md)・[PRD169](0169_outside_core_extension_layout_10rfd.md)・[PRD183](0183_temporl_skip_followups_10rfd.md)・[PRD194](0194_config_key_full_names_in_messages_10rfd.md)（§11）。

## 1. 目的と問題

### 1.1 問題

1. **anet の初期化方式では、Dopamine の IQN・Rainbow の Network の初期値を作れない。** Dopamine は全層の重みを、分散が 0.577/fan_in の一様分布で初期化し、bias を 0 にする（P1・P3）。anet の `default`（torch の既定）は分散が 1/(3·fan_in) で、標準偏差は Dopamine の 0.76 倍になり、bias も一様乱数のまま残る（P4・P6）。`he` は正規分布で fan_out、`xavier` は fan_in と fan_out の両方を使う（P5）。
2. **Breakout で Dopamine の IQN に残る差のうち、どれだけが初期値の幅によるのかを確かめられない。** 切り分けで効いたのは初期値だけだった。he・xavier から torch の既定へ替えると、学習側の it 15〜19 の平均が同じ seed で 59.6 から 89.6 になった。それでも Dopamine の IQN の 119.7〜177.3 には届いていない（§5.3、P7）。
3. **初期化の設定を読む並びが 14 か所に写されていて、読み手ごとに読み方と止まり方が違う。** 欄を 1 つ足すと 14 か所を直すことになる（P10）。block の 9 か所は、不正な値を最初の forward まで止めず、ERROR の文に設定キーが出ない（P13）。DQN と Rainbow は `nonlinearity` を読まない（P19）。未知の `nonlinearity` は、黙って `relu` として扱われる（P22）。

### 1.2 ゴール

- **G1**: 設定で、どの層にも「重みを、分散が倍率÷fan_in の一様分布で初期化し、bias を 0 にする」初期値を指定できる。
- **G2**: Dopamine と同じ初期値の Breakout の Run を流せる状態にし、比べ方と読み方を Atari の枠へ渡す（D1）。
- **G3**: 既存の初期化方式と、新しい方式を使わない正しい設定の Run の学習の結果を変えない。
- **G4**: 初期化の設定を読む 14 か所が 1 つの口で読み、不正な値は、どの読み手でも読んだ直後に設定キーつきで止まる（D5）。

### 1.3 仮説と、確かめの置き場

- **仮説**: Breakout で anet の IQN（B 群）と Dopamine の IQN に残る差（it 15〜19 の平均で 89.6 対 119.7〜177.3）の一部は、初期値の幅と bias の初期値の違いにある。
- **比べ方と終わり方**: 5M の Run の段取り（seed・本数）と結果の読みは、本 PRD の受入に入れず、実装のあと Atari の枠で 13 bat と実験記録に書く（D1）。本 PRD は、比べ方と読み方の材料を §9.1 に書いて渡す。本 PRD の受入は、その Run を流せることの確認までである（§9.2 の 9）。
- **終わり方（本 PRD）**: 効果が出なくても、方式と読み取りの口は残す。既定の初期値は変えないので（§1.4）、使わない設定の Run には効かない。

### 1.4 非ゴール（退けた案と、本当の制約か）

| 非ゴール | 理由 | 本当の制約か |
|---|---|---|
| Atari.txt の既定の初期値（conv は `he`、Linear と Head は `xavier`）を変えること | 効果を確かめてから、Atari の枠で実験の段取りと一緒に決める（P12） | 制約ではない。実験の結果が要る先送りである。Breakout の 5M の結果が出たら決める |
| 13 bat へのステージの追加と、5M の Run の結果の読み | Atari の枠で行う（D1） | 制約ではない。別の関心（実験の段取り）である |
| 既存の `he` の fan_out を fan_in に直すなど、既存の方式の中身を変えること | 既存の設定の Run の初期値が変わり、G3 に反する | 制約ではない。クリーンブレークで移せるが、基準の Run との比較が切れるので、直すなら別の判断にする |
| fan_out・fan_avg・正規分布の形と、それを選ぶ設定キー | 作らない。確定した要求は fan_in・一様分布だけで（P1）、形が 1 つしか無いものに列挙のキーを作らない（P11） | 制約ではない。先送りではなく、今の要求に無い形である。別の形を使う構成が決まったら、別の関心として別の PRD で足す |
| `manual_gain` が 0 以下のとき gain を `nonlinearity` から決める今の決まりを改めること | 既存の方式（`orthogonal`）の中身で、本 PRD の方式は `manual_gain` を使わない（D2） | 制約ではない。別の関心で、Atari.txt の RND の `orthogonal` の設定の書き換えを伴う |
| GraphViz の設定と観測の正規化の設定の読み取りの口、方策の検査の置き場の移動 | 読み取りの口の課題（[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)）に残す（D6） | 制約ではない。別の関心である |
| block の ERROR の文に block 名を入れること | 入れると、block のほかの検査（`res.droppath_rate` など）の文も一緒に変わる。エラーが示す設定キーを完全キーにする [PRD194](0194_config_key_full_names_in_messages_10rfd.md) が扱う（D10、P30） | 制約ではない。block の検査の全体に関わる別の関心である |
| Dopamine とのほかの差（padding、τ を混ぜる位置、目標値で行動を選ぶ τ の本数、打ち切りの扱い） | 別の関心。padding と τ を混ぜる位置は、すでに効かなかった（§5.3） | 制約ではない |

### 1.5 機構とゴールの対応

| 機構 | ゴール | 寄与 |
|---|---|---|
| 初期化方式 `variance_scaling`（D2・D3） | G1 | Dopamine の初期値を設定で作れる |
| 設定キー `<接頭辞>.variance_scale`（D2） | G1・G2 | 分散の倍率を層ごとに指定できる。Dopamine の倍率と、幅だけ torch に合わせる倍率を使い分けられる |
| 初期化の設定の読み取りの口（D5） | G1・G4 | 14 か所が同じ 8 キーを同じ読み方で読み、欄を足すときの読み忘れが無くなる |
| block の読み手の、読んだ直後の検査（D10） | G4 | block でも、不正な値が Network の構築時に設定キーつきで止まる |
| `nonlinearity` の列挙の検査（D8） | G4 | 未知の値が黙って `relu` にならない |
| 流せることの確認の Run（D19） | G2 | ステージ 57 の構成に新しい初期値の引数を足した設定が、構築を通る |

## 2. 前提

P1〜P13 は、2026-11-21 のグリルの最初に共通の前提として示し、説明が要る番号は挙がらなかった。P15〜P20 は 1 つの問いに添えた前提、P21〜P30 は起草と、別モデルの検討のあとに足した前提である。

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実（コード） | Dopamine（e7d780d）の `RainbowNetwork` と `ImplicitQuantileNetwork` は、conv 3 層・τ の射影（`dense_quantile`）・Linear 512・出力の kernel を `VarianceScaling(scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')` で初期化する。bias の初期化は指定せず、Keras の既定（0）になる | [atari_lib.py](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py) 304〜364 行・367〜444 行（グリルで本文をもう一度取得し、中身の一致を確かめた。行番号は起票時のもの） | 済み |
| P2 | 事実（コード） | 同じ版の `NatureDQNNetwork` は kernel の初期化を指定せず、Keras の既定（glorot uniform）になる。1/√3 の倍率は IQN と Rainbow の Network だけである | 同 251〜301 行 | 済み |
| P3 | 事実（文書） | Keras の `VarianceScaling` は、uniform のとき [−limit, limit]、limit = √(3·scale/n) から引く。mode が fan_in なら n は重みの入力の数である。重みの分散は scale/n になる | [TensorFlow の VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling) | 済み |
| P4 | 事実（コード） | anet の `WeightInitializer::Initialize` は、`default` では weight にも bias にも触らない（torch の既定のまま）。ほかの方式は weight を初期化し、bias を 0（`constant` は定数）にする | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 22〜77 行（26 行・66〜76 行） | 済み |
| P5 | 事実（コード） | anet の `he` は `kaiming_normal_` の fan_out、`xavier` は `xavier_uniform_` である | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 36 行・42 行 | 済み |
| P6 | 事実（文書） | torch の Linear・Conv2d の既定は、weight も bias も U(−√k, √k)、k = 1/fan_in（Conv2d は groups ÷（入力 channel × kernel の要素数））から引く。重みの分散は 1/(3·fan_in) になる | [torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)、[torch.nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | 済み |
| P7 | 事実（測定） | Breakout の切り分けで、効いたのは初期値だけだった | §5.3 | 済み |
| P8 | 事実（測定） | 10 月のサンプル効率優先（SN 付きの残差 conv）の qbert でも、初期値を he から PyTorch の既定に戻すと序盤の評価が上がった（1 seed）。実時間優先の構成では、戻しても上がらなかった | [atari/2026-10-23_btr-qbert-repro-init](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md) | 済み |
| P9 | 過去の判断 | 初期化方式は文字列で指定し、未知の値は即座に止める | [ADR 0008](../adr/0008-weight-init-mode-string.md) | 済み |
| P10 | 既存部品の働き | 初期化の設定は、読み手ごとに 1 キーずつ読む。読み手は 14 か所ある。読み手が読まない設定キーを書くと、block では構築時に ERROR で止まり、Agent・RND では Run の構築の終わりの WARN の一覧に出る | §5.2、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1699 行、[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 548 行 | 済み |
| P11 | 守る線（規約） | 形が 1 つしか無いものに、列挙の値を持つキーを作らない。2 つ目が要るときに足す | [実装 品質確認観点](../quality/impl.md) の「名前」の規約 5 | 済み |
| P12 | 過去の判断 | 効果の確かめと Atari の構成の既定は、Atari の枠で実験の段取りと一緒に決める | [PRD121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md) §1.3 の終わり方（D19） | 済み |
| P13 | 既存部品の働き | 不正な値の止まり方が読み手で違う。Agent・RND の 5 か所は、読んだ直後に設定キーの全体つきで止まる。block の 9 か所は、最初の forward の初期化の直前に `WeightInitConfig.<キー>` の名前で止まり、block 名も接頭辞（`init1`・`init2` など）も文に出ない | §5.1 | 済み |
| P14 | 事実（測定） | ステージ 60（B 群の seed 1 を torch の既定の初期値に）とステージ 61（PER あり・τ 8 本の構成の seed 1 を torch の既定の初期値に）の結果 | 欠け。前提なしで決めた（§5.3 の末尾。本 PRD の決定はこの結果で変わらず、5M の Run の段取りは Atari の枠で決める。D1） | 対象外 |
| P15 | 事実（文書） | Keras の `VarianceScaling` の引数は scale・mode・distribution で、既定は 1.0・fan_in・truncated_normal である。JAX の `variance_scaling` も同じ引数名を持つ。Dopamine の JAX 版も Rainbow・QR・IQN が同じ書き方をする。torch.nn.init に同じ名前の関数は無く、近いのは `kaiming_uniform_`（幅 = gain × √(3/fan)）である | §5.4 | 対象外 |
| P16 | 事実（計算） | 1/√3 ≈ 0.577 は、分散の倍率として読むと Dopamine の幅になり、標準偏差の倍率（gain）として読むと torch の既定の幅になる | §5.4 | 対象外 |
| P17 | 既存部品の働き | `manual_gain` は 14 か所すべてが読む。使うのは `orthogonal` だけで、0 以下なら `nonlinearity` から gain を決める | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 43〜56 行、§5.2 | 対象外 |
| P18 | 既存部品の働き | 設定の型ごとの読み取りを 1 か所に置く口（`ConfigReader`）が既にあり、`RewardScalerConfig` と `ActionPolicyConfig` が使う。どちらの口も、読むときに口の中で検査する。`RewardScalerConfig` の口は入口を 2 つ持ち、1 つは読んで全部を検査し、もう 1 つは組み合わせの検査を残して、読み手があとで検査を呼べるようにする。型に口があると、`ANET_READ_CONFIG` はその口を通る | [scaler.hpp](../../src/core/base/include/anet/scaler.hpp) 146〜191 行、[dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 377〜404 行、[config.hpp](../../src/core/base/include/anet/config.hpp) 223〜240 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 159 行・250〜251 行 | 対象外 |
| P19 | 既存部品の働き | DQN の 3 か所と Rainbow は `nonlinearity` を読まず、"linear" を使う。block の 9 か所と RND は設定から読む。`nonlinearity` を使うのは `he` と `orthogonal` の gain の計算だけである | §5.2、[nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 37〜56 行 | 対象外 |
| P20 | 既存部品の働き | 今、`DefaultDQNAgent.head.init.nonlinearity` を書くと、読まれない設定キーとして Run の構築の終わりの WARN の一覧に出る | §5.2（P10 と P19 の帰結） | 対象外 |
| P21 | 既存部品の働き | 価値の分解の Head（`intrinsic.value.head.init`）は、方式の列挙をほかの設定キーの列挙と同じ時点で先に検査し、数値をあとで検査する | [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 223 行・242 行 | 対象外 |
| P22 | 事実（コード） | `GetNonlinearityType` は、`relu`・`linear`・`tanh`・`leaky_relu` のどれでもない名前を `relu` として扱う | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 152〜159 行 | 対象外 |
| P23 | 既存部品の働き | 読み手は、読む前に自分の既定を置く。ResBlock は `init1`・`init_ds` が `he`、`init2` が `constant`、CNBlock は 3 つとも `trunc_normal`（`trunc_std` 0.02）、Conv1d・Conv2d は `he`、skip の Head は Q の Head の値を引き継ぐ。既定を置かない読み手（Linear の block、DQN と Rainbow の Head、RND の Head）は、型の既定の `xavier` になる。`nonlinearity` の型の既定は `relu` である | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 1659〜1661 行・1911〜1916 行・3032〜3044 行・3067 行・3104 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 609 行、[nn.hpp](../../src/core/base/include/anet/nn.hpp) 65〜67 行 | 対象外 |
| P24 | 事実（コード） | ImageCls と MuZero の Head は、初期化の設定を設定ファイルから読まず、コードで直接組み立てて `WeightInitializer::Initialize` に渡す | [image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) 593 行、[muzero_proto_agent.hpp](../../src/algo/muzero/src/muzero_proto_agent.hpp) 31〜38 行 | 対象外 |
| P25 | 事実（コード） | 設定の実数の値は、数値の文字列をそのまま読む。式（1/√3 など）は書けない | [config.cpp](../../src/core/base/src/config.cpp) 425 行 | 対象外 |
| P26 | 事実（検索） | 出荷設定で `nonlinearity` を書く行は [nn.txt](../../apps/runner/config/nn.txt) の 3 行（10 行はコメント、299 行・307 行は値が `relu`）だけで、`head.init.nonlinearity` を書く行は無い | `git grep -n -i nonlinearity -- apps/`（2026-11-21、3 件） | 対象外 |
| P27 | 既存部品の働き | Run フォルダの `config/config_data.txt` は、Agent を組み立てる前の、解決済みの入力の写しである。読まれない設定キーも出る。読み手が読んだ値の記録が外に出るのは Agent だけで、`config/<Agent>.txt`、ログの `<Agent> config=` の行、DefaultDQNAgent の checkpoint の設定の文字列に出る。`config/<Agent>.txt` の実数は丸めて出る（3.125e-4 が `0.000312` になる）。block と RND の読み手の記録は、外に出ない | [RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 387〜388 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 317〜318 行・526 行、[rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp) 39〜40 行、ステージ 59 の Run の `config/DefaultDQNAgent.txt` の 214 行（2026-11-21 に確認） | 対象外 |
| P28 | 過去の判断 | block の読み手が動くのは、Network の構造が使う block を作るときだけである。定義しただけで使わない block は、Run では検査しない | [PRD117](0117_warn_policy_and_unread_config_10rfd.md) D13、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1674〜1699 行 | 対象外 |
| P29 | 事実（コード） | checkpoint を読むとき、保存された設定の文字列はログへ出すだけで、今の設定と比べない | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 575〜578 行 | 対象外 |
| P30 | 過去の判断 | [PRD194](0194_config_key_full_names_in_messages_10rfd.md)（2026-11-21 にグリル、未実装）は、`src` のエラー・WARN・INFO が示す設定キーを、設定ファイルにそのまま書ける完全キーにする。読み口が、読んだ欄ごとに完全キーを記録する仕組みを足す。今の block の検査の文は相対キーで書かれている。読まれない設定キーの ERROR だけは、block を作る側が block 名を足している | [PRD194](0194_config_key_full_names_in_messages_10rfd.md) の D1・D2・D4・§6.7、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1699 行 | 対象外 |

## 3. 背景

サブ機では、Taiga ら（Rainbow＋RND、ICLR 2020）と anet の Montezuma の差を調べている。Dopamine の Q 側に寄せた構成に Taiga らの式の RND を載せた Run は、鍵より先へ進まなかった。そこで、まず Q 側（IQN）が易しいゲームで Dopamine の IQN の公開値と同じように学べるかを、Breakout と Qbert で確かめた。Qbert は Dopamine の帯に入り、Breakout は下回った。Breakout の切り分けの結果は §5.3 に書く。

読み取りの口の課題（[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)）は、2026-11-10〜11 のリファクタリング候補の抽出で見つかった。初期化の設定・GraphViz の設定・観測の正規化の設定の 3 つの型を、使う側が 1 キーずつ読んでいる。本 PRD は初期化の設定に欄を足すので、グリルで開発者が、このうち初期化の分を本 PRD で片付けると決めた（§10）。

## 4. 用語

新しい用語は足さない。`CONTEXT.md` を変えない（D25）。

## 5. 現状（2026-11-21 にコードと Run で確認）

### 5.1 初期化方式と検査

- 方式は `default`・`xavier`・`he`・`orthogonal`・`constant`・`trunc_normal` の 6 つで、`ValidateWeightInitConfigMode` が列挙を検査する（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 735〜744 行）。`ValidateWeightInitConfig` は、方式によらず数値の引数の有限性と `trunc_std > 0`・`trunc_a < trunc_b` を検査する（同 746〜764 行）。`nonlinearity` の値は検査しない（P22）。
- 適用は `WeightInitializer::Initialize` の 1 か所で、Body の Linear・Conv・残差の block・CNBlock、DQN の Head、MuZero の Head、汎用の Linear の Head が呼ぶ（P4）。`Initialize` は、重みに触る前に `ValidateWeightInitConfig` を呼ぶ（[nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 25 行）。
- 検査を呼ぶ時機は読み手で違う。Agent・RND の 5 か所は、設定を読んだ直後に、設定キーの接頭辞を渡して呼ぶ（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 242 行・324 行・616 行、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 35 行、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 613 行）。block の 9 か所は、読んだあとに呼ばない。module は最初の forward で層を作り、そのときの `Initialize` の中の検査が、接頭辞を渡さずに止める（[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 560 行）。文は `Invalid WeightInitConfig.trunc_std=0; expected > 0.` の形になる（P13）。
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

初期化の設定を読む所は 14 か所ある（`git grep -n -E "ANET_READ_CONFIG\([a-z_]+, *[a-z_.0-9]*init[a-z0-9_]*\.mode\)" -- src` の全件、14 件）。どの読み手も `mode`・`manual_gain`・`constant_val`・`trunc_std`・`trunc_a`・`trunc_b` の 6 キーを 1 行ずつ読む。`nonlinearity` を読むかは読み手で違う。

| 読み手 | 設定の接頭辞 | `nonlinearity` | 位置 |
|---|---|---|---|
| ResBlock | `init1`・`init2`・`init_ds` | 読む | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 1678〜1700 行 |
| CNBlock | `init_dw`・`init_pw1`・`init_pw2` | 読む | 同 1926〜1950 行 |
| Linear | `init` | 読む | 同 3036〜3042 行 |
| Conv1d | `init` | 読む | 同 3073〜3079 行 |
| Conv2d | `init` | 読む | 同 3110〜3116 行 |
| DefaultDQNAgent | `intrinsic.value.head.init`・`head.init`・`skip.head.init` | 読まない（"linear" を使う。`skip.head.init` は `head.init` の値を引き継ぐ） | [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 216〜222 行・317〜323 行・609〜615 行 |
| RainbowAgent | `head.init` | 読まない（"linear" を使う） | [rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 28〜34 行 |
| RND | `head.init` | 読む | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 565〜571 行 |

- 1 キーずつ読む行は、全部で 94 行ある（下の「読み手の検索」を `src` にかけた件数。nn_modules.cpp 63 行、default_dqn_agent.hpp 18 行、intrinsic_rnd.cpp 7 行、rainbow_agent.hpp 6 行）。
- DefaultDQNAgent は、価値を分解しない構成で `intrinsic.value.head.init.*` に既定と違う値が書かれていると、キーごとに知らせる（同 286〜295 行の `report_init`、6 キー）。
- 設定ファイルでは、[agent.txt](../../apps/runner/config/agent.txt) 17〜22 行が `intrinsic.value.head.init.*` の既定を 6 行並べ、[nn.txt](../../apps/runner/config/nn.txt) 5〜21 行が方式ごとの書き方をコメントで示す。
- ImageCls と MuZero の Head は、設定を読まずにコードで直接組み立てる（P24）。読み手に数えない。

読み手の検索（縦棒を含むので、表に書かずここに置く）:

```text
git grep -n -E "ANET_READ_CONFIG\([a-z_]+, *[a-z_.0-9]*init[a-z0-9_]*\.(mode|manual_gain|nonlinearity|constant_val|trunc_std|trunc_a|trunc_b|variance_scale)\)" -- src
```

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
- 初期値の効きを見た対は、seed 2 の 1 対（ステージ 57 と 59）だけである。seed 1 を torch の既定の初期値にしたステージ 60（`run_20261121-093204_bo_dqiqn_initdef_s1`、ステージ 50 と対）は 2026-11-21 09:32 に始まり、改訂の時点で走っている。ステージ 61（PER あり・τ 8 本の構成の seed 1）はそのあとに走る。どちらの結果も本 PRD には無い（P14）。

### 5.4 ほかのライブラリの名前と、倍率の値の意味（2026-11-21 に各ページを取得して確認）

- Keras の `VarianceScaling(scale, mode, distribution)` は、既定が scale 1.0・mode `fan_in`・distribution `truncated_normal` である。mode は `fan_in`・`fan_out`・`fan_avg`、distribution は `truncated_normal`・`untruncated_normal`・`uniform` を受け付ける。uniform の幅は ±√(3·scale/n) である（[TensorFlow の VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling)）。
- JAX の `jax.nn.initializers.variance_scaling(scale, mode, distribution)` は同じ引数名を持ち、既定は無い（[JAX の variance_scaling](https://docs.jax.dev/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html)）。
- Dopamine の JAX 版（master）は、Rainbow・QR・IQN の Network が `variance_scaling(scale=1.0 / jnp.sqrt(3.0), mode='fan_in', distribution='uniform')` を使い、DQN の Network は `xavier_uniform()` を使う（[jax/networks.py](https://github.com/google/dopamine/blob/master/dopamine/jax/networks.py)）。
- torch.nn.init に `variance_scaling` という関数は無い。近いのは `kaiming_uniform_(tensor, a, mode, nonlinearity)` で、幅は gain × √(3/fan) である。gain は標準偏差の倍率で、`nonlinearity` から計算され、自由な倍率の引数は無い（[torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html)）。
- 同じ 1/√3 ≈ 0.577 が、読み方で別の初期値を指す。分散の倍率として読むと、重みの分散は 0.577/fan_in（Dopamine の幅）になる。標準偏差の倍率として読むと、重みの分散は 0.577²/fan_in ＝ 1/(3·fan_in)（torch の既定の幅）になる。Dopamine の幅を標準偏差の倍率で書くと 0.7598（0.577 の平方根）である。

## 6. 確定事項

### 6.1 決定の一覧

2026-11-21 のグリルで決めた。D1・D2・D5〜D7 は問いへの答え、D3・D4・D8〜D10・D22〜D25 は規約・現行コード・ほかの決定から決まるもの、D11〜D21 は推奨の既定を採った細部である。D5 の入口の形、D8 の検査の時点、D19〜D21 の確かめ方は、別モデルの検討のあとに直した（§10）。

| # | 決定 | 理由（よりどころ） |
|---|---|---|
| D1 | **G2 の範囲。** 本 PRD の受入は、Dopamine と同じ初期値の Breakout の Run を流せることの確認まで（D19）にする。5M の Run の段取り（seed・本数）と結果の読みは、実装のあと Atari の枠で 13 bat と実験記録に書く。本 PRD には、仮説・比べ方・読み方を材料として残す（§9.1） | P12 の、効果の確かめは Atari の枠で段取りと一緒に決める扱いによる。5M の Run はサブ機で約 2 時間かかり、commit・pull・build を経て流すので、受入に入れると実装の工程がその完走を待つ |
| D2 | **方式の名前と倍率。** 方式は `<接頭辞>.mode = variance_scaling`、倍率は新しい設定キー `<接頭辞>.variance_scale` にする。値は分散の倍率で、重みの分散は variance_scale ÷ fan_in になる。既存の `manual_gain`（標準偏差の倍率）は使わず、値の意味も揃えない | P1・P15 の、Dopamine・Keras・JAX がこの名前と値の意味で書くことによる。P16 のとおり、同じ 0.577 が読み方で別の初期値になるので、Dopamine のソースの数字をそのまま書ける形にする。`manual_gain` を使うと、ソースの数字を書いた Run が torch の幅で流れ、ERROR にも WARN にもならない。`manual_gain` は 0 以下を自動と読む決まりも持つ（P17） |
| D3 | **形。** fan_in・一様分布だけにし、fan と分布を選ぶ設定キーを作らない。bias は 0 にする | P1 の、Dopamine の形はこれだけであることと、P11 による。bias は、P4 の、`default` 以外の方式がどれも 0 にする扱いに揃える |
| D4 | **fan_in の数え方と乱数。** fan_in は torch の数え方に従う（Linear は入力の数、Conv は入力 channel × kernel の要素数 ÷ groups）。乱数は、既存の方式と同じ torch の既定の乱数を使う | P6 の数え方は、P3 の Keras の数え方と同じ数になる。乱数の系統を足すと、同じ seed の既存の Run との連続が切れる（G3） |
| D5 | **読み取りの口。** 初期化の設定の読み取りを 1 つの口にまとめ、14 か所はその口で読む。口は入口を 2 つ持つ。1 つは、読んで、その場で `ValidateWeightInitConfig` まで行う。もう 1 つは、読むだけで検査しない。block の 9 か所と、DQN の `head.init`・`skip.head.init`、Rainbow の `head.init` の 12 か所は、読んで検査する入口を使う。価値の分解の Head（`intrinsic.value.head.init`）と RND の 2 か所は、読むだけの入口を使い、今の位置で検査を呼ぶ | 開発者の選択（§10）。P13 の差を無くし、欄を足すときの読み忘れを無くす。入口を 2 つにするのは、P18 の `RewardScalerConfig` の口と同じ形である。12 か所は口を通すだけで検査され、検査の呼び忘れが起きない。2 か所は、P21 の検査の順と、RND のほかの設定キーとの検査の順（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 593〜613 行）を保つために、読むだけの入口を使う |
| D6 | **口の範囲。** 口にまとめるのは初期化の設定だけにする。GraphViz の設定、観測の正規化の設定、方策の検査の置き場は、読み取りの口の課題に残す | 開発者の選択（§10）。本 PRD が欄を足すのは初期化の設定だけで、ほかの 2 つの型は別の関心である。3 つの型の口をまとめて作る案は採らない。制約ではない（§1.4） |
| D7 | **`nonlinearity` を揃える。** どの読み手も同じ 8 キー（今の 7 キーと `variance_scale`）を読む。DQN の `head.init`・`intrinsic.value.head.init` と Rainbow の `head.init` は、読む前に "linear" を既定として置く。`skip.head.init` は、今と同じく `head.init` の値を引き継ぐ | P19 の、差が gain の既定だけに効くことによる。D9 の、読み手が既定を置く仕組みで、書かないときの結果が今と同じになり、口に分岐が要らない。P26 のとおり、`head.init.nonlinearity` を書く出荷設定は無い |
| D8 | **`nonlinearity` の検査。** 受け付ける値は `relu`・`linear`・`tanh`・`leaky_relu` の 4 つにする。それ以外は、方式によらず ERROR で止める。検査する時点は、方式の列挙と同じにする（§8.1） | P21 の、列挙を数値より先に確かめる順による。`nonlinearity` は列挙なので、方式と同じ時点で検査しないと、価値の分解の Head で、ほかの設定キーの数値の ERROR が先に出る。P22 の、未知の名前を黙って `relu` として扱う今の動きは、[AGENTS.md](../../AGENTS.md) の「設定値の扱い」（未知の列挙値は止める。黙って既定値へ置き換えない）に反する。D7 で DQN と Rainbow が新しく読む設定キーになるので、不正な値の扱いを決める必要がある。P26 のとおり、新しく止まる出荷設定は無い |
| D9 | **読み手ごとの既定。** 口は、渡された設定の今の値を既定として読む。読み手ごとの今の既定（P23）は、D7 の "linear" を除いて、今の値のまま保つ。`skip.head.init` は、新しい `variance_scale` も含めて、8 キーのどれも `head.init` の値を既定にする | P23 の仕組みをそのまま使う。既定を口に持たせる案（読み手の種類を口に渡す）は、口に読み手の知識が入るので採らない。既定を変えると既存の設定の Run の初期値が変わり、G3 に反する。これは制約である |
| D10 | **不正な値の止まり方。** block の 9 か所は、Network の構築時、読んだ直後に、block の中の相対の設定キー（`init1.variance_scale` など）つきで止まる。既存の 7 キーの不正な値も同じに変わる。Agent・RND の止まり方は今と同じである。`WeightInitializer::Initialize` の中の検査は、消さずに置く | D5 の帰結。相対の設定キーで出すのは、block のほかの検査（[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 1709 行の `res.droppath_rate`）と同じ形である。block 名を含む完全キーにするのは、P30 の PRD194 が block の検査の全体について行う（§11）。`Initialize` の中の検査を消す案は採らない。P24 のとおり、設定を通さず直接組み立てる経路があり、消すと検査を通らない経路ができる。これは制約である |
| D11 | **`variance_scale` の既定値。** 1.0 にする（`skip.head.init` だけは、D9 のとおり `head.init` の値を既定にする）。Dopamine に合わせるときは 0.5773502691896258 を書く | P15 の Keras の既定と同じで、特定の手法に結び付かない。P25 のとおり式は書けないので、数値を書く。ほかの数値のキー（`trunc_std` など）と同じく既定を持たせる |
| D12 | **`variance_scale` の値域と検査。** 有限で 0 より大きい実数にする。方式によらず検査し、反したら ERROR で止める | `trunc_std` と同じ扱い（§5.1）にし、方式で検査を分けない |
| D13 | **方式が `variance_scaling` でないときに書いた `variance_scale`。** D12 の検査だけを行い、値は使わない。WARN は足さない | `trunc_std`・`constant_val` の今の扱いと同じにする。使わない値の知らせは、D18 の価値の分解の Head の仕組みだけが持つ。方式を書き忘れて倍率だけを書いた設定は止まらないので、Run を組むときは D19 と同じく、接頭辞ごとの方式の行を設定の記録で確かめる |
| D14 | **ERROR の文。** 今の文の形に揃える。方式の文は、受け付ける値の並びに `variance_scaling` を足す。`nonlinearity` は `Invalid <設定キー>.nonlinearity='<値>'; expected relu, linear, tanh or leaky_relu.`、`variance_scale` は `Invalid <設定キー>.variance_scale=<値>; expected finite value.` と `…; expected > 0.` の形にする | [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 740〜741 行・754 行・758 行の今の文の形による |
| D15 | **口が読む順。** `mode`・`manual_gain`・`nonlinearity`・`constant_val`・`trunc_std`・`trunc_a`・`trunc_b`・`variance_scale` の順に読む | block と RND が今読む順（§5.2）の末尾に足す形にする。DQN と Rainbow の設定の記録（P27）では、`nonlinearity` の行が `manual_gain` の次に入る |
| D16 | **nn.txt。** 方式ごとの書き方の例に、`#init.mode ?= variance_scaling` と `#init.variance_scale ?= 1.0` の行を足す。コメントに、fan_in の一様分布で bias が 0 になることと、Dopamine の Rainbow・IQN の値 0.5773502691896258 を書く | 今の例の並び（§5.2）に揃える。P15 のとおり Keras の既定の分布は一様でないので、名前から形を取り違えないように例で補う |
| D17 | **agent.txt。** `intrinsic.value.head.init.*` の既定の並びに、`nonlinearity ?= linear` と `variance_scale ?= 1` を足す | 今の 6 行の並び（§5.2）に揃える |
| D18 | **使わない設定の知らせ。** 価値を分解しない構成で `intrinsic.value.head.init.*` を書いたときの知らせに、`nonlinearity` と `variance_scale` を足す。`nonlinearity` は、D7 の既定 "linear" と違うときに知らせる | 今の 6 キーの扱い（§5.2）に揃える |
| D19 | **流せることの確認。** 13 bat のステージ 57 の引数に、§7 の例の 12 個の引数を足した設定で、Run 名に `tmp` を含めた短い Run を起動する。構築を通ること、入力の写しに 12 行があること、Agent の設定の記録に Head の方式と倍率が出ることを確かめる（§9.2 の 9） | D1 の受入を、観測できる結果で確かめる。P27 のとおり、入力の写しは読まれなくても出るので、読んだことは別に確かめる。block は、P10 のとおり、読まれない設定キーがあると構築で止まるので、構築を通れば読んだことになる。ステージ 59・60 を組むときに行った設定の確認の Run と同じ手順である |
| D20 | **分布のテストの許容幅。** 重みの絶対値の最大が √(3·s/fan_in) の (1 + 1e-6) 倍以下、標本の分散が s/fan_in の ±5% の内、bias がすべて 0 とする（s は `variance_scale`） | 一様分布の標本 8,192 個（Conv2d 4→32、8×8）で、分散の相対の標準誤差は約 1% である。±5% はその約 5 倍に当たる。幅の端に 1e-6 の余裕を置くのは、幅の計算の丸めで端をわずかに超えうるためである |
| D21 | **既存の挙動の確かめ。** 3 つで確かめる。(1) 新しい方式を使わない設定で、同じ seed・決定的な backend の短い Run の metrics を、実装前に保存した実行体の Run と比べて一致する。対象は、LunarLander の既定の 100,000 exp step と、Breakout のステージ 57 の構成の 30,000 exp step にする。比べる道具と範囲は §9.2 の 1 に書く。(2) DQN と Rainbow が出荷設定から読んだ設定の採取が、実装の前後で、足した設定キーの行を除いて一致する（§9.2 の 10）。(3) 初期化の設定を書かない block の初期値が P23 の既定の方式の形であり、初期化の設定を書かない RND の Head が読む値が P23 の既定である（§9.2 の 11） | G3 の確かめ方。(1) は、315 ニューラルネットワークのトピック別まとめの「変えるときの条件」（[PRD084](done/0084_noisynet_10rfd.md) §12・§13）と同じ比べ方である。(1) の 2 つの Run が通る読み手は、Linear・Conv2d の block と DQN の Head だけなので、口に替える残りの読み手は (2) と (3) で押さえる。(2) は、出荷設定を実際の設定の組み立てに通して採る既存のテスト（[dqn_config_baseline_test.cpp](../../src/algo/dqn/src/dqn_config_baseline_test.cpp) 25〜73 行）を使う。P27 のとおり block と RND の読み手の記録は外に出ないので、(3) は、block を初期値の形で、RND をテストから読む設定の値で確かめる |
| D22 | **Atari.txt と bat。** 本 PRD では変えない。確かめの Run は bat の引数だけで組める（§7） | D1 と P12 による。本 PRD で Atari.txt に Dopamine の初期値のプロファイルを足す案は採らない。制約ではなく、Atari の枠で実験の段取りと一緒に決めるためである（§1.4） |
| D23 | **設計文書。** §8.5 に挙げた設計文書の行を直す。英語版は、翻訳元の情報の扱いに従って合わせる | [実装 品質確認観点](../quality/impl.md) の「検証と報告」の、英語版を編集するときの決まりによる |
| D24 | **ADR。** 起こさない | 既定を変えない方式の追加と、既にある読み取りの口の形（P18）の適用で、戻しやすい。P9 の ADR 0008 の判断（文字列で指定し、未知の値を止める）とも矛盾しない |
| D25 | **用語。** `CONTEXT.md` を変えない | 初期化方式の項目は今も無く（`CONTEXT.md` を「初期化方式」「WeightInit」「variance」で検索して 0 件）、方式の名前は設定の値である。方式の名前を用語集に足す案は採らない。制約ではないが、用語集はドメインの語を載せる所で、設定の値の一覧は設計文書が持つ |

### 6.2 退けた案

| 案 | 退けた理由 |
|---|---|
| 方式の名前を `fan_in_uniform` にする | 挙動は D2 と同じで、分布と fan が名前から読める。ただし Dopamine・Keras の設定と名前が対応しない。主要なライブラリにこの名前の初期化は無い |
| 倍率に既存の `manual_gain` を使う | 新しい設定キーも読み手の変更も要らないが、D2 の理由のとおり、Dopamine のソースの数字を書いた Run が黙って torch の幅で流れる |
| Keras と同じく fan と分布を選ぶ設定キーを同時に作る | D3 の理由のとおり、形が 1 つしか無い |
| 今の読み方のまま、14 か所に `variance_scale` を読む 1 行を足す | Run までが最短で、既存の挙動に波及しない。AI はこれを推奨したが、開発者が読み取りの口を作る形を選んだ（§10）。採らないことで、P13 の差が残らない |
| 口は読むだけにし、検査は 14 か所がそれぞれ呼ぶ | 起草の最初の形。P18 のとおり、今ある口は読むときに検査していて、形が揃わない。検査の行が 14 か所に残り、あとから足す読み手が呼び忘れうる。別モデルの検討で分かり、D5 の入口が 2 つの形に改めた |
| 読み取りの口の課題の全体を、別の PRD で先に片付ける | Dopamine の初期値の Run が、別の PRD のグリルと実装のあとになる |
| DQN と Rainbow が `nonlinearity` を読まない今の形を保ち、口に「読まない」指定を持たせる | 口に分岐が増え、読み手が 2 種類になる |
| 5M の Run の読みまでを本 PRD の受入に入れる | D1 の理由のとおり、実装の工程が Run の完走を待つ |

### 6.3 起票時の未決表との対応

| 起票時の行 | 行き先 |
|---|---|
| 1 方式の名前 | D2 |
| 2 倍率の設定キー | D2・D11・D12 |
| 3 fan と分布の選び方 | D3 |
| 4 bias | D3 |
| 5 既定値 1.0 | D11 |
| 6 検証の範囲 | D12・D13 |
| 7 読み手 | D5・D6（14 か所に足す既定を、読み取りの口を作る形に変えた） |
| 8 設定ファイル | D16・D17 |
| 9 Atari.txt と bat | D1・D22 |
| 10 サブ機への届け方 | 外した。本 PRD の契約に効かない。commit と push は開発者が行う今の運用（[AGENTS.md](../../AGENTS.md) の「Git 操作・コミットメッセージルール」）による |
| 11 確かめの Run の本数 | D1（Atari の枠で決める。材料は §9.1） |
| 12 設計文書と ADR | D23・D24 |
| 13 用語 | D25 |

## 7. 仕様

- `<接頭辞>.mode = variance_scaling` のとき、weight を [−√(3·s/fan_in), √(3·s/fan_in)] の一様分布から引き（s は `<接頭辞>.variance_scale`）、bias を 0 にする。重みの分散は s/fan_in になる（D2・D3）。
- fan_in は torch の数え方による。Linear は入力の数、Conv は入力 channel × kernel の要素数 ÷ groups である（D4）。
- `variance_scaling` は、`manual_gain` と `nonlinearity` を使わない（D2）。
- 例（Breakout の確かめの Run で bat に渡す 12 個の引数。Dopamine と同じ倍率）:

```text
net.block.[AtariConv32].init.mode=variance_scaling net.block.[AtariConv32].init.variance_scale=0.5773502691896258
net.block.[AtariConv64].init.mode=variance_scaling net.block.[AtariConv64].init.variance_scale=0.5773502691896258
net.block.[AtariConv64Final].init.mode=variance_scaling net.block.[AtariConv64Final].init.variance_scale=0.5773502691896258
net.block.[AtariLinear512].init.mode=variance_scaling net.block.[AtariLinear512].init.variance_scale=0.5773502691896258
net.block.[AtariIQNTauProj512].init.mode=variance_scaling net.block.[AtariIQNTauProj512].init.variance_scale=0.5773502691896258
DefaultDQNAgent.head.init.mode=variance_scaling DefaultDQNAgent.head.init.variance_scale=0.5773502691896258
```

- 幅だけを torch の既定と同じにし、bias を 0 にするときは、`variance_scale=0.3333333333333333` を書く（§5.4）。
- プロファイルで書くときの例: `DefaultDQNAgent.@baseline : head.init.mode ?= variance_scaling`、`DefaultDQNAgent.@baseline : head.init.variance_scale ?= 0.5773502691896258`。

## 8. 実装契約

### 8.1 設定と失敗の扱い

`<接頭辞>` は §5.2 の 14 か所の接頭辞である。足す・変える設定キーは次の 3 つで、ほかの 5 キー（`manual_gain`・`constant_val`・`trunc_std`・`trunc_a`・`trunc_b`）の型・値域・既定は今のままである（退けた案と理由は §8.3）。

| 設定キー | 型 | 値の範囲 | 既定 |
|---|---|---|---|
| `<接頭辞>.mode` | 文字列 | `default`・`xavier`・`he`・`orthogonal`・`constant`・`trunc_normal`・`variance_scaling` | 読み手ごとの今の既定（P23）。`skip.head.init` は `head.init` の値 |
| `<接頭辞>.nonlinearity` | 文字列 | `relu`・`linear`・`tanh`・`leaky_relu` | block と RND は `relu`。DQN の `head.init`・`intrinsic.value.head.init` と Rainbow の `head.init` は `linear`。`skip.head.init` は `head.init` の値（D7・D9） |
| `<接頭辞>.variance_scale` | 実数 | 有限で 0 より大きい | 1.0（D11）。`skip.head.init` は `head.init` の値（D9） |

`skip.head.init` は、8 キーのどれも、書かなければ `head.init` の値になる（D9）。たとえば `head.init` に `variance_scaling` と 0.5773502691896258 を書いて skip を有効にすると、skip の Head も同じ方式と倍率で初期化される。

判定は次の表で、上から順に当て、最初に当たった行で扱う。どの行も、`mode` によらず当てる（D8・D12）。

| # | 条件 | 扱い |
|---|---|---|
| 1 | `mode` が上の 7 つのどれでもない | ERROR で止める。文に設定キー、指定した値、受け付ける 7 つの値を入れる |
| 2 | `nonlinearity` が上の 4 つのどれでもない | ERROR で止める。文に設定キー、指定した値、受け付ける 4 つの値を入れる |
| 3 | `manual_gain`・`constant_val`・`trunc_std`・`trunc_a`・`trunc_b`・`variance_scale` のどれかが有限でない（NaN・無限大） | ERROR で止める。文に設定キーと値を入れる |
| 4 | `trunc_std` が 0 以下 | ERROR で止める（今と同じ） |
| 5 | `trunc_a` が `trunc_b` 以上 | ERROR で止める（今と同じ） |
| 6 | `variance_scale` が 0 以下 | ERROR で止める。文に設定キーと値を入れる |
| 7 | 上のどれにも当たらない | `mode` に従って初期化する。`variance_scaling` でない `mode` では `variance_scale` を使わず、知らせない（D13） |

行 1・2 は列挙の検査、行 3〜6 は数値の検査である。列挙の検査だけを先に呼ぶ読み手があるので（P21）、列挙の検査は、数値の検査と分けて呼べる形を保つ（D8）。

止める時機と文の設定キーは、読み手で次のとおりである（D5・D10）。

| 読み手 | 止める時機 | 文の設定キー |
|---|---|---|
| block の 9 か所 | Network の構築時。設定を読んだ直後で、最初の forward より前 | block の中の相対の設定キー（例: `init1.variance_scale`） |
| DQN の `head.init`・`skip.head.init`、Rainbow の `head.init` | 設定を読んだ直後（今と同じ） | 設定キーの全体（例: `DefaultDQNAgent.head.init.variance_scale`） |
| DQN の `intrinsic.value.head.init` | 行 1・2 は、ほかの設定キーの列挙の検査と同じ時点。行 3〜6 は、そのあとの数値の検査の時点（P21 の順） | 設定キーの全体 |
| RND の `head.init` | RND のほかの設定キーの検査のあと（今と同じ位置） | 設定キーの全体 |
| 設定を通さない直接の指定（ImageCls・MuZero） | 初期化の直前（今と同じ） | `WeightInitConfig.<キー>`（今と同じ） |

block の読み手が動くのは、Network の構造が使う block を作るときだけである。定義しただけで使わない block の不正な値は、今と同じく Run では止まらない（P28）。G4 の「どの読み手でも」は、動いた読み手を指す。

### 8.2 読み取りの口

- 口は、初期化の設定を、渡された接頭辞の下の 8 キーで読む。読む順は D15 による。読んだ値は、今と同じく、読み手の設定の記録に残す。
- 口は入口を 2 つ持つ（D5）。読んで検査する入口は、読んだ直後に §8.1 の判定を当て、文の設定キーには、読み手の接頭辞を付けた設定キーを使う。読むだけの入口は検査せず、読み手が §8.1 の時点で検査を呼ぶ。
- 口は、渡された設定の今の値を既定として読む。読み手は、読む前に自分の既定を置く（D9）。
- §5.2 の 14 か所は、初期化の設定をこの口で読む。1 キーずつ読む行を残さない（§9.2 の 5）。
- 設定キーの名前と接頭辞は、今の名前のまま使う（退けた案と理由は §8.3）。別名は作らない。

外に出る記録は次のように変わる（P27）。

- Agent の設定の記録（`config/DefaultDQNAgent.txt`・`config/RainbowAgent.txt`、ログの `config=` の行、DefaultDQNAgent の checkpoint の設定の文字列）: DQN の 3 つの接頭辞と Rainbow の `head.init` に、`nonlinearity` と `variance_scale` の行が増える。`nonlinearity` は `manual_gain` の次、`variance_scale` は末尾に入る（D15）。DefaultDQNAgent は、checkpoint を読むときに設定の文字列を比べないので（P29）、今までの checkpoint は今と同じに読める。
- 入力の写し（`config/config_data.txt`）: agent.txt に足す 2 行（D17）の分だけ増える。
- block と RND の読み手の記録は外に出ないので、変わるものは無い。

### 8.3 変えないもの

| 変えないもの | 退けた案 | 本当の制約か |
|---|---|---|
| 既存の 6 つの方式の初期化の結果と、乱数の消費の順 | 既存の方式の中身も合わせて直す案（§1.4） | G3 による制約である |
| 新しい方式を使わない正しい設定の Run の学習の結果 | 出荷設定の既定の初期値を新しい方式に替える案（§1.4） | G3 による制約である |
| `manual_gain`・`constant_val`・`trunc_std`・`trunc_a`・`trunc_b` の型・値域・既定 | 既存の数値のキーの値域も合わせて見直す案 | 制約ではない。別の関心である |
| 14 か所の設定キーの名前と接頭辞 | 接頭辞を 1 つの形に揃える案 | 制約ではない。別の関心で、出荷設定と bat の書き換えを伴う |
| Agent・RND の 5 か所で検査が走る位置と、価値の分解の Head の検査の順 | 14 か所とも、読んだ直後に全部の検査を行う案 | P21 の順を保つための制約である。順が変わると、2 つの設定キーが同時に不正なときに先に出る ERROR が変わる（[dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp) 2080〜2092 行が断定している） |
| `manual_gain` が 0 以下のとき gain を `nonlinearity` から決める決まり | 0 以下を ERROR にして必ず書かせる案（§1.4） | 制約ではない。別の関心である |
| `WeightInitializer::Initialize` の中の検査 | 検査を読み手だけに置く案 | P24 による制約である（D10） |

### 8.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ConfigReader` の `WeightInitConfig` 用の口 | 新規 | 初期化の設定の 8 キーを、渡された接頭辞で読み、記録する。読んで検査する入口と、読むだけの入口を持つ | NN の公開の口（nn） |
| WeightInitConfig | 変更 | `variance_scale` を持つ | NN の公開の口（nn） |
| WeightInitializer | 変更 | `variance_scaling` の初期化を行う | NN の Head・初期化（nn_head） |
| ValidateWeightInitConfigChoices（関数。今の `ValidateWeightInitConfigMode` を改名する） | 変更 | 方式と `nonlinearity` の列挙を検査する。受け付ける方式に `variance_scaling` を足す。改名の理由は、方式に加えて `nonlinearity` も検査するので、名前を検査の中身に合わせるためである | NN の組み立て（nn_impl） |
| ValidateWeightInitConfig（関数） | 変更 | 列挙の検査に続けて、`variance_scale` の有限性と値域を検査する | NN の組み立て（nn_impl） |
| GetNonlinearityType（関数） | 変更 | 4 つのどれでもない名前を止める。検査のあとでは届かない、念のための止めである | NN の標準 module（nn_modules） |
| ResBlock・CNBlock・Linear・Conv1d・Conv2d の各 Factory の設定 | 変更 | 初期化の設定を、口の読んで検査する入口で読む | NN の標準 module（nn_modules） |
| DefaultDQNAgentConfig | 変更 | 3 つの Head の初期化の設定を口で読む。"linear" を既定として置く。使わない設定の知らせに 2 キーを足す | DefaultDQNAgent（default_dqn_agent） |
| RainbowAgentConfig | 変更 | Head の初期化の設定を口で読む。"linear" を既定として置く | RainbowAgent（rainbow_agent） |
| RndNoveltyEstimatorConfig | 変更 | Head の初期化の設定を、口の読むだけの入口で読む | 内的報酬（intrinsic_rnd） |

### 8.5 移行対象

互換層（旧い形の別名や分岐）は作らない。過去の Run・実験記録・ADR は書き換えない。検索は 2026-11-21 に、追跡しているファイルの全体へ `git grep -i -l -F` でかけた。

**`trunc_std`（1 キーずつ読む並びと、キーの一覧を書く箇所）: 28 件**

- コード（直す、7 件）: [nn.hpp](../../src/core/base/include/anet/nn.hpp)、[nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp)、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp)、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp)、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp)、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp)。
- テスト（直す、2 件）: [nn_test.cpp](../../src/core/base/src/nn_test.cpp)、[dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp)。断定が変わる所は下に挙げる。
- 設定（直す、2 件）: [agent.txt](../../apps/runner/config/agent.txt)（D17）、[nn.txt](../../apps/runner/config/nn.txt)（D16）。
- 設計文書（直す、2 件）: [130 jp](../design/130_neural_networks.jp.md)・[130 en](../design/130_neural_networks.en.md)。
- 検証資材（直さない、1 件）: `src/core/base/testdata/prd121/run_validation.py`。PRD121 の確かめの道具で、当時の既定キーを並べる。今の受入に使わない。
- 課題記録（直さない、1 件）: 読み取りの口の課題。読み手の数を当時のまま記録している。
- ほかの PRD・実装計画（直さない、4 件）: `098_temporl_skip_policy_22impl.md`、`150_residual_alpha_gate_10prd.md`、`done/028_cnblock_convnext_10prd.md`・`done/028_cnblock_convnext_20impl.md`。
- 実験記録の config の写し（直さない、8 件）: `docs/experiments/default-dqn/atari/config/` の、Montezuma の Run（Run 名に `_mz_` を含む）の設定の写し。過去の Run の記録である。
- 本 PRD（1 件）。

合計は 7＋2＋2＋2＋1＋1＋4＋8＋1＝28 件。

**`nonlinearity`（D7・D8 で読み方と検査が変わる）: 606 件**

- コード（直す、6 件）: nn.hpp、nn_head.hpp、nn_modules.cpp、default_dqn_agent.hpp、rainbow_agent.hpp、intrinsic_rnd.cpp（どれも上のコードの 7 件の内）。
- コード（直さない、1 件）: `src/algo/muzero/src/muzero_proto_agent.hpp`。設定を通さず "linear" を直接書く（P24）。値は 4 つの内である。
- テスト（直す、1 件）: nn_test.cpp。
- 設定（直す、1 件）: nn.txt。`nonlinearity` を書く 3 行（P26）は、値が 4 つの内なので直さない。直すのは D16 の例の行である。
- 検証資材（直さない、28 件）: `src/core/base/testdata/prd072/baseline/` の 17 件（固定 commit の golden）と、`src/core/base/testdata/prd092/config/` の 11 件（PRD092 の Run の設定の写し）。値はすべて `relu` である。
- 課題記録（直さない、1 件）: 読み取りの口の課題。
- ほかの PRD・実装計画（直さない、5 件）: `098_temporl_skip_policy_22impl.md`、`116_rnd_lifelong_novelty_10prd.md`、`182_source_layout_10prd.md`、`done/027_weight_init_mode_string_10prd.md`・`done/027_weight_init_mode_string_20impl.md`。
- 実験記録（直さない、560 件）: `docs/experiments/` の Run の設定の写し。値はすべて `relu` である。
- wiki（直さない、2 件）: 315 ニューラルネットワークと 324 DQN 系列のトピック別まとめ。本 PRD のグリルの書き戻しで足した行である。
- 本 PRD（1 件）。

合計は 6＋1＋1＋1＋28＋1＋5＋560＋2＋1＝606 件。

**`trunc_normal`（方式の一覧を書く箇所）: 153 件**

- 直す（8 件）: `src/core/base/` の 5 件（nn.hpp・nn_head.hpp・nn_impl.cpp・nn_modules.cpp・nn_test.cpp）、設計文書 130 の jp・en の 2 件、`apps/runner/config/nn.txt` の 1 件。
- 直さない（144 件）: `apps/runner/config/DropMerge.txt`・`nn_cnx.txt` の 2 件（`trunc_normal` を使う block の定義で、方式の一覧ではない）、ADR 0008 の 1 件、`docs/memo/` のほかの PRD・実装計画 7 件、課題記録 3 件、`src/core/base/testdata/` の 14 件（PRD072 の golden 6 件と PRD092 の Run の設定の写し 8 件）、`docs/experiments/` の 117 件。
- 本 PRD（1 件）。

合計は 8＋144＋1＝153 件。

**`ValidateWeightInitConfig`（検査の呼び出しと、その説明）: 17 件**

- コード（直す、6 件）: nn.hpp、nn_head.hpp、nn_impl.cpp、default_dqn_agent.hpp、rainbow_agent.hpp、intrinsic_rnd.cpp。block の 9 か所の呼び出しを足す nn_modules.cpp は、今は呼んでいないので、この検索には当たらない。
- テスト（直す、1 件）: nn_test.cpp。
- 設計文書（直す、4 件）: 130 の jp・en、[200 jp](../design/200_dqn_agents.jp.md)・[200 en](../design/200_dqn_agents.en.md)。
- 規約（直さない、1 件）: `AGENTS.md` の「汎用機構と利用側の責任境界」の例。検査を `ValidateWeightInitConfig` に置くという例は、変更のあとも成り立つ。
- ほかの PRD・実装計画（直さない、4 件）: `098_temporl_skip_policy_22impl.md`、`116_rnd_lifelong_novelty_10prd.md`、`121_extrinsic_intrinsic_value_decomposition_20impl.md`、`128_dqn_agent_refactor_bundle_10prd.md`。
- 本 PRD（1 件）。

合計は 6＋1＋4＋1＋4＋1＝17 件。

このうち、改名する `ValidateWeightInitConfigMode`（§8.4）は、固定文字列で引くと本 PRD のほかに 5 行、4 件に当たる。直すのは nn.hpp の 75 行、nn_impl.cpp の 735 行・748 行、default_dqn_agent.hpp の 223 行である。`121_extrinsic_intrinsic_value_decomposition_20impl.md` の 190 行は、当時の記録なので直さない。

**ERROR の文と `GetNonlinearityType`**

- `expected default, xavier`: 2 件。nn_impl.cpp（直す。D14）と本 PRD。
- `GetNonlinearityType`: 6 件。nn_head.hpp・nn_modules.cpp（直す）、`182_source_layout_10prd.md`・`done/027_weight_init_mode_string_10prd.md`・`done/027_weight_init_mode_string_20impl.md`（直さない）、本 PRD。

**直す設計文書の行**

| 文書 | 行 | 直すこと |
|---|---|---|
| [130 jp](../design/130_neural_networks.jp.md) | 299 行（§7.1） | 方式に `variance_scaling` を足す。`nonlinearity` の列挙と `variance_scale` の検査、block が読んだ直後に検査すること、読み取りの口を書く |
| [200 jp](../design/200_dqn_agents.jp.md) | 526 行 | `head.init.*` を読み取りの口で読むことと、`nonlinearity` の既定が "linear" であることを書く |
| [200 jp](../design/200_dqn_agents.jp.md) | 656 行 | Rainbow の `head.init.*` の「6 件」を 8 件にし、読み取りの口で読むと書く |
| 130 en・200 en | 上の行に当たる行 | D23 のとおり合わせる |

[212 jp](../design/212_rnd_lifelong_novelty.jp.md) の 171 行・444 行と [214 jp](../design/214_value_decomposition.jp.md) の 98 行は、`head.init.*` を今の設定キーの決まりとして指すだけなので、直さない。`CHANGELOG.md` は、リリースの前にまとめて更新する決まりなので、本 PRD の実装では直さない。

**断定が変わる、または足す既存のテスト**

| テスト | 位置 | 直すこと |
|---|---|---|
| 初期化の設定の共通の検査 | [nn_test.cpp](../../src/core/base/src/nn_test.cpp) 2002〜2046 行 | 受け付ける方式の並びに `variance_scaling` を足す。数値のキーの並びに `variance_scale` を足す。`nonlinearity` の列挙と `variance_scale` の値域の断定を足す |
| 方式ごとの初期化 | 同 2068〜2116 行 | `variance_scaling` の分布の断定を足す（§9.2 の 2） |
| Rainbow の Head の初期化の読み取り | [dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp) 1192〜1212 行 | 「6 件」の設定キーと記録の断定を 8 件にする |
| skip の Head の不正な値 | 同 1320 行 | 不正な値の並びに `variance_scale` と `nonlinearity` を足す |
| 価値の分解の Head の不正な値 | 同 2059 行 | 不正な値の並びに `variance_scale` と `nonlinearity` を足す |
| 価値の分解の Head の列挙が、ほかの設定キーの数値より先に止まること | 同 2080〜2092 行 | 列挙の設定キーの並び（2083 行）に `intrinsic.value.head.init.nonlinearity` を足す（D8） |
| 価値の分解の検査の順 | 同 2107〜2113 行 | 直さない。列挙を先に、数値をあとで検査する順（P21）の断定で、`variance_scale` の検査は数値の検査の時点に入る（§8.1） |
| 使わない設定の知らせ | 既存のテストは無い（`git grep -n -F "intrinsic.value.head.init" -- "src/**/*_test.cpp"` の 7 件に、知らせの断定は無い） | 足す（§9.2 の 7） |
| RND の Head の不正な値 | [intrinsic_test.cpp](../../src/core/base/src/intrinsic_test.cpp) 727 行 | 不正な値の並びに `variance_scale` と `nonlinearity` を足す |

block の不正な値の時機や文を断定する既存のテストは無い（`git grep -n -E "WeightInitConfig\.[a-z_]+" -- "src/**/*_test.cpp"` の当たりは nn_test.cpp の 2056 行・2063 行の 2 件で、どちらも設定を通さない直接の指定の断定であり、変わらない）。新しく止まる設定を使うテストと出荷設定も無い（P26）。

## 9. 成功の測り方と受入

### 9.1 効果の確かめ（Atari の枠へ渡す材料。決定ではない）

5M の Run の段取りと読みは、実装のあと Atari の枠で決める（D1）。次はそのときの材料である。

- **Run の作り方**: 13 bat のステージ 57 の引数に、§7 の例の 12 個の引数を足す。Run 名の例は `run_{t}_bo_dqiqn_initvs_s2`。
- **読む指標**: 学習側のゲームの点を 250,000 exp step ごとに平均した it 15〜19 の平均と、it 4・9・19 の値（§5.3 と同じ集計）。
- **並べる相手**: seed 2 はステージ 57（59.6、he・xavier）とステージ 59（89.6、torch の既定）。seed 1 はステージ 50（70.3）とステージ 60（結果は本 PRD に無い。P14）。Dopamine の IQN の 5 本は 119.7〜177.3。seed による差の目安は、seed だけが違うステージ 50 とステージ 57 の差 10.7 である。この目安は 1 対から取った値なので、下の境目とあわせて、Atari の枠で複数の seed のぶれを基準に決め直す。
- **seed と本数の決め方の目安**: ステージ 60 が seed 1 でもステージ 50 より 10.7 を超えて高ければ、初期値の効きは 2 つの seed で見えたことになり、新しい初期値の Run は seed 2 の 1 本から始められる。超えなければ、効きを見た対は seed 2 の 1 対だけなので、新しい初期値の Run も 2 つの seed で流して読む。
- **読み方の目安**（seed 2 の 1 本を、上から順に当てる）:

| it 15〜19 の平均 | 読み | 次の候補 |
|---|---|---|
| 120 以上 | 初期値（幅と bias）で Dopamine の IQN の帯に入る | seed 1 でも確かめる。Dopamine に寄せた構成（`run.@rainbow_like`）の Montezuma を、この初期値で流し直す |
| 100 以上 | 差の一部が初期値で縮む | 初期値の確かめを終え、残る差（目標値で行動を選ぶ τ の本数など）の切り分けへ移る |
| 79 以上 | ステージ 59 と同じ帯（89.6 ± 10.7）。幅と bias の違いは効かない | 初期値の確かめを終え、ほかの差へ移る |
| 79 未満 | ステージ 59 より悪い | 幅と bias を分ける Run を流す |

- **幅と bias を分ける Run**: ステージ 59 との差は、幅（標準偏差 1.32 倍）と bias（一様乱数か 0 か）の 2 つが重なる。分けるには、`variance_scale=0.3333333333333333`（幅は torch の既定と同じで、bias だけ 0）の Run を足す。

### 9.2 正しさの受入

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 既存の方式の同一性（D21） | 新しい方式を使わない設定で、同じ seed・決定的な backend（`backend.@deterministic`）の短い Run を、実装前に保存した実行体と、実装後の実行体で流す。2 つの Run の `metrics.jsonl` を、それぞれ `src/core/base/testdata/prd084/compare_metrics.py` の `normalize` にかけ、出力を同じ道具の `compare` で比べて、終了コードが 0 である。比べる範囲は、数値の scalar の全部（tag・step・値の組と、tag ごとの点の数）である。除くのは、道具の既定で除かれる tag の群（`90_perf/`・`91_resource/`・`92_resource/`・`47_agent_noisy/` で始まる tag）だけである。理由は、性能と資源の指標は実時間に依って同じ実行体でも値が変わるためである。`47_agent_noisy/` は、道具が既定で除く NoisyNet の追加の指標の群で、道具の既定をそのまま使うために除く。対象は次の 2 つで、どちらも終了コードが 0 である。(a) LunarLander の既定の設定を、100,000 exp step で止めた Run。(b) 13 bat のステージ 57 の引数の構成を、30,000 exp step で止めた Run。どちらの長さも、学習の更新が始まる `update_warmup_steps`（LunarLander.txt の設定は 76,800 以下、`run.@rainbow_like` は 20,000）を超える |
| 2 | 初期値の分布（D2〜D4・D20） | テストで、`variance_scaling` の Linear（入力 256・出力 512）と Conv2d（入力 4・出力 32・kernel 8×8）について、`variance_scale` が 0.5773502691896258 と 1.0 のそれぞれで、重みの絶対値の最大が √(3·s/fan_in) の (1 + 1e-6) 倍以下であり、標本の分散が s/fan_in の ±5% の内にあり、bias がすべて 0 である |
| 3 | 判定（§8.1） | テストで、§8.1 の判定の表の行 1〜6 のそれぞれが ERROR で止まり、文が D14 の形である。行 1 の文に `variance_scaling` を含む 7 つの値、行 2 の文に 4 つの値が入る。行 3 は `variance_scale` に NaN・正の無限大・負の無限大、行 6 は 0 と −1 を書いたときに当たる。行 2 は `nonlinearity` に `selu` を書いたときに当たり、`mode` が `xavier` でも止まる。価値の分解の Head では、`intrinsic.value.head.init.nonlinearity` に `selu`、`learner.intrinsic.loss_weight` に −1 を同時に書くと、文に `intrinsic.value.head.init.nonlinearity` を含む ERROR で止まる |
| 4 | 読み手ごとの止まり方（D5・D10） | テストで、§5.2 の 14 か所のそれぞれについて、`<接頭辞>.variance_scale` に 0 を書くと ERROR で止まる。数える単位は読み手の接頭辞で、14 件すべてである。block の 9 件は、Factory が module を作る呼び出しで止まり（forward を呼ばない）、文に `<接頭辞>.variance_scale`（例: `init1.variance_scale`）が入る。Agent・RND の 5 件は、設定を読む呼び出しで止まり、文に設定キーの全体が入る |
| 5 | 読み取りの口（D5） | §5.2 の「読み手の検索」を `src` にかけた結果が 0 行である（実装前は 94 行）。検索する範囲は `src` の下の追跡しているファイルの全体で、除くものは無い |
| 6 | `nonlinearity`（D7〜D9） | テストで、DefaultDQNAgent と RainbowAgent の `head.init.nonlinearity` に `relu` を書くと、読んだ設定と設定の記録の両方が `relu` になる。書かないと `linear` になる。DefaultDQNAgent の `head.init` に `mode = variance_scaling`・`nonlinearity = tanh`・`variance_scale = 0.5` を書き、`skip.head.init` に何も書かないと、読んだ `skip.head.init` の方式が `variance_scaling`、`nonlinearity` が `tanh`、`variance_scale` が 0.5 になる。`skip.head.init.variance_scale` に 2 を書くと、`skip.head.init` だけが 2 になり、`head.init` は 0.5 のままである |
| 7 | 使わない設定の知らせ（D18） | テストで、価値を分解しない構成で `intrinsic.value.head.init.variance_scale` に 2、`intrinsic.value.head.init.nonlinearity` に `relu` を書くと、設定キーごとに `Ignored` の WARN が 1 回ずつ出る。既定と同じ値（1 と `linear`）を書いたときは出ない |
| 8 | 設定ファイル（D16・D17） | `apps/runner/config/nn.txt` に `#init.mode ?= variance_scaling` と `#init.variance_scale ?= 1.0` で始まる行がある。`apps/runner/config/agent.txt` に `intrinsic.value.head.init.nonlinearity ?= linear` と `intrinsic.value.head.init.variance_scale ?= 1` を含む行がある。`out\test\Debug\anet-core-base-test.exe "[shipped_config]"` が通る |
| 9 | 流せることの確認（D1・D19） | 13 bat のステージ 57 の引数に §7 の例の 12 個の引数を足し、Run 名に `tmp` を含めた Run が、ログに最初の `train_step=` の行を出す所まで進む（block の 5 つの接頭辞は、読まれない設定キーがあると構築で止まるので、ここまで進めば読んだことになる）。その Run の `config/config_data.txt`（入力の写し）に、§7 の例の 6 つの接頭辞のそれぞれについて `mode = variance_scaling` と `variance_scale = 0.5773502691896258` の行（計 12 行）がある。`config/DefaultDQNAgent.txt`（Agent が読んだ設定の記録）に、`DefaultDQNAgent.head.init.mode = variance_scaling` の行と、値が `0.577` で始まる `DefaultDQNAgent.head.init.variance_scale` の行がある（この記録は実数を丸めて出す。P27）。ログの `Unread config keys` の一覧に `variance_scale` を含む行が無い |
| 10 | DQN と Rainbow が読む値の同一性（D21） | [dqn_config_baseline_test.cpp](../../src/algo/dqn/src/dqn_config_baseline_test.cpp) の `[prd061-capture]` のテストを、実装前に保存した実行体と、実装後の実行体で、別々の採取先（環境変数 `ANET_PRD061_OUTPUT`）へ流す。採取した入力ごとの JSON を比べ、`typed`（読んだ設定）と `values`（解決済みの入力）が、次の設定キーの行を除いて一致する。除くのは、DQN の 3 つの接頭辞と Rainbow の `head.init` の下の `nonlinearity` と `variance_scale` である。理由は、本 PRD が足す行だからである。比べる範囲は、採取が出す入力の全部（[manifest.json](../../src/core/base/testdata/rfd0072/manifest.json) の入力のうち、テストが飛ばす ImageCls と GridMaze_muzero を除いたもの）である |
| 11 | block と RND の既定（D9・D21） | テストで、初期化の設定を書かない設定から作った block に forward を 1 回通したあとの重みが、P23 の既定の方式の形である。重みの数が 8,192 個以上になる大きさで作る。ResBlock の 2 つ目の conv（`init2`）はすべて 0 である。ResBlock の 1 つ目の conv（`init1`）と、入力と出力の channel の数が違うときに作られる shortcut の conv（`init_ds`）、Conv1d・Conv2d は、標本の分散が 2/fan_out の ±5% の内にある。CNBlock の 3 つの層は、標本の標準偏差が 0.02 の ±5% の内にある。Linear は、重みの絶対値の最大が √(6/(fan_in + fan_out)) の (1 + 1e-6) 倍以下で、標本の分散が 2/(fan_in + fan_out) の ±5% の内にある。RND は、初期化の設定を書かない設定を読むと、Head の方式が `xavier`、`nonlinearity` が `relu`、`variance_scale` が 1.0 である |

## 10. グリルの材料（2026-11-21）

開発者の発言と選択、AI の案を分けて書く。この節は決定ではなく、決定は §6 にある。

**開発者の選択**

- ゴールと非ゴールは、起票時の G1〜G3 のままでよいとした。説明が要る前提の番号は挙げなかった。
- G2 は、受入を「流せる所まで」にする案を選んだ（D1）。
- 方式の名前と倍率は、Keras の語（`variance_scaling` と `variance_scale`）の案を選んだ（D2）。
- 読み方は、AI が推奨した「14 か所に 1 行ずつ足す」ではなく、「読み取りの口をこの PRD で作る」を選んだ（D5）。選択肢には、block の既存キーの止まる時機と文が変わること、`nonlinearity` の差を追加で決めること、読み取りの口の課題を分けることになることを書いてあった。
- `nonlinearity` は「揃える」を選んだ（D7）。
- 細部 11 件は「推奨で埋める」を選んだ（D11〜D21）。
- ステージ 60・61 の結果（P14）は、はじめ「先にそろえる」を選んだ。そのあと G2 の段取りを Atari の枠で決めるとしたので、この結果で変わる本 PRD の決定は無くなった。AI がそれを示して聞き直し、開発者は「起草を進め、レビューに出す前に終わっていれば結果を足す。終わっていなければ結果待ちと書いて仕上げる」を選んだ。

- 起草のあと、ゴールに G4 を足すことを確かめられ、「ゴールを変える」を選んだ。
- レビューのあとの起票の候補 4 件から、読み取りの口の分の課題の起票、読み取りの口の課題への追記、PRD194 の順序の節への 1 行の追加の 3 件を選んだ。ステージ 60・61 の結果を本 PRD に足す候補は、見送った（§13）。

**AI の案（起草で足したもの）**

- `nonlinearity` の未知の値を ERROR にする D8 は、グリルの問いには無く、起草で P22 に気づいて足した。D7 で新しく読む設定キーになるので、規約から決まると判断した。起草のあとに開発者へ示し、問いに戻す指定は無かった。
- G4 は、読み取りの口を作る選択を受けて起草で足した。

**別モデル（Opus）の検討（起草のあと）**

会話を知らない別モデルに草案を読ませ、12 件の論点を受けた。結論への反対は 1 件で、開発者が選んだ決定への反対は無かった。開発者に聞く論点（ゴール・優先度・許容する不利益の選択）は無いと判断し、次のとおり扱った。

| 論点 | 扱い |
|---|---|
| 価値の分解の Head で、`nonlinearity` を数値と同じ時点で検査すると、列挙を先に確かめる順（P21）に反する | 直した。列挙の検査を方式と同じ時点にした（D8、§8.1）。起草で AI が置いた時点の誤りで、開発者の選択ではない |
| 今ある 2 つの口は、口の中で検査している。「口は読むだけ」は、その形と違う | 直した。入口を 2 つにした（D5、§6.2）。「口は読むだけ」は、グリルで AI が P18 から導いたとして示した事項で、P18 の読みが不正確だった。外から見える挙動（止まる時機、文、検査の順）は変わらない |
| Run フォルダの `config/config_data.txt` は入力の写しで、読んだことを示せない | 直した。受入 9 の確かめ先を分け、P27 を足した |
| 「config dump」が何を指すかが決まっていない | 直した。§8.2 と D15 を、外に出る記録の名前で書いた（P27） |
| 既存の挙動の確かめが、口に替える読み手の一部しか通らない | 受入 10・11 を足した（D21） |
| 方式を書き忘れて倍率だけを書くと、黙って別の方式で流れる | 決定は変えず、D13 の理由に、設定の記録で確かめることを足した |
| block 名が文に出ない理由が、前例だけでは足りない | 非ゴールと D10 の理由を直した。PRD194 が扱う（P30） |
| `GetNonlinearityType` を止めても、検査のあとでは届かない | クラス一覧に、念のための止めと書いた。受入では断定しない |
| 幅の端の断定は、丸めで 1e-6 ほど超えうる | D20 と受入 2 に余裕を足した |
| 前提の表の抜け（今ある口の形、Linear と RND の既定、入力の写し、使わない block） | P18・P23 を直し、P27・P28 を足した |
| 非ゴールを外しても、読み取りの口を作らない案より単純な案は無い | 変えない |
| §9.1 の seed のぶれの目安が 1 対だけによる | §9.1 に、Atari の枠で決め直すと足した |

## 11. ほかの PRD・課題との順序

- **読み取りの口の課題の残り**（[issue-20261111-cdc102](../management/issues/issue-20261111-cdc102.md)。GraphViz の設定、観測の正規化の設定、方策の検査の置き場。PRD は無い）: 本 PRD が先でも後でもよい。本 PRD が先なら、残りは同じ口の形（P18）で足せる。残りが先に起こされるなら、初期化の設定は本 PRD が持つので、そちらの範囲から外す。
- **[PRD150](0150_residual_alpha_gate_10rfd.md)**（未実装。ResBlock を触る）: どちらが先でもよい。本 PRD が先なら、PRD150 は ResBlock の初期化の設定を口で読む形の上に足す。PRD150 が先なら、本 PRD は PRD150 が足した設定キーに触らず、初期化の 3 つの接頭辞だけを口に替える。
- **[PRD169](0169_outside_core_extension_layout_10rfd.md)**（未実装。`WeightInitializer` の置き場を扱う）: 中身は変えないので、どちらが先でもよい。本 PRD が先なら、PRD169 は口も一緒に動かすかを決める。
- **[PRD183](0183_temporl_skip_followups_10rfd.md)**（未実装。skip の Head の複数の射影を `skip.head.init.*` で初期化する）: どちらが先でもよい。PRD183 が `skip.head.init.*` の読み手を足すなら、本 PRD のあとでは口で読む。
- **[PRD194](0194_config_key_full_names_in_messages_10rfd.md)**（未実装。エラー・WARN・INFO が示す設定キーを完全キーにする。3 つのステージを持つ。P30）: どちらが先でもよいが、扱いが違う。
  - 本 PRD が先の場合: block の検査の文は、D10 のとおり相対の設定キーで書く。PRD194 のステージ 3 が、block のほかの検査と一緒に完全キーへ直す。口の読んで検査する入口は、文の設定キーを実行時に接頭辞から組み立てるので、PRD194 の集計（文の中の名前を数える）には現れない。PRD194 の実装の計画に、この口と `ValidateWeightInitConfig` を足す必要がある。
  - PRD194 が先の場合: 本 PRD が足す文は、PRD194 の規約（完全キー）で書く。block の検査の設定キーは、読み口の記録から引く。§8.1 の block の行と §9.2 の 4 の、block の文の期待（`init1.variance_scale`）は、完全キー（`net.block.[<名前>].init1.variance_scale` の形）に読み替える。
  - PRD194 のステージ 2（完全キーを運ぶ仕組みを足す段階）の途中には、本 PRD を入れない。初期化の設定の型に欄と口を足すと、仕組みの取り直しが要る（PRD194 の §6.7）。PRD194 のステージの切れ目と、ステージ 3 の途中には入れてよい。ステージ 3 の途中に入れた場合は、PRD194 が、本 PRD の足した文を同じステージで直す。
- PRD194 のほかの PRD は、段階を持たない実装として並ぶ。本 PRD の実装の途中にほかの PRD が入る場合は、§9.2 の 5 の検索を取り直し、足された読み手を口に替えてから終える。

## 12. issue の対応表

読み取りの口の課題は、初期化・GraphViz・観測の正規化の 3 つの型を持つ。本 PRD が扱うのは初期化の分だけなので、その分の課題を、読み取りの口の課題を親にして起こした（表の 2 行目）。

| ID | 題 | area | 根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261121-533820](../management/issues/issue-20261121-533820.md) | 重みの初期化に、分散を倍率÷fan_in にする一様分布の方式が無い | 315_nn, 324_dqn | [nn_head.hpp](../../src/core/base/include/anet/nn_head.hpp) 22〜77 行、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 735〜764 行 | D1〜D4・D11〜D14・D16・D17・D19〜D25 | 1・2・3・8・9 |
| [issue-20261121-d4feb6](../management/issues/issue-20261121-d4feb6.md) | 初期化の設定（WeightInitConfig）を 14 か所が 1 キーずつ読み、読み方と不正な値の止まり方が読み手ごとに違う | 316_config, 315_nn, 324_dqn | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 1678〜1700 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 216〜222 行、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 28〜34 行、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 565〜571 行 | D5〜D10・D15・D18・D21 | 3・4・5・6・7・10・11 |

## 13. 未決

| # | 事項 | 扱い |
|---|---|---|
| U1 | 初期化の設定の読み取りの口の分の課題（§12） | 起こした。[issue-20261121-d4feb6](../management/issues/issue-20261121-d4feb6.md) |
| U2 | ステージ 60・61 の結果（P14）が本 PRD に無い | 本 PRD の決定には効かない。Atari の枠で 5M の Run の段取りを決めるときに、実験記録へ書く（§9.1） |
| U3 | PRD194 の「ほかの PRD との順序」に、本 PRD との順序（§11）が書かれていなかった。PRD194 の集計は、口が実行時に組み立てる文を拾わない | 直した。PRD194 の §6.7 に、本 PRD との順序の行を足した（2026-11-21） |

## 14. 参照

- Dopamine e7d780d（2019-12-19）: [atari_lib.py](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py)、[baselines/data/breakout.json](https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/baselines/data/breakout.json)。JAX 版: [jax/networks.py](https://github.com/google/dopamine/blob/master/dopamine/jax/networks.py)。
- [TensorFlow の VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling)、[JAX の variance_scaling](https://docs.jax.dev/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html)、[torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html)、[torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)、[torch.nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。
