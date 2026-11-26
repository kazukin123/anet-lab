# Survey: TorchRL の設計と現状 ― Transforms、実装の参考、実験結果の比較元

## 元の材料

[Survey: TorchRL の設計と現状 ― Transforms、実装の参考、実験結果の比較元](../../../../reports/torchrl_survey_2026-11-20.md) blob:5d16433927d0b8ad64bc58a77b63c655811fc806

## 要点

- TensorDict は、先頭の形（batch size）を共有するテンソルの辞書である。インデックスや reshape は全エントリーの batch の次元に一度にかかる。値に別の TensorDict を入れられ、キーは `td["a","b"]` のように入れ子にできる。【公式コード】[tensordict@v0.15.6 (2026/11), `tensordict/_td.py` L251–289]
- TransformedEnv の step は、`inv` で入力を逆変換し、base env の step を実行し、`_step` で出力を変換する、の順に進む。Compose は順変換を前から、逆変換を後ろから回す。v0.15.0 では `inv` の docstring が「`_inv_call` を呼ばない」と書く一方、コードは呼んでいる。【公式コード】[rl@v0.15.0 (2026/11), `_base.py` L671–691・L1373–1422]
- ReplayBuffer に付けた Transform は、読み出し（`sample`・`__getitem__`）で順変換が、書き込み（`extend`・`add`・`__setitem__`）で逆変換（`inv`）が呼ばれる。`transform` 引数の説明も「`sample` のときに実行する」と書いている。`add` は Transform が時間の次元を期待するため、一時的に次元を足す。【公式コード】[rl@v0.15.0 (2026/11), `torchrl/data/replay_buffers/replay_buffers/base.py` L216–224・L1772–1794・L2265–2273・L2413–2455]
- TorchRL の調査対象には、anet の C++ アプリがそのまま呼べる C++ API はなかった。
- 版を固定した公式コードから読める機能と、全ライブラリを同条件で測った速度の証拠とは別である。

## 内容

### 2. 設計の全体像

依存は PyTorch・NumPy・tensordict に絞り、環境・分散・ロガーは任意の依存にしている。アーキテクチャやデータ収集の設定で、偏った選択はしない方針である。番号を振った「設計原則」の一覧は論文に無い。【論文】[TorchRL論文 (2023/11), 付録 I]

制約は 3 つある。batch size は推論せず既定は `[]`、`device` を指定すると全エントリーをその device へ移す、キーは文字列か文字列のタプルに限る。【公式コード】[tensordict@v0.15.6 (2026/11), `_td.py` L291–301]

v0.15 の spec のクラスは Binary、Bounded、Categorical、Composite、MultiCategorical、MultiOneHot、NonTensor、OneHot、Stacked、StackedComposite、Unbounded（continuous・discrete）などである。【公式文書】[TorchRL文書 (2026/11), `data_specs.rst`]

### 3. Transforms

下位: `transform_observation_spec`・`transform_reward_spec`・`transform_done_spec`・`transform_action_spec`・`transform_state_spec`。既定は恒等写像である。

公式チュートリアルは、環境側の Compose と、`("next", ...)` のキーを足した同じ Compose を ReplayBuffer に付け、重ねたキーを除いて保存し、抽出した結果がオンラインの結果と一致することを確かめている。理由には、メモリの消費と「学習時と推論時で窓の長さが違う場合」を挙げる。窓の切り出しは関数 `functional.cat_frames` としても公開されている。【公式文書】[rl@v0.15.0 (2026/11), `tutorials/sphinx-tutorials/rb_tutorial.py` L719–791]、[TorchRL文書 (2026/11), `envs_transforms.rst` L350–366]

並列 env では、各プロセスの中で変換してもメインプロセスで変換しても結果は同じとされる。ただし ActionMask は並列 env 自体に付ける。[rl@v0.15.0 (2026/11), `tutorials/sphinx-tutorials/torchrl_envs.py` L651–683]、[TorchRL文書 (2026/11), `envs_transforms.rst` L410–412]

### 4. Collector と ReplayBuffer と損失

LossModule は学習するパラメータを持つ部品で、入出力とも TensorDict、損失は `"loss_<名前>"` のキーで返す。【公式文書】[TorchRL文書 (2026/11), `objectives.rst`]

DQNLoss の主な既定は `loss_function="l2"`、`delay_value=True`（target ネットを持つ）、`double_dqn=False` で、既定の value estimator は TD0 である。DistributionalDQNLoss は離散の支持点上の分布を出すネットに使う。value estimator には TD0、TD1、TDLambda、GAE、VTrace、MultiAgentGAE がある。【公式コード】[rl@v0.15.0 (2026/11), `torchrl/objectives/dqn.py` L35–63・L166–193・L376–400・`objectives/utils.py` L53–69]

target の更新は `SoftUpdate`（Polyak 平均）と `HardUpdate`（`step()` を N 回呼ぶごとにコピー）である。v0.15.0 のリリースノートは、HardUpdate が N+1 回ごとにコピーしていた誤りを直したと書いている。【公式コード】【公式リリース】[rl@v0.15.0 (2026/11), `objectives/utils.py` L589–679]、[TorchRL v0.15.0リリース (2026/11), Training correctness]

### 5. 収録している手法

環境は ALE/Pong-v5（gymnasium）。スクリプトは `repeat_action_probability` を指定していない。ALE の文書では v5 の既定は 0.25 である。【第三者（ALE 文書）】[ALE文書 (2026/11), Environments の表]

NoisyLinear は、自作の Rainbow で評価時の成績が悪いという報告（Issue #3069、2025-07）を受け、2026-01 の修正でノイズの有無を exploration type で決めるように変わり、`std_init` の既定も 0.1 から 0.5 へ見直された。【公式 Issue・コード】[pytorch/rl Issue #3069 (2025/07)]、[pytorch/rl commit 870ee20a (2026/01)]

内的報酬の部品は RND だけで、v0.14.0 で入った。参照実装は MuJoCo の PPO＋RND だけである。【公式コード】【公式 PR】[rl@40407917 (2026/11), `objectives/rnd.py` L19–60・`envs/transforms/rnd.py` L80]、[TorchRL v0.14.0リリース (2026/09)]

### 6. 公開している実験結果

論文の Table 1 は、オンラインの単一エージェントの最終報酬を 5 seed の平均±標準偏差で示し、元論文の結果とだけ比べている。Atari は A2C・PPO が 40M フレーム、IMPALA が 200M フレーム、MuJoCo（v3）は 1M ステップである。【論文】[TorchRL論文 (2023/11), §4・Table 1]

Atari の条件のうち、frameskip 4 と DeepMind の前処理に従うことは付録に書かれている。一方、環境の版、sticky actions の有無、評価の手順（テストのエピソード数、「最終報酬」の定義）は本文と付録で確かめられなかった。DQN は実験の節に無く、付録の損失の使用例にだけ出る。【論文】[TorchRL論文 (2023/11), 付録 G.1・付録 E]

offline（Table 2）は IQL、DT、oDT を D4RL で 50,000 回更新した結果で、seed の数（5 と 3）とデータセット名（medium-replay と medium）が表の説明と付録の図で食い違う。マルチエージェントは VMAS の 3 課題で、RLlib との比較は図と文だけで数値の表は無い。【論文】[TorchRL論文 (2023/11), Table 2・Figure 2・Figure 13・Figure 15]

### 7. 実時間の性能

論文の Table 5 は、データの収集速度で TorchRL の collector がほかのライブラリを上回ると示す。条件は AWS の単一ノード（CPU 96 コア、Run ごとに A100 1 枚）、worker 32 で、単位は本文で「data collection speed」とだけ書かれる。【論文】[TorchRL論文 (2023/11), §4・Table 5]

DreamerV3（v0.14.0）では、GB200 で 1 更新の中央値が、本家 JAX 22.04 ms に対し、TorchRL の compiled scan だけでは 358.51 ms（16.3 倍遅い）、CUDA graph を足すと 17.83 ms（1.24 倍速い）である。replay の抽出と環境の収集は除く。【公式リリース】[TorchRL v0.14.0リリース (2026/09), Measured performance]

Issue #1325（2023-06）: オクラホマ大の研究員が、PongNoFrameskip-v4・並列 8〜16・ランダム方策で、TorchRL 約 2,600 FPS、Gym AsyncVectorEnv 約 11,000 FPS と報告した。EPFL の博士課程の利用者の追試でも 2,866 対 11,588 だった。[pytorch/rl Issue #1325 (2023/06)]

### 8. C++ から使えるか

tensordict の C++ 拡張は廃止: PR #1916（2026-10）で `tensordict/csrc` を Python に置き換え、いまは純 Python の wheel である。互換用の `tensordict._C` は 0.17 で消える予定。[tensordict PR #1916 (2026/10)]、[tensordict README (2026/11), Install]

libtorch だけの C++ から使う道: C++ 向けの API・文書・CMake のパッケージは見当たらず、Issue の検索でも、libtorch の話題は import エラーの報告と依存の更新だけだった。公式が示すのは、学習済みの方策を書き出す道（torch.export → AOTInductor の C++ バイナリ、ONNX Runtime、v0.15.0 の `export_onnx_policy`）である。[rl@40407917 (2026/11), `tutorials/sphinx-tutorials/export.py` L329–359]、[TorchRL v0.15.0リリース (2026/11), Highlights]

### 9. 採用状況と方向

リリースは頻繁で、2025-11-20 からの 1 年で TorchRL は 10 本、main へのコミットは約 991 件だった。tensordict は 2026-10-15 から 2026-11-11 の間に v0.15.0〜v0.15.6 の 7 本を出した。【公式リリース】[GitHub API (2026/11)]

コミット数の最多は vmoens（TorchRL 1,277、tensordict 544）だが、取得時点の最終コミットは TorchRL で 2026-09-19、tensordict で 2026-09-18 だった。[GitHub API (2026/11)]

tensordict では 2026-08-03 に初コミットした Peter Sharpe（GitHub の自己記載は NVIDIA、tensordict の保守者）が 1,034 コミットで最多である。[GitHub API (2026/11)]

### 10. 批判と懸念

`ReplayBuffer.extend` と `sample` を続けて compile すると eager の約 5 倍遅い、という報告（Issue #2501、2024-10）がある。PR #2529 で試験の範囲では eager と同等以上になったが、Issue は開いたままである。[pytorch/rl Issue #2501 (2024/10)]、[pytorch/rl PR #2529 (2024/11)]

LossModule・TensorDictSequential・ProbabilisticActor が compile できないという報告（Issue #2312、2024-07）は、応答が無いまま開いている。[pytorch/rl Issue #2312 (2024/07)]

### 11. anet との対応（筆者の整理）

RND: 参照実装は PPO＋RND の HalfCheetah だけで、学習結果の数字が無い（§5.3）。

比較元の候補は §6.4 の表にある。どれも条件を個別に確かめる必要がある。anet の Atari の既定のプロトコルは `v5_noop0`（sticky 0.25、noop_max 0、episodic_life false。220）である。

### 12. 総合評価

実装の参考: ReplayBuffer の分け方、PER の CUDA 実装、Transform の spec の規則、重みの同期は読む価値がある。C++ からは使えず、DQN 系は手薄で、API の変化も速い（§8、§5.2、§4.4）。

性能: 論文は、収集の速度でほかのライブラリを上回ると示す。第三者は、CPU の単純な環境では gym より遅いと報告している。compile と CUDA graph の効果は、公式で 1〜4 倍、派生の LeanRL で 2.7〜6.8 倍とされる（§7）。

### 13. 調査の限界

論文の Atari の条件（環境の版、sticky actions、評価の手順）は見つからなかった。ICLR の OpenReview の査読コメントは読んでいない。

Reddit は検索できず、Hacker News に TorchRL 単独のスレッドは見つからなかった。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
