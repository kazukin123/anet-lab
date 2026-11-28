# Survey: 外的と内的の価値を分ける位置と、内的の n-step の和をエピソードの境でどう扱うか

Date: 2026-11-27
Scope: 外的報酬と内的報酬の価値を分ける手法（Agent57、MEME、Burda らの RND、NGU）が、Network のどこから分けているかと、その理由を、論文の本文・付録・図と公開実装のコードで確かめる。あわせて、価値ベースで内的の割引率や終端の扱いを外的と分ける公開実装が、n-step の和と Replay の系列をエピソードの境でどう扱うかを、コードで確かめる。内的報酬の手法の全体像は [内的報酬の手法の全体像](intrinsic_reward_landscape_survey_2026-11-18.md)、Replay に入れた内的報酬の論点は [エピソード内の新しさを価値ベースに載せる論点](episodic_bonus_value_based_survey_2026-11-18.md) にあり、本冊はそこで扱っていない部分（分ける位置、公開実装の n-step と終端）に絞る。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。著作権への配慮から逐語の引用は 1 か所（15 語未満）にとどめ、ほかは要約と位置（節・付録・表・図、実装はファイルと関数）で示す。論文は ar5iv の HTML 版と arXiv の PDF（Agent57 の図 9・10 は PDF の画像）を読み、実装は公開リポジトリの今の版を取得して読んだ（版は §6）。「読み取り」と書いた文は筆者の推論で、原典の主張ではない。

## 目次

1. [1. 問いと結論](#1-問いと結論)
2. [2. 外的と内的の価値を分ける位置と理由](#2-外的と内的の価値を分ける位置と理由)
   - [2.1 Agent57](#21-agent57)
   - [2.2 MEME](#22-meme)
   - [2.3 Burda らの RND の論文と公開実装](#23-burda-らの-rnd-の論文と公開実装)
   - [2.4 ほかの公開実装](#24-ほかの公開実装)
   - [2.5 同じ条件での比較](#25-同じ条件での比較)
3. [3. 価値ベースでの割引率・終端・n-step の扱い](#3-価値ベースでの割引率終端n-step-の扱い)
   - [3.1 RND の公開実装と CleanRL の GAE（on-policy の参照）](#31-rnd-の公開実装と-cleanrl-の-gaeon-policy-の参照)
   - [3.2 価値ベースの公開実装の n-step と終端](#32-価値ベースの公開実装の-n-step-と終端)
   - [3.3 Replay の系列がエピソードの境をまたぐ実装](#33-replay-の系列がエピソードの境をまたぐ実装)
   - [3.4 つなぐ・切るを比べた実験](#34-つなぐ切るを比べた実験)
4. [4. 批判と懸念](#4-批判と懸念)
5. [5. 総合評価](#5-総合評価)
6. [6. 調査の限界](#6-調査の限界)
7. [7. 出典一覧](#7-出典一覧)

## 1. 問いと結論

**問い 1**: 外的報酬と内的報酬の価値を分けるとき、出力の Head だけを分ける形（胴体を共有）と、Network ごと分ける形を、同じ条件で比べた事例はあるか。それぞれを採る代表の手法（RND の PPO、NGU、Agent57、MEME など）は、どこから分けているか（最終の層だけか、もっと手前か）。分ける位置の理由は何と書かれているか。

**結論 1**:
- 2 つの形を同じ条件で比べた事例は見つからなかった。Agent57 は「2 つの Network」と「合わせた報酬の 1 つの Network（NGU）」を比べ、MEME は torso を共有したまま「Head 2 つ」と「Head 1 つ」、「合わせた損失」と「別々の損失」を比べた（§2.5）。
- 分ける位置は手法ごとに違う。Agent57 は畳み込みから Dueling の Head まで全部を 2 組持つ。MEME は torso と LSTM を共有し、Head ごとに隠れ層 1024 を 2 つ持つ。Burda らの公開実装は価値側の層まで共有し、最後の 1 出力の全結合だけを分ける。NGU は分けない（§2.1〜§2.4）。
- 理由は、Agent57 が「報酬ごとの尺度と分散・optimizer の状態を分ける」、MEME が「Network ごとは計算が重く、低い層の特徴を共有できない」である。Burda らは Head を分ける理由を書くが、分岐の位置の理由は書かない（§2.1〜§2.3）。

**問い 2**: 価値ベース（DQN 系、n-step の収益）で、内的報酬の割引率を外的と分けたとき、また内的を非エピソード的（エピソードの終わりで切らない）にしたとき、n-step の和をエピソードの境でつなぐか切るかを、どう実装しているか（公開実装で確かめる）。つなぐ・切るを比べた実験はあるか。Replay の系列がエピソードの境をまたぐ実装はあるか。

**結論 2**:
- 読んだ価値ベースの公開実装は、どれも n-step の和をエピソードの終わりで止め、次のエピソードの報酬を足さない。内的を終端で切らない形は Unity ML-Agents にあり、RND と curiosity の信号だけ done で bootstrap を切らないが、目標値は 1 step で、終端の step の観測から bootstrap する（§3.2）。DI-engine の NGU は done を無視する設定を持つが、外的と内的を分けずに目標値の全体に掛かる（そのときに系列がエピソードをまたぐかは確かめていない）。
- 公開の Agent57 の再実装（deep_rl_zoo）は 2 つの Head を同じ終端の mask で切り、Taïga らの Rainbow＋RND は内的を足した報酬を Dopamine の n-step（最初の終端で止まる）に渡す（§3.2）。
- 価値ベースでつなぐ・切るを比べた実験は見つからなかった。比べた例は Burda らの PPO だけである（§3.4）。
- Replay の系列がエピソードの境をまたぐ実装はある。Acme の系列の adder は次のエピソードへ続ける選択肢を持ち、deep_rl_zoo の系列の切り出しは既定でまたぐ（IMPALA が使う）。R2D2・NGU・Agent57・MEME の論文と、Acme・deep_rl_zoo の R2D2 系の設定はまたがない（§3.3）。

**調べた範囲と日付**: 2026-11-27 に、Agent57・MEME・RND・BYOL-Explore・Temporally-Extended ε-Greedy の論文（ar5iv と arXiv の PDF）と、RND の公開実装、CleanRL、deep_rl_zoo、Taïga らの公開実装、Dopamine、DI-engine、Unity ML-Agents、Acme のコードを読んだ。web 検索は 6 回（§6）。

**見つからなかったこと**: Head だけ分ける形と Network ごと分ける形を同じ条件で比べた事例。価値ベースで内的の n-step の和をエピソードの境でつなぐ実装と、つなぐ・切るを比べた実験。Agent57 の原典の実装（公開されていない）。

## 2. 外的と内的の価値を分ける位置と理由

### 2.1 Agent57

- **位置**: 外的と内的は、同じ構造の別々の 2 つの Network で、重みは重ならない（θ = θ^i ∪ θ^e）。図 9・10 では、2 つの Network がそれぞれ畳み込みの torso（32・64・64 チャンネル）、全結合 512、LSTM 512、Dueling の Head（全結合 512 を 2 本に分けて 1 出力と 18 出力）を持ち、共有する層は無い。2 つの Network の入力は、どちらも観測と、前の行動・前の外的報酬・前の内的報酬・方策の番号 j である。[Badia Agent57 (2020/03), §3.1、付録 E Learner、付録 F 図 9・10]
- **学び方**: 2 つは同じ系列から、外的報酬と内的報酬で別々の transformed Retrace の損失を作り、target 方策は合わせた Q の greedy で共通にする。単純な勾配降下のもとでは、合わせた報酬で 1 つの Q を学ぶのと目的が等価になる（付録 B）。変換 h を使うかどうかは成績に大きく効かなかったとして、恒等の分け方を使う（付録 H.3）。[Badia Agent57 (2020/03), §3.1、付録 B・C・H.3]
- **理由**: 各 Network が自分の報酬の尺度と分散に合わせられることと、optimizer の状態を分けられることを挙げる。§4.2 では、内的報酬の重みを上げると、NGU の 1 つの Network は方策の番号 j ごとに大きく違う値を表す必要があり、より柔軟な表現が要る、と説明する。この効果は、コインを取ったらエピソードが終わる設定だから出るもので、終わらなければ探索と活用の方策は似たものになる、とも書く。[Badia Agent57 (2020/03), §3.1、§4.2、図 5]
- **比較の条件**: 結果は 3 seed（Agent57 だけ 6 seed）の平均で、50 エピソードの窓の平均、学習中の最大を最終値とする。10 本の challenging set で、分割を外すと 20% を超えて下がった。[Badia Agent57 (2020/03), §4、§4.2、図 4]

### 2.2 MEME

- **位置**: 外的と内的の Q で、NFNet 系の torso と LSTM（隠れ 1024）を共有し、LSTM の出力（とその入力）を Head へ渡す。Head は外的と内的に Dueling の Head を 1 つずつ持ち、それぞれ隠れ層 1024 を 2 つ持って、行動の数 × 方策の組の数（16）を出す。方策の Head は別の MLP である。[Kapturowski MEME (2022/09), §4 C2、付録 B.1〜B.4、付録 D、図 2]
- **理由**: Agent57 の別々の Network は計算が重く、一般的な低い層の特徴を共有できないので標本効率を落としうる、として torso を共有する。行動を決めるのは合わせた Q なので、損失も合わせた Q に掛ける（合わせた損失）。そのため外的と内的の分解は厳密な意味を持たなくなるが、β を掛ける形の暗黙の帰納バイアスは残る、と書く。[Kapturowski MEME (2022/09), §4 C2・式 6〜7]

torso を共有して Head を分けた、と原典は次のように書く。

```
a shared torso for the two Q-functions while retaining separate heads

2 つの Q 関数で torso を共有し、Head は別々に保つ
```
[Kapturowski MEME (2022/09), §4 C2]

- **Head 1 つとの比較**: 合わせた値 Q_e + βQ_i を 1 つの Head で推定しても、提案の形と同程度だった。原典は、分けた Head の帰納バイアスは、安定化のために足したほかの部品のもとではもう重要でない、と結論する。この節は games・seed・フレーム数を書かない。[Kapturowski MEME (2022/09), 付録 H.4、図 7]
- **エピソードの境**: Replay の系列はエピソードの境をまたがない。割引率は方策の組ごとに 1 つで、合わせた報酬に掛ける（最大 0.9997、最小 0.97）。外的と内的で別の割引率は書いていない。[Kapturowski MEME (2022/09), §3、付録 D・E、表 2]

### 2.3 Burda らの RND の論文と公開実装

- **論文**: 収益は報酬について線形なので、外的と内的の価値を別々に学んで足せる。Head を分けると、割引率と終端の扱いを報酬ごとに変えられる。非定常な内的報酬に対して、価値関数への追加の教師信号にもなる、と書く。両方をエピソード的にした設定では、Head 2 つは 1 つより良くならず、1 つのほうがわずかに良かった（図 4、5 seed、γ_E = γ_I = 0.99）。外的の割引率を 0.999 にすると大きく良くなり、内的を 0.999 にすると悪くなった（図 5、10 seed）。表 5 の既定は γ_E 0.999・γ_I 0.99 である。[Burda RND (2018/10), §2.3、§3.2〜3.3、図 4・5、付録 A 表 5・表 6]
- **論文が書かないこと**: 付録 A.5 の方策の構造は、畳み込みの encoder と、RNN 版の GRU を書くが、2 つの価値の Head がどこで分岐するかを書かない。分岐の位置の理由も書かない。[Burda RND (2018/10), 付録 A.5]
- **公開実装（既定の RNN 版、cnn_gru_policy_dynamics.py の apply_policy）**: 畳み込み 3 層・全結合・GRU を方策と価値で共有する。そのあと、価値側だけに残差の全結合（fc2val）を 1 つ通し、方策側には別の残差の全結合（fc2act）を通す。外的と内的の価値（vf_ext・vf_int）は、fc2val の出力に、それぞれ 1 出力の全結合を 1 つずつ載せたもので、2 つの価値が別に持つのはこの最後の層だけである。CNN 版（cnn_policy_param_matched.py）も、GRU が全結合 448 に替わるだけで同じ分け方である。[RND 公開実装 (2018/11), policies/cnn_gru_policy_dynamics.py・cnn_policy_param_matched.py]

### 2.4 ほかの公開実装

- **CleanRL（ppo_rnd_envpool.py）**: 畳み込みと全結合 256・448 を方策と 2 つの価値で共有し、2 つの価値だけが共有する全結合 448（extra_layer）を通した出力に入力を足して、外的と内的の 1 出力の全結合（critic_ext・critic_int）を 1 つずつ載せる。方策は別の 2 層を持つ。RND の公開実装と同じく、2 つの価値が別に持つのは最後の層だけである。[CleanRL ppo_rnd_envpool (2023/11), Agent クラス]
- **Unity ML-Agents**: 報酬の信号（外的、curiosity、RND、GAIL など）ごとに、共有の Body の上へ 1 出力の線形の Head を 1 つずつ置き（ValueHeads）、割引率も信号ごとに設定から読む。SAC の Q1・Q2 は、それぞれ Body を 1 つ持ち、その上に信号ごとの Head を持つ。[ML-Agents (2026/10), torch_entities/decoders.py の ValueHeads、networks.py の ValueNetwork、sac/optimizer_torch.py]
- **deep_rl_zoo の Agent57（公開の再実装。DeepMind の実装ではない）**: Agent57Conv2dNet は NGU と同じ Network（畳み込みの torso、LSTM、Dueling の Head）を外的用と内的用に 2 つ作り、層を共有しない。同じリポジトリの NGU の Network は Q を 1 つだけ出す。[deep_rl_zoo (2024/02), networks/value.py]
- **NGU（原典）**: 外的と内的を分けず、32 個の (β, γ) の組を 1 つの Network で表す。Breakout と Beam Rider で R2D2 に届かない理由を、内的信号で学んだ表現が活用側に干渉するためと推測し、表現を共有しなければ改善するという仮説を述べる。この 2 点は [エピソード内の新しさを価値ベースに載せる論点](episodic_bonus_value_based_survey_2026-11-18.md) §2.4・§4.2 に原典の位置つきで書いてあり、本冊では読み直していない。

### 2.5 同じ条件での比較

- Agent57 の分割の ablation は、2 つの Network（Agent57 の形）と、合わせた報酬で学ぶ 1 つの Network（NGU）の比較で、torso を共有して Head だけを分ける形は比べていない。原典は、torso を共有する形を代わりの案として論じていない。[Badia Agent57 (2020/03), §3.1、§4.2、図 4・5、付録 H.1]
- MEME の ablation は 8 本（Frostbite、H.E.R.O.、Montezuma's Revenge、Pitfall!、Skiing、Solaris、Surround、Tennis）・3 seed・1B フレームで、学習曲線の下の面積をゲームごとの最大で割って比べる。合わせた損失は、Agent57 の別々の損失と比べた（図 16）。Head 1 つの比較は §2.2 のとおりである。torso の構造の比較（図 22）と Head の構造の比較（図 24）もあるが、torso を共有する形を外的・内的で別々の Network に戻す比較は無い。[Kapturowski MEME (2022/09), §5.2、付録 H.4・L、図 16・22・24]
- Burda らの Head 2 つと 1 つの比較（§2.3）は、Head の数の比較で、Network ごとに分ける形は入っていない。[Burda RND (2018/10), §3.2、図 4]
- 検索で、Head だけと Network ごとを同じ条件で比べた論文や再現は見つからなかった（§6）。

## 3. 価値ベースでの割引率・終端・n-step の扱い

### 3.1 RND の公開実装と CleanRL の GAE（on-policy の参照）

- **RND の公開実装（ppo_agent.py の update）**: 内的の GAE は、`use_news` が偽（run_atari.py の既定は 0）のとき次の step の終端の印を 0 とみなし、エピソードの境で bootstrap も GAE の累積も切らない。外的は終端の印で切る。割引率は内的が `gamma`、外的が `gamma_ext` で、run_atari.py の既定はどちらも 0.99、README の実行例は `--gamma_ext 0.999` を付ける。λ は 0.95、advantage は内的 1・外的 2 の係数で足す。rollout は並列 env の連続した step なので、内的の収益は env のリセットをまたいで続く（読み取り: GAE の和は rollout の長さ 128 の中で取り、その先は価値で bootstrap する）。[RND 公開実装 (2018/11), ppo_agent.py の update、run_atari.py、README；Burda RND (2018/10), 付録 A 表 5]
- **CleanRL**: 内的の `int_nextnonterminal` を常に 1.0 にし、外的は done で切る。既定は外的 0.999・内的 0.99 である。[CleanRL ppo_rnd_envpool (2023/11), GAE の計算]
- BYOL-Explore は V-MPO（on-policy）で、Atari の内的報酬をエピソード的に扱い、非エピソード的な RND の公開値と比べるとより難しい設定だと書く。報酬は外的と内的の線形和の 1 本で、割引率は 1 つである。[Guo BYOL-Explore (2022/06), §4.1・§4.2、付録 C]

### 3.2 価値ベースの公開実装の n-step と終端

- **deep_rl_zoo の Agent57**: 外的と内的の 2 つの Network の損失を、同じ関数（_calc_retrace_loss）で作る。その関数は、どちらの損失にも合わせた報酬（外的＋β×内的）と、done で 0 にした同じ割引を渡す。割引率は方策ごとに 1 つで 2 つの Network に共通であり、引数の外的・内的の割引率は、方策の組の割引率の範囲の両端に使われる。系列の切り出しはエピソードをまたがない（cross_episode=False）。優先度は外的 0.8・内的 0.2 で足す。読み取り: 論文の「外的報酬と内的報酬で別々の損失」とは違い、2 つの Network が同じ目標を学ぶ。[deep_rl_zoo (2024/02), agent57/agent.py の Actor・Learner._update_q_network・_calc_retrace_loss；Badia Agent57 (2020/03), §3.1]
- **Taïga らの公開実装（Dopamine の Rainbow＋RND）**: 内的報酬を外的報酬に足してから Dopamine の step と end_episode に渡すので、Replay に入るのは合わせた 1 本の報酬である。設定（rainbow_rnd.gin）は n-step 3（update_horizon）、γ 0.99、RND の報酬の倍率 0.00005、合わせた報酬の clip なし、生命の喪失を終端にしない、である。[BBE 実装 (2026/02), intrinsic_motivation/intrinsic_dqn_agent.py の RNDDQNAgent._add_intrinsic_reward、base_intrinsic_agent.py、configs/rainbow_rnd.gin]
- **Dopamine の Replay**: n-step の和は、窓の中の最初の終端で止める（その step までの報酬だけを割り引いて足す）。終端の印なしにエピソードが終わる窓は、抽選の対象から外す。読み取り: Taïga らの構成では、内的の収益も game over で外的と一緒に切れる。[Dopamine (2026/11), tf/replay_memory/circular_replay_buffer.py の is_valid_transition・sample_transition_batch]
- **DI-engine の NGU**: Q は 1 つで、`ignore_done`（既定は偽）を真にすると、目標値の計算で done を無視する。説明は、終端が偽物の env のための設定と書く。外的と内的を分けて扱う設定は無い。[DI-engine NGU (2025/01), ding/policy/ngu.py の設定の表と _data_preprocess_learn]
- **Unity ML-Agents**: 報酬の信号の部品は「終端を無視する」印（ignore_done）を持ち、RND と curiosity は真、GAIL は偽である。印の説明は、終わりの無い報酬の正の偏りを抑えるためで、終端のとき次のエピソードの報酬で今の収益を作るべきだ、と書く。SAC の目標値は 1 step で、印が真の信号だけ `(1 − done)` を掛けずに、信号ごとの割引率で target の値を足す。コードの注釈は、curiosity や GAIL の survivor bonus を減らすためと書く。PPO でも、終端のときに印が偽の信号の次の値だけを 0 にする。終端の遷移の次観測は、終端の step が返した観測である。読み取り: 内的は終端で切らないが、次のエピソードの報酬も観測も目標値に入らない。[ML-Agents (2026/10), reward_providers/base_reward_provider.py・rnd_reward_provider.py・curiosity_reward_provider.py・gail_reward_provider.py、sac/optimizer_torch.py の sac_q_loss、optimizer/torch_optimizer.py の get_trajectory_value_estimates、agent_processor.py]
- **Acme の n-step の adder（NStepTransitionAdder）**: 報酬・環境の割引・追加の割引を同じ木の形にそろえ、葉ごとに n-step の収益と割引の積を作る（成分ごとの割引率を表せる）。エピソードの終わりでは、残りの短い遷移を書き出し、窓を次のエピソードへ延ばさない。[Acme (2026/10), adders/reverb/transition.py の docstring・_write_last・_compute_cumulative_quantities]
- Temporally-Extended ε-Greedy の R2D2＋RND は、合わせ方、終端の扱い、内的の割引率を書いていない（割引率は R2D2 系に共通の 0.997 だけ）。[Dabney ez-greedy (2020/06), 付録 B.3、表 3]

### 3.3 Replay の系列がエピソードの境をまたぐ実装

- **Acme の系列の adder（SequenceAdder）**: エピソードの終わりの扱いを 4 つ（WRITE、CONTINUE、ZERO_PAD、TRUNCATE）から選ぶ。CONTINUE は、最後の系列を次のエピソードの最初の step まで続けて書く。既定は ZERO_PAD（0 で埋める）である。Acme の R2D2 の builder は TRUNCATE を指定する。CONTINUE を使う Acme の agent は確かめていない。[Acme (2026/10), adders/reverb/sequence.py の EndBehavior・SequenceAdder.__init__、agents/jax/r2d2/builder.py]
- **deep_rl_zoo の系列の切り出し（replay.py の Unroll）**: エピソードをまたぐかを引数（cross_episode）で選び、既定はまたぐ。IMPALA の agent はまたぎ、R2D2・NGU・Agent57 の agent はまたがない。またがないとき、エピソードが系列の長さに足りずに終わると、同じエピソードの前の系列の末尾で埋めて長さをそろえ、前の系列が無ければ書き出さない。[deep_rl_zoo (2024/02), replay.py の Unroll、r2d2・impala・ngu・agent57 の agent.py]
- 原典では、Agent57 と MEME の Replay の系列はエピソードの境をまたがない。NGU も同じである（[エピソード内の新しさを価値ベースに載せる論点](episodic_bonus_value_based_survey_2026-11-18.md) §2.5・§4.5）。[Badia Agent57 (2020/03), 付録 E；Kapturowski MEME (2022/09), 付録 D]

### 3.4 つなぐ・切るを比べた実験

- 内的をエピソード的にするか非エピソード的にするかを比べたのは、Burda らの PPO（図 4）で、非エピソード的のほうが探索した部屋が多かった。[Burda RND (2018/10), §3.2、図 4]
- BYOL-Explore は、自分たちが再実装した RND は、エピソード的な設定では訪れた部屋が少なかったと書くが、BYOL-Explore 自身のエピソード的・非エピソード的の比較は無い。[Guo BYOL-Explore (2022/06), §4.2]
- 価値ベース（DQN 系、Retrace、SAC）で、内的の n-step の和をエピソードの境でつなぐ形と切る形を比べた実験は見つからなかった。Agent57 の random coin の説明（§2.1）は、エピソードが終わることが分割の効果の条件だと書くが、内的の終端の扱いを比べてはいない。[Badia Agent57 (2020/03), §4.2]

## 4. 批判と懸念

- **分解の意味**: MEME は、合わせた損失にすると外的と内的の分解は厳密な意味を持たなくなると書き、Head 1 つでも同程度だったとする。分けた Head の効果は、安定化の部品の有無で変わりうる、という原典の読みである。[Kapturowski MEME (2022/09), §4 C2、付録 H.4]
- **Network ごとの費用**: MEME は Agent57 の 2 つの Network を、計算の重さと標本効率の点で批判する。逆に NGU は、表現の共有が活用側に干渉するという仮説を述べる。2 つの見方を同じ条件で比べた結果は無い（§2.5）。[Kapturowski MEME (2022/09), §4 C2]
- **再実装と原典のずれ**: deep_rl_zoo の Agent57 は、2 つの Network に同じ合わせた報酬を学ばせ、割引率も共通にしている（§3.2）。原典の「別々の報酬で別々の損失」とは違うので、この再実装の結果を原典の分割の効果として読めない（読み取り）。
- **非エピソードの bootstrap の観測**: ML-Agents の説明は「次のエピソードの報酬を使うべき」と書くが、実装は終端の step の観測から bootstrap する（§3.2）。説明と実装の間にずれがある（読み取り）。

## 5. 総合評価

1. **分ける位置**: Network ごと（Agent57、deep_rl_zoo の Agent57）、torso と LSTM を共有して隠れ層つきの Head を分ける（MEME）、価値側の層まで共有して最後の層だけ分ける（Burda らの公開実装、CleanRL、ML-Agents）の 3 段がある。NGU は分けない（§2）。
2. **理由**: Network ごとにする理由は報酬ごとの尺度・分散・optimizer の状態、共有する理由は計算と低い層の特徴の共有である。最後の層だけ分ける実装の位置の理由は、論文にも実装にも書かれていない（§2.1〜§2.4）。
3. **比較**: 分ける位置を同じ条件で比べた事例は無く、あるのは分けるか分けないか（Agent57）、Head の数（Burda ら、MEME）の比較である（§2.5）。
4. **n-step と境**: 読んだ価値ベースの公開実装は、n-step の和（または 1 step の目標）をエピソードの終わりで閉じる。内的を終端で切らない価値ベースの例は ML-Agents の SAC で、終端の観測から bootstrap する。系列がエピソードをまたぐ選択肢は Acme と deep_rl_zoo にあるが、R2D2 系の設定ではまたがない（§3.2・§3.3）。
5. **実験**: つなぐ・切るの比較は PPO（Burda ら）にしか無い（§3.4）。

## 6. 調査の限界

- **検索**: web 検索は 6 回。語は「intrinsic extrinsic value separate networks versus shared torso separate heads ablation」「intrinsic value extrinsic value separate critic networks shared encoder two heads」「Agent57 separate networks shared representation ablation reproduction」「non-episodic intrinsic reward DQN n-step return across episode boundary replay」「non-episodic intrinsic reward value-based DQN Rainbow terminal episodic ablation」「intrinsic reward ignore episode termination bootstrap Q-learning separate discount SAC」の組み合わせで、手法名と一般の語だけを使った。DOMiNO（2 つの価値の Head を使うと検索の要約にあった）、マルチエージェントの好奇心の論文（arXiv 2202.12174）は本文を読んでいない。
- **読めなかったもの・確かめていないもの**: Agent57 の付録 H.1 の表の全体と seed の数（本文 §4 の「3 seed」に頼った）。MEME の付録 H.4 の games・seed（§5.2 の ablation の設定が H.4 にも当たるかは書かれていない）と図 7・16・22・24 の値。RLeXplore の 2 Head の分岐の位置。Acme で CONTINUE を使う agent。DI-engine の NGU で `ignore_done` を真にしたときの系列の作り方（ding/rl_utils の n-step のデータ作り）。R2D2 の原典。
- **実装の版**: RND の公開実装 f75c0f1e（2018-11-22）、CleanRL の ppo_rnd_envpool.py 35896b1f（2023-11-28）、deep_rl_zoo 19e3844c（2024-02-21）、google-research の bonus_based_exploration の最終変更 5b09c22d（2026-02-09）、Dopamine の circular_replay_buffer.py 5254dbed（2026-11-10）、DI-engine の ngu.py 64efcb36（2025-01-27）、ML-Agents の develop 4649106d（2026-10-29。sac/optimizer_torch.py の最終変更は df96d5c8、2022-10-20）、Acme の sequence.py c8d096b1（2026-10-06）。Taïga らの実装は dopamine-rl の版を固定しておらず、当時の Dopamine の n-step が今の版と同じかは確かめていない。
- **要約ツール**: 論文の本文は要約ツールを通して読んだ。Agent57 の §3.1・§4・§4.2・付録 E・F は PDF の画像で確かめた。コードは取得したファイルを grep と行の表示で確かめた。
- **被引用数**: Semantic Scholar の API が回数制限で応答せず、NGU（381、2026-11-27）のほかは、別の報告が 2026-11-18 に取った値を写した。BYOL-Explore 以外の新しく引いた論文（Temporally-Extended ε-Greedy）は未取得である。

## 7. 出典一覧

被引用数は Semantic Scholar の値。NGU は 2026-11-27、ほかは 2026-11-18 の取得。論文は arXiv の頁を挙げ、読んだのは同じ番号の ar5iv の HTML 版（https://ar5iv.labs.arxiv.org/html/<番号>）で、Agent57 は arXiv の PDF（https://arxiv.org/pdf/2003.13350）の頁の画像も読んだ。実装は、挙げたファイルを raw.githubusercontent.com から取得して読んだ。

- [Badia Agent57, 2020/03] Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020（査読あり）. https://arxiv.org/abs/2003.13350 — 所属 DeepMind。被引用 613。
- [Kapturowski MEME, 2022/09] Steven Kapturowski, Víctor Campos, Ray Jiang, Nemanja Rakićević, Hado van Hasselt, Charles Blundell, Adrià Puigdomènech Badia. "Human-level Atari 200x faster." ICLR 2023（査読あり）. https://arxiv.org/abs/2209.07550 — 所属 DeepMind。被引用 45。
- [Burda RND, 2018/10] Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. "Exploration by Random Network Distillation." ICLR 2019（査読あり）. https://arxiv.org/abs/1810.12894 — 所属 OpenAI、University of Edinburgh。被引用 1,838。
- [Badia NGU, 2020/02] Adrià Puigdomènech Badia ほか. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020（査読あり）. https://arxiv.org/abs/2002.06038 — 所属 DeepMind。被引用 381。本冊では読み直していない（§2.4）。
- [Guo BYOL-Explore, 2022/06] Zhaohan Daniel Guo ほか. "BYOL-Explore: Exploration by Bootstrapped Prediction." NeurIPS 2022（査読あり）. https://arxiv.org/abs/2206.08332 — 所属 DeepMind。被引用 109。
- [Dabney ez-greedy, 2020/06] Will Dabney, Georg Ostrovski, André Barreto. "Temporally-Extended ε-Greedy Exploration." arXiv 2020/06（査読の有無は arXiv の頁で確かめていない）. https://arxiv.org/abs/2006.01782 — 所属は未確認。被引用は未取得。
- [RND 公開実装, 2018/11] OpenAI. random-network-distillation（policies/cnn_gru_policy_dynamics.py・cnn_policy_param_matched.py、ppo_agent.py、run_atari.py、README.md、f75c0f1e）. https://github.com/openai/random-network-distillation — 論文の著者の公開実装。
- [CleanRL ppo_rnd_envpool, 2023/11] Costa Huang ほか. CleanRL（cleanrl/ppo_rnd_envpool.py、35896b1f）. https://github.com/vwxyzjn/cleanrl — 公開実装（査読なし）。
- [deep_rl_zoo, 2024/02] Michael Hu. deep_rl_zoo（deep_rl_zoo/agent57/agent.py、networks/value.py、replay.py、r2d2・impala・ngu の agent.py、19e3844c）. https://github.com/michaelnny/deep_rl_zoo — 個人の公開の再実装（DeepMind の実装ではない）。
- [BBE 実装, 2026/02] Google Research. google-research/bonus_based_exploration（intrinsic_motivation/base_intrinsic_agent.py・intrinsic_dqn_agent.py・intrinsic_rainbow_agent.py、configs/rainbow_rnd.gin、5b09c22d）. https://github.com/google-research/google-research/tree/master/bonus_based_exploration — Taïga らの論文の公開実装。
- [Dopamine, 2026/11] Google. Dopamine（dopamine/tf/replay_memory/circular_replay_buffer.py、5254dbed）. https://github.com/google/dopamine — 公開実装。
- [DI-engine NGU, 2025/01] OpenDILab. DI-engine（ding/policy/ngu.py、64efcb36）. https://github.com/opendilab/DI-engine — 公開実装（査読なし）。
- [ML-Agents, 2026/10] Unity Technologies. ml-agents（ml-agents/mlagents/trainers の torch_entities/components/reward_providers/*.py、torch_entities/decoders.py・networks.py、sac/optimizer_torch.py、optimizer/torch_optimizer.py、agent_processor.py、develop 4649106d）. https://github.com/Unity-Technologies/ml-agents — 公開実装。
- [Acme, 2026/10] Google DeepMind. Acme（acme/adders/reverb/sequence.py・transition.py、acme/agents/jax/r2d2/builder.py、c8d096b1）. https://github.com/google-deepmind/acme — 公開実装。
- [既存の報告 episodic, 2026/11] 本リポジトリの報告. "Survey: エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点." [episodic_bonus_value_based_survey_2026-11-18.md](episodic_bonus_value_based_survey_2026-11-18.md) — NGU の事実（§2.4・§2.5・§4.2・§4.5）の出典。
