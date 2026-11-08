# RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 PRD

> 2026-11-06 起票（暫定）。質疑を省き、決めきれない点は §8 の未決表に置いた。確定事項（§4）の各行は暫定の既定であり、会話で合意した決定ではない。
> 担当: 起票と精査を同じ枠でまとめる。理由は、実装の枠の予算が 2026-11-07 朝に切れる前に部品と対照腕を作り始めるためで、ユーザーの裁定（2026-11-06）で**暫定のまま実装へ出す**。実装は §4 の暫定の既定に従い、従えない事情があれば実装計画に理由を書いて変えてよい（質問しない）。設計の採否は [924](924_intrinsic_reward_novelty_10prd.md) の再グリルで見直す。
> 2026-11-06: 口の形（§5.1 の 4 操作と Builder、RND の本体を非公開に閉じること）はユーザーが確認した。同日夜に操作の名前・独立性・device の規約を見直した（2 行下）。
> 2026-11-06（実装中）: §9 の移行対象に設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md) の新設を追記した。起票時の書き漏れで、実装中にユーザーが要望した成果物を記録したもの。英訳は別途。
> 2026-11-06（実装中）: 口の形を見直した。`Multiplier` → `Estimate`、`Update` → `Learn` に改名し、`Learn` を渡された観測で自立して学習する操作にし、device の縛りを外した（§5.1）。改名の理由（impl.md §名前の 9）: Multiplier は名詞で処理が読めない。Update は統計の更新と紛れる。Score は anet で成績の語、Train は Train 側と torch の training mode の語なので避けた。独立化の理由: NGU では採点が行動時、補助学習が Learner 側の Replay 抽出になり、隣接した呼び出しで同じ観測を渡す前提の契約では分離できない。実装は直前の採点の計算グラフを持ち越していて、呼び順の誤りが黙って通っていた。
> 2026-11-07（実装中）: 補助学習の刻みと件数を lane 数（num_envs）に依らない形に改めた（D6・D7・D12・§5.1〜§5.7・§6.2・§8 の 4・5・12・13）。置き場は Push 側（facade の intake）のままとし、exp step 基準の credit と lane の抽選を facade に置き、`Learn` を「渡された観測すべてで 1 step」にした。葉は `intrinsic.rnd.train_fraction` を消して facade の `intrinsic.learn_ratio`・`intrinsic.learn_batch_size` にした（impl.md §名前の 9: Train 側の語を避けて `learn_` に揃え、Learner の `replay_ratio`・`replay_batch_size` と同じ形にする）。`warmup_steps` は採点した観測の件数にした。`Learn` は進めたかを bool で返す（実装中の確認で追加。warm-up 中に何もしなかったことを、呼ぶ側が数えられるようにするため）。経緯は §8.1。
> 2026-11-07（別の枠の報告を受けて改訂。暫定のまま）: Atari の RND の Network が意図と別の形で組まれていた不具合（§2.1）を直し、Taiga ら（ICLR 2020）の Rainbow＋RND へ anet の設定だけで寄せられる形にした（§1.2・§1.3・§2.1・§3・D1・D2・D4・D5・D6・D8・D12・D13・§5.1〜§5.7・§6.2〜§6.4・§7・§8 の 1・2・14〜17・§8.1 の 6・7・§9・§11）。target と predictor を別々の Network 設定にし（`intrinsic.rnd.target.net.*`・`intrinsic.rnd.predictor.net.*`。別の枠の案 `target_net`・`predictor_net` は、役割の違いを葉の名前ではなく役割の階層で表す impl.md §名前の 1 に合わせて改めた）、Head の初期値を `intrinsic.rnd.head.init.*` で設定にし、倍率の clip と統計をモード値にし（`intrinsic.rnd.multiplier_mode`・`intrinsic.rnd.statistics_mode`）、採点と補助学習に使う観測を `intrinsic.observation` で選べるようにし、`intrinsic.rnd.adam_eps`・診断 4 つ・checkpoint の書式 3 を足した。既定の式（`clipped`・`whiten`・`next_state`）は変えない。Atari の `run.@rnd` の既定の Network は公開実装の形（3 conv・leaky ReLU・直交 √2・predictor に FC 512 ×2・出力 512）にし、NGU の同形・128 は 924 で決める。各既定は別の枠の推奨をユーザーが「一旦おすすめの内容で」と受けたもので、暫定の既定であり合意した決定ではない。§8.1 の 6 にあった「÷ 標準偏差」「[−1, 1] に clip」は原典のコードで確かめて訂正した。経緯は §8.1 の 7。
> 2026-11-07（夜）: §6.4 の GridMaze から bonus の符号と減少の判定を外した（ユーザーの指示）。GridMaze の観測は (x, y) の 2 値で新しさが最初の数千 step で尽きるため（結果は §7 の GridMaze の行）。§8 に 18（predictor の周期リセット案）を追加した。
> 起点: [924 §10](924_intrinsic_reward_novelty_10prd.md) は「RND 単独はそれだけでも検証できる独立の腕」とし、§9.1 の比較腕に「RND 単独」を置いている。924 は未決が 23 件あり、エピソード内の新しさ・Replay の契約・方策の組を一度に決めないと実装へ出せない。RND は Replay も Actor も触らずに作れる唯一の部品なので、先に切り出す。
> 関連: [924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md)（内的報酬の全体。本 PRD はその第一の腕）、索引 issue [issue-20261106-16b7af](../management/issues/issue-20261106-16b7af.md)、[DQN 系 Agent 設計 §6.2・§8.2](../design/200_dqn_agents.jp.md)、[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)（補助 Network の置き場。§4 D3 で関係を書く）、[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)（checkpoint 読込時の optimizer options）、[CONTEXT.md](../../CONTEXT.md) の「報酬スケーラ」。

## 1. 問題とゴール

### 1.1 問題

anet の DQN 系 Agent には、観測の新しさに応じて Agent 自身が足す報酬（内的報酬）が無い。[924](924_intrinsic_reward_novelty_10prd.md) はエピソード内の新しさ（NGU の k 近傍、E3B）、RND による一生の新しさ、方策の組を 1 本で扱うが、採点点・Replay の契約・方策の表現が未決で、実装の単位にできない。

RND（Random Network Distillation）は、固定の乱数 Network の出力を学習する Network が真似し、その予測誤差を新しさとする方式である。一生の履歴に対する新しさなので lane 別の記憶を持たず、採点は観測 1 つで閉じる。924 §2.5 の見立てでは、KFM のボス戦の壁には効かない（挑み始めると画面に慣れて消える）ので、この腕は KFM では**対照腕**であり、判断に効くのは GridMaze の到達率と、部品・費用・統計の確認である。

### 1.2 ゴール

1. RND による一生の新しさの内的報酬を、`DefaultDQNAgent` の学習報酬に足せる。既定は無効で、無効のときは今と同じ学習系列（metrics checksum の一致）になる。
2. 一生の新しさの口（§5.1）と RND の実装を、924 の最終形でもそのまま使える部品として作る。Replay・Actor・NetworkModel の契約は変えない。
3. 内的報酬の大きさ・予測誤差の統計・報酬全体に占める割合・費用を指標で見られる。
4. 腕 `run.@rnd` を Atari・GridMaze・LunarLander で回せる。Atari では Taiga らの条件（`run.@rnd_taiga`）を設定だけで選べ、Rainbow＋RND と並べられる（寄せない差は §2.1 の表）。効果の判定は 924 §9.1 に従う（§7）。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 扱いと理由 |
| --- | --- |
| エピソード内の新しさ（k 近傍・E3B）、Embedding と逆動力学 | 別の関心。924 の再グリルで採点点と Replay の契約を決めてから作る。技術の制約ではなく順序の選択 |
| 方策の組（β_i・γ_i を変えた方策）、β=0 の評価 | 924 の問い 1・11 で決める。本 PRD の価値は 1 本で、評価も内的報酬込みで学んだ Q を使う |
| 内的報酬を外的報酬と別に Replay へ保存すること | Replay の契約の変更（924 問い 4）。本 PRD は Push の前に報酬へ畳む仮の運び方（§4 D2）を使い、924 の契約が入った時点で置き換える。制約ではなく移行の順序 |
| 2 本の価値 Head（RND 論文の V_E＋V_I）と非エピソード的な内的収益 | 価値の分解は 924 §6 の別の選択肢。本 PRD では終端で切る 1 本の価値のまま |
| Actor での採点（924 の準備が推す最終形） | 採点を Actor に置くと 1 step 遅れる採点を Replay の注釈で運ぶ契約が要る。facade で次観測を採点すれば終端観測も採点でき契約を増やさない。最終形への移行は 924 |
| 評価（configured eval・EvalPanel）での採点と診断 | eval の runner は `UpdateFromBatch` を呼ばず、採点点が facade なので構造上届かない。評価で内的報酬を見たくなったら 924 の採点点の問いで決める |
| C51 の Head（Taiga らの Rainbow の Q 側）と、Dopamine の PER・Q の初期値の細部 | 別の関心（Q の手法）。anet は IQN で代用し、Q 側の差は §2.1 の「本家との残る差」に記す。技術の制約ではなく、Rainbow＋RND の再現の範囲を RND の部品に限る選択。C51 を作るかは `run.@rainbow_like` の結果を見て別に決める |
| 原典のコードを WSL で動かす再現環境 | anet で寄せても再現できなかったときに具体化する（実験の結果が要る先送り。§8 の 14）。今は作らない |

## 2. 現状（2026-11-06 にコードと文書で確認）

- 内的報酬・RND の実装は無い（`core`・`apps/runner` を `intrinsic`・`novelty`・`rnd` で検索して該当なし）。メトリクスの群番号 56 は設定・コードとも未使用（`56_` で当たる行はすべて群の中の項目番号）。
- `DefaultDQNAgent::UpdateFromBatch`（[default_dqn_agent.cpp:624](../../core/anet-core/src/default_dqn_agent.cpp)）は `unique_lock` の中（:632）で報酬を `reward_scaler_->Scale`（:635）に通し、`obs_norm` の統計を更新し（:638-639）、スケール済み報酬で `BatchExperience` を組み直して（:643-648）内側の Learner へ渡す（:651）。Replay の `Push` は内側の [dqn_based_agent.cpp:4030](../../core/anet-core/src/dqn_based_agent.cpp) にあり、同じロックの範囲内である。`learner.enabled=false` では Push しない（:4025-4027）。
- 遷移の終端は `BatchExperience.next_state` の `done`・`truncated` で届く（[rl.hpp:423-427](../../core/anet-core/include/anet/rl.hpp)）。facade が受ける観測は Actor の frame stack を通す前の生の観測で、done の lane では終端観測が入る（[924 §4.6](924_intrinsic_reward_novelty_10prd.md) の訂正候補のとおり、終端観測が届くのは facade だけ）。
- Network は `NetworkBuilder::BuildNetwork`（[nn.hpp:390](../../core/anet-core/include/anet/nn.hpp)）が `NetworkConfig` と観測の `TensorSpecMap` から組む。`Network::CopyTo`・`Clone`・`CreateForwardExecutor` は :214-220。観測の正規化部品は `ObservationNormalizer`（[scaler.hpp:19](../../core/anet-core/include/anet/scaler.hpp)、Factory は :64-67）。名前付き seed は `SeedMaker::MakeNamedSeed`（[random.hpp:100](../../core/anet-core/include/anet/random.hpp)）。
- checkpoint は header の識別子 `DefaultDQNAgent`（第 1 版）／`DefaultDQNAgent-2`（skip 有効）で形を区別し（[default_dqn_agent.cpp:29-30・:305](../../core/anet-core/src/default_dqn_agent.cpp)）、読込は `LoadNetwork`（:334）。識別子以外は ERROR（:355-358）、skip 有効の checkpoint を無効で読むと ERROR（:359）、skip 無効の checkpoint を有効で読むと skip を新規初期化して WARN 1 回（:368-370）。reward scaler と obs_norm の統計は保存していない（:314-315 の `@todo`）。
- 設定の接頭辞は `DefaultDQNAgent`（[default_dqn_agent.hpp:102](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。節は `learner.`・`actor.[k].`・`reward_scaler.`・`obs_norm.`・`skip.` など。schedule を書ける値は `ProfiledValue`（[schedule.hpp:354](../../core/anet-core/include/anet/schedule.hpp)）で、`ANET_READ_CONFIG` から自動で読む。
- facade の統計をメトリクスへ出す経路は `$agent <key> @learn`（例: [metrics_scalar.txt:152](../../apps/runner/config/metrics_scalar.txt) の `reward_scaler.scale`）。腕ごとの tag は `run.@skip : metrics.scalar.[50_agent_skip/...] = ...` の形で腕のプロファイルにだけ足す（[Atari.txt:280](../../apps/runner/config/Atari.txt)）。
- 無効時の同一性の受入は PRD098 の段 0 の手順（[testdata/prd098/README.md](../../core/anet-core/testdata/prd098/README.md) の「段0・smoke」、`run_validation.py` の before／after／compare）。
- 毎 step の経路に別スレッドのロック待ち・同期の転送を足さない規約は [impl.md:321-326](../quality/impl.md)。本 PRD の採点はロックの中の学習スレッドで行い、新しいロックは足さない。転送は、観測を RND の Network の device へ送る H2D と誤差を戻す D2H が step ごとに加わる（§5.3）。

### 2.1 2026-11-07 の追記（Atari の Run を担当した別の枠の調査と、原典コードの確認）

Atari の Run を担当した別の枠（Codex、2026-11-07）が Run の checkpoint と設定から見つけたもの。表の値はその枠の測定と確認で、Taiga らの式と呼び方は Claude が原典のコードで確かめた（§11）。

**Atari の RND の Network が意図と別の形で組まれている。** `nn.txt` の `Conv1d_S`・`Conv1d_M`・`Conv2d_S`・`Conv2d_M`（[nn.txt:123-148](../../apps/runner/config/nn.txt)）は `out_channels`・`kernel_size`・`stride`・`padding` を block の直下に書くが、`Conv1d`・`Conv2d` の読み込み（[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) の `Conv1dModuleFactory`・`Conv2dModuleFactory`、3020 行・3059 行付近）は `conv.out_channels` などを読む。直下の葉は黙って無視され、`ConvConfig` の既定（同 2969 行。128 チャンネル、3×3、stride 1、padding 0）で組まれる。`ConvProj32`（nn.txt:152-156）と Atari.txt の `AtariConv*` は `conv.*` で正しい。この 4 block を `structure` で使う現用の設定は Atari.txt の `run.@rnd`（[Atari.txt:10](../../apps/runner/config/Atari.txt)）だけで、nn.txt の見本 `@Conv1D`・`@Conv2D`・`@ViT_*` はどの env 設定からも参照されていない。`core/anet-core/testdata/prd092/config/` の 4 本にも同じ葉があるが、Run の設定の写し（記録）で、構造には使っていない。

| | 意図（起票時の PRD、212 §4.3） | 実物 |
| --- | --- | --- |
| 畳み込み | 32@8/4、64@4/2（padding 1） | 128@3/1、128@3/1 |
| Flatten | 6,400 | 819,200 |
| 1 Network のパラメータ | 854,240 | 105,006,592 |

確かめた値: 105,006,592 × 4 byte × 4（target、predictor、Adam の 2 状態）＝ 1,680.1MB。`agent_close.anet` の増分（`mz_rt_rnd_s1` 1,759.36MB と `mz_rt_base_s1` 79.02MB の差 1,680.3MB）と一致する。KFM の `kfm_10m_rnd_s04` も同じ 1,759MB。RND ありの速さが 2.4 倍遅い主因の候補でもある。影響する Run は 11/06〜07 の Atari の RND 全部（KFM s04、Montezuma s1・s2）。GridMaze・LunarLander は `MLP_FC*` で影響なし。下の比較では実物の形でも新しさの判別は落ちていないので、実証された害は速さと checkpoint の大きさであり、これらの Run の成績を「判別できなかった結果」とは読まない。

**ほかに直すもの。** Head の初期値は `RndHeadFactory` に `WeightInitConfig{}` を渡して固定している（[intrinsic_rnd.cpp:109-110](../../core/anet-core/src/intrinsic_rnd.cpp)。既定の mode は xavier、[nn.hpp:70](../../core/anet-core/include/anet/nn.hpp)）。畳み込みの既定の初期値は he（`Conv2dModuleFactory`）。`bonus_share` の分母は Σ|外的| ＋ Σ内的（[default_dqn_agent.cpp:719-724](../../core/anet-core/src/default_dqn_agent.cpp)）で、内的報酬が負になると壊れる。

**公開実装の Network と初期値。**

| | Burda ら（[RND 公開コード](https://github.com/openai/random-network-distillation)） | Taiga ら（[bonus_based_exploration](https://github.com/google-research/google-research/tree/master/bonus_based_exploration)） | CleanRL（[ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)） | NGU（D8 の出典、付録 H.2） | anet の実物（起票時） |
| --- | --- | --- | --- | --- | --- |
| target | 3 conv ＋ FC 512 | 同左 | 同左 | predictor と同形、出力 128 | 2 conv ＋ Head の Linear 128 |
| predictor | 3 conv ＋ FC 512 ×3 | 同左 | 同左 | target と同形 | target と同形 |
| 活性化 | leaky ReLU（0.2）、FC は ReLU | 同左 | leaky ReLU（0.01）、FC は ReLU | — | ReLU |
| 初期値 | 直交 √2、bias 0 | 同左 | 同左 | — | 畳み込み he、Linear xavier、bias 0 |
| 畳み込みの padding | VALID（Flatten 3,136） | SAME（Flatten 7,744） | VALID | — | — |
| 入力 | 1 枚 | 1 枚 | 1 枚 | — | 1 枚 |

形の違いの条件は Burda ら §2.2.1 の「target が predictor のモデルクラスの中にあること」で、同形でも深くても満たす。Burda らのコードは target の出力の分散と最大（`feat_var`・`max_feat`）を監視している（§5.5 の `target_feature_*` の出典）。

**Taiga らの Rainbow＋RND の式と呼び方（原典のコードで確認。ファイルは §11）。**

- 採点と学習に使う観測は、その遷移で行動した観測 x_t。`base_intrinsic_agent.py` の `step` と `end_episode` が、新しい観測を記録する前の `self._observation` を渡す。終端の遷移も x_{T−1} で採点し、0 にしない。
- 観測の正規化は `clip((x − obs_mean) ÷ obs_std, −5, 5)`。変数名は `std` だが中身は分散に相当する再帰（`std ← (std × n ＋ δ² × n ÷ (n＋1)) ÷ (n＋1)`）で、平方根を取らない。観測の統計は正規化後の値で更新する（フィードバック）。初期値は mean 0・std 1。n は agent の `training_steps` で、最初の採点のとき 1（`begin_episode` で 1 増えたあと）、以後 agent step ごとに 1 増える（エピソード開始でも 1 増えるので、採点した観測の件数より少し多い）。
- 誤差 e は 512 次元の MSE の平均。誤差の統計も同じ再帰で、z ＝ (e − mean) ÷ std は更新後の値で作る。内的報酬 ＝ 0.00005 × z。z は clip せず、負の値もそのまま入る。
- predictor は Adam（lr 1e-4、eps 1e-5）で、eval でない agent step ごとに、その 1 件で 1 回更新する（Q の `update_period` 4・batch 32 とは別）。
- `rainbow_rnd.gin` は `RNDRainbowAgent.clip_reward = False`。外的報酬は Dopamine の Runner が [−1, 1] に clip し（`clip_rewards` の既定 true）、外的＋内的の合計は clip しない。
- Network は 3 conv（32@8/4、64@4/2、64@3/1、SAME padding）＋ leaky ReLU（0.2）、直交 √2・bias 0。target は FC 512 だけ、predictor は FC 512（ReLU）×2 ＋ FC 512。

**手元の比較（別の枠。PyTorch の写し、同じ画面で 1 件ずつ 6 万 step、2 seed）。** 画面は Montezuma のランダム方策と、状態の保存・復元で作った別の部屋・鍵あり。写しなので判断材料であり、受入の基準にしない。一式は `.scratch/prd116/repro-20261107/`（追跡外）。

| 形・初期値 | target の出力の分散 | 見慣れた画面の誤差 | 鍵の誤差の比 | 鍵の判別（AUROC） | 別の部屋の誤差の比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A anet の実物（1 億パラメータ） | 0.0013〜0.0015 | 0.00003 | 119〜135 倍 | 1.00 | 約 5,600 倍 |
| A2 意図の形、he/xavier | 0.0025〜0.0027 | 0.00018 | 35〜36 倍 | 1.00 | 1,640〜1,700 倍 |
| B 意図の形、直交 | 0.17 | 0.011 | 36〜44 倍 | 1.00 | 1,660〜1,810 倍 |
| C 3 conv・leaky・512、predictor 同形、直交 | 0.22〜0.28 | 0.02 | 14〜20 倍 | 1.00 | 490〜700 倍 |
| D C の predictor に FC ×2（公開実装の形） | 0.22〜0.28 | 0.04 | 7〜11 倍 | 1.00 | 245〜610 倍 |
| F D の target だけ he/xavier | 0.003〜0.004 | 0.0026〜0.0035（下がらない） | 1.8〜2.1 倍 | 0.80〜0.87 | 74〜138 倍 |
| E D を Taiga らの統計で | 0.39〜0.55 | 0.23〜0.30 | 2.3〜3.8 倍 | 0.94〜0.98 | 28〜107 倍 |

- 実物の大きな Network（A）は判別が落ちていない。問題は意図と違うことと 1 step の費用（この比較でも約 5 倍）。
- target と predictor の尺度が揃っていれば、初期値の種類で判別は大きく変わらない（A2 と B）。target だけ小さいと predictor が誤差を下げられず判別が落ちる（F）。target は学習で直らないので、尺度を揃えることが条件になる。
- predictor を深くすると、この比較の範囲では比は小さくなる（C と D）。6 万 step 時点の値で、公開実装がなぜ深くしたかの根拠は Burda らの論文に見当たらない。長い学習での優劣は分からない（§8 の 2）。
- Taiga らの統計（E）は分散に相当する値で割るため正規化後の値の散らばりが小さくなり、平均から外れた画素が ±5 に張り付いて、この比較では判別が弱い。Taiga らの結果はこの条件で出ているので、再現には要る。既定にはしない（D5）。
- ランダム方策で測った Taiga らの式のボーナスは、中央値 −1e-4、上位 1% ＋1e-4、最大 ＋6e-4、止まった画面 −7e-5。anet の Run（`clipped`）は正だけで平均 ＋5e-6、最大約 7e-4。

**本家との残る差（直さない。実装記録に写す）。**

| 項目 | Taiga ら | anet |
| --- | --- | --- |
| Q の手法 | C51（51 atom、±10） | IQN。C51 は無い（§1.3） |
| 畳み込みの padding | SAME（Flatten 7,744） | VALID（Flatten 3,136）。Burda らと同じ |
| Q の入力の埋め込み | IQN でも Dense 512 は τ の後 | `AtariNature` は τ の前にも Linear 512 |
| Q の初期値 | variance scaling（1/√3、fan_in、uniform） | he・xavier |
| PER の細部 | √priority、重みの正規化は Dopamine の式 | anet の PER |
| `Learn` の入力 | 採点と同じ正規化済みの観測 | 統計の更新後に正規化し直した観測 |
| 統計の n | agent step（エピソード開始でも 1 増える） | 採点した観測の件数 |
| 実行環境 | TF1、ALE は当時の版 | anet、ALE 0.12 |

## 3. 用語（暫定）

[CONTEXT.md](../../CONTEXT.md) に無い概念で、既存の語（報酬スケーラ、spatial exploration）では言い換えられない。確定は 924 §8 の 20 の裁定に合わせ、本 PRD では次の語を使う。

| 語 | 意味 |
| --- | --- |
| 内的報酬 | Env の外的報酬とは別に、Agent が探索のために算出して学習報酬へ足す値 |
| 一生の新しさ | Run の全履歴に照らした観測の新しさ。エピソードごとに戻らない（NGU の lifelong novelty） |
| 新しさの倍率 α | 一生の新しさを、1 を基準にした倍率で表した値。既定（`clipped`）では [1, L] に収め、1 が「見慣れた」。`unclipped` では 1 ＋ z で、1 を下回る値も取る（D4）。NGU ではエピソード内ボーナスに掛ける倍率、本 PRD では α−1 を内的報酬にする |

## 4. 確定事項（暫定の既定）

各行は暫定の既定である。理由が他の決定や事実に依存するときは名指す。

| # | 決定 | 理由 |
| --- | --- | --- |
| D1 | **採点点は facade。** `DefaultDQNAgent::UpdateFromBatch` が、reward scaler を通した後・内側の Learner へ渡す前に、`intrinsic.observation` で選んだ観測（既定は `next_state`。D5）を採点する | 遷移の 4 つ組と終端観測が揃う唯一の場所（§2）。Replay・Actor の契約を変えずに済む。ロックの中の経路に forward が乗る費用は §6.3 で測る |
| D2 | **運び方は報酬へ畳む（仮）。** 学習報酬 = スケール済み外的報酬 ＋ β × (α − 1)。採点する観測が `next_state`（既定）のとき、done の lane の遷移は内的報酬 0（truncated は採点する）。`state` のときは mask しない（D5・§5.3）。β は `intrinsic.beta`（ProfiledValue、schedule 可） | 外的・内的を別に保存する契約は 924 問い 4。終端観測に内的報酬を付けると終端そのものが報酬になるので 0 にする（924 §5.5 の 2 と同じ扱い）。`state` では採点する観測が終端観測ではないので mask の理由が無く、Taiga らも終端の遷移に付ける（§2.1）。D1・D5 に依存 |
| D3 | **公開ヘッダ（`anet::rl::intrinsic`）に置くのは一生の新しさの口と生成の口（Builder）だけ。RND の本体は専用の非公開ヘッダと cpp に閉じ、NetworkModel の外で自分の Network・optimizer・統計を所有する。** DQN は RND の型も `intrinsic.rnd.*` の葉も見ず、`intrinsic.enabled`・`intrinsic.beta` を読んで Builder に設定の接頭辞を渡す | 924 の最終形でも同じ口を使う。RND の本体は DQN と独立に試験・差し替えでき、2 つ目の方式（RECODE 型など）を足すときも DQN 側は変わらない。[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) は「primary の出力を入力に取り、同じ target 同期を持つ補助 Network」を NetworkModel に置く判断で、RND は観測を入力に取り target 同期を持たないので適用範囲の外。矛盾ではないが、適用範囲の注記は 924 の再グリルで足す（§8 の 9） |
| D4 | **内的報酬は NGU の倍率の形で作り、clip はモード値で選ぶ。** e_t = ‖predictor(x_t) − target(x_t)‖²（次元方向の平均）、z_t = (e_t − μ_e)/σ_e。`intrinsic.rnd.multiplier_mode ?= clipped` では α_t = clip(1 + z_t, 1, L)、`unclipped` では α_t = 1 + z_t（負の値もそのまま）。内的報酬 = α_t − 1。μ_e・σ_e の作り方は D5 の統計のモードに従う。warm-up 中はどちらも α=1 | 924 の RND の倍率と同じ式・同じ統計なので、エピソード内ボーナスが入ったときに部品を変えない（既定は変えない）。Taiga らの Rainbow＋RND は z を clip せず 0.00005 倍して足す（§2.1）ので、再現には `unclipped` が要る。RND 原典の e/σ（平均を引かない）は §8 の 1 の比較候補。L は `intrinsic.rnd.multiplier_max`（`unclipped` では読まない。§5.4） |
| D5 | **入力は facade に届く生の観測（最新 1 フレーム）で、どちらの観測を使うかは `intrinsic.observation ?= next_state`（`next_state`｜`state`）、統計の作り方は `intrinsic.rnd.statistics_mode ?= whiten`（`whiten`｜`variance_feedback`）で選ぶ。** `next_state` は遷移の次観測 x_{t+1}（今と同じ）、`state` は行動した観測 x_t（Taiga ら）。`whiten` は RND 専用の `ObservationNormalizer` で移動平均・標準偏差により白色化して clip し、誤差の統計は Welford。採点は更新前の統計で行い、採点の後に更新する。`variance_feedback` は Taiga らの式（§5.2）で、観測は正規化後の値で統計を更新し（フィードバック）、分散に相当する値で割り、誤差の z は更新後の統計で作る。lane 順に 1 件ずつ更新する | RND 原典 §2.4（固定の target は尺度に合わせられないので正規化が必須、最後の 1 フレーム、Algorithm 1 の順）。既存部品の再利用（AGENTS.md の共通処理を先に探す）。Q 入力用の `obs_norm`（Atari は pass_through）とは独立。Taiga らの統計は写しの比較で判別が弱い（§2.1 の E）ので既定にせず、再現のためのモードにする。採点する観測は Burda ら・anet が次観測、Taiga らが行動した観測で、再現には選べる必要がある。観測の選択と mask の有無を 1 つのキーで表すのは、mask の理由（終端観測）が観測の選び方から決まるため（D2） |
| D6 | **warm-up。** 採点した観測の件数が `intrinsic.rnd.warmup_steps` に達するまでは統計だけ更新し α=1（内的報酬 0）を返し、`Learn` は何もしない。件数は intake で呼ぶなら exp step と同じで、既定 12,800 は N=128 の採点 100 回と同じ長さ。0 なら warm-up を置かない（Taiga らの条件） | RND 原典は乱数行動で観測統計を作ってから学習する。anet では方策をそのまま動かし、ボーナスだけ止める。採点の回数で数えると warm-up の長さが lane 数に比例する（§8.1） |
| D7 | **predictor の学習は on-policy で、刻みと件数は lane 数に依らない。** facade は intake ごとに credit を N × `intrinsic.learn_ratio` ÷ k だけ貯め（k = min(`intrinsic.learn_batch_size`, N)）、credit が 1 以上の回数だけ、その intake の観測から k lane を重複なく選んで `Learn` を呼ぶ。端数は次の intake へ繰り越し、1 intake の step 数は ⌊N ÷ k⌋ を上限にする。`Learn` は渡された観測すべてで MSE を 1 回 backward し optimizer を 1 回進め、進めたかを返す（warm-up 中は進めず false）。credit は結果に依らず消費する。母集団は intake の観測（Push 側。§8 の 12、924 問い 13）。optimizer は AdamW をモジュールが私有し、`intrinsic.rnd.learning_rate`・`.weight_decay` で設定する | exp step あたりの step 数（`learn_ratio` ÷ `learn_batch_size`）と 1 step の件数が、N・RR・replay batch のどれにも依らない（§8.1）。Learner の `replay_ratio`・`replay_batch_size`（U = N × RR ÷ B）と同じ形。RND 原典の間引き（§3.4、「内的報酬が消える速さ」を揃える）は N=128 で割合 0.25 の手動の補正で、`learn_ratio` 0.25 が同じ量。Replay からの抽出を要らなくし（924 問い 8 を先取りしない）、Learner の optimizer に group を足すと役割表・checkpoint・[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) へ波及する |
| D8 | **target と predictor は別々の Network 設定 `intrinsic.rnd.target.net.*`・`intrinsic.rnd.predictor.net.*`（どちらも必須）から観測仕様で組み、出力はどちらも `intrinsic.rnd.output_dim` 次元の Linear の Head（初期値は `intrinsic.rnd.head.init.*`、既定 xavier）。** target は乱数初期化のあと固定し、predictor から同期しない。重みは共有しない。Atari の `run.@rnd` の既定は公開実装の形（3 conv 32@8/4・64@4/2・64@3/1、padding 0、leaky ReLU 0.2、直交 √2・bias 0。target は Head の FC 512 だけ、predictor は FC 512（ReLU）×2 の後に Head。`output_dim` 512）。GridMaze・LunarLander は今の MLP を両方に写し、挙動を変えない | 形の違いの条件は Burda ら §2.2.1（target が predictor のモデルクラスに入る）で、同形でも深くても満たす。別々の設定にするのは、predictor だけ深くする形を葉の有無ではなく構造で表すため（impl.md §名前の 1。役割の違いは `target.`・`predictor.` の階層で表す）。Atari の既定を公開実装の形にするのは比較対象（Taiga ら）と揃えるためで、NGU の同形・128（付録 H.2）との優劣は写しの比較では決まらず（§2.1 の C・D）、924 で決める（§8 の 2）。構造は Body カタログから選ぶので、ベクトル観測（MLP）と画像観測（CNN）を同じ仕組みで扱う（924 のゴール） |
| D9 | **無効化は `intrinsic.enabled ?= false` の葉 1 つ。** 無効のときは Network・optimizer・統計・seed のどれも作らず、採点も forward も無い | 有効化の口を 1 つに保つ（924 問い 9 の推奨）。無効時の同一性（§6.1）。RND しか無い間は方式の列挙キーを作らない（impl.md「名前」5。924 で 2 つ目が入るときに足す） |
| D10 | **評価専用 Run（`learner.enabled=false`）では採点も更新も行わず、`intrinsic.enabled=true` なら 1 回 WARN** | 学習が無く報酬は使われない。意図された休止はエラーにしない（AGENTS.md Fail-Fast の表） |
| D11 | **メトリクスは群 56（`56_agent_in`）、`$agent intrinsic.<key> @learn` で読む。tag は腕のプロファイルにだけ足す** | 群 56 は未使用（§2）。無効時に `metrics.scalar.defs` を変えないため腕にだけ足す。tag は群名と `/` を含めて 40 文字以内 |
| D12 | **checkpoint は `intrinsic` 節を learner の後に足し、header の識別子を `<今の識別子>;intrinsic` にする。** 節には RND の書式（`RndNoveltyEstimator-3`）と統計のモード、target・predictor・optimizer・統計（`whiten` は観測の正規化と Welford、`variance_feedback` は obs_m・obs_v・err_m・err_v・n）・target の出力の統計・採点した観測の件数と、facade の credit の端数・lane 抽選の乱数状態・累計回数を入れる。統計のモードが設定と違えば ERROR。有効で節が無い checkpoint は ERROR、無効で節がある checkpoint は ERROR、無効で節が無い checkpoint は今のまま（バイト同一） | 固定 target を作り直すと predictor の教師が変わる（924 §4.9）。統計を保存しないと再開後の α の尺度が変わる。旧 checkpoint の有効読みは 924 問い 21 の推奨 (a)（互換分岐を残さない）。skip の先例（WARN 1 回）は採らない |
| D13 | **腕は `run.@rnd`。** Atari・GridMaze・LunarLander に `intrinsic.enabled = true`、2 つの Network、tag の宣言を置く。既定値は `agent.txt` の `@baseline` に `?=` で置く。Atari には Taiga らの条件を重ねる `run.@rnd_taiga`（β 0.00005、`learn_ratio` 1・`learn_batch_size` 1、`observation` state、`unclipped`、`variance_feedback`、warm-up 0、`adam_eps` 1e-5）を置き、Q 側の `run.@rainbow_like`（2026-11-07 に別の枠が追加済み）の後ろに `run.@rnd > run.@rnd_taiga` と重ねる | 機能ごとの @ プロファイルは作らず、既存の `run.@skip` と同じ形（§2）。GridMaze・LunarLander は害の確認と到達率の測定に使う（§7）。Taiga らの条件を 1 つのプロファイルに束ねるのは、再現の腕と anet の既定の腕を bat の 1 行で切り替えるため |
| D14 | **SHARED scope との併用は制限しない** | RND は lane 別の状態を持たない。924 のエピソード内の記憶で初めて要る制約 |

## 5. 公開契約

### 5.1 一生の新しさの口（インタフェース）

名前空間 `anet::rl::intrinsic`、公開ヘッダ `include/anet/intrinsic.hpp`。anet-core の公開 API（`NetworkBuilder`・`Network`・`ObservationNormalizer`・`SeedMaker`・`Serializable`）だけに依存し、DQN の型を参照しない。

| 操作 | 受け取るもの | 返すもの | 副作用 |
| --- | --- | --- | --- |
| `Estimate` | lane ごとの観測（[B, …]。device は任意） | 新しさの倍率 α [B]（有限。`clipped` では [1, L]、`unclipped` では 1 ＋ z で 1 未満も負もある。入力と同じ device） | 誤差の統計と観測の統計を更新する（順は D5 のモードに従う。`whiten` は採点の後、`variance_feedback` は観測の正規化の後・誤差の z の前）。warm-up 中は統計だけ更新して 1 を返す。次の呼び出しへ状態を持ち越さない |
| `Learn` | 学習に使う観測（[B, …]。device は任意。`Estimate` に渡したものと同じでなくてよい） | 学習を進めたか（bool。warm-up 中は進めず false） | 渡された観測すべてで predictor の補助学習を 1 step 進める（抽選や刻みは持たない。統計は読むだけで更新しない）。warm-up 中は何もしない。target は変えない。`Estimate` とは独立で、順序も回数も縛らない |
| `Diagnostics` | なし | 直近の採点と更新の診断値（§5.5 の key） | なし |
| `Save`／`Load` | archive | — | `Serializable`。§5.6 の節の中身 |

生成の口（Builder）は同じ公開ヘッダに置く。

| 操作 | 受け取るもの | 返すもの | 副作用 |
| --- | --- | --- | --- |
| `LifelongNoveltyEstimatorFactory::Create` | 設定（`ConfigData`）と設定の接頭辞（例: `DefaultDQNAgent.intrinsic.rnd`）、観測の仕様（`TensorSpecMap`）、device、Agent の `SeedMaker` | 口の実体（所有権つき） | 接頭辞の下の葉を自分で読んで検証し（§5.4）、反していれば ERROR。Network の構築と seed の派生 |

DQN が見るのは口と Builder だけで、RND の型・設定 DTO・葉の名前は非公開ヘッダ（`src/intrinsic_rnd.hpp`）に閉じる。テストは非公開ヘッダを含めて RND を直接試験する。

本 PRD の呼び方は、1 intake につき facade が `intrinsic.observation` で選んだ観測（既定は次観測）で `Estimate` を呼び、α から内的報酬を作って終端のマスク（次観測のときだけ）と β を掛け（D2）、credit（D7）が許す回数だけ、同じ観測から選んだ lane で `Learn` を呼ぶ。口は α までを返し、β・終端・刻み・抽選・どちらの観測かを知らない。2 つの操作は独立で、どちらをいつ呼んでもよく、Save/Load は呼び出しの途中を除きどの時点でもよい。device は入力に従い、計算の device は Builder に渡す。CPU を前提にしない。NGU（924）でも `Estimate` は行動時、`Learn` は Push 側（intake）の刻みのままとし、口は変えない。母集団を Replay に変えるか、Push 側を消すかは §8 の 12・13（経緯は §8.1）。924 でエピソード内の新しさが入るときは、同じ α をエピソード内ボーナスの倍率として使う。日本語との対応は、採点＝`Estimate`、補助学習＝`Learn`。

RND 以外の実装（RECODE 型など）が入るときも、この 4 つの操作で足りる見込みである。足りなくなったときは 924 の再グリルで口を変える。

### 5.2 RND の実装の契約

- target は観測仕様と `intrinsic.rnd.target.net.*`、predictor は `intrinsic.rnd.predictor.net.*` から組んだ Network で、どちらも末尾に `output_dim` 次元の Linear の Head を持つ（Head の初期値は `intrinsic.rnd.head.init.*`。既存の `WeightInitConfig` の検証に通す）。2 つの構造は違ってよい。target は seed `intrinsic.target`、predictor は seed `intrinsic.predictor` で初期化する（`SeedMaker::MakeNamedSeed`）。seed が同じなら構造と初期値が再現する。
- 誤差 e_t は出力の二乗誤差の次元方向の平均。倍率は `multiplier_mode` に従う（D4）。`whiten` では μ_e・σ_e は一生の累積（Welford）、`variance_feedback` では下の式。どちらも割る値が 0 のときは α=1。
- `whiten` の観測の正規化は RND 専用の `ObservationNormalizer`（`intrinsic.rnd.obs_norm.*`）で、採点は更新前の統計、更新は採点の後。`variance_feedback` は推定器の中の小さな統計（§5.7）が obs_m・obs_v・err_m・err_v・n を持ち、`obs_norm.clip_range` だけを読む。lane 順に 1 件ずつ次を行う（forward は lane 数分まとめてよく、統計の更新だけ順に行う）。

```text
初期値: obs_m = 0, obs_v = 1, err_m = 0, err_v = 1, n = 0

Estimate（lane ごと）:
  n += 1
  p = clip((x − obs_m) ÷ obs_v, −clip_range, clip_range)
  e = mean((predictor(p) − target(p))²)        # 次元方向の平均
  d = p − obs_m;  obs_m += d ÷ n;  obs_v = (obs_v × n ＋ d² × n ÷ (n＋1)) ÷ (n＋1)
  d = e − err_m;  err_m += d ÷ n;  err_v = (err_v × n ＋ d² × n ÷ (n＋1)) ÷ (n＋1)
  z = (e − err_m) ÷ err_v                        # 更新後の値を使う
  α = 1 ＋ z（multiplier_mode に従って clip）

Learn: p = clip((x − obs_m) ÷ obs_v, …) で 1 step。統計は更新しない
```

  obs_m・obs_v は観測と同じ形（成分ごと）、err_m・err_v はスカラー。Taiga らのコードの変数名は `std` だが中身は上の再帰で、平方根を取らない（§2.1）。
- Adam の eps は `intrinsic.rnd.adam_eps`（既定 1e-8）。optimizer の options へ渡し、読込時の options の置き換えと WARN にも含める。
- target の出力の統計（§5.5 の `target_feature_var`・`target_feature_max`）は採点のたびに更新する。一生の累積の分散は次元ごとに持ち、その平均を出す。
- 補助学習は渡された観測すべてで MSE を 1 回 backward して optimizer を 1 回進め、進めたかを bool で返す（warm-up 中は進めず false）。lane の抽選と刻みは facade（§5.3）が持ち、RND は乱数を持たない。
- 乱数は Agent の `SeedMaker` から名前付きで派生し（lane の抽選は facade の `intrinsic.learn`）、既存の名前（`replaybuffer`・`learner`・`network`・`plasticity_probe` など）の派生と消費列を変えない。

### 5.3 facade の接続

`DefaultDQNAgent::UpdateFromBatch` の中で、`reward_scaler_->Scale` の後・内側の Learner へ渡す `BatchExperience` を組む前に、次を行う（有効のときだけ）。

1. `obs` は `intrinsic.observation` が `next_state` なら `next_state.obs`、`state` なら `state.obs`（どちらも最新 1 フレームの生の観測。`state.obs` が 1 フレームであることは実装で確かめる）。`alpha = lifelong->Estimate(obs)`。
2. 内的報酬 = β × (alpha − 1)。`next_state` のときは `next_state.done` の lane を 0 にする。`state` のときは mask しない。
3. 学習報酬 = スケール済み外的報酬 ＋ 内的報酬。
4. credit += N × `learn_ratio` ÷ k（k = min(`learn_batch_size`, N)。N < `learn_batch_size` なら構築時に WARN 1 回）。credit ≥ 1 の間、`obs` から k lane を重複なく選んで `lifelong->Learn(選んだ観測)` を呼び、credit −= 1（warm-up 中で false が返っても消費し、繰り越さない）。true が返った回数を `learn_count` に数える。1 intake の step 数は ⌊N ÷ k⌋ を上限にし、超えた credit は次の intake へ繰り越す。
5. 診断値を `GetScalar` の `intrinsic.*` で読めるよう保持する。

追加する処理は 1 intake あたり、`Estimate` で forward 2 回（target・predictor、B 観測）と統計の更新、`Learn` で平均 B × `learn_ratio` ÷ k 回の forward 2 回と backward 1 回（各 k 観測。既定の 0.25・32 と N=128 なら intake ごとに 1 回・32 観測）、転送は観測の H2D（`Estimate` で B、`Learn` で k ずつ）と誤差の D2H、診断値の CPU 化。既存のロックの中で行い、新しいロック・スレッドを足さない。`Learn` の forward の重複は、`learn_ratio` が 0.5 以下なら backward が k 観測だけになる分で相殺される見積もり。`run.@rnd_taiga`（`learn_ratio` 1・`learn_batch_size` 1）では intake ごとに N 回の backward（各 1 観測）と N 回ずつの H2D・D2H が入り、費用は step 数（N 倍）に比例して増える。単価は未測定で §6.3 で測り、ABBA の群に `run.@rnd_taiga` も足す。

### 5.4 設定

接頭辞は `DefaultDQNAgent.`。既定値は `agent.txt` の `@baseline` に `?=` で置く。

```text
DefaultDQNAgent.@baseline : intrinsic.enabled ?= false
DefaultDQNAgent.@baseline : intrinsic.beta ?= 0.1                 # ProfiledValue。内的報酬の重み
DefaultDQNAgent.@baseline : intrinsic.observation ?= next_state   # 採点と補助学習に使う観測。next_state｜state
DefaultDQNAgent.@baseline : intrinsic.learn_ratio ?= 0.25         # exp step あたりの補助学習の件数（Learner の replay_ratio と同じ形）
DefaultDQNAgent.@baseline : intrinsic.learn_batch_size ?= 32      # 補助学習 1 step の件数
DefaultDQNAgent.@baseline : intrinsic.rnd.output_dim ?= 128       # target・predictor の Head の出力次元
DefaultDQNAgent.@baseline : intrinsic.rnd.head.init.mode ?= xavier      # Head の初期値（既存の WeightInitConfig の葉）
DefaultDQNAgent.@baseline : intrinsic.rnd.head.init.manual_gain ?= 0    # 正なら orthogonal の gain。0 なら nonlinearity から求める
DefaultDQNAgent.@baseline : intrinsic.rnd.multiplier_mode ?= clipped    # clipped｜unclipped
DefaultDQNAgent.@baseline : intrinsic.rnd.multiplier_max ?= 5     # L（clipped のときだけ読む）
DefaultDQNAgent.@baseline : intrinsic.rnd.statistics_mode ?= whiten     # whiten｜variance_feedback
DefaultDQNAgent.@baseline : intrinsic.rnd.warmup_steps ?= 12,800  # 採点した観測の件数（intake なら exp step）。N=128 の 100 intake と同じ。0 なら warm-up 無し
DefaultDQNAgent.@baseline : intrinsic.rnd.learning_rate ?= 1e-4
DefaultDQNAgent.@baseline : intrinsic.rnd.weight_decay ?= 0
DefaultDQNAgent.@baseline : intrinsic.rnd.adam_eps ?= 1e-8
DefaultDQNAgent.@baseline : intrinsic.rnd.obs_norm.clip_range ?= 5   # ObservationNormalizer の既存の葉（variance_feedback もこれだけ読む）

# GridMaze・LunarLander の例。同じ構造を 2 回書かず、1 つの定義を両方から参照する
run.@rnd : DefaultDQNAgent.intrinsic.enabled = true
run.@rnd : DefaultDQNAgent.intrinsic.rnd.target.net.branch.[feature].$ = net.branch.@RndMlp
run.@rnd : DefaultDQNAgent.intrinsic.rnd.target.net.body.output.[features] = feature
run.@rnd : DefaultDQNAgent.intrinsic.rnd.predictor.net.branch.[feature].$ = net.branch.@RndMlp
run.@rnd : DefaultDQNAgent.intrinsic.rnd.predictor.net.body.output.[features] = feature

# Atari の例。層カタログは公開実装の形（名前は案。層の順で付ける）
net.block.[AtariRndConv1].type = Conv2d
net.block.[AtariRndConv1].conv.out_channels = 32
net.block.[AtariRndConv1].conv.kernel_size = 8
net.block.[AtariRndConv1].conv.stride = 4
net.block.[AtariRndConv1].conv.padding = 0
net.block.[AtariRndConv1].init.mode = orthogonal
net.block.[AtariRndConv1].init.manual_gain = 1.4142135623730951
# AtariRndConv2 は 64@4/2、AtariRndConv3 は 64@3/1。padding 0、初期値は同じ
# AtariRndFC は linear.out_features 512、初期値は同じ。AtariRndLeaky は type LeakyReLU、negative_slope 0.2
run.@rnd : DefaultDQNAgent.intrinsic.enabled = true
run.@rnd : DefaultDQNAgent.intrinsic.rnd.output_dim = 512
run.@rnd : DefaultDQNAgent.intrinsic.rnd.head.init.mode = orthogonal
run.@rnd : DefaultDQNAgent.intrinsic.rnd.head.init.manual_gain = 1.4142135623730951
run.@rnd : DefaultDQNAgent.intrinsic.rnd.target.net.branch.[feature].bind = grid
run.@rnd : DefaultDQNAgent.intrinsic.rnd.target.net.branch.[feature].structure = AtariRndConv1 > AtariRndLeaky > AtariRndConv2 > AtariRndLeaky > AtariRndConv3 > AtariRndLeaky > Flatten
run.@rnd : DefaultDQNAgent.intrinsic.rnd.target.net.body.output.[features] = feature
run.@rnd : DefaultDQNAgent.intrinsic.rnd.predictor.net.branch.[feature].bind = grid
run.@rnd : DefaultDQNAgent.intrinsic.rnd.predictor.net.branch.[feature].structure = AtariRndConv1 > AtariRndLeaky > AtariRndConv2 > AtariRndLeaky > AtariRndConv3 > AtariRndLeaky > Flatten > AtariRndFC > ReLU > AtariRndFC > ReLU
run.@rnd : DefaultDQNAgent.intrinsic.rnd.predictor.net.body.output.[features] = feature
run.@rnd_taiga : DefaultDQNAgent.intrinsic.beta = 0.00005
run.@rnd_taiga : DefaultDQNAgent.intrinsic.observation = state
run.@rnd_taiga : DefaultDQNAgent.intrinsic.learn_ratio = 1
run.@rnd_taiga : DefaultDQNAgent.intrinsic.learn_batch_size = 1
run.@rnd_taiga : DefaultDQNAgent.intrinsic.rnd.multiplier_mode = unclipped
run.@rnd_taiga : DefaultDQNAgent.intrinsic.rnd.statistics_mode = variance_feedback
run.@rnd_taiga : DefaultDQNAgent.intrinsic.rnd.warmup_steps = 0
run.@rnd_taiga : DefaultDQNAgent.intrinsic.rnd.adam_eps = 1e-5
```

Atari のチェーンは、実時間優先の既存の連鎖の後ろに `run.@rainbow_like`（Q 側。別の枠が 2026-11-07 に追加済み）、RND ありはさらに `run.@rnd > run.@rnd_taiga` を重ねる。プロファイルが勝つか、別名の葉が負けないかは解決して確かめる（§6.2 の「出荷設定の形」）。

| キー | 型・値域 | 反したとき |
| --- | --- | --- |
| `intrinsic.enabled` | bool | 型違いは ERROR |
| `intrinsic.beta` | ProfiledValue、全端点が有限で ≥ 0 | ERROR（キー・値・期待を含む） |
| `intrinsic.observation` | `next_state`｜`state` | それ以外は ERROR（キー・値・候補を含む） |
| `intrinsic.learn_ratio` | 0 < 値 ≤ 1 | ERROR |
| `intrinsic.learn_batch_size` | 整数 ≥ 1。N より大きいときは N 件で学習し WARN 1 回 | ERROR（値域） |
| `intrinsic.rnd.target.net.*`・`intrinsic.rnd.predictor.net.*` | 既存の `NetworkConfig`（Body。Head は持たない）。どちらも必須 | 片方でも無ければ ERROR（無いキーを含む）。構造は既存の NN の構築時検証（未知参照・rank・shape） |
| `intrinsic.rnd.output_dim` | 整数 ≥ 1 | ERROR |
| `intrinsic.rnd.head.init.*` | 既存の `WeightInitConfig`（`mode`・`manual_gain` など） | 既存の `ValidateWeightInitConfig`（ERROR） |
| `intrinsic.rnd.multiplier_mode` | `clipped`｜`unclipped` | それ以外は ERROR（キー・値・候補を含む） |
| `intrinsic.rnd.multiplier_max` | 有限 > 1。`unclipped` では読まない（検証は行い、解決した値が実装の既定 5 と違えば WARN 1 回） | ERROR |
| `intrinsic.rnd.statistics_mode` | `whiten`｜`variance_feedback` | それ以外は ERROR（キー・値・候補を含む） |
| `intrinsic.rnd.warmup_steps` | 整数 ≥ 0 | ERROR |
| `intrinsic.rnd.learning_rate` | 有限 > 0 | ERROR |
| `intrinsic.rnd.weight_decay` | 有限 ≥ 0 | ERROR |
| `intrinsic.rnd.adam_eps` | 有限 > 0 | ERROR |
| `intrinsic.rnd.obs_norm.*` | 既存の `ObservationNormalizerConfig`。`variance_feedback` では `clip_range` だけ読む（ほかは検証だけ行い、解決した値が実装の既定と違えば葉ごとに WARN 1 回） | 既存の検証 |

- `intrinsic.enabled`・`intrinsic.beta`・`intrinsic.observation`・`intrinsic.learn_ratio`・`intrinsic.learn_batch_size` は DQN が読む。`intrinsic.rnd.*` は Builder（§5.1）が接頭辞を受けて読み、検証する。DQN 側に RND の葉の読み込みを書かない。
- `intrinsic.enabled=false` のときは `intrinsic.*` の他の葉を読まず、検証もしない（未指定のまま無効にできる）。
- `intrinsic.enabled=true` で `intrinsic.rnd.target.net` と `intrinsic.rnd.predictor.net` のどちらかが無いときは ERROR。片方から他方を補わない。
- モードが読まない葉（`unclipped` の `multiplier_max`、`variance_feedback` の `clip_range` 以外の `obs_norm.*`）は、値の検証を行ったうえで、解決した値（`?=`・個別指定・プロファイル・CLI のどれで与えられたかは問わない）が実装の既定値と違えば、構築時に葉ごとに WARN を 1 回出し、値は使わない。実装の既定値は RND の設定 DTO が持つ値で、`multiplier_max` は 5、`obs_norm.*` は RND が型の既定から変えている 4 葉（`use_centering` true・`use_robust_update` false・`post_process_type` none・`clip_range` 5。[intrinsic_rnd.cpp:62-65](../../core/anet-core/src/intrinsic_rnd.cpp)）とそれ以外の `ObservationNormalizerConfig` の既定。このうち `agent.txt` の `@baseline` に `?=` があるのは `multiplier_max` と `obs_norm.clip_range` で、その値が実装の既定と一致することは §6.2 の「既定の一致」で確かめる（PRD072 の `check_default_leaves.py` は `?=` と `=` の使い分けの静的検査で、値の一致は見ない）。`obs_norm` のほかの葉は RND の内部の既定で、起票時のまま `@baseline` に置かない。したがって `@baseline` の `?=` のままなら WARN は出ず、Run プロファイル・個別指定・CLI で実装の既定と違う値を書いたときだけ出る。読まない値に別の意味を持たせない（AGENTS.md §設定値の扱い）。
- `learner.enabled=false` かつ `intrinsic.enabled=true` は ERROR にせず WARN 1 回（D10）。
- 値域の検証は構築時（設定読み込み）に行う。観測の形・device の不一致は NN の既存の初期化検証に任せる。
- 0 や負に別の意味を持たせない（0 は 0 として検証に掛かる）。

### 5.5 メトリクス

`$agent intrinsic.<key> @learn` で読む。群は `56_agent_in`。

| tag | key | 意味 |
| --- | --- | --- |
| `56_agent_in/01_bonus_mean` | `intrinsic.bonus_mean` | 内的報酬（β 倍後）の lane 平均 |
| `56_agent_in/02_bonus_max` | `intrinsic.bonus_max` | 同、lane 最大 |
| `56_agent_in/03_bonus_share` | `intrinsic.bonus_share` | Σ\|内的報酬\| ÷ (Σ\|外的報酬\| ＋ Σ\|内的報酬\|)。両方 0 なら NaN。負の内的報酬でも [0, 1] に収まる（起票時の Σ内的報酬の分母は負で壊れるので改めた） |
| `56_agent_in/04_bonus_min` | `intrinsic.bonus_min` | 同、lane 最小（`unclipped` では負になる） |
| `56_agent_in/05_bonus_pos_ratio` | `intrinsic.bonus_pos_ratio` | 内的報酬が正の lane の割合 |
| `56_agent_in/11_error_mean` | `intrinsic.error_mean` | μ_e |
| `56_agent_in/12_error_std` | `intrinsic.error_std` | σ_e（`variance_feedback` でも標準偏差 √err_v を出す） |
| `56_agent_in/13_error_step` | `intrinsic.error_step` | この step の e_t の lane 平均 |
| `56_agent_in/14_target_feature_var` | `intrinsic.target_feature_var` | target の出力の次元ごとの一生の分散の平均（1 lane でも値が出る形。Burda らの `feat_var`） |
| `56_agent_in/15_target_feature_max` | `intrinsic.target_feature_max` | target の出力の絶対値の最大（この step。Burda らの `max_feat`） |
| `56_agent_in/21_clip_ratio` | `intrinsic.clip_ratio` | α が L に当たった lane の割合（`unclipped` では 0） |
| `56_agent_in/31_predictor_loss` | `intrinsic.predictor_loss` | 補助学習の直近の step の loss（まだ 1 回も進めていなければ NaN。読込直後も次の step まで NaN） |
| `56_agent_in/32_learn_count` | `intrinsic.learn_count` | 補助学習の累計 step 数（`Learn` が true を返した回数）。exp step に対する傾きが `learn_ratio` ÷ `learn_batch_size` になる（N 不変の確認用） |

warm-up 中は bonus 系が 0、error 系は値を持つ。無効のときは key が存在せず（`std::nullopt`）、腕に tag を足さないので defs は変わらない。

### 5.6 checkpoint

- header の識別子: 有効なら `DefaultDQNAgent;intrinsic` または `DefaultDQNAgent-2;intrinsic`。無効なら今のまま。
- `intrinsic` 節（learner の後）: RND の本体（書式の識別子 `RndNoveltyEstimator-3`、統計のモード、target の重み、predictor の重み、optimizer の状態、統計（`whiten` は観測の正規化の統計と μ_e・σ_e と件数、`variance_feedback` は obs_m・obs_v・err_m・err_v・n）、target の出力の統計、採点した観測の件数）の後に、facade の credit の端数・lane 抽選の乱数状態・累計回数。RND の本体を先に置くのは、形の違う checkpoint を RND の書式の検査で ERROR にするため。書式が `RndNoveltyEstimator-3` でないもの、統計のモードが設定と違うものは ERROR（checkpoint の値と設定の値を含む）。
- 読込の判定（上から順に当てる）: (1) 識別子が既知の形でない → ERROR。(2) 設定が有効で節が無い → ERROR。(3) 設定が無効で節がある → ERROR。(4) 節があり、RND の書式か統計のモードが設定と違う → ERROR。(5) それ以外 → 節があれば復元し、無ければ今の読込のまま。
- optimizer の options は [ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) と同じく現在の設定（`learning_rate`・`weight_decay`・`adam_eps`）で置き換え、違えば WARN し、moment と step を引き継ぐ。

### 5.7 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
| --- | --- | --- | --- |
| `LifelongNoveltyEstimator` | 新規（起票時。2026-11-07 の改訂では変えない） | §5.1 の 4 操作。方式に依らない一生の新しさの入口 | RL 共通基盤の内的報酬（`anet::rl::intrinsic`、`include/anet/intrinsic.hpp`） |
| `LifelongNoveltyEstimatorFactory` | 新規（起票時。2026-11-07 の改訂では変えない） | 設定の接頭辞・観測の仕様・device・seed から口の実体を作る。葉の読み込みと検証 | 同上（`include/anet/intrinsic.hpp`、実装は `src/intrinsic_rnd.cpp`） |
| `RndNoveltyEstimator` | 新規（起票時）→ 変更（2026-11-07） | §5.2 の RND。別々に組んだ target・predictor、optimizer、モード別の統計、target の出力の統計を所有し、保存する | 同上（`src/intrinsic_rnd.hpp`・`src/intrinsic_rnd.cpp`。DQN からは見えない） |
| `RndNoveltyEstimatorConfig` | 新規（起票時）→ 変更（2026-11-07） | `intrinsic.rnd.*` の DTO と値域の検証（`target.net`・`predictor.net`・`head.init`・`multiplier_mode`・`statistics_mode`・`adam_eps` を含む） | 同上（`src/intrinsic_rnd.hpp`） |
| `VarianceFeedbackStatistics` | 新規（非公開） | `variance_feedback` の obs_m・obs_v・err_m・err_v・n の更新・読み出し・保存（§5.2 の式） | 同上（`src/intrinsic_rnd.hpp`・`src/intrinsic_rnd.cpp`） |
| `DefaultDQNAgent` | 変更 | 有効時の生成、採点する観測の選択と mask の有無（`intrinsic.observation`）、採点・畳み込み・補助学習の credit と lane の抽選（§5.3）、`GetScalar` の `intrinsic.*`（`bonus_share` の式と `bonus_min`・`bonus_pos_ratio` を含む）、checkpoint の節 | DefaultDQNAgent |
| `DefaultDQNAgentConfig` | 変更 | `intrinsic.*` の節（`enabled`・`beta`・`observation`・`learn_ratio`・`learn_batch_size`・`rnd`） | DefaultDQNAgent |

## 6. 受入

### 6.1 無効時の同一性

- PRD098 の段 0 の手順（[testdata/prd098/README.md](../../core/anet-core/testdata/prd098/README.md)）で、着手前に保存した実行体と新しい実行体を、Atari・DropMerge・GridMaze・LunarLander の既定（`intrinsic.enabled=false`）で動かし、全 scalar 点の checksum と `metrics.scalar.defs` が一致する。
- 有効の checkpoint の読み書きが無効の checkpoint のバイト配置を変えない（無効で保存したファイルが着手前と同一）。

### 6.2 正しさ（anet-core-test、タグ `[prd116]`）

| 観点 | 観測する結果 |
| --- | --- |
| target の固定 | 同じ入力で `Learn` を 100 回進めても target の出力と重みが変わらない |
| predictor の学習 | 同じ入力で `Learn` を進めると predictor の loss が単調に近く下がり、e_t が下がる |
| α の値域 | `clipped` では任意の入力で α が有限かつ [1, L]。`unclipped` では有限で、平均より小さい誤差の lane が 1 未満（負の内的報酬）になる。割る値が 0 のとき α=1。warm-up 中はどちらも 1 |
| warm-up | 採点した観測の件数が `warmup_steps` に達するまでの `Estimate` が全 lane で 1 を返し（N を変えても同じ exp step 数で終わる）、μ_e・σ_e・正規化の統計は更新されている。その間の `Learn` は何もしない |
| 統計の順 | 採点が更新前の統計で行われる（同じ入力を 2 回渡した 1 回目と 2 回目の α が、更新順どおりの値になる） |
| 操作の独立 | `Learn` に `Estimate` と別の観測を渡すと、その観測の loss で学習する。`Estimate` を 2 回続けても、`Learn` を単独で呼んでも、状態の取り残しが無い |
| device | 入力を CUDA にすると α も CUDA で返り、CPU 入力の結果と一致する（CUDA が無い環境では skip） |
| Save の時点 | `Estimate` と `Learn` の間に Save/Load を挟んでも、以後の α と loss が挟まない場合と一致する |
| 刻みの不変 | 同じ exp step 数（N × intake 数）を N=2 と N=4 で流すと、`learn_count` と `Learn` に渡した観測の総件数が一致する。1 intake の step 数が ⌊N ÷ k⌋ を超えず、credit の端数が次の intake へ繰り越される |
| 抽選 | 同じ intake の中で同じ lane を 2 度渡さない。`learn_batch_size` > N のとき N 件で学習し、WARN が 1 回出る |
| 終端のマスク | `next_state`（既定）で、`next_state.done` の lane の学習報酬がスケール済み外的報酬と一致し、truncated の lane には内的報酬が付く |
| 採点する観測 | `intrinsic.observation=state` で `state.obs` を採点・学習し（`Estimate`・`Learn` に渡る観測が `state.obs`）、done の lane にも内的報酬が付く |
| 層カタログ | `Conv2d_S > ReLU > Conv2d_M > ReLU > Flatten` を Atari の観測（1×84×84）で組むと Flatten が 6,400 で、Head 128 のとき 1 Network 854,240 パラメータ。`Conv1d_S > ReLU > Conv1d_M > ReLU > Flatten` を 1 チャンネル × 長さ 84 の観測で組むと Flatten が 640（長さ 84 → 20 → 10、64 チャンネル） |
| 出荷設定の形 | Atari の `run.@rnd` を解決して組むと、target が 1,677,984、predictor が 2,203,296 パラメータになる（`[shipped_config]`）。`run.@rnd_taiga` を重ねた解決で、§5.4 の 8 葉がプロファイルの値になる |
| 別々の構造と Head の初期値 | `target.net` と `predictor.net` が別の形を取れ、形が違っても `Estimate`・`Learn` が通る。`head.init.mode` を変えると同じ seed でも Head の重みが変わる。片方の `net` が無いと ERROR になり、無いキーを含む |
| `variance_feedback` の式 | §5.2 の式を手計算した 3 step（N=1）の値と一致する。N=2 では lane 順の更新になる（lane 1 の統計が lane 0 の更新後の値）。保存→復元で統計の続きが一致する |
| 読まない葉の WARN | `unclipped` で `multiplier_max` を実装の既定 5 以外（例 10）にすると WARN が 1 回だけ出て、5 と書いても出ない。`variance_feedback` で `obs_norm.use_centering` を実装の既定（true）以外にすると WARN が 1 回だけ出て、`clip_range` を変えても出ない |
| 既定の一致 | `agent.txt` の `@baseline` の `intrinsic.rnd.multiplier_max` と `intrinsic.rnd.obs_norm.clip_range` の `?=` の値が、RND の設定 DTO の既定（5・5）と一致する（`[shipped_config]`。読まない葉の WARN が `@baseline` のままで出ないことの前提） |
| 診断の追加 | `bonus_min`・`bonus_pos_ratio`・`target_feature_var`・`target_feature_max` が `GetScalar` で読める。負の内的報酬でも `bonus_share` が [0, 1] に収まる |
| Adam の eps | `adam_eps` が optimizer の options に入り、読込で現在の設定に置き換わる（違えば WARN） |
| 畳み込みの位置 | 内側の Learner が受ける報酬 = reward scaler の出力 ＋ β × (α − 1)（偽の Learner で受け取って比べる） |
| 保存・復元 | 保存→復元の前後で、同じ入力の α と、μ_e・σ_e・正規化の統計・採点した観測の件数・credit の端数・抽選の列が一致する |
| 読込の判定 | §5.6 の (1)〜(4) がそれぞれ ERROR になり、メッセージに識別子（書式・統計のモード）と設定の状態を含む |
| 設定の検証 | §5.4 の表の各行で、反した値が ERROR になり、キー・値・期待が含まれる。`enabled=false` では他の葉を読まない |
| seed の再現 | 同じ seed で 2 回構築した target の出力が一致し、既存の名前付き seed の派生と消費列が変わらない（無効時の同一性で確かめる） |
| メトリクス | 有効時に §5.5 の key が `GetScalar` で読め、無効時は `std::nullopt` |

### 6.3 性能

- 群は有効（`run.@rnd`）と無効の 2 つに、Atari では `run.@rnd > run.@rnd_taiga` を足した 3 つ。群を交互に（ABBA の要領の巡回で）3 巡以上回し、群ごとの実時間の中央値と Tracy の内訳（target forward・predictor forward・backward・統計・転送）を記録する。環境は Atari（B=64）と GridMaze。
- 合否の閾値は置かない。数値は基準 Run の事前測定と本測定を見て 924 問い 20 で置く。本 PRD では内訳と実時間を記録に残すことを受入にする。

### 6.4 腕の smoke

smoke はどれも seed 1 本・100k exp step の 1 Run で、合否は各箇条の条件を、その Run の記録された点のうち `56_agent_in/32_learn_count` が 0 より大きい点（warm-up 後。`learn_count` が 0 の点は除く）で満たすこと。「有限」の条件は、§5.5 が NaN を定める場合を除いて見る: `03_bonus_share` は同じ点の `01_bonus_mean` が 0 のとき、`31_predictor_loss` は同じ点の `32_learn_count` が 0 のときの NaN を許し、それ以外の NaN は否。機構の確認であり成績の判定ではない。

- GridMaze で `run.@rnd` を回し、落ちない・群 56 の全指標が全点で有限（上の NaN の例外を除く）・`56_agent_in/32_learn_count` が 0 より大きい点がある。bonus の符号と減少は判定しない（2026-11-07 に外した。観測が正規化した (x, y) の 2 値で、盤面は `seed_mode = process` で固定なので、RND の入力は 25 通り以下しか無い。ε = 1 から始まる探索が最初の数千 step で全部を踏み、predictor が覚え切って一生の誤差平均を下回り続け、既定（`clipped`・`whiten`）の bonus は 0 になる。「同じ盤面に慣れる」は確かめられるが、新しさの判別の確認には使えない。結果は §7 の GridMaze の行）。
- Atari（KFM）で `run.@rnd` を回し、落ちない・群 56 の全指標が全点で有限（上の NaN の例外を除く）・`56_agent_in/21_clip_ratio` の全点が 1 未満。2026-11-07 の改訂後は公開実装の形の既定の Network で取り直す。
- Montezuma で `run.@rainbow_like > run.@rnd > run.@rnd_taiga` を回し、落ちない・群 56 の全指標が全点で有限（上の NaN の例外を除く）・`56_agent_in/04_bonus_min` の点のうち負の点が 9 割以上（z の負が報酬に入っている。warm-up 0 なので最初の点から数える）。

## 7. 成功の測り方

効果の判定は [924 §9.1](924_intrinsic_reward_novelty_10prd.md) に従い、本 PRD では機構の指標（§6.4）までを受入にする。924 の腕として回す Run の候補は次のとおりで、合否の数値は 924 問い 20・21 で置く。

| Run | 見るもの | 本 PRD での位置づけ |
| --- | --- | --- |
| GridMaze: 通っていないセルからの到達率 | RND だけで収束後も被覆が続くか。924 §2.7 の見立ては「続かない」。2026-11-07 の Run（256 lane・400k exp step、`run_20261107-212611`）では見立てどおり続かなかった: bonus は warm-up 直後の 9 点以外 0、predictor の誤差は 1e-9、手で通っていないセルへ誘導すると穴へ入る（未踏ではなく off-path の Q の粗さ。§8 の 18） | 判断に効く。続くなら方策の組の優先度が下がる |
| LunarLander: 既定との比較 | 害の確認（易しい env で成績を落とさないか）。2026-11-08 の `run.@rnd`（β 0.1）3 seed × 2M は、評価側（UQE full `21_eval/02_policy_reward`）の 1.5〜2M が 181.1〜225.5 で、11/05 の基準 9 本の 188.6〜225.7 と同じ帯。害は見えない。ただし bonus は採点 5〜10 万件で消えた（predictor の誤差が whiten の一生平均を下回り続ける。GridMaze と同じ構造）ので、bonus が続く状態での害は未確認。[実験記録](../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md) | 924 §9.1 の害の確認 |
| KFM 10M × 数 seed、16 seed の基準と比較 | 挑む窓の本数、撃破÷命中、費用 | 対照腕。924 §2.5 の見立てでは壁は越えない |
| Montezuma: `run.@rainbow_like > run.@rnd > run.@rnd_taiga`、評価側の Greedy の平均点 | Taiga らの Rainbow＋RND（Montezuma）と並ぶ範囲に入るか。寄せない差は §2.1 の表 | 本家再現の確認。合否の数値は 924 問い 21 で置く |

## 8. 未決表（924 の再グリルで決める）

| # | 事項 | 暫定の既定と候補 |
| --- | --- | --- |
| 1 | 内的報酬の式 | 既定は α−1（NGU の倍率の形、`clipped`）。Taiga らの z をそのまま使う形は `multiplier_mode=unclipped` にした（2026-11-07）。残る候補: RND 原典の e_t／σ_e（平均を引かない、上限なし）。差は尺度と下限 |
| 2 | target と predictor の形（Atari の既定） | 2026-11-07 に別々の Network 設定にして決着（D8）。残るのは Atari の既定の形で、暫定は公開実装の形（predictor に FC 512 ×2、出力 512）。候補: NGU の同形・出力 128。写しの比較では同形のほうが誤差の比が大きく（§2.1 の C・D）、長い学習での優劣は 924 の腕で確かめる |
| 3 | 誤差統計の窓 | 既定は一生の累積。候補: EMA（係数をキーに） |
| 4 | 補助学習の刻みと件数の既定値 | 既定は `learn_ratio` 0.25・`learn_batch_size` 32（N=128 の旧既定と同じ量。exp step あたり 1/128 回・32 件）。候補: Taiga らと同じ毎 exp step 1 回・1 件（`learn_ratio` 1・`learn_batch_size` 1。費用は N 倍の backward）、件数 64 で回数を半分。比較対象との差は §8.1 の 6。指標は内的報酬が消える速さ（§6.4） |
| 5 | warm-up の既定値 | 既定は採点した観測 12,800 件（N=128 の旧既定と同じ長さ）。候補: RND 原典の正規化の初期化の長さに揃える |
| 6 | 終端遷移の内的報酬 | 既定は 0。候補: 採点して付ける（終端観測は facade に届く） |
| 7 | β を掛ける位置 | 既定は reward scaler の後。候補: scaler の前（スケールに乗せる） |
| 8 | 評価での採点 | 本 PRD では無し（構造上届かない）。924 の採点点の問いで決める |
| 9 | ADR 0062 の適用範囲の注記と、RND の Network の所有・補助学習の置き場と刻み（§8.1）を記す新 ADR | 本 PRD では起こさない。924 で起こす |
| 10 | 用語の CONTEXT.md への追加 | §3 の 3 語。924 §8 の 20 と一緒に確定する |
| 11 | 仮の運び方（D2）を 924 の注釈経路へ置き換える時期 | 924 の Replay 契約の段が入るとき。置き換えは同じ変更で行い、畳む経路を残さない |
| 12 | 補助学習の母集団と置き場 | 既定は Push 側（facade が採点した intake の観測を exp step 基準の credit で渡す）。候補: Learner の step で Replay から抽出した観測を渡す（NGU の置き場）。2 つは等価でない（§8.1）。924 問い 13 と一緒に決める |
| 13 | NGU 対応後に Push 側（intake での補助学習）を残すか | 既定は 924 の移行時にクリーンブレークで消す（1 関心 1 機構。RND 単独の腕は NGU の機構でエピソード内の β を 0 にして作る）。候補: 即時に反応する一生の新しさが要るなら残す。§8.1 |
| 14 | 本家との残る差（§2.1 の表）をどこまで寄せるか | 既定は寄せない（C51・padding・Q の初期値・PER の細部・`Learn` の入力・n の数え方・実行環境）。`run.@rainbow_like` と `run.@rnd_taiga` の結果で再現が成らなければ、原典のコードを WSL で動かす再現環境を作って差を切り分ける（実験の結果が要る） |
| 15 | Head の葉の置き場 | `intrinsic.rnd.output_dim` は `rnd` 直下、初期値は `rnd.head.init.*`。Head の設定の root の扱い（設定 root 所有の PRD、Head の `net.head` 候補）が決まれば揃える |
| 16 | 設定の読まれない葉の検出 | 本 PRD の範囲外（§2.1 の不具合は 4 block の葉を直して閉じる）。`net.block` の未知の葉を構築時に検出する仕組みは [issue-20261107-0c34dd](../management/issues/issue-20261107-0c34dd.md) に起票した（2026-11-07、開発者の選択） |
| 17 | `learn_batch_size` > N のときの縮小 | 既定は N 件で学習し構築時に WARN 1 回（2026-11-07 未明の改訂）。レビューで、明示した値を N へ丸めて続ける挙動が AGENTS.md §設定値の扱いの clamp に当たると指摘された（2026-11-07 昼）。候補: ERROR にして N 以下を要求する。924 で `learn_batch_size` の既定値（§8 の 4）と一緒に決める |
| 18 | predictor の周期リセット（ユーザー案、2026-11-07） | 未決。一生の新しさが尽きた盤面で「最近の経験に対する新しさ」を作る案。predictor と誤差統計を一定間隔で初期化し warm-up をやり直す（target は固定のまま）。初期化の直後は predictor が訪問の多い状態から先に覚えるので、訪問の少ない状態の誤差がしばらく高く残り、その間だけ訪問回数の逆順に近い bonus が出る。統計を一緒に初期化しないと直後に全 lane が α = L に張り付く。近い事例は Q 側の周期リセット（primacy bias、BBF）とエピソード単位のリセット（NGU のエピソード記憶、E3B、NovelD のエピソード内ゲート）で、RND の predictor を周期的に初期化する既知の事例は 2026-11-07 時点で見つけていない。3 の EMA だけでは収束し切った誤差の差（1e-9）を拾えない。候補の葉: `intrinsic.rnd.reset.mode ?= none`（none｜predictor）と `intrinsic.rnd.reset.interval`（learn step）。エピソード内の新しさ（924）との優劣は 924 で決める。2026-11-08 に暫定実装（50,000 件ごと）で試した: GridMaze 1 本では warm-up 直後の窓にしか bonus が出ず、LunarLander 3 seed × 2M では bonus は続くが評価側の終盤は同じ帯で利得が無く、1 seed が途中で崩れて Q と勾配が膨らんだ。実装は戻し、考えと結果は [issue-20261108-227b05](../management/issues/issue-20261108-227b05.md) と[実験記録](../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md)に残した |

### 8.1 補助学習の置き場と刻みの経緯（2026-11-07）

2026-11-06 夜に口を独立させた（冒頭の記録）あと、11/07 未明の問答で決めた。発言と判断を分けて残す。

1. 事実の整理（Claude）: Push 側（facade の intake）の補助学習は RR と replay batch に元から依らない。lane 数 N には 3 か所で依っていた。1 step の件数（割合 × N）、exp step あたりの step 数（1 ÷ N）、warm-up の長さ（採点回数 × N）。RND 原典（§3.4）の間引きは、環境数を増やしたときに predictor の学習に使う経験を 32 環境ぶんに固定する手動の補正で、128 環境のとき割合 0.25。anet はこの 0.25 を割合のまま持ち込んでいた。
2. ユーザーの発言の順: (a) N は env ごとにほぼ固定で RR は頻繁に変えるので、RR に引きずられない形がよい。(b) RND は NGU への通過点で、分散学習も将来の候補なので、NGU 方式（Learner の step で Replay から学習する Sample 側）を強く意識したい。部品としての RND ができていれば NGU 対応は問題ない。残る論点は NGU 対応後に Push 側を潰すか。(c) NGU 完成までは特性研究として RND の評価を進めるので、比較対象と実績のある Push 側で進める。無くす可能性が高くても、N に依存しない形で実装する。
3. Push 側と Sample 側は等価ではない（Claude の整理）。頻度の単位（exp step／learn step）、母集団（今の方策の観測を多くて 1 回／Replay の混合を RR 回ほど）、反応の速さ（採点の直後／Replay から引かれるまで遅れる）、忘れ方（方策が行かなくなった領域を忘れる／buffer にある間は覚える）。分散処理との相性は直感と逆で、actor-learner 型は勾配計算を learner に集めるため Sample 側になる。NGU の置き場は構成の都合で、アルゴリズムの必然ではない（原典の付録 A: actor が埋め込み網で r^i_t を計算して Replay へ入れ、learner が抽出した系列の末尾 5 フレームで行動予測網と RND の predictor を学習する）。RND 原典は同期型で Push 側。anet は同期型で facade が両方を持つ。
4. 不変性と置き場は別（Claude の整理）。exp step 基準の credit を facade に置けば、Sample 側でも RR に不変にできる。RR に依るのは学習が起きる時刻の粒度だけ。
5. 判断（本改訂の暫定の既定）: 置き場は Push 側、刻みは facade の credit（Learner の U と同じ形）、`Learn` は渡された観測で 1 step、母集団は intake の観測。924 で変えるとすれば母集団の 1 か所だけで、credit と `Learn` は残る。NGU 対応後に Push 側を消すかは §8 の 13。
6. 比較対象の確認（2026-11-07。原典のコード [bonus_based_exploration](https://github.com/google-research/google-research/tree/master/bonus_based_exploration) の `intrinsic_motivation/intrinsic_rewards.py`・`base_intrinsic_agent.py`・`configs/rainbow_rnd.gin`）: Taiga らの Rainbow＋RND は Push 側。agent の `step`・`end_episode` が、行動した観測 1 件で `compute_intrinsic_reward` を呼び、その中で eval でなければ predictor の学習 op を 1 回走らせ（batch 1）、観測と誤差の移動統計を更新し、(誤差 − 平均) ÷ 分散に相当する値 × `reward_scale`（0.00005）を外的報酬（Dopamine の Runner が [−1, 1] に clip 済み）に足し、合計は clip せずに Replay に入れる（2026-11-07 に訂正。起票時は「÷ 標準偏差」「合計を [−1, 1] に clip」と書いていたが、`RNDRainbowAgent.clip_reward = False` と `_update_moments` の再帰を読み違えていた。式の詳細は §2.1）。Rainbow の `update_period` 4・batch 32 は Q の学習だけに効き、predictor は毎 agent step 更新する。採点する観測は行動した観測 x_t（anet と RND 原典は次観測）。exp step あたりの predictor の step 数と 1 step の件数で並べると、Taiga 1 回・1 件、anet の既定 1/128 回・32 件、RND 原典 約 1/1,000 回・約 1,000 件（rollout 128 × 128、4 epoch × 4 minibatch、割合 0.25 から計算）、NGU 約 1/13,000 回・320 件（learner 5 更新/秒 × 64 系列 × 末尾 5 フレーム、actor 256 × 260 step/秒から計算）。Adam の進みは lr × step 数が主なので、Taiga の predictor は anet の既定より 2 桁速く慣れる。KFM・Montezuma の結果を Taiga らと並べるときはこの差を添える。
7. 本家再現への寄せ（2026-11-07 昼）: Atari の Run を担当した別の枠（Codex）が、RND ありの Run の checkpoint が 1.7GB 大きいことから、Atari の RND の Network が層カタログの葉の書き方の不具合で 1 億パラメータの別物になっていたことを見つけ（§2.1）、Taiga らの Rainbow＋RND の公開コードを読んで §8.1 の 6 の誤りと、再現に要る条件（行動した観測、分散に相当する値で割る統計、clip しない合計、毎 step 1 件の学習、公開実装の形の Network と直交初期値）を整理し、写しの比較（§2.1）で形と初期値の効き方を見た。その枠が推奨した内容（C51 は入れない、`run.@rnd` の既定の Network を公開実装の形へ、Taiga らの統計はモードとして入れて既定にしない、WSL の再現は保留）をユーザーが「一旦おすすめの内容で」と受け、本改訂は Claude が原典のコードで式を確かめたうえで PRD へ写した。D4・D5・D8 の既定を変えずにモード値と設定で選べる形にしたのは、924 の前提（NGU の倍率・同じ統計）を崩さず、比較対象と揃えた腕を同じ実装で回すためである。各判断は暫定の既定で、合意した決定ではない。

## 9. 移行対象

新しい契約へ移すもの（同じ変更で更新する）。旧契約の互換層は作らない。

- コード: `core/anet-core/include/anet/intrinsic.hpp`（新規、口と Builder）、`src/intrinsic_rnd.hpp`・`src/intrinsic_rnd.cpp`（新規、RND の本体）、`default_dqn_agent.hpp`／`.cpp`（`enabled`・`beta` の読み込み、採点・畳み込み、checkpoint の節、GetScalar）、`CMakeLists.txt`。2026-11-07 の改訂: `intrinsic_rnd.hpp`／`.cpp`（2 つの Network 設定、Head の初期値、倍率と統計のモード、`adam_eps`、target の出力の統計、書式 3）、`default_dqn_agent.hpp`／`.cpp`（`intrinsic.observation` の読み込みと採点する観測の選択・mask、`bonus_share` の式、`bonus_min`・`bonus_pos_ratio`）。
- テスト: `core/anet-core/src/intrinsic_test.cpp`（新規、`[prd116]`）、既存の DefaultDQN の checkpoint テストは無効時の形を変えないので断定は変わらない（`DefaultDQNAgent` の header の断定を検索して確かめる）。2026-11-07 の改訂: §6.2 の追加行に対応するテスト（出荷設定の形と既定の一致は `[shipped_config]`）。既存の `[prd116]` テストの `intrinsic.rnd.net` と `RndNoveltyEstimator-2` の断定は、2 つの Network 設定と書式 3 に読み替える。
- 設定: `apps/runner/config/agent.txt`（`@baseline` の既定。2026-11-07 未明の改訂で `learn_ratio`・`learn_batch_size` を足し、`train_fraction` を消し、`warmup_steps` を件数へ）、`Atari.txt`・`GridMaze.txt`・`LunarLander.txt`（`run.@rnd` の腕と tag。`32_learn_count` を足す）。2026-11-07 昼の改訂: `nn.txt` の `Conv1d_S`・`Conv1d_M`・`Conv2d_S`・`Conv2d_M` の葉を `conv.*` へ（旧キーの別名・互換読み・案内は残さない）、`agent.txt` の `@baseline` に `observation`・`head.init.*`・`multiplier_mode`・`statistics_mode`・`adam_eps`、`Atari.txt` に `AtariRnd*` の層カタログ・`run.@rnd` の 2 つの Network（`output_dim` 512、Head の初期値）・`run.@rnd_taiga`・tag 4 つ（`run.@rainbow_like`・`run.@rainbow_like_impala`・`run.@no_munch` は 2026-11-07 に別の枠が追加済み）、`GridMaze.txt`・`LunarLander.txt` の `run.@rnd` を 2 つの Network と tag 4 つへ。`core/anet-core/testdata/prd092/config/` と `docs/experiments/*/config/` にある Run 設定の写し（同じ葉を含む）は記録なので触らない。
- 検証資材: `core/anet-core/testdata/prd116/README.md`（段 0 の比較手順の参照と、§6.3・§6.4 の手順。2026-11-07 昼の改訂で 23impl の案内と Montezuma の smoke を足す）。
- 設計文書: [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) の §6.2（前処理に内的報酬の採点を足す）、§7.2（設定グループに「内的報酬」）、§8.2（`intrinsic` 節と識別子）、§9.1（群 56）。2026-11-07 未明の改訂では §6.2 の RND の段落（採点のグラフの再利用・独立 RNG の lane 抽選・採点回数の warm-up は旧い記述）と §7.2 の葉を日英とも現在形に直す。2026-11-07 昼の改訂では §6.2（採点する観測・倍率と統計のモード・別々に組む 2 つの Network）、§7.2（葉）、§8.2（書式 3 と統計のモード）、§9.1（診断 4 つと `bonus_share` の式）を日英とも直す。
- 設計文書の新設: [212_rnd_lifelong_novelty.jp.md](../design/212_rnd_lifelong_novelty.jp.md)（RND の概念・計算・設定・指標の現在形。2026-11-06 にユーザーが実装中に要望し、起票時の本節に書き漏れていた項目として追記した。英訳は別途。2026-11-07 未明の改訂では §3.4〜§3.6・§3.8・§4.2・§4.4・§4.5・§5.1・§5.2 を追随させる。2026-11-07 昼の改訂では §3.1〜§3.3（別々に組む 2 つの Network、統計と倍率のモード、採点する観測）、§3.5（warm-up 0）、§4.2（葉の表）、§4.3（Atari の構造と層カタログの名、`run.@rnd_taiga`）、§4.4（診断 4 つ、`bonus_share`）、§4.5（書式 3）、§5.1〜§5.3（費用・検証の記録・まだ言えないこと）を追随させる。英語版は anet-translate-docs で別途）。
- 実装記録と索引: `docs/memo/116_rnd_lifelong_novelty_23impl.md`（新規。§2.1 の「本家との残る差」の表を写す）、索引 issue [issue-20261106-16b7af](../management/issues/issue-20261106-16b7af.md) の対応内容に 1 行。
- wiki: [intrinsic_novelty](../wiki/mechanisms/intrinsic_novelty.md) に本 PRD を出典として足す（起票時に実施。2026-11-07 昼の改訂で Taiga らの式と、写しの比較で分かった尺度の条件を足した）。

## 10. 924 との関係と、後で消すもの

- 本 PRD は 924 の部品（一生の新しさの口と RND）を先に作り、924 の比較腕「RND 単独」を成立させる。924 §8 の 7・9・18 の一部（RND の統計・倍率・保存）に暫定の既定を与える。
- 924 の再グリルで採点点が Actor になれば、`Estimate` を呼ぶ側が facade から Actor に移り、口はそのまま残る。Actor 側で統計を更新しない採点が要るなら、口に「更新なしの採点」を足す（924 で決める）。
- D2 の畳む経路は仮で、924 の Replay の注釈経路が入る変更で削除する。残すと「外的・内的を別に保存する」契約と二重になる。
- 補助学習の置き場は Push 側のまま 924 へ持ち越す。Sample 側（Learner の step で Replay から）へ変えるか、Push 側を消すかは 924 で決める（§8 の 12・13、§8.1）。credit と `Learn` の形は置き場に依らず残る。

## 11. 参照

- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.2（構造）、§2.4（観測と報酬の正規化、Algorithm 1）、§3.4（学習データの間引き）。
- Badia ら、[Never Give Up](https://arxiv.org/abs/2002.06038)、ICLR 2020。§2 式 1（α の clip）、付録 H.2（RND の構造、出力 128）、付録 B.3（L=5）、付録 A（actor が r^i_t を計算して Replay へ入れ、learner が抽出した系列の末尾 5 フレームで predictor を学習する）。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。易しいゲームでの害の確認の条件。公開コードは [bonus_based_exploration](https://github.com/google-research/google-research/tree/master/bonus_based_exploration)（predictor の学習の置き場と式は §8.1 の 6・§2.1）。2026-11-07 に読んだファイル: `intrinsic_motivation/intrinsic_rewards.py`（`RNDIntrinsicReward` の `_preprocess`・`_update_moments`・`_build_train_op`・`compute_intrinsic_reward`・`_target_network`・`_prediction_network`）、`intrinsic_motivation/base_intrinsic_agent.py`（`step`・`end_episode` が渡す観測）、`intrinsic_motivation/intrinsic_dqn_agent.py`（`RNDDQNAgent._add_intrinsic_reward` の `clip_reward`）、`intrinsic_motivation/intrinsic_rainbow_agent.py`、`configs/rainbow_rnd.gin`。あわせて Dopamine の `discrete_domains/run_experiment.py`（`clip_rewards` の既定 true）と `tf/agents/dqn/dqn_agent.py`（`begin_episode`・`step` の `_train_step` で `training_steps` が増える）。
- Burda らの公開コード [random-network-distillation](https://github.com/openai/random-network-distillation)（`policies/cnn_policy_param_matched.py`・`utils.py`。`feat_var`・`max_feat` の監視）。CleanRL の [ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)。形と初期値は §2.1 の表。
- [924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md) §2.5・§4.3・§7.1・§8・§9.1・§10。
