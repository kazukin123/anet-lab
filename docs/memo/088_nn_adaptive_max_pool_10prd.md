# 出力の大きさで指定する max pooling（AdaptiveMaxPool2d）暫定 PRD

> 2026-10-09 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した（暫定の既定を置いてある）。
> 2026-10-23 改訂: Run 名を今のフォルダ名に合わせた（2026-10-15〜17 の改名。対応は DropMerge の workspace の改名の対応表）。中身は変えていない。
> 起点: DropMerge の NN で、行動（NOOP と 58 列への DROP）の列の位置を `Flatten > Linear_2048` が消している。列を残したまま縦方向だけを畳みたいが、今の `MaxPool2d` は正方形のカーネルしか書けず、出力の大きさで指定する pooling も無いので、設定では書けない。Atari でも、BTR の `AdaptiveMaxPool2d(6)` を `MaxPool2d` の組み合わせで近似してきた（`AtariPool6`）。
> 関連: [Atari: BTR の NN 構成の移植検証](../experiments/default-dqn/atari/2026-09-12_btr-network-port.md)、[Atari: BTR との差分の腕](../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md)、[Atari-5 の記録](../experiments/default-dqn/atari/2026-08-27_atari5.md)、[DropMerge の探索記録](../experiments/default-dqn/dropmerge/README.md)、[DropMerge の長期 Run](../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、[BTR のコード読み](../../reports/btr_code_reading_2026-08-31.md)、[設計 130 ニューラルネットワーク](../design/130_neural_networks.jp.md)、[084_noisynet_10prd.md](084_noisynet_10prd.md)（DropMerge で NEET の無い基準を作った NoisyNet）。

## 1. 目的と問題

出力の高さと幅を指定する max pooling のブロック `AdaptiveMaxPool2d` を、NN の設定で使えるようにする。入力の大きさに依らずに出力の大きさを決められる。用途は次の 2 つ。

1. **DropMerge: 列の位置を残す縮約**
   - 行動は、行動 0 が NOOP、1〜58 が盤面の列への DROP である（`core/envs/dropmerge1/src/DropMergeEnv.cpp:994-996`）。
   - 今の NN は、盤面（46×58）を畳み込みで 12×15 に縮め、Transformer で 180 トークン（12 行 × 15 列）として処理する。Transformer を出た時点では、列の位置が残っている。
   - その直後の `Flatten > Linear_2048` で列の位置が消える。V と A のストリームは、2048 次元の特徴から列を学び直している。この 1 層だけで 47,187,968 パラメータあり、ネット全体（約 53.3M）の 89% を占める。
   - 縦方向だけを畳んで列を残す形（例: 12×15 → 2×15）にしたいが、今のブロックでは書けない。
2. **Atari: BTR の pooling をそのまま書く**
   - BTR は、畳み込みの後に `AdaptiveMaxPool2d(6)` を置く。今の Atari の設定は、これを `MaxPool2d`（kernel 2、stride 2、padding 1）で近似している（`AtariPool6`）。出力が 6×6 になるのは 11×11 の入力のときだけで、窓の切り方も BTR とは違う（§2.3）。

**ゴール:** `AdaptiveMaxPool2d` を NN のブロックとして足し、DropMerge で列を残す縮約を設定で書けるようにする。既存の設定（`MaxPool2d` を使うものを含む）は、今と同じに動く。

**非ゴール:** 平均の pooling、正方形でない `MaxPool2d`、DropMerge で NN 構成を比べる実験そのもの、トークン列を盤面へ戻す専用のブロック（§6）。

## 2. 現状

### 2.1 コードで確認したこと

- NN のブロックは、`core/anet-core/src/nn_modules.cpp` の登録表で名前と実装が結ばれる。pooling は `MaxPool2d` だけである（同 `:3181`）。出力の大きさで指定する pooling と、平均の pooling は無い。
- `MaxPool2d` の設定は `pool.kernel_size`・`pool.stride`・`pool.padding` で、どれも 1 つの整数（正方形）である（同 `:2712-2750`）。
- トークン列と盤面の変換に使えるブロックはある。
  - `SpatialPositionalEmbedding2D` は、`[B, C, H, W]` に位置埋め込みを足し、行優先のトークン列 `[B, H·W, C]` にして返す（同 `:1949`、`:1983` の `out.flatten(2).transpose(1, 2)`）。
  - `Permute` は batch を含む全次元を並べ替える。`Reshape` は batch 次元を保ったまま、残りの形を変える（同 `:734-743`、`:762-775`、登録は `:3162-3163`）。
  - したがって、Transformer の出力 `[B, 180, 128]` は、`Permute`（0 2 1）と `Reshape`（128 12 15）で `[B, 128, 12, 15]` に戻せる。行優先の並びはそのまま保たれる。
- `Linear` のブロックの設定は出力の次元だけを持ち（例: `apps/runner/config/DropMerge.txt:966-971`、`:973-975`）、入力の次元は前の段の出力で決まる。

### 2.2 DropMerge の NN（2026-10 の基準、`run.@iqn32_stratified`）

- **main_feature:** 盤面 46×58（`Embed5846_v2`、`DropMerge.txt:957-959`）を `DMImpalaConv64 > DMImpalaPool > DMImpalaRes64(*2) > DMImpalaConv128 > DMImpalaPool > DMImpalaRes128(*2) > SiLU > PosEmbed2D > DMImpalaTrans128 > Flatten > Linear_2048 > SiLU` で 2048 次元にする。
  - `DMImpalaPool` は `MaxPool2d`（kernel 3、stride 2、padding 1、`:1045-1048`）で、2 回で 46×58 → 23×29 → 12×15 になる。
  - `DMImpalaTrans128` は 4 層の Transformer（d_model 128、8 head、`:1146-1154`）。
- **IQN の τ:** `IQNCosEmb > IQNTauProj2048 > SiLU` で 2048 次元にし（`:1174`）、main_feature と要素ごとに掛ける（`:1175`）。
- **V と A:** それぞれ `HeadFC1024 > SiLU`（`:1168-1169`）。
- **パラメータ数:** 約 53.3M。保存時の `model_size`（426,098,754 バイト）を、online と target の 2 本分の float32 とみなして 8 で割った概数である。そのうち `Linear_2048` が 23,040 × 2,048 + 2,048 = 47,187,968。

### 2.3 Atari の背景（実験記録）

- **adaptive pooling だけが残っている:** 2026-08-27 時点で、BTR の構成要素のうち未実装だったのは、Spectral Norm・Munchausen・NoisyNet・Adaptive Maxpool・100 本の評価だった（[Atari-5 の記録](../experiments/default-dqn/atari/2026-08-27_atari5.md) の 33 行目）。その後 Spectral Norm（PRD065）、Munchausen（PRD067）、NoisyNet（PRD084）は実装済みで、adaptive pooling だけが残っている。
- **今の近似:** `AtariPool6` は、`MaxPool2d`（kernel 2、stride 2、padding 1）を Flatten の前に置き、11×11×64 = 7,744 を 6×6×64 = 2,304 にしている（[BTR との差分の腕](../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md) 探索ブロック 06、`apps/runner/config/Atari.txt:1016-1019`）。
  - 出力の大きさは BTR と同じだが、窓は違う。11 → 6 の場合、`AdaptiveMaxPool2d` の窓は幅 2〜3 で隣と重なる。`AtariPool6` の窓は幅 1〜2 で重ならない。
  - 4×4 の版（`AtariPool4`、kernel 3、stride 3、padding 1）も、11×11 の入力のときだけ 4×4 になる（[BTR NN 構成の移植検証](../experiments/default-dqn/atari/2026-09-12_btr-network-port.md) 探索ブロック 08）。
- **効果**（Breakout、RR4、[BTR NN 構成の移植検証](../experiments/default-dqn/atari/2026-09-12_btr-network-port.md)）:
  - BTR の NN 構成の部品のうち、Pool6 の寄与が最大だった（12–15M で +55.1）。RR1 では同じ追加が −28 で、符号が逆になる（まとめ 4、探索ブロック 02）。
  - 4×4 まで縮めると、6×6 よりさらに良かった（12–15M で 466.1 対 455.0、18–20M で 487.2 対 461.4）。差は後半ほど開き、BTR の 6×6 は最適点ではなかった（まとめ 4b、探索ブロック 08）。
  - BTR の NN 構成の利得は replay ratio に条件付きで、旧 net に対して RR1 で −54.1、RR2 で −38.2、RR4 で +133.9 だった（まとめ 13、探索ブロック 10）。
- **ここから言えること:** 出力の大きさは、入力の大きさから kernel・stride・padding を逆算するより、直接の値として振れる方が実験しやすい。4×4・6×6 のような用量反応を並べるときも同じである。

### 2.4 DropMerge の背景（2026-10 の Run）

- **RR4 は NEET で負けていた:** RR4 と、residual の畳み込みの Spectral Norm を、今の NN（`Flatten > Linear_2048`）で試した Run がある（`run_20261006-135602_dmfb_20_rr4sn`）。
  - 序盤は基準より 13% 速かった。しかし 10M から NEET（合法な DROP があるのに NOOP を続ける）に入り、100M まで抜けなかった。
  - 評価専用の UQE target は 1,522 で、基準（`run_20261005-130039_dmfb_16_ndt20`、1,806）より 16% 低かった（2026-10-08 集計）。
- **NEET の入口は ε のラダーだった:**
  - ε を NoisyNet に置き換えた Run（`run_20261008-015339_dmfb_21_nzrep`）は、100M まで NEET が出なかった。評価専用の成績は 1,873 / 1,788 / 1,844（UQE target / online / Greedy）で、基準と同等だった。
  - ε も NoisyNet も無い Run も NEET は出なかったが、成績は 1,644 に留まった（2026-10-09 集計）。
- **ここから言えること:** NEET の無い基準ができたので、Atari で効いた「高い replay ratio と、共有トランクを持たない構成」の組を、DropMerge で試せる状態になった。その NN を設定で書くのに、本 PRD のブロックが要る。
- **長期の目標との関係:** DropMerge の目標は、長期 Run（700M〜1000M）の最終成績である。7〜8 月の長期 Run（QR51、累計約 14 億 step）では、終盤の伸びの鈍化が課題だった（[DropMerge の長期 Run](../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)）。NN 構成はその打ち手の候補の 1 つで、効くかどうかは本 PRD の範囲外の実験で確かめる。

## 3. 提案の骨子

### 3.1 ブロック `AdaptiveMaxPool2d`

1. 入力 `[B, C, H, W]` を、`[B, C, H_out, W_out]` へ max pooling する。窓の切り方は libtorch の `AdaptiveMaxPool2d` に従う。
2. 設定キーは `pool.output_size` の 1 つだけ。高さと幅を、2 つの正の整数として空白区切りで書く（例: `pool.output_size = 2 15`）。
3. 不正な設定と入力は、ネットの組み立て時に、ブロック名と値を示して止める。
   - `pool.output_size` の値が 2 つでない、整数でない、0 以下。
   - 入力の階数が 4 でない（入力の形を示す）。
   - 出力の高さか幅が、入力より大きい（窓が 1 になって同じ値が並ぶだけになるので、黙って許さない）。
4. 「入力の大きさのまま」を表す特別な値は持たない。その次元を残したいときは、入力の大きさをそのまま書く（DropMerge なら幅に 15 と書く）。
5. AMP（BF16）の中でも、精度の扱いを変えずに動かす。

### 3.2 DropMerge での使い方（設定の例。採否は実験で決める）

Transformer の出力を盤面の形に戻し、縦方向だけを 2 段に畳む例である。

```text
net.block.[DMTokensToMap].type = Permute
net.block.[DMTokensToMap].permute.dims = 0 2 1
net.block.[DMMap12x15].type = Reshape
net.block.[DMMap12x15].reshape.dims = 128 12 15
net.block.[DMColPool2].type = AdaptiveMaxPool2d
net.block.[DMColPool2].pool.output_size = 2 15
net.block.[IQNTauProj3840].type = Linear
net.block.[IQNTauProj3840].linear.out_features = 3840

main_feature:  … > PosEmbed2D > DMImpalaTrans128 > DMTokensToMap > DMMap12x15 > DMColPool2 > Flatten
tau_embedding: IQNCosEmb > IQNTauProj3840 > SiLU
```

- main_feature の出力は 128 × 2 × 15 = 3,840 次元になる。τ の射影も 3,840 次元に合わせる。V と A の FC は、入力が 3,840 次元になる。
- パラメータは約 53.3M から約 9.9M に減る（`Linear_2048` の 47.2M が消え、V・A の FC が 4.2M → 7.9M、τ の射影が 0.13M → 0.25M に増える）。
- 縦の段数（1・2・3 段 → 1,920・3,840・5,760 次元）は、実験で振る値である。Atari では、出力を小さくした方が良かった（§2.3 の 4×4）。
- 横を 15 のまま残すと、トークン 1 列が盤面の約 4 列（58 ÷ 15）に当たる住所が残る。

### 3.3 Atari での使い方

- BTR と同じ `AdaptiveMaxPool2d(6)` を、既存の `AtariPool6` とは別の名前のブロックとして足す（例: `AtariAPool6`）。
- 既存の `AtariPool6`・`AtariPool4` と、それを使う profile は変えない。過去の Run と比べられるようにするためである。

## 4. User Stories

1. 研究者として、DropMerge の NN で、列の位置を残したまま縦方向だけを畳みたい。行動（落とす列）と特徴の位置を揃え、4,700 万パラメータの 1 層を外した構成を試すため。
2. 研究者として、pooling の出力の大きさを、入力の大きさに依らない値として振りたい。Atari の 4×4・6×6 のような用量反応を、kernel・stride・padding の逆算なしで並べるため。
3. 研究者として、BTR の NN 構成を、近似ではなくそのまま再現して比べたい。
4. フレームワークの利用者として、既存の設定と Run の結果が変わらないでほしい。

## 5. 決めること（詳細設計で詰める）

| # | 項目 | 暫定の既定 |
|---|---|---|
| 1 | 平均の版（`AdaptiveAvgPool2d`）も入れるか | 入れない。実際に要るのは max だけで、BTR も max を使う |
| 2 | 正方形でない `MaxPool2d`（kernel・stride・padding の 2 値指定）も入れるか | 入れない。列を残す縮約は、本 PRD のブロックで書ける |
| 3 | `pool.output_size` の書式 | 正の整数 2 つ（高さ、幅）。1 つの値で正方形を表す書き方は持たない |
| 4 | 出力が入力より大きいとき | 組み立て時に止める（§3.1 の 3） |
| 5 | 決定的モード（`backend.@deterministic`）で、CUDA の backward が例外にならないか | 要確認。adaptive pooling の CUDA の backward は、libtorch の決定的アルゴリズムの対象外として例外を出す演算に入っている可能性がある。例外になるなら、決定的モードでの扱い（止めるか、別の実装にするか）を決める |
| 6 | トークン列を盤面へ戻す専用のブロック | 作らない。`Permute` と `Reshape` で書ける。`Reshape` に盤面の大きさを直書きするので、盤面の大きさを変えるときは一緒に直す |
| 7 | DropMerge の NN profile を、本 PRD の成果物に含めるか | 含める。§3.2 の形を `net.branch.@…` として `DropMerge.txt` に置く。既定の `run.$` は変えない |
| 8 | Atari の BTR 準拠のブロック | 足す（§3.3）。既存の `AtariPool6` は変えない |
| 9 | 新しく記録する指標 | 足さない。既存の可塑性と Q の指標で足りる |

## 6. 対象外

- 平均の pooling と、正方形でない `MaxPool2d`（§5 の 1・2）。
- DropMerge で、replay ratio と NN 構成を組み合わせて比べる実験そのもの。実験記録で扱う。
- Transformer の出力を、列ごとに共通の `Linear` で畳む別の形。今のブロックの組み合わせで書けるかは未確認で、要るなら別に確かめる。
- 3 次元の pooling。

## 7. 依存と順序

- 単独で作れる。NN の部品の追加で、Agent と Env には触れない。
- DropMerge の「RR4・共有トランクの撤去・Spectral Norm」の腕は、本 PRD の実装の後に積む。それまでは、NN を変えない B512/RR2 の腕を先に流す。

## 8. 受け入れ条件

1. **単体テスト**が通る。
   - 形: `[2, 128, 12, 15]` → `[2, 128, 2, 15]`、`[2, 64, 11, 11]` → `[2, 64, 6, 6]`。
   - 値: 小さな入力で、libtorch の `adaptive_max_pool2d` の結果と一致する。
   - 勾配: backward が通り、入力に勾配が届く。
   - 不正値: `pool.output_size` の値が 2 つでない・0 以下・入力より大きい、入力の階数が 4 でない、のそれぞれで、組み立て時に止まる。
2. **既存の設定が変わらない**: `MaxPool2d` を使う既存の構成（Atari の効率優先チェーン、DropMerge の `run.@iqn32_stratified`）で、短い Run の metrics checksum が、実装前の実行体と一致する（`backend.@deterministic`）。
3. **DropMerge の smoke**: §3.2 の profile で、学習（4,000 exp）、checkpoint の保存、評価専用での読み込みが通る。ログの `[E]` が 0 で、パラメータ数（`model_size`）が §3.2 の概算と合う。
4. **設計文書**: [設計 130](../design/130_neural_networks.jp.md) のブロック一覧に、`AdaptiveMaxPool2d` と `pool.output_size` が載る（英語版は翻訳の手順で追随する）。

## 9. 参照

- [BTR, 2025] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." ICML 2025. https://arxiv.org/abs/2411.03820（コード読み: [btr_code_reading_2026-08-31.md](../../reports/btr_code_reading_2026-08-31.md)）
- PyTorch: `torch.nn.AdaptiveMaxPool2d`。https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html
