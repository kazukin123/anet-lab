# NN 内部表現の 2D 埋め込み可視化（t-SNE / PCA）暫定 PRD

> 凍結中(再開条件: 着手予定なし。NN 埋め込み可視化が必要になったら)

- 起票日: 2026-08-20
- 状態: draft（バックログ。着手予定なし。正式番号は着手時に採番）
- 対象: `core/anet-core`（`nn_impl.cpp` の可視化関数口、`observers`、`heat_map`）、`viewers/metrics-tools`、`apps/runner/config`
- 起点: Mnih et al. 2015 "Human-level control through deep reinforcement learning"（*Nature* 518, p.532）Figure 4 の t-SNE 図
- 依存: なし（既存 PRD への依存は現時点で見つかっていない）

> 本 PRD は実装着手を意味しない。設計分岐の洗い出しと、確認済みのコード事実の保全が目的。案の最終選択は未了。

## Context / 何を作りたいか

### 原典 Figure 4 の中身

- 対象は Space Invaders 1 本のみ。学習済み DQN に**実時間 2 時間プレイさせて経験した状態**を集める。
- 埋め込む対象は生ピクセルではなく **最終隠れ層（出力の 1 つ手前の全結合層）の活性ベクトル**。
- その集合に t-SNE をかけて 2 次元散布にし、各点を **DQN が予測した状態価値 V（= max_a Q(s,a)）** で着色（濃赤=高、濃青=低）。
- 選ばれた点にはゲーム画面のサムネイルを吹き出しで添える。

キャプションが主張しているのは 3 点:

1. 敵が全部揃った画面と、ほぼ倒し終えた画面が**どちらも高 V** で近接する（クリアすると敵満載の次画面が来るので将来報酬として等価）。
2. 中途半端に倒しかけた画面は**低 V** で分離される（直近報酬が少ない）。
3. バンカー（防御壁）の有無で**見た目が違う画面が近接**する（ステージ終盤ではバンカーは価値に寄与しない）。

つまり図の主張は「**最終隠れ層は知覚的類似性ではなく価値的・意味的類似性で状態を並べる表現を、報酬信号だけから獲得している**」。1 と 3 は「見た目が違うのに近い」、2 は「見た目が近いのに離れる」例になっている。

### anet で何が嬉しいか

現状の NN 可視化は `SweepedHeatMap`（state 空間を 2 軸 sweep）と `Conv2dVisualizer`（フィルタ可視化）で、いずれも**入力側**の可視化である。埋め込み可視化は**表現側**を見る初めての手段になる。特に Atari / DropMerge のように obs 次元が高く sweep が使えない環境では、現状「NN が何を等価とみなしているか」を見る手段が無い。

## 現状の事実（コード確認済み）

2026-08-20 時点、branch `main`。

| 事実 | 根拠 |
|---|---|
| `Network` は Body + Head 分割済み。Body の出力は `TensorDict`、Head は `Forward(const TensorDict& feature_dict)` を受ける | `core/anet-core/include/anet/nn.hpp:101`, `:119-146` |
| `Network::Forward` は head が無ければ body 出力（features）をそのまま返す | `core/anet-core/src/nn_impl.cpp:1325-1336` |
| **`Network::GetTensorDictFunction("forward")` が body 単体を返すのは head が無い場合だけ**。head 有りでは必ず `body → head` を通す経路しか公開されていない | `core/anet-core/src/nn_impl.cpp:1340-1373` |
| Head が公開する key は `forward` / `forward.q` / `q_values` / `forward.v` / `v_values` / `forward.a` / `a_values`。**`feature` 相当の予約 key は未使用** | `core/anet-core/src/dqn_based_heads.cpp:58, 135, 147, 156, 228` |
| network の選択は `policy-net.` / `target-net.` の prefix routing | `core/anet-core/src/dqn_based_agent.cpp:458-485` |
| `HeatMap` は「任意 (x,y,value) の散布を 2D ヒートマップ化」する `ImageSource`。`AddDataBatch(xv, yv, vv)` と `max_points`、`HM_MeanMode` 等の flags を持つ | `core/anet-core/include/anet/heat_map.hpp:49-86` |
| 画像系 Observer は `TaggedTrainObserver` / `TaggedLearnObserver` + `ImageProvider` の組で、`IntervalGate` による interval 制御を持つ。既存は `HeatMapVectorObserver` / `TimeHistogramObserver` / `SweepedHeatMapObserver` / `Conv2dVisualizationObserver` | `core/anet-core/include/anet/observers.hpp:72, 120, 184, 276` |
| 値の抽出は `ScalarProbe` / `VectorProbe`、NN 出力からの集約は `ValueExtractFunction` と `extractor::MaxExtractor` 等 | `core/anet-core/include/anet/probe.hpp:14-38, 304-380` |
| `MetricsLogger` が扱えるのは scalar / json / wxImage / ImageSource / GraphViz。**[N, D] の行列を吐く API は存在しない** | `core/anet-core/include/anet/metrics_logger.hpp:162-187` |
| 画像 Observer は config 駆動。`image.<name>.type` と `network_key = policy-net.forward`、`output_key = q`、`extractor_name = max` 等の形式 | `apps/runner/config/metrics_image.txt:66-100` |
| `metrics_image.txt` は `_main.txt` から `$include` されている | `apps/runner/config/_main.txt:5` |
| `.venv` に **scikit-learn 1.9.0 と numpy 2.5.1 が既に入っている**（`sklearn.manifold.TSNE` が import だけで使える）。torch は入っていない | `.venv/Lib/site-packages/` |
| Python 側ツールは `viewers/metrics-tools/` に集約。`inspect_run.py`（subcommand 分割済み）、`tb_bridge.py`（jsonl → tfevents）、`metrics_source.py` | `viewers/metrics-tools/` |
| 背景スレッドで重い処理を回すと Train が止まる前例がある（`EpisodeEvalObserver` の `WaitBackgroundEval`） | `core/anet-core/include/anet/observers.hpp:218-240` |

## 分解：4 つの関心と充足状況

t-SNE 図は独立した 4 関心の合成であり、**新規実装が必要なのは ③ だけ**である。

| 関心 | 役割 | 既存部品 | 不足 |
|---|---|---|---|
| ① 収集 | どの状態集合を N 件集めるか | `VectorProbe` / `ISweepInputGenerator` / `EvalRunner` | rollout 中の feature を N 件貯める Collector |
| ② 抽出 | NN のどの層のベクトルを取るか | `Network::GetTensorDictFunction` | **body-only key が無い**（§問題 C） |
| ③ 射影 | D 次元 → 2 次元 | なし | **本体** |
| ④ 描画 | (x, y, value) の散布 | `HeatMap` | ほぼ不要（§問題 E） |

## 問題（設計分岐）

### A. t-SNE をどこで回すか（最大の分岐、未決）

**案 A: C++ in-process**（libtorch で自前実装、Observer 化してライブ表示）

- 既存 `SweepedHeatMapObserver` と同じ形に収まり、Runner パネルに出せる。
- t-SNE は O(N^2)。N=4000 で秒〜十数秒。**学習スレッドを止める**。背景スレッド化しても `EpisodeEvalObserver` と同じ轍を踏む危険がある（§現状の事実）。
- 実装量が最大。Barnes-Hut を入れるなら更に増える。

**案 B: オフライン Python**（C++ は特徴量を dump、`viewers/metrics-tools/` で射影と作画）

- `sklearn.manifold.TSNE` が既に使える。C++ 側の仕事は「[N, D] float 行列 + 色用スカラ + step を Run フォルダに吐く」だけになる。
- `inspect_run.py` が subcommand 分割済みなので `inspect_run.py tsne <run>` が自然な置き場。
- ライブ性は無い。

**案 C: TensorBoard Embedding Projector**

- `tensors.tsv` / `metadata.tsv` / `sprite.png` を吐くだけで、t-SNE・PCA・UMAP はブラウザ側が回す。`tb_bridge.py` が既にある。
- **原典 Figure 4 に最も近い体験**（点をホバーするとサムネイルが出る）が実装ゼロで得られる。あの図の価値の大半はサムネイル添付で、自前描画ではそこが一番高くつく（§問題 F）。
- 射影パラメータの再現性・自動化は弱い。

C と B は **C++ 側の出力仕様が同じ**（[N, D] + メタ）なので、① ② + dump を作れば両方に乗る。案 A は「ライブで見たい」という要件が実際に立ってから。

現時点の暫定順位は **C → B → A**。ただし未決。

### B. t-SNE は run 間・step 間で比較できない

埋め込みは回転・スケール・クラスタ配置が実行ごとに変わる。interval で連番出力して動画にすると「動いて見えるが意味がない」図ができる。原典 Fig.4 も**学習済みエージェントの 1 枚**であって時系列ではない。

対策候補:

- (a) 前回の埋め込みを `init=` に渡す warm start
- (b) 「学習後の 1 枚」と割り切り、interval 出力を提供しない
- (c) 決定的な PCA を併記する（PCA は符号だけ揃えれば run 間比較できる）

→ **射影器を差し替え可能にして PCA も選べる**形が素直。`projector = tsne | pca | none` を想定。

### C. 特徴量取得口が無い

原典の「最終隠れ層」は anet では Body 出力 `TensorDict` に対応する。しかし現状 head 有りの `Network` から body 単体を取る key が無い（§現状の事実）。

最小変更案: `Network::GetTensorDictFunction` に予約 key `"feature"` を追加し、`body_->Forward(state_input)` をそのまま返す。既存 head key（`forward*` / `*_values`）と衝突しない。6 行程度。

未決点:

- Body は branch を持ちうる（`NetworkBranchConfig`）ため出力 `TensorDict` が複数 key になる場合がある。どの key を採るか（`kKey_DefaultOutput` 既定 + `feature_key` で上書き？）。
- Dueling head の場合、head 内部の中間層も「最終隠れ層」の候補になりうる。原典 DQN は Dueling ではないので厳密対応は無い。
- AMP 有効時、body 出力は BF16 になりうる。dump 前に FP32 化するか。

### D. 色（value）の決め方

原典は V = max_a Q(s,a)。anet には `extractor::MaxExtractor` が既にある。他の候補:

- argmax action（離散色。方策の分割面が見える）
- TD 誤差 / PER 優先度（学習が困っている領域が見える）
- episode 進行度、報酬、終端理由
- train 由来 / eval 由来のフラグ（原典 Extended Data の図はこれ）
- ImageCls なら正解ラベル（§段取り案 3）

→ 既存 `HeatMapVectorObserver` の `value_probe` と同じ思想で**外から Probe を挿す**形が一貫する。案 B / C なら「色用スカラを複数列 dump しておいて後で選ぶ」で済む。

### E. 描画粒度

`HeatMap` はビン集約するため、原典のような 1 点 1 マーカーの散布とは見た目が変わる。`max_points` と `HM_MeanMode` である程度は寄せられる。真の散布が要るなら `ScatterPlot : ImageSource` を 1 つ足す判断になる。

**案 C を採るならこの分岐自体が消える。**

### F. サムネイル添付

原典 Fig.4 の説得力の本体。自前描画で HeatMap 画像に obs サムネイルを合成するのは実装量が大きい。案 C なら sprite シートを吐くだけ。**案 C を推す最大の理由**。

### G. 収集元（未決）

- `eval_rollout`: 原典に最も近い（学習済み方策で実際に遊んだ状態）。eval 経路への差し込みが要る。
- `replay_sample`: ReplayBuffer から N 件サンプル。実装は最も軽いが、方策分布ではなく履歴分布になる（古い方策の状態が混ざる）。
- `train_stream`: Train event を流し見て貯める。オンライン分布そのものだが探索ノイズを含む。

リザーバサンプリングで N 件に間引く前提。重複状態の間引きをするかも未決。

### H. コストと hot path

- 前処理の作法として **D=512 をいきなり t-SNE に入れず PCA で 50 次元に落とす**（原典の元手法もこれ）。perplexity 5〜50、N は 2k〜10k。
- 収集そのもの（forward の feature を貯める）は hot path に乗る。interval と max_points で有界にする。
- 案 B / C なら射影コストは学習プロセス外に出る。

## 目標契約（案・未確定）

1. **既定 OFF**。config で明示的に有効化しない限り hot path に一切乗らない。
2. C++ 側の責務は「① 収集 + ② 抽出 + dump」まで。射影と作画は外に出す（案 B / C 前提）。
3. dump 先は **Run フォルダ内**（`runs/<name>/embed/`）。Run フォルダだけで完結し、外部に主データを置かない。
4. 色用スカラは複数列を同時に dump できる（後から選び直せる）。
5. 射影器は差し替え可能（`tsne` / `pca` / `none`）。
6. `Network` への追加は予約 key 1 本のみ。既存の forward 経路・数値挙動を変えない。

## config 素案（`metrics_image.txt` の作法に合わせた場合）

```
image.tsne.value.type            = Embedding2D
image.tsne.value.interval        = 0            # 0=学習中は出さない（終了時のみ）
image.tsne.value.image_width     = 1024
image.tsne.value.image_height    = 1024
image.tsne.value.embed.source       = eval_rollout   # eval_rollout | replay_sample | train_stream
image.tsne.value.embed.max_points   = 4000
image.tsne.value.embed.network_key  = policy-net.feature   # ★新規予約 key
image.tsne.value.embed.feature_key  = default              # body 出力 TensorDict の key
image.tsne.value.embed.projector    = tsne      # tsne | pca | none(=dump のみ)
image.tsne.value.embed.pca_dim      = 50
image.tsne.value.embed.perplexity   = 30
image.tsne.value.embed.warm_start   = true
image.tsne.value.embed.probe.value.source = agent
image.tsne.value.embed.probe.value.key    = action.max_q
image.tsne.value.embed.sprite.enabled     = true
image.tsne.value.embed.sprite.obs_key     = image
```

キー名・階層はすべて暫定。`Embedding2D` という type 名も仮。

## 段取り案

1. `Network::GetTensorDictFunction("feature")` 追加（6 行程度）
2. FeatureCollector（rollout 中の feature + 色用スカラを N 件リザーバサンプリング）+ TSV / バイナリ dump
3. **ImageCls で正しさを検証**。正解ラベルで色分けすればクラスタがラベルと一致するか一目で判定でき、実装の検証手段になる。RL の図で配線の正しさを判定するのは不可能。過学習調査の道具としても使える見込み
4. TensorBoard Embedding Projector で見る（Python 実装ゼロ）
5. 必要なら `inspect_run.py tsne` で PNG 生成
6. さらに必要なら C++ ライブ Observer（案 A）

## スコープ外

- UMAP / PaCMAP 等 t-SNE 以外の非線形射影の内製
- 埋め込みの定量評価（trustworthiness、kNN 一致率など）
- 埋め込み空間を使った学習側の改変（表現正則化、探索ボーナス等）。本 PRD は可視化のみ
- MetricsLogger への tensor 行列ログ型の追加（dump は別経路で足りるなら不要）

## 未調査事項

- ImageCls 経路で正解ラベルを Observer 側から取得できるか（§段取り案 3 の前提）
- Body の branch 構成時に feature `TensorDict` が実際にどういう key 集合になるか
- 案 C の sprite 生成に必要な obs 画像化が、既存 `ImageSource` / `Conv2dVisualizer` 経路をどこまで再利用できるか
- eval 経路への Collector 差し込み点（`EvalRunner` のどこで feature を掴むか）
- 原典が t-SNE 前に PCA を挟んでいたか（van der Maaten の標準手順ではあるが、DQN 論文 Methods での明記は未確認）
