# 可塑性メトリクス（srank / dormant unit 率）暫定 PRD

> 状態: 暫定メモ。実装候補 3 案と決定事項 D1〜D11 は未確定。詳細は別途グリルで詰める。本 PRD は実装着手を意味しない。
> 起点: 2026-08-25〜26、「多くのメトリクスが 5-6M 付近まで向上→急落→緩回復」の機序分析。
> **2026-08-26〜27 の `replay_ratio` ラダー（8 / 4 / 2 / 1）で機序側はほぼ決着した**ため、本 PRD の位置づけは
> 「仮説の検証手段」から **「代理指標が尽きた後の唯一の測定器」** へ移っている（後述 §なぜ今か）。
> 将来の保護機構（Spectral Norm / reset）導入時の効果測定器になる点は変わらない。
> 関連: `docs/experiments/default-dqn/atari/2026-08-17_baseline.md`（探索ブロック 15 / 20 / 21 / 23 / 24）、
> `frozen/999_nn_embedding_tsne_10prd.md`（予約 key `feature` の 6 行案を共用）、
> `940_target_update_step_axis_10prd.md`（RR ラダーに乗る `soft_update_tau` の交絡。§測定上の注意）、
> `911_metrics_record_channel_10prd.md`（独立。本 PRD は既存 scalar 経路だけで完結する）、
> 将来: Spectral Norm（BTR）/ shrink-and-perturb・ReDo（BBF/Sokar）導入 PRD。

## Context（背景・目的）

### 機序はどこまで決着したか

数値の正本は探索記録（探索ブロック 21 / 23 / 24）。要点だけ再掲する。

`replay_ratio` 以外を完全に固定した 100M の単変数ペア／トリプルで、`q_gap` の前半の落ち込みは次のようになった。

| | 過渡ピーク | 深さ | 100M 時点の `q_gap` |
|---|---|---|---|
| RR8（探索ブロック 15） | 86k grad | **−62%**（崩壊、回復せず） | — |
| RR4 | 86k grad（5.5M exp） | **−31.5%** | 0.0571 |
| RR2 | 145k grad（18.5M exp） | **−11.8%** | 0.0798 |
| RR1 | **前半の過渡なし**（80M まで単調上昇） | **—** | 0.0734 |

**当初の H2「過渡の位置は固定 grad step で決まる」は棄却された**（ピークが 86k 対 145k grad と 1.69 倍ずれ、
exp 軸では 3.36 倍ずれる。どちらの軸にも揃わない）。生き残ったのは弱い形である。

> **損傷は勾配 step あたりで蓄積し、修復は新規サンプルの流入で進む。破綻点はその比 = `replay_ratio` で決まる。**

`replay_ratio` は「収集した遷移が平均何回サンプルされるか」（`U = N·RR/B` より exp_step あたりの抽出数 = RR。
`dqn_based_agent.cpp:1804-1808`）なので、**RR = 1 は各サンプルを 1 回だけ使う点**にあたり、そこで過渡が消える。
これは Kumar 2021 / Sokar 2023 / Nikishin 2022 の "updates per datum" と同型で、
**文献の枠組みが本コードベースで再現された**ことになる。

### なぜ今か（本 PRD の価値が上がった理由）

**代理指標が尽きた。** これまで可塑性の代理に使ってきた 2 つが、どちらもスコアの予測子でないと分かった。

| 指標 | 95-100M の値（RR1 / RR2 / RR4） | スコア順（RR1 > RR2 > RR4）との関係 |
|---|---|---|
| `q_gap`（定常値） | 0.0734 / **0.0798** / 0.0571 | **単調でない**（RR2 が最大） |
| BF16 の ULP 余裕 | **1.48** / 1.66 / 1.25 | **単調でない**（RR1 が最小なのに最良） |

`q_gap` について使えるのは**過渡の深さ**だけであって定常のレベルではない、というのが探索ブロック 24 の結論である
（探索ブロック 23 で「定常での識別性が高いほど良い」と読める書き方をしたのは RR1 が反証した）。
ULP 余裕は探索ブロック 19 以来 6 例目の反例で、監視対象から外してよい段階にある。

つまり **いま何かを触っても、表現が健康になったのか壊れたのかを測る量が存在しない**。
保護機構（Spectral Norm / reset）を入れるにも、NN 容量を上げるにも、先に測定器が要る。

もう 1 つ、決着していない観測がある。**後期の `grad_norm` / `loss` / `td_mean` の上昇が RR 依存である**
（RR4 は 45-50M を底に +27% / +14% / +11%、RR2 / RR1 は平坦）。探索ブロック 22 では β スケジュールに帰属させたが、
両 Run の `per_beta_step` が同じ 50M であることから探索ブロック 23 で反証された。
**「RR4 の後期の働きは学習ではなく損傷への対処」という読みは、本 PRD の 2 指標があれば直接検証できる。**

### 文献との対応

| 文献 | 主張 | 本記録との対応 |
|---|---|---|
| Kumar et al. 2021 (arXiv:2010.14498) "Implicit Under-Parameterization" | bootstrap 更新を重ねるほど penultimate 特徴の実効ランク（srank）が低下。データあたり更新回数が多いほど悪化 | 「損傷は勾配 step、修復は新規データ」の弱い H2 と一致 |
| Sokar et al. 2023 (arXiv:2302.12902) "Dormant Neuron Phenomenon" | dormant ニューロン率は学習とともに単調増加し、**replay ratio が高いほど速く増えて性能崩壊と同期**。ReDo（dormant の再初期化）で高 RR が回復 | **RR ラダー 4 点（−62% / −31.5% / −11.8% / なし）が単調で、この予言と形が一致する。最も直接的な対応** |
| Nikishin et al. 2022 (arXiv:2205.07802) "Primacy Bias" / reset | 更新過多で早期経験に過適合。定期 reset で回復 | 過渡が「同じデータを繰り返し当てること」で生じる、という読みと整合 |
| BBF (Schwarzer et al. 2023) / BTR (2411.03820) | γ=0.997 を reset / Spectral Norm とセットで運用 | 本コードベースは両機構とも未実装（grep ヒットゼロ。探索ブロック 15） |

数値・図の詳細は記憶ベースのため、実装時に原典を引き直すこと。

## 計測の定義

対象は **penultimate 特徴 = `AtariLinear512 > ReLU` の出力（512 次元）**。全 BODY プロファイル
（`@AtariNature` / `@AtariImpala` / `@AtariImpalaX2` / `@AtariImpalaViT`）がこの形で終端することを確認済み
（`apps/runner/config/Atari.txt:512-522`。次元は `net.block.[AtariLinear512].linear.out_features` = 512、同 :411）。
IQN では fusion（`main_feature * tau_embedding`）の**前**で測る — τ 変調前の共有表現が「健康状態」の対象。

### srank_δ（実効ランク）

バッチ N 件の特徴行列 Φ ∈ R^{N×512} の特異値 σ₁ ≥ … ≥ σ_d に対し

```
srank_δ(Φ) = min{ k : (σ₁+…+σ_k) / (σ₁+…+σ_d) ≥ 1−δ }      δ = 0.01（Kumar et al. 慣例）
```

最終層は Q(s,a) = w_a·φ(s) の線形読み出しなので、φ が低ランク部分空間に潰れると行動間の差 (w_a−w_b)·φ(s) はその部分空間への射影しか使えない。**q_max 単調上昇 + q_gap 侵食はランク崩壊の署名**であり、srank はそれを表現側で直接測る。

### dormant unit 率

ニューロン i の正規化活性スコア

```
s_i = E_x[|h_i(x)|] / ( (1/H) Σ_k E_x[|h_k(x)|] )
```

に対し s_i ≤ τ なら dormant。τ = 0（厳密な死。ReLU では勾配が流れない）と τ = 0.025（Sokar らの実用値）の 2 本を出す。

### 出力スカラー（案）

| tag（案） | 内容 |
|---|---|
| `43_agent_plast/01_srank` | srank_0.01（0〜512 の整数値） |
| `43_agent_plast/02_srank_ratio` | srank / 512 |
| `43_agent_plast/03_dormant_ratio` | τ=0.025 の dormant 率 |
| `43_agent_plast/04_dead_ratio` | τ=0 の dormant 率 |
| `43_agent_plast/05_feature_norm` | E[‖φ‖₂]（値スケール成長の随伴観測） |

群番号 43 は空き（現用は 37_qtd / 38_loss / 39_per / 40_rs / 41_on / 42_env / 90_perf）。EMA 版は viewer 用に 1〜2 本検討（D7）。

## 現行コードで確定している事実（実装の下地）

1. **Body/Head 分割済み**。`Network::Forward` は body → head の順で、head が無ければ body 出力（= 特徴）をそのまま返す（`core/anet-core/src/nn_impl.cpp:1320-1338`）。head 入口で FP32 へキャストしており、**body 出力は BF16 でありうる**（同 :1333）。
2. **予約 key `feature` は未使用**。head が公開する key は `forward` / `forward.q` / `q_values` / `forward.v` / `v_values` / `forward.a` / `a_values` のみ（`core/anet-core/src/dqn_based_heads.cpp:58,135,147,156,228`）。t-SNE PRD が「`Network::GetTensorDictFunction` に予約 key `feature` を足して `body_->Forward(state_input)` を返す。6 行程度」という最小変更案を保持している（`frozen/999_nn_embedding_tsne_10prd.md:108`）。**同じ 6 行を本 PRD が共用できる**。
3. **learner は毎 update でバッチ forward 済み**。`ForwardOnlineWithTrain`（`dqn_based_agent.cpp:2624, 2791, 2929`）が replay バッチ（`replay_batch_size` = 256）を流しており、q_gap 等のバッチ統計も同じ場所で計算している（`MakeQuantileMetrics`、同 :2449-2482）。
4. **TraceSink という層タップが既にある**が別用途（NN activation の viewer 送出）で、actor 側の `MakeActionTraceSink` は env 0 だけを slice する（`core/anet-core/src/rl.cpp:20-25`）。名前と機構の衝突に注意（record チャネル PRD の D1 と同種の論点）。
5. メトリクス配線は既存 idiom で足りる: agent 側スカラーは `$agent <key> @learn`（例 `41_agent_on` 群）、update 結果系は `$result <key>`（`37_agent_qtd` 群）。どちらの経路でも新規タグは config 追加のみで viewer / inspect_run に乗る。
6. **過去 Run から遡って計算することはできない**。Run が残す重みは `agent_close.anet`（終了時 1 点）だけで、途中の checkpoint は無い。**検証には再 Run が要る**（§検証計画）。
7. **spatial exploration 下では replay バッファが ε 混合**になる（`CreateSpatialLaneTensor` が env ごとに固定 ε を割り当て、`.flip({0})` により env[0] が `eps_end`。`dqn_based_agent.cpp:1086`）。replay バッチで測る量は「バッファに入っている状態の質」を含む（探索ブロック 26 と同じ論点。D3）。

## 実装候補（詳細グリルで選択）

| 案 | タップ方法 | 追加 forward | 備考 |
|---|---|---|---|
| A | learner の既存 forward に sink / 出力 key で相乗り | 無し | 最小コスト。ただし `ForwardOnlineWithTrain` は head 込みの `Network::Forward` なので body 出力を外へ出す口が要る（=予約 key と同じ改修が Forward 経路側に要る） |
| B | 予約 key `feature`（t-SNE PRD の 6 行）で **別 forward** を N update に 1 回 | 有り（バッチ 256 の body forward 1 回 / N update） | 既存 forward 経路を一切触らない。N=100 なら追加コストは実質ゼロ。t-SNE PRD と改修が完全共用 |
| C | `QuantileMetrics` / UpdateResult を拡張して learner 内で直接計算 | 無し | 配線は最短だが learner が SVD を抱える。quantile 系 learner 3 実装（QR/IQN/…）への重複に注意 |

計算コスト: `svdvals`（256×512、FP32 化後）は sub-ms。dormant は平均絶対活性のみ。**N=100 learn step に 1 回で全指標を出しても overhead は 0.1% 未満**の見込み（実測は受入時）。

## 予測（この計測が付けば何が判定できるか）

RR ラダーの過渡の深さ（RR8 −62% / RR4 −31.5% / RR2 −11.8% / RR1 なし）が**既知の答え合わせ用データ**になる。

| 観測 | 判定 |
|---|---|
| srank 低下・dormant 増加の**大きさが RR に単調**で、RR1 では起きない | **弱い H2 確定**。「損傷 ∝ 勾配 step / 修復 ∝ 新規データ」の直接証拠 |
| 表現は健康（srank 平坦・dormant 微増）なまま `q_gap` だけ振動 | 可塑性説の棄却。値の再配分であって容量損耗ではない |
| RR4 の後期（45M〜）で dormant が増え続け、RR1 では平坦 | **「RR4 の後期の `grad_norm` / `loss` / `td_mean` 上昇は損傷への対処」の確認**（探索ブロック 23 の未特定事項） |
| 劣化位置が grad 軸にも exp 軸にも揃わず、深さだけが RR に従う | `q_gap` で観測された形の再現。代理指標と直接指標が同じ絵を描く（相互検証） |
| e04（谷 −20%）と baseline（−54%）で dormant 増分が谷の深さと共変 | 「被覆の良い探索が表現を守る」の直接証拠（探索ブロック 20 と接続）。※ラダー設定をまたぐので D3 の注意が要る |

将来の Spectral Norm / ReDo / reset 導入時は、この 2 指標がそのまま**効果測定器**になる（ReDo は dormant 率がトリガー指標そのもの）。

## 検証計画

過去 Run からは計算できない（§事実 6）ので、実装後に短い再 Run が要る。過渡は RR4 で 5.5M、RR2 で 18.5M exp に出るので
**25M 予算で 3 本**取れば全域を覆える。

| 腕 | 予算 | 所要（実測 throughput から） | 見るもの |
|---|---|---|---|
| RR4 | 25M | 約 2.7h | 深い過渡（−31.5%）が srank / dormant に出るか |
| RR2 | 25M | 約 1.5h | 中間（−11.8%）。深さが単調か |
| RR1 | 25M | 約 1.0h | 過渡なし。srank / dormant も平坦か |

合計約 5.2h。**RR4 と RR1 の 2 本（3.7h）でも判定の中核は取れる**（単調性は 3 点目で確認）。
`replay_ratio` 以外は探索ブロック 23・24 と同一構成にすること。

## 測定上の注意

- **`soft_update_tau` の交絡が RR ラダーに乗る。** τ は更新あたりなので RR を半分にすると target network の遅れが
  exp 軸で 2 倍になる（RR4 64k / RR2 128k / RR1 256k exp step）。可塑性指標の RR 依存性を読むときも同じ交絡が入るため、
  **勾配 step 軸での比較を主に読む**か、`940_target_update_step_axis_10prd.md` の正規化を先に入れる。
- **測定間隔を learn_step で切ると、RR が低いほど点が疎になる。** eval で同じ問題が起きている（RR4 は 10M 窓あたり 63 点、
  RR1 は 15.7 点。探索ブロック 24）。過渡の位置比較をするなら疎さが効くので、D2 で軸を決める。

## 決めるべきこと（未確定）

| # | 論点 | 備考 |
|---|---|---|
| D1 | 実装候補 A/B/C の選択 | B が t-SNE PRD と共用でき、既存 forward 経路を触らない点で有力 |
| D2 | 測定間隔の**軸**と既定 ON/OFF | learn_step（損傷の時計）か exp_step（点の密度が RR に依らない）か。interval 設定 key（例 `learner.plasticity_metrics.interval`、0=OFF）。既定 ON か |
| D3 | 測定バッチ | 手元の replay バッチ（文献と同流儀）か、固定プローブ集合か。**spatial exploration 下では前者が ε 混合になる**（事実 7）。同一ラダー設定どうしの比較（= RR ラダー）は成立するが、**ラダー設定をまたぐ比較は成立しない**（探索ブロック 26 と同じ制約）。まず前者、後者は拡張 |
| D4 | 測定点 | main_feature（fusion 前）のみか、fusion 後も出すか。branch 構成で出力 TensorDict が複数 key の場合にどの key を採るか（t-SNE PRD の同一論点） |
| D5 | BF16 の扱い | body 出力は BF16 でありうる。SVD 前に FP32 へ cast（固定でよいか） |
| D6 | 決定論との両立 | `svdvals` の CUDA 実装が deterministic 設定下で通るか。ダメなら CPU 転送で計算（N=100 なら許容） |
| D7 | タグ構成 | 上記 5 本 + EMA の要否。群番号 43 でよいか |
| D8 | 他 agent への展開 | DefaultDQNAgent のみか、RainbowAgent / ImageClsAgent にも足すか（当面 Default のみが最小） |
| D9 | τ と δ の値 | δ=0.01 / τ={0, 0.025} を既定にするか。config 化するか |
| D10 | **target net 側も測るか** | online だけか、target も出して両者の乖離を見るか。`soft_update_tau` の交絡（§測定上の注意）を直接観測できる利点がある |
| D11 | **検証 Run を受入に含めるか** | §検証計画の 2〜3 本を受入条件に入れるか、実装完了は smoke だけで切って検証は別枠にするか |

## スコープ外

- **保護機構そのもの**（Spectral Norm / shrink-and-perturb / ReDo）。本 PRD は計測のみ。導入判断はこの計測の結果を見てから別 PRD。
- **t-SNE / 埋め込み可視化**（frozen/999_nn_embedding_tsne）。予約 key を共用するだけで、可視化側の凍結状態は変えない。
- **record チャネル**（911_metrics_record_channel）。本 PRD の出力は集約済みスカラーなので既存 scalar 経路で完結する。
- **`soft_update_tau` の正規化**（940_target_update_step_axis）。本 PRD は交絡を注記するだけで、修正はしない。
- 重みノルム・層別勾配などの追加指標。欲しくなったら同じタップに足せる。

## 受入基準（案）

1. 新タグが `inspect_run.py tags` で `status=ok`・`count>0`。
2. srank が学習初期（ランダム重み）で高く、**RR4 の既知の過渡域（5-15M exp）で低下**する方向の変化を示す（絶対値の妥当性はこのスモークで確認）。
3. throughput 劣化が Run 間ブレ幅（±3%）以内。
4. `interval=0` で完全に不活性（forward 追加も SVD も走らない）。
5. determinism ON 構成でエラーにならない（D6 の解決を含む）。
6. RR1 構成で srank / dormant が**平坦**であること（既知の「過渡なし」と一致。D11 を受入に含める場合）。
