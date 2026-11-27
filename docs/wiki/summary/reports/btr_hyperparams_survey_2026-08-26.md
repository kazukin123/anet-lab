---
title: "Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル"
page_kind: サマリ
material: "[[reports/btr_hyperparams_survey_2026-08-26.md|btr_hyperparams_survey_2026-08-26]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-26
---
# Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル

## 元の材料

[Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル](../../../../reports/btr_hyperparams_survey_2026-08-26.md) blob:7f0f1101cf0c54a534980d005b8d0350228b1cf8

## 要点

- BTR は Rainbow DQN を基点とし、6 要素を追加、4 要素を維持、2 要素を削除した構成である。Double DQN の削除理由は、Munchausen RL が次状態の argmax を使わないため機能的に不要になることである。
- したがって BTR は sticky actions 0.25 と NoOp reset (max 30) を併用している。
- Torch やラッパーの既定値を確認しないと、論文が省略した開始時の no-op を落とす。
- PER の重要度サンプリングが意図せぬ実装になっている。
- 個々の構成要素の IQM 差は一部図でしか示されず、数値表として確認できる範囲に限界があった。

## 内容

### 1. 書誌情報と調査方法

本調査では、論文に記載がない項目・論文と実装が食い違う項目を切り分けるため、論文と公式実装の双方を突き合わせた。以下の各節では、論文にしか無い情報・コードにしか無い情報・両者が一致する情報を区別して記す。

信頼度の階層として、§6 では公式ドキュメント（層1）、公式リポジトリのソース（層2）、原論文（層3）を分けて扱う。第三者の解説記事・ブログは根拠として使用していない。

### 2. アルゴリズム構成 — Rainbow DQN からの差分

Noisy Networks は維持されている。 Dueling・PER・Noisy Networks の除去実験の結果、PER と Noisy Networks は有益と判明して残され、Dueling は有意差が無かったが Rainbow DQN からの継続性のため残された。

### 3. ハイパーパラメータ

本体の optimizer は素の Adam であり、Table D6 に weight decay の行が存在しない。学習率は Rainbow DQN の 6.25e-5 よりやや高い 1e-4 が最良だったと本文が述べている。

実装の既定分岐も `weight_decay` 引数を渡しておらず 0 である。Adam epsilon は batch size に連動する定義になっている。

weight decay については、Appendix H に AdamW（decay 1e-4）の試行が「有意差なし」として記録されている。

hard update（重みの丸ごとコピー）であり、500 gradient steps ごとである。これは 32,000 環境ステップ（128,000 フレーム）に相当する。

### 4. ネットワーク構造

幅スケール 2、adaptive maxpool 出力 6×6。3 ブロックのチャネル数は [16×width, 32×width, 32×width]、畳み込み部の出力は 6×6×32×width となる。

adaptive maxpool の採用理由は、学習の高速化と入力解像度非依存性である。

Table D6 は "All Convolutional Residual Layers" と記す。より具体的には、各 Impala CNN ブロック内の residual layer（Conv 3x3 + ReLU が 2 つ）に適用される。

実装では block 先頭のダウンサンプル用 conv には適用されておらず、residual 内の 2 conv のみが対象である。 したがって対象は 3 ブロック × 2 residual × 2 conv = 12 層になる。

### 5. Atari 環境設定と前処理

論文が与える環境設定は 8 項目のみである。noop も frame skip も max pooling も含まれない。

論文本文・付録・全表のどこにも `no-op` / `noop` / `random start` の語が出現しない。 一方、公式実装は Gymnasium の `AtariPreprocessing` を fork した自前ファイルを使い、その既定 `noop_max=30` を上書きしていない。

reward clipping は学習ループ内でのみ適用され、スコア集計はクリップ前に行われる。したがって報告スコアは raw score である。

Gymnasium の `AtariPreprocessing` を丸ごと複製し `life_information` 引数のみ追加した自前ファイルである。docstring は Gymnasium 版のまま残っている。Stable-Baselines3 / CleanRL / Dopamine のいずれも使用していない。

### 6. v5 世代における NoOp reset の系統分裂

本節は BTR 固有ではなく、Atari RL 一般における noop の扱いを一次資料横断で確認したものである。

`AtariEnv.__init__` にも noop 系引数は存在せず、`reset()` は `self.ale.reset_game()` を呼ぶだけである。raw env は noop start を含まない。 公式ドキュメントも v5 の変更点として sticky の復活と stochastic frame-skipping の廃止のみを挙げ、noop には言及していない。

公式ドキュメントとソースの両方で `noop_max=30` が既定である。

### 7. 評価プロトコル

online net である。 `agent.net`（online）の state_dict のみを評価プロセスへ渡しており、`agent.tgt_net` は評価経路に現れない。評価前に NoisyNet のノイズが無効化される。

200M フレーム時点の値ではなく、学習中の最良評価を報告している旨が脚注で明示される。

### 8. 報告スコアと出所

682 は Table A1 の BTR 列の値であり、raw score、200M フレームまでの学習中に得られた最大スコア（100 エピソード平均）である。

リポジトリ公開の実測 CSV とは一致しない。 `results.csv`（200 回の評価、各回 100 エピソード平均）の Breakout 行は、最大 676.58、200M フレーム時点（最終評価）602.01、末尾 10 回平均 605.54 であった。論文値 682 との差の理由は原典に説明がない。

一方 Table A1 の DQN (Nature) / Rainbow 列については、キャプションに出所の明記が無い。 キャプションが出所を示すのは M-IQN と FE-Rainbow の 2 列のみである。BTR 著者が自前で再実験したという記述も論文中に存在しない。

### 9. Ablation

本文中に数値として与えられているのは Impala の +142% IQM だけである。 他成分の IQM 変化率は Figure 5 / B3 / C6 のグラフのみで提示され、数値表は存在しない。

200M フレーム学習済み最終エージェント、3 seeds。原典が与えるコンポーネント別の定量比較はこの表のみである。

この 2 要素は性能を下げることを論文自身が認めたうえで、計算アクセシビリティのために採用している。

Table 2 の Score ε=0 も、`w/o Maxpool` の 406k が BTR 本体の 330k を上回っており、この記述と整合する。

### 11. 批判・懸念

Machado 準拠の主張と実装が一致しない可能性がある。 論文は Machado et al. (2018) に従って評価すると述べるが、実装は sticky と noop を併用している。Machado 原典は sticky を no-op の代替として提案しており（§6.3）、併用を推奨してはいない。ただし原典に併用の禁止文言も無いため、これを違反と断定することはできない。

報告値と公開実測値が一致しない。 Table A1 の Breakout 682 に対し、公開 `results.csv` の最大は 676.58 である。差の理由は原典に説明がない。

ablation の定量値がほとんど本文に無い。 数値で与えられるのは Impala の +142% IQM のみで、他成分はグラフのみである。componentwise の表（Table 2）は Phoenix 単一環境・3 seeds に限られる。

### 12. 総合評価

BTR は Rainbow DQN を基点に 6 要素を追加し、200M フレームの Atari-60 学習をデスクトップ PC（RTX 4090）で 11.5 時間に収めた構成である。設計思想として明確なのは、性能と計算コストのトレードオフを意識的に性能側へ全振りしていない点である。replay ratio を 1/64 に抑えたのも、vectorization と maxpool を性能低下と引き換えに採用したのも、いずれも実時間を優先した判断として論文中に明記されている。

ハイパーパラメータの記載密度は Table D6 が 33 行と比較的高いが、環境側（Table D4、8 行）が薄く、前処理の再現に必要な情報がコードにしか無い。論文単独での再現は困難で、公式実装が事実上の仕様書になっている。

比較値としての Breakout 682 を用いる際は、(a) best checkpoint であること、(b) online net の評価であること、(c) 100 エピソード平均であること、(d) 公開実測 CSV の最大 676.58・最終 602.01 と一致しないことの 4 点を明示しないと、他手法との比較が成立しない。

### 13. 調査限界

本調査の一次資料取得は 2026-08-26 時点のものである。GitHub 上のソースは main ブランチであり、以降の変更は反映されていない。

論文引用は arXiv v2（HTML 版）に基づく。v1（2024/11）との差分は本調査の対象外である。なお v1 のタイトル表記は "On A Desktop PC"、v2 は "on a Desktop PC" である。

PER の β の初期値・終値・スケジュール長は原典で確認できなかった（論文中で "beta" は Adam の β1/β2 としてのみ登場する）。

### 14. 付録A: anet-lab 現行設定との差分（調査対象外の自プロジェクト情報）

本節は BTR 原典の記述ではない。 左列は §3〜§7 で確定した BTR の値、右列は anet-lab の Run `run_20260825-183923_atari_breakout_apex_e04_100m` の実行時 config ダンプ（`config/config_data.txt` および `config/DefaultDQNAgent.txt`）から取得した実測値である。

γ 0.997 / batch size 256 / n-step 3 / replay capacity 1,048,576 / weight decay 0 / Adam lr 1e-4 / Huber κ 1.0 / Impala 幅スケール 2 / dueling 採用・512 per stream / IQN cos 埋め込み 64 / sticky 0.25 / episodic life false / max episode frames 108,000 / grayscale・84x84・frame stack 4 / reward clip [-1,1] / replay ratio（BTR の 1/64 は batch 256 ÷ 64 env steps = 4 samples/env-step であり、anet-lab の `replay_ratio = 4` と同義）。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
