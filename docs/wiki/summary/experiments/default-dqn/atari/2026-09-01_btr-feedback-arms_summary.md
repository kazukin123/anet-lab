---
title: "Atari Breakout: BTR ソース差分の腕別スクリーニング — 2026-09-01"
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md|default-dqn/atari/2026-09-01_btr-feedback-arms]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
updated: 2026-11-28
---
# Atari Breakout: BTR ソース差分の腕別スクリーニング — 2026-09-01

## 元の材料

[Atari Breakout: BTR ソース差分の腕別スクリーニング — 2026-09-01](../../../../../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md) blob:1a374cb97b4e1fd30912932f0442ccbb956994f4

## 要点

- 同一設定の複製対で 50M のブレ幅が実測できた。
- policy churn は群の成績を順序付けない。
- head 活性の効果は adv_stream 側だけで再現し、value_stream 側はほとんど寄与しない。
- BTR 構造の上積みは複製で消えた。 複製対 3.07% / 4.18% は V/A 基準 3.48% を挟む位置にあり、まとめ 18 の「最終窓で最高」は単独 Run の性質だった（探索ブロック 13 追記）。
- `soft_update_tau` の掃引は 0.008 / 0.020 / 0.050 / hard の順序を決められない。
- Munchausen RL が壁突破率をほぼ 2 倍にした。
- Munchausen は 100M まで登り続けた。

## 内容

### 探索ブロック 01: 同一設定の複製対

`run_20260831-181826` と `run_20260901-071420` は解決後の `DefaultDQNAgent.txt` が完全一致する同一設定 Run である。

同一設定の 2 本で eval 窓平均が 32.4 点、壁突破が 52 本（相対 27%）離れた。突破本数の Poisson 標準偏差は 218 本に対し約 15 本であり、実測のばらつきはその約 1.8 倍にあたる。

### 探索ブロック 02: per_beta 0.2 固定（RR1）

`A1.learner.per_beta_start = per_beta_end = 0.2`。参照はアニール版（0.4 → 1.0、`per_beta_step` = 25M）。

`per_beta_step` は `${@vars.half_exp_step}` を参照する唯一の学習ハイパラだったため、固定化により予算の異なる Run 同士の prefix 比較が成立するようになった。

### 探索ブロック 03: per_beta 0.2 固定（RR4）

RR4 は突破率が 0.58% → 0.80% へ動いたが、RR1 の複製対の下限 1.18% には届かない。

### 探索ブロック 04: PER 初期優先度 fixed 0.3

`max` 初期化では eviction 比が 50M を通して 0.24〜0.26 で平坦、`fixed 0.3` では単調に悪化した。`max_prio_` は 1.0 で初期化され `std::max` でしか更新されない単調高水位である（`replay_buffer_impl.cpp:961,1062`）。

### 探索ブロック 05: head/fusion ReLU

eval1 の 5M 窓平均: 466.7 / 419.9 / 464.6 / 550.5 / 490.3。40-45M の 550.5 は本キャンペーンと前キャンペーンを通じて最高窓である。

エピソード数は 14531 で全群中最少、eval の `20_game_len_mean`(45-50M) は 3518.7 で基準の 3266.5 より長い。`30_lives_mean` は両者 5.0。train 平均スコアは 209.5 で基準の 179.8 を上回る。

### 探索ブロック 06: 6×6 pooling

`AtariPool6`（MaxPool2d kernel 2 / stride 2 / padding 1）を ReLU と Flatten の間に挿入し、11×11×64 = 7744 を 6×6×64 = 2304 にした。BTR の `AdaptiveMaxPool2d(6,6)` 後の `conv_out_size` と同値。共有トランク `AtariLinear512 > AtariLN512 > ReLU` は残している。

壁突破率 1.42%(226 本)、`≥600` 28 本。複製対 1.18〜1.52%、21〜30 本の内側。

### 探索ブロック 07: SN + he init residual ブロック

壁突破率 0.43%(76 本) は無保護 RR1 対照の 0.47%(78 本) を下回り、全群中最下位。train 平均スコア 158.6 も最低で、エピソード数 17497 は最多。

2026-08-30 の 5 群スクリーニングは RR8 崩壊 assay 上で SN を測っており、RR1 では未測定だった。

### 探索ブロック 08: 群横断の機序指標（45-50M）

壁突破率の順（高い順）は ReLU 3.65% > pool6 1.42% > 基準 1.35%(対平均) > PER fix 0.75% > SN 0.43%。 `01_action_churn_ratio` の順（低い順）は ReLU 0.0102 < SN 0.0251 < PER fix 0.0270 < 基準 0.0291 < pool6 0.0329。両者は一致しない。

### 探索ブロック 09: policy churn の基準値

`35_agent_churn/*` は step 軸が learner update 数（50M / batch 256 = 約 195k）である。`37_agent_qtd/*` 等の experience step 軸とは横軸が一致しない。

`03_q_delta_abs` は 5.7 倍、`17_q_gap` は 6.2 倍になり、比は 0.316 ± 0.014 で平坦だった。 `13_target_q_delta_abs / 17_q_gap` は 4.18 → 1.72 へ単調に低下し、`11_target_disagreement` は 0.439 → 0.339 とこれに追随した。50M 時点でも online と target の argmax は probe 状態の約 1/3 で一致しない。

### 探索ブロック 10: q_delta / q_gap 比の architecture 依存

ハイパラのみを変えた PER 群では比が基準と同じ帯（0.26〜0.30）に収まり、churn も近い。活性関数を変えた ReLU 群では比が最大で churn が最小となり、比と churn の対応が逆転した。

### 探索ブロック 11: head/fusion ReLU の分解

探索ブロック 05 の群は 3 箇所を同時に変えていた。1 箇所ずつに分けた。

`17_q_gap` は head/fusion ReLU が最小（0.0458、基準 0.0591）で、churn も最小だった。 行動ギャップが狭いほど argmax は脆いはずで、向きが逆である。前キャンペーンで観測した `03_q_delta_abs / 17_q_gap` ≈ 0.31 の不変量も破れる（head/fusion ReLU 0.378 で最大、churn 最小）。ハイパラのみ変えた群（PER fixed 0.3）では 0.26〜0.30 に収まっていた。

### 探索ブロック 12: head/fusion ReLU の 100M

最終窓が壁突破率で最高。eval 90-100M 平均 566.3。旧バンドル RR1 100M（`run_20260826-185555`）の 90-100M 487.1 を 16% 上回る。

### 探索ブロック 13: BTR 構造ラダー

`iqn_fusion.bind = main_feature * tau_embedding` なので融合次元は main_feature 次元に等しい。共有トランクを `iqn_fusion.structure` へ移すことで融合点だけを動かした。

段 3 は複製で上積みが消えた（2026-09-03 追記）。 複製 r2 は 3.07%（453 本、n=14776）で、r1 の 4.18% と合わせた対 3.07% / 4.18% は V/A 基準 3.48% を挟む。45-50M 窓も r2 は 443.4 で r1 の 450.6 を下回る。「最終窓で最高」（まとめ 18）は単独 Run の性質であり、V/A 基準に対する上積みは複製幅に埋もれる。 探索ブロック 23 で hard C=125 と合成しても上積みが出なかったのは、これで整合する。

### 探索ブロック 14: hard update C=125

`A2.model.soft_update_tau = 0` / `hard_update_interval = 125`。BTR の C=500 は RR4 で 32,000 exp step ごとの同期にあたる。RR1 は 1 grad = 256 exp step なので、env 時間で揃えると C=125 になる。

`15_target_sync_age` が初めて値を出した。155,326-174,742 窓で 64.1。C=125 の周期 0〜124 の一様平均 62 とほぼ一致する（probe の `interval` 503 と C=125 が互いに素なため周期を偏りなくサンプルできている）。

### 探索ブロック 15: actor_approx 初期優先度

`A2.learner.per_initial_priority_mode = actor_approx`。Atari で初めて `39_agent_per` の actor 系メトリクスが値を出した（`max` モードでは全 Run で 0 だった）。

非有限ヒントによる `max` への退避はゼロなので、`actor_approx` を実際に測っている（NDEBUG 下では黙って退避する経路がある）。推定は中央値比 1.01 でほぼ不偏、順位相関は 0.50 まで上がる。

### 探索ブロック 16: eval1 と train の乖離

hard C=125 は train エピソード長が +11%、eval エピソード長が −8% で向きが逆転する。 中央値も train +56%（243 対 156）に対し eval −1.5%（425 対 431）。

train は actor（online のクローン、`sync_interval` 400 learner step）で行動しており、`eval1` は target net を評価している（`Atari.txt:1171`）。hard C=125 の target 遅れは 0〜125 update で、actor の 0〜400 より小さい。

### 探索ブロック 17: BTR 式探索（空間 ε ラダー撤去）

eval が単調に下降する唯一の群で、25-50M 平均 366.0 は RR1 系全群で最下位。

### 探索ブロック 18: churn と成績の対応（横断）

β0.2 基準（0.030）と V/A ReLU 族（0.0095〜0.0134）は分離するが、族の中では順序が付かない。族内では churn が最大の群が壁突破率も最大である。

SN + he 群（前記録の探索ブロック 07）は churn 0.0251 で群中 2 番目に低く、壁突破率 0.43% で最下位だった。

### 探索ブロック 19: head 活性の V / A 分離

churn 系は完全に A ストリームだけで決まる。 V のみの churn 0.0262 は基準 0.0301 とほぼ同じ、系統差率 0.146 も基準 0.138 と同じ、target 不一致 0.3696 は基準より高い。

### 探索ブロック 20: LeakyReLU（硬いゼロと折れ点の分離）

機序は ReLU と区別できない。 churn 0.0111 対 0.0104、系統差率は 0.066 で完全一致、target 不一致も 0.2107 対 0.2097。`41_probe_dormant_ratio` は 0.0305 対 0.0261 で ReLU 寄り。

### 探索ブロック 21: soft_update_tau の掃引

キー 1 つで hard C=125 が完全に再現される。 壁突破率・事象数・train 平均・eval2 のすべてが一致する。

### 探索ブロック 22: eval1（target net）と eval2（online net）

`train.eval_schedule.[eval2].interval = 1009` で online net 評価を有効化した。eval1 の 1000 と互いに素にして同位相発火を避けている。

### 探索ブロック 23: BTR 構造と hard C=125 の合成

churn・dormant・`q_gap` は両者の中間、`03_q_delta_abs` と `11_q_max_mean` は両方を超えて最大、target 不一致は両方より低い。

### 探索ブロック 24: τ 埋め込みの ReLU（融合 2304）

BTR は cos 埋め込みに ReLU を掛けている（`networks.py:219` の `torch.relu(self.cos_embedding(cos))`）。探索ブロック 11 で融合 512 の τ ReLU を測ったときは純粋なコストだったが、融合 2304 で測り直した。

### 探索ブロック 25: ViT ハイブリッド（新基準）

Linear 実次元は `15488 → 512`（11×11×128）で、他群の 7744 → 512 の 2 倍。ViT が効いた場合にトランク容量の寄与と分離できないため、対照（conv 16/32/32 + トランク 15488→512、Transformer 無し）は未測定である。

### 探索ブロック 26: `eval_episodes` の較正（PRD 060）

V/A ReLU 基準のまま eval だけ `eval_batch_size = 10` / `eval_episodes = 10` にした。lane と本数を同数にすると各 lane が 1 本ずつ担当し、`eval_episodes < group_count` の WARN も出ない。

### 探索ブロック 27: 複製幅の再測定と τ 掃引の撤回

soft008 の複製が soft020 を下回った。 探索ブロック 21 の「hard C=125 の完全な再現」も、0.008 を頂点とする読み方も、いずれも n=1 の産物である。残るのは τ=0.001（3.48%）が他の全点（4.51〜6.29%）より低いことだけで、0.008 / 0.020 / 0.050 / hard の順序は判定できない。

### 参照

2026-08-31_rr4-ceiling-ln512.md — RR4 天井・LN512・432 点の壁・eval1 の検出力。本記録の判定基準はこちらに依存する。

### 探索ブロック 28: Munchausen RL（PRD 067）

対照 6.33% は基準 3 本の内側で、Double 除去は壁突破率を動かさない。 差 5.2pt は基準の複製幅 0.71pt の 7 倍で、判定不能領域には入らない。

### 探索ブロック 29: Munchausen の 100M 延長（チェックポイント継続）

探索ブロック 28 で「50M は Munchausen には早い打ち切りの可能性がある」と記録した。 `A3.auto_load_file` で 50M のチェックポイントを読み込み、+50M を継ぎ足して確かめる。

### 探索ブロック 30: 学習停止での断面評価（`learner.enabled=false`）と BTR 統計の整合

50M・ε=0 の 602.5 / 583.8 は BTR の 602.01 と同水準である。100M の 695.4 / 708.2 は BTR が報告する 682（200 評価点の最大＝選択バイアスで上振れした値）を上回る。

### 探索ブロック 31: RR4 × Munchausen（env 時間整合 C=500）

実時間は RR4 7h58m 対 RR1 2h46m で 2.9 倍。断面評価（探索ブロック 30）では ε=0 で 454.0 対 602.5 / 583.8（−130〜−149 点）、ε=0.01 で 455.8 対 554.3（−98.5 点）。 学習 Run の複製幅（ε=0 で 18.7 点）の 7 倍以上で、判定不能領域には入らない。

## 主要な概念

- [Breakout](../../../../concepts/subjects/breakout.md)
- [Target更新の比較](../../../../concepts/methods/target_update.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [可塑性指標と成績](../../../../concepts/phenomena/plasticity.md)
