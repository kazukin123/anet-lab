# target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`）暫定 PRD

> 状態: 暫定メモ。案 A〜D と決定事項 D1〜D8 は未確定。詳細は別途グリルで詰める。本 PRD は実装着手を意味しない。
> 起点: 2026-08-26〜27、`replay_ratio` を 8 → 4 → 2 → 1 → 0.5 と下げるラダー（探索記録 探索ブロック 15 / 23 / 24）で、
> **`soft_update_tau` が「更新あたり」なので RR を半分にすると target network の遅れが exp 軸で 2 倍になる**ことが判明したこと。
> 単変数 A/B のつもりが 2 変数動いていた。
> 関連: `docs/experiments/default-dqn/atari/2026-08-17_baseline.md`（探索ブロック 23 §考察・pending「`soft_update_tau` を RR に追随させた対照」）、
> `062_plasticity_metrics_10prd.md`（同じ RR ラダーを検証セットに使う）。

## Context（背景・目的）

`replay_ratio` は本記録で最大のレバーだが、これを動かすと **target network の遅れが道連れで動く**。
`SoftUpdate()` は learn_step ごとに呼ばれるため `soft_update_tau` は「勾配更新 1 回あたり」の EMA 係数であり、
同じ τ でも RR が変われば「経験 1 step あたりどれだけ target が online へ寄るか」が変わる。

exp_step 軸で見た target EMA の時定数は `(1 / τ) × B / RR` になる（B = `replay_batch_size`）。
`num_envs` には依存しない。τ = 0.001 / B = 256 での実値:

| `replay_ratio` | 更新/exp_step | exp 軸の時定数 | 100M に対する比 |
|---|---|---|---|
| 8 | 1/32 | 32,000 | 0.03% |
| 4 | 1/64 | 64,000 | 0.06% |
| 2 | 1/128 | 128,000 | 0.13% |
| 1 | 1/256 | 256,000 | 0.26% |
| 0.5 | 1/512 | 512,000 | 0.51% |

**RR を半分にするたび遅れが 2 倍**になる。遅い target = 安定した bootstrap 目標なので、
向きは観測された成績改善と同じであり、**exp 軸の効果量に混じった寄与を排除できない**
（勾配 step 軸の比較は同じ更新回数なら EMA 進行も同一なのでこの交絡を受けず、そちらでも RR2/RR1 が勝つため
結論自体は保たれる。探索ブロック 23・24）。

`replay_batch_size` を動かしても同じ drift が起きる。**RR/B という比が本当のノブ**であり、
現在の config はそれを直接書けない。

同種の問題が `hard_update_interval` にもある（後述。ただし文献整合の観点が逆に働く）。

## 現行コードで確定している事実

| 事実 | 位置 |
|---|---|
| `UpdateTarget(learn_step)` は 1 勾配更新ごとに呼ばれる | `dqn_based_agent.cpp:2368`（`UpdateTargetNetwork(vars_.learn_step)`） |
| `hard_update_interval > 0` なら `learn_step % interval == 0` で hard、そうでなければ毎回 soft | `dqn_based_agent.cpp:431-442` |
| `SoftUpdate()` は `online_net_->SoftCopyTo(*target_net_, soft_update_tau)` のみ | `dqn_based_agent.cpp:444-446` |
| 宣言は `float soft_update_tau = 0.01f` / `int hard_update_interval = -1` | `agent.hpp:113-114` |
| 更新回数は `earned_credit = num_envs × replay_ratio / replay_batch_size`（float） | `dqn_based_agent.cpp:1804-1808` |
| **同じ関数の直後で `UpdatePerBeta(counts.exp_step)` は exp 軸を使っている** | `dqn_based_agent.cpp:2369` |
| `train_actor.sync_interval` は `ProfiledValueConfig<step_t>` で、周期評価は `step.exp_step`、age 判定は `step.train_step` | `dqn_based_agent.cpp:1689-1698`、`agent.hpp:176` |

**同一ループ内で 3 つの軸が混在している**（target=learn_step、PER beta=exp_step、actor sync=exp_step 評価 + train_step age）。

## 問題の整理

### (1) `soft_update_tau` — 正規化したい本体

τ は更新あたりの量なので、RR / B を動かすと exp 軸の挙動が変わる。
「同じ設定で RR だけ変えた」という単変数実験が成立しない。

### (2) `hard_update_interval` — 単純に exp_step へ変えると文献整合が崩れる

現在は learn_step 単位。**これは文献と一致している** — Nature DQN 2015 の `C = 10,000` は
パラメータ更新回数であり、Dopamine の `target_update_period = 8000` も `training_steps`（勾配 step）で数える。
`Atari.txt:319` の `@nature` プロファイルはこの前提で書かれている。

したがって「learn_step は誤りだから exp_step へ」ではなく、**軸を明示的に選べるようにするか、
既定をどちらに置くかの設計判断**である。soft 側と同じ drift は hard 側にも同じだけ存在する
（RR 半減で hard update の exp 間隔が 2 倍）。

## 案（グリルで選択）

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A: exp 時定数で書く** | 新キー `model.target_lag_exp_steps`（仮）を追加し、実行時に `τ = B / (RR × T)` を導出。`soft_update_tau` は排他で残す | 意図（「経験 N step ぶん遅らせたい」）を直接書ける。RR / B を変えても不変 | 導出値なので config dump に出る実効 τ と設定値が別。RR が実行中に変わらない前提が要る |
| **B: 基準 RR での正規化フラグ** | `model.normalize_target_update = true` のとき `τ_eff = τ × (RR_ref / RR)`。`RR_ref` は既定 4 | 既存 config の意味を変えずに追随だけ入る | 基準値というマジックナンバーが増える。B の変化を拾わない |
| **C: 軸を選べるようにする** | `model.target_update_axis = learn_step \| exp_step` を追加。exp_step 選択時は soft も hard も exp 基準で発火 | soft/hard を同じ機構で扱える。`sync_interval` の前例と揃う | soft を exp 基準にすると「毎 exp step 微小に寄せる」か「K exp step ごとに τ で寄せる」かの再設計が要る |
| **D: 何もせず文書化のみ** | CONTEXT.md と Atari.txt に「τ は更新あたり」と明記し、RR を動かす実験では τ も併せて動かす運用にする | 実装コストゼロ | 人間の注意力が防波堤。今回まさに見落とした |

案 A と C は排他ではない（A = soft の書き方、C = hard の軸選択）。

## 決定事項（未確定）

| # | 論点 | メモ |
|---|---|---|
| D1 | 案 A〜D の選択 | soft は A、hard は C という組み合わせが有力。要検討 |
| D2 | 既存 config の互換 | `soft_update_tau` を残すか置き換えるか。両方書いたら fail-fast か、片方優先か |
| D3 | 導出値の可視性 | 案 A のとき、実効 τ を `config_data.txt` へ出すか Property 側へ出すか（CONTEXT.md の Module Config / Property の区別） |
| D4 | `hard_update_interval` の既定軸 | learn_step のままか exp_step か。`@nature` プロファイルは文献整合上 learn_step が要る |
| D5 | 軸変更時の単位取り違え防止 | 同じキー名で単位だけ変わると既存 config が静かに壊れる。キー名を分けるか、軸指定を必須にするか |
| D6 | `replay_batch_size` 依存の扱い | RR だけでなく B も分母。案 B は B を拾わないが、それでよいか |
| D7 | 他 agent への展開 | `NetworkModel` は `DefaultDQNAgent` / `RainbowAgent` 共通。`ImageClsAgent` に target network は無い |
| D8 | 検証手段 | RR2 で「RR4 と同じ exp 軸時定数」（τ = 0.002）にした 1 本を回し、探索ブロック 23 の exp 軸の効果量から target lag ぶんを差し引けるか |

## 受け入れ観点（案）

- 同一 `target_lag_exp_steps` で `replay_ratio` を 4 / 2 / 1 と変えたとき、**exp 軸の target 遅れが一致する**こと（実効 τ をログか dump で確認）。
- 既存の `soft_update_tau = 0.001` / `replay_ratio = 4` 構成が、値・挙動とも変わらないこと（既定では新機構が無効）。
- `@nature` プロファイル（hard update / `hard_update_interval = 10,000`）が Nature 2015 と同じ更新回数基準で動き続けること。
- 単位取り違えが fail-fast すること（D5）。

## 非目標

- target network 機構そのものの変更（Polyak / hard の是非、二重 target 等）。
- `train_actor.sync_interval` の軸統一。現状 exp_step 評価 + train_step age で混在しているが、本 PRD の範囲外。
- 探索ブロック 23・24 の結論の見直し。勾配 step 軸の比較はこの交絡を受けないため、`replay_ratio` が効くという結論は保たれる。
