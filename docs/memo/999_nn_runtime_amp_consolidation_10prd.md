# NN 実行設定への AMP の集約（全 Agent 対象）

2026-10-03、[PRD 084: NoisyNet](084_noisynet_10prd.md) の追加グリルから切り出した暫定 PRD。番号は 999 とし、方式・対象 Agent・公開契約は未裁定とする。084 の実装・完了の前提条件にはしない。

## 1. 問題と目的

AMP（autocast）の有無と FP16 / BF16 の選択は、現在 DefaultDQN では Actor の ActionPolicy 設定（`actor.[key].policy.use_amp` / `use_amp_bf16`）、Learner 設定（`learner.use_amp` / `use_amp_bf16`）、target policy 設定（`target_policy.use_amp` / `use_amp_bf16`）に分かれている。ImageCls は別の `bf16.*` 設定を持ち、MuZero と Rainbow は AMP 設定を持たない。084 の NN 実行設定（`nn_runtime`、[CONTEXT.md](../../CONTEXT.md)）は「その NN を今回どの条件で使うか」を名前付きで表す場所であり、精度はその候補になる。

084 では精度を集約しない判断をした（§8.2、[ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-runtime-key.md)）。理由は、ノイズが target 構築全体で 1 つの契約なのに精度は target 行動選択と価値評価で別であること、`Network::Forward` が精度を適用する形にすると DefaultDQN 以外の Agent と診断の経路へ波及すること、NoisyNet の目的に必要でないことである。本 PRD は、精度を NN 実行設定へ集約するなら全 Agent を同じ契約で移行することを前提に、必要性と方式を別途検討する。

## 2. 現行契約で確認したこと（2026-10-03）

- DefaultDQN の hard target では、target 行動選択は target policy 側の `Autocast` が外側の Learner の autocast を切るため、`@bf16` では FP32 になる。target 価値評価と Munchausen の追加 forward は Learner 側の設定に従い BF16 になる。この非対称は現行の実効精度であり、移行で保存するか揃えるかは裁定事項である。
- Learner の autocast scope は forward だけでなく損失計算まで覆う。精度の適用者を Network に移すと scope が forward だけに縮む。
- policy_churn と replay_fit は FP32 固定、plasticity の probe チャネルは Learner と同じ autocast で動く。
- 勾配スケーリングは Learner の現在値計算の精度（FP16 のときだけ GradScaler）に従う。
- ImageCls は `bf16.enabled` / `bf16.learner` / `actor.[key].bf16` と自前の `Autocast` を持つ。Rainbow は `ActionPolicyConfig` / `LearnerConfig` を共有するが AMP 設定を読まない。MuZero は AMP を使わない。
- `use_amp` を書いている設定ファイルは `agent.txt` だけで、`@bf16` プロファイルが Learner 側だけを有効にしている。

## 3. 別途裁定する事項

| 論点 | 決める内容 |
|---|---|
| 必要性 | 設定の分散以外に、集約で解消する実害があるか。無ければ本 PRD を中止する |
| 適用者 | 精度を Network が適用するか、呼び出し側が従来どおり autocast guard を張るか。損失計算の scope をどうするか |
| 用途の粒度 | DefaultDQN で Actor・Learner 現在値・target 行動選択・target 価値評価（追加 forward を含む）の 4 用途か、target を 1 つに揃えるか。現行の非対称を保存するか |
| 対象 Agent | ImageCls・MuZero・Rainbow の設定と呼び出しをどう移行するか。二重運用を残さないために同一変更で揃える範囲 |
| 診断 | plasticity の probe チャネルの精度（現行は Learner と同じ）、policy_churn・replay_fit の FP32 固定をどの設定が決めるか |
| 勾配スケーリング | 現在値の精度との整合をどこで保証するか |
| 移行 | `use_amp` / `use_amp_bf16` / `bf16.*` の削除とクリーンブレーク、`@bf16` プロファイルの書き換え、無効時回帰の判定（同 seed の metrics checksum 一致） |

## 4. 対象外

本起票では実装、設定キーの確定、084 への組み込みを行わない。084 の `Linear.force_fp32`（構造設定）は本 PRD の対象ではなく、集約後も構造側に残す。

## 5. 参照

- [PRD 084: NoisyNet](084_noisynet_10prd.md) §8.2、§13。
- [ADR 0048](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-runtime-key.md): AMP を束ねなかった理由。
- [DQN 系 Agent 設計](../design/200_dqn_agents.jp.md)、[NN 設計](../design/130_neural_networks.jp.md)。
