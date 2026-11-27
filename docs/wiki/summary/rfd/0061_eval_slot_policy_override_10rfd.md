---
title: Actor 設定カタログと eval スロットごとの方策 / network 指定
page_kind: サマリ
material: "[[docs/memo/done/0061_eval_slot_policy_override_10rfd.md|0061_eval_slot_policy_override_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-26
---
# Actor 設定カタログと eval スロットごとの方策 / network 指定

## 元の材料

[Actor 設定カタログと eval スロットごとの方策 / network 指定](../../../memo/done/0061_eval_slot_policy_override_10rfd.md) blob:d2f332e20c19d730c2562da9681407569a2a71ef

## 要点

- 一つの Run の中で、評価スロットごとに方策と利用するネットワークを選べるようにする。
- Breakout の比較では評価時 ε と対象ネットワークを Run 全体でしか変えられず、同一の学習結果で測れない条件があった。
- Actor の設定をカタログとして持ち、評価の用途に応じて選ぶ。
- スロットの選択と個別指定を設定解決の共通規則に従わせ、明示した上書きを保つ。
- 評価結果が実際に使った Actor と対応するよう、Runner・Observer の参照先もそろえる。

## 内容

### 背景と実害

A. 文献比較の土俵が 1 Run で組めない。 Nature DQN 2015 は ε=0.05 / 30 本、BTR は ε=0.01 / 100 本。ε が agent 共通なのでどちらか一方しか置けない。
D. online net を 2 通りの ε で評価できない。 `IsForTarget` が Eval1 にだけ true を返す。

### 現行コードで確定している事実(2026-09-07 時点)

ImageClsActor は run_mode を保持するだけで参照しない。Env 側の RunMode 依存は `IsTrain / IsEval` のみ(ImageCls の dataset 選択と episode scope、CartPole の初期化)。Eval1 と Eval2 を区別する Env・metrics・GUI は存在しない。
QR51 は `run.@qr51_control`、IQN32 は現行チェーンで起動できる。

### ゴールと非ゴール

非ゴール: 本数(060 済)、スケジュール(052 済)、background snapshot 順序(912)、best checkpoint(913)、eval1 / eval2 の運用判断、eval ε の段階スケジュールそのもの(BTR 相当の ε=0 評価は常時 ε=0 の別スロットで実現する)。

### 確定契約

`Actor` は `Module` を実装する(`GetScalar` / `GetTensor` / `GetTensorVector`)。Actor の実効設定は Agent dump の `actor.[key].*` で確認する。`Runner::GetActor()` を追加する。
ProfiledValue で ε / τ を書けるようにする「枠」はこの `UpdateSchedule(step)` フックまでとし、ProfiledValue 自体は導入しない(`eps_start / eps_end / eps_decay_steps` は現状維持)。
Actor Q ヒントの送出は役割を見ず Agent の PER 設定(`use_per && ACTOR_APPROX`)だけで決める。全 Actor が付け、eval runner は無視する。
EvalPanel の `model_sync.mode = shared` は廃止する(clone は参照タグの Actor 設定が決める。`frame / time / episode` の同期スケジュールは GUI の責務のまま)。EvalPanel は `app.*.eval_panel.eval_config_tag` で参照するタグの `actor` と env prefix を鏡写しする。
`EventField::ACTOR` と `$actor` を追加し、`event.runner->GetActor()` で解決する。`@train` / `@learn` / `@episode_end` / `@session_end` のいずれも runner を持つので `$actor` を使える。`$eval.[tag] $actor epsilon` でスロット別の ε を観測できる。

### 実装ノート

過去 Run artifact・実験記録・ADR は当時の記録として変更しない。
受入: 全テスト緑、`inspect_run_test.py` 緑、smoke Run の `config_data.txt` がキー名置換を除いて一致。
`dqn::Actor`: `policy_` を専有し、`MakeAction` 冒頭で `policy_->UpdateSchedule(step)`。`GetScalar("epsilon" / "uqe_tau")` を policy へ委譲、実効 Actor 設定は Agent dump の `actor.[key].*` に既定補完後の値として記録する。
`IsForTarget`、`train_policy_` / `eval_policy_`、`GetScalar` の `train_policy.` / `eval_policy.` / bare `epsilon` / `uqe_tau` 経路を削除。`UpdateFromBatch` は `target_policy_->UpdateSchedule(counts)` だけ残す。
移行後の dump を対応表で照合し、全フィールド一致を確認する。UQE では不使用の `eps_*` など「効かないフィールド」も差があれば明示して一致させる(判定を policy 種別に依存させない)。

### 却下した案と経緯

`use_optimistic_target` の廃止(2026-09-08 検討): 実効値は明示指定で同じになるが、optimistic 化のたびに `target_policy.*` を丸ごと書き換える(戻しも含む)保守コストが戻る。コピー元を `actor.[train].policy` に変えて維持する。
IQN spec K の専用キー(`net.taus_shape` 等)/ カタログ全項目と Learner の最大 K: Learner の N で足りる。設定項目やカタログ走査を増やさない。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [316 設定管理](../../concepts/framework/316_config.md)
