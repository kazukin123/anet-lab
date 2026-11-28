---
title: ANET フレームワーク全体概要
page_kind: サマリ
material: "[[docs/design/010_framework_overview.jp.md|010_framework_overview.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/326_ppo.md|326 Agent個別実装 PPO系列]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# ANET フレームワーク全体概要

## 元の材料

[ANET フレームワーク全体概要](../../../design/010_framework_overview.jp.md) blob:c9a07de2edfea794701b26542623e95224a4e952

## 要点

- ANET は Env・Agent・Runner・Observer の契約を分け、共通の実行基盤で学習と評価を行う。
- Env が観測の形と行動の仕様を宣言し、Actor が行動を選び、Learner が蓄積した経験を使う。
- 経験数・学習更新・環境の進行を別々の step 軸で数える。メトリクスを比較するときは記録軸をそろえる。
- Agent、Env、Network の追加は公開の拡張点を通し、設定から実装を構築する。
- 具象 Agent には DQN 系、PPO 系、MuZero 試作、ImageCls がある。PPO 系は方策を直接学び、Replay を使わずに rollout をためて学ぶ。
- Run の設定と出力を Run ディレクトリに集め、Runner の画面と事後の分析ツールから参照する。

## 内容

### 2. ANET の概要

- ANET は、環境との相互作用から方策を学習する強化学習実験を、C++ の単一アプリケーション上で実行・観察するためのフレームワークです。libtorch によるテンソル計算とニューラルネットワーク、wxWidgets による GUI、Run 単位のメトリクス記録を統合しています。
- 本プロジェクトは、学習アルゴリズムや実装方式を検証する個人実験コードとして継続開発されています。安定した公開 API を提供する製品ライブラリではなく、設定や内部構造は実験に合わせて更新されます。

### 3. 基本概念

- Observation の代表的なキーは、低次元ベクトル用の `vector`、画像・格子用の `grid`、合法手を示す `action_mask` です。各 Env が `EnvSpec` を通じて shape、dtype、値域などの契約を定義します。
- Actor — Observation から Action を作る。Train と Eval で別インスタンスを作成できる。
- 比較では step 軸の名前と、その軸を持つ Runner の両方を合わせる。学習と評価の `exp_step` は同名でも別の座標なので、同じ横軸として重ねられない。

### 4. ソフトウェア構成

- `AnetRLRunner` が設定と実行をまとめる入口です。学習処理は `anet-core` の抽象と共通実装を経由して具象 Agent・Env を利用します。メトリクスは Runner と Observer から Run ディレクトリへ記録され、Viewer や補助ツールが実行プロセスとは独立して読み取ります。
- コードマップでは、`src/algo/` に DQN、PPO、MuZero、ImageCls の Agent 実装と系統別テストを置く。

### 5. 機能一覧

- 共通基盤の上に具象 Agent と Env があり、Runner と分析アプリケーションがそれらを利用する。
- Agent の具象実装は `DefaultDQNAgent`、`RainbowAgent`（Run では使わない）、`PPOAgent`（方策を直接学ぶ PPO 系の離散行動の実装。Replay を使わず rollout をためて学ぶ）、`MuZeroAgent`（試作）、`ImageClsAgent`。

### 6. 機能仕様

- Agent 共通（6.4）からは、DQN 系と PPO 系の構成、方策ベースの Agent が使う共通部品の文書を引く。具象実装（6.12）からも PPO 系の具象構成を引く。
- Run管理は、一回の実行に必要な Env、Agent、Runner、Observer を設定から構築し、その反復、評価、停止、成果物の lifetime をまとめます。Train は直列の `SerialTrainRunner` または学習と Env 処理を重ねる `PipelineTrainRunner` を選択でき、Eval は Learner を持たない `EvalRunner` で進めます。

### 7. 設定、Run、成果物

- `metrics.jsonl` — scalar、設定、成果物参照などの時系列メトリクス。
- `config/config_data.txt` — include、merge、コマンドライン上書き後の全 ConfigData。
- `images/` — step ごとに出力した画像。
- Agent 保存ファイル — Runner が保存した Network と optimizer 等の状態。

### 8. 基本的な設計原則

- 設定値、Tensor shape・dtype・device、EnvSpec などの契約を境界で検証し、意味のない状態で処理を継続しない。
- Actor が Learner に依存しない方向を保ち、Train と Eval の行動選択を分離できるようにする。
- Event と Observer を介して実行処理とメトリクス・可視化を分離し、同じ Run を GUI と外部ツールの両方から観察できるようにする。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [326 Agent個別実装 PPO系列](../../concepts/framework/326_ppo.md)
- [RunnerApp](../../concepts/apps/110_runner_app.md)
