---
title: NoisyNet (Noisy Networks for Exploration) PRD
page_kind: サマリ
material: "[[docs/memo/done/0084_noisynet_10rfd.md|0084_noisynet_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# NoisyNet (Noisy Networks for Exploration) PRD

## 元の材料

[NoisyNet (Noisy Networks for Exploration) PRD](../../../memo/done/0084_noisynet_10rfd.md) blob:1fcd310d3eb06525fa5019e33c8ec32dc44e09d6

## 要点

- Atari を主な対象に、学習可能なパラメータノイズを既存の ε-greedy や UQE と比較・併用できるようにする。
- ネットワークの noisy な層は平均とノイズの尺度を持ち、用途ごとにいつノイズを引き直すかを指定する。
- Actor と Learner、online と target、環境間や遷移間でのノイズの共有を区別する。
- 設定の解釈は準備時に済ませ、通常の forward は設定ロックを取らない。
- 既定の forward や部分 forward は平均だけを用い、外部ベースラインのスコア再現を受入条件に置かない。

## 内容

### 目的と問題

現在の DefaultDQN は ε-greedy に加え、IQN の分位点配置や UQE による行動選択を持つ。「IQN は探索しない」という前提は置かない。一方、学習可能なノイズスケール、ノイズの保持期間、環境・遷移間の共有、target へのノイズ適用を比較する契約はまだない。これらを明示して、探索と学習のどちらの変更が結果へ寄与したかを調べられるようにする。

### 適用範囲と構造

既存 `Linear` に NoisyNet の設定を追加し、既存 Head の最終射影にも対応する設定を設ける。独立した公開ブロック型 `NoisyLinear` を増やすことは採用しない。Body と Head はノイズ合成・射影の演算を共用し、Head は現在の最終射影所有を維持する。
`train()` / `eval()`、勾配制御、IQN の τ、AMP は今回のカタログへの集約対象に含めない。これらは現在の呼び出し側・アルゴリズムの責任に残し、Noisy の sample / μ-only とは独立に扱う。

### Actor のノイズ契約

保持するのは ε であり、μ・σ や方策全体ではない。学習や Train Actor network snapshot の更新で μ・σ が変わっても、保持方式に定めた境界まで ε と抽選時の時計の値を維持する。
ε-greedy・UQE との併用を許す。層への NoisyNet 適用や実行設定の選択だけでは、既存の探索設定を変更しない。NoisyNet 単独を比較する profile では、ε-greedy の確率 0、risk-neutral な行動選択など、比較に必要な値を利用側で明示する。

### Learner と target のノイズ契約

ε は「実行状態の箱（用途）× network の層（module 実体）」ごとに保持し、同じ箱・同じ層・同じ時計の値なら共有、それ以外は独立とする。
Munchausen の current / next を \([2B,\ldots]\) にまとめる場合も、元の同一遷移の対応を保つ。batch を結合したことを理由に current / next の ε を独立にしてはならない。
Munchausen 有効時に `online_reuse` を選ぶ場合は、現在値の出力を bonus が要求するノイズ方式で再利用できるかを、各計算部品が検証する。Body の Noisy Linear と Head の Noisy 最終射影のどちらも対象にする。

### SN・精度・optimizer

bias は SN の対象外とする。ノイズを加える前の μ を正規化するため、ノイズ込み重みのスペクトルノルムを同じ範囲へ抑える保証はない。batch 共有・独立の両方式を扱い、SN 側の μ の計算と各共有単位の摂動を合成する。
受け入れ条件の SN の「参照計算」は、同じ μ・初期 u/v・更新条件で既存の SN 推定を適用し、その結果へ指定した摂動を加える計算を指す。厳密な最大特異値で除した結果との一致や、ノイズ込み重みへの厳密な上限は要求しない。
SN 計算と Head は既存の FP32 契約を維持する。Head は最終射影を含めて保護し、低精度で射影した出力を後から FP32 へ cast するだけの実装にしない。
通常 AdamW と FusedAdamW の双方で parameter group を構成し、σ の二重登録・登録漏れを防ぐ。weight decay が探索へ与える影響は比較対象であり、非ゼロなら必ず探索が失われるとは扱わない。

### 所有権・同期・保存・再現性

module 内の暗黙の「現在の ε」やグローバル切替で呼び出し間の状態を共有しない。Actor snapshot へ ε を紛れ込ませず、同期を再抽選の境界にしない。shared network を複数の Actor が `shared_lock` で同時に forward する構成では、module に ε を持たせると読み手どうしで競合する。
固定した構成・seed・呼び出し順序で再現できること、および診断の購読・有効化が学習や行動選択の数値系列へ干渉しないことを保証範囲とする。並列環境数、他 lane の終了順、prefetch 構成等を変えても lane ごとの乱数列が一致する保証には広げない。
NoisyNet の設定・parameter 名・保存形式を具体化する際はクリーンブレーク方針に従う。現用設定・テスト・ドキュメントを同じ変更で移行し、旧 checkpoint のためだけの alias や自動変換を追加しない。過去の Run artifact は書き換えない。

### 設定検証と公開契約

列挙値は、方式が `sample` / `mu_only`、保持が `call` / `count` / `episode`、共有が `batch` / `sample` とする。語は呼び出し側に依らず、`sample` の共有は Actor では env、Learner では遷移ごとに当たる。標準の選択をコードへ埋め込まず、設定側から方式と許可された保持・共有を選べるようにする。
ノイズ設定を使わない計算部品では、その設定群は休眠し、そのための ε 抽選・保持処理を要求しない。NN 実行設定全体を無効扱いにはしない。μ-only の保持・共有等の未使用項目も、その利用を前提とする検証の対象にはしない。
NN実行設定の項目内容は、それを使用する具象部品が準備時に型・列挙値・値域を検証する。未参照項目と、部品が利用しない設定群は休眠し、内容を検証しない。構造設定のσ₀、weight decay、probeの件数・間隔などは既存の設定検証を維持する。正の件数を要求する項目に0や負値を渡した際に、全件扱い・自動clamp・別方式へのfallbackを加えない。

### Testing・受け入れ条件

`dqn_based_agent.cpp` と `default_dqn_agent.cpp` に出る `noisy` / `sigma` の語が、optimizer の parameter 分類と診断の関数の中だけにある。Actor・Policy・forward 経路・設定検証には 0 件である。
実行設定を使う Module の追加・項目追加は、その具象クラス内部の設定型・解釈・検証だけで済む。既存の `NetworkForwardConfig`、伝達経路、Actor・Learner、他の Module の差分が 0 行である。
スコア改善、σ の一方向の減少、特定ゲームでの優位性は受け入れ条件に含めない。期待した改善が出なかったことも研究結果として残す。

## 主要な概念

- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
