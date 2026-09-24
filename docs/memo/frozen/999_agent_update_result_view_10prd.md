# Agent Update Result View 暫定 PRD

> 凍結中(再開条件: PRD 030 MixUp/CutMix の可視化要求が実運用で出たら)

> 状態: 暫定メモ。実装方針の確定前に、ImageCls MixUp/CutMix 検討から出た可視化ギャップを記録する。
> 背景 PRD: `030_imagecls_mixup_cutmix_10prd.md`

## Context

現行の `ImageClsView` は Env 側 View として実装されており、`TrainEvent` の `experience.state.obs` と `action_info` を表示する。
そのため、Env が出した単一画像、hard label、Actor の推論結果を確認する用途には合っている。

一方、MixUp/CutMix を `ImageClsLearner::UpdateFromBatch` 側で実装する場合、実際に loss へ入る mixed grid、paired target、lambda、混合方式などは Learner 内部の一時データになる。
この情報は Env の observation ではなく、Agent の学習処理、主に `BatchUpdateResult` または `LearnEvent` に近い情報である。
したがって、現行の Env View の位置付けのままでは MixUp/CutMix 後の学習入力を表示できない。

## Problem

Agent/Learner 側で生成・計算された学習時情報を、人間が確認する標準的な可視化手段が弱い。

具体例:

- MixUp/CutMix 後の mixed image を表示できない。
- paired target、lambda、CutMix bbox、mix mode を確認できない。
- loss に使った target contract と、Env View に出る hard label の違いを UI 上で区別しにくい。
- `BatchUpdateResult` に scalar 以外の debug tensor を入れても、それを View として扱う導線がない。

## Goal

Env View とは別に、Agent/Learner の update result 由来データを表示する仕組みを用意する。

この PRD の主目的は、MixUp/CutMix のように「Env observation ではないが学習に使った中間データ」を確認できる可視化経路を定義することである。

## Non-Goals

- `ImageClsView` を MixUp/CutMix 専用 UI に変更しない。
- Env の `state.obs` に Agent/Learner 内部の mixed data を逆流させない。
- MixUp/CutMix 実装そのものは本 PRD の対象外とする。
- すべての Agent に一律の詳細 UI を強制しない。

## Proposed Direction

### 1. Env View と Agent Update View を分ける

`ImageClsView` のような Env View は、従来通り Env observation と Actor 推論結果を表示する。
学習時に Learner が作る中間データは、別の Agent Update View または Agent Debug View として扱う。

表示対象の整理:

| 種別 | 由来 | 例 |
|---|---|---|
| Env View | `TrainEvent.experience`, `action_info` | 元画像、true label、pred、Top probabilities |
| Agent Update View | `LearnEvent.update_result_list`, `BatchUpdateResult` | loss、accuracy、mixed image、lambda、paired label、CutMix bbox |

### 2. BatchUpdateResult に tensor/debug 情報を載せる

`BatchUpdateResult` は scalar だけでなく `GetTensor` / `GetTensorVector` を持っているため、Agent 側の debug tensor の出口として使える可能性がある。
ImageCls MixUp/CutMix では、必要最小限の情報だけを update result に持たせる。

候補:

- `mix.image`: 表示用の mixed grid サンプル。全 batch ではなく先頭数件だけ。
- `mix.lambda`: mix ratio。
- `mix.mode`: scalar または enum 相当の値。
- `mix.target_a`, `mix.target_b`: hard label id。
- `mix.bbox`: CutMix の矩形。

### 3. View/Observer 側は LearnEvent を購読する

Env View は `TrainEvent` 前提だが、Agent update result の自然なイベントは `LearnEvent` である。
View または metrics image observer が `LearnEvent` の `update_result_list` から tensor/debug 情報を取り出して描画できるようにする。

ただし、UI 更新頻度と tensor サイズには注意する。
混合画像の可視化は常時全 batch ではなく、低頻度、先頭サンプル、または明示設定時だけに限定する。

## Acceptance Criteria

1. Env View と Agent Update View の責務がドキュメント上で区別されている。
2. `ImageClsView` は pre-mix の Env observation 表示として維持される。
3. Agent/Learner 側の `BatchUpdateResult` 由来 tensor を表示する経路が定義される。
4. MixUp/CutMix の mixed image、lambda、paired target を確認できる設計になっている。
5. 表示のために Env state や replay/experience の契約を歪めない。
6. 既定では追加表示は無効または低頻度で、通常学習の性能へ大きく影響しない。

## Risks / Open Questions

- `BatchUpdateResult` に画像 tensor を載せる場合、CPU/GPU 転送と lifetime をどこで切るか。
- View と metrics image observer のどちらに寄せるべきか。
- Agent ごとに異なる debug tensor 名を許容するか、共通の命名規約を作るか。
- `LearnEvent` 購読 View が現行 GUI 構造に自然に入るか。
- MixUp/CutMix 可視化は常設 UI ではなく、まず metrics image として保存するだけで十分か。

## Follow-Up Candidates

1. 現行 `BatchUpdateResult::GetTensor` / `GetTensorVector` の利用実績を調査する。
2. `metrics.image` observer が `update_result_list` を画像ソースとして扱えるか確認する。
3. ImageCls MixUp/CutMix 実装後、最小 debug payload を `ImageClsUpdateResult` に追加するか判断する。
4. 必要なら Agent Update View の共通インターフェースを別 PRD に分離する。
