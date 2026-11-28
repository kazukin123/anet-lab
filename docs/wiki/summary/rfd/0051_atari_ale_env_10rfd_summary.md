---
title: ALE (Atari) Env 統合 — C++ 直結・Single seam・自前前処理
page_kind: サマリ
material: "[[docs/memo/done/0051_atari_ale_env_10rfd.md|0051_atari_ale_env_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
updated: 2026-11-28
---
# ALE (Atari) Env 統合 — C++ 直結・Single seam・自前前処理

## 元の材料

[ALE (Atari) Env 統合 — C++ 直結・Single seam・自前前処理](../../../memo/done/0051_atari_ale_env_10rfd.md) blob:75afd96b495060baf602c3170c33aa25a9b4ca76

## 要点

- Atari Learning Environment を C++ から利用する Env を追加し、設定でゲームを切り替える。
- 既存の単一環境の契約に合わせて ALE を包み、観測の前処理をプロジェクト内で扱う。
- 最初の対象は Pong と Breakout で、同じ実行基盤から学習・評価する。
- エミュレータのフレーム数と Agent step を区別して予算と成績を比較する。
- RFD は学習の目安を示し、Env 統合の確認と長時間学習での成績確認を分けている。

## 内容

### Context（背景・目的）

ENV バリエーション拡充の初手として Atari 2600（ALE: Arcade Learning Environment）を導入する。ALE は DQN 以来の事例が最も厚い標準ベンチマークであり、既知スコアとの突き合わせで anet-lab のアルゴリズム実装を外部基準で検証できるようになる。
環境実装は SingleDiscreteEnv として登録し、並列化・auto-reset・frame stack はすべて既存機構（ThreadPoolDiscreteEnv / stucker / ReplayBuffer stack_count）に乗る。env 自身の責務は「1 ALE インスタンスの駆動と標準前処理（skip / max-pool / resize）で単フレーム uint8 観測を出す」ことに限定する（ADR 0026）。

### アーキテクチャ（ADR 0026）

RB には単フレーム uint8 が保存され、stack はサンプル時に展開される（ストレージ増なし）。wrap 境界は PRD 050 の history margin が守る。

### Env 仕様

`frame_skip`。skip ループは env 自前（max-pool のため中間フレームが必要）。
max-pool は前 Step のフレームを混ぜない。real game over で1フレームしか実行できなければ、その1枚をそのまま使う。
`episodic_life` の lives 判定は skip 窓完走後に1回だけ行い、life lossで残りの反復を中断しない。窓内の全報酬は同じ Step に帰属する。

### ビルド統合

標準雛形（STATIC lib / include PUBLIC / pch / test / bin 出力 / source_group）は imagecls1 と同型。
Debug は `_ITERATOR_DEBUG_LEVEL`/CRT 整合のため必ず Debug の ale.lib（MSVC LNK2038 対策）。RelWithDebInfo は Release 側 lib を使う（両方 /MD・IDL=0 で整合）。
`AtariEnv-test` は module と同時にビルドされる（gating の内側なので ALE 無し環境ではそもそも生成されない）。

### テスト計画（120_environments §8 チェックリスト準拠）

`PixelwiseMax` / `ResizeGrayscale` の golden テスト（合成入力 → 期待出力固定。area 補間の縮約値を数点ピン留め）。
config 既定値・プリセット合成（v5/classic）の読み込み検証。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
