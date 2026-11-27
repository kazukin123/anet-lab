---
title: ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD
page_kind: サマリ
material: "[[docs/memo/done/0078_replay_frame_history_integrity_10rfd.md|0078_replay_frame_history_integrity_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
updated: 2026-11-26
---
# ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD

## 元の材料

[ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD](../../../memo/done/0078_replay_frame_history_integrity_10rfd.md) blob:2e8b80b62dbace3973a5280e2d0472315aae6896

## 要点

- ReplayBuffer の観測履歴と bootstrap 用 stack に、エピソード境界の扱いに由来する二つの不整合が見つかった。
- 履歴の過去方向の走査には、入力が示す episode_start を使う。
- 各 lane の最初の実 Push と、その後の done・truncated との対応を storage へ書く前に検証する。
- 終端 dummy と実観測を区別し、リングを上書きするときは履歴開始情報も置き換える。
- 長さ1のエピソードを正しい入力として扱い、実際のサンプルと bootstrap の観測を公開経路から確認する。

## 内容

### 問題と目的（Problem Statement / Solution）

ReplayBufferから返るTensorのshape、replay item key、n-step長が正しくても、観測stackの一部が別時刻の観測や過剰なpaddingに置き換わる。学習側はこれを正常な経験として受け取り、誤った状態を学習に使う可能性がある。
公開サンプルが、入力された同一lane・同一エピソードの履歴に一致することを保証する。履歴不足のpaddingが許されるのはepisode由来の不足だけであり、ring上書きで失われた履歴は既存のsampleable range契約で除外する。サンプル集合を余分に狭めて不具合を隠す対応も採らない。

### 調査結果と確度

初回実行は100条件成功後の101番目で失敗した。残り91条件を独立実行した結果は20成功・71失敗。追加91条件のプロセス実行時間合計は586.032秒で、成功20条件だけの集計は180検査地点、延べ7,308 key、49,185サンプル照合だった。単一プロセスずつ実行し、メモリ観測例はworking set 519 MiB / private memory 931 MiB。全プロセスの生涯最大メモリは採取していない。
未検証: CUDA転送経路、matrix外の全構成。stack=1の96条件成功も、全機能・全入力の無欠陥証明ではない。

### 不具合A: 初回満杯時の起動時padding汚染

Aは`next_obs`にも及ぶ。時刻0、n=1、stack=4なら`next_obs`のframeは論理位置-2〜1で、同じ折り返しが起きる。アッセイは最初の不一致で止まるため、記録には`obs`しか残っていない。

### 不具合B: 未確定metadataによるbootstrap stackの過剰padding

非terminalサンプルのbootstrap入力が誤るため、bootstrap価値、Bellmanターゲット、TD誤差、Learner優先度更新へ波及する可能性がある。
wrap前の初期値だけでなく、wrap後の旧世代の実終端等が原因となり得るため、運用中にも繰り返し起こる可能性がある。頻度は未測定。

### 共通の設計要求と対応方針（Implementation Decisions）

Push検証の規則: laneごとに`expected = lane_expects_start`（初期値true）、`actual = state.episode_start[lane]`。`expected != actual`なら、lane、write cursor（logical index）、期待値、実値を含めて`ANET_SYSTEM_ERROR`。全laneの検証をどのlaneの書込みよりも前に終える。検証後、実観測を履歴開始`actual`で書き、truncatedならdummy（履歴開始false）を書き、`lane_expects_start = done \|\| truncated`とする。dummyは検証の対象にならない。
Atari 1M slotで約1 MiB。全観測のstack複製、無制限の履歴保持、追加の画像Tensorコピーは導入しない。Pushにはslot当たり1 byteの履歴開始の書込みに加え、全laneの事前検証とlane状態の更新が増える。Sampleの走査は2つのaccessor読みから1 byte読みへ置き換わる。既存のProfileRangeを維持し、Push/Sampleの時間を同条件で変更前後に測定・比較して報告する。固定の性能劣化率を合否基準にはせず、明確な悪化が出た場合は原因を調査する。

### テスト方針・受け入れ基準（Testing Decisions）

失敗時は設定、seed、lane、入力時刻、key、対象が`obs`か`next_obs`か、stack内のframe位置、期待値・実値を出す。条件ごとの実行結果と未完了地点を区別する。

### スコープ外（Out of Scope）

Env継ぎ目の構造契約（ADR 0034）自体の変更、Actor側`StackerActionContext`の変更。
CUDA転送・CUDAイベント管理の修正と新規検証。共通抽出の修復を転送経路の全面保証とは扱わない。

### 調査中に見つかった関連事項（Further Notes）

shutdown修正直後の1回で`_CrtIsValidHeapPointer` / `is_block_type_valid`を観測した。同じseedの後続6回と合同回帰、追加91条件では同じ症状を確認していない。想定原因は特定できておらず、影響範囲も未確定。shutdown修正で解消したとも、ReplayBufferのA/Bが原因とも扱わない。再発時は独立した再現条件・stack・実行ログを採取して対応範囲を裁定する。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
