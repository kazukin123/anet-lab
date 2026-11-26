# 希少なエピソードを後から調べるための記録案

## 元の材料

[RFD0144　希少なエピソードを後から調べるための記録案](../../../memo/0144_episode_forensics_10rfd.md) blob:62061ee4d57d4d5805637bd896edc6aac006f752

## 要点

- 希少な挙動を事後に調べるための要求を整理した暫定メモである。捕捉方法・記録形式・導入順は未決である。
- 長期 Run の Q 値の一過性の跳ねや、長時間の NOOP、例外的な終端を後から元のエピソードまで追いたい。
- 集約メトリクスだけでは、その値を出した lane・状態・行動の時系列を復元できない。
- 特殊な挙動を見つけたときに、元の事象と前後の流れを保存する候補を整理する。
- 常時全件を記録する負荷を避けるため、捕捉条件、記録の粒度、保持量を併せて決める。

## 内容

### Problem Statement

長期 Run では、スカラー指標のスパイク、NEET 方策、異常に長い NOOP 列、珍しい終端、急激な Q 値・TD 誤差・PER 優先度の変化など、短時間だけ現れる特徴的な挙動がある。
長期 Run は実時間と計算資源の費用が大きい。特徴的な挙動を人間がリアルタイムで目撃して手動録画する運用に依存せず、該当エピソードを自動で捕捉し、Run 終了後でも詳細調査できる証拠を残したい。

### Solution

全 Env lane の直近エピソードについて、Action、報酬、終端、時刻軸、識別子、Observation の要約または設定された診断 payload を、容量制限付きの Episode Journal に保持する。
Env、Actor、Learner、Replay、または利用者操作から特徴的な挙動を検出したら、そのエピソードと関連遷移を Forensics Bundle へ昇格し、Run artifact として永続化する。

### User Stories

長期 Run の実施者として、特徴的なエピソードを自動保存したい。リアルタイムで画面を監視し続けなくても希少事象を失わないためである。
Run 分析者として、メトリクス上の Q 値スパイクから最大値を生んだ Sample 要素を特定したい。batch 集約値だけでなく原因候補の遷移を確認するためである。
Run 分析者として、Learner で検出した Sample を元のエピソードへ紐付けたい。現在動いている Env の映像と過去の Replay 経験を混同しないためである。

### Implementation Decisions

Env instance と lane を含む Run-scoped な `EpisodeId` を発番する。
Learner trigger が発生する前に元履歴が期限切れになる可能性があるため、保持 horizon は Replay の滞留時間との関係を設定・メトリクスで確認可能にする。
active episode が trigger された場合、trigger 前の Journal を固定し、終端または設定された post-trigger step まで追記してから Bundle を確定する。

### Testing Decisions

テストは private container の形ではなく、入力イベント、発生 trigger、保存された Bundle、明示された不完全状態という外部契約を検証する。
RFD0035 の replay item key テストを先例とし、物理 slot の再利用と永続 identity を混同しないことを検証する。
元履歴が保持期限切れの場合、Bundle が incomplete として確定し、利用可能な trigger/Sample 証拠は失われないことを確認する。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
