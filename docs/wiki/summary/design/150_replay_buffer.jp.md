---
title: ReplayBuffer
page_kind: サマリ
material: "[[docs/design/150_replay_buffer.jp.md|150_replay_buffer.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
updated: 2026-11-26
---
# ReplayBuffer

## 元の材料

[ReplayBuffer](../../../design/150_replay_buffer.jp.md) blob:28e027973a98fd58f1f75caab77631e69698e5a0

## 要点

- ReplayBuffer は Experience を蓄え、Learner に観測、行動、N-step 収益や終端情報を含む Sample を返す。
- フレームの束と N-step の窓は、lane とエピソード境界を考慮して作る。
- 優先度付き抽選ではサンプルに重みと更新用の識別子を添え、学習後の優先度を元の項目に反映する。
- 事前取得と CUDA 転送を使う場合も、Sample のテンソルと CPU に残す識別用の情報を区別する。
- 診断用の抽選は一様・非復元や抽選履歴別の条件に対応する。通常の学習 sample と目的を区別して読む。

## 内容

### 2. 基本概念と外部contract

- RunnerからのRewardは`[N]`である。Agentが複数の報酬成分を使う場合だけ、取り込みからReplayへ渡す報酬を`[N,C]`にする。成分の数Cは構築時の`ReplayBufferConfig.reward_component_count`（既定1）で固定する。1成分のshapeと計算順は変えない。成分の意味は利用側Agentが所有する。
- Push・Sample・一様probe・履歴probeはcallerから成分ごとの現在のγの列を受け取り、builderとextractorはγを永続保持しない。Push時にはそのγで成分ごとのn-step収益を保存する。初期priorityの完成待ちentryには、同じPushで評価した成分ごとの収益とbootstrap割引率を保持する。

### 3. コンポーネント定義

- `ExperienceSamples` — ReplayBufferから抽出しLearnerへ渡すminibatch。
- `ReplayExperienceStorage` — lane別ring storageへObservation、Action、infoと実slotの履歴開始を保持する。未書込slotとdummy slotの履歴開始はfalse。
- `ExperienceQueueController` — laneごとの遷移からN-step targetを確定する。
- `ExperienceSampleExtractor` — indexからstack、N-step、next stateをminibatchへ組み立てる。

### 6. 処理フロー

- 成分数が2以上で窓規則が指定された場合は、構築時に成分数と窓規則の指定を示して拒否する。窓の収益を成分へ展開しない。
- Pushはn-stepの確定とは別に、各遷移の1-step reward、done／truncated、`BatchActionInfo.info["decision"]`を保持する。decisionはboolのlane列で、省略時は全laneを判断とみなす。打ち切り後のdummy slotは終端観測の保存専用であり、windowの途中には入れない。
- `UpdatePriorities`はin-flight fetchとqueued Pushの完了を待ってからinnerへ委譲し、sampleとmutationの順序を固定する。futureは同時に最大1件であり、worker例外はfutureの`get()`または次の同期境界でcallerへ再送出する。

### 7. 設定・lifetime・エラー・性能特性

- shutdown時はprefetch workerと未完了eventを回収してからstorageを破棄する。
- 共通 ReplayBuffer は Agent の checkpoint に含まれず、再開時は空の buffer を構築する。成分別の step 報酬と収益は復元されず、成分数と再計算フラグは新しい Run の設定から決まる。
- `PrefetchingReplayBuffer`は呼出時までに受理したPushとin-flight prefetchをFIFO順でsettleしてからinnerへ委譲するが、通常prefetched batchは消費・並べ替えない。このためprobeの有無で通常sample列を変えない。
- Prefetching版は既存probeと同じFIFO待機を行い、受理済みPushとprefetchをsettleしてから委譲する。次回の通常batchは保持する。DQN の群抽出 RNG は、内側 Learner が Agent の基の seed と `replay_fit_probe` の名前から構築時に作って所有し、購読の再設定でも保持する。

### 8. テストと拡張時の確認事項

- RFD0121検証資材: 成分ごとの収益と割引、定数の保持とscheduleの再計算、初期優先度、窓拒否、1成分の同一性。
- RFD0128検証資材: 全フィールドの転送・走査・文字列化、CPU/CUDAでの未定義保持、変更前の実行体との比較。
- RFD0104検証資材: action_change・episode_endの窓、終了・dummy・ring wrap・転送の確認。
- 可変窓の判断境界・終端・truncated・stack・ring折返し・一様抽選・sampleable遅延は RFD0098検証資材 の公開 ReplayBuffer テストを使う。window を追加する実装は通常転送・prefetch・probe経路も含め、規則無しのRNGとsample列を維持する。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
