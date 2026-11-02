# replay の slot に 1 step の reward と終わり方を持ち、サンプル時に差し替え可能な窓の規則で可変長の区間を組む

> 2026-10-29 注記（[PRD104](../memo/104_temporl_skip_auxiliary_only_10prd.md) D3・D4・D7）: Consequences の「窓は判断の旗と終わり方で区切る」は判断で区切る規則（`decision`）の話で、PRD104 は skip の出力を行動に使わない構成（全 Actor がスキップ長を常に 1 にする補助学習のみモード）に限り、行動の変化で閉じる規則（`action_change`）と、上限 J と終わり方だけで閉じる規則（`episode_end`）を足した。1 step の記録、sampleable の遅れ、sampling 分布、item key、PER の ready 基準、Push と Extract の lock は変わらない。

n-step return は Push の時点で確定し、slot には結果（`target_returns_`・`terminals_`・`actual_n_steps_`）だけが残るため（[ADR 0044](0044-replay-frame-history-start-from-episode-start-at-push.md) は END を slot 単位に持たないと決めた）、起点から同じ行動が続いた区間のような可変長の窓をサンプル時に組めなかった。**slot に 1 step の reward（scale 後）、終わり方（なし／done／truncated）、判断の旗を 1 step の記録として常に持ち、抽出が窓の規則（まず TempoRL の、判断で決めた区間の中の窓、上限 J）で起点ごとに長さ k を決めて割引報酬と k 先の観測を組み、sampleable の列挙の遅れを max(1, n_step, 上限) に広げる（優先度の完成と eviction 統計の遅れは変えない）**ことを決定する（[PRD 098_temporl_skip_policy](../memo/098_temporl_skip_policy_10prd.md)）。論文どおりの skip 用の別 replay は PER・prefetch・ready・世代 key が 2 系統になり obs を複製するとメモリが 2 倍になり、Push 時に区間を確定して slot に足す形は同じ起点から短い sub-skip を取れないため採らない。ADR 0044 のうち、履歴開始を Push 時の `episode_start` から確定し extractor がそれだけを stack の境界に読む判断は維持し、END を slot に持たない部分だけを改める。次の判断まで遡る n-step と n-step の目標値の補正は、同じ記録に欄を足し、窓の規則を差し替えることで乗る。

## Considered Options

- **skip 用の replay を別に持つ（論文の形）**: 既存に触れないが、2 つのバッファの整合を検査する場所が無く、他の可変長の窓にも使えない。却下。
- **Push 時に区間を確定して同じ slot に skip 版（k_max・R・done）を足す**: サンプル時の走査は無いが、1 step の reward が無いので R を途中で切れず、論文の j(j+1)/2 の sub-skip を捨てる。却下。
- **slot に 1 step の記録を持ち、サンプル時に組む（採用）**: slot +5 B（Atari +0.07%、DropMerge +0.18%）、サンプルごとに最大 J 回の走査と加算。変更が extractor と slot の配列に閉じる。

## Consequences

- 1 step の記録は skip の有効・無効によらず書く（分岐を減らし、他の PRD の土台にする）。サンプルの値には無効では使わない。
- sampleable の列挙の遅れと容量の下限検査が窓の上限を含む（優先度の完成と eviction 統計は n_step の遅れのまま）。無効では上限 1 で今と同一。
- sampling 分布、item key、PER の completion と eviction 統計の ready 基準、Push と Extract の lock、index の昇順は変えない。
- 窓は判断の旗と終わり方で区切る。判断の境界を越えて同じ行動を数えると、次の判断で同じ行動を選び直すかが次の状態に依存するため、確率的な env で選び直しの偏りが入る。dummy slot の action が 0 で NOOP と同じ値になりうるが、終わり方で止まるので跨がない。
- 詳細は [PRD 098_temporl_skip_policy](../memo/098_temporl_skip_policy_10prd.md) §5.1・§5.2。
