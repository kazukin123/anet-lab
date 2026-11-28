---
id: issue-20261125-de8cfd
title: Actor Q ヒントの入力（ActorQHintValues）の組み立てが Actor と step 診断の 2 か所にあり、共有しているのは最後の計算だけである
status: 解決
raised: 2026-11-25
phase: 監査
scope: src/algo/dqn/src/dqn_actor.cpp の Actor::MakeAction と dqn_assembler.cpp の DqnAgentStepDiagnosticsProvider::Observe
kind: コード
area:
  - 321_agent_core
type: リファクタリング
domain: 学習基盤
ai_judge: 残件無し
ai_judged: 2026-11-28
docs:
  - docs/adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md
rfd: "[[0207_dqn_small_fixes_and_refactors_10rfd]]"
priority: 中
depends: []
parent:
value: 3
urgency: 5
risk: 3
size: 2
wsjf: 5.5
resolved: 2026-11-28
resolution: 対応済み
memo:
tags: []
---

## 背景

2026-11-25 の監査（観点 architecture、範囲 321_agent_core）で見つけた。分類は重複生成源（locality）である。開発者には質問していない。

ADR 0058 と RFD0094 は、step trace の近似優先度を、Actor と共有するヒント計算で作ることで、Replay の初期優先度との一致を守ると決めた。今、共有の範囲は最後の関数 `MakeActorQHint` だけである。

ホットスポット補正: `dqn_actor.cpp` は、`git log --since=2026-08-27 --follow` で 90 日に 9 回変わっており、コードのファイルの上位 5%（7 回以上）に当たる。urgency を 3 から 5 へ 1 段上げた。

## 事象

- Actor の側は `dqn_actor.cpp:1588-1601` で組む。`q_values` の有無の検査、Munchausen の値、TBO、分解なら 4 つの aux（`kExtrinsicQValues`・`kIntrinsicQValues`・`kExtrinsicActionScores`・`kCombinedActionScores`）の取り出し、外的の bootstrap である。
- step 診断の側は `dqn_assembler.cpp:127-148` で、同じ検査と同じ 4 つの aux の取り出しを別の文で書く。Munchausen の値は、`DQNActionInfo` の scalar を番号（`scalar(4)`・`scalar(5)`）で引く（[issue-20261125-5ebf2f](issue-20261125-5ebf2f.md)）。
- 外的の bootstrap への変換 `extrinsic_bootstrap_mode == kBootstrapExtrinsic` は、`dqn_assembler.cpp:145` と `default_dqn_agent.cpp:938` の 2 か所にある。
- step 診断は、分解のとき `MakeActorQHint` を 2 回呼ぶ（136・146 行）。1 回目は、合わせた Q の `q_sa`・`state_value` を列に出すためだけに使う。読み出しは、1 回目が要素ごとの `item<float>()`（163 行）、2 回目が accessor（167 行）で、2 通りある。
- 検索語と件数（テストを除く）: `MakeActorQHint\(` の呼び出し 3、`kExtrinsicActionScores`・`kCombinedActionScores` は書き込み 1・取り出し 2 か所ずつ。

## 影響

- ヒントの入力を足すとき（RFD0147 の分位点の対の優先度、RFD0172 の複数の Head、Munchausen の値の増減）、2 か所を同じに直す。step 診断を直し漏らすと、Evaluation Trace の `approx_priority`・`extrinsic_q_sa` などの列が、Replay の初期優先度と黙って食い違う。

## 対応方針

- 「`DQNActionInfo` の aux と、ヒントの設定から `ActorQHintValues` を作る」関数を 1 つ置き、Actor と step 診断が使う。合わせた Q の値は、その関数の別の出力として 1 回の計算で返す。`ActorQHintConfig` を分ける（[issue-20261125-fa1be2](issue-20261125-fa1be2.md)）なら、その設定を受ける。
- 性能の区分は「中立」（Actor は step ごと。step 診断は pane か trace の要求があるときだけ）。
- トピック別まとめの「変えるときの条件」では、RFD0094 §4.2 D6・D7・§8 の 3（actor_approx の checksum の一致、通知の時機）と RFD0161 §6.2 D11 が当たる。
- テスト: `dqn_concrete_test.cpp:3512`・3598（step 診断と Replay の完了の一致）、3634（`MakeActorQHint` の直接の呼び出し）、`dqn_actor_test.cpp:1395`・1624、`dqn_munchausen_test.cpp:289`・342 が通す。
- 不採用の形・ADR との突き合わせ: ADR 0058（Actor の設定で eval_panel だけにヒントを作らせない）と ADR 0075（列の数をモードで変えない）を守る。
- [issue-20261123-c3b5ef](issue-20261123-c3b5ef.md)（n 先の hint を待って Estimate に渡す手順の写し）は、同じ `Observe` の別の部分の写しなので、同時に直すかを決める。

## 対応内容

- 2026-11-28 Actor と step 診断の Q ヒントの入力を共通化した。分解時の合成 Q、各 Head、初期優先度の対応を確かめるテストを追加した。 [RFD0207 実装メモ](../../memo/0207_dqn_small_fixes_and_refactors_20impl.md)


- 2026-11-28 実 ReplayBuffer の公開読み出しで得た最初の格納位置の初期優先度と、step 診断の値を同じ lane ごとに照合した。重点テストが通過し、短縮の actor_approx 構成は 38,112 点が一致した。 [検証結果](../../memo/0207_dqn_small_fixes_and_refactors_20impl.md#検証結果)
$1
- 2026-11-28 状態を `対応中` から `解決` へ変更。コミットの準備で解決の候補を選んだ。AI 判定は「残件無し」。Actor Q ヒントの入力を共通化し、Replay 初期優先度と step 診断を同じ lane で照合した。
