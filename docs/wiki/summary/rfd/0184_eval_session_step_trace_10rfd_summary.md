---
title: 評価セッションの遷移を Run に記録する
page_kind: サマリ
material: "[[docs/memo/0184_eval_session_step_trace_10rfd.md|0184_eval_session_step_trace_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# 評価セッションの遷移を Run に記録する

## 元の材料

[RFD0184　評価セッションの遷移を Run に記録する](../../../memo/0184_eval_session_step_trace_10rfd.md) blob:f50a2e6b703bcfee5b5d3d9e408db4b8c7cce910

## 要点

- スケジュール評価の集計だけでは個々の状態での行動・価値と、その直後の結果を追えなかった。
- Qbert の残機喪失や飛び降りを調べるため、評価セッションの各 step を Run に記録する暫定案を置いた。
- 採用するエピソード、lane、step の識別子を対応させ、複数回の評価の記録を混同しないようにする。
- 既存の評価集計と手元の Evaluation Trace の役割を分け、後から調べるためのデータを選べるようにする。
- 同じ評価条件の記録と成績を照合し、記録の有無で評価方策が変わらないことを確かめる。

## 内容

### 記録を求めた理由

Qbert を Greedy で評価した3ゲームでは、最後の残機ではない残機を失っても状態価値がほぼ下がらなかった。より多くのエピソードで確かめるには、セッション全体の集計に加え、各遷移の行動・価値・報酬を後で読める記録が要る。

### 未合意の記録方法

起草時の既定は、評価セッションと列を宣言し、採用エピソードの遷移だけを trace チャネルへ1遷移1行で出す案だった。行動選択時の値と環境の Step の結果を同じ行に置き、後から決まる列を待って出す。学習側への適用や EvalPanel の手動の評価とは分ける。

### 案を採用した場合の確認

記録の宣言がない場合は収集の部品を作らない。記録を有効にした場合は採用されなかったエピソードの混入、終了時に行が欠ける場合、複数の評価をまたぐ識別を検査する。行の形や未予定セッションをどう扱うかは再グリルで見直す暫定条件である。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Qbert](../../concepts/subjects/qbert.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
