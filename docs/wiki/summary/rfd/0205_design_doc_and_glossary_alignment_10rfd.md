---
title: 用語集と設計 140・160 を今のコードと ADR に合わせる小さい直しの案
page_kind: サマリ
material: "[[docs/memo/0205_design_doc_and_glossary_alignment_10rfd.md|0205_design_doc_and_glossary_alignment_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-27
---
# 用語集と設計 140・160 を今のコードと ADR に合わせる小さい直しの案

## 元の材料

[RFD0205　用語集と設計 140・160 を、今のコードと ADR に合わせる小さい直しをまとめて行う](../../../memo/0205_design_doc_and_glossary_alignment_10rfd.md) blob:32341be981a18fc9bf850a2b9697c73731a1724a

## 要点

- 暫定の起票で、確定した決定は無い。§6・§7 は §5 の未決表の暫定の既定を前提にした試案である。
- 2026-11-23・11-25 の監査（観点 drift）で見つかった、文書だけの直しの課題 10 件を題材にする。直し方はどれも課題に書いてある。
- 開発者が、仕様の調整がほぼ要らない小さい課題を集めて夜のうちに実装させたいと述べ、置き場ごとに 3 本（文書・基盤とアプリ・DQN 系）に分けた。本書はその文書の 1 本で、3 本のうち最初に実装する。
- 用語集は、network snapshot と評価セッションに `clone_model` の条件を、target return に γ の schedule の成分の作り直しを、Agent checkpoint に RND と報酬スケーラの保存を書き、step の軸（train・exp・update・learn・agent）と lane の 6 項を足す。
- 設計 212 と 200 の、checkpoint の読み込みを指す「再開」を「引き継ぎ」に改め、020 の引き継ぎの例を `A3.auto_load_file` にする。
- 設計 160 と 140 に、200 §11.2 と同じ形の ADR の節を設ける。160 は図の node の ID の重なり、コードマップ、FPS の選択肢を直す。140 は trace の行が時刻を持つという文を今の契約に戻し、§6 の節番号・置き場、図、コードマップを直す。
- コード、ADR の本文と注記、ほかの RFD の本文は変えない。ADR 0047 に「時刻は未実装」の注記を足さない（実装の状態は課題が持つ）。英訳は翻訳の skill で追随させる。

## 内容

### 問題と範囲

食い違いは、用語集の定義が今の動きと合わないもの（5 件）、設計文書から ADR へたどれないもの（2 件）、設計文書の図・並び・属性が今の形と合わないもの（3 件）に分かれる。`sim_step` の扱い、ほかの避ける語の残り、ADR へのたどり方の全体の方針、160 §7.4 の移動、画像の Observer の置き場、観測の正規化の統計の保存は、それぞれ別の課題が持つので扱わない。

### 未決表に置いた既定

agent step は「Agent が 1 回の行動選択で 1 lane に渡す 1 回の Env の Step」とし、`$agent_step`（Evaluation Trace の列の取り出し先）を避ける語に書く。140 の §6 は 6.1 step座標系・6.2 trace チャネル・6.3 Evaluation Trace の列・6.4 `env_step`・6.5 TempoRL と振り直す。140 の §6 の節番号を書く参照は RFD の本文の 6 行だけで、直さない。RFD0081 の「140 §6.x」は §6.2 に当たる。実装の順は本書、RFD0206、RFD0207 で、build とテストは要らない。

### 受入

用語集の各項の文字、設計文書の ADR のリンクの番号、図の node の ID、節の見出し、英訳の blob、リンク切れ、避ける語を、検索で確かめる。題材の課題の事象を 1 件ずつ照らした結果を実装計画に書く。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
