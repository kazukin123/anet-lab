---
id: issue-20261125-bdbcac
title: 設計140 が本文で書く仕組みの ADR 8 本（評価セッションの集約・エピソード長・trace の行の時刻・forward の capture・遷移の結果・eval schedule・WARN の表・nn_trace の渡し方）を、どの設計文書もリンクしない
status: 未着手
raised: 2026-11-25
phase: 監査
scope: docs/design/140_observability.jp.md（と英訳）の ADR への参照
kind: 文書
area:
  - 317_metrics
  - 318_rl_core
type: 不整合
domain: 学習基盤
ai_judge: 未確認
ai_judged:
docs:
  - docs/design/140_observability.jp.md
rfd:
priority: 中
depends: []
parent:
value: 2
urgency: 3
risk: 1
size: 2
wsjf: 3
resolved:
resolution:
memo:
tags: []
---

## 背景

2026-11-25 の監査（観点 drift、範囲 `docs/design/140_observability.jp.md`）で見つけた。分類は ADR 入口なしである。開発者には質問していない。

AGENTS.md は、ADR を読むときは設計文書の該当章からの参照を優先するとしている。設計 200 と 160 の同じ種類の欠けは [issue-20261118-7f7757](issue-20261118-7f7757.md) と [issue-20261123-0fcb2a](issue-20261123-0fcb2a.md) が扱い、200 は §11.2 に ADR の一覧を設けて直した。ADR へのたどり方の全体の方針は [issue-20261114-fbcbca](issue-20261114-fbcbca.md) が扱う。

ホットスポット補正: 140 は 90 日の変更回数が 38 回で、設計文書（jp）の中で 200・160 に次いで多い。urgency を 2 から 3 へ 1 段上げた。

## 事象

140 がリンクする ADR は 0028・0029・0037 の 3 本である。140 が本文で仕組みを書いているのに、どの設計文書（`docs/design/*.jp.md`）からもリンクされず、`ADR 番号` の文字でも出てこない ADR が次の 8 本ある（`adr/NNNN` と `ADR NNNN` で検索）。0059 は 0060 で取り消されているので数えない。0045・0057・0082 は 0fcb2a が挙げる。

| ADR | 決めたこと | 140 の節 |
| --- | --- | --- |
| [0034](../../adr/0034-eval-session-aggregation-in-batchenv-decorator.md) | 評価セッションの N 本の集約は `EvalSessionEnv` が受け持つ | §5.1（151 行）、trace チャネル（327 行） |
| [0043](../../adr/0043-episode-steps-shared-scope-is-step-count.md) | エピソード長は SHARED でも Step の回数で、lane 数を掛けない | §5.1（151 行） |
| [0047](../../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) | 鏡写しの評価の trace の購読と、trace の行の時刻 | trace チャネル（325 行。時刻は未実装で、[issue-20261125-195ad5](issue-20261125-195ad5.md)） |
| [0055](../../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) | forward の持ち帰りは鍵要求型の capture 1 つ | §5.4 |
| [0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) | 遷移の結果は Step の結果で運び、速報値か確定値かは event で分ける | `env_step` の節（469〜475 行） |
| [0070](../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md) | eval schedule は既定で学習直後に評価せず、offset でずらす | §6（289 行）、§7.2 |
| [0071](../../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) | WARN の意味と、読まれない設定キーの検出 | §6（293・309 行）、§7.1 |
| [0072](../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md) | 行動選択に添える入力は Runner の step と `MakeAction` の引数で都度渡す | §5.4 |

- 上の決定の要点と 140 の該当の節は、0047 の時刻を除いて食い違わない。

## 影響

- 140 の該当の節を読んでも、なぜその形なのか（評価の集約を Env の decorator に置く、SHARED のエピソード長に lane 数を掛けない、終端の値を env へ聞かない、nn_trace を登録しない、など）にたどり着けない。ADR を読むべき変更で、ファイル名の一覧を引くしかない。

## 対応方針

- 200 §11.2 と同じく、140 の「関連文書」（§10）に ADR の節を設け、上の 8 本と、今ある 0028・0029・0037 を番号順に、対応する節とともに並べる。英訳も同じに直す。文書だけの直しで、コードは変えない。
- 0fcb2a（160）と同じ形で、同時に直せる。

## 対応内容
