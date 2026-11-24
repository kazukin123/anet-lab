---
name: anet-impl-rfd
description: Implement a repo RFD through docs-aware questions, a saved implementation plan, self-check, and review, then report unresolved items and stop. Use when the user invokes /anet-impl-rfd with a RFD filename such as 0013_sample_prefetch_10rfd.md, or asks to implement a *_10rfd.md file while saving a *_2ximpl.md plan.
---

# RFD 実装起動

## クイックスタート

通常の呼び出しは次の形として扱う:

```text
/anet-impl-rfd 0013_sample_prefetch_10rfd.md
```

RFD ファイル名だけが渡された場合は、次の意図として解釈する:

1. RFD を解決して読む。
2. RFD に暫定の印があれば、RFD のレビューを受ける。
3. コード、`CONTEXT.md`、ADR に照らして未決事項監査ゲートを通す。
4. 実装判断を変える未解決点だけを質問する。
5. ブロッカーが 0 になったら、合意した実装計画を次に対応する `*_2ximpl.md` として保存する。
6. 実行が許可されている場合は、その保存済み計画から実装を開始する。
7. 自己点検、レビューと差し戻し対応を行い、未決を示して停止する。

## 必須参照

作業前に次を読む:

- [RFD-IMPLEMENTATION.md](RFD-IMPLEMENTATION.md)（「プロセス契約との対応」まで適用する）
- [プロセス契約](../../../docs/process/protocol.md) と [レジストリ](../../../docs/process/registry.md)
- [anet-create-rfd](../anet-create-rfd/SKILL.md)
- [CONTEXT-FORMAT.md](../anet-create-rfd/CONTEXT-FORMAT.md)
- [ADR-FORMAT.md](../anet-create-rfd/ADR-FORMAT.md)

このスキルでは、`anet-create-rfd` のドメイン用語、コード照合、`CONTEXT.md`、ADR のルールを適用する。`../anet-create-rfd/` は編集しない。
コード変更を伴う実装では、利用可能なら `tdd` skill も読み、`RFD-IMPLEMENTATION.md` の TDD 実装フェーズを適用する。

## ワークフロー

1. `RFD-IMPLEMENTATION.md` の規則で RFD パスを解決する。
2. RFD、現在の実装、近いテスト、設定、`CONTEXT.md`、関連 ADR を読む。RFD に暫定の印があれば、`RFD-IMPLEMENTATION.md` の「暫定の印がある RFD」に従って RFD のレビューを受ける。
3. 未決事項監査ゲートの前に、[実装 レビュー観点](../../../docs/quality/impl.md)の「実装計画」を読み、読んだ節の名前を応答に 1 行で示す。その後、`RFD-IMPLEMENTATION.md` の未決事項監査ゲートを通し、RFD の決定点を repo evidence、ユーザー判断、`2ximpl` 前提、RFD 範囲外へ分類する。
4. リポジトリ探索で解ける疑問は、ユーザーへ質問せず探索で解決する。
5. 回答によって実装挙動または計画形状が変わる場合だけ、ユーザーへ一問ずつ質問する。
6. ユーザー判断が必要なブロッカーが 0 であることと、計画に固定する前提を短く提示する。
7. 最終計画を、解決済みの `*_2ximpl.md` ファイルへ日本語で保存する。
8. 保存済みの `*_2ximpl.md` を実装の正本として扱う。計画を保存したあと、TDD または編集の前に、`RFD-IMPLEMENTATION.md` の「書く前に観点を読む」に従って変更種類に応じた節を読み、読んだ節の名前を応答に 1 行で示す。コード変更を伴う場合は、計画内の TDD 順序に従って実装する。

9. `RFD-IMPLEMENTATION.md` の「プロセス契約との対応」に従って自己点検、レビューと差し戻し対応、未決の提示と完了報告まで進め、停止する。

## 出力ルール

- ユーザー向け要約と保存する実装計画は、ユーザーが別指定しない限り日本語で書く。
- 実行時ログ、設定キー、API 名、ファイルパス、エラーメッセージは、リポジトリ既存の言語規約を保つ。
- プランモードでは提案計画で止め、ファイル編集をしない。
- 実行モードでは、未決事項監査ゲートを通過してから実装計画を保存し、コード変更前にその計画を正本にする。コード変更時は TDD の red-green-refactor を守る。

レビューに出す前にRFDの課題（`rfd`でそのRFDを指す課題）のうち今回の実装の範囲に当たる課題の対応内容へ今回の対応を追記し、完了の報告前にそれらの課題を残件無し（実装したあとも事象が残るもの、段階の途中で残りの段階を持つものは残件有り）とする。RFDの実装済みは、RFDの課題から一覧の道具が導く。状態は、計画を保存したときに同じ課題を対応中にし、解決はコミットの準備で持ちかける。工程末の4区分とこの場で直す流れはレジストリ、対応内容の追記と判定の更新はRFD-IMPLEMENTATION.mdに従う。
