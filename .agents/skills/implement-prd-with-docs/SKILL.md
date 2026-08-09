---
name: implement-prd-with-docs
description: Kick off implementation from a repo PRD file, using docs-aware grilling before saving an implementation plan. Use when the user invokes /implement-prd-with-docs with a PRD filename such as 013_sample_prefetch_10prd.md, or asks to implement a *_10prd.md file while saving a *_2ximpl.md plan.
---

# PRD 実装起動

## クイックスタート

通常の呼び出しは次の形として扱う:

```text
/implement-prd-with-docs 013_sample_prefetch_10prd.md
```

PRD ファイル名だけが渡された場合は、次の意図として解釈する:

1. PRD を解決して読む。
2. コード、`CONTEXT.md`、ADR に照らして未決事項監査ゲートを通す。
3. 実装判断を変える未解決点だけを質問する。
4. ブロッカーが 0 になったら、合意した実装計画を次に対応する `*_2ximpl.md` として保存する。
5. 実行が許可されている場合は、その保存済み計画から実装を開始する。

## 必須参照

作業前に次を読む:

- [PRD-IMPLEMENTATION.md](./PRD-IMPLEMENTATION.md)
- [grill-with-docs](../grill-with-docs/SKILL.md)
- [CONTEXT-FORMAT.md](../grill-with-docs/CONTEXT-FORMAT.md)
- [ADR-FORMAT.md](../grill-with-docs/ADR-FORMAT.md)

このスキルでは、`grill-with-docs` のドメイン用語、コード照合、`CONTEXT.md`、ADR のルールを適用する。`../grill-with-docs/` は編集しない。
コード変更を伴う実装では、利用可能なら `tdd` skill も読み、`PRD-IMPLEMENTATION.md` の TDD 実装フェーズを適用する。

## ワークフロー

1. `PRD-IMPLEMENTATION.md` の規則で PRD パスを解決する。
2. PRD、現在の実装、近いテスト、設定、`CONTEXT.md`、関連 ADR を読む。
3. `PRD-IMPLEMENTATION.md` の未決事項監査ゲートを通し、PRD の決定点を repo evidence、ユーザー判断、`2ximpl` 前提、PRD 範囲外へ分類する。
4. リポジトリ探索で解ける疑問は、ユーザーへ質問せず探索で解決する。
5. 回答によって実装挙動または計画形状が変わる場合だけ、ユーザーへ一問ずつ質問する。
6. ユーザー判断が必要なブロッカーが 0 であることと、計画に固定する前提を短く提示する。
7. 最終計画を、解決済みの `*_2ximpl.md` ファイルへ日本語で保存する。
8. 保存済みの `*_2ximpl.md` を実装の正本として扱う。コード変更を伴う場合は、計画内の TDD 順序に従って実装する。

## 出力ルール

- ユーザー向け要約と保存する実装計画は、ユーザーが別指定しない限り日本語で書く。
- 実行時ログ、設定キー、API 名、ファイルパス、エラーメッセージは、リポジトリ既存の言語規約を保つ。
- プランモードでは提案計画で止め、ファイル編集をしない。
- 実行モードでは、未決事項監査ゲートを通過してから実装計画を保存し、コード変更前にその計画を正本にする。コード変更時は TDD の red-green-refactor を守る。
