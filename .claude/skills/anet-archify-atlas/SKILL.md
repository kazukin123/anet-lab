---
name: anet-archify-atlas
description: Explicitly invoked workflow for investigating the current anet-lab checkout and generating its five-map architecture atlas with the required Archify skill. Use only when the user invokes $anet-archify-atlas; do not use for a single diagram.
---

# ANET Archify Atlas

anet-lab の現行 checkout を根拠から調査し、全体像から実行時詳細までを5種類の Archify マップとして `docs/archify/` に出力する。

## 起動条件

- `$anet-archify-atlas` で明示的に指定された場合だけ実行する。
- 最初に、利用可能なスキルから名前が正確に `archify` であるものを探し、その `SKILL.md` を最後まで読む。この確認が完了するまで、候補を含む成果物を作成・変更しない。
- `archify` を発見できない、または全文を読めない場合は、次のエラーを返して終了する。

  ```text
  ERROR: anet-archify-atlas requires the "archify" skill, but it is unavailable or unreadable. No artifacts were created.
  ```

- Archify が定める実行経路と fallback だけを利用する。Mermaid、draw.io、独自レンダラーなど、Archify 外の代替経路を追加しない。
- Plan Mode では調査と計画だけを行い、成果物を生成しない。実行可能なモードでのみ、以下のワークフローを進める。

## ワークフロー

1. `git rev-parse --show-toplevel` などのリポジトリ情報から root を特定する。root から適用範囲にある作業規約を読み、Git・ビルド宣言・プロジェクト資料の複数の手掛かりから対象が anet-lab であることを確認する。単一の固定パスだけを識別条件にしない。
2. [Atlas contract](references/atlas-contract.md) を全文読み、5図の役割、公開マニフェスト、品質基準を確定する。
3. 現在の Git revision を取得し、リポジトリを動的に調査する。
   - `rg --files`、ファイル名検索、内容検索、ビルドターゲット調査を組み合わせる。
   - 設計索引、用語集、ADR、ownership、設定契約、ビルド定義、実行 entrypoint、runtime/data/state flow を、名称を決め打ちせず役割と内容から発見する。
   - 現行動作はコードと設定、用語と設計意図は設計資料と ADR を優先する。
   - build output、Run artifact、vendor 実装などは事実の主要根拠にしない。外部依存は manifest、build declaration、公開境界から確認し、外部コード本体を広く読まない。
   - 資料と実装が食い違う場合は推測で統合しない。実証できる関係だけを描き、不一致を Atlas index に残す。
4. 5図を通して使う canonical term、主要な責任境界、主経路、外部依存、信頼境界を先に整理する。現在の識別子は保持し、説明文は日本語にする。
5. OS の一時領域へ5つの Archify JSON 候補を作る。各図について、読み込んだ Archify スキルの type router に従い、該当 schema、common schema、対応例だけを読む。候補作成後の検証・修正・delivery・visual-check も Archify の契約に従う。
6. 5図すべてを公開前に完成させる。
   - `meta.quality_profile` は `showcase` とする。
   - 実証された主経路と3つ以上の主要要素を含め、主要要素は最大12個に抑える。システム構成図は8〜12コンポーネントとする。
   - 詳細、責任、例外、根拠はカードへ集約し、説明のためだけにエッジを増やさない。
   - 各要素に載せる根拠は、判定力の高いリポジトリ相対参照を最大3件までとする。
   - `validate --quality showcase --json` で9件すべての artifact check が成功し、composition errors と warnings がともに0になることを必須とする。
   - validation 成功後の JSON は編集しない。`deliver` が非0で終了した図は成功扱いしない。
7. 一時領域で生成したHTMLに `visual-check --json` を実行する。
   - 画像を確認できる場合は、生成された contact sheet と light/dark screenshots を実際に目視する。
   - 目視で問題を見つけた場合は、1図につき最大2回まで原因を限定して候補を修正し、validation、delivery、visual-check をやり直す。
   - visual-check が overflow または capture failure を報告した場合は公開しない。
   - Chrome/Chromium または目視手段が利用できない場合は警告にとどめ、index の `visual_review` を `skipped` とする。目視していないものを reviewed と記録しない。
   - visual-check の画像、HTML、JSON sidecar は確認後に一時領域から削除し、永続化しない。
8. 全5図が契約を満たした場合だけ、`docs/archify/` へ公開する。
   - [Atlas contract](references/atlas-contract.md) の固定マニフェストだけを置換対象にする。
   - 既存の公開版がある場合は一時バックアップを取り、全ファイルの置換が完了するまで保持する。
   - 公開中に失敗した場合は、置換対象だけを前回版へ戻し、今回新設した対象は除去する。無関係なファイルには触れない。
   - 5組の JSON と HTML を公開した後、完了マーカーとして `README.md` を最後に置換する。
9. 今回作成した一時ファイルとバックアップだけを片付ける。`docs/archify/` にあるマニフェスト外のファイルは削除しない。

## 停止条件

- いずれかの図で、実証された主経路と3要素を確保できなければ、空の図や推測図を作らず Atlas 全体を更新しない。
- Archify の validation または delivery が失敗し、2回連続の限定修正でも最良のエラー数を改善できなければ停止する。
- 証拠不足が一部に限られ、各図の最小条件を満たす場合は、既知範囲だけを描いて不足を index に記録する。
- 失敗時は、どの段階で止まったか、前回版を維持できたか、未解決の診断を正確に報告する。

## 完了報告

- 5図と Atlas index のリポジトリ相対パスを閲覧順に示す。
- 各図の type、showcase 検証結果、specification/artifact SHA-256、`visual_review` を示す。
- evidence gap と資料・コードの不一致を要約する。
- C++ の build/test はこのスキルの作業に含めず、実行していないことを明記する。
