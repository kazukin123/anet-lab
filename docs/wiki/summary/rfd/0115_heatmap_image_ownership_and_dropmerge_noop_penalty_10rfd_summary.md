---
title: 画像メモリの所有権と DropMerge の NOOP 判定
page_kind: サマリ
material: "[[docs/memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md|0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-28
---
# 画像メモリの所有権と DropMerge の NOOP 判定

## 元の材料

[RFD0115　画像メモリの所有権と DropMerge の NOOP 判定](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md) blob:f6204149a174f9458713396717ca2791b68a6f23

## 要点

- 画像を作る四つの経路で、メモリの確保と破棄方法を wxWidgets の所有権契約に合わせる。
- 確保に失敗したときは画素に触れる前に止め、対象と要求寸法をエラーに含める。
- 画像の寸法・RGB 値・画像種別と、返却後に画像自身がメモリを解放する寿命を保つ。
- DropMerge では行動番号だけで判断せず、その行動モードで選んだ命令が NOOP かを調べる。
- 四つの行動モードを同条件で比較し、NOOP への罰が1回の Env step に正しく反映されるか確認する。

## 内容

### 問題とゴール

画像生成では、メモリを確保する方法と、画像を破棄するときの解放方法が合っていない。DropMergeでは、行動モードごとのNOOPの意味と、NOOPへの罰の判定が合っていない。どちらも既存の責務の中で直せる不具合である。
題材は画像メモリの課題とDirectモードの課題。RFDの作成・レビュー完了と、不具合の実装・受入完了は区別する。今回は前者までを行う。

### 起草時点の現行コードで確認した事実

同ファイル653〜690行の `processAction` は、直接指定モードと移動モードを分ける。移動モードで `noop_override=true` なら、選択されたNOOPを中央方向への移動へ変換するが、呼び出し元が受けた命令の意味は変えていない。また、DROPは落下中などに実行されないことがある。

### 確定事項

2.1の4経路では、画像が所有するメモリを、その画像の契約に適合する方法で確保・解放する。画像を返したあとも、返された画像自身の寿命に従って正しく解放される。公開APIと画像の寸法・RGB画素値・画像種別は維持する。空のTimeHeatMapのScale描画は従来どおり1×1の黒画像である。

### 実装契約

汎用のImageSource・MetricsLogger・Env基盤へ、DropMerge固有の条件を持ち込まない。
新規・削除するクラスはない。TimeHistogramは委譲先の修正を受けるため受入に含めるが、それ自体の契約は変更しない。内部の確保方法、関数分割、テストの配置は、上記契約の中で実装計画に記す。

### 受入条件と成功の測り方

加えて両個体の罰を0にした場合に、報酬を含む全結果が一致することを4モードすべてで確認する。累積報酬は返されたStep結果が持つ現在のエピソードの値を読み、View用の前エピソード履歴を代用しない。Step報酬と累積報酬の両方を検証し、スコアを報酬の代わりに比較しない。終了stepの結果も確認対象とする。

### 現用の追随先

この一覧が移行対象の正本である。実装工程で内容を確認し、挙動に関する説明とテストを同じ変更内でそろえる。今ある設定キーと値の移行は不要だが、それを理由にテストや説明の確認を省かない。

### 起票時の検証記録

相対リンク切れなし、115の番号重複なし、禁止された状態行なし、UTF-8/LF、公開できない値の混入なしを確認した。

## 主要な概念

- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
