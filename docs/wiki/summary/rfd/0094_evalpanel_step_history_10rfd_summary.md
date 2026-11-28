---
title: Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD
page_kind: サマリ
material: "[[docs/memo/done/0094_evalpanel_step_history_10rfd.md|0094_evalpanel_step_history_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
updated: 2026-11-28
---
# Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD

## 元の材料

[Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD](../../../memo/done/0094_evalpanel_step_history_10rfd.md) blob:7c3fc78323982e3a216afd7734f361d5670e4981

## 要点

- EvalPanel の1遷移を1件の step trace とし、行動、報酬、Q、優先度や Env の値を同じ場面で追えるようにする。
- scalar と trace で分かれていた値の参照先の解決をまとめ、メトリクスの配線を RunManager から分ける。
- 番号付きの列を宣言し、各遷移時点で実際に利用できる値を記録する。
- Evaluation Trace パネルの表で直近の履歴を選択・再生すると、EvalPanel、QValuePanel、Conv2d をその時点の表示へ切り替える。
- 手元で履歴を振り返る機能と、trace チャネルへ永続出力する計画の担当範囲を区別する。

## 内容

### 目的と問題

EvalPanel（鏡写しインスタンス）で遊ばせているゲームを、1 遷移ずつ追う手段が無い。行動・報酬・Q・優先度・env の途中の値を並べて見られず、ある場面で Agent が何を見て何を選び、その結果どうなったかを、あとから確かめられない。

### 背景（KFM）

TD 誤差による優先度が珍しい経験を目立たせないと考えられる理由は、報酬クリップの下では撃破の報酬が雑魚と同じ +1 であること、撃破の先の価値（出口の換算）がまだ target に入っていないこと、優先度は引かれたときにしか更新されないこと、IQN の TD は τ の揺らぎを含むこと。どれも推測で、場面ごとに確かめる手段が無い。

### 現状（2026-10-19 にコードで確かめたこと。§3.6 は 2026-10-22）

列は一律95pxで、内容に応じた幅になっていない。上部には文字の7ボタンがあり、再生と一時停止が別になっている（EvalTracePanel.cpp、構築とInitialize）。
報酬行の背景は明るい緑、終端行は明るい橙、文字は黒の固定色で、ダークテーマのシステム色に合わせていない。同じファイルのTableが配色している。
メインツールバーの Step（EvalPanel を 1 step 進める）のアイコンは、右向き三角の右に縦棒を置いた形である（同ファイル、`kStepSvg`）。動画のプレイヤーで一般的なコマ送りの形（縦棒の右に右向き三角）とは、順が逆である。

### 確定事項

数値: 値が整数（小数部が 0）なら整数で、そうでなければ有効数字 4 桁で出す。NaN は空欄、done・truncated は 0/1（論理値の表示をそろえ、空欄を「値が無い」だけに使うため。得点のような大きな整数を丸めないため）。
性能の見積もり: 列の幅を決める処理は、新しい行の表示値 C 件と、あとで決まった表示値 K 件だけを対象にする（表示する 1 step あたり C+K 件の幅の判定）。持っている全行を step ごとに測り直さず、列を広げるのは、それまでより長い値が出たときだけにする。学習の step の順序と worker の構成は変えない。
Trace モードに入ったら、EvalPanel を一時停止する（調べている間に、選んでいる行が上限で消えないようにするため）。Live に戻っても再開はしない。再開は EvalPanel の Resume で行い、それが Live に戻るきっかけにもなる（2026-10-19 のユーザーの裁定）。
再生／一時停止は1個のボタンとし、再生中は一時停止のアイコンとツールチップ`Pause`、LiveまたはTraceの一時停止中は再生のアイコンと`Play`を出す。押すと現在の状態に応じた操作を行い、最新行への到達などで再生が止まった場合も表示を同期する。

### User Stories

研究者として、EvalPanel で遊ばせているゲームの直近の遷移を、行動・報酬・Q・近似優先度・env の途中の値を並べた表で見たい。
研究者として、選んだ時点から時系列で自動に送って、場面の流れを見たい。
研究者として、KFM のボス撃破や出口の換算の step で、近似優先度がふつうの step より高いかを確かめたい。
研究者として、env ごとに見たい値（Atari の得点・残機・RAM）を、設定で列に足したい。

### 成功の測り方と受入基準

近似の確からしさ: `actor_approx` で回した Run で `39_agent_per/01_actor_learner_rank_corr` を確かめ、近似と学習側の優先度がどれだけ揃うかを把握する。
段階 1 の等価性: 編集前と後で、同 seed・決定的モードの短い Run（Atari、DropMerge、LunarLander の `run.@cpu`、GridMaze）の metrics checksum、`metrics.scalar.defs`・`metrics.trace.defs` の中身、WARN の行が一致する。
単体テスト（Agent の step 部品）: 合成したヒント・報酬の系列について、`approx_priority` が replay の初期優先度の推定（`InitialPriorityCompleter` 経由の値）と一致する。n step の途中の終端、打ち切り（空欄）、TBO、Munchausen の有無、reward scaler の倍率を含む。
学習の値が変わらない: ヒントのヘルパーを切り出す前と後で、`actor_approx` の同 seed・決定的モードの短い Run の metrics checksum が一致する。
単体テスト（列の宣言）: 構築時の fail-fast（番号が整数でない・0 以下、鍵が 2 つ、target が無い、未知の target、`$agent`・`$exp`・`$update_result`・`$env`、`$runner` のエピソード集計以外への集約の接頭辞、未知の属性、同じ属性の重複、空の `label:`、`display:name` を数値の列に、連続行動の `action` と `display:name`）と、正しい宣言の解析結果（番号順、label の既定）。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
