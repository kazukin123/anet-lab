# PRD094 速度表記・最小幅・列見出し配色の追加調整

## 概要

ユーザーの追加指定を実装する。低速表示をx0.125/x0.25/x0.5へ統一し、Evaluation Traceの全操作バーと左端からアクション列までが収まる最小幅を設定する。列見出しはLive/Traceで同じ標準配色とし、Traceの目印は赤枠とcaptionに保つ。正本PRDと日本語設計・ガイドを同期する。

## 主な変更

- TracePlayerのGetSpeedLabelと速度操作のstatus通知を小数表記にする。倍率の値・段階は維持する。
- EvalTracePanelは操作バー全体の必要幅と、設定順で最初のaction列までの現在の列幅の合計（罫線・スクロールバーの余白を含む）の大きい方を最小幅にする。action列が無ければ操作バーだけを基準とする。連続行動は最初のaction.[n]列までを対象とする。初期化・列幅の変更・手動幅・開き直し・テーマ/フォント/DPI変更で更新する。残りの列は横スクロールのままで、固定列は追加しない。
- RunnerFrameのAddAuxPaneとレイアウト幅縮小処理は各pane/windowの最小幅を尊重する。全補助pane一律100までTraceを縮める経路を除く。Trace以外へ新たな固定幅を要求しない。
- wxGridのoverlay選択は行選択時に全列見出しを強調するため、同機能を無効にして通常のシステム色による行選択へ切り替える。列見出しの配色をLive/Traceで変えず、操作バー赤枠を維持する。
- PRD D10/D12・受入、設計160・ガイド020・検証READMEと関係wikiを同期する。既存20/21impl、他作業のdirty変更は保持する。

## テスト

公開のTracePlayer速度操作、wxGridの行選択・列幅イベントと描画、window/paneのGetMinSizeと実サイズを通す。最初に速度表記1件をRED→GREEN、その後最小幅1件、列見出し1件をそれぞれRED→GREENとする。過去結果を新規成功に流用しない。

## 検証

VsDevCmd経由で分離Debug/ReleaseのAnetRLRunner-gui-test・anet-core-testをビルドする。build.pyは初回CMake configure後は--skip-configureで同じ分離設定を使う。Debug/Release×light/darkの通常[prd094]と描画[trace_ui_visual]、終了回帰、core関連と全体CTestを実行する。新しい速度段階すべて、操作バー全体とアクション列、列幅成長・手動幅・フォント/テーマ追従を検証する。ログは.scratch/prd094/ui-polish-20261022へ保存する。終了時にcacheを元へ戻し、通常Runner実行体をhashで保護する。PRDと実装を別々に既存の独立レビュー担当へ渡す。

## 前提

追加質問への回答は「表の列見出し（step、actionなど）」。ユーザー判断の必要なブロッカーは0。通常のactive/inactive pane caption配色は維持し、Trace専用の列見出し強調は行わない。既存Runは停止・再起動せず、通常binを上書きしない。GPUを使う学習Run・性能測定は追加しない。実OSテーマ/DPI移動と人による実Runner確認は未実施なら明記する。stage・commit・pushは行わない。

通常step/tickへ新たなロックや同期を追加しない。最小幅の再計算は列幅/外観が実際に変わった時に限定し、paneが最小幅未満の時だけ再配置する。既存のGetSpeedLabel/CreateTraceSnapshot/ReturnToLive等と同じ意味の名前を使い、public APIや設定キーは追加しない。

## 追加指定

列幅を節約するため、既定のtruncated列はlabel:truncで表示する。取得キーと値はtruncatedのまま。common.txt、現用設定比較器、PRD/ガイド/描画の例を同期する。恒久テストへ一度きりの比較fixtureは復活させず、設定解決の前後比較を検証ログへ残す。

## 実施記録

速度表記は新規[trace_speed_label]で旧分数の3項目失敗をREDとして確認し、小数へ変更後[trace_speed_label],[trace_player]の2件39項目成功（red-speed.json、green-speed.json）。倍率自体と端の動作は維持した。

最小幅は旧版の操作バー必要幅322pxに対しGetMinSize=1でREDを確認した。修正後、公開pane最小幅・実dock幅・action列の右端・手動幅・フォント変更・開き直しの1件9項目が成功（red-minimum.json、green-minimum.json）。

列見出しのテスト初回ビルドはwxGrid内部型のGetColLabelWindowを使って失敗した。公開のGetGridColLabelWindowへ訂正後、修正前の行選択で列見出し2列が強調され、LiveとTraceのRGBも異なる3失敗をREDとして確認した（red-header-0.logはコンパイル失敗、red-header-v2-result.jsonはビルド成功、red-header.jsonはbehaviorのRED）。その後、overlay選択の無効化だけを追加した。

列見出しは修正後1件8項目成功（green-header.json）。行選択後も見出しが非強調で、Live/Traceの見出し画像のRGBが全画素一致する。最小幅の追加回帰としてfloatとaction列無し、GUI上のx0.5表示を確認する。設定解決比較はscratchだけの独立probeで行い、恒久target/testへ一度限りのfixtureを追加しない。

## 検証・自己点検の途中結果

- Debug本体・GUI・coreとscratchだけの設定probeのビルド成功。最小幅（dock/float、action列無し、手動幅、フォント、開き直し）と見出し描画、GUI上のx0.5を合わせた4件35項目が成功（final-focus.json）。
- ConfigManagerでCartPole/Atari/DropMerge/LunarLander/GridMaze/ImageClsの5834キーを解決比較し、全環境でapp.eval_trace_panel.columns.[8]へのlabel:trunc追加だけが差分（config-probe-result.json、config-equivalence.json、resolved-*.json）。設定比較器の期待値も同期した。恒久テストへのone-off復活は無い。check_default_leaves.pyはaudited3474・errors0。
- PRDの30品質項目を自己点検し、同じPRD独立レビュー担当が1往復でOK。原文は.scratch/prd094/ui-polish-20261022/review-prd-20261022-2334.md。倍率とtruncatedキー維持、最小幅・見出し非強調・ラベルと受入の整合を確認した。新規不足・範囲外検出・開発者への確認事項はなし。
- 実装品質62項目を適用範囲に従い自己点検した。計画との対応・public event/描画によるRED→GREEN・設定比較・既存契約の維持を確認。新しい公開API、設定キー、Agent所有権、ログは無い。最小幅は既存の幅変更時と外観変更時だけ再計算し、paneの実幅が不足した時だけmanager.Updateを追加する。通常step/tickへ新しい同期や全行走査を追加していない。
- 変更テキストのLF/BOM維持、diff check成功、共有契約のproject語0（static-checks.json）。対象外のexisting dirtyは保持する。Debug GUI/core全体、Releaseビルド/GUI、cache復元と保全は進行中で、最終結果を追記する。

## 未実施・未確認

実OSテーマの切替・モニタ間DPI移動、人が実Runnerで操作する確認、既存PRD094のAtari実速度・学習速度・メモリ受入は実施していない。自動のlight/dark別processとwx通知・フォント変更・描画比較を実機受入と混同しない。学習Runや性能比較Runは追加しない。

独立レビュー途中で、設計160のmanager.Update条件の古い限定が新しい最小幅不足時の再配置と不整合と指摘された。最小幅が変わり実際のpane幅が不足した場合も含める説明へ同期した。コード変更はない。また開始時snapshotとのwiki差分に、別作業によるdumpリンク6件の削除とPRD095のlog行が混在している。今回の責任範囲はRunner wikiのPRD094の成立の経緯1行の追記更新と、今回のPRD094 writeback行だけであり、他作業の変更は保持・今回の判定外とする。

品質59の動詞比較として、新設UpdateMinimumSizeは既存UpdateAppearance・UpdateIcons（EvalTracePanel.cpp）と同じく、UIプロパティを現在の条件へ更新するUpdateを用いる。公開名の追加や異なる意味での動詞転用はない。

## 最終検証結果

- VsDevCmd経由の分離Debugビルド（AnetRLRunner-gui-test、anet-core-test、scratch設定probe）とReleaseビルド（GUI/core）が成功。コマンド全文・終了コードはbuild.pyの各*-result.jsonとログへ保存した。
- Debug/Release各々でlight/darkの通常GUIは12件104項目、描画回帰は1件38項目がすべて成功。終了回帰は各構成3反復、各2件15項目が成功（gui-Debug-results.json、gui-Release-results.json）。Debugのlight-narrowとdark-traceのPNGを読み、全操作バーとaction列、trunc見出し、標準配色、赤枠を確認した。
- Debugの関連core（[prd094],[observers],[runner]）57件2194項目成功。全体CTestは742件中741成功・既存skip1、150018項目成功（core-results.json、core-ctest.log、core-ctest-details.log）。速度の全7段階、最小幅、列見出しは修正前のbehaviorのREDと修正後GREENを記録済み。
- READMEの既存期待値維持範囲を速度の表示変更に合わせ、元のUI調整21implと追加調整22implへの参照を区別した。独立レビューの文書同期指摘に対応したもので、追加のコード変更はない。
- 今回新たに未解決となった課題候補はなし。実OSテーマ/DPI・人による実Runner確認と既存の速度/メモリ受入は従来の未確認事項のまま保持する。

Debug/ReleaseのCMake注入cacheは開始時へ復元し、どちらもconfigure成功・復元一致（cleanup-checks.json）。対象外9ファイルと通常Runner Debug/Release実行体のSHA-256が開始時と一致した。分離した検証実行体をafter-binへ固定した。最終LF/BOM・git diff --check・比較器のPython構文検査も成功（static-final.json）。stage・commit・pushは実施していない。

## 完了報告

実施内容: 小数速度表記、操作バーとaction列を基準とした最小幅、Live/Traceで同じ列見出し配色、trunc表示を実装し、PRD・日本語設計・ガイド・検証README・関係wikiを同期した。上記Debug/Release・GUI・core・設定比較・静的検査が完了した。
変更ファイル: owned.jsonの15ファイル。前からの他作業の差分は保持した。
レビュー結果: PRDは1依頼でOK。計画・実装は同じ独立an-reviewへの1依頼の中で文書指摘を修正し、最終OK（差し戻し0件）。最終応答原文は.scratch/prd094/ui-polish-20261022/review-20261022-2345-impl.md。実OSテーマ/DPI・人による実Runnerと既存の速度/メモリ受入は未確認のまま。根拠なく見送った指摘・新しい範囲外検出はなし。
決まらなかったこと: 新規の課題候補なし。既存の未確認事項は本書の未実施・未確認欄に保持した。
判断してほしいこと: 今回追加分の仕様・実装に関する判断待ちはなし。
次にできること: 変更内容を確認し、人の判断でコミットする。通常実行体の差し替え、Run、stage・commit・pushへは進まず停止する。

## 追加対応計画：reward列順（2026-10-22）

ユーザー指定により既定列をstep、ep_step、action、reward、Q_sa、V_s、done、trunc、prio、prio_pctの順へ変更する。common.txtと設定比較器、PRD D13・日本語ガイド/設計とwikiを同期する。列順だけの設定変更でC++の変更はなく、新規恒久テストは追加しない。設定解決の全キーを前後比較し、列4〜6以外の変化がないことと既定葉監査を確認する。学習Run・性能測定は行わない。

列順追加対応の検証: scratchのConfigManager probeをVsDevCmd経由でDebugビルドし、6環境5834キーの全値を比較。列4〜6のreward/Q_sa/V_sの入替えだけが差分で、他の全キーは一致した。比較器COLUMNSも実効10列に一致。既定葉監査3474件・errors0、Python構文検査・LF/BOM維持・diff check成功。DebugのCMake注入は復元済み。検証資材は.scratch/prd094/reward-order-20261022。C++本体の変更がないためGUI/core全体の再実行は行っていない。前工程snapshotとのhash比較で通常Release実行体の変化を観測したが、mtime23:44:53は今回の分離ビルドより前であり、古い基準との比較を今回の保全違反とは扱わない。通常binへの書込みは行っていない。

列順追加対応の完了報告: commonと比較器、PRD・日本語guide/design・wikiを同期し、自己点検と別々のPRD/計画実装レビューが各1依頼でOK（不足/範囲外検出/確認事項なし）。commonの直前コメントも行動と報酬を隣接させる説明へ訂正した。レビュー原文はreward-order-20261022のreview-*-prd.mdとreview-20261022-2354-impl.md。新たに決まらなかったこと・判断待ち・課題候補はなし。前からの実機/性能未確認は維持。列4〜6以外の値は不変を確認済みで、stage・commit・pushはせず停止する。
