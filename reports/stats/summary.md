# anet-stats summary (2026-10-22, HEAD e885a5ae)

期間: 2026-10-15 〜 2026-10-22 (前回 digest: 2026-10-14)

## 期間の活動
- 稼働日 8、コミット 33、Run 起動 69、連続稼働 38 週
- コード 追加 7,842 / 削除 2,043(うちテスト 追加 3,756)、docs 追加 6,091 / 削除 914、config 追加 149 / 削除 92
- 主種別 feat、主 Topic #13 317_可視化/メトリクス、コード変更コミットの docs 同時更新 9/11
- PRD: 直下 0xx 16 本、9xx 23 本、未コミット 51 件(+383/-433)

## テスト充足(C++)
- 全体 test/prod 0.70、被テスト(直接) 53/123、被テスト(間接込み) 94/123
- 年代(境界 2026-05-30): 以前ファイル 46/68 被テスト、以後 48/55、以前に書かれた行の割合 52%
- 薄いカテゴリ: 110 RunnerApp 0.00、325 AGENT個別実装 MuZero系列 0.00、319 GUI 共通基盤 0.04
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1148行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/scaler_impl.cpp(767行,pre)、core/anet-core/src/muzero_based_agent.hpp(258行,pre)、core/anet-core/src/scaler_impl.hpp(209行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 0.97 は厚い
- [watch] 設定キーが 4 週で +135(現在 3373)
- [watch] 直下の 0xx PRD が 16 本(WIP 多め)
- [good] 38 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1148 行、pre)
- [info] 期間中の bulk 除外 83,854 行: docs/experiments/default-dqn/dropmerge/config/run_20261017-173200_dmfb_44a_21_b1024.txt(1,420), docs/experiments/default-dqn/dropmerge/config/run_20261017-174838_dmfb_44b_21_b2048.txt(1,420), docs/experiments/default-dqn/dropmerge/config/run_20261018-074714_dmfb_42h_34a_b1024rr2.txt(1,420)

## 参考
- TODO/FIXME 72(4 週前 72)、台帳 open 26
- 直近の完了週: 2026-09-21週: 稼働 6日 / コード +4,986 -950 / Run 82, 2026-09-28週: 稼働 2日 / コード +391 -210 / Run 67, 2026-10-05週: 稼働 5日 / コード +8,009 -2,610 / Run 53, 2026-10-12週: 稼働 6日 / コード +5,465 -1,117 / Run 55
