# anet-stats summary (2026-10-07, HEAD 7ee30c42)

期間: 2026-09-20 〜 2026-10-07 (前回 digest: 2026-09-19)

## 期間の活動
- 稼働日 10、コミット 58、Run 起動 141、連続稼働 36 週
- コード 追加 6,316 / 削除 1,292(うちテスト 追加 2,896)、docs 追加 5,421 / 削除 380、config 追加 155 / 削除 303
- 主種別 feat、主 Topic #7 311_開発プロセス/開発ツール/テスト基盤、コード変更コミットの docs 同時更新 18/27
- PRD: 直下 0xx 8 本、9xx 18 本、未コミット 93 件(+5,146/-844)

## テスト充足(C++)
- 全体 test/prod 0.65、被テスト(直接) 47/120、被テスト(間接込み) 90/120
- 年代(境界 2026-05-30): 以前ファイル 47/71 被テスト、以後 43/49、以前に書かれた行の割合 55%
- 薄いカテゴリ: 319 GUI 共通基盤 0.00、325 AGENT個別実装 MuZero系列 0.00、314 テンソル処理 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1148行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/scaler_impl.cpp(745行,pre)、core/envs/gridmaze1/src/GridMazeEnv.cpp(288行,pre)、core/anet-core/src/muzero_based_agent.hpp(258行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 0.67 は厚い
- [watch] 設定キーが 4 週で +342(現在 3334)
- [info] TODO/FIXME が 4 週で +5(参考値)
- [watch] 直下の 0xx PRD が 8 本(WIP 多め)
- [good] 36 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1148 行、pre)
- [info] 期間中の bulk 除外 93,912 行: docs/agents/harness_map_architecture.html(14,906), docs/agents/harness_map_workflow.html(14,867), docs/experiments/default-dqn/atari/config/run_20260930-044852_a5_qbert_btr50m_env64_uqe.txt(1,189)

## 参考
- TODO/FIXME 72(4 週前 67)、台帳 open 4
- 直近の完了週: 2026-09-07週: 稼働 7日 / コード +1,912 -264 / Run 32, 2026-09-14週: 稼働 6日 / コード +6,341 -3,361 / Run 45, 2026-09-21週: 稼働 6日 / コード +4,986 -950 / Run 81, 2026-09-28週: 稼働 2日 / コード +391 -210 / Run 40
