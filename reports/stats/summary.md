# anet-stats summary (2026-09-17, HEAD 83a2a81)

期間: 2026-08-20 〜 2026-09-17 (前回 digest: 無し)

## 期間の活動
- 稼働日 23、コミット 134、Run 起動 177、連続稼働 33 週
- コード 追加 25,591 / 削除 6,470(うちテスト 追加 10,398)、docs 追加 17,579 / 削除 3,207、config 追加 5,147 / 削除 4,141
- 主種別 feat、主 Topic #3 030_機能追加、コード変更コミットの docs 同時更新 31/70
- PRD: 直下 0xx 3 本、9xx 15 本、未コミット 19 件(+0/-0)

## テスト充足(C++)
- 全体 test/prod 0.57、被テスト(直接) 47/120、被テスト(間接込み) 90/120
- 年代(境界 2026-05-30): 以前ファイル 51/75 被テスト、以後 39/45、以前に書かれた行の割合 57%
- 薄いカテゴリ: 319 GUI 共通基盤 0.00、325 AGENT個別実装 MuZero系列 0.00、314 テンソル処理 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1155行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/scaler_impl.cpp(745行,pre)、core/envs/gridmaze1/src/GridMazeEnv.cpp(288行,pre)、core/envs/cartpole2/src/CartPoleEnv.cpp(245行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 0.76 は厚い
- [watch] 設定キーが 4 週で +581(現在 3085)
- [info] TODO/FIXME が 4 週で +5(参考値)
- [good] 33 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1155 行、pre)
- [info] 期間中の bulk 除外 321,614 行: docs/archify/anet_lab_20_training_step_sequence.html(14,959), docs/archify/anet_lab_00_system_architecture.html(14,925), docs/archify/anet_lab_30_experiment_dataflow.html(14,878)

## 参考
- TODO/FIXME 72(4 週前 67)、台帳 open 0
- 直近の完了週: 2026-08-17週: 稼働 6日 / コード +6,844 -1,955 / Run 62, 2026-08-24週: 稼働 4日 / コード +9,420 -2,716 / Run 35, 2026-08-31週: 稼働 5日 / コード +7,609 -1,476 / Run 58, 2026-09-07週: 稼働 7日 / コード +1,912 -264 / Run 32
