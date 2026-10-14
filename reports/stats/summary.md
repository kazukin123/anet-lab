# anet-stats summary (2026-10-14, HEAD 60127df1)

期間: 2026-10-08 〜 2026-10-14 (前回 digest: 2026-10-07)

## 期間の活動
- 稼働日 5、コミット 13、Run 起動 23、連続稼働 37 週
- コード 追加 7,972 / 削除 2,604(うちテスト 追加 4,253)、docs 追加 1,526 / 削除 403、config 追加 102 / 削除 0
- 主種別 feat、主 Topic #11 315_ニューラルネットワーク、コード変更コミットの docs 同時更新 7/8
- PRD: 直下 0xx 8 本、9xx 18 本、未コミット 112 件(+2,904/-1,304)

## テスト充足(C++)
- 全体 test/prod 0.65、被テスト(直接) 47/120、被テスト(間接込み) 90/120
- 年代(境界 2026-05-30): 以前ファイル 44/68 被テスト、以後 46/52、以前に書かれた行の割合 54%
- 薄いカテゴリ: 319 GUI 共通基盤 0.00、325 AGENT個別実装 MuZero系列 0.00、314 テンソル処理 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1148行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/scaler_impl.cpp(745行,pre)、core/envs/gridmaze1/src/GridMazeEnv.cpp(288行,pre)、core/anet-core/src/muzero_based_agent.hpp(258行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 0.80 は厚い
- [watch] 設定キーが 4 週で +264(現在 3334)
- [info] TODO/FIXME が 4 週で +5(参考値)
- [watch] 直下の 0xx PRD が 8 本(WIP 多め)
- [good] 37 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1148 行、pre)
- [info] 期間中の bulk 除外 9,614 行: docs/experiments/default-dqn/lunarlander/config/run_20261006-233110_ll_qr_nzadd_s1.txt(829), docs/experiments/default-dqn/lunarlander/config/run_20261006-233110_ll_qr_nzrep_s1.txt(829), docs/experiments/default-dqn/lunarlander/config/run_20261006-234614_ll_qr_nzrep_s2.txt(829)

## 参考
- TODO/FIXME 72(4 週前 67)、台帳 open 4
- 直近の完了週: 2026-09-14週: 稼働 6日 / コード +6,341 -3,361 / Run 45, 2026-09-21週: 稼働 6日 / コード +4,986 -950 / Run 81, 2026-09-28週: 稼働 2日 / コード +391 -210 / Run 40, 2026-10-05週: 稼働 5日 / コード +8,009 -2,610 / Run 32
