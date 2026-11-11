# anet-stats summary (2026-11-11, HEAD 56e49a8c)

期間: 2026-11-05 〜 2026-11-11 (前回 digest: 2026-11-04)

## 期間の活動
- 稼働日 7、コミット 82、Run 起動 65、連続稼働 41 週
- コード 追加 5,481 / 削除 909(うちテスト 追加 3,078)、docs 追加 8,239 / 削除 1,102、config 追加 311 / 削除 137
- 主種別 docs、主 Topic #7 311_開発プロセス/開発ツール/テスト基盤、コード変更コミットの docs 同時更新 18/21
- PRD: 着手済み 5 本、未着手 59 本、未コミット 17 件(+367/-19)

## テスト充足(C++)
- 全体 test/prod 0.78、被テスト(直接) 57/127、被テスト(間接込み) 106/127
- 年代(境界 2026-05-30): 以前ファイル 52/66 被テスト、以後 54/61、以前に書かれた行の割合 48%
- 薄いカテゴリ: 325 AGENT個別実装 MuZero系列 0.00、319 GUI 共通基盤 0.07、313 プロファイリング 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1149行,pre)、core/anet-core/src/scaler_impl.cpp(903行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/muzero_based_agent.hpp(259行,pre)、core/anet-core/src/scaler_impl.hpp(217行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 1.13 は厚い
- [watch] 設定キーが 4 週で +477(現在 3811)
- [good] 41 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1149 行、pre)
- [info] 期間中の bulk 除外 70,895 行: docs/experiments/default-dqn/gridmaze/data/2026-11-07_rnd_performance_events.csv(4,618), reports/stats/stats.json(4,363), docs/experiments/default-dqn/atari/config/run_20261108-232006_mz_tg_rnd_s1.txt(1,530)

## 参考
- TODO/FIXME 72(4 週前 72)、本版の課題 110
- 直近の完了週: 2026-10-12週: 稼働 6日 / コード +5,465 -1,117 / Run 55, 2026-10-19週: 稼働 7日 / コード +13,518 -3,299 / Run 78, 2026-10-26週: 稼働 7日 / コード +6,199 -620 / Run 128, 2026-11-02週: 稼働 7日 / コード +8,311 -520 / Run 93
