# anet-stats summary (2026-11-18, HEAD a9a78864)

期間: 2026-11-12 〜 2026-11-18 (前回 digest: 2026-11-11)

## 期間の活動
- 稼働日 6、コミット 67、Run 起動 29、連続稼働 42 週
- コード 追加 12,231 / 削除 3,793(うちテスト 追加 6,176)、docs 追加 9,051 / 削除 1,768、config 追加 121 / 削除 56
- 主種別 feat、主 Topic #20 324_AGENT個別実装 > DQN系列、コード変更コミットの docs 同時更新 18/24
- PRD: 着手済み 6 本、未着手 55 本、未コミット 86 件(+2,082/-492)

## テスト充足(C++)
- 全体 test/prod 0.84、被テスト(直接) 57/127、被テスト(間接込み) 110/127
- 年代(境界 2026-05-30): 以前ファイル 53/63 被テスト、以後 57/64、以前に書かれた行の割合 45%
- 薄いカテゴリ: 325 AGENT個別実装 MuZero系列 0.00、319 GUI 共通基盤 0.07、313 プロファイリング 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1154行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/muzero_based_agent.hpp(260行,pre)、core/anet-core/include/anet/str_util.hpp(88行,pre)、core/anet-core/src/str_util.cpp(77行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 1.13 は厚い
- [watch] 設定キーが 4 週で +558(現在 3931)
- [good] 42 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1154 行、pre)
- [info] 期間中の bulk 除外 98,099 行: docs/experiments/default-dqn/atari/data/2026-11-12_rnd_performance_taiga_events.csv(26,895), docs/experiments/default-dqn/atari/data/2026-11-12_rnd_performance_on_events.csv(5,255), docs/experiments/default-dqn/atari/data/2026-11-12_rnd_performance_taiga_facade_events.csv(3,034)

## 参考
- TODO/FIXME 72(4 週前 72)、本版の課題 149
- 直近の完了週: 2026-10-19週: 稼働 7日 / コード +13,518 -3,299 / Run 78, 2026-10-26週: 稼働 7日 / コード +6,199 -620 / Run 128, 2026-11-02週: 稼働 7日 / コード +8,311 -520 / Run 93, 2026-11-09週: 稼働 7日 / コード +14,033 -4,227 / Run 25
