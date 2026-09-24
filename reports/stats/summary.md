# anet-stats summary (2026-09-19, HEAD c07622e)

期間: 2026-09-18 〜 2026-09-19 (前回 digest: 2026-09-17)

## 期間の活動
- 稼働日 2、コミット 2、Run 起動 9、連続稼働 33 週
- コード 追加 129 / 削除 4(うちテスト 追加 82)、docs 追加 33 / 削除 4、config 追加 0 / 削除 0
- 主種別 fix、主 Topic #24 040_不具合対応、コード変更コミットの docs 同時更新 0/1
- PRD: 直下 0xx 3 本、9xx 15 本、未コミット 91 件(+2,370/-1,735)

## テスト充足(C++)
- 全体 test/prod 0.58、被テスト(直接) 47/120、被テスト(間接込み) 90/120
- 年代(境界 2026-05-30): 以前ファイル 51/75 被テスト、以後 39/45、以前に書かれた行の割合 57%
- 薄いカテゴリ: 319 GUI 共通基盤 0.00、325 AGENT個別実装 MuZero系列 0.00、314 テンソル処理 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1148行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/scaler_impl.cpp(745行,pre)、core/envs/gridmaze1/src/GridMazeEnv.cpp(288行,pre)、core/envs/cartpole2/src/CartPoleEnv.cpp(245行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 0.76 は厚い
- [watch] 設定キーが 4 週で +730(現在 3232)
- [info] TODO/FIXME が 4 週で +5(参考値)
- [good] 33 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1148 行、pre)

## 参考
- TODO/FIXME 72(4 週前 67)、台帳 open 0
- 直近の完了週: 2026-08-17週: 稼働 6日 / コード +6,844 -1,955 / Run 62, 2026-08-24週: 稼働 4日 / コード +9,420 -2,716 / Run 35, 2026-08-31週: 稼働 5日 / コード +7,609 -1,476 / Run 58, 2026-09-07週: 稼働 7日 / コード +1,912 -264 / Run 32
