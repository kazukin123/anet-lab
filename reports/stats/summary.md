# anet-stats summary (2026-11-04, HEAD ba59ce5c)

**digest は追記しない**: 前回 digest が 2026-11-04 まで書かれており、それ以降の新しい期間が無い(同日の再実行)。 期間を指定して書き直す場合は `--since <YYYY-MM-DD>` を使う。以下の規模・テスト・プロセスの数字は現在値なので参照してよい。

期間: 2026-11-05 〜 2026-11-04 (前回 digest: 2026-11-04)

## 期間の活動
- 稼働日 0、コミット 0、Run 起動 0、連続稼働 40 週
- コード 追加 0 / 削除 0(うちテスト 追加 0)、docs 追加 0 / 削除 0、config 追加 0 / 削除 0
- 主種別 None、主 Topic -、コード変更コミットの docs 同時更新 0/0
- PRD: 着手済み 0 本、未着手 65 本、未コミット 168 件(+4,633/-2,758)

## テスト充足(C++)
- 全体 test/prod 0.76、被テスト(直接) 55/123、被テスト(間接込み) 102/123
- 年代(境界 2026-05-30): 以前ファイル 53/67 被テスト、以後 49/56、以前に書かれた行の割合 49%
- 薄いカテゴリ: 325 AGENT個別実装 MuZero系列 0.00、319 GUI 共通基盤 0.06、313 プロファイリング 0.07
- 未テスト上位: core/anet-core/src/muzero_based_agent.cpp(1149行,pre)、core/anet-core/src/config_impl.cpp(588行,post)、core/anet-core/src/scaler_impl.cpp(767行,pre)、core/anet-core/src/muzero_based_agent.hpp(259行,pre)、core/anet-core/src/scaler_impl.hpp(209行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 1.11 は厚い
- [watch] 設定キーが 4 週で +404(現在 3651)
- [good] 40 週連続で稼働
- [info] 未テストで最大は core/anet-core/src/muzero_based_agent.cpp(1149 行、pre)

## 参考
- TODO/FIXME 72(4 週前 72)、本版の課題 110
- 直近の完了週: 2026-10-05週: 稼働 5日 / コード +8,009 -2,610 / Run 53, 2026-10-12週: 稼働 6日 / コード +5,465 -1,117 / Run 55, 2026-10-19週: 稼働 7日 / コード +13,518 -3,299 / Run 78, 2026-10-26週: 稼働 7日 / コード +6,199 -620 / Run 128
