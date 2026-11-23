# anet-stats summary (2026-11-23, HEAD 748587f8)

**digest は追記しない**: 前回 digest が 2026-11-23 まで書かれており、それ以降の新しい期間が無い(同日の再実行)。 期間を指定して書き直す場合は `--since <YYYY-MM-DD>` を使う。以下の規模・テスト・プロセスの数字は現在値なので参照してよい。

期間: 2026-11-24 〜 2026-11-23 (前回 digest: 2026-11-23)

## 期間の活動
- 稼働日 0、コミット 0、Run 起動 0、連続稼働 43 週
- コード 追加 0 / 削除 0(うちテスト 追加 0)、docs 追加 0 / 削除 0、config 追加 0 / 削除 0
- 主種別 None、主 Topic -、コード変更コミットの docs 同時更新 0/0
- RFD: 着手済み 6 本、未着手 55 本、未コミット 293 件(+7,677/-5,044)

## テスト充足(C++)
- 全体 test/prod 0.86、被テスト(直接) 69/147、被テスト(間接込み) 136/147
- 年代(境界 2026-05-30): 以前ファイル 60/64 被テスト、以後 76/83、以前に書かれた行の割合 42%
- 薄いカテゴリ: 319 GUI 共通基盤 0.07、313 プロファイリング 0.07、314 テンソル処理 0.17
- 未テスト上位: src/algo/muzero/src/muzero_based_agent.cpp(1154行,pre)、src/core/base/src/config_impl.cpp(588行,post)、src/algo/muzero/src/muzero_based_agent.hpp(260行,pre)、src/core/base/include/anet/diag.hpp(74行,post)、src/core/base/include/anet/str_util.hpp(88行,pre)

## insight flags
- [good] 直近 4 週のテスト投資率 1.80 は厚い
- [watch] 設定キーが 4 週で +437(現在 3964)
- [good] 43 週連続で稼働
- [info] 未テストで最大は src/algo/muzero/src/muzero_based_agent.cpp(1154 行、pre)

## 参考
- TODO/FIXME 72(4 週前 72)、本版の課題 189
- 直近の完了週: 2026-10-26週: 稼働 7日 / コード +5,094 -501 / Run 128, 2026-11-02週: 稼働 7日 / コード +7,130 -380 / Run 93, 2026-11-09週: 稼働 7日 / コード +8,188 -1,008 / Run 25, 2026-11-16週: 稼働 7日 / コード +20,903 -6,320 / Run 37
