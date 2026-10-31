# Freeway

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

RAM の 0xEA は効果音のタイマーで、91 に入った回数が車に当たった回数と一致した（エミュレータ調査の 2 ゲーム）。面とボスが無く、渡った回数は得点そのものなので、RAM の番号は 4 だけを置いた。

## 分かっていること

- RAM 0x8E はニワトリの縦の位置（下端 6、上端 177）で、上を押すと 1 フレームに 1 増え、渡りきると 6 へ戻った。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0xEA は車に当たると 92 か 91 から 65 まで減り、得点では 142 から 129 までの別の範囲を使った。91 に入った回数は上だけを押した 1 ゲームで 52 回、上と待ちを混ぜた 1 ゲームで 35 回で、どの回も 12 フレーム以内にニワトリが 7〜13 押し戻された。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- 既定設定の RAM 定義は 4 = `car_hit`（0xEA reach_count:91）だけで、面とボスが無く渡った回数は得点そのものなので 1〜3 は置かない。（出典: [220_atari_env.jp.md](../../design/220_atari_env.jp.md)、§4.9 RAM メトリクス。[082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、D21・Context「beam_rider と freeway の RAM」）

## 分かっていないこと

- DOWN を多く使う方策でも、0xEA が 91 に入る回数が車に当たった回数と一致するか。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、既知の限界と後続「freeway の 0xEA」）

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [220 §4.9 RAM メトリクス](../../design/220_atari_env.jp.md)。
- [PRD082](../../memo/082_atari_ram_progress_metrics_10prd.md)。
