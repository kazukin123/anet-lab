# BeamRider

## 今の結論

RAM の 0x80 はセクター番号、0x94 はボスを魚雷で倒したフレームだけ 141 になり、0xD3 は魚雷の残りだった（エミュレータ調査）。ボスを倒すと残機ぶんのボーナスが入り、その間は残機が減って見える。

## 分かっていること

- RAM 0x80 はセクター番号（1 始まり）で、セクターが変わるフレームで 1 ずつ増え、ランダムな行動の 12 ゲームでは死亡とゲームオーバーで増えなかった。15 機を倒しきったあとのボスの場面で死ぬと、復帰と同じフレームで増えた。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0xD4 はそのセクターで残っている敵の数で、セクターの開始で 15、白い円盤を倒すたびに 1 減り、15 機目のあと 0 → 255 になった。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0x94 は、ボスを魚雷で倒したフレームで 1 回だけ 141 になった（セクター 2・3 で 1 回ずつ、得点は 360・390）。倒さない枝と倒していない 2 万フレームでは 141 に入らず、ゲームオーバーでは 140 で止まった。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- RAM 0xD3 は魚雷の残り（セクターの開始で 3）で、魚雷の行動のフレームだけ 1 減り、死亡では減らなかった。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- ボスを倒すと残機の数だけボーナス（セクター 2 は 120 点、3 は 130 点）が 1 回ずつ入り、そのたびに残機の 0x85 が 1 減って最後に戻った。ALE の `lives()` も 3 → 1 → 3 と動くので、episodic_life を有効にしたプロファイルではボスを倒すたびに残機を失った扱いになる。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context「beam_rider と freeway の RAM」）
- 既定設定の RAM 定義は、1 = `sector_clear`（0x80 inc_count）、2 = `boss_kill`（0x94 reach_count:141）、4 = `torpedo_fired`（0xD3 dec_count）である。ボスは魚雷 1 発で倒れるので 3 は置かない。（出典: [220_atari_env.jp.md](../../design/220_atari_env.jp.md)、§4.9 RAM メトリクス。[082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、D21・Context「beam_rider と freeway の RAM」）

## 分かっていないこと

- セクター 1 と、セクター 4 より先でも、ボスを倒したときに 0x94 が 1 回だけ 141 になるか。（出典: [082_atari_ram_progress_metrics_10prd.md](../../memo/082_atari_ram_progress_metrics_10prd.md)、既知の限界と後続「beam_rider の確認範囲」）

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [220 §4.9 RAM メトリクス](../../design/220_atari_env.jp.md)。
- [PRD082](../../memo/082_atari_ram_progress_metrics_10prd.md)。
