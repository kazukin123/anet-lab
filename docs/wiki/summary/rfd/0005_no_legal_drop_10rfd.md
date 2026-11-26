# DropMergeEnv の NoLegalDrop 終端追加

## 元の材料

[DropMergeEnv の NoLegalDrop 終端追加](../../../memo/done/0005_no_legal_drop_10rfd.md) blob:fa344dc5f27cea865bff90e5dd80b6262b042361

## 要点

- DropMerge で盤面が埋まり、どの位置への DROP でも次の個体を置けない局面が生じていた。
- この状況で NOOP を続けると、終局を待つだけの時間がエピソードに入る。
- 置ける DROP がなくなった局面を終端理由 NoLegalDrop として扱う。
- 判定は実際の生成位置の余白と drop noise が取り得る範囲を用いる。
- 既存の終端理由と区別して集計し、盤面と Q 値の観察で見つかった現象を追えるようにする。

## 内容

### NOOP が続いた盤面

`direct_noop` の評価で500ステップの無投下タイムアウトに達した局面の盤面と Q 値を調べると、多くはどの列に投下しても生成時に衝突する状態だった。DROP にはゲームオーバーのペナルティーが付き、NOOP なら追加報酬なしで待てる。この違いから、NOOP の回数だけでは操作を放棄したかどうかは分からない。

### いつ NoLegalDrop にするか

学習と評価の両方で、`direct_noop` の NOOP 時にだけ判定する。まだゲームオーバーでなく、投下装置が空いており、投下待ちの物体がなく、現在の果物のランクが1から `kFruitTypeCount` までの範囲にあることを確かめる。合体要求も破棄待ちの物体もなく、盤面の物理状態が静止したうえで、現在の果物を置ける列が一つもない場合に限り終了する。

各列について、実際の投下と同じ高さ・果物半径・壁の制限・衝突の余白を使う。投下ノイズで到達し得る横座標の範囲を果物の占有区間と照合し、置ける点が一つでも残れば継続する。判定の負荷は静止判定・列走査・区間判定ごとに測れる。

### 報酬・Replay・終了理由

NoLegalDrop は `done=true`、`truncated=false` で、無投下タイムアウトに上書きさせない。ゲームオーバーのペナルティーを付けず、Replay では終端以降の価値を bootstrap しない。同じ盤面で DROP を選んで衝突すれば、従来の SpawnBlocked とペナルティーが適用される。エピソード終了時に `term_reason_no_legal_drop` を記録し、Timeout・SpawnBlocked・Overflow・MaxStep と区別する。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
