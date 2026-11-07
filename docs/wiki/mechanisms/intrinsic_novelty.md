# 内的報酬による探索の比較

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

NGU 型のボーナスと E3B の比較結果は、学習器・環境・合成方法に依存する。MiniHack での結果を Atari の優劣へ持ち込めず、anet の BTR を基にした構成での効果はまだ測っていない。KFM の命中にボーナスが出るかも、逆動力学の特徴がバーの差を保つかに依存する。

## 分かっていること

- 文献の比較条件: NGU 原論文は R2D2 を基にした再帰型の学習器を使う。RLeXplore の共通学習器上のモジュール比較とは構成と予算が違い、公開スコアから同じ条件での順位は付けられない（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.8）。
- 文献の比較条件: Henaff ら（2023）の NGU 型ボーナスとの比較は主に MiniHack で、Montezuma's Revenge では E3B 単独と RND・併用でも結果が異なる。E3B を NGU の無条件な上位互換とする根拠にはならない（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.8）。
- 未検証の仮説: KFM のバーの縮みを逆動力学の特徴が保存すれば、報酬 0 の命中にも新しさの手応えを与えられる。ただし自機の姿勢だけで行動を予測できる可能性があり、制御可能な変化が必ず特徴に残る保証は無い（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.5）。
- 調査の範囲: anet の BTR を基にした構成での NGU／E3B の効果は未測定であり、今回の調査では直接比較に使える BTR＋NGU 全体の実績も確認できていない（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§3「現状」・§2.8「NGU・E3B・BTR の関係と、比較結果の適用範囲」）。
- 切り出し: RND による一生の新しさは lane 別の記憶を持たず、facade（`DefaultDQNAgent::UpdateFromBatch`）で次観測を採点して報酬へ畳めば Replay・Actor の契約を変えずに腕にできる。KFM では対照腕で、判断に効くのは GridMaze の到達率と費用・統計の確認（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§1.1・§4 D1・D2・§7）。
- 切り出し: 一生の新しさの口は「倍率 α を返す採点・補助学習・診断・保存」の 4 操作で、RND の本体は DQN から見えない非公開の実装に閉じる暫定の設計（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§5.1・§5.7）。
- 比較対象の実装: Taiga ら（ICLR 2020）の Rainbow＋RND は、行動した観測 x_t を agent step ごとに 1 件で採点・学習し、誤差の z を分散に相当する値で割って 0.00005 倍し、Runner が [−1, 1] に clip した外的報酬に足す。合計は clip せず、負の値もそのまま入る。観測の統計は正規化後の値で更新する。Burda ら・anet が採点するのは次観測 x_{t+1}（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1・§8.1 の 6）。
- RND の尺度の条件: target と predictor の出力の尺度が揃っていれば、初期値の種類（he／xavier と直交）で新しさの判別は大きく変わらず、target だけ小さいと predictor が誤差を下げられず判別が落ちる。PyTorch の写しで Montezuma の画面を 6 万 step 学習した判断材料で、受入の基準ではない（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1）。
- 設定の落とし穴: anet の層カタログ `Conv2d_S`・`Conv2d_M` は葉の階層が読み込みと合わず、2026-11-06〜07 の Atari の RND の Run は既定の 128@3/1 ×2（1 Network 約 1 億パラメータ）で動いていた。写しの比較では判別は落ちず、実証された害は速さと checkpoint の大きさ（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1）。

## 分かっていないこと

- 同じ anet の学習器・経験予算で、個々の訪問ベクトルとの局所的な近さを測る NGU と、まだ十分に経験していない特徴方向を評価する E3B のどちらが、KFM・GridMaze の外的成績を改善するか（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.8「NGU・E3B・BTR の関係と、比較結果の適用範囲」）？
- 探索による経験の増加が、点数を目的とする方策の未訪問状態からの到達率へ転移するか（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、§2.7）？
- RND の target と predictor の形は、公開実装の形（predictor を深く、出力 512）と NGU の同形・出力 128 のどちらが長い学習で判別を保つか。写しの 6 万 step では同形のほうが誤差の比が大きい（出典: [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)、§2.1・§8 の 2）？

## 撤回・更新

- 2026-10-27: PRD の 2026-10-21 改訂記録にある「命中のたびにボーナスが付く」という説明 → 特徴がバーの差を保つかに依存する未検証の仮説へ限定した（出典: [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)、冒頭の 2026-10-21 改訂記録・§2.5「新しさのボーナスが KFM に効くと見込む理由」）。

## 関連

- [KFM](../subjects/kung_fu_master.md)。
- [評価と比較の条件](evaluation.md)。
- [内的報酬の PRD](../../memo/924_intrinsic_reward_novelty_10prd.md)（機構・設計案と未決の正本）。
- [RND の腕の暫定 PRD](../../memo/116_rnd_lifelong_novelty_10prd.md)（924 の第一の腕。一生の新しさの口と RND の実装）。
