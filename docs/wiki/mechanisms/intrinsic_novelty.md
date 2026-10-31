# 内的報酬による探索の比較

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

NGU 型のボーナスと E3B の比較結果は、学習器・環境・合成方法に依存する。MiniHack での結果を Atari の優劣へ持ち込めず、anet の BTR を基にした構成での効果はまだ測っていない。KFM の命中にボーナスが出るかも、逆動力学の特徴がバーの差を保つかに依存する。

## 分かっていること

- 文献の比較条件: NGU 原論文は R2D2 を基にした再帰型の学習器を使う。RLeXplore の共通学習器上のモジュール比較とは構成と予算が違い、公開スコアから同じ条件での順位は付けられない（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.8）。
- 文献の比較条件: Henaff ら（2023）の NGU 型ボーナスとの比較は主に MiniHack で、Montezuma's Revenge では E3B 単独と RND・併用でも結果が異なる。E3B を NGU の無条件な上位互換とする根拠にはならない（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.8）。
- 未検証の仮説: KFM のバーの縮みを逆動力学の特徴が保存すれば、報酬 0 の命中にも新しさの手応えを与えられる。ただし自機の姿勢だけで行動を予測できる可能性があり、制御可能な変化が必ず特徴に残る保証は無い（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.5）。
- 調査の範囲: anet の BTR を基にした構成での NGU／E3B の効果は未測定であり、今回の調査では直接比較に使える BTR＋NGU 全体の実績も確認できていない（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§3「現状」・§2.8「NGU・E3B・BTR の関係と、比較結果の適用範囲」）。

## 分かっていないこと

- 同じ anet の学習器・経験予算で、個々の訪問ベクトルとの局所的な近さを測る NGU と、まだ十分に経験していない特徴方向を評価する E3B のどちらが、KFM・GridMaze の外的成績を改善するか（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.8「NGU・E3B・BTR の関係と、比較結果の適用範囲」）？
- 探索による経験の増加が、点数を目的とする方策の未訪問状態からの到達率へ転移するか（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.7）？

## 撤回・更新

- 2026-10-27: PRD の 2026-10-21 改訂記録にある「命中のたびにボーナスが付く」という説明 → 特徴がバーの差を保つかに依存する未検証の仮説へ限定した（出典: [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、冒頭の 2026-10-21 改訂記録・§2.5「新しさのボーナスが KFM に効くと見込む理由」）。

## 関連

- [KFM](../subjects/kung_fu_master.md)。
- [評価と比較の条件](evaluation.md)。
- [内的報酬の PRD](../../memo/999_intrinsic_reward_novelty_10prd.md)（機構・設計案と未決の正本）。
