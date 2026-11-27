---
title: "Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り"
page_kind: サマリ
material: "[[reports/sil_value_based_lower_bound_survey_2026-11-24.md|sil_value_based_lower_bound_survey_2026-11-24]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/self_imitation.md|自己模倣による価値学習]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り

## 元の材料

[Survey: 自己模倣（SIL）を価値ベースへ広げた手法と、確率的な環境での偏り](../../../../reports/sil_value_based_lower_bound_survey_2026-11-24.md) blob:eedbb4fe4bbfd824329d77b0778b8e74b43e5ec6

## 要点

- 「確率的な環境」は出典の語に合わせ、遷移の確率性、報酬の確率性、行動の繰り返しの確率（sticky actions）を含む。sticky actions は、各時刻に確率 ς で直前の行動が実行される設定で、標準は ς = 0.25 である [Machado ALE (2017/09), §5.2]。
- Montezuma's Revenge で、SAIL-IQN は最終 833（期間平均 513）、RND-IQN は 161（45）、IQN は 0。SAIL-IQN に RND と n = 3 を足すと最終 5180 で、Taïga らの RND-Rainbow と並ぶ、と著者は述べる [Ferret SAIL (2020/12), §4.4 図 8]。
- SAIL-IQN の論文は、IQN にも同じ報酬の書き換えを掛けるとだけ書き、分位点への掛け方は書かない [Ferret SAIL (2020/12), §4.3]。
- 記憶の書き方を平均に寄せる: MFEC は、max を外して収益をそのまま書くと楽観は弱まるが、予備実験では性能が下がったと書く（Atari を決定的と明記した文脈）[Blundell MFEC (2016/06), 脚注 1、§4.1]。MBEC は近傍の記憶へ距離に応じて書き込み、平均への収束を示し、確率的な古典制御 9 課題で MFEC・NEC・DQN を上回った（数値は図だけ）[Le MBEC (2021/11), §3.2、§4.2、付録 A.2]。
- 価値ベースの学習器で、sticky actions のある Atari の難探索ゲームまで数値を出したのは、調べた範囲では SAIL だけだった。Montezuma's Revenge は SAIL-DQN が 0、SAIL-IQN が 833 で、RND と n = 3 を足すと 5180 だった（§3）。

## 内容

### 1. 対象と読み方

「下限」は、価値の推定が実際の収益より低いときだけ引き上げる片側の掛け方を指す。「両側」は、記憶の値へ上下どちらからも寄せる掛け方を指す。

区分: 査読つきの論文、査読なしのプレプリント、著者の公式実装、著者側のブログ、第三者の記事を、出典一覧で分ける。

### 2. 価値ベースへ広げた手法

損失は、収益 R が Q を上回るときだけ二乗誤差を掛ける片側の形 ½((R − Q)₊)² である。根拠は、決定的な環境では行動方策の収益が最適ソフト Q の下限になることで、この関係が成り立つのは「環境が決定的なとき」と条件つきで書かれている [Oh SIL (2018/06), §4.2 式 (11)〜(13) とその直後]。

Bellman・下限・n-step の演算子の凸結合を考え、下限を含むと不動点の偏りは正の向きに限られ、上は最適 Q で抑えられると示す [Tang LBQ (2020/06), §4 式 (9)、定理 2]。

### 3. 難探索ゲームの数値

SAIL の論文は総学習量を数で書いていない。図 2 の横軸は 0〜100M フレーム、図 10 の横軸は 0〜200M フレーム [Ferret SAIL (2020/12), §4.1、図 2・図 10]。

EMDQN、GEM、DB-ADP の論文には、難探索ゲームの個別の値が無い（§2.3・§2.6・§2.7）。

### 4. 記憶の持ち方と設定の値

値は公式実装の既定か起動スクリプトの値。論文の値が違うものは §4.3 に書く。

SAIL: 論文は加算の項に ReLU を掛けず [−1, 1] に切る。IQN 版のコードは同じ形だが、DQN 版のコードは relu を掛けてから [0, 1] に切るので負の加算が出ない [sail_rl 実装, agents/sail_dqn.py の _build_target_q_op、agents/sail_iqn.py の _build_target_quantile_values_op]。論文は IQN を n-step なしで使うと書くが、IQN の gin には n = 3 の指定がある。実行時にどちらの値になるかは未確認 [Ferret SAIL (2020/12), §4.3; sail_rl 実装, configs/atari_iqn.gin]。

### 5. 分布型での扱い

分布型（C51・QR-DQN・IQN）と組み合わせた報告は、調べた範囲では SAIL-IQN だけだった（検索語は §10）。

公式実装では、Q(s, a) と max の Q を、目標網の分位点を 32 個平均した値から取る。加算の項は遷移ごとの 1 つのスカラーで、報酬に足してから、64 個の目標分位点のすべてに同じ値を足す（分布全体の平行移動）。分位点 Huber 損失（κ = 1）は IQN のまま [sail_rl 実装, agents/sail_iqn.py の _build_target_quantile_values_op、configs/atari_iqn.gin]。

### 6. 確率的な環境での偏り

SIL 本体: 確率性の有無や強さを変えて SIL の悪化量を示した報告は見つからなかった（検索語は §10）。

### 7. 偏りへの対策と効果

拡散係数（後ろ向き更新）: EBU は、確率的な有限 MDP で拡散係数 β が十分小さければ収束すると示し、確率的な遷移を入れた MNIST の迷路で他の手法を上回った [Lee EBU (2018/05), §4.2 定理 2、§5.1 図 2(c)]。

### 8. 批判と懸念

決定性を前提にしてよいとする側の主張: 近決定的な状況は多いので、それを使う仕組みは学習を速くする [Blundell MFEC (2016/06), §2; Lin EMDQN (2018/05), §2.2]。下限による正の偏りは最適 Q で上から抑えられ安全で、実験では有益だった [Tang LBQ (2020/06), §4.2]。運のよい遷移の系列を確実に繰り返せない問題は、自分たちの実験では大きな障害にならなかった [Ecoffet FRTE (2020/04), 本文 "Policy-based Go-Explore"]。

### 9. 総合評価

記憶の持ち方は、SIL 本体と Tang が別の優先度つきの記憶、SAIL と Optimality Tightening が replay の遷移に収益を書き足す形、EMDQN と DB-ADP が別の表である。上位何本のエピソードだけを残す方式は見つからなかった。SIL の公式実装は、正の報酬を含むエピソードだけを入れる（§4）。

### 10. 調査の限界

版の違い: SAIL は arXiv の初版と AAMAS 版の Table 1・2 の数値の一致を確かめた。初版では sticky なしの相対改善の値が本文と図 9 の説明文で違う。Go-Explore の Nature 版は掲載前の preprint を読んだ。Machado らは arXiv 版を読み、JAIR 版は読んでいない。

## 主要な概念

- [自己模倣による価値学習](../../concepts/methods/self_imitation.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
