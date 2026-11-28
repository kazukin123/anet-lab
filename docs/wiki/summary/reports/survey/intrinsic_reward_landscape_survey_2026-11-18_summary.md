---
title: 内的報酬による探索の手法と比較条件
page_kind: サマリ
material: "[[reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md|intrinsic_reward_landscape_survey_2026-11-18]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/pseudo_counts.md|訪問回数と擬似カウントによる探索]]"
  - "[[docs/wiki/concepts/methods/return_and_explore.md|訪問済み状態へ戻る探索]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
updated: 2026-11-28
---
# 内的報酬による探索の手法と比較条件

## 元の材料

[内的報酬による探索の手法と比較条件](../../../../../reports/survey/intrinsic_reward_landscape_survey_2026-11-18.md) blob:bc9c452ca4b49855d17fc7fb338e12d62b9f174a

## 要点

- 探索の方法を、予測誤差、訪問回数・密度、エピソード内の新しさ、訪問済み状態へ戻る方式に分けた。何を入力に採点し、いつ新しさが消えるかで適用条件が変わる。
- RND は決定的な固定ターゲットで遷移の予測不能性を避けるが、操作して新奇な画像を出し続ける環境では成績が落ちた。ICM の逆モデル表現は行動に無関係な雑音を除ける一方、操作できない重要な情報を落とす場合がある。
- 擬似カウントは密度の変化から訪問回数を推定する。ハッシュや特徴の選び方で、未訪問状態への汎化と無関係な変化への反応が変わる。
- NGU はエピソードごとの記憶と全期間の RND を組み合わせる。MiniHack と Montezuma の比較では、文脈間で共有する構造の量によって、どちらの新しさが効くかが違った。
- Go-Explore は有望な状態を覚え、そこへ戻ってから探索する。状態復元を使う版、目標に戻る方策を学ぶ版、成功軌跡を頑健な方策にする工程を区別する。
- 同じ Rainbow での比較と、PPO 上で正規化などを揃えた比較の両方で、最良のボーナスはゲーム依存だった。論文間の得点はフレーム予算、基準実装、評価方法と併記する。
- 調査は 2026-11-18 時点。数値は HTML 等の原典の抽出に依存し、一部の表・版・実装条件は未照合である。

## 内容

### 対象・予測誤差

ICM は観測・行動・次観測から逆モデルの特徴を学び、特徴の順予測誤差を報酬とする。RND は次観測だけを使い、固定ランダム target を predictor が予測する。観測を標準化して切り、内的報酬は内的 return の標準偏差で割る。Disagreement は現在の特徴と行動から複数の順モデルを走らせ、予測のばらつきで採点する。AMA は予測誤差からモデルの予測共分散の項を引く。

sticky actions がある Montezuma では、ランダム特徴の順モデルが部屋の境界で往復した。これは ICM そのものとの比較ではない。RND でも、行動で無関係な画像を出せる環境では新しい画面が報酬の源になった。ICM は VizDoom の画面の 40% を白色雑音にしても高得点を保ったが、すぐに操作できない情報を特徴が捨てる可能性がある。その性質だけを変えた RND との直接比較は見つからなかった。

BYOL-Explore は未来の埋め込みを複数 step 先まで予測し、target は online encoder の移動平均。比較では固定 target が劣り、一段予測で遅くなる環境があった。Atari の主要な比較は学習中最高点で、Montezuma は 13,518、RND 4,554、ICM 300。基準の資源や構造の一致は不明。SND は一枚の観測に自己教師ありで学習する target を使い、同じ PPO で学習し直した RND と比べた。Montezuma の最高点は SND-V 14,973、RND 6,689。すべての組の差が有意だったわけではない。

### 訪問回数と密度

CTS は同じ状態をもう一度学習する前後の密度から実数の訪問回数を推定する。経験分布なら実際の回数に一致するが、学習により確率が下がらないという条件がある。PixelCNN は負の予測利得を切り、時間とともに縮尺を変える。#Exploration はハッシュで離散化して数え、φ-EB は価値近似と同じ特徴の上に密度を置く。NGU や E3B の回数に似た式については、著者が挙げた着想の出典を区別する。

Montezuma の CTS 付き DQN は 50M フレームで 15 部屋、ボーナスなしは 2 部屋。疎な報酬ではボーナスを価値へ速く伝える Monte Carlo 収益との併用が重要だった。PixelCNN は 57 ゲーム中 52 で最大得点が CTS を上回ったが、Montezuma では CTS が高かった。手設計のハッシュに敵の位置を含めると、遠くの敵の動きが新しい状態として数えられ、成績が落ちた。状態がほとんど再出現しない環境では、エピソード内の初訪問を判定する項も機能しにくい。

### エピソード内の新しさと方策の組

NGU は逆モデルで学んだ埋め込みをエピソード内の記憶と比較し、その報酬に全期間の RND から作った 1〜5 倍の係数を掛ける。行動と無関係な変化を除くことと、学習期間を通じてなじみの状態の価値をゆっくり下げることを分担する。Agent57 は内外の価値 Network を分け、actor ごとに方策の組から選ぶ。MEME は胴体を共有し、中央で方策を選び、全方策の学習、価値の正規化、トレースの打ち切りなど複数の変更を加えた。MEME は 1B フレーム予算のうち 390M で Atari 全 57 本の人間基準を上回った。

NovelD は次状態での RND の新しさの増分を初訪問でだけ与える。E3B は同じエピソードで見ていない特徴方向を測り、RE3 は固定埋め込みの k 近傍距離、RIDE は連続する特徴の変化と訪問回数を使う。Henaff らでは、異なるエピソードで共有できる構造が少ないとエピソード内、多いと全期間の探索が強かった。MiniHack では両方の積が改善し、Montezuma では全期間の RND にほぼ並んだ。和では同様の改善は得られなかった。

### 訪問済み状態へ戻る探索

Go-Explore は、訪問してボーナスが消えた先を再び目指せなくなることを detachment、戻る途中に探索の揺らぎで精密な行動列が崩れることを derailment と呼んだ。復元で戻る版はシミュレータ状態を保存し、見つけた軌跡を模倣して頑健な方策を学習する。policy-based 版はセルを目標とする方策で実際に戻り、sticky actions 下で Montezuma 97,728（12B frames）、Pitfall 20,093（10B）。セルの設計にはゲーム知識を使った。

DTSIL は記憶した軌跡に条件付けた方策で戻り、MEGA は達成済みで密度の低い目標へ戻る。Cell-Free Latent Go-Explore はセルの設計を学習した表現に置き換えた。IGE は保存先・戻り先・探索行動の判断に基盤モデルを使ったが、実験は決定的な環境に限られる。ELLM は観測の文章化と問い合わせ、Motif はオフラインの LLM 選好を必要とする。Motif の報酬はエピソード内のメッセージの出現回数で減衰する。

### 統一条件での比較と近年の動向

Taïga らは Rainbow、sticky 0.25、200M frames、5 seed に揃えた。Montezuma は RND 5,199.2、ICM 2,795.9、ε-greedy 780。別の難探索ゲームでは ε-greedy が同等のことが多く、60 ゲーム全体では NoisyNets 以外に有意な上積みは報告されなかった。RLeXplore は PPO 上の 8 手法で、観測と報酬の正規化、初期化、内外の価値 head の分け方が結果に影響することを示した。ALE 5 本では Montezuma・Venture は RND、Private Eye は PseudoCounts、Seaquest は E3B、Gravitar は外的報酬のみが最高だった。Craftax-1B では ICM・E3B・RND が PPO を上回らず、PPO-RNN が最も強かった。

### 採点の入力・計算費用・ベンチマーク

観測だけで採点できる方法と、行動や次観測、履歴、環境復元を要する方法では必要な資源が異なる。PixelCNN は通常 1 step に評価 2 回と更新 1 回、E3B は逆行列の更新を含み報告例でほかの手法より 1.5〜2 倍遅い。BYOL-Explore の Atari は学習 TPUv2 4 個・actor CPU 400 個・推論 TPUv2 4 個。AMA は RTX 2080Ti で Space Invaders 約 12 時間と報告した。RND・ICM の追加費用を実時間で比較できる値は確認できなかった。ベンチマーク横断の順位を作る根拠はなく、MiniGrid、MiniHack、Habitat、NetHack、Procgen、Craftax の各論文が使った基準は異なる。

### 批判・総合評価・調査の限界と出典

ボーナスは学習と逆の速度で消えたり、無関係な変化を追わせたりする。論文で使われたデータ量、学習中最大か最終平均か、特権情報と復帰操作の有無を揃えて読む必要がある。調査では Taïga らの arXiv 拡張版と会議版の表の一致を確認していない。RND の CNN/RNN 表記、NGU のカーネル定数、Agent57・MEME の方策選択、RLeXplore の sticky の確率など、本文と図・付録の食い違いが残る。一部の表は PDF 原表と未照合で、要旨のみを読んだ研究もある。

## 主要な概念

- [内的報酬による探索の比較](../../../concepts/methods/intrinsic_novelty.md)
- [訪問回数と擬似カウント](../../../concepts/methods/pseudo_counts.md)
- [訪問済み状態へ戻る探索](../../../concepts/methods/return_and_explore.md)
- [Montezuma's Revenge](../../../concepts/subjects/montezuma_revenge.md)
- [LLM の外部知識と探索](../../../concepts/methods/llm_prior_knowledge.md)
- [計測の負荷と比較条件](../../../concepts/measurement/diagnostic_cost.md)
- [報酬尺度と割引率](../../../concepts/methods/reward_scale_discount.md)
