---
title: TempoRL の Q とスキップ価値でのノイズ指定
page_kind: サマリ
material: "[[docs/memo/done/0106_temporl_skip_forward_noise_key_10rfd.md|0106_temporl_skip_forward_noise_key_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
updated: 2026-11-26
---
# TempoRL の Q とスキップ価値でのノイズ指定

## 元の材料

[RFD0106　TempoRL の Q とスキップ価値でのノイズ指定](../../../memo/done/0106_temporl_skip_forward_noise_key_10rfd.md) blob:828b71d1781591a8a4a364752b570b33323447e3

## 要点

- 行動を選ぶ Q の forward と、何 step 続けるかを選ぶ forward のノイズを独立に変更する。
- Actor ごとに共通設定とスキップ専用設定を選べるようにし、既定では同じ設定を使う。
- スキップ長を選ぶときの IQN の τ は、既存のスキップ専用方策から指定する。
- 同じ Qbert の checkpoint で評価条件だけを変え、NoisyNet と τ のどちらが到達率の差に関わるかを比較する。
- 学習中の行動選択も変える実験は、この評価結果を見て決める。

## 内容

### 目的と問題

学習データを集める Train Actor は、`full`・NoisyNet のノイズあり・τ ランダムで動く。学習中の 8 面突破率が評価より低いことと、学習の中盤の遅れにも、この「`full` でノイズが効く」ことが関わっている可能性がある。仕組みの候補は 2 つある。
仮説 1（固定で誤りが長引く）: Q 側のノイズで Greedy と違う行動を選ぶと、`full` ではその行動を、その行動に対して読んだスキップ長 j のあいだ続ける。毎 step 選び直す `single` やスキップなしでは、誤りは 1 step で終わる。
τ にも同じ 2 つの経路がある（Q 側で行動を選ぶときの τ と、skip 側でスキップ長を選ぶときの τ）。
今は skip 側の NN 実行設定を Q 側と別に指定できないので、仮説 1 と 2 を分けて測れない。

### 起草時点の現状

既定の構成で NoisyNet と skip が両方有効な短い検証の構成は、PRD103 の検証資材の 6 構成に無い（skip 有効の GridMaze・LunarLander は NoisyNet 無効）。LunarLander には NoisyNet を有効にする `run.@noisy` がある（LunarLander.txt:91-95）。
qbert のサンプル効率優先の 50M の終了 checkpointを、`[greedy]` の評価 session で各 200 ゲーム評価した 8 面突破率（実験記録、探索ブロック02）。「両方」が ε=0 の Train Actor と同じ選び方である。
中盤の遅れ（ε=0 の評価が 30M 以降に無効より遅れ、学習側の 8 面突破率が 50% を超える 2.5M の窓が 50M まで現れない）は、遮断・合算・重み 0.3 の 3 本で記録されている。分割の学習 Run の経過は記録に無い。
探索ブロック01 では、分割の評価 Run（2 session、計 400 ゲーム）の実時間が 7.5 分だった（起動・読込・終了を含む。同じ記録、探索ブロック01 の表）。これから、1 条件（200 ゲーム）の評価 Run は約 5 分（起動・読込・終了の分を含めて切り上げた）と見積もる。分割の分母（Greedy と両方の差）は、ノイズで 24.5 ポイント（66.0 − 41.5。ブロック02 の「ノイズ」）、τ で 15.5 ポイント（66.0 − 50.5）だった。

### ゴール・非ゴール・仮説と成功の測り方

skip 側のノイズで j が入れ替わる割合などの新しい診断指標。消える案は noisy probe の skip 版を足す案で、本当の制約ではない（足せる）が、ステージ 0 の評価で切り分けられる見込みなので、読みが「決めきれない」になるまで先送りする。

### 実装契約

skip 側の forward の NN 実行オブジェクトを、`shared` ならその Actor の `nn_forward_key` の NN 実行設定で、`dedicated` なら `skip.nn_forward_key` の NN 実行設定で作る。seed の派生（skip 用の名前付き seed）、時計の種別、forward の回数と順序は変えない。
テスト: 設定を手で組むテストは、モード値が無くても C++ 側の既定 `shared` で今と同じに動くので直さない。skip の資源（`SkipActorResources`）を直接組んで Actor を作るテストは、skip 側の NN 実行設定を渡す形に合わせる。
互換の仕組みは残さない（`shared` はモード値として明示されたもので、互換の読み替えではない）。

### 検証と受入

k_Q'・k_S'（分母を r_G − r_TB_indep にした値。独立の抽選の形なので、ノイズの h と並べるときに使う）
モード値が `shared｜dedicated` 以外なら、skip の有効・無効によらず、構築時にキーと値を含むエラーになる。
`skip.nn_forward_key` に宣言されていない名前が書かれていれば、モードと skip の有効・無効によらずエラーになる。
`skip.enabled = true` で、`dedicated` なのにキーの無い Actor があると、構築時に該当する Actor のキー名をすべて含むエラーになる。`skip.enabled = false` なら同じ設定で構築できる。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [Qbert](../../concepts/subjects/qbert.md)
