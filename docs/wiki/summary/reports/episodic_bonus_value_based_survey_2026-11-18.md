---
title: "Survey: エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点"
page_kind: サマリ
material: "[[reports/episodic_bonus_value_based_survey_2026-11-18.md|episodic_bonus_value_based_survey_2026-11-18]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/pseudo_counts.md|訪問回数と擬似カウントによる探索]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/methods/return_and_explore.md|訪問済み状態へ戻る探索]]"
updated: 2026-11-26
---
# Survey: エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点

## 元の材料

[Survey: エピソード内の新しさのボーナスを価値ベース（DQN・Replay）に載せるときの論点](../../../../reports/episodic_bonus_value_based_survey_2026-11-18.md) blob:1f37d213578a0387a7ec48beb4b196868df0bf23

## 要点

- エピソード内の新しさのボーナスは、同じエピソードの中で見たかどうかで報酬を出し、エピソードの始めに記憶を空にする。PPO・IMPALA のような on-policy の手法では、rollout の中でその場で計算してすぐ学習に使える。Replay を使う off-policy の価値ベースでは、行動した時点の記憶で計算した報酬が Replay に残り、学習はあとで別の時点に行う。本冊は、この時間差から生じる論点（記憶の持ち方、報酬の保存と古さ、価値の分け方、多段の収益）を、原典がどう扱ったかで整理する。
- NGU は actor が内的報酬を計算し、観測・行動・合成した報酬・内的報酬などを Replay に入れる。learner は保存された値をそのまま使い、内的報酬を Network の入力にもする。learner で計算し直すとは書いていない。[Badia NGU (2020/02), 付録 A]
- エピソード内の回数や新しさは、そのエピソードのそれまでの文脈が要る。RLeXplore はこのため off-policy ではエピソード全体を引く必要があるとした。[Yuan RLeXplore (2024/05), 付録 C.1・G]
- DEEP は、ボーナスを足す方策はボーナスが 0 に近づくまで偏るが、task の報酬だけで学ぶ task 方策は偏らないとし、報酬が密な環境で性能を落とさず、疎な環境でデータ効率が数倍になったと報告する（SAC と組んだ実験）。[Whitney DEEP (2021/01), Abstract、§4.2〜4.3]
- 方策の分離の副作用: 探索と活用の方策を分けると分布のずれで不安定になりうる。共有した表現は活用側に干渉しうる。[Schäfer DeRL (2021/07), §5.6；Badia NGU (2020/02), §4.2]

## 内容

### 2. NGU 系の実装

NGU・Agent57・MEME は R2D2 系の分散した価値ベースの学習で、エピソード内の新しさを使う代表例である。

エピソードの始めにメモリを完全に空にする（MEME も同じ）。容量は 30,000 件で、満杯なら古いものから上書きする ring buffer、k 近傍の k は 10 である。Agent57 も同じ値を使う。MEME は容量と k を書いていない。[Badia NGU (2020/02), §2、付録 A.2、付録 F.2 表6；Badia Agent57 (2020/03), 付録 G 表3；Kapturowski MEME (2022/09), 付録 G.2・表2]

### 3. NGU 以外のエピソード内ボーナスと off-policy

エピソード内のボーナスを価値ベースで使い、Montezuma's Revenge・Pitfall の結果を報告した例は、NGU 系（R2D2 系）のほかに見つからなかった。

近い報告として、RE3 と Rainbow（Replay 全体の k 近傍）は Montezuma's Revenge を含む Atari 6 本で Rainbow 単独より学習が速かった。[Seo RE3 (2021/02), §4.3、Fig.13・15、付録 D]

### 4. Replay に入れた内的報酬の古さ

NGU は、混ぜた報酬が予測できない形で変わると問題が部分観測になりうるとし、内的報酬を入力に入れ再帰の状態を持つことで対処する。保存値の古さを論じた記述は見つからなかった。Agent57 も同様である。[Badia NGU (2020/02), §3、付録 A；Badia Agent57 (2020/03), 付録 E]

### 6. 批判と懸念

古さを論じない代表例: 価値ベースでエピソード内のボーナスを使う代表の NGU・Agent57・MEME は、保存した内的報酬と埋め込みの古さに触れていない（§2.2・§2.3）。古さを名指しした DEEP は査読未確認で、主な実験は連続制御である。[Whitney DEEP (2021/01)]

Atari での価値ベースの証拠の薄さ: NGU 系以外に、エピソード内のボーナスを価値ベースで難探索の Atari に使った報告は見つからなかった（§3.5）。

### 7. 総合評価

メモリ: NGU 系はエピソードの始めに空にする ring buffer（30,000 件、k=10）を使い、費用はメモリ全体への距離計算である。E3B の行列は rank-1 更新で次元の 2 乗の費用になる。並列の actor・lane での持ち方を明記した原典はない（§2.1・§3.2）。

### 8. 調査の限界

照合: 本文は要約ツールを通して読み、数値は PDF で 1 語ずつ照合していない。

読めなかったもの: R2D2 の原典（OpenReview の確認画面）、NGU・MEME の末尾の一部、SOFE と DeepCS の本文（要約・アブストラクトだけ）、RLeXplore の Table 14 の基盤アルゴリズム。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [訪問回数と擬似カウントによる探索](../../concepts/methods/pseudo_counts.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [訪問済み状態へ戻る探索](../../concepts/methods/return_and_explore.md)
