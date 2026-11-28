---
title: "Survey: LLM の事前知識で RL の探索と報酬を補助する手法"
page_kind: サマリ
material: "[[reports/survey/llm_prior_knowledge_rl_survey_2026-10-18.md|llm_prior_knowledge_rl_survey_2026-10-18]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/llm_prior_knowledge.md|LLMの外部知識と探索]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
updated: 2026-11-28
---
# Survey: LLM の事前知識で RL の探索と報酬を補助する手法

## 元の材料

[Survey: LLM の事前知識で RL の探索と報酬を補助する手法](../../../../../reports/survey/llm_prior_knowledge_rl_survey_2026-10-18.md) blob:4357d167329e14c7d1a4a4738d72e6732e6346d9

## 要点

- LLMの知識を補助報酬や探索へ移す実証はある。ただし、三つの起点論文がすべてAtariで有効だったわけではない。
- Atariの探索補助は、文章イベントを人手で列挙する方式だけではない。
- MotifはNetHackが出すメッセージ、ELLMは状態・遷移captionを利用する。ELLMのCrafter公式コードではaction_successによるgateがあり、単に画像にSentenceBERTをかける実装ではない。ICLR Motif (2024), §3.1、ELLM code (閲覧2026/10), CrafterLMGoalWrapper.step
- ALEの一般能力評価はゲーム固有情報を抑えることを重視し、設定差が比較を歪める問題を指摘する。
- Atariでの正の実証はあるが、ゲーム・報酬・観測の条件付き。Motif/ELLMの原結果をAtariへ移して評価しない（§2–3）。

## 内容

### 2. 起点となる三手法

Read and Rewardは、説明書から接触の意味を読み取り、検出した接触を報酬へ変える。 RoBERTaによるQA抽出の後、Macawが物体へ接触すべきかを判定する。実験の報酬は±5で、対象イベントはagentと物体のbounding boxの交差に限る。戦術や長期目標を自由に生成する仕組みではない。

Motifは、LLMをオフラインの評価者として使い、RL中のLLM呼び出しをなくす。 Llama 2の文章ペアへの選好から報酬モデルを学習し、それを固定して使う。NetHackでは組込みのゲームメッセージをcaptionとする。10 seedsの実験で内発報酬のみでもscore直接最適化を上回るが、oracle到達には幻覚を利用する抜け道があり、真の課題達成と分ける必要がある。

### 3. ALE Atari の実証と効果量

LLM-Explorerは報酬を生成せず、DQNの探索時に選ぶ行動の分布を変える。 LLMに渡すのは課題の説明、直近episodeの行動履歴、総報酬で、現在の画像・状態は渡さない。

### 4. 呼び出し頻度と費用

iLLM行の訂正根拠はSupplement §B、§D.7。同資料§D.8にはAtariのモデル別比較もあり、今回の追加確認を探索の別冊へ記録した。

### 5. 出来事を検出する方法

OCAtariは物体抽出を提供するが、報酬にすべき出来事の意味までは自動的に決めない。 RAM ExtractionはRAMと物体属性の対応を解析し、Vision Extractionは色・位置等の知識を使う。RAMでは点滅・遮蔽中の物体も追えるため、純画像条件と観測できる情報が異なる。論文はRAM抽出の平均約50倍の速度を報告するが、これはLLMやRL全体の高速化率ではない。

### 6. CLIP 等の VLM 報酬との違い

CLIP報酬は、画像が文章の目標へどれほど合うかを測る。LLMの文章上の判断と、どこで知覚を行うかが異なる。 VLM-RMはCLIPで画像と目標文を埋め込み、baseline promptとの区別も使う。CartPole、MountainCar、Humanoidで評価し、Atariの実証ではない。標準MountainCarの見た目では学習できず、texture等の変更が必要だった例もある。

上表は起点三論文とVLM-RM、VLM-CaRの方式からの整理。VLM-CaRはCLIP類似度の別名ではない。 開始・目標画像から物体検出と完了判定のコードを生成し、expert/random trajectoriesで検証する。評価対象はMiniGrid、Pandas-Gym、CLIPortで、コードの組立てや進行状態には人手も入る。

### 7. 外部知識を使う結果の報告

本調査では、結果を読むために次の情報を別欄で保持する。これは新しいベンチマーク規則の引用ではなく、上記の条件差を混ぜないための整理である。

### 9. 総合評価と調査の限界

物体やcaptionの正しさと、望む出来事の達成を分けて検証する必要がある。報酬モデルの高い値だけでは本来の成功と一致しない例がある（§5・§8）。

## 主要な概念

- [LLMの外部知識と探索](../../../concepts/methods/llm_prior_knowledge.md)
- [内的報酬による探索の比較](../../../concepts/methods/intrinsic_novelty.md)
- [Atari横断](../../../concepts/subjects/atari.md)
- [kung_fu_master（KFM）](../../../concepts/subjects/kung_fu_master.md)
