# Survey: LLM の事前知識で RL の探索と報酬を補助する手法

Date: 2026-10-18
Scope: reports/INDEX.md の優先度10。Read and Reap the Rewards、Motif、ELLMを起点に、ALE Atariでの実証、効果量、LLMの呼び出し頻度と費用、キャッシュ、出来事の検出、外部知識を使う結果の報告、CLIP等のVLM報酬との違いを一次論文と公式コードから調べる。

## Table of Contents

1. [結論と手法の見取り図](#1-結論と手法の見取り図)
2. [起点となる三手法](#2-起点となる三手法)
   - [2.1 Read and Reap the Rewards](#21-read-and-reap-the-rewards)
   - [2.2 Motif](#22-motif)
   - [2.3 ELLM](#23-ellm)
3. [ALE Atari の実証と効果量](#3-ale-atari-の実証と効果量)
   - [3.1 Read and Reward の条件付きの改善](#31-read-and-reward-の条件付きの改善)
   - [3.2 iLLM の内発報酬](#32-illm-の内発報酬)
   - [3.3 LLM-Explorer の探索分布](#33-llm-explorer-の探索分布)
4. [呼び出し頻度と費用](#4-呼び出し頻度と費用)
5. [出来事を検出する方法](#5-出来事を検出する方法)
   - [5.1 物体検出と意味の接続](#51-物体検出と意味の接続)
   - [5.2 RAM 解析と OCAtari](#52-ram-解析と-ocatari)
6. [CLIP 等の VLM 報酬との違い](#6-clip-等の-vlm-報酬との違い)
7. [外部知識を使う結果の報告](#7-外部知識を使う結果の報告)
8. [批判と留保](#8-批判と留保)
9. [総合評価と調査の限界](#9-総合評価と調査の限界)
10. [出典リスト](#10-出典リスト)

表記の約束: 数表は原典の値で、図の目視概算は【目視】と記す。異なるゲーム・予算・評価指標の倍率は合成しない。金額・時間は論文当時の報告であり、現在の見積額ではない。公開コードの確認日は本書の日付。外部知識の区分は本調査による整理で、論文共通の公式分類ではない。

## 1. 結論と手法の見取り図

**LLMの知識を補助報酬や探索へ移す実証はある。ただし、三つの起点論文がすべてAtariで有効だったわけではない。** Read and RewardはAtariの4ゲーム、MotifはNetHack、ELLMはCrafterとHousekeepを評価している。後二者の結果をAtariの効果量として扱えない。[NeurIPS R&R (2023/12), §4](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)、[ICLR Motif (2024), §4](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)、[ICML ELLM (2023/07), §4](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

| 方式 | LLM等から得るもの | RL中に使うもの | 原論文の評価対象 |
|---|---|---|---|
| Read and Reward | 説明書に基づく接触の良し悪し | 検出した接触への補助報酬 | Atari 4本 |
| Motif | 出来事の文章ペアへの選好 | 選好を蒸留した報酬モデル | NetHack |
| ELLM | 状況に合う目標 | 目標と遷移記述の類似度報酬 | Crafter、Housekeep |
| iLLM | 行動生成・履歴要約の事前知識 | それらの予測誤差による内発報酬 | Atari等 |
| LLM-Explorer | 次のepisode群で試す行動の分布 | ε探索の一様分布を置換 | Atari、MuJoCo |

上表の出典: 起点三論文の§3、および [ACML iLLM (2025), §3](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)、[NeurIPS LLM-Explorer (2025/10), §3](https://arxiv.org/html/2505.15293v2)。

## 2. 起点となる三手法

### 2.1 Read and Reap the Rewards

**Read and Rewardは、説明書から接触の意味を読み取り、検出した接触を報酬へ変える。** RoBERTaによるQA抽出の後、Macawが物体へ接触すべきかを判定する。実験の報酬は±5で、対象イベントはagentと物体のbounding boxの交差に限る。戦術や長期目標を自由に生成する仕組みではない。

```text
we ignore all interactions that do not contain the agent.

agentを含まない相互作用はすべて対象外とする。
```
[NeurIPS R&R (2023/12), §3・§4.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)

### 2.2 Motif

**Motifは、LLMをオフラインの評価者として使い、RL中のLLM呼び出しをなくす。** Llama 2の文章ペアへの選好から報酬モデルを学習し、それを固定して使う。NetHackでは組込みのゲームメッセージをcaptionとする。10 seedsの実験で内発報酬のみでもscore直接最適化を上回るが、oracle到達には幻覚を利用する抜け道があり、真の課題達成と分ける必要がある。

```text
instead fully rely on the knowledge acquired offline.

代わりに、オフラインで獲得した知識へ完全に依存する。
```
[ICLR Motif (2024), §3–4・Fig.1・App.A.8.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)

### 2.3 ELLM

**ELLMは状態の文章から目標を提案し、実際の遷移の文章と照合する。** SentenceBERTの類似度を用い、同一episodeで達成済みの目標は再報酬しない。主にシミュレータ由来のcaptionを使う。Crafterの5M steps・5 seedsではepisode当たり実績数が約6、RND等は3未満（§4.1本文の概数）。主実験は行動集合等を変更したCrafterであり、標準CrafterのscoreやAtariの成績ではない。

```text
ELLM requires states and transition captions.

ELLMは状態と遷移のcaptionを必要とする。
```
[ICML ELLM (2023/07), §3–5・Fig.4・App.C](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

## 3. ALE Atari の実証と効果量

### 3.1 Read and Reward の条件付きの改善

**改善の数値は、報酬の遅延と物体情報の条件を添えて読む必要がある。** Table 1のA2Cは下表。ただしTennis、Ms. Pac-Man、Breakoutは元の報酬をepisode終了時へ遅延させ、物体位置はRAM由来で与えている。[NeurIPS R&R (2023/12), §4.2–4.3・Table 1](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)

| ゲーム（原表の表記） | A2C | A2C＋R&R |
|---|---:|---:|
| Tennis | −23 | −5 |
| Pacman | 452 | 580 |
| Breakout | 2 | 14 |

**この表を一貫した改善の確証にはできない。** 別の3 seeds表である公刊PDF Table 2は、Breakout A2C＋R&Rが−4.5±3.0、baselineが2.1±0.3で、captionの一貫改善という記述と整合しない。arXiv v4では一部のPPO欄の符号も公刊PDFと異なる。誤植と推測して値を修正しない。[NeurIPS R&R (2023/12), Table 2](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)、[arXiv R&R (2024/07), Table 2](https://arxiv.org/html/2302.04449v4)

112倍の速度改善も、本文§4.8は遅延なし、Table 4 captionは遅延ありと条件が一致しない。標準ALEで同一最終性能に112倍速く達したとの確定値には使わない。Skiingの少ないframesの主張も、Agent57と同一性能の比較ではない。[NeurIPS R&R (2023/12), §4.6・§4.8・Table 4](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)

### 3.2 iLLM の内発報酬

**Atariの探索補助は、文章イベントを人手で列挙する方式だけではない。** iLLMは観測と直前行動をHopfield moduleで事前学習済みtoken embeddingへ写像し、行動生成・履歴要約を予測する誤差を内発報酬にする。LLM推奨行動の直接実行ではない。下表は100M training steps、10 seedsの平均±標準偏差であり、論文のstepsをALE生framesへ無条件に読み替えない。

```text
Instead of directly performing the LLM-recommended action

LLMが推奨した行動を直接実行する代わりに
```
[ACML iLLM (2025), §3・§4.3・Table 3](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)

| ゲーム | RND | iLLM(hop) |
|---|---:|---:|
| Montezuma’s Revenge | 456±55 | 2,632±277 |
| PrivateEye | 598±110 | 4,422±376 |
| Gravitar | 192±34 | 4,044±559 |
| Pitfall | −11±3 | 125±24 |
| Seaquest | 2,612±315 | 18,851±2,930 |

Seaquestでは別baselineのAPTが19,989±2,873でiLLMを上回る。テキスト観測が必要なiLLM(obs)とELLMはこのAtari比較にはない。KungFuMasterとQbertもこの表にはない。[ACML iLLM (2025), Table 3](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)

### 3.3 LLM-Explorer の探索分布

**LLM-Explorerは報酬を生成せず、DQNの探索時に選ぶ行動の分布を変える。** LLMに渡すのは課題の説明、直近episodeの行動履歴、総報酬で、現在の画像・状態は渡さない。

```text
we only sample the actions and rewards of the agent and exclude the states.

エージェントの行動と報酬だけを取り出し、状態は除外する。
```
[NeurIPS LLM-Explorer (2025/10), §3.2–3.4](https://arxiv.org/html/2505.15293v2)

15/26ゲームをbaselineの安定学習に基づいて選び、100k–500k steps・3 seedsで比較している。HNSは(score−random)/(human−random)を百分率で表したhuman-normalized scoreである。平均HNSは14.27%→19.59%、中央値は3.18%→3.62%。論文の+37.27%は平均HNSの相対増加であり、全ゲームのscoreが37%増えた意味ではない。主設定のQbertは306.07→301.97と低下する。Appendix JのKL制限を加えた別設定では316.92に改善するが、8面到達を評価した結果ではない。[NeurIPS LLM-Explorer (2025/10), §4.1・Table 7・Table 13](https://arxiv.org/html/2505.15293v2)

## 4. 呼び出し頻度と費用

**LLMへ論理的に問い合わせる頻度、APIのcache miss回数、RL中の小さなモデルの推論回数は別々に数える。** ELLMの実装は毎step目標を取得する一方、同じ問い合わせなら保存済み応答を返す。cache keyにはモデル、prompt、生成設定が含まれ、文章embeddingも再利用する。[ELLM code (閲覧2026/10), GPTLanguageModel.predict_options](https://github.com/yuqingd/ellm/blob/main/text_crafter/lm.py)、[CrafterLMGoalWrapper.step / text_reward](https://github.com/yuqingd/ellm/blob/main/text_crafter/text_crafter/goal_wrapper.py)

| 手法 | 頻度・再利用 | 原典で確認できる費用と限界 |
|---|---|---|
| Read and Reward | 説明書QA・物体別の判定と、ゲーム中の接触検出を分ける構成 | 実呼出回数・cache運用・金額の定量報告は未確認。「最初の1回だけ」とは断定しない |
| Motif | オフライン注釈→報酬モデル蒸留。RL中のLLM呼出なし | 500kペアの70B注釈はV100×8構成で最大15 GPU-daysと記載。報酬モデル学習約1h、RL約24h（各V100×1） |
| ELLM | 毎stepの目標取得に応答cache。Run間でも共有 | cacheなし5M stepsならAPI待ち27h・数千ドルという当時の試算。共有cacheが育つとほぼ問い合わせ不要と報告 |
| iLLM | 各stepでLLMを使い、履歴contextをcache | backboneはTransfo-XL 280M。context cacheはLLM推論の除去ではない。今回ドル額は確認できず |
| LLM-Explorer | 主設定は毎episode、履歴要約→分布生成の2段階。100 actionsを抽出 | GPT-4o mini、500k-step Qbertは$1.32・9.95h。全20 Atari＋MuJoCo平均$1.30はAtari単独平均ではない |

出典: [NeurIPS R&R (2023/12), §3](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)、[ICLR Motif (2024), App.A.8.4](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)、[ICML ELLM (2023/07), App.O](https://proceedings.mlr.press/v202/du23f/du23f.pdf)、[ACML iLLM (2025), §3–4](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)、[NeurIPS LLM-Explorer (2025/10), §4.1–4.3・Table 1](https://arxiv.org/html/2505.15293v2)。

ELLMの公開コードの価格定数と論文の当時価格は一致せず、コードのbudgetも文字数に基づく推定である。現在の再現費用はこの値から確定しない。[ELLM code (閲覧2026/10), prices / budget_used](https://github.com/yuqingd/ellm/blob/main/text_crafter/lm.py)

## 5. 出来事を検出する方法

### 5.1 物体検出と意味の接続

**文章を理解できても、ゲーム内で何が起きたかを正しく検出できるとは限らない。** Read and RewardのSkiingはSPACEの物体候補、CLIPのクラス対応、複数物体追跡を組み合わせる。他3本は約15枚の手作業ラベルからRAM→座標の回帰を作る。重なった物体の検出失敗も報告されている。[NeurIPS R&R (2023/12), §4.3・Fig.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)

MotifはNetHackが出すメッセージ、ELLMは状態・遷移captionを利用する。ELLMのCrafter公式コードではaction_successによるgateがあり、単に画像にSentenceBERTをかける実装ではない。[ICLR Motif (2024), §3.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)、[ELLM code (閲覧2026/10), CrafterLMGoalWrapper.step](https://github.com/yuqingd/ellm/blob/main/text_crafter/text_crafter/goal_wrapper.py)

### 5.2 RAM 解析と OCAtari

**OCAtariは物体抽出を提供するが、報酬にすべき出来事の意味までは自動的に決めない。** RAM ExtractionはRAMと物体属性の対応を解析し、Vision Extractionは色・位置等の知識を使う。RAMでは点滅・遮蔽中の物体も追えるため、純画像条件と観測できる情報が異なる。論文はRAM抽出の平均約50倍の速度を報告するが、これはLLMやRL全体の高速化率ではない。

```text
by introducing OCAtari, that performs resource-efficient extractions of the object-centric states

物体中心の状態を効率よく抽出するOCAtariを導入することで
```
[RLJ OCAtari (2024), Abstract・§2.2・App.K](https://arxiv.org/html/2306.08649)

公開コードの版99c874675df6b76a33a80b57776c123fbcd051afでは、Qbertのcube色・player・敵、KungFuMasterの敵種別・player/敵のhealth barをRAMから構成している。これは対象ゲームの物体情報が実装済みという証拠であり、Qbertの8面達成やボス戦の報酬判定器が検証済みという証拠ではない。後半は本調査における証拠範囲の区別である。[OCAtari code (2025/06), qbert.py](https://github.com/k4ntz/OC_Atari/blob/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram/qbert.py)、[kungfumaster.py](https://github.com/k4ntz/OC_Atari/blob/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram/kungfumaster.py)

## 6. CLIP 等の VLM 報酬との違い

**CLIP報酬は、画像が文章の目標へどれほど合うかを測る。LLMの文章上の判断と、どこで知覚を行うかが異なる。** VLM-RMはCLIPで画像と目標文を埋め込み、baseline promptとの区別も使う。CartPole、MountainCar、Humanoidで評価し、Atariの実証ではない。標準MountainCarの見た目では学習できず、texture等の変更が必要だった例もある。

```text
Without the custom textures, we are not able to successfully train an agent

独自のテクスチャなしでは、エージェントを成功裏に学習させられない。
```
[ICLR VLM-RM (2024), §3–4.1](https://arxiv.org/pdf/2310.12921)

| 比較軸 | 文章知識からの補助 | CLIP型報酬 | VLMがコードを生成する報酬 |
|---|---|---|---|
| 知覚との接続 | 別の検出器・caption・履歴を入力 | 画像をVLMへ直接入力 | 生成した画像処理・判定コード |
| 報酬の形 | 接触の符号、選好モデル、目標達成等 | 画像と目標の埋め込み類似度等 | subtask完了判定 |
| 各stepの大モデル | 設計次第 | 通常は画像encoderの推論 | 生成後はコード実行へ移せる |
| 注意点 | captionの誤り、LLMの好みと目的のずれ | 見た目・空間関係への認識限界 | 生成コードの誤りと検証データへの依存 |

上表は起点三論文とVLM-RM、VLM-CaRの方式からの整理。**VLM-CaRはCLIP類似度の別名ではない。** 開始・目標画像から物体検出と完了判定のコードを生成し、expert/random trajectoriesで検証する。評価対象はMiniGrid、Pandas-Gym、CLIPortで、コードの組立てや進行状態には人手も入る。

```text
we only assemble these scripts together into a single class

これらのスクリプトを一つのクラスに組み立てる作業は人間が行う。
```
[ICML VLM-CaR (2024/07), §4–5・App.D](https://arxiv.org/pdf/2402.04764)

## 7. 外部知識を使う結果の報告

**原論文は外部情報や変更した条件を記しているが、手法間で統一された別ランキングを使っているわけではない。** R&RはmanualとRAM・delayed rewardを、Motifは事前データとLLMへのpromptを、ELLMはcaptionと環境変更を明記する。比較するときはこれらを読み落とさない。[NeurIPS R&R (2023/12), §4](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)、[ICLR Motif (2024), §3–4・App.A.6](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)、[ICML ELLM (2023/07), App.C・I](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

ALEの一般能力評価はゲーム固有情報を抑えることを重視し、設定差が比較を歪める問題を指摘する。

```text
without game-specific information

ゲーム固有の情報なしに
```
[JAIR ALE (2018), §1・§3.1](https://arxiv.org/html/1709.06009)

本調査では、結果を読むために次の情報を別欄で保持する。これは新しいベンチマーク規則の引用ではなく、上記の条件差を混ぜないための整理である。

| 区分 | 記録する情報 |
|---|---|
| 外部の知識 | model/version、事前学習、manual/Wiki、課題説明、prompt全文 |
| 追加の観測・教師 | RAM、物体ラベル、captionerの学習データ、expert/random動画、事前収集量 |
| 環境・学習 | 元の報酬と補助報酬、遅延化、action集合、事前学習とオンライン予算 |
| 評価 | 補助報酬を混ぜないゲームscore、seed数、評価時点、sticky actions、frame skip、終端 |
| 費用 | データ収集、注釈、蒸留、RLの時間・GPU、API token/call、cold/warm cache |

ALEの現行公式文書もsticky actionsとminimal action setの明記を求めている。報酬や知識の条件を分けても、これらの差は残る。[ALE docs (閲覧2026/10), Stochasticity / Minimal Action Set](https://ale.farama.org/env-spec/)

## 8. 批判と留保

**良い文章上の判断を集めても、最終目標の達成は保証されない。** Motifのoracleでは幻覚状態で敵をoracleと認識する報酬ハックを確認し、oracle-sober課題を別途調べている。ELLMでも、目標の達成を繰り返し報酬すると簡単な行動の反復へ偏る。[ICLR Motif (2024), §4.2・App.A.8.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)、[ICML ELLM (2023/07), App.L](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

R&Rの公開先は確認時READMEのみで、実装からcacheや呼出頻度を検証できなかった。原論文の数値・条件の不整合と合わせ、再現可能性の留保が大きい。[RnR official code (閲覧2026/10), repository / README](https://github.com/Holmeswww/RnR)

LLM-Explorerの改善は選別した15ゲームと短い予算の結果である。iLLMの5ゲーム・100M stepsと、R&Rの遅延報酬条件を同じ改善率ランキングに並べない。これは各実験の母集団と条件を守るための本調査の扱いである。[NeurIPS LLM-Explorer (2025/10), §4.1](https://arxiv.org/html/2505.15293v2)、[ACML iLLM (2025), §4.3](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)、[NeurIPS R&R (2023/12), §4.2](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)

## 9. 総合評価と調査の限界

本調査の結論は三つである。

1. Atariでの正の実証はあるが、ゲーム・報酬・観測の条件付き。Motif/ELLMの原結果をAtariへ移して評価しない（§2–3）。
2. stepごとの大モデル呼出を避ける実証済みの設計は、オフライン蒸留、応答cache、episode単位の更新、生成コードへの移行。それぞれ事前計算・cache依存・適用範囲が異なる（§4・§6）。
3. 物体やcaptionの正しさと、望む出来事の達成を分けて検証する必要がある。報酬モデルの高い値だけでは本来の成功と一致しない例がある（§5・§8）。

調査上の限界:

- 検索は指定三論文とその近傍、Atari＋LLM exploration/reward、CLIP/VLM報酬の一次資料を中心に行った。2026年10月までの全研究を網羅したsystematic reviewではない。
- 今回確認した論文に、KungFuMasterのボス攻略やQbertの8面到達を直接改善した実証は見つからなかった。Qbertの全episode scoreや物体抽出の対応は、その代替証拠にしない（§3.3・§5.2）。
- 実装の再実行、ゲームイベントの検出精度測定、現行API費用の測定は行っていない。文献の費用と現在の運用費用を分けた。
- 被引用数はSemantic Scholar公開APIがHTTP 429で取得できず、統一DBでの確認は未達。R&RのみAriel大学ページのScopus表示16件を確認した。検索サービス経由の表示であり、Scopusを直接照会した当日値とは保証しない。その他の論文は取得不可とする。[Ariel research portal (閲覧2026/10)](https://cris.ariel.ac.il/en/publications/read-and-reap-the-rewards-learning-to-play-atari-with-the-help-of-2/)
- 個人の役割は下記研究の著者・共同筆頭・連絡著者として記す。所属は論文掲載時。個別の職位・現在所属は確認していない。
- キューの関連欄はAtari campaignで、特定の関連PRDは指定されていない。PRDへの反映事項として確定したものはない。PRD・実験記録は変更していない。

## 10. 出典リスト

論文は査読付きのproceedingsを優先し、公式コード・公式文書は実装や運用の確認として分ける。被引用数は上記の例外を除き取得不可。

[NeurIPS R&R, 2023/12] Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom M. Mitchell. “Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals.” NeurIPS 2023. Wu/Fan/Liang/Mitchell: CMU、Azaria: Ariel University、Li: CMU/Microsoft Research。全員研究著者。[公刊PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/034d7bfeace2a9a258648b16fc626298-Paper-Conference.pdf)。初出2023/02、比較版は[arXiv v4、2024/07](https://arxiv.org/html/2302.04449v4)。

[ICLR Motif, 2024] Martin Klissarov, Pierluca D’Oro, Shagun Sodhani, Roberta Raileanu, Pierre-Luc Bacon, Pascal Vincent, Amy Zhang, Mikael Henaff. “Motif: Intrinsic Motivation from Artificial Intelligence Feedback.” ICLR 2024. Klissarov/D’Oro共同筆頭。所属: Mila、FAIR at Meta、UT Austin、Université de Montréal、McGill（著者ごとの対応はp.1）。[公刊PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8de97fa77a09258b22880c230a83445-Paper-Conference.pdf)。初出2023/09。

[ICML ELLM, 2023/07] Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, Jacob Andreas. “Guiding Pretraining in Reinforcement Learning with Large Language Models.” ICML 2023, PMLR 202:8657–8677. Du/Watkins共同筆頭。所属: UC Berkeley EECS、University of Washington、MIT CSAIL、Inria Flowers（対応はp.1）。[公刊PDF](https://proceedings.mlr.press/v202/du23f/du23f.pdf)。

[ACML iLLM, 2025] Nicolas Bougie, Narimasa Watanabe. “Exploring Beyond Curiosity Rewards: Language-Driven Exploration in RL.” PMLR 260:127–142. 両著者Woven by Toyota, Tokyo。公式書誌は2025、PDFの会議表記はACML 2024。[正式ページ](https://proceedings.mlr.press/v260/bougie25a.html)、[PDF](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)。

[NeurIPS LLM-Explorer, 2025/10] Qianyue Hao, Yiwen Song, Qingmin Liao, Jian Yuan, Yong Li. “LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models.” NeurIPS 2025. Tsinghua University, Electronic Engineering / BNRist。Hao/Song共同筆頭、Li連絡著者。初出2025/05、参照v2は2025/10/23。[参照本文](https://arxiv.org/html/2505.15293v2)。

[ICLR VLM-RM, 2024] Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, David Lindner. “Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning.” ICLR 2024. 所属は順にFAR AI、Vertebra、ETH AI Center、Anthropic、ETH Zurich。初出2023/10。[PDF](https://arxiv.org/pdf/2310.12921)。

[ICML VLM-CaR, 2024/07] David Venuto, Mohammad Sami Nur Islam, Martin Klissarov, Doina Precup, Sherry Yang, Ankit Anand. “Code as Reward: Empowering Reinforcement Learning with VLMs.” ICML 2024, PMLR 235:49368–49387. 所属: Mila/McGill、Google DeepMind、UC Berkeley（対応はp.1）。[正式ページ](https://proceedings.mlr.press/v235/venuto24a.html)、[PDF](https://arxiv.org/pdf/2402.04764)。

[RLJ OCAtari, 2024] Quentin Delfosse, Jannis Blüml, Bjarne Gregori, Sebastian Sztwiertnia, Kristian Kersting. “OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments.” Reinforcement Learning Journal 1:400–449. 所属: TU Darmstadt / Hessian.AI / DFKI（対応は本文冒頭）。初出2023/06。[本文](https://arxiv.org/html/2306.08649)、[公式コード](https://github.com/k4ntz/OC_Atari)。

[JAIR ALE, 2018] Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, Michael Bowling. “Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.” JAIR 61:523–562. 所属: University of Alberta、Google Brain、Franklin & Marshall、DeepMind、Microsoft Research（対応は本文冒頭）。[本文](https://arxiv.org/html/1709.06009)。初出2017/09。

[Official code/docs, 閲覧2026/10] 著者・管理団体による一次資料。[RnR](https://github.com/Holmeswww/RnR)、[Motif](https://github.com/facebookresearch/motif)、[ELLM lm.py](https://github.com/yuqingd/ellm/blob/main/text_crafter/lm.py)、[ELLM goal_wrapper.py](https://github.com/yuqingd/ellm/blob/main/text_crafter/text_crafter/goal_wrapper.py)、[OCAtari参照版](https://github.com/k4ntz/OC_Atari/tree/99c874675df6b76a33a80b57776c123fbcd051af)、[ALE Environment Specifications](https://ale.farama.org/env-spec/)。

[Ariel research portal, 閲覧2026/10] Ariel University. R&R研究業績レコード。被引用数表示の確認に限って参照。[研究業績ページ](https://cris.ariel.ac.il/en/publications/read-and-reap-the-rewards-learning-to-play-atari-with-the-help-of-2/)。
