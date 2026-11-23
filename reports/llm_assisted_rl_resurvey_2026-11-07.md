# LLMを活用したRLの再サーベイ

調査日: 2026-11-07（JST）  
対象: 外部環境で学ぶRLエージェントをLLM・VLMで補助する研究。2025〜2026年を中心に、比較の起点となる2023〜2024年の研究を含む。
前回: [LLMの事前知識でRLの探索と報酬を補助する手法](llm_prior_knowledge_rl_survey_2026-10-18.md)  
状態: 本冊は主要論文21本。追加調査を含む11冊で重複を除き47論文を確認。全文未確認の資料は個別に明記。サブエージェントは使用していない。

全体の入口: [47論文の索引・関心別の読み順・比較費用の考え方](llm_rl_reading_guide_2026-11-07.md)

## 目次

1. [1. 対象と読み方](#1-対象と読み方)
2. [2. 報酬と観測をコードとして生成する](#2-報酬と観測をコードとして生成する)
   - [2.1 Eureka](#21-eureka)
   - [2.2 R*](#22-r)
   - [2.3 PoRSE](#23-porse)
   - [2.4 LIMEN](#24-limen)
   - [2.5 EvoNav](#25-evonav)
3. [3. 映像から報酬を学ぶ](#3-映像から報酬を学ぶ)
   - [3.1 RL-VLM-F](#31-rl-vlm-f)
   - [3.2 RoboReward](#32-roboreward)
   - [3.3 Large Reward Models](#33-large-reward-models)
   - [3.4 Dopamine-RewardとDopamine-RL](#34-dopamine-rewardとdopamine-rl)
   - [3.5 Robometer](#35-robometer)
4. [4. 学習する課題の順序を作る](#4-学習する課題の順序を作る)
   - [4.1 LEACL](#41-leacl)
5. [5. 探索と行動を補助する](#5-探索と行動を補助する)
   - [5.1 LLM-Explorer](#51-llm-explorer)
   - [5.2 Exploration with Foundation Models](#52-exploration-with-foundation-models)
   - [5.3 TextAtari](#53-textatari)
   - [5.4 Generative Code Optimization](#54-generative-code-optimization)
   - [5.5 ULPS](#55-ulps)
6. [6. 学習則そのものを探索する](#6-学習則そのものを探索する)
   - [6.1 Evolutionary Discovery of RL Algorithms](#61-evolutionary-discovery-of-rl-algorithms)
7. [7. 比較条件と失敗を調べる](#7-比較条件と失敗を調べる)
   - [7.1 BIRD](#71-bird)
   - [7.2 Diagnostic-Driven Refinement](#72-diagnostic-driven-refinement)
   - [7.3 Policy-Invariant Reward Shaping from LLM Feedback](#73-policy-invariant-reward-shaping-from-llm-feedback)
   - [7.4 CriticHack](#74-critichack)
8. [8. 費用とローカル実行の証拠](#8-費用とローカル実行の証拠)
   - [8.1 費用の単位を揃える](#81-費用の単位を揃える)
   - [8.2 小型モデルで何が確かめられているか](#82-小型モデルで何が確かめられているか)
9. [9. 到達点と調査の限界](#9-到達点と調査の限界)
10. [10. 書誌と探索記録](#10-書誌と探索記録)
    - [10.1 著者と所属](#101-著者と所属)
    - [10.2 被引用数](#102-被引用数)
    - [10.3 一次資料一覧](#103-一次資料一覧)
    - [10.4 探索と確認の範囲](#104-探索と確認の範囲)

## 1. 対象と読み方

今回の中心は「LLMに毎回行動を決めさせる」方式だけではない。報酬のプログラム、観測の変換、課題の難易度、学習の更新則をLLMが作り、通常のRLが環境との相互作用を担う方式を含める。次表は本報告の整理であり、各論文が同一の分類体系を提唱したという意味ではない。

| LLMが作るもの | RL中に残る処理 | 代表資料 |
|---|---|---|
| 報酬・観測のコード | 生成済みの関数を実行。外側の探索で再生成 | [Eureka](https://arxiv.org/abs/2310.12931)、[LIMEN](https://arxiv.org/abs/2605.03408) |
| 画像間の好みや進捗のラベル | 学習した報酬モデルで採点 | [RL-VLM-F](https://proceedings.mlr.press/v235/wang24bn.html)、[RoboReward](https://arxiv.org/abs/2601.00675) |
| 課題の分解・難易度の指定 | 自動カリキュラムとPPO | [LEACL](https://arxiv.org/abs/2607.23515) |
| 探索用の行動分布 | RL方策と探索分布を併用し、定期的にLLMへ問い合わせ | [LLM-Explorer](https://arxiv.org/abs/2505.15293) |
| 行動のプログラム・学習の更新則 | コードとして実行 | [Generative Code Optimization](https://arxiv.org/abs/2508.19506)、[Evolutionary Discovery](https://arxiv.org/abs/2603.28416) |

評価で特に注意するのは、LLM呼び出しの削減と、RLを何度も学習する費用の削減が別であることだ。BIRDは候補の生成から方策の学習・評価までを共通化して比較する。

```text
under matched feedback conditions and policy-training budgets

フィードバックの条件と方策学習の予算を揃えた条件で
```
[arXiv / BIRD (2026/10), Abstract](https://arxiv.org/html/2610.04364v1)

以下の数値は著者の報告であり、この調査で再現実験を実行した値ではない。LLM自体の数学・コード能力をRLHFやRLVRで高める研究は対象外とする。視覚モデルはVLMと明記する。

## 2. 報酬と観測をコードとして生成する

### 2.1 Eureka

Eurekaは、環境のコードと課題の説明から報酬コードを作り、RLの結果を次の生成へ返す起点となる研究である。GPT-4などによる進化的探索を使う。29環境・10種類のロボット形態で、人手報酬を上回った課題の割合を83%と報告するが、これはAtariの成績ではない。報酬を作り終えた後の計算と、その報酬を探すための学習試行を区別して読む。[arXiv / Eureka (2024/04), Abstract](https://arxiv.org/abs/2310.12931)

```text
perform evolutionary optimization over reward code

報酬コードに対して進化的最適化を行う
```
[arXiv / Eureka (2024/04), Abstract](https://arxiv.org/abs/2310.12931)

### 2.2 R*

R*は報酬の構造と係数を分けて探索する。LLMによる変更と部品単位の交叉で構造を作り、LLMが生成した複数の評価関数の投票で軌跡の比較ラベルを得て、係数を選好学習で調整する。PPOによるロボット制御の検証であり、評価関数の投票そのものが人間の目的を保証するわけではない。係数の調整まで毎回文章生成に任せる設計とは異なる。[PMLR / R* (2025/07), §3–4](https://proceedings.mlr.press/v267/li25v.html)

```text
reward structure evolution and parameter alignment optimization

報酬構造の進化と、パラメータの整合を図る最適化
```
[PMLR / R* (2025/07), Abstract](https://proceedings.mlr.press/v267/li25v.html)

### 2.3 PoRSE

PoRSEは目標に向かう報酬に加え、探索のために状態をまとめる写像もLLMに作らせる。写像先での訪問回数を使い、課題に関係する状態を探索する。方策を引き継ぎながら候補を評価・選別するため、変更のたびに最初から学び直す方式とは異なる。24のロボット操作・移動課題で検証されている。意味を持つ状態の入力があり、画素だけのAtariに同じ結果があるという報告ではない。[ICLR / PoRSE (2026), §3–4](https://openreview.net/pdf?id=1vXMfIYFZp)

```text
without requiring policies to train from scratch

方策を最初から学習し直すことを必要とせず
```
[ICLR / PoRSE (2026), Abstract](https://openreview.net/pdf?id=1vXMfIYFZp)

### 2.4 LIMEN

LIMENはシミュレータの状態から観測と報酬の両方を生成する。XLand-MiniGridの3課題、Panda、Go1の計5課題で、どちらか片方だけの探索と比較する。入力は意味の分かるシミュレータ状態であり、軌跡の成功を採点する基準も外から与える。画像を見ただけで課題の仕様まで自律発見した実験ではない。生成した候補は短い学習でふるいに掛ける。[arXiv / LIMEN (2026/05), §2, §4–5](https://arxiv.org/html/2605.03408v1)

```text
both observation mappings and reward functions must be generated

観測の写像と報酬関数の両方を生成する必要がある
```
[arXiv / LIMEN (2026/05), Abstract](https://arxiv.org/abs/2605.03408)

### 2.5 EvoNav

EvoNavは人が動く中でのロボット移動を対象とし、報酬候補を解析的な指標、軽量な軌跡生成、本格的な方策学習の順で選別する。初期の安い評価が最終評価をどこまで予測するかも検討する。主眼はLLMのトークン節約より、全候補を最後まで学習しないことにある。大きな公開モデルも比較されるが、小型モデルを個人用GPUで動かす測定ではない。[arXiv / EvoNav (2026/05), §4.3, Appendix C.7](https://arxiv.org/html/2605.11859v1)

```text
finally, to full policy training

最後に、方策の本格的な学習へ進む
```
[arXiv / EvoNav (2026/05), Abstract](https://arxiv.org/abs/2605.11859)

## 3. 映像から報酬を学ぶ

### 3.1 RL-VLM-F

RL-VLM-FはVLMに画像の組を比較させ、その好みから別の報酬関数を学ぶ。毎回VLMに数値を直接出させる方式の不安定さを避けるための設計で、課題の文章と視覚観測を用いる。古典制御、剛体・関節物体・変形物体の操作で評価する。教師となるVLMと、RLが使う報酬モデルの役割を分けた例として重要である。[PMLR / RL-VLM-F (2024/07), Abstract](https://proceedings.mlr.press/v235/wang24bn.html)

```text
learn a reward function from the preference labels

好みを表すラベルから報酬関数を学習する
```
[PMLR / RL-VLM-F (2024/07), Abstract](https://proceedings.mlr.press/v235/wang24bn.html)

### 3.2 RoboReward

RoboRewardは成功例に偏るロボット動画を、指示の変更と途中での切り取りにより失敗・部分成功の例へ広げる。4B・8Bの報酬モデルと評価用データを公開し、8Bを実ロボットRLでも検証する。高価なラベル生成を事前に行い、繰り返しの採点は学習済みモデルへ移す方式である。評価対象は短いロボット課題であり、公開モデルというだけで任意のゲームの報酬に使えるとは示していない。[arXiv / RoboReward (2026/01), §4–5](https://arxiv.org/html/2601.00675v2)

```text
no model excels across tasks

すべての課題で優れるモデルはない
```
[arXiv / RoboReward (2026/01), Abstract](https://arxiv.org/abs/2601.00675)

### 3.3 Large Reward Models

LRMはQwen3-VL-8B-Instructを、進捗・完了・時間順序の比較という別々の報酬モデルに調整する。模倣学習で初期化した方策をManiSkill3でPPO改善する実験と、実機4課題での進捗による重み付き行動模倣を分けている。後者を実機のオンラインRL成功例として数えない。9月のv3では問い合わせ間隔と遅延も検討されている。[arXiv / LRM (2026/09), §III–IV](https://arxiv.org/html/2603.16065v3)

```text
Instead of querying a VLM reward online during robot control

ロボット制御中にVLM報酬へオンラインで問い合わせる代わりに
```
[arXiv / LRM (2026/09), §IV-D](https://arxiv.org/html/2603.16065v3)

### 3.4 Dopamine-RewardとDopamine-RL

この研究は複数視点から課題の進捗を学ぶ報酬モデルと、最適方策を変えない形で報酬を加える手法を組み合わせる。事前のデータは3,400時間超で、新課題への適応には専門家の軌跡を1本使う。150回のオンライン試行で成功率95%という例は、この事前学習と適応を経た結果であり、最初から1時間で全系を作った意味ではない。[CVF / Dopamine-Reward (2026/06), Abstract, §3](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.html)

```text
from a single expert trajectory

専門家の軌跡1本から
```
[CVF / Dopamine-Reward (2026/06), Abstract](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.html)

### 3.5 Robometer

RobometerはQwen3-VL-4Bを基盤とし、成功動画内の進捗と、成功・失敗の動画間の優劣を同時に学ぶ。LIBEROの2課題では5 seedで疎報酬より2〜4倍のサンプル効率を報告し、実機でも既存のπ₀をDSRLで改善する。人が場面を戻す作業は残る。追加適応のLoRAは単一RTX A6000で約8時間だが、大規模事前学習全体の費用ではない。長い課題で過去の特定イベントを記憶する能力には限界がある。[arXiv / Robometer (2026/05), §III–V, Appendix F](https://arxiv.org/html/2603.02115v2)

```text
manual human intervention is required only for physical scene resets

人による介入は物理的な場面のリセットに必要となる
```
[arXiv / Robometer (2026/05), §IV-Q3](https://arxiv.org/html/2603.02115v2)

## 4. 学習する課題の順序を作る

### 4.1 LEACL

LEACLは課題を分解し、各課題のパラメータと難易度の尺度をLLMで作り、自動カリキュラムへ渡す。GPT-4o miniとPPOを用い、LIBEROを拡張した環境の長い操作5課題で検証する。各条件で独立な5方策を学習し、疎な成功報酬を使う。LLMが密な報酬を設計する方式とは違うが、課題を判定する条件や環境を変化させる仕組みは必要である。[arXiv / LEACL (2026/07), §III–V](https://arxiv.org/html/2607.23515v1)

```text
using only sparse reward signals

疎な報酬信号だけを使って
```
[arXiv / LEACL (2026/07), Abstract](https://arxiv.org/abs/2607.23515)

## 5. 探索と行動を補助する

### 5.1 LLM-Explorer

前回扱ったLLM-Explorerをv2で再確認した。最近の行動・報酬の履歴から探索用の分布を作り、定期更新する方式であり、各状態の最適行動をLLMが与えるわけではない。15ゲームの平均改善37.27%を全ゲーム共通の改善と読めない。Qbertの500k条件では基準306.07に対し301.97であり、特定面の攻略を保証する結果ではない。[arXiv / LLM-Explorer (2025/10), §3–4, Table 1](https://arxiv.org/html/2505.15293v2)

```text
recent action-reward trajectories

最近の行動と報酬の軌跡
```
[arXiv / LLM-Explorer (2025/10), §1](https://arxiv.org/html/2505.15293v2)

### 5.2 Exploration with Foundation Models

この研究は、目標を説明できることと、細かな操作を実行できることの差を調べる。GPT-4oを疎報酬のAtari 7ゲームで調べ、Montezumaでは0点などの限界を報告する。PPOとVLMを組み合わせた改善実験は、VLMの戦略が正しく操作も簡単なFreewayを選んだ100k steps・5 seedの実証である。難探索ゲーム全般での改善と広げて読まない。[arXiv / Exploration with Foundation Models (2025/09), §4–5](https://arxiv.org/html/2509.19924v1)

```text
a controlled, best-case scenario

条件を制御した、最も有利な状況
```
[arXiv / Exploration with Foundation Models (2025/09), Abstract](https://arxiv.org/abs/2509.19924)

### 5.3 TextAtari

TextAtariは状態を文章で示して言語エージェントを評価するベンチマークで、Qwen2.5-7B、Gemma-7B、Llama3.1-8Bを比較する。説明書の付与や反省などの条件を分けるが、人間との大きな性能差を報告する。入力は画像から推論した記述ではなく、AtariARIによるRAMのラベルをテンプレートで文章化する。[arXiv / TextAtari (2025/06), §2.2–3](https://arxiv.org/html/2506.04098v2)

タイトルは100Kだが、v2の実験手順は23ゲーム・1,000 interaction step・5 seedと明記し、PDFの7頁と付録21頁でも確認した。100K stepを通して評価済みという意味でも、通常のAtari 100k学習設定でもない。DQNへ組み込んで学習を改善した証拠とも区別する。[v2 §3.1、Appendix E.3](https://arxiv.org/pdf/2506.04098v2)

```text
a horizon of 1000 interaction steps (for cost saving)

評価の長さは1,000 interaction step（費用節約のため）
```
[arXiv / TextAtari v2（2025/06）, §3.1](https://arxiv.org/pdf/2506.04098v2)

公開commit `f95cb5af4fd761520801b0af216dbd7acc392320`では、wrapperは既定frameskip 4・sticky action確率0でRAMラベルを返す。Qbertの文章化器は座標・敵の列・得点・tile色を使う。画素だけの観測条件へ結果を移さない。静的な確認であり、実験全runの設定との対応やQbert第8面の攻略は未確認。[固定wrapper](https://github.com/Lww007/Text-Atari-Agents/blob/f95cb5af4fd761520801b0af216dbd7acc392320/envs/atari/represented_atari_game.py#L38)、[Qbertの文章化](https://github.com/Lww007/Text-Atari-Agents/blob/f95cb5af4fd761520801b0af216dbd7acc392320/envs/atari/Qbert_translator.py#L7)

同commitのREADMEではObscuringを「無作為方策の軌跡を、その旨を知らせずに与える」と説明し、v2本文の名詞をitemへ置換する条件とは異なる。シナリオ名だけで論文と公開版を対応させない。[固定README](https://github.com/Lww007/Text-Atari-Agents/blob/f95cb5af4fd761520801b0af216dbd7acc392320/README.md#-benchmark-structure)

### 5.4 Generative Code Optimization

この研究は方策をPythonプログラムとして表し、実行の記録と自然言語のフィードバックで改良する。Pong・Breakout・Space Invadersで、OCAtariの物体情報、Claude 3.5 Sonnet、ゲーム別の構造と段階別フィードバックを使う。RL方策への軽い補助機能ではなく、方策を表す方法から変える研究である。公開リポジトリに他ゲームのファイルがあっても、論文での評価対象が増えるわけではない。[arXiv / Generative Code Optimization (2025/08), §3–5, Appendix A–C](https://arxiv.org/html/2508.19506v1)

```text
policies are represented as Python programs

方策をPythonプログラムとして表す
```
[arXiv / Generative Code Optimization (2025/08), Abstract](https://arxiv.org/abs/2508.19506)

### 5.5 ULPS

ULPSはA*が作った正解の行動列でBERTを調整し、複数回のdropout推論から求める不確かさでPPOとの混合率を変える。MiniGridで検証するが、汎用生成LLMの知識だけを使った実験ではない。中間の鍵取得・扉開放の報酬があり、全実験はseed 42の1条件である。99.17%は正解行動との一致率であり、RLの成功率とは区別する。[arXiv / ULPS (2026/06), §II–III](https://arxiv.org/html/2606.06673v1)

```text
All experiments were carried out on a fixed random seed (42).

すべての実験は固定した乱数seed（42）で行われた。
```
[arXiv / ULPS (2026/06), §III-B](https://arxiv.org/html/2606.06673v1)

## 6. 学習則そのものを探索する

### 6.1 Evolutionary Discovery of RL Algorithms

LLMに実行可能な更新則を書かせ、複数環境の学習結果で選ぶ。探索時にはactor–critic、TD誤差、bootstrapを禁じ、新しい構成を促す。GPT-5.2とClaude 4.5 Opusを各2探索seed、10世代で比較する。最終成績は環境別のハイパーパラメータ探索後の最良値を含むため、未知環境へ無調整で転移した成績とは異なる。報酬の自動生成より、探索対象の範囲が広い。[arXiv / Evolutionary Discovery (2026/03), §3.5–5](https://arxiv.org/html/2603.28416v1)

```text
after post-evolution hyperparameter optimization

進化的探索の後にハイパーパラメータを最適化してから
```
[arXiv / Evolutionary Discovery (2026/03), §1](https://arxiv.org/html/2603.28416v1)

## 7. 比較条件と失敗を調べる

### 7.1 BIRD

BIRDは14方式を共通の実装で表現し、うち6方式と30の設計上の選択を比較する。主要比較の基準は5反復×8候補で、報酬を選ぶ際に正解の採点基準を使う群と、VLMに評価させる群を分ける。ただし方法別の実行数には例外があり、全方式が40方策とは限らない（[配布実装の確認](llm_assisted_rl_reproducibility_2026-11-07.md#4-bird)）。多くの工夫は別課題や別構成では効果が小さいか不明確だった。高いベンチマーク得点と望ましい振る舞いの不一致も調べるが、人による評価は2人と小規模である。[arXiv / BIRD (2026/10), §4, Appendix I–J](https://arxiv.org/html/2610.04364v1)

ERAという提案法の成績だけでなく、共有実装による比較条件の統一がこの資料の価値である。個々の原論文とBIRDでモデル・予算が違う場合、その数値をそのまま原論文の再現失敗と判定しない。比較対象・予算・評価源は同論文の§4.1とAppendix Iに明記されている。[arXiv / BIRD (2026/10), §3–4](https://arxiv.org/html/2610.04364v1)

### 7.2 Diagnostic-Driven Refinement

報酬が出過ぎる、APIの意味を取り違える、といった不具合を診断してコードを直す研究である。PPOと構造化されたMiniGridを中心に、1〜4回のLLM呼び出しを使う。大幅な改善値はあるが、10月のv2では同じ指示を与えた3モデル×2課題の比較において、多重比較補正後に修正の有意な優位性は残らなかった。4呼び出し・同一episode予算の候補選択も競争力がある。密な報酬の移動課題では診断が誤作動した。[TMLR / Diagnostic-Driven Refinement (2026/09), §9.4–9.8; arXiv v2 (2026/10)](https://arxiv.org/html/2605.28918v2)

```text
none of the six contrasts survives family-wise correction

6つの比較はいずれも、多重比較の補正後には有意性が残らない
```
[arXiv / Diagnostic-Driven Refinement (2026/10), Abstract](https://arxiv.org/abs/2605.28918)

### 7.3 Policy-Invariant Reward Shaping from LLM Feedback

このpreprintはLLMの採点をポテンシャルとして使う理論を示し、ローカルQwen2.5-14Bによる計画も調べる。しかし実環境のpilotでは計算費用のため報酬整形を無効にしており、成功率はPPOの28.1%に対し9.3%だった（3 seed、信頼区間は重なる）。計画の語彙と完了判定が合わず、目標が先へ進まなかったと報告する。[arXiv / Hounwanou et al. (2026/08), §4.7–5](https://arxiv.org/html/2608.18008v1)

```text
the total number of subgoal advances was zero

小目標が次へ進んだ回数は合計で0だった
```
[arXiv / Hounwanou et al. (2026/08), §5](https://arxiv.org/html/2608.18008v1)

理論については本調査で注意点を見つけた。式(6)が遷移後も古い目標で採点するため、目標が切り替わる場合の一般的な不変性は成立しない反例を構成できる。これは本調査の指摘であり、著者自身の結論ではない。[別冊の数式照合・反例・検算](llm_reward_shaping_guarantees_2026-11-07.md#3-本調査で構成した反例)に条件をすべて記した。

### 7.4 CriticHack

CriticHackは、報酬と成功率が上昇しても、別の物体を操作する失敗が同時に増え得ることを検証する。Robometerで引き出し操作の方策を最適化した5学習・512評価seedの比較では、成功が10.2ポイント、対象を取り違える失敗が10.9ポイント増えた。シミュレータの完了信号との比較もある。ただしシミュレーションは2課題、方策全体の更新は1種類の報酬モデルであり、すべての視覚報酬に同じ失敗率があるとはいえない。[arXiv / CriticHack (2026/10), §5, §9, Appendix C.2](https://arxiv.org/html/2610.02527v1)

```text
while reward and task success both rise

報酬と課題の成功率の両方が上がる間にも
```
[arXiv / CriticHack (2026/10), Abstract](https://arxiv.org/abs/2610.02527)

## 8. 費用とローカル実行の証拠

### 8.1 費用の単位を揃える

次の値は論文ごとに課題・機材・価格時点が違う。横並びの順位表にはしない。金額は論文内の記載であり、現在のAPI価格ではない。

| 研究 | 費用として報告されたもの | 読む際の境界と出典 |
|---|---|---|
| LIMEN | 主実験5探索の合計11.43 GPU時間、3.76M tokens、LLM費用$17.50。単一NVIDIA L4、Claude Sonnet 4.6 | 5課題合計であり、1候補の費用ではない。[Appendix B.1, 計算費用の表](https://arxiv.org/html/2605.03408v1) |
| EvoNav | CARDの36 GPU時間に対し9.2時間。LLM tokensは72Kから720Kへ増加 | GPU試行を減らし、LLMの利用を増やす構成。[Appendix C, Table 9](https://arxiv.org/html/2605.11859v1) |
| LRM | RTX 6000 Adaで進捗360±19 ms、完了340±5 ms、比較7093±2307 ms | 1回のHTTP往復。起動・モデル読込を除く。RL全体の速度ではない。[Table VIII](https://arxiv.org/html/2603.16065v3) |
| Diagnostic-Driven Refinement | 報酬設計1回につき1〜4 LLM呼び出しとCPUでの学習 | seedごとの繰り返しを含む総費用ではない。[§1, §9](https://arxiv.org/html/2605.28918v2) |
| Dopamine-RL | 実ロボットで約1時間・150試行という例 | 報酬モデルの大規模事前学習と課題適応は別。[Abstract](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.html) |

### 8.2 小型モデルで何が確かめられているか

「公開された重みがある」「単一GPUに載る」「十分速い」「対象課題でRLが改善する」はそれぞれ別の確認項目として扱った。今回の資料から確認できた範囲は次のとおりである。

| 研究 | 確認できたモデルと用途 | この調査で確認していないこと |
|---|---|---|
| RoboReward | 4B・8Bをロボットの報酬予測用に学習、8Bで実機RL。[§4–5](https://arxiv.org/html/2601.00675v2) | Atariへの転用、個人用GPUでRLと同居した速度 |
| LRM | Qwen3-VL-8Bを用途別に調整し、RTX 6000 Adaで遅延測定。[§III-C, Table VIII](https://arxiv.org/html/2603.16065v3) | 無調整の8Bで同じ成績になること、他GPUでの速度 |
| Robometer | Qwen3-VL-4Bを調整したモデルでシミュレーション・実機RL。[§III–IV](https://arxiv.org/html/2603.02115v2) | 長期記憶が必要なAtari、GPU共有時の総実時間 |
| TextAtari | 7B・8Bの言語エージェントを評価。[§3](https://arxiv.org/html/2506.04098v2) | RL学習の改善、GPUを共有した総所要時間 |
| EvoNav | Llama 3.1の70B・405B、GPT-OSS-120Bなどを比較。[Table 8](https://arxiv.org/html/2605.11859v1) | 7B程度の報酬コード生成器での同等性 |

この表では、モデルを小さくしただけで同等の結果になるとは判定していない。教師モデルによる事前のラベル付けと、実行時の小型報酬モデルを分けるRoboRewardの構成は、二つの費用を混同せずに読める具体例である。[arXiv / RoboReward (2026/01), §4.2](https://arxiv.org/html/2601.00675v2)

## 9. 到達点と調査の限界

前回から追加された大きな論点は、LLMが出す助言の内容だけでなく、報酬・観測・課題・更新則のどこを生成対象とするかである。[LIMEN (2026/05)](https://arxiv.org/abs/2605.03408)、[LEACL (2026/07)](https://arxiv.org/abs/2607.23515)、[Evolutionary Discovery (2026/03)](https://arxiv.org/abs/2603.28416)は、その異なる範囲を具体的に検証している。

効果の読み方については、指示・フィードバック・学習予算を揃えた比較を優先する。BIRDとDiagnostic-Driven Refinementの改訂版は、複雑な仕組みの追加だけでは改善を説明し切れないことを示す資料である。[BIRD (2026/10), §4.2](https://arxiv.org/html/2610.04364v1)、[Diagnostic-Driven Refinement (2026/10), §9.4, §9.8](https://arxiv.org/html/2605.28918v2)

Atariでは、今回精読した資料にもKungFuMasterのボス攻略やQbert第8面を直接改善した証拠は見つからなかった。これは検索・精読範囲内の未確認であり、不可能という主張ではない。LLM-Explorerのゲーム別スコア、Freewayの有利な条件での実証、TextAtariの言語エージェント評価、3ゲームのプログラム方策は、別々の結果として保持する。[§5.1](#51-llm-explorer)、[§5.2](#52-exploration-with-foundation-models)、[§5.3](#53-textatari)、[§5.4](#54-generative-code-optimization)

本調査では学習実験・モデルダウンロード・有料API実行を行っていない。量子化による劣化、GPU共有時の速度、対象環境への転移、全失敗候補を含む総費用は再現していない。ロボットの意味付き状態や専門家動画を使う結果を、画素だけのAtariへ適用済みの根拠として扱わない。

報酬モデルの評価と、それで方策を最適化した結果も分ける。Robometerの成功例とCriticHackの対象取り違えは、課題と初期方策が異なる条件での結果であり、どちらかを消して一つの評価へまとめない。[Robometer §IV](https://arxiv.org/html/2603.02115v2)、[CriticHack §5・§9](https://arxiv.org/html/2610.02527v1)

公開実装の既定値・再試行・採点処理は[再現条件の別冊](llm_assisted_rl_reproducibility_2026-11-07.md)へ、目標切り替え時の数式の指摘と小さな検算は[保証の別冊](llm_reward_shaping_guarantees_2026-11-07.md)へ分けた。

追加調査では、[制御コードと学習環境の5論文](llm_generated_control_and_environments_2026-11-07.md)、[小型モデルと費用の4論文](llm_rl_small_models_and_costs_2026-11-07.md)、[探索に渡す情報の2論文](llm_rl_exploration_information_2026-11-07.md)、[更新則発見の3論文](llm_rl_algorithm_discovery_2026-11-07.md)、[言葉を環境へ結び付ける3論文](llm_rl_language_grounding_2026-11-07.md)、[小目標と振り返りの4論文](llm_rl_goals_and_reflection_2026-11-07.md)、[行動助言の3論文](llm_rl_action_advice_2026-11-07.md)、[蒸留と先読みの2論文](llm_rl_distillation_and_planning_2026-11-07.md)を別冊にした。本文と合わせて47論文であり、原著の結果・公開コードの状態・本調査の推論を区別している。

探索の別冊ではiLLMの補足資料を読み、前回の問い合わせ頻度の説明を訂正した。小目標の別冊ではLAGEAの単一RTX 4090・3B VLMという実行条件を追加した。実時間表は一部課題の平均改善であり、全課題で速いという結果ではない。目標画像と成功ラベルなど、LLMとは別に用意する情報も整理した。[全体の読み順と研究上の問い](llm_rl_reading_guide_2026-11-07.md)

## 10. 書誌と探索記録

### 10.1 著者と所属

下表の所属は確認した論文版の記載。役割は論文の著者・責任著者・同等貢献など、原典が明記するものに限る。現在の雇用先を推定していない。著者全員の氏名はリンク先にある。

| 資料 | 著者・役割 | 原典の所属 | 発表と確認版 |
|---|---|---|---|
| [Eureka](https://arxiv.org/abs/2310.12931) | Yecheng Jason Maら全9名、著者 | NVIDIA、UPenn、Caltech、UT Austin | ICLR 2024、arXiv v2 2024-04-30 |
| [R*](https://proceedings.mlr.press/v267/li25v.html) | Pengyi Liら全7名、Jianye Haoが責任著者 | Tianjin University | ICML 2025、PMLR 267 |
| [PoRSE](https://openreview.net/pdf?id=1vXMfIYFZp) | Yanbin Chang・Junfan Linが同等貢献、Changxin Huangが責任著者、全6名 | Shenzhen University、Peng Cheng Laboratory、Shenzhen MSU-BIT University | ICLR 2026、採録PDF |
| [LIMEN](https://arxiv.org/html/2605.03408v1) | Akshat Singh Jaswal、Ashish Baghel、Paras Chopra、著者 | Lossfunk | RLC 2026（[著者README](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/README.md)で確認）、arXiv v1 2026-05-05 |
| [EvoNav](https://arxiv.org/html/2605.11859v1) | Zhikai Zhao・Chuanbo Huaが同等貢献、全7名 | KAIST、Radical Numerics、UC Riverside、Omelet／AI4CO | arXiv v1 2026-05-12 |
| [RL-VLM-F](https://arxiv.org/html/2402.03681v2) | Yufei Wangら全7名、著者 | CMU、USC | ICML 2024、PMLR 235 |
| [RoboReward](https://arxiv.org/html/2601.00675v2) | Tony Lee・Andrew Wagenmaker・Karl Pertschが主要貢献者、全6名 | Stanford、UC Berkeley | arXiv v2 2026-01-08 |
| [LRM](https://arxiv.org/pdf/2603.16065v3) | Yanru Wuら全7名、Jiageng Mao・Yue Wangが同等の指導貢献 | Vitor GuiziliniがToyota Research Institute、他6名がUSC（PDF第1頁で確認） | arXiv v3 2026-09-30 |
| [Dopamine-Reward](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.html) | Huajie Tan・Sixiang Chen・Yijie Xuが同等貢献、Shanghang Zhangが責任著者、全15名 | Peking University、BAAI、University of Sydney、CAS Institute of Automation | CVPR 2026、採録版。arXiv版と題名・著者順の一部が異なる |
| [Robometer](https://robometer.github.io/) | Anthony Liang・Yigit Korkmazが同等貢献、Stephen Tu・Erdem Bıyık・Jesse Zhangが同等の指導貢献、全17名 | USC、UT Dallas、MIT、独立研究者、UW、Ai2、NVIDIA | RSS 2026、arXiv v2 2026-05-13 |
| [LEACL](https://arxiv.org/abs/2607.23515) | Faraz Heraviら全6名、著者 | UT Austin、NTU、Sony AI | IROS 2026、arXiv v1 2026-07-26 |
| [LLM-Explorer](https://arxiv.org/html/2505.15293v2) | Qianyue Hao・Yiwen Songが同等貢献、Yong Liが責任著者、全5名 | Tsinghua University / BNRist | [NeurIPS 2025本会議](https://papers.nips.cc/paper_files/paper/2025/hash/869fdb8fb32c561c66e16611e250fee1-Abstract-Conference.html)、arXiv v2 2025-10-23 |
| [Exploration with Foundation Models](https://arxiv.org/abs/2509.19924) | Remo Sasso、Michelangelo Conserva、Dominik Jeurissen、Paulo Rauber、著者 | Queen Mary University of London | NeurIPS 2025 FoRLM Workshop、v1 2025-09-24 |
| [TextAtari](https://arxiv.org/abs/2506.04098) | Wenhao Liら全11名、著者 | Tongji、ECNU、Shanghai Jiao Tong、CUHK Shenzhen、Bank of Communications、独立研究者（著者と所属の個別対応はHTMLの崩れがあり未確定） | arXiv v2 2025-06-10 |
| [Generative Code Optimization](https://arxiv.org/html/2508.19506v1) | Zhiyi Kuangが責任著者、Ryan Rong・YuCheng Yuanが同等貢献、Allen Nie、全4名 | Stanford University | ICML 2025 PRAL Workshop、v1 2025-08-27 |
| [ULPS](https://arxiv.org/html/2606.06673v1) | Ujjwal Bhatta・Utsabi Dangol・Sumaly Bajracharyaが同等貢献、Rodrigue Rizk・KC Santosh、全5名 | University of South Dakota | IEEE CAI 2026、v1 2026-06-04 |
| [Evolutionary Discovery](https://arxiv.org/abs/2603.28416) | Alkis Sygkounas、Amy Loutfi、Andreas Persson、著者 | Örebro University | GECCO 2026採録、v1 2026-03-30 |
| [BIRD](https://arxiv.org/html/2610.04364v1) | Logan M. Bhamidipaty・Lauren Robsonが同等貢献、全5名 | Anthropic Fellows Program、University of Edinburgh、Anthropic | arXiv v1 2026-10-03 |
| [Diagnostic-Driven Refinement](https://arxiv.org/html/2605.28918v2) | Youting Wang、Yuan Tang、Bowen Liu、Xuan Liu、Dingyan Shang、著者 | 全員Independent Researcher | TMLR 2026-09、arXiv v2 2026-10-10 |
| [Policy-Invariant Reward Shaping from LLM Feedback](https://arxiv.org/pdf/2608.18008v1) | Christophe D. Hounwanouが責任著者、John Emeka Eze、Yaé U. Gaba | AIMS Rwanda、AIRINA Labs、SMU、ACAS | arXiv v1 2026-08-18、採録先未確認 |
| [CriticHack](https://arxiv.org/pdf/2610.02527v1) | Jiaxuan Luo・Xingguo Xu・Shanshan Wangが同等貢献、Luoが責任著者、Yuhan Zhou・Zhen Zhang、全5名 | Johns Hopkins、Dalian University of Technology、Rutgers、Baidu、UC Santa Barbara | arXiv v1 2026-10-01、採録先未確認 |

### 10.2 被引用数

2026-11-07にOpenAlexの`cited_by_count`を取得した。タイトルとDOIが一致したレコードだけを採用した。収録遅延や会議版との分離があるため、研究全体の総被引用数ではない。未取得と0件を区別し、優劣の順位には使わない。

| 資料 | OpenAlexで確認した件数 | レコード |
|---|---:|---|
| Eureka | 139 | [W4387839203](https://openalex.org/W4387839203) |
| R* | 未取得 | 検索結果のタイトル不一致 |
| PoRSE | 未取得 | 検索結果のタイトル不一致 |
| LIMEN | 0 | [W7160410645](https://openalex.org/W7160410645) |
| EvoNav | 0 | [W7161013377](https://openalex.org/W7161013377) |
| RL-VLM-F | 28 | [W4391631669](https://openalex.org/W4391631669) |
| RoboReward | 1 | [W7118501563](https://openalex.org/W7118501563) |
| LRM | 1 | [W7138850231](https://openalex.org/W7138850231) |
| Dopamine-Reward | 未取得 | 検索結果のタイトル不一致 |
| Robometer | 0 | [W7133361213](https://openalex.org/W7133361213)、RSS版DOIのレコード |
| LEACL | 0 | [W7171511008](https://openalex.org/W7171511008) |
| LLM-Explorer | 6 | [W4415025005](https://openalex.org/W4415025005)、会議版DOIのレコード |
| Exploration with Foundation Models | 未取得 | 検索結果のタイトル不一致 |
| TextAtari | 0 | [W4416074961](https://openalex.org/W4416074961) |
| Generative Code Optimization | 未取得 | 検索結果のタイトル不一致 |
| ULPS | 0 | [W7162992581](https://openalex.org/W7162992581)、IEEE版DOIのレコード |
| Evolutionary Discovery | 0 | [W7167969131](https://openalex.org/W7167969131)、GECCO版DOIのレコード |
| BIRD | 0 | [W7220726394](https://openalex.org/W7220726394) |
| Diagnostic-Driven Refinement | 2 | [W7162754949](https://openalex.org/W7162754949) |
| Policy-Invariant Reward Shaping from LLM Feedback | 0 | [W7203792414](https://openalex.org/W7203792414) |
| CriticHack | 0 | [W7220443208](https://openalex.org/W7220443208) |

Semantic Scholar APIはHTTP 429で取得できなかった。OpenAlexの検索では無関係な論文が先頭に返ったものがあり、その数を転記していない。

### 10.3 一次資料一覧

- [Eureka, 2024/04] Yecheng Jason Ma et al. “Eureka: Human-Level Reward Design via Coding Large Language Models.” ICLR 2024. [論文](https://arxiv.org/abs/2310.12931)、[著者のプロジェクト](https://eureka-research.github.io/)。
- [R*, 2025/07] Pengyi Li et al. “R*: Efficient Reward Design via Reward Structure Evolution and Parameter Alignment Optimization with Large Language Models.” ICML 2025. [論文・書誌](https://proceedings.mlr.press/v267/li25v.html)。
- [PoRSE, 2026] Yanbin Chang et al. “Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring.” ICLR 2026. [論文](https://openreview.net/pdf?id=1vXMfIYFZp)。
- [LIMEN, 2026/05] Akshat Singh Jaswal, Ashish Baghel, Paras Chopra. “Discovering Reinforcement Learning Interfaces with Large Language Models.” arXiv. [論文](https://arxiv.org/abs/2605.03408)、[実装](https://github.com/Lossfunk/LIMEN)。
- [EvoNav, 2026/05] Zhikai Zhao et al. “EvoNav: Evolutionary Reward Function Design for Robot Navigation with Large Language Models.” arXiv. [論文](https://arxiv.org/abs/2605.11859)。
- [RL-VLM-F, 2024/07] Yufei Wang et al. “RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback.” ICML 2024. [論文・書誌](https://proceedings.mlr.press/v235/wang24bn.html)。
- [RoboReward, 2026/01] Tony Lee et al. “RoboReward: General-Purpose Vision-Language Reward Models for Robotics.” arXiv. [論文](https://arxiv.org/abs/2601.00675)。
- [LRM, 2026/09] Yanru Wu et al. “Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models.” arXiv v3. [論文](https://arxiv.org/html/2603.16065v3)。
- [Dopamine-Reward, 2026/06] Huajie Tan et al. “General Process Reward Modeling for Robotic Reinforcement Learning.” CVPR 2026. [採録版](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.html)、[著者のプロジェクト](https://robo-dopamine.github.io/)。
- [Robometer, 2026/05] Anthony Liang et al. “Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons.” RSS 2026. [論文v2](https://arxiv.org/html/2603.02115v2)、[著者のプロジェクト](https://robometer.github.io/)。
- [LEACL, 2026/07] Faraz Heravi et al. “LEACL: LLM-Enhanced Automatic Curriculum Learning for Reinforcement Learning in Long-Horizon Manipulation Tasks.” IROS 2026. [論文](https://arxiv.org/abs/2607.23515)。
- [LLM-Explorer, 2025/10] Qianyue Hao et al. “LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models.” NeurIPS 2025. [論文](https://arxiv.org/abs/2505.15293)、[本会議の書誌](https://papers.nips.cc/paper_files/paper/2025/hash/869fdb8fb32c561c66e16611e250fee1-Abstract-Conference.html)、[実装](https://github.com/tsinghua-fib-lab/LLM-Explorer)。
- [Exploration with Foundation Models, 2025/09] Remo Sasso et al. “Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches.” NeurIPS 2025 FoRLM Workshop. [論文](https://arxiv.org/abs/2509.19924)。
- [TextAtari, 2025/06] Wenhao Li et al. “TextAtari: 100K Frames Game Playing with Language Agents.” arXiv v2. [論文](https://arxiv.org/abs/2506.04098)。
- [Generative Code Optimization, 2025/08] Zhiyi Kuang et al. “Learning Game-Playing Agents with Generative Code Optimization.” ICML 2025 PRAL Workshop. [論文](https://arxiv.org/abs/2508.19506)、[実装](https://github.com/ameliakuang/LLM-Game-Playing-Agents)。
- [ULPS, 2026/06] Ujjwal Bhatta et al. “Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning.” IEEE CAI 2026. [論文](https://arxiv.org/abs/2606.06673)。
- [Evolutionary Discovery, 2026/03] Alkis Sygkounas, Amy Loutfi, Andreas Persson. “Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models.” GECCO 2026. [論文](https://arxiv.org/abs/2603.28416)。
- [BIRD, 2026/10] Logan M. Bhamidipaty et al. “A Bird’s-Eye View of Iterative Reward Design.” arXiv. [論文](https://arxiv.org/abs/2610.04364)、[実装](https://github.com/safety-research/bird)。
- [Diagnostic-Driven Refinement, 2026/10] Youting Wang et al. “When LLM Reward Design Fails: Diagnostic-Driven Refinement for Sparse Structured RL.” TMLR 2026. [改訂論文](https://arxiv.org/abs/2605.28918)。
- [Policy-Invariant Reward Shaping from LLM Feedback, 2026/08] Christophe D. Hounwanou, John Emeka Eze, Yaé U. Gaba. “Policy-Invariant Reward Shaping from LLM Feedback: A Framework for Hybrid RL Agents.” arXiv. [論文v1](https://arxiv.org/html/2608.18008v1)。
- [CriticHack, 2026/10] Jiaxuan Luo et al. “CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization.” arXiv. [論文v1](https://arxiv.org/html/2610.02527v1)。

### 10.4 探索と確認の範囲

検索語はLLM / reinforcement learning / reward code / exploration / curriculum / local model / Qwen / Atariと、見つけた論文の題名・後続文献を組み合わせた。PMLR、OpenReview、CVF、arXiv、著者の実装を技術的主張の根拠とした。第三者のまとめやレビューは候補の発見に使い、数値は一次資料へ戻した。

特に版を区別したものはLRMのv3、Diagnostic-Driven Refinementのv2、TextAtariのv2、RoboRewardのv2、LLM-Explorerのv2である。Workshop採録を本会議採録へ読み替えていない。OpenReviewとCVFでは直接URLが一時的に拒否され、検索結果から取得済みの同じ公式PDFを読み直した例がある。

網羅的な系統的レビューではない。検索順位・公開範囲・取得できた版による偏りがある。被引用数の未取得、TextAtariの著者と所属の個別対応、公開実装の動的な再現性、ローカルGPUでの速度は、それぞれ別の未確認事項として残した。

文書の目次と見出しの一致、内部リンク、短い原文引用と和訳・出典の書式、LF・BOMを保存後に検査した。理論の反例だけは小さな有理数計算で検算した。これはRLの再現実験に数えない。
