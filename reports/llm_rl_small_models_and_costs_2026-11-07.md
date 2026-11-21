# 小型モデルと計算費用から見たLLMとRL

確認日: 2026-11-07（JST）  
対象: 小型モデルの役割と、学習費用の削減箇所を確認できる4論文。[再サーベイ本文](llm_assisted_rl_resurvey_2026-11-07.md)と[制御・世界モデル・環境生成の別冊](llm_generated_control_and_environments_2026-11-07.md)を補う。  
方法: 一次論文・著者コードの静的調査。サブエージェント、学習、モデル取得、有料API呼び出しは使っていない。  
範囲: 外部環境の制御・報酬設計。数学回答のRLは比較対象に含めず、DERLではALFWorld・ScienceWorldだけを扱う。

## 目次

1. [1. 対象と書誌情報](#1-対象と書誌情報)
2. [2. VLM-AR3L：小型VLMの判定を軽い報酬モデルへ移す](#2-vlm-ar3l小型vlmの判定を軽い報酬モデルへ移す)
3. [3. SCOUT：軽いRL方策で集めた経験をLLMへ渡す](#3-scout軽いrl方策で集めた経験をllmへ渡す)
4. [4. EARS：報酬を作る段階の環境サンプリングを省く](#4-ears報酬を作る段階の環境サンプリングを省く)
5. [5. DERL：小さな報酬提案器も学習させる](#5-derl小さな報酬提案器も学習させる)
6. [6. VLM-AR3Lの公開コードで確認した条件](#6-vlm-ar3lの公開コードで確認した条件)
7. [7. 費用と効果を分けて読む](#7-費用と効果を分けて読む)
8. [8. 調査の限界](#8-調査の限界)

## 1. 対象と書誌情報

「小型モデルでできる」という主張は、何を担当させたかまで必要である。比較画像へのラベル付け、方策そのもの、報酬式の提案は、それぞれ入力・更新方法・費用が違う。本冊はこの違いを中心に読む。

| 論文 | 著者・所属・役割 | 確認版・掲載先 | 被引用数 |
|---|---|---|---:|
| [VLM-AR3L](https://ijcai.org/proceedings/2026/110) | Kuan-Chen Chen（第一著者）、Winston Chen、Min-Chun HuはNational Tsing Hua University CS。Wei-Fang SunはNVIDIA AI Technology Center | IJCAI 2026、[arXiv v1](https://arxiv.org/abs/2607.00483) 2026-07-01 | [0](https://openalex.org/W7167029937) |
| [Language-based Trial and Error Falls Behind in the Era of Experience](https://proceedings.mlr.press/v306/wang26fn.html) | Haoyu Wang（第一著者）ら10名。NTU、SenseAuto、Sichuan University、Sun Yat-sen University。Dacheng Taoが責任著者 | ICML 2026、PMLR 306。初稿01-29、確認版[arXiv v3](https://arxiv.org/abs/2601.21754) 06-08 | [0](https://openalex.org/W7126271297) |
| [Specifying Reward Functions for RL Without Environment Sampling](https://arxiv.org/abs/2609.15544) | Stephane Hatgis-Kessell（第一・責任著者、Stanford）、W. Bradley Knox（UT Austin）、Emma Brunskill（Stanford） | 2026-09-14 v1、preprint | [0](https://openalex.org/W7213366420) |
| [Dual-Evolving Reinforcement Learning for Automated Reward Design](https://arxiv.org/abs/2512.13399) | Sitao Cheng（Waterloo）、Tianle Li（HKU）、Xuhan Huang（CUHK Shenzhen）の3名が共同第一著者。Xunjian Yin（Duke）、Difan Zou（HKU） | 初稿2025-12-15、確認版v4 2026-10-27、preprint | [0](https://openalex.org/W4417458470) |

所属と役割は論文著者欄と[DERLの著者ページ](https://sitaocheng.github.io/derl/)による。被引用数は2026-11-07のOpenAlex取得値。収録遅れがあるため、0を実際の総引用数の断定に使わない。SCOUTは会議の掲載をPMLRで確認し、本文はv3を読んだ。PMLR配布PDFとの全差分照合はしていない。

## 2. VLM-AR3L：小型VLMの判定を軽い報酬モデルへ移す

VLMが画像対の好みを判定し、そのラベルから状態の採点器と、前後の画像を比べる採点器を学習する。毎step大きなVLMへ問い合わせる代わりに、学習済み報酬モデルを使う。[論文 §3–4](https://arxiv.org/html/2607.00483v1)

```text
both exhibit clear task-specific strengths

両モデルには、課題ごとの明確な強みがある。
```
[VLM-AR3L（2026/07）, §5.4](https://arxiv.org/html/2607.00483v1)

この「両モデル」はPhi-3.5とMiniCPM-o 2.6を指す。主比較はGemini-2.0-Flashだが、前者2つを使ったRLのablationもある。各課題100画像対の正解率ではPhi-3.5がMilk Cow・Shear Sheepで0.90、CartPoleでは0.56。万能な小型判定器という結果ではない。学習曲線は3 seeds・checkpointごと5評価episodesで、表は最良成功率である。[同 §5.2–5.4、Table 1、Figure 5](https://arxiv.org/html/2607.00483v1)

著者はL40 48GBで実験し、RTX 3090 24GBでも再現可能と記す。1 seedは通常24時間以内、問い合わせ数は直接採点より約20分の1。この数字は総学習時間が20分の1という意味ではなく、全課題の中央値でもない。[同 Appendix B.3](https://arxiv.org/html/2607.00483v1)

## 3. SCOUT：軽いRL方策で集めた経験をLLMへ渡す

SCOUTではDQN・PPOの小さな方策が先に外部環境を探索し、得た軌跡でLLMをSFTした後、multi-turn PPOを行う。LLMからDQNへ知識を与える方式とは向きが逆である。[論文 §2](https://arxiv.org/html/2601.21754v3)

```text
Scouts operate primarily on CPUs

探索用の小さな方策は主にCPU上で動く。
```
[SCOUT（2026/06）, Table 2](https://arxiv.org/html/2601.21754v3)

Qwen2.5-3Bの平均0.86は、6種類のsymbolic課題をまとめた値で、2048には最大tileを正規化した指標が入る。全課題の成功確率86%とは読めない。小さなScout-DQN自体の平均も0.83であり、大型化だけの優位性を示す比較ではない。[同 Table 1](https://arxiv.org/html/2601.21754v3)

「60% GPU時間削減」はRubik’s Cube Rotation3の200学習stepに限る。8 H100でDirect PPOが24 GPU-h、SCOUTはSFT＋PPOが9.6 GPU-h、別にCPU探索0.17時間。LLM側の学習を1枚のGPUで済ませた証拠ではない。v3には画像入力のFrozenLake・Sokobanもあるが、Atari全体への適用は示していない。[同 §4.2、Table 3、§4.4](https://arxiv.org/html/2601.21754v3)

## 4. EARS：報酬を作る段階の環境サンプリングを省く

EARSはLLMが測定可能な特徴を定義し、特徴値を仮想的に組み合わせた軌跡への好みから重みを学ぶ。環境と相互作用しないのは報酬を作る手続きであり、実験の評価まで環境dataが不要という意味ではない。[論文 §3–4](https://arxiv.org/html/2609.15544v1)

```text
rather than reward functions that can train policies

方策を訓練できる報酬関数そのものを主な評価対象としているわけではない。
```
[EARS（2026/09）, Appendix A](https://arxiv.org/html/2609.15544v1)

主指標は500本の実軌跡に対する順位相関TACで、得た報酬によるRLの最終成績ではない。さらに合成の正解ラベルを使う実験では、Appendix Bのラベル生成器を評価軌跡集合の順位に合わせて学習する。これは自然言語で好みを与える実験とは別条件であり、「完全にdataを使わず、独立な実機評価で成功」とまとめてはいけない。[同 §4.1、Appendix A–B](https://arxiv.org/html/2609.15544v1)

確認版はGemini 3を使う。交通・感染症・血糖の3シミュレータの報酬整合性を扱い、実機運用や医療上の有効性を検証したものではない。[同 §4、Table 1、Appendix E.1](https://arxiv.org/html/2609.15544v1)

## 5. DERL：小さな報酬提案器も学習させる

DERLは0.5Bの提案器が、既存の報酬部品を数式で組み合わせる。各式で内側の方策を学習し、validationの成功率を提案器へ返して、その重みも更新する。新しい部品の意味を何もないところから作る設定ではない。[論文 §3–4](https://arxiv.org/html/2512.13399v4)

```text
The inner-loop dominates the cost.

計算費用の大部分は、内側の方策学習が占める。
```
[DERL（2026/10）, Appendix H](https://arxiv.org/html/2512.13399v4)

ALFWorld・ScienceWorldでは内側がQwen2.5-1.5B、4 seeds。未学習の課題種類を含むL2でも改善する。最後の方策学習の予算は揃えるが、その前の報酬探索費用は追加である。DERL-baseの探索は40〜128回の内側学習、DERL-fullも8本の全学習相当と最終1本を要する。小さな提案器というだけで、探索全体が軽いとは言えない。[同 §4.2、Appendix G–H](https://arxiv.org/html/2512.13399v4)

内側の学習を通して勾配を微分しているわけではない。validation成績を報酬にした提案器のpolicy gradientであり、「meta-gradient」はその区別と併せて読む。[同 Appendix I](https://arxiv.org/html/2512.13399v4)

## 6. VLM-AR3Lの公開コードで確認した条件

確認したrepositoryは[1224K/VLM-AR3L](https://github.com/1224K/VLM-AR3L/tree/ef2b1ab54451041cb3bf10ceba134794d0186962)、commitは`ef2b1ab54451041cb3bf10ceba134794d0186962`。以下は実行結果ではなく、その版の読解である。

配布既定値は小型ローカルモデルではなく`gemini2.0`。GymはSAC、MineDojoはPPOで、同じ名前の報酬でも正負の値はそれぞれ±1、±0.1。`k=16`、`absolute_alpha=0.5`、`confidence_threshold=0.52`は共通である。[Gym設定](https://github.com/1224K/VLM-AR3L/blob/ef2b1ab54451041cb3bf10ceba134794d0186962/config/gym.yaml)、[MineDojo設定](https://github.com/1224K/VLM-AR3L/blob/ef2b1ab54451041cb3bf10ceba134794d0186962/config/minedojo.yaml)

小型モデルの実装は含まれる。PhiはHugging FaceのモデルをCUDAへ読み、dtypeはauto。MiniCPMはbfloat16を指定する。これはロード方法の証拠であり、4bit量子化やCPUのみの動作・性能を確かめた記録ではない。[Phi入口](https://github.com/1224K/VLM-AR3L/blob/ef2b1ab54451041cb3bf10ceba134794d0186962/VLM/phi_infer.py#L15)、[MiniCPM入口](https://github.com/1224K/VLM-AR3L/blob/ef2b1ab54451041cb3bf10ceba134794d0186962/VLM/MiniCPM_infer.py#L16)

SAC側では画像の順序を入れ替えて2回採点し、両方向が閾値を満たしたときだけ正負の相対報酬を与える。片方だけの高い確信では0になる。絶対報酬との加重和で方策を学習し、報酬モデル更新後にはReplay Bufferも再採点する。[run_sac.py: 相対報酬](https://github.com/1224K/VLM-AR3L/blob/ef2b1ab54451041cb3bf10ceba134794d0186962/run_sac.py#L541)、[同: 再採点](https://github.com/1224K/VLM-AR3L/blob/ef2b1ab54451041cb3bf10ceba134794d0186962/run_sac.py#L481)

したがって、実装を移す場合の費用はVLM問い合わせだけではない。画像保持、2方向の報酬推論、報酬モデル学習、保存済み遷移の再採点も含む。この費用の切り分けは上の実装からの本調査の整理で、anet-lab上の測定値ではない。

## 7. 費用と効果を分けて読む

4方式の比較から、次の情報を別々に持つ必要がある。これは本調査による読み方の整理である。

| 観点 | 確かめる内容 |
|---|---|
| 小型化した対象 | ラベル付け用VLM、外部環境を操作する方策、報酬提案器のどれか |
| 省いた処理 | オンライン推論、初期探索、報酬設計時の実軌跡収集のどれか |
| 残っている処理 | 報酬学習、方策学習、事前学習、候補の棄却と再評価 |
| 成功の指標 | ラベル正解率、軌跡順位の相関、固定方策の成功率、探索中の最良値 |
| 費用の単位 | GPU-h、実時間、環境step、問い合わせ数、全候補を含む総量 |
| 報告の範囲 | 1課題のablationか、全課題共通の条件か、未使用課題への汎化か |

小型モデルの候補を選ぶ際には、モデルのparameter数だけでなく、対象課題の採点を確かめた証拠と、報酬学習・方策学習を含む総費用を並べる必要がある。本冊の結果だけから、7B・8Bを1つ選んで全てのAtariへ適用できるとは結論できない。

追加で確認したiLLMの補足資料には、AtariでTransfo-XL 280MとGPT2系・Llama2/3を比べた結果がある。4-bit使用の記載、バッチ実行、内発報酬を外した比較を[探索の別冊 §2–3](llm_rl_exploration_information_2026-11-07.md)へまとめた。小型モデルの有用性を検討する証拠を、ロボットの画像採点だけに限定しない。

さらにLAGEAでは、Qwen2.5-VL-3Bと外部RLを単一RTX 4090で使った報告を確認した。実時間は一部課題の平均でわずかに短縮し、課題別には悪化もある。目標画像・環境の成功ラベル・状態入力方策という条件を含め、[小目標と振り返り §5–6](llm_rl_goals_and_reflection_2026-11-07.md)へ記録した。

## 8. 調査の限界

- 学習、GPU使用量、推論遅延、API費用を独立に測定していない。公開コードの設定と、著者の報告を区別した。
- VLM-AR3Lの小型モデルは実課題に応じて成績が変わる。今回の確認でAtariへの転用の成功例は得られていない。
- SCOUTはLLM方策を育てる方向であり、既存のRL方策をLLMで助ける主題の隣接研究として含めた。これをDQNへの追加機構として分類しない。
- EARSの仮想特徴の実現可能性、TACと実際の最適化後の行動の関係は、追加で調べる必要がある。合成ラベルの構成条件と自然言語による指定条件を分けて再現する必要もある。
- DERLの小型提案器は限られた式の言語を使う。任意のPython報酬を0.5Bで設計できるとの証拠ではない。数学課題に限定された追加ablationの結果を、外部制御へ転用していない。
- [磁気マイクロロボットへの応用（MARSS 2025）](https://research.aalto.fi/en/publications/application-of-large-language-models-in-magnetically-manipulated-/)も候補として見つけたが、著者稿PDFが403で取得できなかったため、モデル規模の比較や成績を本冊の証拠には加えていない。
