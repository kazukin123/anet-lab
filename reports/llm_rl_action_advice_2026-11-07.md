# LLMの行動助言をRLのどこへ接続するか

調査日: 2026-11-07（JST）  
対象: 行動事前分布、観測に加えるヒント、行動の実行可能性を用いる3論文。  
関連: [全体の読み方](llm_rl_reading_guide_2026-11-07.md)、[小目標と振り返り](llm_rl_goals_and_reflection_2026-11-07.md)。  
状態: 一次資料の読解。IEEE論文は公開検索で取得できた本文と大学の書誌を用い、PDF全体は未確認。学習実験は実施していない。

## 目次

1. [1. 同じ助言でも接続先が違う](#1-同じ助言でも接続先が違う)
2. [2. LLM Priors：候補の中で価値を学ぶ](#2-llm-priors候補の中で価値を学ぶ)
3. [3. 観測ヒント：従うかどうかを方策が学ぶ](#3-観測ヒント従うかどうかを方策が学ぶ)
4. [4. APM：行動の直後の効果を学んで文章にする](#4-apm行動の直後の効果を学んで文章にする)
5. [5. 候補数とcacheの小さな計算例](#5-候補数とcacheの小さな計算例)
6. [6. 正確な価値があっても事前分布は残る](#6-正確な価値があっても事前分布は残る)
7. [7. 比較で揃える条件](#7-比較で揃える条件)
8. [8. 書誌と未確認事項](#8-書誌と未確認事項)

## 1. 同じ助言でも接続先が違う

「右へ動くとよい」という助言には、少なくとも三つの渡し方がある。方策に参考情報として見せる、探索する行動の確率を変える、学習で比べる行動の集合自体を狭める、である。最後の方法では、行動を選ぶ段階だけでなく、価値の更新でも除かれた行動を比較しなくなる。

| 方式 | LLMへ渡すもの | RLへの接続 | 読むときの注意 |
|---|---|---|---|
| LLM Priors | 文章の観測と利用可能な行動 | 候補分布、価値更新の候補、KLによる制約 | 有限個の候補と元の全行動を区別する |
| 観測ヒント | ASCIIの状態、課題、過去の行動 | 推奨行動と有無を観測へ追加 | 無視できる構造と、実際に無視を学べる証拠は別 |
| Agent-Specific Prompt Engineering | 学習した行動の効果 | ε探索時の行動選択 | 効果の教師と事前学習の費用が必要 |

この表と以降の接続先の分類は、本調査による比較の整理である。

## 2. LLM Priors：候補の中で価値を学ぶ

ICLR 2025の本論文は、固定したLLMを行動の事前分布として使う。DQN-PriorとCQL-Priorは主にQwen1.5-7B、OvercookedのSalad課題では14Bが出す候補から行動を選び、次状態の価値更新も候補に限定する。状態・行動の表現にはBERTを使う。方策側の実装ではFlan-T5 SmallにKL項を加える。[公刊論文 §3–4、7頁のLLM Setting](https://proceedings.iclr.cc/paper_files/paper/2025/file/797be96e4481c3fe5d675c1ba5352969-Paper-Conference.pdf)

```text
uses cached action candidates for previously seen observations

以前に見た観測には、保存済みの行動候補を使う
```
[LLM Priors Appendix 9.4（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/file/797be96e4481c3fe5d675c1ba5352969-Paper-Conference.pdf)

主評価は文章化したALFWorld、Overcooked、Frozen Lake。自由文を規則で行動へ写す方式を採用する。Saladのoffline比較の約1,000対12,000件は、実際には1,186対11,939遷移で、良い例と悪い例の構成比も異なる。Atari frame数や総費用の90%削減ではない。[§4、Appendix 9.1–9.3、Table 11](https://proceedings.iclr.cc/paper_files/paper/2025/file/797be96e4481c3fe5d675c1ba5352969-Paper-Conference.pdf)

cache比較はTomatoの1 training seed。逐次問い合わせ31.1分・2,933回から13.8分・434回へ短縮したが、GPU使用量は約28,340 MB。計算機は40 GB A100を2枚搭載する。24 GB単一GPUの実証とは読めない。候補数を増やす比較もあるが、理論の極限はKL正則化付き目的である。[Proposition 1、Appendix 9.3–9.4、Table 12](https://proceedings.iclr.cc/paper_files/paper/2025/file/797be96e4481c3fe5d675c1ba5352969-Paper-Conference.pdf)

本調査の解釈では、候補の再利用は費用と提案の揺れを減らす一方、最初の提案の偏りを保持する。この条件を§5で、論文の性能値とは独立に計算する。

実行時だけ候補生成器を小さくする比較もある。Table 2では7B・5候補で学習したQを固定し、評価時に4Bへ替える。ALFWorld Pickの未見課題でCQL-Priorは4B・5候補が0.17、7B・5候補が0.46であり、4B単独の1候補は0だった。同じ小型候補生成器でも学習した価値を使う利点を示すが、4Bだけで学習した結果や、7Bと同等の成績ではない。さらに候補数を増やすと、学習中に見なかった状態・行動の組が混じり、成績は必ずしも単調に上がらない。§5の包含確率と、学習済みQでの実際の選択精度を区別する。[公刊論文 Table 2、§4.3.1、8頁](https://proceedings.iclr.cc/paper_files/paper/2025/file/797be96e4481c3fe5d675c1ba5352969-Paper-Conference.pdf)

## 3. 観測ヒント：従うかどうかを方策が学ぶ

JainとGrossmannは、Llama3-70bが出す推奨行動とヒントの有無をPPOの観測に追加する。LLMはASCII状態・課題・直近5行動を受け取る。5または10 stepごとに問い合わせ、他のstepでは中立値と「なし」を渡す。[arXiv v1 Methodology、Appendix](https://arxiv.org/html/2510.08779v1)

```text
computational cost of frequent LLM queries

頻繁なLLM問い合わせによる計算費用
```
[観測ヒント論文 Conclusion and Discussion（2025/10）](https://arxiv.org/html/2510.08779v1)

3種類の単一部屋BabyAIを3Mまたは5M frame学習する。PickupLocの最終成功率は17.2%から29.5%で、差は12.3ポイント、相対改善は約71%。GoToObjの50%到達は1.08Mから120K frameへ減るが、9倍という値は実時間ではない。課題文だけ追加した比較もある。[Table 2–3](https://arxiv.org/html/2510.08779v1)

30件の助言を人が調べた結果と、学習後の成功率は異なる測定である。付録の小目標ヒントは、正解の計画器由来でも改善を示していない。本文は問い合わせ費用を限界として挙げるが、本調査ではseed数・GPU条件・総問い合わせ時間を確定できなかった。[Results、Appendix](https://arxiv.org/html/2510.08779v1)

参考情報にとどめれば、誤りから逃れる選択肢は残る。しかし、選択肢が存在することだけでは、有限の経験で無視する判断を獲得できるとは言えない。「観測へ入れるだけなら害はない」という保証として扱わない。

## 4. APM：行動の直後の効果を学んで文章にする

Guらは、キャラクターごとにAction Possible Model（APM）を事前学習し、その出力を文章にしてLLMへ渡す。LLMの選択はε探索の分岐で使う。Street Fighter IIIの15行動ごとにモデルを持ち、直後の状態変化を所定のクラスへ対応させる。割引率は0。長期の勝敗価値を予測するものではない。[IEEE本文 §III・IV-B](https://doi.org/10.1109/ACCESS.2026.3683466)

```text
initial exploration performance under short-term training settings

短期間の学習条件での初期探索性能
```
[Agent-Specific Prompt Engineering Introduction（2026/04）](https://doi.org/10.1109/ACCESS.2026.3683466)

対戦相手を制御した事前学習を使い、0と分類した行動を候補から除く。行動名を抽象ラベルへ変え、候補順序も無作為化する。Reacherでは距離変化、MiniGridでは状態変化の有無をクラスに使う。追加二環境の比較は300学習episodeと100評価episodeである。[§IV-B–D・G](https://doi.org/10.1109/ACCESS.2026.3683466)

追加確認した§IV-Hでは主モデルはGPT-4o-mini。ローカルLlama-3-8Bの比較は行動の効果クラスの一致であり、モデルを替えたRL学習の成績と同一視できない。同期実行でAPIは約3〜6秒、Intel i5のCPU上の8Bは約30秒/応答と記し、ゲーム内100秒の最大200問い合わせで最大約20分・100分と見積もる。[IEEE本文 §IV-H](https://doi.org/10.1109/ACCESS.2026.3683466)

公開検索で取得できた本文を確認したが、IEEEのPDF全体は取得できていない。勝率「19.3%改善」の分母、独立学習seed、APM事前学習の総費用は未確定として残す。長期学習後の成績や画像だけのAtariへ一般化しない。[大学の書誌](https://pure.dongguk.edu/en/publications/agent-specific-prompt-engineering-for-llm-guided-rl-exploration/)

この方式の研究上の問いは、「数値をLLMへ直接読ませるか」だけではない。どの状態変化を有用なクラスとするか、その判定をどのデータで学ぶかも含む。距離が減ることと最終的に目的を達成できることは、一般には同じではない。

## 5. 候補数とcacheの小さな計算例

以下は本調査による説明用の計算であり、論文の測定値ではない。ある状態で最良の行動をLLMが一回に提案する確率を\(p\)とし、独立に\(k\)回提案すると仮定する。少なくとも一回含まれる確率は、

\[
P(\text{候補に含む})=1-(1-p)^k.
\]

| 仮定した一回の確率 | 候補数 | 最良行動を一度以上含む確率 |
|---|---:|---:|
| 0.1 | 1 | 10% |
| 0.1 | 5 | 40.951% |
| 0.1 | 10 | 65.132% |
| 0.1 | 20 | 87.842% |

候補を増やせば改善するが、5候補でも半数以上は最良行動を含まない。この計算には独立性が必要で、同じ候補を重ねて出すような相関があれば、そのまま適用できない。

最初の候補集合を同じ観測に対して永久に使い、選択も更新もその集合内に限るとする。この追加条件の下では、最初に欠けた行動は、Qの推定を正確にしても集合へ戻らない。新しい独立候補を毎訪問で作るなら、\(n\)訪問後の一度以上の包含確率は\(1-(1-p)^{kn}\)だが、同じcacheを使うなら\(1-(1-p)^k\)のままである。

これはcache一般の欠点を断言するものではない。良い候補を安定して再利用する利益もある。意味が異なる二状態が同じ文章になるなら、cacheのキーが十分な情報を保持しているかという別の問いも生じる。完全観測か部分観測か、履歴をキーへ入れるかを区別する必要がある。

## 6. 正確な価値があっても事前分布は残る

本節も本調査の計算例である。§2の論文の極限分布は、正規化定数を除けば\(q(a)\propto p(a)\exp(Q(a)/\alpha)\)となる。論文自身がKL付き目的として述べており、元の無制約目的の最適方策と同じだという主張ではない。[Proposition 1、Appendix 8.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/797be96e4481c3fe5d675c1ba5352969-Paper-Conference.pdf)

二行動で\(Q(\mathrm{good})=1\)、\(Q(\mathrm{bad})=0\)、\(p(\mathrm{good})=0.01\)、\(\alpha=1\)なら、

\[
q(\mathrm{good})=\frac{0.01e}{0.01e+0.99}\simeq0.02672.
\]

価値が正確でも、事前分布に強く逆らう最良行動の確率は約2.67%にとどまる。\(\alpha\)を小さくすれば価値の影響は強くなるが、\(p=0\)の行動はこの分布では0のままである。これは二行動の固定価値についての計算で、関数近似や有限データ下の収束定理ではない。

したがって、助言を使わない探索経路を残す場合と、候補集合から完全に除く場合を分けて読むことが重要になる。誤った助言を直すのに「もっと学習すればよい」のか、候補を作る仕組みを変える必要があるのかが異なる。

## 7. 比較で揃える条件

| 問い | 比較で分けるもの | 何が分かるか |
|---|---|---|
| 行動の意味が必要か | LLM提案、同数の無作為候補、同じ情報の規則 | 候補数削減と事前知識の効果 |
| 学習目標を変えたか | 行動選択だけ、更新対象も限定、KLも追加 | 探索と目的の変化 |
| 助言の誤りから回復できるか | 誤った候補を固定、再生成、全行動を選ぶ経路 | 誤りの持続と修正 |
| ヒントの抽象度が合うか | 操作、小目標、正解の小目標 | 推論精度と学習しやすさ |
| 費用を減らせたか | 同step、同実時間、cache構築を含む全費用 | 相互作用の節約と総計算の節約 |

これは今後の研究を読むための比較軸であり、anet-labへの実装計画ではない。DQNという名前が共通でも、文章状態・可変の行動候補と、画像・固定の離散操作では、入力と行動の意味づけを作る負担が異なる。

## 8. 書誌と未確認事項

被引用数は2026-11-07にOpenAlexの同名レコードを照合した値で、全サービスの総数ではない。新しい研究の0件は品質の評価を意味しない。

| 論文 | 著者・所属・役割 | 公開・位置づけ | OpenAlex |
|---|---|---|---|
| Efficient Reinforcement Learning with Large Language Model Priors | Xue Yanが第一著者、Haifeng ZhangとJun Wangが責任著者。CASIA・UCAS、UCL、Bristol、Huaweiの共同研究 | 初版2024-10-10。ICLR 2025公刊25頁版を本文の根拠とした | [4件、preprintレコード](https://openalex.org/W4403365060) |
| Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations | Vaibhav Jainが第一著者、DFKI・Saarland。共著Gerrit GrossmannはDFKI。責任著者の別記は未確認 | arXiv v1、2025-10-09。紙面のAAAI著作権表示だけで本会議採択とは判断しない | [0件](https://openalex.org/W4417117819) |
| Agent-Specific Prompt Engineering for LLM-Guided RL Exploration | Bonwoo Guが第一著者、Jinseok Oh、Yunsick Sung。Dongguk University。責任著者の記載はPDF全体を含む再確認が必要 | IEEE Access 14、60963–60983、2026-04-13公開 | [0件](https://openalex.org/W7154174003) |

所属と役割は公刊論文冒頭、arXivの著者情報、[Donggukの書誌](https://pure.dongguk.edu/en/publications/agent-specific-prompt-engineering-for-llm-guided-rl-exploration/)による。行動事前分布の全著者はXue Yan、Yan Song、Xidong Feng、Mengyue Yang、Haifeng Zhang、Haitham Bou Ammar、Jun Wang。

3研究とも、本調査では公式の実験コードを固定commitで検査できていない。候補抽出の失敗時挙動、cacheの実際のキー、ヒントを除く評価条件などを実装で追認したとは扱わない。§5–6の計算は独立した有限例として[検算スクリプト](tools/check_llm_action_advice.py)で確認する。
