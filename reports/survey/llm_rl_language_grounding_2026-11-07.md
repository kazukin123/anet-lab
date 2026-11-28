# LLMの言葉をRLの出来事と行動へ結び付ける

調査日: 2026-11-07（JST）  
対象: 非同期の報酬注釈、形式言語への翻訳、経験で修正するサブゴール依存関係。  
関連: [再サーベイ本文](llm_assisted_rl_resurvey_2026-11-07.md)、[前回調査](llm_prior_knowledge_rl_survey_2026-10-18.md)、[探索に渡す情報](llm_rl_exploration_information_2026-11-07.md)。  
状態: ONI、RLangへの自然言語翻訳、DECKARDの3論文を追加確認。ONIは固定commitも確認。学習・モデル取得・LLM API呼び出しは未実行。

## 目次

1. [1. 言葉が正しいだけでは足りない](#1-言葉が正しいだけでは足りない)
2. [2. ONI：採点を待たずに学習を進める](#2-oni採点を待たずに学習を進める)
3. [3. ONIの固定実装で確認した境界](#3-oniの固定実装で確認した境界)
4. [4. RLang：助言を報酬以外にも翻訳する](#4-rlang助言を報酬以外にも翻訳する)
5. [5. DECKARD：依存関係を経験で確かめる](#5-deckard依存関係を経験で確かめる)
6. [6. Atariへ持ち込む前に区別すること](#6-atariへ持ち込む前に区別すること)
7. [7. 書誌と未確認事項](#7-書誌と未確認事項)

## 1. 言葉が正しいだけでは足りない

「青い鍵で青い扉を開ける」という助言をRLへ渡す場合、少なくとも三つの対応が要る。どの観測が青い鍵なのか、手持ちの行動でどう近づくのか、どの変化を開いた成功と判定するのか、である。本冊でいうgroundingは、このように言葉と環境内の対象・操作・判定を結び付けることを指す。

| 方式 | 言葉を何へ変えるか | 環境側に必要な入口 |
|---|---|---|
| ONI | 出来事の有用性ラベル、そこから学ぶ報酬 | NetHackのメッセージ |
| RLangへの翻訳 | 方策・計画・報酬・遷移についての部分的な知識 | 対象名、述語、利用できるskillの辞書 |
| DECKARD | アイテムの前提関係を表すグラフ | inventoryの項目と数量、採集方策、craft操作 |

これは3研究の比較のための整理である。LLMの文章を受け取る処理が動いても、入口が正しく対応していなければ助言は作用しない。[前回のローカルMiniGrid pilot](llm_assisted_rl_resurvey_2026-11-07.md#73-policy-invariant-reward-shaping-from-llm-feedback)で見つかった語彙不一致は、この問題の一例だった。

## 2. ONI：採点を待たずに学習を進める

ONIは、方策が集めたメッセージの一部をLlama-3.1-8Bに非同期で注釈させる。ラベルの直接検索、分類モデルへの蒸留、ペアの順位学習を比較する。NetHackの4課題を2B step・5 seedで評価し、事前の大規模データ収集を必要とするMotifに近い性能を報告する。Oracleは報酬ハックの既知の問題を含み、真のゲーム攻略と同一視できない。[ONI v4 §3–5](https://arxiv.org/html/2410.23022v4)

```text
hosted on a separate node

別のノード上に配置
```
[ONI v4 §3.1](https://arxiv.org/html/2410.23022v4)

論文ではRL側とLLM側を分ける。RLの基準約32k step/sに対して、注釈の直接検索は約30k、分類モデルを同時学習する場合は約26k。注釈は全観測のごく一部で、直近の候補を優先する。70Bとの比較で8Bが十分だったのはScore・Oracleの条件であり、全ゲームへの小型化保証ではない。[ONI v4 §3.1、Appendix C.1](https://arxiv.org/html/2410.23022v4)

本調査では、これを「LLMを頻繁に呼んでも無料」という結果とは読まない。遅い注釈器を速い方策学習から切り離す代わりに、注釈の少なさと遅れを受け入れる方式である。1台のGPUで共有した場合の競合や、文字メッセージがない環境での画像認識費用は、上記のstep/sからは分からない。

また、モデルの採点を減らした試験は、取得済み注釈を間引く条件である。少ない問い合わせだけを実際に実行して同じ結果を得ることとは、費用上は区別する必要がある。[ONI v4 §5.3](https://arxiv.org/html/2410.23022v4)

## 3. ONIの固定実装で確認した境界

確認commitは `facebookresearch/oni@f0f86e8abea877e045e74f4bf786d246df0638a4`（2024-12-17）。論文v4より古いため、論文の全条件との一致は未確認である。READMEの例もGPUを2枚使い、LLMとRLを分ける。[固定README](https://github.com/facebookresearch/oni/blob/f0f86e8abea877e045e74f4bf786d246df0638a4/README.md)

| 調べた箇所 | 確認したこと | 意味 |
|---|---|---|
| `llm_reward_type` | `online_cls`はretrieval、`online_cls_reward_model`はclassification | 名前だけから分類モデルの有無を決められない |
| LLM worker | 蓄積した候補の末尾からbatchを取り出す | 未処理候補を到着順にすべて注釈する設計ではない |
| 分類prompt | 応答に`knowledge`、`analysis`、`label`を要求 | ラベル以外の文にも文字列が現れうる |
| `get_cls_label` | 応答全体に`FOO`があれば1、なければ`BAR`を探す | 最終`label`要素だけを読む実装ではない |
| 同parser | どちらもなければ警告して0 | 解析失敗と否定ラベルが同じ数値になる |

出典: [workerとparser](https://github.com/facebookresearch/oni/blob/f0f86e8abea877e045e74f4bf786d246df0638a4/rl_baseline/sample-factory/sample_factory/algorithms/appo/llm_annotator.py)、[prompt](https://github.com/facebookresearch/oni/blob/f0f86e8abea877e045e74f4bf786d246df0638a4/rlaif/prompts.py)。

たとえば、説明に`FOO`への言及があり、最後が`<label>BAR</label>`で終わる応答でも、このparserの分岐は1を返す。これは静的に構成した例で、公開実験で何件発生したかを測ったものではない。論文の成績を否定する根拠ではなく、モデルを取り替えて再現する際に採点経路を確認する理由となる。

学習中に追加されるラベルを使う場合、同じ出来事の扱いが時間とともに変わりうる。未注釈だったことと、モデルが無益と判断したことも異なる。本調査としては、生成された文章、解析したラベル、学習へ渡した値を区別して読むことが必要だと考える。

## 4. RLang：助言を報酬以外にも翻訳する

Spiegelらは、自然言語の助言をRLangという形式言語へ翻訳し、tabularのDyna-Qへ渡す。報酬だけでなく、方策・計画・遷移の部分知識を扱う。使用可能な対象・述語・skillの辞書を先に渡し、gpt-3.5-turbo-instructで翻訳する。主実験はMiniGridの4課題とVirtualHomeの2課題で、各10 instanceを評価する。[著者公開のRLC 2024 workshop版 §3–4、Appendix E](https://cs.brown.edu/people/gdk/pubs/nl2rlang_tafm.pdf)

```text
semantically-meaningful labels

意味の分かるラベル
```
[RLang翻訳論文 §4.3](https://cs.brown.edu/people/gdk/pubs/nl2rlang_tafm.pdf)

対象へ歩く`go_to()`などは事前に与える。VLMによる対象ラベル付けは追加のデモであり、知覚からskillまでの完全自動獲得ではない。利用者の助言10件では1件が存在しない対象を参照し、有効なprogramでも必要な概念が辞書にないため手順が欠ける例があった。[同 §4.1、§4.3–4.4](https://cs.brown.edu/people/gdk/pubs/nl2rlang_tafm.pdf)

本調査で重要と考えるのは、文の内容によって渡す先が違うことである。「扉を開けよ」は行動についての助言だが、「閉じた扉には通れない」は環境の遷移についての知識である。後者まで正の報酬へ押し込むと、本来何を伝えたかったかを見失う。この研究は、その違いを明示的に扱う比較対象になる。

ただし、辞書を渡したこと自体を誤参照の防止保証とは呼べない。上記の利用者実験に反例があるためである。また、当初の助言が間違っていた場合にどこまで修正できるかは、本調査では確認し切れていない。

検索にはICLR 2026投稿版も現れたが、OpenReviewが確認画面へ転送されたため本文・採否を確認していない。本節は読めた著者公開のworkshop版に限定し、ICLR採択論文とは記載しない。[著者の論文一覧](https://benjaminaspiegel.com/publications/)、[workshopの採択一覧](https://sites.google.com/view/tafm/accepted-papers)

## 5. DECKARD：依存関係を経験で確かめる

DECKARDはMinecraftのアイテム依存関係をLLMで一度生成し、探索中の経験で確認・修正する。LLMは学習中や実行中には呼ばない。subgoal用の方策はVPTからadapterを学び、inventoryで切り替える。LLMの主な効果は、不要なsubgoalの学習を減らすことにある。LLMなし版と最終成功率は同じと記され、探索の効率改善と最終能力を区別する。[ICML 2023本文 §4–6、Figure 4](https://royf.org/pub/pdf/Nottingham2023DECKARD.pdf)

```text
only deterministic transitions

決定的な遷移だけ
```
[DECKARD §7](https://royf.org/pub/pdf/Nottingham2023DECKARD.pdf)

前提は強い。高レベルのcraft操作、inventory、学習済みVPTを使う。付録にはMineCLIPの報酬整形、各subgoalを一つのworld seedで学習すること、必要な道具を初期所持させることも記される。評価はランダムなworld seed。抽象化の自動獲得や確率的な依存関係は対象外である。[DECKARD §5.3、§7、Appendix B.2–B.3](https://royf.org/pub/pdf/Nottingham2023DECKARD.pdf)

したがって、Atariの任意の画面から出来事と前提関係を自動発見できる根拠にはならない。一方、生成した依存関係を最初から真実とせず、環境で確認した部分と未確認の部分を分けて探索する考え方は参考になる。この最後の点は本調査による方法上の評価である。

誤りへの頑健性の実験も、任意の幻覚を網羅したものではない。正しい依存グラフへ所定の辺の追加・削除を加える条件である。実行できないskill名や、観測できない出来事の生成まで直せるとは読まない。[DECKARD §6.3](https://royf.org/pub/pdf/Nottingham2023DECKARD.pdf)

## 6. Atariへ持ち込む前に区別すること

以下は本冊と既調査の結果を基にした、検証対象の整理である。anet-labの変更計画や実験開始の合意を表さない。

| 層 | 例 | 確かめたいこと |
|---|---|---|
| 知識 | 「鍵を取ってから扉へ行く」 | ゲームの規則として正しいか |
| 観測 | 「鍵を取得した」 | 実際に取得した時点を認識したか |
| 翻訳 | `has_key`という条件 | 現在の環境の語彙で評価できるか |
| 採点 | ラベルや数値への変換 | 解析失敗や未判定を成功と混同していないか |
| 行動 | 扉へ到達する方策 | 手持ちの行動で実行できるか |
| 学習 | 補助報酬・探索分布・更新式 | 元の課題の成績を改善したか |
| 費用 | 注釈、知覚、方策学習 | 同じ総費用で改善したか |

たとえばボス戦で必要なのが相手の動作タイミングなら、正しい攻略文があっても、観測がその動作を区別できなければ効かない。逆に、出来事を正確に取得できても、同じ出来事を繰り返して得点できる補助報酬なら、本来の進行を妨げる可能性がある。前回のMotif・ELLMの失敗例と、今回の[報酬整形の反例](llm_reward_shaping_guarantees_2026-11-07.md)は、それぞれ別の層の問題として読む。

LLMの大きさを選ぶ前に、どの層を担当させるのかを決めることが重要である。ONIの8Bの採点能力、iLLMの280Mの表現・予測課題、コード生成に使う大型モデルは、同じ能力の大小比較ではない。[小型モデルの別冊](llm_rl_small_models_and_costs_2026-11-07.md)、[iLLMの補足資料の確認](llm_rl_exploration_information_2026-11-07.md#3-illmの補足資料で更新された実行条件)

## 7. 書誌と未確認事項

| 研究 | 日付・公刊状況 | 著者・所属・役割 | 引用件数の観測 |
|---|---|---|---|
| ONI | 初出2024-10-30。RLJ 6:139–164、RLC 2025。確認本文はarXiv v4、2025-10-23 | Qinqing Zheng・Mikael Henaffが同等貢献、Amy Zhang、Aditya Grover、Brandon Amos。Zheng/Henaff/AmosはMeta、ZhangはMeta・UT Austin、GroverはUCLA | OpenAlex W4404342981: 4 |
| Informing Reinforcement Learning Agents by Grounding Language to Markov Decision Processes | RLC 2024 Training Agents with Foundation Models workshop版。ICLR 2026投稿の本文・採否は未確認 | Benjamin A. Spiegel、Ziyi Yang、William Jurayj、Ben Bachmann、Stefanie Tellex、George Konidaris。Brown University。Spiegelが先頭、個別貢献は未確認 | 題名検索で一致レコードを取得できず。0とはしない |
| DECKARD | 初出2023-01-28。ICML 2023、PMLR 202:26311–26325 | Kolby Nottinghamが第一・連絡著者、Prithviraj Ammanabrolu、Alane Suhr、Yejin Choi、Hannaneh Hajishirzi、Sameer Singh、Roy Fox。UC Irvine、AI2、University of Washington | W4318751399: 22 |

書誌: [ONI公刊情報](https://rlj.cs.umass.edu/2025/papers/Paper6.html)、[ONI公刊PDFの所属](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_6.pdf)、[RLang著者公開PDF](https://cs.brown.edu/people/gdk/pubs/nl2rlang_tafm.pdf)、[DECKARD公刊情報](https://proceedings.mlr.press/v202/nottingham23a.html)。引用件数は2026-11-07のOpenAlex観測値であり、影響力の確定値ではない。[ONI](https://openalex.org/W4404342981)、[DECKARD](https://openalex.org/W4318751399)

今回の追加検索は「LLM grounded subgoal」「online intrinsic rewards」「language to MDP」「DECKARD」。2026年のLOGO、LLMBAなども候補に現れたが、本文確認に達していないため本調査の確認済み論文には含めない。LOGOの出版社ページは403だった。制限の回避は行わず、摘要だけから条件や効果量を補わなかった。

追跡検索では、RLangのICLR 2026投稿版の公開索引に、VLMで環境の構造を認識して語彙ファイルを作る追加実演の記述があった。しかしPDF本体はブラウザー確認画面に移るため取得できず、実演の課題・精度・人の修正量は未確認である。§4の「語彙を先に渡す」という説明は精読した2024年workshop版の実験条件に限り、後続版でも語彙の自動化を試みていないという主張にはしない。表紙は投稿中の表示で、採択済みとも判断しない。[ICLR 2026投稿版の索引で確認した摘要・表紙](https://openreview.net/pdf/5aad3ce5521332cc018580e7b93af3086eeb8d18.pdf)

ONIのparser指摘は静的な構成例で、実データでの発生率は未測定。RLangとDECKARDのコードは今回読んでいない。いずれもAtariのKungFuMasterボスやQbert第8面の改善を直接示す追加証拠にはなっていない。
