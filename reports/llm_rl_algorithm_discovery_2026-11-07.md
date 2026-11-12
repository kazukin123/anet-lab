# LLMでRLの更新則を発見する研究

調査日: 2026-11-07（JST）  
対象: LLMが報酬や行動だけでなく、方策の更新式、optimizer、ゲーム解法、学習コードを提案する研究。  
関連: [再サーベイ本文](llm_assisted_rl_resurvey_2026-11-07.md)、[生成した制御と環境](llm_generated_control_and_environments_2026-11-07.md)、[小型モデルと費用](llm_rl_small_models_and_costs_2026-11-07.md)。  
状態: 3論文の本文・付録と2リポジトリの固定commitを確認。生成コードの実行・学習・API呼び出しは行っていない。

## 目次

1. [1. 何を発見させるか](#1-何を発見させるか)
2. [2. How Should We Meta-Learn：単純な式と大きな更新器の違い](#2-how-should-we-meta-learn単純な式と大きな更新器の違い)
3. [3. 公開LPO実装の探索条件](#3-公開lpo実装の探索条件)
4. [4. Multiagent Learning：生成後の人による絞り込み](#4-multiagent-learning生成後の人による絞り込み)
5. [5. DiscoGen：発見する課題を生成する](#5-discogen発見する課題を生成する)
6. [6. 公開DiscoGenの評価境界](#6-公開discogenの評価境界)
7. [7. アルゴリズム探索の比較条件](#7-アルゴリズム探索の比較条件)
8. [8. 書誌と調査の限界](#8-書誌と調査の限界)

## 1. 何を発見させるか

この分野の出力は、学習済みの行動方策とは限らない。たとえば「方策を大きく変えすぎないための罰則の式」を生成し、その式を使って複数の方策を学習する。外側では式を探索し、内側では通常のRLを実行する。したがって、環境を1回学習する費用と、よい式を見つけるまでの総費用を分ける必要がある。

| 探索対象 | 生成後に行うこと | 近い研究 |
|---|---|---|
| 方策更新の罰則・パラメータ更新器 | 生成した関数でRLを学習 | How Should We Meta-Learn |
| regretの蓄積、戦略の混合方法 | ゲーム木で相手への最適応答を反復 | Multiagent Learning |
| 損失・NN・replay・学習loopなどの指定部分 | 固定した残りのコードへ組み込んで評価 | DiscoGen |
| Actor-CriticやTDを使わない探索法 | 制約内で新しい学習方法を進化 | [Evolutionary Discovery](llm_assisted_rl_resurvey_2026-11-07.md#6-学習則そのものを探索する) |

この表は調査上の整理である。生成箇所が違えば、失敗の意味も違う。損失だけの探索で悪い成績が出た場合と、学習loopを含めた生成でコードが動かなかった場合を、同じ「アルゴリズムの質」でまとめない。

## 2. How Should We Meta-Learn：単純な式と大きな更新器の違い

Goldieらは、LLMによるコード提案、数値的なblack-box探索、学習済み更新器の蒸留などを比較する。LLMはo3-miniで、PPOやSGD/Adamから開始する。少数入力の更新則では有力だが、入力や内部状態が多いOPENでは優位が一様でない。探索を始める式と、触ってよい係数の違いも比較条件に入る。[本文 §4–8](https://arxiv.org/html/2507.17668v2)

```text
single seed

単一のseed
```
[本文 §6](https://arxiv.org/html/2507.17668v2)

各アルゴリズムの発見は1回で、評価には16の環境seedを使う。したがって16回の独立した発見の実証ではない。さらにSGDから始める一部の比較は、評価環境ごとに学習率を調整するため、著者自身がfew-shotの条件とする。未見環境への無調整の転用とは区別される。[本文 §6、§7.2、§8](https://arxiv.org/html/2507.17668v2)

本調査での読み方は、「LLMなら任意の更新器をよくできる」ではなく、**入力と出力が小さく、既存式からの変化を評価しやすい箇所が、比較的扱いやすい**という条件付きの結果である。これは後述のDiscoGenの「変更可能な部分を増やすと失敗が増える」という観察とも整合するが、両論文の数値を直接合算した結論ではない。

## 3. 公開LPO実装の探索条件

確認commitは `AlexGoldie/learn-rl-algorithms@5c52f68641b330dafe3804873cd54265af3a8cec`（2025-09-07）。READMEはLPGを収録対象外と明記するため、論文全実験を単独で再現できるリポジトリとは呼ばない。[固定README](https://github.com/AlexGoldie/learn-rl-algorithms/blob/5c52f68641b330dafe3804873cd54265af3a8cec/README.md)

| 調べた箇所 | 静的に確認した挙動 | 読み取りへの影響 |
|---|---|---|
| `lpo_llm.py` | 既定30世代、Breakout、o3-mini。最初にPPOの罰則を評価 | warm startなしの発見ではない |
| 同 `validate_code` | 10個の入力に対しNaNと出力shapeを検査 | promptで指定する関数の条件をすべて機械検証するものではない |
| `eval_lpo_llm.py` の損失 | 生成関数へ`relu`を適用して罰則に使う | 生の生成式と、実際に学習へ適用する式を区別する |
| 同 `eval_func` | 既定8 run。各呼び出しで`PRNGKey(42)`から分割 | 候補ごとに新しい乱数系列を抽選する評価ではない |
| 同 fitness | 最後の記録の平均をPPO基準値で正規化 | 最良episodeだけを選ぶ評価とは異なる |

出典: [生成・事前検査](https://github.com/AlexGoldie/learn-rl-algorithms/blob/5c52f68641b330dafe3804873cd54265af3a8cec/algorithms/learning_algorithms/llm_discovery/lpo_llm.py)、[損失と評価](https://github.com/AlexGoldie/learn-rl-algorithms/blob/5c52f68641b330dafe3804873cd54265af3a8cec/algorithms/evaluation/eval_lpo_llm.py)。

事前検査は、有限性全般、`r=1`での値、同点での勾配を直接検査していない。`relu`で非負にはできても、同点で値と勾配が0になる条件まで自動的に満たすわけではない。これは公開コードの検査範囲の指摘であり、論文掲載のすべての生成式が条件違反だという判断ではない。

同じ乱数系列で候補を比較することには、候補間の偶然の差を減らす利点がある。ただし、探索の最後に別系列で評価しなければ、その系列への適応と更新式の一般性を分離できない。ここでの8 runという既定値を、論文の最終評価16 seedへ読み替えない。

## 4. Multiagent Learning：生成後の人による絞り込み

LiらはAlphaEvolveでCFRとPSROの部品を進化させる。CFRは各行動を選ばなかった後悔を蓄積して戦略を更新する方法、PSROは相手の戦略集団に対する最適応答を追加する方法である。v3では、生のVAD-CFR・SHOR-PSROから人が要素を削り、WOP-CFR・PM-PSROへまとめ直している。[v3 §3–5](https://arxiv.org/html/2602.16928v3)

実証は小規模ゲームの正確なゲーム木評価で、PSROの応答もvalue iterationによる。深層RLの学習成功を直接示す結果ではない。主評価は同系統のゲームへの転用が中心で、独立したAlphaEvolve探索は1回。人の選択を含み、収束保証もない。[v3 §4、§6 Limitations](https://arxiv.org/html/2602.16928v3)

```text
human-in-the-loop, not fully autonomous

人が判断に加わる方式であり、完全自律ではない
```
[v3 §6 Limitations](https://arxiv.org/html/2602.16928v3)

この結果から受け取れる方法上の示唆は、複雑な生成物を完成品として扱わず、要素を外して何が効いていたかを確かめることにある。本調査では、生成した式を単に簡潔に書き直す作業と、効果を測って不要な要素を除く作業を区別する。後者は別の探索なので、その試行回数と人の判断も記録対象になる。

v1の紹介だけを読むと、複雑な生のアルゴリズムが最終結論のように見える。ここでは2026-05-07のv3の結論を使い、AAMAS形式の原稿と版を混同しない。[版履歴](https://arxiv.org/abs/2602.16928)、[AAMAS形式の公開原稿](https://openreview.net/pdf?id=Sxp4F7HL7o)

## 5. DiscoGen：発見する課題を生成する

DiscoGenは、環境のステージではなく「どの学習部品を、どの環境で発見するか」という課題を生成する。DiscoBenchはその固定評価集合。候補の探索用と最終評価用の環境を分け、最終評価用コードは探索終了後に作る。さらに、探索役のpromptを改善する場合には、そのprompt自身にも別の評価課題が要る。[公刊本文 §1、§4–5](https://raw.githubusercontent.com/mlresearch/v306/main/assets/goldie26a/goldie26a.pdf)

実験は3種類の公開モデルを用いる。既存実装から始める条件は空の関数から始める条件より成功しやすいが、生成物も既存実装に近づく。成功した試行だけを集めた分析でも、全体では固定の既存手法が上回る。個別の成功例を「RL更新則発見の全般的優位」には広げられない。[公刊本文 §5、§8、Appendix I.2](https://raw.githubusercontent.com/mlresearch/v306/main/assets/goldie26a/goldie26a.pdf)

```text
35,000 GPU-hours

35,000 GPU時間
```
[公刊本文 Appendix C.5](https://raw.githubusercontent.com/mlresearch/v306/main/assets/goldie26a/goldie26a.pdf)

開発・未掲載実験を含む推定計算量が上記で、H200を使う。API費用は別に約1,600ドル。個別課題のAPI料金が小さくても、候補の学習費用を含めた探索が安いとは限らない。これは当時の報告値で、現在の価格の見積もりではない。[公刊本文 Appendix C.5](https://raw.githubusercontent.com/mlresearch/v306/main/assets/goldie26a/goldie26a.pdf)

研究対象の数が膨大という点は、独立した難問を同数実証したという意味ではない。設定の組合せ数、実行した探索数、発見した新しい仕組みの数は異なる。本調査では、課題生成器としての貢献と、そこから有用な更新則を安定して発見できるかを分けて評価する。

## 6. 公開DiscoGenの評価境界

確認commitは `AlexGoldie/discogen@98f98a4a6c2bc5a5db01e7d3e0dffbbce651f488`（2026-09-16）。現在のREADMEのOnPolicyRLには`loss, networks, optim, train, activation, targets, schedule`の7項目がある。公刊本文のTable 2は6項目、Appendix A.12の説明は4項目なので、資料間で列挙が一致していない。論文の組合せ実験を最新コードの7項目へ置き換えて解釈しない。[固定README](https://github.com/AlexGoldie/discogen/blob/98f98a4a6c2bc5a5db01e7d3e0dffbbce651f488/README.md)、[公刊本文](https://raw.githubusercontent.com/mlresearch/v306/main/assets/goldie26a/goldie26a.pdf)

損失だけを探索する公開設定は、探索用がMinAtarのBreakoutとFreeway、評価用がAsterixとSpaceInvaders。ほかの変更フラグはfalseである。ALEの画像Atari全般への転用評価ではない。[固定設定](https://github.com/AlexGoldie/discogen/blob/98f98a4a6c2bc5a5db01e7d3e0dffbbce651f488/discogen/discobench_configs/OnPolicyRL_loss.yaml)

`make_files.py`は固定ファイルのコピーと、編集対象をbaselineか空のtemplateから作る処理を分けている。評価を隔離するという考え方の実装上の入口は確認できるが、今回の読み取りだけで全呼び出し経路の隔離を検証したとはしない。[固定生成処理](https://github.com/AlexGoldie/discogen/blob/98f98a4a6c2bc5a5db01e7d3e0dffbbce651f488/discogen/utils/make_files.py)

この仕組みを参考にする場合、候補が変更できる範囲と、評価側が再生成する範囲を先に分ける必要がある。たとえば損失の探索中に環境wrapperや終了条件まで変われば、成績差を損失へ帰属できなくなる。この点はanet-labへの導入案の決定ではなく、研究比較上の条件である。

## 7. アルゴリズム探索の比較条件

以下は3論文と[既調査のEvolutionary Discovery](llm_assisted_rl_resurvey_2026-11-07.md#6-学習則そのものを探索する)を踏まえた、本調査の整理である。新しい実験結果ではない。

| 判断したいこと | 固定・記録する条件 | それだけでは答えられないこと |
|---|---|---|
| 生成した式がよいか | 同じ入力、同じ更新回数、同じ環境予算 | 探索手法としての成功確率 |
| LLM探索が再現するか | 独立した探索を複数回開始 | 固定した最良式の学習seedだけでは代替できない |
| 新しい環境へ移るか | 探索に見せない環境・seed | 評価先で係数調整した場合の無調整転用 |
| 構造が効いたか | 元の式、係数だけの探索、構造の探索を比較 | 既存式の再発見と新しい仕組みの区別 |
| 実用的に安いか | 失敗・修復・候補学習・選別・最終再学習の総費用 | 最後の学習1回の費用だけでは代替できない |
| 簡略化できるか | 要素を外した比較と、簡略化後の独立評価 | 見た目が短いことによる性能保証 |

とくに「試行が動いた」と「既存法よりよい」と「別環境でよい」は別の段階である。失敗した試行を除いた値を出すなら、除く前の成功率も併記する。探索の途中で見た評価環境を、最後に未見と呼ばない。

また、比較対象は既存のRLアルゴリズムだけでは足りない。係数のランダム探索、同じ部品集合の非LLM探索、生成済みの式の固定再生を置くと、LLMのどこに価値があったかを分けやすい。これは実施済みの評価ではなく、今後研究を読む・設計する際の比較軸である。

## 8. 書誌と調査の限界

| 論文 | 版・公刊状況 | 著者・所属・役割 | 引用件数の観測 |
|---|---|---|---|
| How Should We Meta-Learn Reinforcement Learning Algorithms? | arXiv 2507.17668 v2、2025-09-10。RLC/RLJ 2025 | Alexander D. Goldieが第一・連絡著者、Zilin Wang、Jaron Cohen、Jakob N. Foerster、Shimon Whiteson。Cohenは独立研究者、ほかはOxford。末尾2名は共同指導 | OpenAlex W4414885531: 1 |
| Discovering Multiagent Learning Algorithms with Large Language Models | arXiv 2602.16928 v3、2026-05-07。AAMAS 2026形式の原稿も公開。技術的結論はv3を採用 | Zun Li、John Schultz、Daniel Hennes、Marc Lanctot。全員Google DeepMind。Liが先頭。個別の貢献区分は今回未確認 | W7130726257: 0 |
| Procedural Generation of Algorithm Discovery Tasks in Machine Learning（DiscoGen） | ICML 2026、PMLR 306:35600–35715。公刊PDFとarXiv v2を照合 | Goldieがlead・連絡著者。Oxford、UC Santa Barbara、UCL、Wisconsin–Madison、Delft。公刊版は19名、arXiv/READMEは20名で著者列に差がある | W7138916761: 0 |

書誌の一次資料: [RLC公刊PDF](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)、[Multiagent版履歴](https://arxiv.org/abs/2602.16928)、[DiscoGen公刊書誌](https://proceedings.mlr.press/v306/goldie26a.html)。DiscoGenの公刊版ではWang・Hayler・Nathani・Toledo・Thampiratwong・Kaliszがcore contributor、Raileanu・Whiteson・Foersterがequal supervisionと記される。著者列が異なるため、公刊版の引用にはPMLRの書誌を使う。

引用件数は2026-11-07のOpenAlex検索結果であり、実際の影響力や受容の確定値ではない。0は検索サービスの当該レコードの値で、引用が存在しない保証ではない。[W4414885531](https://openalex.org/W4414885531)、[W7130726257](https://openalex.org/W7130726257)、[W7138916761](https://openalex.org/W7138916761)

「LLM RL algorithm discovery」「meta-learn reinforcement learning algorithms」「DiscoGen」の検索と原典の相互参照から追加した3論文で、網羅的なsystematic reviewではない。LLMを使わない学習則発見まで含めた優劣比較は行っていない。公開コードは指定ファイルの静的確認に限り、Windows・libtorchへの移植性やローカル小型LLMでの再発見性能は未検証である。
