# LLMが制御・世界モデル・探索・学習環境を作る研究

確認日: 2026-11-07（JST）  
対象: [LLMを活用したRLの再サーベイ](llm_assisted_rl_resurvey_2026-11-07.md)を補う2026年の5論文。LLM自身の数学・コード回答をRLで鍛える研究は対象外。  
方法: 一次論文と著者の公開コードを単独で調査。学習・モデル取得・有料API呼び出しは実行していない。  
状態: 各論文の実験条件と、公開成果物の再現範囲を確認した記録。anet-labへの採用決定ではない。

## 目次

1. [1. 対象と書誌情報](#1-対象と書誌情報)
2. [2. Code to Control：構造と数値を分けて制御器を作る](#2-code-to-control構造と数値を分けて制御器を作る)
3. [3. ESBM：得点以外の検証を方策の更新に使う](#3-esbm得点以外の検証を方策の更新に使う)
4. [4. Memento 3：世界モデルを文章とコードで保持する](#4-memento-3世界モデルを文章とコードで保持する)
5. [5. PDE：言語条件を変えて行動の候補を広げる](#5-pde言語条件を変えて行動の候補を広げる)
6. [6. PlayTrain：学習するゲーム自体を生成する](#6-playtrain学習するゲーム自体を生成する)
7. [7. 公開コードで確かめた再現の境界](#7-公開コードで確かめた再現の境界)
8. [8. 比べる前に揃えるもの](#8-比べる前に揃えるもの)
9. [9. 調査の限界と残る問い](#9-調査の限界と残る問い)

## 1. 対象と書誌情報

以下では「学習」を、勾配によるニューラル方策の更新と、外部のプログラム・記憶の更新に分ける。どちらも環境から学ぶが、同じ手法ではない。LLMの重みを固定したまま生成物を更新する研究も含めた。具体的な処理は各節の一次資料に対応する。

| 論文 | 著者・所属・役割 | 日付・掲載先 | 被引用数 |
|---|---|---|---:|
| [Code to Control](https://arxiv.org/abs/2609.38733) | Zergham Ahmed（第一著者、Harvard）、Joshua B. Tenenbaum（MIT）、Chris Bates（Harvard / Florida Institute for Human and Machine Cognition）、Samuel J. Gershman（Harvard） | 2026-09-30 v1、preprint | [0](https://openalex.org/W7215079953) |
| [Learning Explicit Behavioral Models with Adaptive Questions and World-Model Probes](https://arxiv.org/abs/2606.07127) | Hikaru Shindo（第一著者）ら8名。TU Darmstadtを中心にhessian.AI、DFKI、Centre for Cognitive Science | 2026-06-05 v1、preprint | [0](https://openalex.org/W7163881306) |
| [Memento 3](https://arxiv.org/abs/2610.11794) | Haoyu Zhao（第一著者、UCL）ら8名。UCL、Huawei Noah’s Ark Lab UK、University of Liverpool。Jun Wangが責任著者 | 2026-10-08 v1、preprint | [0](https://openalex.org/W7221131839) |
| [Prompt-Driven Exploration](https://arxiv.org/abs/2607.08837) | Sunshine Jiang（第一著者）ら9名。MIT、MIT-IBM Computing Research Lab、Improbable AI Lab | 初稿2026-07-09、確認版v4 09-18。[著者ページ](https://xinyunsunshine.github.io/prompt-rl/)はNeurIPS 2026と記載 | [0](https://openalex.org/W7168187889) |
| [PlayTrain](https://arxiv.org/abs/2609.09059) | Ryan Truong（第一著者、Harvard）、Lance Ying（Harvard / MIT）、Samuel J. Gershman（Harvard / Kempner）、Kazuki Irie（Yale CS / Wu Tsai） | 初稿2026-09-08、確認版v2 09-26、preprint | [0](https://openalex.org/W7212023350) |

所属は論文の著者欄、役割は著者順・責任著者表示による。PlayTrainの所属対応は[PDF p.1](https://arxiv.org/pdf/2609.09059)で確認した。被引用数は2026-11-07のOpenAlex API取得値で、研究の有効性や実際の総引用数を保証しない。未収録・収録遅れと、引用が存在しないことを区別する。PDEの採択表示は著者側の一次告知で、会議論文集の照合は未実施。

## 2. Code to Control：構造と数値を分けて制御器を作る

LLMがPythonの制御構造を作り、MuJoCoでは固定した特徴写像の数値パラメータをARSで合わせる。実行時は生成済み関数だけを呼ぶため、LLM推論と世界モデル内の探索が不要になる。[論文 §2](https://arxiv.org/html/2609.38733v1)

```text
require neither LLM inference nor planning at decision time

意思決定時にLLM推論も計画探索も必要としない。
```
[Code to Control（2026/09）, Abstract](https://arxiv.org/html/2609.38733v1)

Atariの入力はOCAtariの物体表現であり、画素から表現を学ぶ比較ではない。6ゲームの値は各run内の最良episodeを選んだ中央値。20M stepのPPOはPong・Breakout・Freewayで上回り、他3ゲームでは下回る。全ゲームでPPOを上回ったという解釈はできない。[同 §3.1–3.2](https://arxiv.org/html/2609.38733v1)

公開記録ではさらに、3 runとも環境seedは42で、差はLLMの生成にある。論文の「3 seeds」を、異なる初期環境での汎化評価と読まないことが重要である。保存済み制御器の再生と、同じ分布の制御器を再発見する実験も分ける。[著者のPROVENANCE](https://github.com/ZerghamAhmed/code-to-control/blob/a716dad972acf147d19f50d8b40e510fb02fff47/PROVENANCE.md#randomness)

## 3. ESBM：得点以外の検証を方策の更新に使う

ESBMは述語・重み付き規則・実行時間を限定したoption・機構の記憶を持ち、得点に加えて質問への回答と遷移予測を検証する。高得点でも世界の理解が誤っている状態を、次の修正へ返すのが狙いである。[論文 §4–6](https://arxiv.org/html/2606.07127v1)

```text
jointly evaluates task score, answerability and active world-model consistency

課題の得点、質問への回答可能性、能動的に検証した世界モデルの整合性を合わせて評価する。
```
[ESBM（2026/06）, Abstract](https://arxiv.org/html/2606.07127v1)

主実験はJAXAtariの3ゲームであり、Kangarooは最初のlevel・落下するヤシの実なし、Kingkongは最初のlevelだけという条件を持つ。評価は5 seeds・20,000 raw frames。厳密な同wrapper比較はPPO・EvoSymbol・ESBMに限られ、表中のDQN等には外部成績と局所adapterが混在する。通常のALE成績表と横並びにしない。[同 §7.1–7.2、Appendix A.6–A.7](https://arxiv.org/html/2606.07127v1)

質問と正解を作る側から回答側への漏洩、抽象状態の欠落、checkpoint復元の失敗が検証品質を左右する。QAの高さだけを一般的な理解の証拠にすることもできない。[同 §9、Appendix A.9](https://arxiv.org/html/2606.07127v1)

## 4. Memento 3：世界モデルを文章とコードで保持する

固定LLMが環境規則の仮説を文章のrulebookへ残し、それを実行コードへ変換する。観測した遷移との不一致から両者を修正する。既知の履歴に一致しても未知状態の正しさまでは決まらない、という問題を明示している。[論文 §1–3](https://arxiv.org/html/2610.11794v1)

```text
Agreement with observed transitions, however, does not guarantee correct generalisation.

観測した遷移への一致は、正しい汎化を保証しない。
```
[Memento 3（2026/10）, §1](https://arxiv.org/html/2610.11794v1)

外部制御の証拠として読むのはALE Pongの事例である。frame skip 4、学習時に+21へ到達したら停止し、その制御器を異なる開始状況3回で評価して全て21対0。評価時のLLM呼び出しはない。過去のRL手法の成績は各論文の条件であり、表のframe比を公平な総計算量比較には使えない。画面全体の再生には残差があり、ARCのcell単位の完全一致をPongへそのまま適用してはいない。[同 §5](https://arxiv.org/html/2610.11794v1)

これは世界モデルと制御器をコードとして獲得した事例であり、DQNに補助報酬を追加した実験ではない。この分類は本調査による整理である。

## 5. PDE：言語条件を変えて行動の候補を広げる

PDEは既存VLAへの指示文を変え、行動ノイズでは得にくい軌跡を探索する。動画を見たVLMが失敗を診断し、候補文を更新する。方策の重みの更新はPPOが担う。言語入力に応じた運動能力がすでにあることが前提で、通常のDQNへ文だけを渡しても成立しない。[論文 §3–5、§6](https://arxiv.org/html/2607.08837v4)

```text
PDE can only elicit capabilities already present in the VLA’s motor repertoire.

PDEが引き出せるのは、VLAがすでに持つ運動能力に限られる。
```
[Prompt-Driven Exploration（2026/09）, §6](https://arxiv.org/html/2607.08837v4)

実機の「約1時間」は128 rollouts相当のロボットデータ収集を指す。事前のSFTや全計算時間込みの所要時間ではない。本文v4は3課題平均を記す一方、確認日の著者ページは4課題と記すため、成績を混ぜない。[同 §5.6](https://arxiv.org/html/2607.08837v4)、[著者ページ（確認2026/11）](https://xinyunsunshine.github.io/prompt-rl/)

付録の10課題suiteではprompt pool構築60k＋PPO 15M step、8 H200・72.5時間。言語による追加処理が小さいことと、実験全体が小規模であることは別である。混合backpropagationは、単一prompt方策の不偏importance-sampling推定量ではなく、結合したsurrogateとして説明される。[同 Appendix B、C.1](https://arxiv.org/html/2607.08837v4)

## 6. PlayTrain：学習するゲーム自体を生成する

LLMがJavaScriptのゲームを作り、同じゲームを人間とRL方策が使う。QuickJSと専用描画処理で環境を実行し、学習器は別に動く。報酬設計だけでなく、問いに合わせて環境を増やす方向である。[論文 §2](https://arxiv.org/html/2609.09059v2)

```text
one node with four H100s and 92 CPU cores

H100を4枚、CPUを92 cores搭載した1 node。
```
[PlayTrain（2026/09）, Table 1](https://arxiv.org/html/2609.09059v2)

1.07M agent-steps/sはこのnode上で、Nature-CNN＋IMPALA、frame skip 1、64×64 RGB、24生成ゲームの幾何平均。PPOの同encoderは185kである。元のALE ROMを置き換えず高速化した値ではなく、生成したcloneの測定である。単core比較には描画解像度と呼び出し経路の差もある。[同 Table 1、Appendix E.1](https://arxiv.org/html/2609.09059v2)

cloneは中核的な仕組みを再現する一方、細部の一致を保証しない。生成物の確認・修正には人間も入る。したがって、生成した環境での成績から元ゲームの攻略性能や無人生成の成功率を直接導かない。[同 §2.1、§3.1、AI use statement](https://arxiv.org/html/2609.09059v2)

## 7. 公開コードで確かめた再現の境界

以下は静的な確認である。公開資料が「再現できた」と記す箇所を、今回こちらで再現したとは扱わない。

| リポジトリ | 確認したcommit | 読んだ範囲 |
|---|---|---|
| [Code to Control](https://github.com/ZerghamAhmed/code-to-control/tree/a716dad972acf147d19f50d8b40e510fb02fff47) | `a716dad972acf147d19f50d8b40e510fb02fff47` | README、PROVENANCE、latency測定 |
| [PlayTrain](https://github.com/heyodog0/playtrain/tree/1471481164034b023d2e19dd871def19a3e4ef06) | `1471481164034b023d2e19dd871def19a3e4ef06` | README、再現記録、runtime・生成器のvalidation |

Code to Controlは保存済みchampionを含み、再生はLLM呼び出しを必要としない。READMEが動作を確認した環境はPython 3.10・Apple siliconのmacOSであり、Windowsでの動作は本調査では未確認。[固定README](https://github.com/ZerghamAhmed/code-to-control/blob/a716dad972acf147d19f50d8b40e510fb02fff47/README.md)

再生と生成過程の忠実な再実行には差がある。PROVENANCEは4つのAtari cellで未commitの環境版を回収できないこと、MuJoCoの2課題で元promptログが上書きされていることを開示する。Flappy Birdの転移評価は5 run中、基準環境で飛べた3つのchampionが対象である。この条件付き成功と、初回から全runが成功する確率を分ける。[固定PROVENANCE](https://github.com/ZerghamAhmed/code-to-control/blob/a716dad972acf147d19f50d8b40e510fb02fff47/PROVENANCE.md)

latency測定は、先に収集した物体状態に対する`policy(state)`だけを計る。環境step・観測抽出・LLMでの生成は時間範囲外で、300状態に各5回、状態内と状態間で中央値を取る。11µs級の数値はその呼び出しに対する測定であり、画素入力から行動までの総遅延ではない。[measure_latency.py](https://github.com/ZerghamAhmed/code-to-control/blob/a716dad972acf147d19f50d8b40e510fb02fff47/scripts/measure_latency.py)

PlayTrainはゲームと学習器を別repositoryに分ける。READMEでは学習器の`env_backend`既定値が`minigrid`であることも明記され、単にゲーム名を変えた条件をPlayTrainの学習と見なせない。[固定README](https://github.com/heyodog0/playtrain/blob/1471481164034b023d2e19dd871def19a3e4ef06/README.md)

検査器の5項目はAPI・決定性・観測・報酬と終了・速度で、速度は省略可能。省略時は結果に`passed=True`と`message="skipped"`を入れるため、集約したpassedだけでは速度を測ったか分からない。検査の実装を見ても元ゲームの意味の一致を判定する項目はない。[runtime/validate.py](https://github.com/heyodog0/playtrain/blob/1471481164034b023d2e19dd871def19a3e4ef06/src/playtrain/runtime/validate.py#L261)、[CLI](https://github.com/heyodog0/playtrain/blob/1471481164034b023d2e19dd871def19a3e4ef06/src/playtrain/gen/validate/validate.py#L34)

PlayTrainの再現資料は、図表ごとに入力data、生成script、cluster jobを対応させる。図の再描画、保存dataの検査、環境の再実行、RLの再学習は異なる作業であり、再現scriptが完走したという一文ではどこまで確認したか決まらない。[reproduction/PROVENANCE.md](https://github.com/heyodog0/playtrain/blob/1471481164034b023d2e19dd871def19a3e4ef06/reproduction/PROVENANCE.md)

## 8. 比べる前に揃えるもの

以下は前節までの条件から導いた、本調査の比較方針である。

| 区別するもの | 同一視すると生じる問題 |
|---|---|
| 環境seed、生成のrun、学習seed、評価episode | 3回の生成を3種類の環境での汎化と読んでしまう |
| raw frames、環境step、agent-decision | frame skipが異なる試行の予算比が変わる |
| 探索中の最高値、固定した最終方策、未使用seed上の成績 | 候補選択の効果と評価成績が混ざる |
| ALE、物体入力のOCAtari、JAXAtari、JS clone | 同じゲーム名でも観測と遷移が違う |
| オンラインLLM呼び出し、オフライン生成、固定コード実行 | 低遅延と学習費用の少なさを取り違える |
| 再描画、成果物再生、再学習、生成からの再発見 | 再現できた部分を過大に広げる |

根拠は[Code to Controlの生成run](https://github.com/ZerghamAhmed/code-to-control/blob/a716dad972acf147d19f50d8b40e510fb02fff47/PROVENANCE.md)、[ESBMのwrapper区分](https://arxiv.org/html/2606.07127v1)、[PlayTrainの再現資料](https://github.com/heyodog0/playtrain/blob/1471481164034b023d2e19dd871def19a3e4ef06/reproduction/PROVENANCE.md)を参照。

これらは選べる研究方向を増やす証拠にはなるが、既存の画素ベースDQNを一つの方式で置き換える根拠にはならない。制御コード、世界モデル、言語条件、環境生成のどこを変えたいかによって、必要な比較群が異なる。

## 9. 調査の限界と残る問い

- 5論文の実験を再実行していない。効果量は著者報告で、今回の独立検証結果ではない。
- ESBMの総token・実時間・完全な実行設定と公開コードは、確認した一次資料から特定できなかった。Memento 3のPongも生成費用を含む同予算比較は未確認。
- PDEは論文版と現在の著者ページに実機課題数の差がある。新しい成績をv4の証拠として扱っていない。
- 小型ローカルLLMで制御器や世界モデルを同等品質に作れるか、未知のゲームで既知攻略知識に頼らず成立するか、失敗を含む生成費用はいくらかは、この5論文だけでは決まらない。
- 生成環境の検査が通ることと、研究者の意図する難しさ・攻略法を持つことは別である。今後確かめるなら、意味の一致を判定する課題と、実装・速度の検査を分ける必要がある。

本冊で使った追加の一次資料は各節へ直接リンクした。[報酬整形の数式点検](llm_reward_shaping_guarantees_2026-11-07.md)と[報酬生成系の公開実装の確認](llm_assisted_rl_reproducibility_2026-11-07.md)は別冊を参照。
