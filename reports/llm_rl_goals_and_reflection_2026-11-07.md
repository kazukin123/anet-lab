# LLMの小目標・行動の制約・失敗の振り返りをRLへ渡す

調査日: 2026-11-07（JST）  
対象: ELLM、SGRL、LLM-TALE、LAGEAの4論文と、SGRL・LAGEAの固定した公開実装。  
関連: [全体の読み方](llm_rl_reading_guide_2026-11-07.md)、[言葉と環境の対応](llm_rl_language_grounding_2026-11-07.md)、[小型モデルと費用](llm_rl_small_models_and_costs_2026-11-07.md)。  
状態: 一次資料の読解と静的な実装確認。学習・モデル取得・LLM API呼び出しは未実行。

## 目次

1. [1. 何を目標にするかと、どう達成するか](#1-何を目標にするかとどう達成するか)
2. [2. ELLM：有用そうな目標を探索の報酬にする](#2-ellm有用そうな目標を探索の報酬にする)
3. [3. SGRL：目標生成をコードにし、行動を制約する](#3-sgrl目標生成をコードにし行動を制約する)
4. [4. LLM-TALE：候補の姿勢を選び、制御器を補正する](#4-llm-tale候補の姿勢を選び制御器を補正する)
5. [5. LAGEA：失敗の説明を各stepの学習信号へ変える](#5-lagea失敗の説明を各stepの学習信号へ変える)
6. [6. 公開実装で確認した再現条件](#6-公開実装で確認した再現条件)
   - [6.1 SGRLの乱数と行動mask](#61-sgrlの乱数と行動mask)
   - [6.2 LAGEAの入力情報とGPU共有](#62-lageaの入力情報とgpu共有)
7. [7. 調査から分けられる比較](#7-調査から分けられる比較)
8. [8. 書誌と確認できなかった範囲](#8-書誌と確認できなかった範囲)

## 1. 何を目標にするかと、どう達成するか

「石を集めよう」と提案できても、現在の画面から石を識別できるか、どの操作で採れるか、採れたことを何で判定するかは別に残る。本冊は、LLMの役割が近く見える4方式を、その接続の違いから読む。

| 方式 | LLMに任せる部分 | RLへ渡す形 | LLMとは別に用意する情報・機構 |
|---|---|---|---|
| ELLM | 次に試す有用そうな行動 | 目標達成の内発報酬、目標の埋め込み | 状態・遷移の文章化、達成済み目標の管理 |
| SGRL | 目標生成コードと優先度、関連する行動 | 観測への目標追加、行動mask | 状態をコードへ渡す処理、既存の環境報酬 |
| LLM-TALE | 操作の順序と複数の接近・把持方法 | 目標姿勢、残差制御の目標 | 物体状態、PD制御器、距離などの内発報酬 |
| LAGEA | 失敗の種類と修正の説明 | 学習した視覚表現からの報酬 | 成功ラベル、目標画像、LIVの視覚表現 |

表は各論文と後述の実装を比較するための本調査の整理である。同じ「言語で探索を助ける」でも、画像だけから一から行動を学ぶ方式と、既存の制御器を調整する方式を同列に扱えない。

## 2. ELLM：有用そうな目標を探索の報酬にする

ELLMは、状態説明から提案した目標と遷移の説明の類似度を報酬にする。DQNによる事前学習をCrafterとHousekeepで調べる。同じepisodeで達成済みの目標を除く処理も必要で、LLMだけの効果とは分けて読む。[ICML 2023論文 §3–4、Appendix L](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

```text
mostly assumes a pre-existing captioning function

主として、既存の文章化関数があることを仮定する
```
[ELLM §3.5（2023/07）](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

主条件は画像に加えて状態説明も入力する。学習した文章化器の追加試験は847件の人手ラベルと900件の合成ラベルを使い、単語の重なりによる誤報酬も残る。事前学習後は、単純なfine-tuningより事前学習方策を探索時に使う方法がよく、goal-conditioned版には下流課題の小目標列を与える。API結果のcacheもrun間で再利用する。[§3.5、§4.1、Appendix J/M/O](https://proceedings.mlr.press/v202/du23f/du23f.pdf)

したがって本調査では、「文章化まで無料」「下流課題の情報なし」「新規runでも問い合わせが少ない」という三つの読み方を避ける。何を再利用したかを揃えると、意味のある探索目標を選ぶ効果を考えやすい。

## 3. SGRL：目標生成をコードにし、行動を制約する

SGRLは、LLMへの毎stepの問い合わせを、再利用する目標生成関数へ置き換える。目標の優先度を使い、関連しない行動をmaskする強さを学習中に変える。本節の実験値は2025年のarXiv v1を対象とし、2026年公開の改題された雑誌版の追加実験と混ぜない。[v1 §3](https://arxiv.org/html/2509.22008v1)

```text
reusable, structured goal generation function

再利用できる、構造化された目標生成関数
```
[SGRL v1 §3.1（2025/09）](https://arxiv.org/html/2509.22008v1)

Craftax-Classicの5M step・5 seedではscoreがSGRL 33.8±1.5、PPO 24.8±5.7。一方、速度は1,850対13,530 step/sで、PPOより遅い。SGRLは頻繁にLLMを呼ぶ比較法より速い、という範囲にとどまる。生成はDeepSeek-R1/V3、RL側はRTX 4090という条件であり、小型LLMの同居を実証した結果ではない。[§5、Table 1、Appendix B](https://arxiv.org/html/2509.22008v1)

この比較から言えるのは、同じstep数での成績と処理速度の交換条件があることまでである。同じ実時間でPPOが何点に達するかは、Table 1だけでは分からない。後述の公開実装には乱数の扱いにも確認点がある。

## 4. LLM-TALE：候補の姿勢を選び、制御器を補正する

LLM-TALEは、GPT-4oで操作の順序と複数の把持・接近方法を学習前に作り、cacheする。RLは候補を価値と探索用の重みで選び、PD制御器に足す補正を学ぶ。LLM自身が低水準の全動作を選ぶ方式ではない。[v2 §IV–V](https://arxiv.org/html/2509.16615v2)

```text
requires access to object state

物体状態へのアクセスを必要とする
```
[LLM-TALE v2 §VII（2026/04）](https://arxiv.org/html/2509.16615v2)

主評価はManiSkillとRLBenchの計6課題・3 seed。物体状態、既存の制御器、距離・速度から作る密な報酬を使う。実機PutBoxは15試行中14成功。比較するText2RewardだけManiSkill2、他はManiSkill3という差も明記されている。[§V–VI](https://arxiv.org/html/2509.16615v2)

本調査としての関心は、「LLMの提案を正解として固定せず、経験で候補を選ぶ」点にある。ただし、候補がすべて不適切だった場合や、未対応の複雑な物体形状まで解決したとは読めない。生の画像からAtariの行動候補を同様に作れるかは別の問いである。

## 5. LAGEA：失敗の説明を各stepの学習信号へ変える

LAGEAはQwen2.5-VL-3Bのepisode単位の説明をGPT-2で埋め込み、LIVの画像表現へ対応付け、目標と説明への近づきを報酬にする。重要な時点へ重みを付け、失敗時に補助報酬を加える。単一RTX 4090・24 GBを実験機として明記している。[ICML 2026論文 §3–4、Appendix D](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chowdhury26a/chowdhury26a.pdf)

```text
despite the extra cost of VLM inference

VLM推論の追加費用があるにもかかわらず
```
[LAGEA Appendix C.2（2026/07）](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chowdhury26a/chowdhury26a.pdf)

MT10固定目標の平均成功率はFuRL 76%から80%、ランダム目標は64.6%から70.4%。実時間表は主結果の5 seedと異なる3 seed・9条件で、平均94.90分から92.36分。たとえば固定目標のbutton-pressは99.71分から162.12分へ遅くなる。平均の短縮を全課題へ広げられない。[Tables 1–2/9](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chowdhury26a/chowdhury26a.pdf)

これは「小型VLMと外部RLを一つのGPUで使う」具体的な報告として有用である。ただし、実装には目標画像と環境からの成功ラベルがあり、VLMがそのすべてを推定しているわけではない。差分報酬に`tanh`などを重ねるため、通常のpotential-based shapingの方策不変性を自動的に引き継ぐとも読まない。

## 6. 公開実装で確認した再現条件

### 6.1 SGRLの乱数と行動mask

対象は `jiejieqi/SGRL@cbb1b0a6ac201491e501cbc7b182945795c11d52`（2025-12-26）。

`models/actor_critic.py`の`get_marsk_logits`は、呼び出すたびに`jax.random.PRNGKey(0)`を作り、`epsilon * (1 - mask)`でBernoulliを生成する。同じshape・mask・epsilonなら同じ解除結果になり、呼ぶたびに独立な試行を引く処理ではない。名前にある`marsk`は原文のままである。[固定実装](https://github.com/jiejieqi/SGRL/blob/cbb1b0a6ac201491e501cbc7b182945795c11d52/models/actor_critic.py)

学習側は環境用・行動用の乱数を分割しているが、mask関数にはそのkeyを渡していない。PPO更新でも保存したmaskとepsilonを渡して関数を再評価する。論文のBernoulliによる解除と配布コードの実際の乱数系列を区別する必要がある。[学習側](https://github.com/jiejieqi/SGRL/blob/cbb1b0a6ac201491e501cbc7b182945795c11d52/train.py)

これは静的に確認できる処理上の差であり、論文の結果がどれほど変わるかは測定していない。修正版の学習や、論文当時の全runの追跡も行っていない。

### 6.2 LAGEAの入力情報とGPU共有

対象は `monaf-chowdhury/LaGEA@0f8d761729665abc03635410560fd4c1fe853f41`（2026-05-01）。arXiv v3より古いため、最終版との完全一致は未確認である。

| 読んだ経路 | 確認した挙動 | 再現時に意味すること |
|---|---|---|
| `setup_exp` | `data/oracle/.../s0_cN.npz`を無条件に読み、成功画像から目標を作る | READMEの「optional」を、配布経路でファイル不要という意味には取れない |
| policy作成 | `env.observation_space`の39次元状態を方策へ渡す | 画像は報酬側で使う。純粋な画像入力方策の試験とは違う |
| モデル作成 | Qwen 3Bはbfloat16・`device_map="auto"`、GPT-2もCUDA | 配布設定は4-bitではない |
| episode終了 | 重要時点から報酬用の重みを作る一方、VLM入力には等間隔64 frameを渡す | 「重要時点の抽出」と「VLMへ渡すframe選択」を同じ処理とみなさない |

出典: [固定した学習処理](https://github.com/monaf-chowdhury/LaGEA/blob/0f8d761729665abc03635410560fd4c1fe853f41/experiments/train_lagea.py)、[README](https://github.com/monaf-chowdhury/LaGEA/blob/0f8d761729665abc03635410560fd4c1fe853f41/README.md)。

さらに、VLMのpromptには環境が観測した成功・失敗を明示する。JSONの型を検査して最大10回試すが、10回とも不適合だった場合の独立した停止分岐はなく、最後の解析結果を返す。今回、失敗率は測っていない。[promptと返却処理](https://github.com/monaf-chowdhury/LaGEA/blob/0f8d761729665abc03635410560fd4c1fe853f41/feedback/qwen_feedback.py)

報酬生成では、potentialの差を`tanh`へ通し、feedback側はさらに時点の重みを掛ける。後段にはclipもある。個々の`tanh`が有界であることと、重みを掛けた最終信号の上限や方策不変性は別に確かめるべき事項である。[報酬モデル](https://github.com/monaf-chowdhury/LaGEA/blob/0f8d761729665abc03635410560fd4c1fe853f41/models/projection.py)、[後段の整形](https://github.com/monaf-chowdhury/LaGEA/blob/0f8d761729665abc03635410560fd4c1fe853f41/utils/train_utils.py)

## 7. 調査から分けられる比較

ここからは本調査の考察であり、anet-labで実験した結果ではない。

**目標の意味と、目標が使う情報を分ける。** LLMの目標が有効でも、比較法には渡していない状態説明や依存関係が効いている可能性は残る。同じ情報から非LLMの規則で目標を選ぶ比較は、言語モデルの寄与を狭く問うために役立つ。

**行動の提案と禁止を分ける。** 目標を観測に加えるだけなら別の行動も選べる。maskは選択可能な行動そのものを減らす。誤った助言への耐性を問うなら、正しい目標での成績だけでなく、必要な行動が一時的に候補から落ちた場合も別の条件になる。

**説明の正しさと、説明を入力した効果を分ける。** 成功ラベルを既にVLMへ渡している場合、「成功を見抜いた」のではなく「与えられた結果を説明した」可能性がある。説明を固定文に替える、内容を入れ替える、同じ成功ラベルだけ渡すなどの比較は、意味内容が必要かを調べる問いになる。これは追加実験の採用決定ではない。

**問い合わせを減らす方法には違いがある。** ELLMのcache、SGRLのコード、LLM-TALEの学習前の候補、LAGEAのepisode後の説明は、再利用する対象がそれぞれ異なる。費用を比べるなら、cacheやコードを作る初回と、再利用後を分ける。[費用の整理](llm_rl_reading_guide_2026-11-07.md#4-費用を比較するための計算)

## 8. 書誌と確認できなかった範囲

被引用数は2026-11-07にOpenAlex APIを照合した値で、品質評価ではない。公開直後の0件は、未収録の引用がないことを保証しない。

| 論文・版 | 著者・発表時の所属と役割 | 発表先・日付 | 被引用数 |
|---|---|---|---|
| [ELLM](https://proceedings.mlr.press/v202/du23f.html) | Yuqing Du、Olivia Watkinsは共同筆頭・連絡著者、Trevor Darrell、Pieter AbbeelとともにUC Berkeley。Zihan Wang、Abhishek GuptaはWashington。Cédric ColasはMIT/Inria、Jacob AndreasはMIT | ICML 2023、PMLR 202:8657–8677。初稿2023-02-13 | [83](https://openalex.org/W4321011670) |
| [SGRL v1](https://arxiv.org/abs/2509.22008v1) | Yajie Qiが筆頭、Wei Wei、Lin Li、Lijun Zhang、Zhidong Gao、Da Wang、Huizhong Song。全員Shanxi University。雑誌版ではWei Weiが連絡著者 | 初稿2025-09-26。改題した雑誌版はNeural Networks、2026-10-15 online、巻の表示は2027年 | [雑誌版0](https://openalex.org/W7221986504) |
| [LLM-TALE v2](https://arxiv.org/abs/2509.16615v2) | Jelle Luijkxが筆頭、Runyu Ma、Jens KoberとともにTU Delft。Zlatan AjanovićはRWTH Aachen | 著者サイト・公開実装はICRA 2026。v2は2026-04-14 | [0](https://openalex.org/W4415251931) |
| [LAGEA](https://proceedings.mlr.press/v306/chowdhury26a.html) | Abdul Monaf Chowdhuryが筆頭・連絡著者、Safaeid Hossain Arib、Rabeya AkterとともにDhaka。Akm Moshiur Rahman MazumderはIndependent University, Bangladesh | ICML 2026、PMLR 306:20566–20588。初稿2025-09-27、arXiv v3は2026-08-24 | [0](https://openalex.org/W4415332103) |

SGRLの改題・雑誌版の情報は[出版社ページ](https://www.sciencedirect.com/science/article/pii/S0893608026012244)と[PubMed書誌](https://pubmed.ncbi.nlm.nih.gov/42914943/)で照合した。出版社本文は取得制限により未精読で、arXiv v1から追加された条件を確認済みとはしない。LLM-TALEの所属と会議表記は[著者サイト](https://llm-tale.github.io/)で確認した。

4研究とも、今回の読解だけではKungFuMasterのボス攻略やQbert第8面への効果を確定できない。実装確認も選んだ経路に限り、全コードの監査や論文スコアの再現を完了したものではない。
