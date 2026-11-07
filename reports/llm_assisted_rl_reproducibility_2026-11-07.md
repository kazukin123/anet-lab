# LLMでRLを補助する公開実装の再現条件

確認日: 2026-11-07（JST）  
対象: [再サーベイ本文](llm_assisted_rl_resurvey_2026-11-07.md)のうち、公開コードと配布モデルを確認できたもの。  
方法: 著者の公開リポジトリを固定commitで読む静的調査。学習、モデル取得、有料API呼び出しは実行していない。

## 目次

1. [1. 確認対象と読み方](#1-確認対象と読み方)
2. [2. LLM-Explorer](#2-llm-explorer)
3. [3. LIMEN](#3-limen)
4. [4. BIRD](#4-bird)
5. [5. Robo-Dopamine](#5-robo-dopamine)
6. [6. RoboReward](#6-roboreward)
7. [7. 再現結果を比べるために残す情報](#7-再現結果を比べるために残す情報)
8. [8. 未確認事項](#8-未確認事項)

## 1. 確認対象と読み方

論文の実験条件、配布コードの既定値、今回実行して確かめたことを区別する。最後の区分に入る学習結果はない。コードから決定的に読める分岐と、実際に特定環境で発生した障害も区別する。

| 著者リポジトリ | 確認したcommit | 主に読んだ範囲 |
|---|---|---|
| [LLM-Explorer](https://github.com/tsinghua-fib-lab/LLM-Explorer/tree/0c374651fd04c9fd72ff83f12a66fcd79b9dd581) | `0c374651fd04c9fd72ff83f12a66fcd79b9dd581` | AtariのDQN＋LLM、README |
| [LIMEN](https://github.com/Lossfunk/LIMEN/tree/cf0b44da37f7e7ee038920c860a8cb00c610c9af) | `cf0b44da37f7e7ee038920c860a8cb00c610c9af` | easy_pickup設定、LLMクライアント、評価器、探索制御 |
| [BIRD](https://github.com/safety-research/bird/tree/a24ef8257bb0b86726154b55da66ef1732589769) | `a24ef8257bb0b86726154b55da66ef1732589769` | 論文条件の起動スクリプト、既定設定、クライアント、README |
| [Robo-Dopamine](https://github.com/FlagOpen/Robo-Dopamine/tree/2c714abca66329f8e2113587226bc9556726f6c2) | `2c714abca66329f8e2113587226bc9556726f6c2` | 動画採点例、依存一覧、README、ファイル一覧 |

RoboRewardは著者のモデルカードを確認した。以下でコードを引用するリンクはcommitを固定し、モデルカードは確認日を基準とする。

## 2. LLM-Explorer

AtariのDQN入口は、直前episodeの報酬と間引いた行動列を文章にし、分析と行動分布の生成を2段階で行う。既定のモデルは`gpt-4o-2024-08-06`、更新間隔は1 episode、総step上限は310,000である。論文の100k・500k条件や、本文で比較したモデルを、配布既定値だけで再現したとは扱えない。[DQN_LLM.py: 設定](https://github.com/tsinghua-fib-lab/LLM-Explorer/blob/0c374651fd04c9fd72ff83f12a66fcd79b9dd581/atari_code/DQN_LLM.py#L31)、[同: 問い合わせ](https://github.com/tsinghua-fib-lab/LLM-Explorer/blob/0c374651fd04c9fd72ff83f12a66fcd79b9dd581/atari_code/DQN_LLM.py#L283)

この入口には、短いepisodeと応答形式について再現前に確かめる条件がある。`len(action_record) // sample_rate`をsliceのstepへ渡すため、履歴が既定の100件未満ならstepは0になる。応答の分布はJSONではなく小数の正規表現から末尾の行動数分を取るため、説明中の数値も抽出対象になる。採用時の条件は要素数であり、ここで合計が正であることを検証してはいない。これは静的に読める条件で、論文の実験で実際に失敗したとの主張ではない。[DQN_LLM.py: 間引き](https://github.com/tsinghua-fib-lab/LLM-Explorer/blob/0c374651fd04c9fd72ff83f12a66fcd79b9dd581/atari_code/DQN_LLM.py#L287)、[同: 分布の解析](https://github.com/tsinghua-fib-lab/LLM-Explorer/blob/0c374651fd04c9fd72ff83f12a66fcd79b9dd581/atari_code/DQN_LLM.py#L358)

確認範囲はこのDQN入口である。CURL、Rainbow、連続制御などの全実装へ同じ性質を一般化していない。

## 3. LIMEN

配布設定のまま始めると、論文の主実験とは探索予算が変わる。論文は30反復・Claude Sonnet 4.6を記す一方、確認した`easy_pickup.yaml`は500反復・`anthropic/claude-sonnet-4`である。[論文 Appendix B](https://arxiv.org/html/2605.03408v1)、[固定した設定](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/configs/easy_pickup.yaml)

| 設定項目 | 配布設定の値 | 比較時に意味するもの |
|---|---:|---|
| `max_iterations` | 500 | 外側で評価する候補数の予算に関わる |
| `candidates_per_iteration` | 1 | 1回の生成候補数 |
| `total_timesteps` | 500,000 | 最初の短い学習 |
| `total_timesteps_full` | 1,000,000 | 閾値通過後の学習 |
| `num_seeds` | 3 | 本格学習で集約するseed数 |
| `cascade_thresholds` | `[0.05]` | 短い学習から進める条件 |
| `max_tokens` | 16,384 | 1応答の出力上限 |

表の出典は[同設定の1〜59行](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/configs/easy_pickup.yaml#L1)。値を足して実行済みの環境step数とすることはできない。実際の更新単位への丸め、失敗、再生成、候補の棄却が別にある。

評価器は短い学習で閾値を判定し、通過時に複数seedの本格学習を呼ぶ。本格学習が例外やNaNで終わると、短い学習の指標を返し、評価段階も`SHORT_TRAIN`へ戻す。そのため結果表を作る際にはfitnessだけでなく段階も必要である。[evaluator.py: 短い学習](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/limen/evaluator.py#L181)、[同: 本格学習と失敗時の扱い](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/limen/evaluator.py#L259)

ローカルサーバーへの接続可能性はコード上で確認できる。クライアントはOpenAI互換APIの接続先を設定から受け取り、モデルに応じて`max_tokens`または`max_completion_tokens`を送る。これは通信の接続口があるという証拠であり、7B・8Bモデルが同じ品質の観測・報酬を作れる証拠ではない。[llm_client.py](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/limen/llm_client.py#L112)

費用の保存は候補ごとの生成成功だけを数えているわけではない。コードブロックが返らない場合と、生成コードがcrashした場合の再問い合わせtokenを加算し、checkpointへ入力・出力tokenとモデル別の量、推定費用を保存する。ただし、API側がusageを返さない場合や、実際の請求との一致まではこの調査で検証していない。[controller.py: 再問い合わせ](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/limen/controller.py#L386)、[同: 保存](https://github.com/Lossfunk/LIMEN/blob/cf0b44da37f7e7ee038920c860a8cb00c610c9af/limen/controller.py#L670)

## 4. BIRD

BIRDの既定設定はmockの生成器と評価器である。READMEのtesterで動作確認が済んでも、実LLM・実RLの成績を再現したことにはならない。生成器と評価器は別々にproviderとmodelを持つため、片方を変えた条件を両方変更した条件と取り違えない。[既定設定](https://github.com/safety-research/bird/blob/a24ef8257bb0b86726154b55da66ef1732589769/configs/_default.yaml#L33)、[README](https://github.com/safety-research/bird/blob/a24ef8257bb0b86726154b55da66ef1732589769/README.md)

```text
Numbers will not reproduce exactly

数値は厳密には再現しない
```
[BIRD README / 固定commit (2026-11-07確認)](https://github.com/safety-research/bird/blob/a24ef8257bb0b86726154b55da66ef1732589769/README.md#L161)

著者はLLMのサンプリングに加え、公開版に原実験と違うpromptの細部があると明記する。共有予算にも方法固有の例外がある。READMEは通常40方策、RF-Agentは38と記す。CARDの実行数も別扱いで、`card`と比較用の`card_iter5`は入口が分かれている。全方式を単に「5×8本」と記録すると、この違いを失う。[README: 実験規模](https://github.com/safety-research/bird/blob/a24ef8257bb0b86726154b55da66ef1732589769/README.md#L127)、[paper_cell.py: 方法ごとの指定](https://github.com/safety-research/bird/blob/a24ef8257bb0b86726154b55da66ef1732589769/scripts/paper_cell.py#L102)

論文条件を組むスクリプトは、候補数、RF-Agentの学習回数、最終再学習のstep数、環境の採点基準を使う条件を上書きする。設定ファイル単体と起動時の上書きを合わせて実験条件とする必要がある。[paper_cell.py: method_overrides](https://github.com/safety-research/bird/blob/a24ef8257bb0b86726154b55da66ef1732589769/scripts/paper_cell.py#L128)

## 5. Robo-Dopamine

公開版には論文のGRM-3B・8Bに加え、GRM-2.0の4B・8B Previewがある。READMEは4Bについて8B Previewの約80%の精度と記すが、この一文だけでは評価集合や指標の集約を特定できないため、論文のRL成功率と結び付けない。3Bから4Bへの置換を同一モデルの小変更とも扱わない。[README: 公開履歴とモデル一覧](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/README.md#L27)

推論例は動画からframeを間引き、初期・現在・目標の関係を使うモードを選んで採点する。vLLMの`gpu_memory_utilization=0.9`、文脈上限8,192、最大画像数8、出力上限1,024がコードにある。これらは設定値であり、VRAMの最低必要量や、RLとGPUを共有したときの速度の実測値ではない。[inference.py: モデル初期化](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/examples/inference.py#L433)

採点の後処理にも意味がある。文字列を数値にできないと0を使い、`forward`では進捗の差を`hop`にする。`incremental`では出力をhopとし、正負に応じた更新式でprogressを積み上げる。この動画表示用の計算を、論文の割引率・終端条件を含む報酬整形全体と同一視しない。[inference.py: 後処理](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/examples/inference.py#L577)

READMEはCUDA 12.8以上を想定し、依存一覧はtorch 2.8.0・transformers 4.57.0などを固定している。anet-labのWindows環境でそのまま動作するかは検証していない。確認したファイル一覧からは、データ処理、報酬モデルの学習・評価・推論を確認したが、論文のロボットオンラインRL一式が再現可能であるとは判定していない。[README: Setup](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/README.md#L76)、[requirements.txt](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/requirements.txt)、[ファイル一覧](https://github.com/FlagOpen/Robo-Dopamine/tree/2c714abca66329f8e2113587226bc9556726f6c2)

## 6. RoboReward

著者の4B・8Bモデルカードは、動画と課題指示から最後の状態を1〜5で評価する使い方を示す。基盤はQwen3-VLの対応サイズである。この採点を毎frameの差分や特定のゲームイベントへ変換する契約は、モデルカードにはない。公開された重みの存在と、対象RLの報酬として妥当なことを分ける。[RoboReward-4Bモデルカード](https://huggingface.co/teetone/RoboReward-4B)、[RoboReward-8Bモデルカード](https://huggingface.co/teetone/RoboReward-8B)

## 7. 再現結果を比べるために残す情報

以下は上の差分を踏まえた本調査の整理であり、anet-labの実装仕様や採用済みの計画ではない。

| 記録する区分 | 区別が必要な理由 |
|---|---|
| 論文の版、code commit、起動時の上書きを含む設定 | LIMENの既定反復数、BIRDの実行profileで条件が変わる |
| 生成器・評価器・報酬モデルの識別子 | BIRDの生成と評価は独立。Robo-Dopamineには論文版とPreviewがある |
| 外側の探索seed、候補ごとの学習seed、評価episode | 候補選別の偶然と、選んだ方策の学習・評価の偶然を区別する |
| 生成候補数、構文失敗・再試行・棄却・本格学習の件数 | LIMENの短い学習と本格学習、BIRDの方法別の例外を区別する |
| 入力・出力token、LLM待ち時間、RLのstepと実時間 | API費用だけ減ってもRL再学習の負担が残る |
| 採点源と最終評価の採点源 | 報酬の自己採点がよくても、本来の成功を達成したとは限らない |
| 応答の解析失敗件数とその扱い | LLM-Explorerの数値抽出、Robo-Dopamineの0への置換が結果へ混じり得る |

## 8. 未確認事項

- 小型のローカル生成モデルで、コードの実行成功率とRLの最終性能を保てるか。接続先の変更が可能なことだけでは答えられない。
- 本冊で調べた各実装をRLと同じGPUへ載せた場合の、総実時間・VRAM・学習への影響。別方式のLAGEAには単一RTX 4090の報告があるが、この4実装への実証ではない。[追加調査](llm_rl_goals_and_reflection_2026-11-07.md#5-lagea失敗の説明を各stepの学習信号へ変える)
- 論文と配布既定値の差を揃えた際の再現値。今回の調査は静的確認にとどまる。
- 動画採点モデルをAtariへ移した場合の妥当性、終端や後退を含む採点の一貫性。
- 外部リポジトリの全コード、全入口、全失敗経路の検査。確認箇所を超える品質判定はしていない。
