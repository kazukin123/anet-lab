# LLMの助言を方策へ蒸留する方法と、世界モデルの先読みに使う方法

調査日: 2026-11-07（JST）  
対象: LVLM2P、PriorZeroの2論文。両者の固定commitで学習への接続と評価設定も確認した。  
関連: [読み順と索引](llm_rl_reading_guide_2026-11-07.md)、[行動助言の接続先](llm_rl_action_advice_2026-11-07.md)。  
状態: 一次論文と公開コードの静的な読解。教師APIの呼び出し、環境実行、学習は行っていない。

## 目次

1. [1. 助言をいつ使うか](#1-助言をいつ使うか)
2. [2. LVLM2P：行動分布を小さな方策へ移す](#2-lvlm2p行動分布を小さな方策へ移す)
3. [3. 公開PPO実装で異なる教師分布](#3-公開ppo実装で異なる教師分布)
4. [4. PriorZero：先読みの入口だけを言語で補う](#4-priorzero先読みの入口だけを言語で補う)
5. [5. 知識の転用と実行可能な操作の違い](#5-知識の転用と実行可能な操作の違い)
6. [6. 書誌と確認範囲](#6-書誌と確認範囲)

## 1. 助言をいつ使うか

行動の候補を教える方式には、学習中に小さな方策へ教え込む方法と、実際に行動を決めるときに先読みの出発点として使う方法がある。前者は学習後に教師を外せる。後者は、その場の助言と環境から学んだ予測を組み合わせる。本冊は二つを、問い合わせの場所と誤りを修正する経路から読む。

| 比較点 | LVLM2P | PriorZero |
|---|---|---|
| LLMへの主な入力 | 環境全体の画像と手作りprompt | 履歴の文章と実行可能な行動 |
| 学習されるもの | PPO/A2Cの小さな方策 | 世界モデルとLLMの両方 |
| 助言の接続先 | 教師と学生の行動分布を近づける損失 | MCTSの根で使う行動事前分布 |
| 学習後 | 学生だけで実行 | 先読みとLLM事前分布を使う |
| 費用を読む焦点 | 学習中の教師呼び出し | LLM推論・調整・世界モデル・探索の合計 |

表は本調査の整理である。「LLMを小さい方策へ移す」と「LLMの世界知識を世界モデルへ直接移す」は同じ操作ではない。

## 2. LVLM2P：行動分布を小さな方策へ移す

LVLM2Pは、学生が集めた軌跡の画像をGemini-1.5-Flashへ渡し、状況分析と行動確率の二段階で助言を得る。PPO/A2Cの損失に教師から学生へのKL項を加える。論文は教師の全観測画像を明記し、評価時は教師を外す。[arXiv v1 §III–V](https://arxiv.org/html/2505.11221v1)

```text
the student agent operates independently without further interaction with the LVLM

学生エージェントは、LVLMとの追加のやり取りなしで独立に動作する
```
[LVLM2P §I（2025/05）](https://arxiv.org/html/2505.11221v1)

4種類のMiniGrid、各3 seedで、目標returnへのサンプル効率は平均PPO 2.56倍、A2C 2.86倍。これは実時間の短縮率ではない。LavaGapではsoftな教師分布がhardな分布を上回り、係数0.01がよく、1超では成功率が低下した。手作りの座標質問やfew-shot例は残る。[§V、Fig. 2–4](https://arxiv.org/html/2505.11221v1)

本調査では、総問い合わせ数・API費用・学習実時間を確定できていない。式(1)とAlgorithm 1には「最大化する報酬目的へ正のKLを足す」ように読める符号表記もあり、最小化する実装の損失と照合する必要がある。論文のsoft分布という説明と、次節の公開PPO分岐も一致しない。[式(1)、Algorithm 1](https://arxiv.org/html/2505.11221v1)

評価時に教師が不要であることは明確な利点だが、学習時の総費用が安いことや、小型のローカル教師で同じ結果が出ることまで示すものではない。

## 3. 公開PPO実装で異なる教師分布

公式に案内された`i22024/LVLM2P`のcommit `bbad541b029ba3eff941299d3c937312676aed73`（2024-11-26）を対象とした。論文より前の公開スナップショットであり、論文の全実験を生成した版だとは確認できない。[固定commit](https://github.com/i22024/LVLM2P/commit/bbad541b029ba3eff941299d3c937312676aed73)

`torch_ac/algos/ppo.py`の`train_ppo_from_vlm`分岐には次の処理がある。

| 箇所 | 静的に確認した処理 | 論文との照合点 |
|---|---|---|
| 100–110行 | 更新のsub-batchごとに画像を加工して教師へ問い合わせ | 軌跡数だけでは問い合わせ総数は決まらない |
| 116行 | 既に確率である`dist.probs`へ再びsoftmax | 通常の学生分布そのものへのKLではない |
| 117–120行 | 教師のargmaxだけに0.8、他を0とする | 教師のsoft確率をそのまま保持せず、総和も1ではない |
| 121–140行 | 係数0.01を掛け、PPOの最小化損失へ加算 | この分岐はKL様の項を小さくする符号 |

[固定版PPO更新](https://github.com/i22024/LVLM2P/blob/bbad541b029ba3eff941299d3c937312676aed73/torch_ac/algos/ppo.py#L100)

これは静的な不一致であって、論文値が誤りだという判定ではない。別の未公開版・別の実験分岐の可能性は残り、A2Cの全経路も検査していない。再現するときは、公開分岐をそのまま走らせたのか、論文の数式へ合わせたのかを分ける必要がある。

二重softmaxの違いは小例で分かる。3行動の学生確率が`[1, 0, 0]`に近づいても、その確率へsoftmaxを掛けた最初の成分は\(e/(e+2)\simeq0.5761\)にしかならない。この値は本調査の計算であり、学習結果ではない。教師の分布を一つの値へ潰す処理も含め、softな助言の比較を再現したと扱う前に確認すべき箇所である。

応答解析は数値リストを正規表現で探し、見つからなければ`breakpoint()`へ入る。実行環境によっては対話待ちを起こすので、無人の学習が常に進むとは言えない。[固定版の応答解析270–279行](https://github.com/i22024/LVLM2P/blob/bbad541b029ba3eff941299d3c937312676aed73/torch_ac/algos/vlm_policy/gemini_policy_2turn.py#L270)

## 4. PriorZero：先読みの入口だけを言語で補う

PriorZeroは、LLMの行動事前分布をMCTSの根だけで世界モデル側の分布と混ぜる。先読み中の潜在状態には対応する実際の文章がないため、その先は世界モデルを使う。世界モデルとLLMを交互に更新し、価値推定をLLM調整の信号にする。[arXiv v1 §3](https://arxiv.org/html/2605.12289v1)

```text
the fusion is applied only at the root node

混合は根のノードだけに適用される
```
[PriorZero Appendix B.1（2026/05）](https://arxiv.org/html/2605.12289v1)

主比較はJerichoの4ゲームとBabyAIの18課題、3 seed。失敗したvariantのablationは1 seedに限る。BabyAIではBFSが作る高水準の操作を使い、移動を含む3〜15候補を比較する。低水準の左右回転・前進を一から学ぶ条件ではない。[§4、Appendix B.2–B.3](https://arxiv.org/html/2605.12289v1)

JerichoはQwen2.5-3B、50k stepにH200を4枚で約35時間。BabyAIは7B、100k stepに同4枚で約48時間。3Bというモデルサイズだけで手軽さを判断できない。BabyAI集約returnは0.82対UniZero 0.79で、すべての課題を同様に改善するわけではない。[§4.5、Appendix B.3.3–B.3.4](https://arxiv.org/html/2605.12289v1)

根でだけ呼ぶことは「環境stepごとには呼ばない」という意味ではない。一つの実状態に対する木探索の内部でLLMの呼び出しを増やさない、という違いである。

公開先LightZeroのcommit `3be29ae17452a6fd2d91fe5aea22a757a23e7a69`（2026-10-15）では、`priorzero_policy.py`が根の確率を混ぜ、logへ戻してMCTSへ渡す。論文の固定係数に加え、学習中に重みを変える分岐もあるため、実験設定の選択が必要になる。[固定版の根での混合](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/priorzero/src/priorzero_policy.py#L338)

BabyAI設定では学習・評価とも同じ18個の`data_idx`を使い、リストはlevel番号から1を引いた値である。wrapperは学習時にリストから選び、評価時は順番に選んで、その値をサーバーのresetへ送る。論文の`seed = data_idx // 40`という契約なら、これらはすべて環境seed 0となる。独立した学習seedが3本あることを、未見の環境配置への汎化と読み替えない。サーバー本体の版と実行時の上書きは未確認なので、論文の全runがこの条件だったとは断定しない。[固定設定201–205行](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/babyai/priorzero/src/priorzero_config.py#L201)、[resetの経路260–280行](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/babyai/priorzero/envs/babyai_env.py#L260)

1 GPU用のquick-testスクリプトもあるが、論文の4 H200による成績・実時間と同じ証拠ではない。モデル・BGE encoder・起動ディレクトリには開発環境の絶対パスが残る。この公開版をそのまま別PCで再現できたとは扱わない。[quick test](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/babyai/priorzero/scripts/test_1gpu.sh)、[モデル設定](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/babyai/priorzero/src/priorzero_config.py)

同じ固定版のquick testは`--data_idx 0`を渡すが、呼び出すentryの`ArgumentParser`にはこの引数の宣言がなく、`parse_args()`を使っている。したがって、依存の読み込みとパスの問題を解消して引数解析へ到達しても、この呼び出しは未定義引数として拒否される構成である。これは起動コマンドと解析器を照合した結果であり、GPU学習を実行して得たエラーではない。単に「1 GPU用の手順がある」ことと、保存された手順がそのまま通ることを分ける。[起動スクリプト7行](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/babyai/priorzero/scripts/test_1gpu.sh#L7)、[引数宣言291–302行](https://github.com/opendilab/LightZero/blob/3be29ae17452a6fd2d91fe5aea22a757a23e7a69/zoo/babyai/priorzero/src/priorzero_entry_sync_ddp.py#L291)

## 5. 知識の転用と実行可能な操作の違い

本調査の考察として、二つの異なる「知識の転用」を分けられる。LVLM2Pでは、教師の出力を小さい方策の重みへ移す。PriorZeroでは、学習した世界モデルによる先読みをLLMの助言で誘導し、逆に環境から得た価値でLLMを調整する。

どちらの場合も、教師が見られる情報と、使える操作の粒度を揃えないと意味を取り違える。全体画像を見られる教師と部分観測だけの学生の差、BFSが到達経路を用意する操作とprimitive actionの差は、LLMの推論能力とは別の助けになる。

| 比較の問い | 揃える条件 | 結果の読み方 |
|---|---|---|
| 教師の知識が有用か | 同じ全体情報を持つ規則教師、一様な教師 | 情報の追加と知識の内容を分ける |
| 学生へ正しく移ったか | 同じ損失の符号・分布・係数 | 表記上のKLと実際の更新を分ける |
| 世界モデルが助言を補正したか | 同じ候補操作、同じ探索回数 | 助言と先読みの相互作用を読む |
| 部分観測に耐えるか | 同じ履歴と可視範囲 | 教師だけが知る状態の効果を分ける |
| 実時間も改善するか | 教師呼び出し、学習、探索の合計 | 小さい実行方策と安い学習を分ける |

非LLMの比較を置く意義は、LLMの価値を最初から否定することではない。どの知識を、どの形式で渡すことに効果があるかを特定するためである。この点は画像Atariへ持ち込む場合にも残るが、本調査はその実証を行っていない。

**教師だけが知る状態は、重みへ移すだけでは復元できない。** 説明用に、毎回独立に確率1/2で現れる二つの状態を考える。教師には区別できるが、学生には同じ観測しか見えず、履歴も識別情報を持たないとする。一方では左、他方では右だけが成功する。教師の成功率は1にできるが、学生が左を確率\(p\)で選ぶ成功率は\(p/2+(1-p)/2=1/2\)で、学習量を増やしても超えられない。

教師がそれぞれ正解行動へ確率1を置くなら、学生への平均forward KLは\(-\frac12\log p-\frac12\log(1-p)\)。最小値は\(p=1/2\)で\(\log2\)となり、0にはならない。これは本調査の仮想例であり、LVLM2Pの課題が必ずこの状況にあるという指摘ではない。追加の観測や識別可能な履歴があれば前提は変わる。全観測の教師を使う実験では、学生が必要な情報を持つかを別に確認する理由になる。

**KLの向きも区別する。** LLM Priorsの\(D_{KL}(\pi\|p_{LLM})\)は、事前分布が0の行動へ方策が正の確率を置くと無限大になる。一方、LVLM2Pの\(D_{KL}(\pi_T\|\pi)\)では、教師が0の行動へ学生が正の確率を置くこと自体は無限大を生まない。両者を同じ「助言から外れられない制約」とまとめない。有限のsoftmaxで各確率が正なら、厳密な0の議論と非常に小さい確率の数値的な問題も分ける。[事前分布の目的と計算例](llm_rl_action_advice_2026-11-07.md#6-正確な価値があっても事前分布は残る)

## 6. 書誌と確認範囲

| 論文 | 著者・所属・役割 | 公開 | 被引用数 |
|---|---|---|---|
| Sample Efficient Reinforcement Learning via Large Vision Language Model Distillation | Donghoon Lee、Tung M. Luuが同等貢献。Younghwan Lee、Chang D. YooとのKAIST共同研究 | ICASSP 2025、会議2025-04-06〜11。arXiv v1は2025-05-16 | [OpenAlex公刊レコード1件](https://openalex.org/W4408351960) |
| PriorZero: Bridging Language Priors and World Models for Decision Making | Junyu Xiongが第一著者（USTC）、Yuan Pu（Shanghai AI Laboratory）、Jia Tang（NUAA）、Yazhe Niuが責任著者（CUHK MMLab） | arXiv v1、2026-05-12。採択済みとは本調査では確定しない | [OpenAlex 1件](https://openalex.org/W7161080529) |

著者と所属は各論文冒頭、LVLM2Pの公刊情報は[KAISTの書誌](https://pure.kaist.ac.kr/en/publications/sample-efficient-reinforcement-learning-via-large-vision-language/)を確認した。OpenAlexは2026-11-07の取得値で、異なる版の引用数を合算していない。

LVLM2Pの解析失敗時の挙動、公開分岐と実験版の対応、PriorZeroの環境サーバーと実験コードの版は追加確認が必要である。公開モデルの現在の提供状況や再実行費用は今回測定していない。学習結果を追認した報告としては扱わない。
