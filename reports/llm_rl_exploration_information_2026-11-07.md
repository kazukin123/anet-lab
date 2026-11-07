# 探索を補助するLLMに何を渡すか

調査日: 2026-11-07（JST）  
対象: Atariの探索支援を、LLMへの入力、報酬と表現学習、環境へ戻る操作、推論費用に分けて確認する。  
関連: [再サーベイ本文](llm_assisted_rl_resurvey_2026-11-07.md)、[前回のAtari調査](llm_prior_knowledge_rl_survey_2026-10-18.md)、[小型モデルと費用](llm_rl_small_models_and_costs_2026-11-07.md)。  
状態: iLLMの本文・補足資料とIntelligent Go-Exploreの公刊本文・固定commitを確認。実験は実行していない。

## 目次

1. [1. 入力と介入箇所を分ける](#1-入力と介入箇所を分ける)
2. [2. iLLM：予測誤差と補助学習の両方を使う](#2-illm予測誤差と補助学習の両方を使う)
3. [3. iLLMの補足資料で更新された実行条件](#3-illmの補足資料で更新された実行条件)
4. [4. Intelligent Go-Explore：戻る場所もLLMで選ぶ](#4-intelligent-go-explore戻る場所もllmで選ぶ)
5. [5. 公開実装における環境への戻り方](#5-公開実装における環境への戻り方)
6. [6. Atariの証拠をどう比較するか](#6-atariの証拠をどう比較するか)
7. [7. 未確認事項と読み取りの限界](#7-未確認事項と読み取りの限界)
8. [8. 書誌と探索記録](#8-書誌と探索記録)

## 1. 入力と介入箇所を分ける

「LLMがゲームを理解して探索を助ける」という説明だけでは、実際にどこへ情報が流れるかが分からない。本調査では、次の4通りを区別する。これは論文横断の整理であり、著者共通の分類ではない。

| LLMに渡すもの | LLMから受け取るもの | RL・環境へ効く箇所 | 詳細 |
|---|---|---|---|
| 課題説明、間引いた行動列、episode合計報酬 | 次の期間の探索分布 | 探索時の行動抽選 | [LLM-Explorer](llm_assisted_rl_resurvey_2026-11-07.md#51-llm-explorer) |
| 観測と前の行動を変換した埋め込み、履歴 | 推奨行動と要約の表現 | 内発報酬、方策の入力、補助損失 | 本冊のiLLM |
| 物体配置・軌跡・評価結果 | 制御コード | 生成後の各stepの行動 | [Code to Controlなど](llm_generated_control_and_environments_2026-11-07.md) |
| 訪問済み状態の記録と現在の観測 | 戻る状態、探索行動、保存の要否 | 環境の探索経路 | 本冊のIntelligent Go-Explore |

LLM-Explorerは状態を入力から外すことを明記している。したがって、現在の敵との距離や球の速度を直接見て判断する方式ではない。本文Appendix J.3は、初期位置によって有益な行動が逆になる場合の限界を説明する。ここから「ボスの特定の動作に応じて避ける能力」を読み取ることはできない。[LLM-Explorer §3.2、Appendix J.3](https://arxiv.org/html/2505.15293v2)

一方、画像を渡せば精密な操作まで解決するわけでもない。Exploration with Foundation ModelsのGPT-4o評価では、FreewayとMontezumaで結果が異なり、PPOとの併用の正の実証は有利な条件のFreewayに絞られている。前者の入力不足と後者の制御失敗は、同じ問題ではない。[探索能力評価 §4–5](https://arxiv.org/html/2509.19924v1)

## 2. iLLM：予測誤差と補助学習の両方を使う

iLLMは、固定した言語モデルの出力を方策側の予測headが学ぶ方式である。行動headは言語モデルの推奨行動を、履歴headは要約の平均表現を予測する。正の内発報酬は、その予測誤差の二乗距離から作る。本文は一部でsimilarityという語を使うが、式(3)・(5)は距離である。[iLLM 本文 §3.1](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)

```text
prediction-error loss

予測誤差の損失
```
[iLLM 本文 §3.1.1、式(4)](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)

行動側の報酬は、概略として次の形になる。

\[
r_t^a = \tfrac12\|f^a(\phi(o_t))-\mathrm{onehot}(\bar a_t)\|^2.
\]

ここで \(\bar a_t\) はLLMが選んだ行動で、実際に実行した行動との一致率ではない。予測headはこの誤差を小さくする方向へ学習し、方策は誤差が残る場所へ向かう誘因を受ける。したがって「LLMの推奨に従ったときだけ報酬をもらう」と説明すると式と異なる。これは本調査による式の読み取りである。

画像を使う`hop`版は、灰色画像と直前行動をHopfield写像でtoken embeddingの混合へ変換し、凍結LLMで履歴表現を作る。この表現は方策の入力にも入る。補助損失も方策の表現へ影響するため、効果を内発報酬だけに帰属させない。[iLLM 本文 §3.2、Figure 1](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)

Atariの主比較は5ゲーム、100M training steps、10 seeds。MontezumaではRNDの456±55に対しiLLM(hop)は2,632±277。SeaquestはAPTがiLLMを上回る。stepを無条件に生frameへ換算せず、全Atariへの一律改善とも扱わない。[iLLM 本文 Table 3](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a.pdf)

## 3. iLLMの補足資料で更新された実行条件

補足資料は、Transfo-XL 280Mの4-bit版、温度0、Lamorelとcacheの利用を明記する。各stepの履歴表現は作るが、行動・要約の問い合わせはrollout終了後にまとめて行う。要約の出力上限は64 tokensである。[iLLM Supplement §B、§B.2、§D.7](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a-supp.pdf)

```text
querying of the LLM for the two language tasks took place in-batch

二つの言語課題についてのLLMへの問い合わせは、バッチにまとめて行った
```
[iLLM Supplement §B](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a-supp.pdf)

前回報告の「各stepでLLMを使う」は、履歴表現の計算には当てはまるが、報酬用の二つの問い合わせを逐次実行するとの意味にはできない。今回、前回報告の費用表もこの区別へ訂正した。バッチ化は推論の除去でも、総実時間の改善の実測でもない。

「行動生成」も自由文を生成して後から行動へ変換する方式ではない。補足資料Appendix Aは、実行可能な行動の中からlog probabilityが最大のものを選ぶと明記する。Algorithm 1の8行も有限候補の最大値を取る形である。長い行動説明を必要としない環境を選んだことも著者が認める。小型モデルの評価結果を、自由文で新しい操作を発明する能力の証拠へ広げない。[補足資料 Appendix A、Algorithm 1](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a-supp.pdf)

この区別には実装上の意味がある。候補への採点なら、候補名にない行動や解析不能な自由文を選ぶ問題を避けられる。ただし、候補のうち何が有益かを選ぶ意味的な誤りは残る。これは本調査の整理であり、著者の小型モデルを手元で動作確認したものではない。別冊の[LLM Priors](llm_rl_action_advice_2026-11-07.md)は自由文から規則で候補へ対応付けるため、同じ「LLMが行動を提案する」でも接続方法が異なる。

さらに補足資料はAtariでbackboneを比べている。主本文Table 4のBabyAIだけで小型モデルの可否を判断する必要はない。

| Backbone（論文表記） | Montezumaの成績 |
|---|---:|
| Transfo-XL 280M | 2,632±277 |
| GPT2 137M | 2,299±256 |
| GPT2 870M | 2,524±312 |
| Llama2 7B | 2,953±269 |
| Llama3 8B | 3,220±312 |

上表は100M steps・10 seedsの平均±標準偏差。論文中のモデル名・parameter数をそのまま転記し、正確なcheckpoint IDを推測しない。内発報酬を外して補助headの学習を残す比較でも、Montezumaは1,452±201で、同表のPPO 11±4を上回る。[iLLM Supplement §D.4 Table 3、§D.8 Table 6](https://raw.githubusercontent.com/mlresearch/v260/main/assets/bougie25a/bougie25a-supp.pdf)

この比較からは、同方式における小型backboneの有用性を読める。ただし、公開実装・実測時間・正確な量子化構成を確認できていないため、手元のGPUでの再現費用までは確定しない。また、入力写像が言語の意味を正確に復元することと、結果として学習に役立つことは別である。凍結モデルをランダム重みに置き換える対照は、今回読んだ本文・補足には見当たらず、事前学習された意味知識だけが改善原因だと切り分けた証拠は不足する。後半は本調査の評価である。

## 4. Intelligent Go-Explore：戻る場所もLLMで選ぶ

Intelligent Go-Explore（IGE）は、訪問済み状態を保存し、戻る状態・次の行動・新状態を保存するかを基盤モデルで判断する。評価はGame of 24、BabyAIの文章・画像、TextWorldであり、元のGo-ExploreのAtari実績をIGE自身の実績へ読み替えない。[IGE 公刊本文 §2–4](https://proceedings.iclr.cc/paper_files/paper/2025/file/369a30aac2765950865efd318cef7f76-Paper-Conference.pdf)

```text
Return best discovered trajectory

発見した中で最良の軌跡を返す
```
[IGE 公刊本文 Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/369a30aac2765950865efd318cef7f76-Paper-Conference.pdf)

BabyAIとTextWorldは各25 seedsで評価する。BabyAI-VisualのOpenDoorは8%、PutNextToは52%と課題差がある。戻る操作は可能だと仮定し、確率的環境への拡張と頑健な方策の学習は将来方向としている。成功軌跡を見つけたことと、初期状態から未知条件でも動くRL方策を学習したことを分ける必要がある。[IGE 公刊本文 Figure 3、§2、Appendix F](https://proceedings.iclr.cc/paper_files/paper/2025/file/369a30aac2765950865efd318cef7f76-Paper-Conference.pdf)

当時のAPI費用は1課題平均でBabyAIが2.01 USD、TextWorldが1.28 USD。最終実験全体は2,000 USD未満で、開発中の試行費用は別にある。現在の単価やローカルモデルの費用として使わない。[IGE 公刊本文 Appendix E.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/369a30aac2765950865efd318cef7f76-Paper-Conference.pdf)

## 5. 公開実装における環境への戻り方

IGEの固定commitは`bcb4052e6de36a4389e666acd374832cfab1a042`。BabyAIの公開入口は文章版であり、READMEはGame of 24、BabyAI-Text、TextWorldを案内する。画像版の公開場所についてはissue #3に質問があり、閲覧時点で返答はない。[固定README](https://github.com/conglu1997/intelligent-go-explore/blob/bcb4052e6de36a4389e666acd374832cfab1a042/README.md)、[issue #3](https://github.com/conglu1997/intelligent-go-explore/issues/3)

BabyAIの`evaluate_single_performance`は、同じ環境seedでresetし、保存した行動列を再実行して状態へ戻る。その後で`actions_taken`を0へ戻して探索区間を数える。この関数内の探索step数だけでは、復帰に使う全`env.step`数を表さない。[固定コード L374–403](https://github.com/conglu1997/intelligent-go-explore/blob/bcb4052e6de36a4389e666acd374832cfab1a042/babyai/run.py#L374)

これはコードを読んだ範囲の計数方法であり、論文の全グラフが誤っているとの指摘ではない。環境を保存状態へ直接戻す方式と、開始地点から行動列を再生する方式では、復帰費用も失敗条件も違う。比較するなら両方を記録する必要がある。

Atariへの適用について、著者は2024年のissue回答で、画像をGPT-4oへ渡すか文章へ変換する拡張案を述べている。これは実験結果ではない。今回確認した固定treeにもAtari実験の入口はなかった。[著者の回答](https://github.com/conglu1997/intelligent-go-explore/issues/1#issuecomment-2166851429)

## 6. Atariの証拠をどう比較するか

本調査の比較上の結論は、改善率の大きさより先に「何を与え、何を数えたか」を揃える必要がある、ということである。

| 確認する軸 | 混ぜると誤るもの |
|---|---|
| 入力 | 生画像、RAM由来物体、文章caption、行動と報酬だけ |
| 知識の出所 | 一般事前学習、手書きの課題説明、ゲーム専用の物体抽出、正解方策の軌跡 |
| 学習経路 | 補助報酬だけ、補助損失も追加、方策入力も変更、制御器そのものの置換 |
| 環境への介入 | 通常のreset、任意状態への復帰、同じseedでの軌跡再生 |
| 反復の単位 | 環境seed、LLMの生成run、学習seed、評価episode |
| 予算 | 新規探索のstep、復帰を含む全step、生frame、外側の候補探索、総実時間 |
| 成果 | 軌跡の発見、最高episode score、学習方策の平均、特定の出来事の達成率 |

たとえばiLLMとCode to Controlは、どちらもLLMを使うが、前者は補助報酬と表現学習、後者は生成コードによる制御を中心にする。片方のscoreをもう片方の学習効率の証拠にはできない。IGEの任意状態へ戻れる前提も、通常の環境接続での探索支援とは追加条件が異なる。[制御コードの詳細](llm_generated_control_and_environments_2026-11-07.md#2-code-to-control構造と数値を分けて制御器を作る)

Qbert第8面やKungFuMasterのボス攻略を考える場合、平均scoreだけでその場面への到達と攻略を区別できるとは限らない。今回の調査では、これらの特定場面へのLLM支援の直接実証は得られていない。ここは前回結論を維持する。実装案やPRDを今回合意したものとしては扱わない。

**予測誤差が残ることと、有用な未学習の知識が残ることも違う。** 以下は本調査の数式例である。予測headに見える入力を\(X\)、教師の目標vectorを\(Y\)とすると、二乗誤差は

\[
\mathbb E[\|Y-f(X)\|^2\mid X]
=\mathbb E[\|Y-\mathbb E[Y\mid X]\|^2\mid X]
+\|\mathbb E[Y\mid X]-f(X)\|^2
\]

と分かれる。十分に学習して後半を0にしても、同じ入力から識別できない教師出力の揺れは前半に残る。たとえば同じ入力に対し教師のone-hot行動が\((1,0)\)と\((0,1)\)へ半々に変わるなら、最良予測は\((1/2,1/2)\)。式の前に\(1/2\)を掛けるiLLM型の報酬は平均\(\frac12[(1-\frac12)^2+(0-\frac12)^2]=\frac14\)にとどまり、0にならない。

この例はiLLMで実際に生じた失敗ではない。同論文は温度0を使い、完全な入力に対して教師が決定的なら、その入力に条件付けた揺れは0となりうる。ただし、教師だけが履歴を見てheadが現在観測だけを見る場合には、生成を決定的にしても入力の差が残る。予測誤差を探索報酬へ使う方式を比較するときは、教師の揺れ、headの入力不足、学習不足を区別する必要がある。LLMに意味知識があるだけで、この区別が不要になるわけではない。

## 7. 未確認事項と読み取りの限界

- iLLMの公式公開コードと、Atariのwrapper・frame skip・sticky actionsを再構成できる完全な実行設定は未確認。論文報告と手元での再現性は分けた。
- iLLMは補助head・履歴表現・内発報酬を同時に使う。事前学習の意味知識、時系列処理、追加損失の各寄与を完全には分離できていない。
- iLLMの4-bit記載は補足資料にあるが、全backboneが同じ量子化・推論構成だったかまでは確定しない。
- IGEの画像版コードは今回の固定treeで確認できない。著者への連絡は行っていない。
- LLM-Explorerの状態非依存な分布、iLLMの補助学習、IGEの復帰付き探索のいずれについても、anet-labでの学習改善や総実時間を計測していない。

**追跡した別用途：INSIGHTの説明生成。** QbertとLLMの検索で見つかったINSIGHTは、LLMによる探索改善の本体47論文とは別に、用途を区別するために公刊本文§3–5を追加確認した。Lirui LuoらのPeking University・BIGAI共同研究で、責任著者はQing LiとCong Fang。ICML 2024、PMLR 235:33533–33557に掲載されている。[公刊書誌](https://proceedings.mlr.press/v235/luo24j.html)

約10,000 frameを学習済み方策で集め、FastSAMとDeAotから物体の位置を抽出して知覚器へ移す。その後はPPOによるneural actorと、数式で表す方策を同時に学ぶ。GPT-4が受け取るのは課題の説明、学習済み方策の式、特定の判断を説明する場合には座標と勾配も含む情報であり、説明をRLの更新へ戻す機構ではない。9 Atari・10M environment step・3 seedのQbert成績は16,978.6±1,936.1、neural baselineは17,879.2±1,857.1。これをGPT-4がQbertの学習を改善した証拠にはできない。[公刊PDF §3、§4.1、Table 1](https://raw.githubusercontent.com/mlresearch/v235/main/assets/luo24j/luo24j.pdf)

著者自身が、Pongの生成説明にはパドルの動きの誤りや説得的でない行動条件があり、説明品質の定量評価は今後の課題だと記す。式や感度から説明を作ることと、その説明が行動の理由を忠実に伝えていることも別である。コードと付録全体は未確認で、この比較用の補足を主要47論文の件数へ加えていない。[公刊PDF §4.3–5](https://raw.githubusercontent.com/mlresearch/v235/main/assets/luo24j/luo24j.pdf)

## 8. 書誌と探索記録

| 論文 | 著者・所属・役割 | 公表・参照版 |
|---|---|---|
| Exploring Beyond Curiosity Rewards: Language-Driven Exploration in RL | Nicolas Bougie、Narimasa Watanabe。両者Woven by Toyota, Tokyo。研究著者 | PMLR 260:127–142、公式書誌は2025。PDFはACML 2024表記。本文と14頁のSupplementを確認 |
| Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models | Cong Lu（筆頭）、Shengran Hu、Jeff Clune。全員University of British Columbia / Vector Institute。CluneはCanada CIFAR AI Chairも表記 | 初出2024-05-24、ICLR 2025公刊PDF。arXiv v3は2024-12-03 |

書誌の一次資料は[iLLM公式ページ](https://proceedings.mlr.press/v260/bougie25a.html)と[IGE公刊本文p.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/369a30aac2765950865efd318cef7f76-Paper-Conference.pdf)。OpenAlexを2026-11-07に照会し、IGEは[W4399061885](https://openalex.org/W4399061885)で2件、iLLMは検索候補に一致文献がなく取得不可。0件とはしていない。

iLLMの公式ページのSupplementary PDFリンクは`assets/assets/`という重複を含み404だった。同じ公式リポジトリの`assets/bougie25a/bougie25a-supp.pdf`を確認した。Web読取ツールが本文PDFのMIME型に対応しなかったため、既存のpypdfで公開PDFをメモリ上に読み込んだ。OpenReview側はブラウザ確認画面となり、突破は試みていない。

探索ではAtari、Montezuma、QbertとLLM探索の組み合わせを検索し、関連論文からIGEをたどった。一次論文・著者の公開コードと回答を根拠に用い、検索に出た第三者の自動要約を証拠にしていない。本冊の追加本文確認は2論文。再サーベイ全体の対象は[読み順と索引](llm_rl_reading_guide_2026-11-07.md)にまとめた。
