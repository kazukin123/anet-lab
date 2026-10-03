# Survey: SimPLe(Model-Based Reinforcement Learning for Atari)の受容・批判・後続研究

Date: 2026-09-24
Scope: Kaiser ら "Model-Based Reinforcement Learning for Atari"(手法名 SimPLe、arXiv:1903.00374、ICLR 2020)について、一次情報から次の 7 点を確かめる。(1) 論文の主張と、版ごとの変化。(2) 査読と、著者の所属組織による公式発信。(3) 学術的な再評価と批判。(4) Atari 100k ベンチマークとしての扱い。(5) 後続の world model 研究での位置づけ。(6) 報道・ブログ・SNS での扱い。(7) 企業の world model の取り組みでの参照のされ方。

調査は 5 つの角度に分けて並行して行い、本書に統合した。角度は、書誌・版・査読・被引用、再評価・批判・評価手順、後続の world model 研究、報道・ブログ・SNS、産業シグナルである。

## Table of Contents

1. [論文の概要](#1-論文の概要)
   - [1.1 問題設定と学習ループ](#11-問題設定と学習ループ)
   - [1.2 world model](#12-world-model)
   - [1.3 方策の学習と評価方法](#13-方策の学習と評価方法)
   - [1.4 報告された結果](#14-報告された結果)
   - [1.5 論文自身が認めた限界](#15-論文自身が認めた限界)
2. [書誌・版の変遷・コード](#2-書誌版の変遷コード)
   - [2.1 書誌と被引用数](#21-書誌と被引用数)
   - [2.2 版の変遷](#22-版の変遷)
   - [2.3 プロジェクトページとコード](#23-プロジェクトページとコード)
3. [公式発信](#3-公式発信)
   - [3.1 Google Research ブログ](#31-google-research-ブログ)
   - [3.2 Google AI の X 投稿](#32-google-ai-の-x-投稿)
   - [3.3 共著機関 deepsense.ai の発信](#33-共著機関-deepsenseai-の発信)
   - [3.4 後年の Google の言及](#34-後年の-google-の言及)
4. [ICLR 2020 の査読](#4-iclr-2020-の査読)
   - [4.1 評点と決定](#41-評点と決定)
   - [4.2 査読者の主な指摘](#42-査読者の主な指摘)
   - [4.3 著者の回答と一般コメント](#43-著者の回答と一般コメント)
5. [再評価・批判・再現性](#5-再評価批判再現性)
   - [5.1 van Hasselt らの DER](#51-van-hasselt-らの-der)
   - [5.2 Kielak の OTRainbow と更新回数比](#52-kielak-の-otrainbow-と更新回数比)
   - [5.3 計算コストの見積もり](#53-計算コストの見積もり)
   - [5.4 DreamerV2 の位置づけ](#54-dreamerv2-の位置づけ)
   - [5.5 評価の選び方への批判](#55-評価の選び方への批判)
   - [5.6 サーベイでの扱い](#56-サーベイでの扱い)
   - [5.7 再現性](#57-再現性)
6. [Atari 100k ベンチマークの起点として](#6-atari-100k-ベンチマークの起点として)
   - [6.1 起点が SimPLe だとする記述](#61-起点が-simple-だとする記述)
   - [6.2 評価手順の食い違い](#62-評価手順の食い違い)
   - [6.3 26 ゲームの外側](#63-26-ゲームの外側)
7. [後続の world model 研究](#7-後続の-world-model-研究)
   - [7.1 Atari 100k で評価した world model 系エージェント](#71-atari-100k-で評価した-world-model-系エージェント)
   - [7.2 後続論文による SimPLe の位置づけ](#72-後続論文による-simple-の位置づけ)
   - [7.3 比較表に載る SimPLe の値の 2 系統](#73-比較表に載る-simple-の値の-2-系統)
   - [7.4 後継を名乗る記述の有無](#74-後継を名乗る記述の有無)
8. [第三者の報道](#8-第三者の報道)
   - [8.1 英語](#81-英語)
   - [8.2 中国語](#82-中国語)
   - [8.3 ポーランド語・日本語](#83-ポーランド語日本語)
9. [技術ブログ・解説](#9-技術ブログ解説)
10. [SNS・掲示板](#10-sns掲示板)
    - [10.1 X](#101-x)
    - [10.2 Reddit](#102-reddit)
    - [10.3 Hacker News](#103-hacker-news)
    - [10.4 後継研究の発表時の言及](#104-後継研究の発表時の言及)
11. [産業シグナル](#11-産業シグナル)
    - [11.1 SimPLe を参照している企業の取り組み](#111-simple-を参照している企業の取り組み)
    - [11.2 SimPLe を参照していない企業の取り組み](#112-simple-を参照していない企業の取り組み)
    - [11.3 特許](#113-特許)
12. [批判と留保](#12-批判と留保)
    - [12.1 SimPLe に向けられた批判](#121-simple-に向けられた批判)
    - [12.2 批判の側の留保](#122-批判の側の留保)
13. [総合評価](#13-総合評価)
14. [調査の限界](#14-調査の限界)
15. [出典リスト](#15-出典リスト)
    - [15.1 SimPLe 本体](#151-simple-本体)
    - [15.2 公式発信](#152-公式発信)
    - [15.3 査読・データセット・集計](#153-査読データセット集計)
    - [15.4 学術論文: 再評価・評価手法・サーベイ](#154-学術論文-再評価評価手法サーベイ)
    - [15.5 学術論文: 後続の world model 研究](#155-学術論文-後続の-world-model-研究)
    - [15.6 企業の取り組み: 論文・技術報告・公式発表](#156-企業の取り組み-論文技術報告公式発表)
    - [15.7 第三者の報道](#157-第三者の報道)
    - [15.8 ブログ・解説](#158-ブログ解説)
    - [15.9 SNS・掲示板](#159-sns掲示板)
    - [15.10 コード・ドキュメント・issue](#1510-コードドキュメントissue)
    - [15.11 特許](#1511-特許)

表記の約束:

- 本書は出典の文章を逐語では転載しない。私の運用ルール(著作物の逐語の引用は 1 回答につき 1 か所・15 語未満)に従ったもので、survey スキルが求める「主張ごとに逐語で引用する」形式からは外れている。各主張の根拠は、出典の場所(版・節・図表・ページ・投稿 ID)を示したうえで、要旨を日本語で書いた。逐語の引用は §2.2 の 1 か所だけである。コードブロックには、数値・日付・ID・設定値などの事実データの転記だけを入れた。原文は §15 の URL で確認できる。
- 「本調査の照合」「本調査の数え上げ」「本調査の整理」と書いた箇所は、本調査による照合・集計・分類で、出典の主張ではない。「角度 X の担当者の照合」は、並行して調べた担当(サブエージェント)の照合で、本調査で全件を再確認したわけではない。
- 出典ラベルの年月は初出(arXiv 初版または掲載)の年月である。SimPLe だけは版ごとにラベルを分けた(Kaiser SimPLe v1〜v5)。読んだ版が初出と違う場合は、§15 の各項目に書いた。
- 被引用数は、断りがなければ Semantic Scholar の値で、2026-09-24 に取得した。
- HNS は人間正規化スコア(human-normalized score)で、(エージェントのスコア − ランダムのスコア)/(人間のスコア − ランダムのスコア)である。人間とランダムの基準値は論文によって違う(§6.2)。百分率(33% など)で書く論文と小数(0.332 など)で書く論文がある。
- 「100k」「100K」はエージェントのステップ数(行動を 4 フレーム繰り返した後の数)で、40 万フレームに当たる。

---

## 1. 論文の概要

本節は論文本体(主に v4)の要旨である。版ごとの違いは §2.2 にまとめた。

### 1.1 問題設定と学習ループ

**SimPLe は、学習した動画予測モデルを環境の代わりに使い、その中で方策を学習する model-based の深層強化学習である。** 論文の出発点は、model-free 手法が同じゲームを覚えるのに人間よりはるかに多くの相互作用を要することである。人がゲームの仕組みを予測して学ぶのと同じように、動画予測モデルで相互作用を減らせるかを問う。評価は相互作用 100k 回(実時間のプレイで約 2 時間)の低データ設定で行う [Kaiser SimPLe v4 (2020/02), Abstract・Section 1]。

**学習は 3 段の反復である。実環境でデータを集め、それで world model を教師あり学習し、world model の中で方策を強化学習で更新する。** これを 15 回繰り返す。各回の収集は 6400 相互作用で、初回の前にも収集するので合計は 16 回分になる。前処理はフレームスキップ 4 と縦横 1/2 の縮小である [Kaiser SimPLe v4 (2020/02), Algorithm 1・Section 5・Section 6]。

```
実環境の相互作用: 6400 × 16 = 102,400 回 = 409,600 フレーム(60 FPS で 114 分)
world model の学習: 初回 45K ステップ、2 回目以降は各 15K ステップ(既定)
world model 内での方策の相互作用: 合計 15.2M 回
```

[Kaiser SimPLe v4 (2020/02), Section 5・Section 6]

論文によれば、実データも PPO の学習に直接使うが、量が 15M 対 100K と大きく違うため方策への影響は無視できる [Kaiser SimPLe v4 (2020/02), Section 6]。

### 1.2 world model

**world model の入力は連続 4 フレームと行動、出力は次フレームと報酬である。** 構造は Oh ら(2015)に似た畳み込みネットワークを土台にしている。提案モデルはスキップ接続付きの畳み込みエンコーダ・デコーダで、デコーダ各層の出力に行動の埋め込みを掛けて行動で条件付ける。フレームの出力は画素ごとの 256 色 softmax である [Kaiser SimPLe v4 (2020/02), Section 4・Figure 2]。

**画素の損失は一定値 C 以下を切り捨てる形 `max(Loss, C)` にクリップし、論文はこれがモデルの改善に決定的だったと書く。** 論文の推測では、クリップによって背景の広い領域からの勾配が抑えられ、Pong のボールのような小さく重要な領域に最適化が集中する。softmax 損失では C = 0.03 なので、正しい画素値への確信度が 97% を超えた画素からは勾配が出なくなる [Kaiser SimPLe v4 (2020/02), Section 4 "Loss functions"]。予測を入力に戻したときのずれには scheduled sampling で対処する [Kaiser SimPLe v4 (2020/02), Section 4 "Scheduled sampling"]。

**確率性は、潜在変数を bit に離散化し、推論時には補助の LSTM が bit を自己回帰で生成する形で扱う。** 先に Babaeizadeh ら(2017a)にならった VAE を試したところ、2 つの問題が出た。KL 項の重みがゲームごとに違い、しかも非常に小さい値になる。そのため推論時に学習中に見なかった潜在値が出て、予測が悪くなった。離散潜在はその対策である [Kaiser SimPLe v4 (2020/02), Section 4 "Stochastic Models"]。

**world model は約 74M パラメータで、生成が ALE より 2 桁近く遅い。**

```
推論 約 0.5 秒(バッチ 16)/ 逆伝播 約 0.7 秒(バッチ 2)/ NVIDIA Tesla P100
シミュレータとして 約 32ms/フレーム、ALE は 約 0.4ms/ステップ
```

[Kaiser SimPLe v4 (2020/02), Appendix C]

### 1.3 方策の学習と評価方法

**方策は PPO で学習し、rollout は実データから選んだ始点から短く打ち切る。** 論文は、モデルの誤差が時間とともに積み重なることを根本的な難しさとしている。そのため通常 50 ステップごとに、実データのバッファから一様に選んだ状態で world model を再開する。短い rollout では先の効果を学べないので、rollout 最後のステップの報酬に価値関数の評価値を足す。PPO は 16 の並列エージェントで回す [Kaiser SimPLe v4 (2020/02), Section 5]。

**割引率の既定値は本文の 2 か所で食い違っている(本調査の照合)。** 第 5 節は PPO を γ = 0.95 で使うと書く。第 6.4 節は、特に断らない限り γ = 0.99 とし、0.95 がわずかに良かったと書く [Kaiser SimPLe v4 (2020/02), Section 5・Section 6.4]。

**評価時の方策は、学習した方策のロジットを温度 T で割った softmax である。** 多くの場合 T = 0.5 が最良で、T = 0 まで下げると悪化することが多かった [Kaiser SimPLe v4 (2020/02), Appendix D]。

**比較相手の Rainbow は、Dopamine の実装を低データ向けに調整したものである。** 各ハイパーパラメータの組で Pong を 1M 相互作用まで 5 エージェント学習させ、その平均で選んだ。探索範囲の記載では target_update_period の候補が {50, 100, 1000, 4000} なのに、最良値は候補にない 8000 と書かれている(本調査の照合)。PPO は OpenAI Baselines の標準ハイパーパラメータを使った [Kaiser SimPLe v4 (2020/02), Appendix E]。

### 1.4 報告された結果

**評価対象は 26 ゲームである。選定基準は本文と脚注で書き方が違う。** 本文は、既存の最先端 model-free 手法で解けることを基準にしたと書く。脚注 2 は、SimPLe か Rainbow が 100K 相互作用で非ランダムな結果を出したゲームを選んだと書く [Kaiser SimPLe v4 (2020/02), Section 6・脚注 2]。付録の表 2〜4 には 36 ゲームが載っている。図 3 の 26 ゲームに入らない 10 ゲームは次のとおりである(本調査の数え上げ)。

```
表 2〜4 にあり図 3 にないゲーム: Asteroids, Atlantis, BeamRider, Bowling, FishingDerby,
Gravitar, IceHockey, NameThisGame, Riverraid, YarsRevenge
```

[Kaiser SimPLe v4 (2020/02), Figure 3・Table 2・Table 3]

**主な比較は、SimPLe が 100K で出したスコアに Rainbow や PPO が何ステップで届くかである。** 論文は次のように報告している。
- 半数を超えるゲームで、Rainbow は SimPLe の 2 倍以上のサンプルを要した。
- Freeway では差が 10 倍を超えた。
- PPO との差はさらに大きく、PPO が 10M ステップで届くスコアに SimPLe が届いたゲームもある。

[Kaiser SimPLe v4 (2020/02), Section 1・Section 6・Section 6.1・Figure 3]

**結果は各ゲーム 5 run の平均である。** Bank Heist を除く全ゲームでランダム方策を上回った。5 run の最良値はしばしば平均よりかなり良く、最良値では 6 ゲームで人間の平均スコアを超えた [Kaiser SimPLe v4 (2020/02), Section 6.1]。

**見出しの SimPLe の値は、world model を 5 倍長く学習させた構成(表 2 の "SD long")のものである。** 付録 A によれば、他のアブレーションは資源の制約から短い学習で行った [Kaiser SimPLe v4 (2020/02), Appendix A]。表 3 の SimPLe 列と表 2 の SD long 列は値が一致する(本調査の照合、例: Alien 616.9、Freeway 16.7)。v3 はこの構成を初回 225K・以降 75K ステップの学習とし、計算が非常に重く 3 週間以上かかったと書いている [Kaiser SimPLe v3 (2019/06), Section 5]。

**相互作用の量を変えた結果、低データでは強いがデータが増えると優位が消えると論文は書く。** 20K では悪く、50K で 100K とほぼ同等になり、500K まで伸びてそこで model-free の PPO に並ぶ [Kaiser SimPLe v4 (2020/02), Section 6.2]。

**確率モデルが必要な例として Kung Fu Master が挙げられている。** 敵を全員倒した後の画面は毎回同じで、次に来る敵の組は画面だけからは分からない。そのため決定的モデルでは予測できず、確率モデルはもっともらしい敵を描き分ける [Kaiser SimPLe v4 (2020/02), Section 6.3・Figure 11]。sticky actions を入れた環境でも、world model がその性質を学び、多くの場合、調整なしでほぼ同じ結果になった [Kaiser SimPLe v4 (2020/02), Section 6.3・Figure 6]。

**アブレーションの主な結果は次の 3 点である。**
- 提案した確率的な離散潜在モデルが大差で最良だった。
- rollout 長は 25 でほぼ同等、100 でやや悪化した。
- 実データからのランダム始点をやめて rollout 長を 1000 にすると、Seaquest で大きく悪化した。

[Kaiser SimPLe v4 (2020/02), Section 6.4・Appendix A・Table 1]

**定性分析では、成功と失敗が次のように報告されている。**
- 解けた: Pong と Freeway では最大スコアに到達した。
- 予測の精度: Pong・Freeway・Breakout では、最大 50 ステップ程度なら画素単位で完全に予測できた。
- 害の小さい誤り: Bowling と Pong でボールが 2 つに分かれる、Kung Fu Master で敵の数が変わる、Crazy Climber で鳥が早く出る。
- 最も多い失敗: 小さいが重要な物体。Atlantis と Battle Zone では弾が消える。
- その他の失敗: Private Eye のように場面がテレポートで切り替わる大域的な変化も捉えにくかった。

[Kaiser SimPLe v4 (2020/02), Appendix B]

### 1.5 論文自身が認めた限界

**結論の節は 3 つの限界を挙げる。**
- 最終スコアは、最良の model-free 手法より全体として低い。
- 同じゲームでも run ごとの性能のばらつきが大きい。
- world model の中での学習に要する計算と時間が大きい。

[Kaiser SimPLe v4 (2020/02), Section 7]

v4 は導入の最終段落に、初版の後で低データ向けに調整した Rainbow が SimPLe と互角になったという記述を加えている(§2.2・§5.1)。

## 2. 書誌・版の変遷・コード

### 2.1 書誌と被引用数

**採択版の著者は 14 名で、所属は Google Brain、deepsense.ai、ポーランド科学アカデミー数学研究所、ワルシャワ大学、UIUC、スタンフォード大学である。** 先頭 4 名(Kaiser、Babaeizadeh、Miłoś、Osiński)は同等の貢献で、並び順はランダムと脚注にある。連絡著者は Osiński で、作業の一部を Google Brain のインターン中に行った [Kaiser SimPLe v4 (2020/02), p.1 著者欄・脚注]。v1・v2 の著者は Afroz Mohiuddin を除く 13 名で、Mohiuddin(Google Brain)は v3 で加わった [Kaiser SimPLe v1 (2019/03), p.1] [Kaiser SimPLe v3 (2019/06), p.1]。

**題名は arXiv のメタデータではハイフン付き(Model-Based)、PDF 本体と ICLR ではハイフンなし(Model Based)である。** arXiv の Comments 欄と Journal-ref 欄は空である [arXiv API 1903.00374 (2024/04)] [ICLR 2020 virtual site (2020/04), papers.json]。

**会場は ICLR 2020 である。採否の区分は、OpenReview をもとにした第三者データ 3 件がそろって Spotlight としている。** ICLR 公式サイトの論文データにはセッション名("Wednesday: RL and Planning")しかなく、区分を示す項目は見つからなかった [ICLR 2020 virtual site (2020/04), poster_S1xCPJHtDB] [MReD (2022/03)] [PeerSum (2023/10)] [Paper Copilot paperlists (2025/06)]。共著機関の deepsense.ai も、2020-01 に spotlight への選出を告知している(§3.3)[deepsense.ai (2020/01)]。

**被引用数は集計元によって 1,016〜1,404 と幅がある。** Semantic Scholar は arXiv 版と ICLR 版を 1 つのレコードにまとめている。OpenAlex は 2 つのレコードに分けており、合計は 502 である。Google Scholar の主なクラスタが最も多い。

```
取得日 2026-09-24
Semantic Scholar : citationCount 1,016 / influentialCitationCount 88(arXiv 版と ICLR 版を統合した 1 レコード)
OpenAlex         : arXiv 版 W2920362155 = 418 / ICLR 版 W2994714051 = 84(別レコード)
  arXiv 版の年別: 2019 43 / 2020 93 / 2021 101 / 2022 65 / 2023 57 / 2024 43 / 2025 16
Google Scholar   : 主クラスタ 1,404(11 バージョン)/ 著者欄が Campbell から始まる分かれたレコード 8
```

[Semantic Scholar API (2026/09)] [OpenAlex API (2026/09)] [Google Scholar (2026/09)]

### 2.2 版の変遷

**arXiv には 5 つの版がある。主張の強さが最も大きく変わったのは v3 で、「13 対 13」の自己修正が入ったのは v4 である。** v5 で変わったのは謝辞の 1 文だけである。以下の表は、各版の PDF から抽出したテキストを文単位で比べた結果である(角度 A の担当者の比較を、本調査が抽出テキストで抜き取り確認した)。

| 版 | 日付(UTC) | 主な変更 |
|---|---|---|
| v1 | 2019-03-01 | 初版。著者 13 名。アブストラクトは「model-free 手法より桁違いに少ない相互作用で解く」という書き方。表 1 の SimPLe の値は短い学習の構成のもの |
| v2 | 2019-03-05 | 引用 1 件の追加、誤記 2 件の修正、謝辞の追加。数値は変わらない |
| v3 | 2019-06-11 | 著者に Mohiuddin を追加。アブストラクトから「桁違いに少ない」を削り、「大半のゲームで上回り、一部では 1 桁以上」の文を追加。world model を長く学習させる SD long(225K/75K ステップ)を最良の実験として加え、表の値を差し替え。第 6 節のまとめを「6 ゲームを除く全ゲームで半分未満のサンプル」に強めた。Dyna-DQN・GATS との比較、20K〜1M の結果、36 ゲームの数値付録を追加。結論の限界表現は「大幅に低い」から「低い」に弱めた |
| v4 | 2020-02-19 | ICLR 2020 採択版の体裁。導入に「13 対 13」の段落を追加。第 6 節のまとめを「半数を超えるゲーム」に戻した。sticky actions の実験(§6.3)、データ量の節(§6.2)、world model の規模と速度(付録 C)、Rainbow の調整(付録 E)を追加。結論に計算コストの限界を追加 |
| v5 | 2024-04-03 | 謝辞の計算資源に関する 1 文だけを変更。埋め込み画像 35 点は画素のハッシュが v4 と一致 |

[arXiv abs 1903.00374 (2024/04), Submission history] [Kaiser SimPLe v1 (2019/03)] [Kaiser SimPLe v2 (2019/03)] [Kaiser SimPLe v3 (2019/06)] [Kaiser SimPLe v4 (2020/02)] [Kaiser SimPLe v5 (2024/04)]

**SD long への差し替えで、ゲームによっては値が下がった。** 例として v1 と v3 以降の値を並べる。

```
ゲーム        v1 表 1(短い学習)   v3 表 1 / v4 表 3(SD long)
Freeway       20.3                 16.7
Pong           5.2                 12.8
Breakout      12.7                 16.4
CrazyClimber  39827.8              62583.6
```

[Kaiser SimPLe v1 (2019/03), Table 1] [Kaiser SimPLe v3 (2019/06), Table 1] [Kaiser SimPLe v4 (2020/02), Table 3]

**v3 は SD long の学習について、計算が非常に重く 3 週間以上かかったため、アブレーションは学習サンプルを 1/5 にして行ったと書いている。** v4 の本文は既定の学習量を 45K/15K ステップと書き、最良の結果は 5 倍長い学習で得たと付録 A に書いている [Kaiser SimPLe v3 (2019/06), Section 5] [Kaiser SimPLe v4 (2020/02), Section 5・Appendix A]。

**v4 で追加された段落の内容は次のとおりである。** 初版の公開後、van Hasselt ら(2019)と Kielak(2020)が、Rainbow を低データ向けに調整するとより良い結果が出ることを示した。その結果は SimPLe と互角で、26 ゲームのうち 13 ゲームで model-free 側が、残る 13 ゲームで SimPLe が上回る。段落は、van Hasselt らの比較相手が SimPLe の最初のプレプリント(のちに改善)の結果だったことも注記している。

```
both of the model-free methods are better in 13 games, while SimPLe is better

2 つの model-free 手法はどちらも 13 ゲームで上回り、一方で SimPLe が上回るのは(残りの 13 ゲーム)
```

[Kaiser SimPLe v4 (2020/02), Section 1 最終段落]

### 2.3 プロジェクトページとコード

**論文に載っているプロジェクトページの短縮 URL は、2026-09-24 時点で Google Sites のページに解決する。** 論文中のほかの goo.gl リンクも、同じページ内のアンカーに解決する。

```
https://goo.gl/itykP8 → 302 → https://sites.google.com/view/modelbasedrlatari/home
https://goo.gl/JPi7rB → 302 → 同ページ #h.p_PN9xXX1Zyx1d
https://goo.gl/uiccKU → 302 → 同ページ #h.p_o1uYByIqLxei
```

[SimPLe project page (日付不明)]

**プロジェクトページには、ゲームごとの動画と実験の生データ(pickle)がある。** 冒頭の要約は v1・v2 のアブストラクトと同じ表現(100K 相互作用・400K フレーム・約 2 時間)を使っている。サンプル効率の改善は「1 例で 1 桁、大半で少なくとも 2 倍」としている [SimPLe project page (日付不明), 冒頭・各節]。

**コードは tensor2tensor の rl ディレクトリにあり、TensorFlow 1.13.1 系を前提としている。** README によれば、model-based の学習を一通り回すには数日から 1 週間かかる。学習済みの方策と world model は GCS で配布されている(run 番号 1〜180、1 ゲームにつき 5 run)。README の最終更新は 2019-03-22 である [tensor2tensor rl README (2019/03)] [GitHub API tensor2tensor (2026/09)]。

**tensor2tensor のリポジトリは 2023-07-07 にアーカイブされ、読み取り専用になっている。** ルートの README は T2T を非推奨とし、後継の Trax を勧めている。最終コミットは 2023-04-01、最新リリースは v1.15.7(2020-06-17)である [tensor2tensor GitHub (2023/07)] [GitHub API tensor2tensor (2026/09)]。ICLR 公式ページの Code リンクは別の Google Drive フォルダ("MBRL for Atari")を指している [ICLR 2020 virtual site (2020/04)] [MBRL for Atari Drive (2019/09)]。

## 3. 公式発信

本節は、著者の所属組織(Google、deepsense.ai)が自ら出した発信だけを扱う。第三者の報道は §8、個人のブログは §9、SNS の個人投稿は §10 に分けた。

### 3.1 Google Research ブログ

**Google Research ブログの記事 "Simulated Policy Learning in Video Models" は 2019-03-25 に公開された。著者は Łukasz Kaiser と Dumitru Erhan で、掲載時の肩書きは Google AI の Research Scientist である。** 公開日は arXiv の v2(03-05)と v3(06-11)の間にあたる [Google Research Blog (2019/03), 冒頭]。

**ブログの主張は、約 100K 相互作用(人間の実プレイで約 2 時間)で競争力のある結果を出し、大半のケースで比較手法の 2 倍を超えるサンプル効率を達成した、というものである。** 比較相手は 26 ゲームでの Rainbow と PPO である。Pong と Freeway では、模擬環境だけで学習したエージェントが最高得点に届いた。Freeway・Pong・Breakout では、50 ステップ先までほぼ画素単位で正確に予測できたとしている [Google Research Blog (2019/03), 本文・図の説明]。

**失敗例は 2 種類挙げられている。** 1 つは小さいが重要な物体で、Atlantis と Battlezone の弾が該当する。もう 1 つは、Private Eye のテレポートのような大きな画面変化である [Google Research Blog (2019/03), 失敗例の段落]。

**コードは tensor2tensor の一部として公開され、コマンドラインで遊べる学習済み world model が同梱されている。** 研究は UIUC、ワルシャワ大学、deepsense.ai との共同研究と明記されている [Google Research Blog (2019/03), 末尾・謝辞]。

### 3.2 Google AI の X 投稿

**Google AI の公式 X アカウントは、ブログ公開と同じ 2019-03-25 に SimPLe を紹介した。** 投稿は、SimPLe を Atari 向け強化学習のオープンソースのフレームワークとし、約 10 万回の相互作用で競争力のある結果が出ると述べ、コードの入手先を案内している。反応はいいね 357・返信 6 である(2026-09-24 取得時点)[Google AI (X) (2019/03), 投稿 1110244037305327618]。

### 3.3 共著機関 deepsense.ai の発信

**共著機関の deepsense.ai(ワルシャワの企業)は、Google のブログより 4 日早い 2019-03-21 に、自社ブログと PR Newswire のプレスリリースで成果を発表した。見出しは「人工的な想像力(artificial imagination)」である。** 主な内容は次の 3 点である。
- ネットワークは Pong や Freeway などのゲームを、本物の Atari とほとんど見分けがつかない形で再現したと主張している。
- Osiński は、Unreal Engine 4 のような数百万行のシミュレータのコードを例に挙げ、シミュレーション自体をネットワークに学ばせる狙いを説明している。
- CEO の Tomasz Kułakowski は、マーケティング的な誇張なしに知の境界を押し広げた成果だと述べている。

[deepsense.ai (2019/03), 本文の Osiński 発言段落] [PR Newswire (2019/03), 末尾の CEO 発言段落]

**deepsense.ai の公式 X 投稿(2019-03-21)は、ロボティクスと自動運転での活用が見込まれると添えている。** 反応はいいね 2・返信 0 である [deepsense.ai (X) (2019/03), 投稿 1108742905974394881]。

**deepsense.ai は 2020-01-17 に、論文が ICLR 2020 の spotlight に選ばれたと告知した。** 告知は、spotlight に選ばれるのは投稿の 5% だけだとし、Piotr Miłoś のコメントを載せている。これは §2.1 の第三者データ(Spotlight)と一致する、共著機関側の一次情報である [deepsense.ai (2020/01), 本文]。

### 3.4 後年の Google の言及

**Google Research ブログの DreamerV2 の記事(2021-02-18、Danijar Hafner、Student Researcher)は、SimPLe を含む既存の world model を、最も競争の激しいベンチマークでは上位の model-free 手法と競えるほど正確ではなかった、と位置づけている。** [Google DreamerV2 blog (2021/02), 導入部]

## 4. ICLR 2020 の査読

**OpenReview 本体はボット検証のため閲覧できなかった。本節は OpenReview をもとに作られた第三者のデータセット(PeerSum、MReD、Paper Copilot)による二次情報である。** OpenReview の API は 403(Challenge verification required)を返した [OpenReview API (2026/09)]。

### 4.1 評点と決定

**公式査読は 3 件で、評点は 6・8・6 である。** ほかに各査読への著者回答が 3 件、一般コメントが 1 件、それへの著者返信が 1 件ある [PeerSum (2023/10), row 2925] [Paper Copilot paperlists (2025/06)]。

**決定コメントは、結果の重要性が新規性の弱さを上回るとして採択を勧め、口頭発表(oral)での採択に触れている。一方、採否区分は第三者データ 3 件ともに Spotlight である。** 決定コメントの要旨は次のとおりである。動画予測を使った Atari 向けの model-based RL で、限られた相互作用で際立った性能を出した。多くの研究者の関心事への非常に重要な結果である。査読者は全員が掲載に賛成したが、新規性には意見が分かれた [PeerSum (2023/10), row 2925 meta_review] [MReD (2022/03), 2020-597]。

### 4.2 査読者の主な指摘

**評点 6 の査読(Byg8y2bUKr)は、実験の質(26 ゲーム、強い比較相手、5 シード平均)を評価しつつ、低データの設定だけで評価している点を懸念した。** 付録の図では 500K ステップでサンプル効率の優位が消え、漸近性能でも劣るとして、その図を本文に移し、50M ステップでの model-free の最高性能との差を明記するよう求めた。手法の独創性は高くないが、Atari で機能する model-based RL は文献上の大きな空白を埋めるとして weak accept とした [PeerSum (2023/10), row 2925 review Byg8y2bUKr]。

**評点 8 の査読(SJg3esx3tH)は、結果を非常に印象的としつつ、計算コストを主な技術的懸念に挙げた。** 実データは 1〜2 時間分でも、エージェントは world model の中で 15.2M 回の相互作用を要する。そのため、実環境で Rainbow を 15.2M 回学習させる場合と計算資源を比べるよう求めた。Rainbow を 1M ステップ向けに最適化して比べることの公平性も問い、Ha & Schmidhuber(2018)との違いが環境によるのか手法によるのかも不明確だとした [PeerSum (2023/10), row 2925 review SJg3esx3tH]。

**評点 6 の査読(H1lwnmopFH)は、実環境の軌跡数は大きく減っても PPO の更新回数の総数はあまり減らないので、強化学習そのもののサンプル効率はあまり進んでいないと指摘した。** ALE は実行が安く並列化しやすい一方、world model には GPU が要るとして、総実時間への影響を尋ねた [PeerSum (2023/10), row 2925 review H1lwnmopFH]。

### 4.3 著者の回答と一般コメント

**著者は、計算コストが ALE 上で Rainbow などを直接走らせるより大きいこと、SimPLe の実時間が標準的な model-free 学習より長いことを認め、結論と付録 C に書き加えた。** そのうえで、実世界での経験収集が高価または危険な分野(ロボティクス、自動運転)では利点があるとした。100k に絞ったのは主に計算量の都合で、現状では最先端の model-free 手法の最終性能には並べない可能性が高いとも認めた [PeerSum (2023/10), row 2925 author responses]。

**一般コメントは、van Hasselt ら(arXiv:1906.05243)と Kielak(OpenReview Bke9u1HFwB)を挙げ、低データ向けに調整した model-free 手法が SimPLe と少なくとも同等だと指摘した。** 著者は返信で次のように答え、最終版に Kielak の比較を入れると予告した。
- Kielak の比較では、26 ゲーム中 SimPLe と OTRainbow が 13 ずつで互角である。
- van Hasselt らの「26 ゲーム中 17 で勝つ」は古い SimPLe の結果との比較で、SD long の結果なら互角になる。
- SimPLe は Bank Heist や、探索が要る Hero・Private Eye・James Bond を苦手とする。

[PeerSum (2023/10), row 2925 public comment BJxJlLRpFr・author reply ByxgJ6Jj5H]

**著者回答で「追加した」「最終版に入れる」とされた項目は、arXiv ではどれも v4 で初めて現れる。** 該当するのは、引用の追加(Alaniz、Leibfried ら、Ersen & Sariel、Guzdial ら)、§6.2 の漸近性能の記述、結論と付録 C の計算コストの記述、「13 対 13」の段落である [Kaiser SimPLe v4 (2020/02)] [PeerSum (2023/10), row 2925 author responses]。

## 5. 再評価・批判・再現性

### 5.1 van Hasselt らの DER

**van Hasselt らは SimPLe を研究の動機に挙げ、条件を揃えれば Rainbow DQN の方が少ない経験と計算で SimPLe のスコアを上回ると示した。** 導入部で、パラメトリックモデルによる計画で Atari のデータ効率の良い学習を示した Kaiser らの結果に、一部触発されたと書いている [vanHasselt DER (2019/06), NeurIPS 版 導入部 p.1]。

**論文は SimPLe を、Dyna 型のアルゴリズムを一般化した枠組み(Algorithm 1)に当てはめて数量化している。** 実環境の相互作用は少ないが、モデルから大量のサンプルを取るという整理である。計算面では、SimPLe の大きな画素予測ネットワークを例に、パラメトリックモデルはリプレイからのサンプリングより計算が重いと述べている [vanHasselt DER (2019/06), NeurIPS 版 §2.1 p.4・§4 SimPLe の段落・脚注 2 p.8]。

```
SimPLe: 実環境 K × M = 16 × 6400 = 102,400 回、モデルからのサンプル 19 × 800,000 = 15.2M
DER:    実環境 K = 100,000、M = 1、P = 32 → リプレイからのサンプル 3.2M
DER の Rainbow からの変更: multi-step 3 → 20、学習開始時のリプレイ量 20,000 → 1,600
```

[vanHasselt DER (2019/06), NeurIPS 版 §4・§4.1 p.8]

**結果として、DER は 70,000 相互作用で SimPLe に並び、100,000 では約 25% 上回り、26 ゲーム中 17 ゲームで SimPLe を上回った。** 指標は 26 ゲームの人間正規化スコアの中央値で、5 回の独立な反復から誤差棒を出している [vanHasselt DER (2019/06), NeurIPS 版 §4.2 p.8・Figure 3 p.9]。

**比較に使った SimPLe の値は最初のプレプリント(v1)のものである。** arXiv v1 の付録の表は SimPLe の値の出典を Kaiser ら(2019)と明記しており、値は SimPLe v1 と一致する(例: alien 405.2)。この表で各ゲームの高い方を示す太字を数えると Rainbow 16・SimPLe 10 で、本文の「17/26」とは合わない(角度 B の担当者の集計)。NeurIPS 版の補足資料は確認しておらず、食い違いの理由は分からない [vanHasselt DER (2019/06), arXiv v1 付録 E Table 1 p.13]。SimPLe 側は v4 でこの点を注記し、SD long の値で比べれば 13 対 13 だとしている(§2.2)。

### 5.2 Kielak の OTRainbow と更新回数比

**Kielak は、近年の手法の改善は新しい仕組みではなく、実データ 1 件あたりの学習更新の回数(比 r)を増やしたことによると主張した。** 既定の DQN は r = 1/4 である。SimPLe は実相互作用 6.4k ごとに 800k の模擬ステップを回すので、model-free 部分が DQN なら r = 31.5 になると試算している。新手法にだけ追加の更新を許す比較は不公平だとし、更新回数を増やした Rainbow(OTRainbow)を比較相手に立てた [Kielak OTRainbow (2020/03), §4 p.3–4]。

**Kielak は SimPLe を当時のデータ効率の最先端の model-based 手法と位置づけたうえで、100k では明確な勝者がないと報告した。** OTRainbow と SimPLe はちょうど半数ずつのゲームで勝ち、人間正規化スコアの中央値では OTRainbow が上回る [Kielak OTRainbow (2020/03), §3 p.3・§6.1 p.5–6・Table 1・Table 4]。

```
人間正規化スコアの中央値(100k): OTRainbow 20.42% / SimPLe 10.17% / HRainbow 2.27%
  HRainbow = SimPLe 論文が比較に使った調整済み Rainbow
Freeway: OTRainbow 25 / SimPLe 16.7
SimPLe が HRainbow を上回るゲーム: 26 中 20
学習時間: SimPLe は 3 週間超(Kaiser らの報告として)/ OTRainbow は同じデータ量で 24 時間以内(Intel Haswell 8 コア CPU)
```

[Kielak OTRainbow (2020/03), Table 4・§6.1 p.6・脚注 2]

**Kielak は、SimPLe 論文の比較相手の Rainbow について、調整したとされるハイパーパラメータが開示されていないとも指摘した。** v1 の SimPLe はこの Rainbow を時間をかけて調整したと書くだけだった。v4 では調整の詳細が付録 E に加わっている(§1.3)。Kielak がどの版を参照したかは確認できていない [Kielak OTRainbow (2020/03), §4 p.4] [Kaiser SimPLe v1 (2019/03), Section 6] [Kaiser SimPLe v4 (2020/02), Appendix E]。

**結論として Kielak は、少なくとも Atari では model-free が依然として最先端であり、OTRainbow のような適切なベースラインを使うべきだとした。** [Kielak OTRainbow (2020/03), §7 p.8]。論文は v2(2020-03-31)で "Importance of using appropriate baselines for evaluation of data-efficiency in deep reinforcement learning for Atari" に改題されている。ICLR 2020 に投稿したとあるが、採否は確認できていない。

### 5.3 計算コストの見積もり

**後続論文は、SimPLe の学習に数週間(約 500 時間)かかる点を繰り返し記している。** SPR は値を v3 に基づくと注記し、TWM は SPR から転載している。Kielak は Kaiser らの報告として 3 週間超と書いている。v3 には、SD long の学習が 3 週間以上かかったという記述がある(§2.2)。IRIS の「3 週間」の出どころは書かれていない。各論文の記載は次のとおりである。

```
出典                       SimPLe の学習コスト                              比較対象
SPR 付録 D Table 8         500 時間(P100、100k ステップ、v3 に基づくと注記)  SPR 4.6 時間
IRIS 付録 G                1 環境あたり 3 週間(P100 1 基)                 IRIS 3.5 日(A100)
TWM 付録 A.1 Table 3       500 時間(SPR から転載、P100)                   TWM 23.3 時間
DreamerV3 v1 Table T.1     10 GPU 日(V100 換算、P100 は半分の速度と仮定)  —
DreamerV2 Table 3          accelerator days 40、学習パラメータ 74M、4M フレーム  DreamerV2 10 日・22M・200M
Kielak §6.1                3 週間超(Kaiser らの報告として)                OTRainbow 24 時間以内(CPU)
```

[Schwarzer SPR (2020/07), 付録 D・Table 8 p.18] [Micheli IRIS (2022/09), 付録 G] [Robine TWM (2023/03), 付録 A.1・Table 3] [Hafner DreamerV3 (2023/01), v1 付録 T・Table T.1 p.35] [Hafner DreamerV2 (2020/10), Table 3 p.10] [Kielak OTRainbow (2020/03), §6.1]

**SPR は、SimPLe は 100k で複数のゲームに強い結果を出したが学習に数週間かかり、DER と OTRainbow ははるかに少ない計算で同等以上だ、と整理している。** [Schwarzer SPR (2020/07), §3.1 p.6]。EfficientZero も、SimPLe はかなり効率的だが性能は劣ると位置づけている。そのうえで、調整した Rainbow で同等の結果が出ることを Kielak と van Hasselt らが示した、と要約している [Ye EfficientZero (2021/10), §1・§2.1 p.2]。

### 5.4 DreamerV2 の位置づけ

**DreamerV2 は、Atari で正確な world model を学ぶ過去の試みが競争力のある性能に届かなかった例として SimPLe を挙げ、評価の条件の違いを図 1 の注記で指摘している。** 注記は SimPLe について次の 3 点を、SimPLe の著者の報告に基づくとして挙げている。
- より易しい 36 ゲームの部分集合でしか評価していない。
- 学習ステップが少ない。
- 追加で学習しても性能が伸びない。

「36」は SimPLe の付録の表に載るゲーム数と一致する(§1.4)[Hafner DreamerV2 (2020/10), Figure 1 の注記 p.1・§1]。

**関連研究の節は、SimPLe が 400k と 2M 環境ステップで評価され、それ以降は収穫逓減が報告されていると書く。** そのうえで、この低データ域の最高値(gamer 正規化中央値 0.28)でも人間の水準には遠いとし、DreamerV2 は 200M フレームで評価すると説明している [Hafner DreamerV2 (2020/10), §4 "SimPLe" の段落 p.10]。

### 5.5 評価の選び方への批判

**Korkmaz(AAAI 2026)は、Atari 100k の 26 ゲームの選び方そのものを選択バイアスだと批判している。** 論文の主題は、深層 RL アルゴリズムの性能の順位がデータ量の領域をまたいで単調には保たれない、ということである。その流れで、高データ域向けに設計された Rainbow を低データ域の比較相手に使った研究として、SimPLe、DrQ、CURL、SPR、EfficientZero を挙げている。問題にしているのは、高データ域での順位が低データ域にもそのまま当てはまるという暗黙の仮定である。さらに Atari 100k は、提案論文の手法(SimPLe)か、その仮定の影響を受ける Rainbow のどちらかで成績の良かったゲームを選んで作られた、と論じている [Korkmaz AAAI2026 (2026/07), Abstract・§6 "Datasets are Created and Founded on Implicit Assumptions"]。選定基準の原文は §1.4 に書いた。

**Neves ら(プレプリント、2021)は、SimPLe の評価手順が明らかでないと指摘している。** 挙げている点は 3 つである。
- 全エピソードの平均で評価したのか、チェックポイントごとの最後のエピソードで評価したのかが分からない。後者は Machado らが推奨する方式である。
- ALE の確率性の機能を使ったかどうかが分からない。
- 学習の停止基準がフレーム数ではなくアルゴリズムの反復回数である。

[Neves COMPER (2021/11), §2・§5 冒頭]

### 5.6 サーベイでの扱い

**Plaat らの model-based RL のサーベイは、SimPLe を潜在モデル(latent model)の代表例として紹介している。** 2020 年のプレプリントは、SimPLe を学習=VAE と LSTM、計画=MPC、RL=PPO、応用=Atari と分類し、26 ゲーム・100k で Rainbow よりサンプル効率が高いと記している。SimPLe 固有の限界は挙げていない [Plaat Survey2020 (2020/08), §3.2.2]。2021 年版(AI Review 2023)は、この種の潜在モデル手法の多くが複数のネットワークと異なる学習・計画アルゴリズムを組み合わせた複雑な設計だと述べる文脈で SimPLe を挙げている [Plaat Survey2021 (2021/07), §3.1.3・§3.2.2・§3.3.2]。

**Moerland らと Luo らのサーベイの arXiv 版には、SimPLe への言及が見つからなかった。** PDF の全文検索で確認した [Moerland Survey (2020/06)] [Luo Survey (2022/06)]。

### 5.7 再現性

**公式の README は論文の結果を再現できるはずだとしているが、実行・再現を尋ねる GitHub issue の多くは未回答のまま残っている。** 確認できたものは次のとおりである。
- #1563(2019-04): 学習済み方策の結果を再現しようとしたが、論文の値と一致しなかった。オープンで回答なし。
- #1609(2019-06): 比較に使った調整済み Rainbow のハイパーパラメータの所在を尋ねた。オープンで回答なし。
- #1830(2020-07): 推奨バージョンの組み合わせで実行時にモジュールのエラーが出た。コメント欄に回避策が共有されている。
- #1891(2021-06)と #1579(2019-05): 実行時のエラーの報告。いずれも回答なし。

[tensor2tensor rl README (2019/03)] [tensor2tensor issue #1563 (2019/04)] [tensor2tensor issue #1609 (2019/06)] [tensor2tensor issue #1830 (2020/07)] [tensor2tensor issue #1891 (2021/06)] [tensor2tensor issue #1579 (2019/05)]。リポジトリがアーカイブ済みであることは §2.3 に書いた。

**第三者の PyTorch による再実装が 1 件ある(thomas-schillaci/SimPLe)。** README は 6 環境の結果を論文の値と並べている。各環境 1 回だけのフル学習で、スコアは実環境での平均累積報酬の最大値だと注記している。論文の値に対する比率は、Alien 137.7%、Freeway 108.9%、Kangaroo 198.1%、Krull 82.6%、MsPacman 89.3% である。表に載っている論文の値は SimPLe v1 の値と一致する(角度 B の担当者の照合)[Schillaci SimPLe-PyTorch (2020/10), README]。

## 6. Atari 100k ベンチマークの起点として

### 6.1 起点が SimPLe だとする記述

**後続の論文と公式ドキュメントの多くが、Atari 100k を SimPLe(Kaiser ら)が導入したものと明記している。**
- EfficientZero: Atari 100k は SimPLe が最初に提案し、多くのサンプル効率の研究が使っていると書く [Ye EfficientZero (2021/10), §5.1 p.7]。
- DrQ: Kaiser らが最近導入したベンチマークとして紹介している [Kostrikov DrQ (2020/04), §4.2]。
- TWM: Kaiser ら(2020)が、ALE の 26 ゲームで 100K 相互作用(400K フレーム、約 2 時間)に制限する設定を提案したと説明している [Robine TWM (2023/03), 実験節の冒頭]。
- BBF: Kaiser ら(2020)が Atari 100K を導入したとしている [Schwarzer BBF (2023/05), §3 p.3]。
- DIAMOND: SimPLe が world model を Atari に適用し、サンプル効率に焦点を当てた Atari 100k ベンチマークを導入したと書く [Alonso DIAMOND (2024/05), §7]。
- DreamerV3: Atari100k の説明(26 ゲーム、40 万フレーム、ゲーム時間 2 時間相当)に Kaiser らの文献番号を付けている。Nature 版には「SimPLe の元の Atari100k プロトコルからの逸脱」という題の拡張データ表がある(PMC の HTML 抽出)[Hafner DreamerV3 (2023/01), v2 p.9] [Hafner DreamerV3 Nature (2025/04), Extended Data Table 3]。
- EDELINE: SimPLe が world model を Atari に持ち込み、Atari 100k を提案して基礎を築いたと書いている [Lee EDELINE (2025/02), §2.2]。
- Agarwal ら: Atari 100k を Kaiser らに由来する ALE の派生ベンチマークとして扱っている [Agarwal Precipice (2021/08), §3 p.4]。
- Dopamine の公式 README: Atari 100k のリンク先が SimPLe の arXiv である [Dopamine atari_100k (2026/09), README]。

**SPR だけは、この設定を Kaiser ら(2019)と van Hasselt ら(2019)の 2 本に帰属させている。** [Schwarzer SPR (2020/07), §4 p.6]。個人ブログでも、Google DeepMind の Pablo Samuel Castro が、Kaiser らが SimPLe とともに相互作用を 10 万回に制限したベンチマークを提案し、それが後に「Atari 100k」と呼ばれるようになったと書いている [psc-g.github.io (2024/12)]。

### 6.2 評価手順の食い違い

**Atari 100k の評価手順は論文ごとに食い違っており、その点を複数の論文と公式実装が指摘・明記している。**
- Agarwal ら: 既報は主に 3 run か 5 run で、まれに 10 run や 20 run と、run 数がそろっていない。学習中の最大評価値を報告する方式(CURL、SUNRISE)は、最終性能の報告とは比べられないとし、DER を CURL の方式で評価すると CURL の報告値を大きく超えることを示した。DrQ のコードは評価に非標準の ε を使っており、標準に直した DrQ(ε) を別に用意した [Agarwal Precipice (2021/08), §3 p.4–5・Figure 5・脚注 5]。
- SPR: 1 ゲームあたり 500,000 フレームだけで評価する方式では、1 エピソードが最長 108,000 フレームあるため完了するエピソードが 4 本しか取れない場合があり、問題だとした。長さに関係なく 100 エピソードで評価することを推奨し、seed を 10 に増やした [Schwarzer SPR (2020/07), §4.1 p.8]。
- DrQ: Kaiser らと van Hasselt らを慣行として挙げ、学習の終了時に 125k 環境ステップで評価して 5 seed で平均した [Kostrikov DrQ (2020/04), §4.2]。
- BBF: Atari 100K には sticky actions がなく、TWM・IRIS・DreamerV3 も使っていないと指摘した [Schwarzer BBF (2023/05), §6 p.8・脚注 3]。
- DreamerV3 v1: 手法ごとに、ライフ情報の利用、early reset、評価用の別エピソード、解像度の変更、ゲームごとのハイパーパラメータが違うと一覧にした [Hafner DreamerV3 (2023/01), v1 Table T.1 p.35]。
- Dorner(プレプリント): 1,000 万フレーム未満の結果は 26 ゲームの中央値で、no-op 行動を使ったかどうかが分からないと述べている [Dorner (2021/02), 付録 B.1]。

**Dopamine の Atari 100k の設定は、sticky actions を使わないことをコメントで明記している。** 評価時の ε とエピソード長の上限は次のとおりである。

```
sticky_actions = False(コメントで Atari 100K は sticky actions を使わないと明記)
epsilon_eval: DER・OTRainbow・DrQ(ε) = 0.001 / DrQ = 0.05
max_steps_per_episode = 27_000(エージェントのステップ数)
報告値: 学習後に 100 エピソードの平均(README)
```

[Dopamine atari_100k (2026/09), configs/*.gin・README]

**人間正規化の基準値も論文ごとに違い、各論文が出典を明記している。** van Hasselt らは van Hasselt ら(2016)、Kielak は人間スコアに Mnih ら(2015)、SPR は Wang ら(2016)を使っている。SimPLe 自身は Pohlen ら(2018)の "Avg. Human" を使った [vanHasselt DER (2019/06), NeurIPS 版 §4.2] [Kielak OTRainbow (2020/03), §5] [Schwarzer SPR (2020/07), §4] [Kaiser SimPLe v4 (2020/02), Appendix D]。

### 6.3 26 ゲームの外側

**BBF は、Atari 100K が ALE の 55 ゲームのうち 26 しか使わず、sticky actions もない点を挙げたうえで、含まれない 29 ゲームの方が有意に難しいと報告した。** 同時に、BBF が DQN と比べて Atari 100k に過適合した証拠はないとしている [Schwarzer BBF (2023/05), §6 p.8・Figure 11]。


## 7. 後続の world model 研究

本節では、SimPLe の後に Atari 100k で評価された world model 系のエージェント(2021〜2026 年)を扱う。集計値はどれも各論文の自己申告である。評価手順(seed 数、sticky actions、評価エピソードなど)は論文ごとに違うため(§6.2)、論文をまたいだ単純な比較はできない。企業が主体の world model の取り組みは §11 で扱う。

### 7.1 Atari 100k で評価した world model 系エージェント

**SimPLe の後、Atari 100k で評価した world model 系エージェントは少なくとも 21 本ある。** 自己申告の平均 HNS は 1.0〜2.4 の範囲にある。後続論文の表に載る SimPLe の値は 0.33 か 0.44 である(§7.3)。表は arXiv の初出順に並べた。「表の SimPLe 列」はその論文の比較表に SimPLe があるか、「本文の言及」は本文で SimPLe に触れているかを示す。

| 手法 | 初出 / 会場 | world model の種類(出典の記述の要約) | 自己申告の集計値(平均 / 中央値 / IQM) | 表の SimPLe 列 | 本文の言及 | 被引用数 |
|---|---|---|---|---|---|---|
| EfficientZero | 2021-10 / NeurIPS 2021 | MuZero Reanalyze が土台。隠れた抽象状態の上のモデルと MCTS | 1.943 / 1.090 / —(Table 1) | あり(0.443 / 0.144) | あり | 340 |
| IRIS | 2022-09 / ICLR 2023 | 離散オートエンコーダと自己回帰 Transformer | 1.046 / 0.289 / 0.501(Table 1) | あり(0.332 / 0.134) | あり | 373 |
| DreamerV3 | 2023-01 / Nature 2025 | RSSM(表現を softmax 分布のベクトルからサンプリング) | v1 112% / 49%(Table S.1)、v2 125% / 49%(Table 9) | あり(33% / 13%) | あり | 1,415(arXiv 版) |
| TWM | 2023-03 / ICLR 2023 | Transformer-XL と確率的な離散潜在状態 | 0.956 / 0.505 / —(Table 1) | あり(0.332 / 0.134) | あり | 167 |
| HarmonyDream | 2023-09 / 会場は未確認 | DreamerV3 系 | 平均 136.5%(HTML 抽出) | 未確認 | 語なし(HTML 抽出) | 28 |
| STORM | 2023-10 / NeurIPS 2023 | Transformer とカテゴリカル VAE | 126.7% / 58.4% / —(Table 2) | あり(33% / 13%) | あり | 150 |
| Hieros | 2023-10 / 会場は未確認 | S5 層を使う world model | 120 / 56 / 53(Table 1、25 ゲーム) | あり(34 / 11 / 13) | あり | 10 |
| REM | 2024-02 / ICML 2024 | トークン型(RetNet と並列の観測予測) | 1.222 / 0.280 / 0.673(Table 1) | あり(0.332 / 0.134) | あり | 15 |
| EfficientZero V2 | 2024-03 / ICML 2024 | EfficientZero の拡張(木探索) | 2.428 / 1.286 / —(Table 1) | あり(0.443 / 0.144) | あり | 45 |
| DIAMOND | 2024-05 / NeurIPS 2024 | 画像空間の拡散 world model | 1.459 / 表になし / 0.641(Table 1) | あり(0.332、IQM 0.130) | あり | 334 |
| DART | 2024-06 / ICML 2024 | VQ-VAE のトークンと GPT 型の world model、方策は ViT | 1.022 / 0.790 / 0.575(Table 1) | なし | あり | 11 |
| Δ-IRIS | 2024-06 / ICML 2024 | 時刻間の確率的な差分を符号化する離散 AE と自己回帰 Transformer | 1.39 / 表になし / 0.65(付録 C Table 8) | あり(0.33、IQM 0.13) | あり(付録 C) | 36 |
| Drama | 2024-10 / ICLR 2025 | Mamba-2 の系列モデルと VAE | 105% / 27% / —(Table 1) | あり(33% / 13%) | あり | 8 |
| OC-STORM | 2025-01 / 会場は未確認 | STORM に物体の特徴を加えたもの | 134.8% / 43.8%(HTML 抽出) | なし | あり(HTML 抽出) | 8 |
| EDELINE | 2025-02 / プレプリント | 拡散と Mamba | 1.866 / 0.817 / 0.940(Table 1、3 seed) | あり(0.332 / 0.134) | あり | 4 |
| Simulus | 2025-02 / プレプリント | モジュール構成のトークン型 | 1.645 / 0.982 / 0.990(Table 9) | なし | 語なし(Kaiser らはベンチマークの出典として引用) | 4 |
| TWISTER | 2025-03 / ICLR 2025 | Transformer、カテゴリカル VAE、行動条件付きの対照予測符号化 | 162% / 77% / —(Table 2) | あり(33% / 13%) | あり | 30 |
| EMERALD | 2025-07 / ICML 2025 | 空間的な潜在状態と MaskGIT による予測 | 134% / 51% / —(付録 Table 11) | あり(33% / 13%) | あり | 5 |
| DyMoDreamer | 2025-09 / NeurIPS 2025 | RSSM と dynamic modulation | 156.6% / 71.3% / —(付録 Table 2) | なし | あり | 2 |
| EAWM | 2026-01 / ICLR 2026 | イベントを認識する表現(DreamerV3・Simulus の上に構築) | EADream 1.290 / 0.651 / 0.593、EASimulus 1.818 / 0.773 / 1.004(Table 1) | なし | 語なし(導入で Kaiser らを引用) | 1 |
| Optimistic WM | 2026-02 / プレプリント | DreamerV3・STORM に楽観的な探索を加えたもの | O-DreamerV3 の平均 152.68%(本文 p.7) | なし | なし(HTML 抽出) | 1 |

[Ye EfficientZero (2021/10)] [Micheli IRIS (2022/09)] [Hafner DreamerV3 (2023/01)] [Robine TWM (2023/03)] [Ma HarmonyDream (2023/09)] [Zhang STORM (2023/10)] [Mattes Hieros (2023/10)] [Cohen REM (2024/02)] [Wang EZ-V2 (2024/03)] [Alonso DIAMOND (2024/05)] [P. Agarwal DART (2024/06)] [Micheli Δ-IRIS (2024/06)] [Wang Drama (2024/10)] [Zhang OC-STORM (2025/01)] [Lee EDELINE (2025/02)] [Cohen Simulus (2025/02)] [Burchi TWISTER (2025/03)] [Burchi EMERALD (2025/07)] [Zhang DyMoDreamer (2025/09)] [Peng EAWM (2026/01)] [Mete OWM (2026/02)]

表の world model の種類と集計値は、各論文の要旨・本文・表から角度 C の担当者が転記したものである。「HTML 抽出」と書いた項目は、PDF のページではなく arXiv の HTML 版から取った。

### 7.2 後続論文による SimPLe の位置づけ

**後続論文による SimPLe の位置づけは、主に 3 通りに分けられる(本調査の整理)。ベンチマークを作った研究、world model の中で方策を学ぶ初期の例、画素空間の world model の初期の例である。**
- ベンチマークの起点: EfficientZero、DIAMOND、EDELINE などが、SimPLe が Atari 100k を提案・導入したと書いている(§6.1)[Ye EfficientZero (2021/10), §5.1] [Alonso DIAMOND (2024/05), §7] [Lee EDELINE (2025/02), §2.2]。
- 想像の中で学ぶ例: IRIS は、Ha & Schmidhuber(2018)がトイ環境で作った想像ベースのエージェントに続き、SimPLe がより難しい Atari 100k で有望さを示したと位置づけている [Micheli IRIS (2022/09), p.1]。TWM は、EfficientZero のように計画に使う系統と、SimPLe のように想像の中で行動を学ぶ系統を対比している [Robine TWM (2023/03), p.1]。Δ-IRIS は、想像の中で学ぶ model-based エージェント 4 本のうちの 1 本として SimPLe を比較に入れた [Micheli Δ-IRIS (2024/06), 付録 C]。DIAMOND は、自身の学習手順が Kaiser ら(2019)などと同様だと書いている [Alonso DIAMOND (2024/05), §3.2]。
- 画素空間の world model の例: TWISTER と EMERALD は、SimPLe を画像データに適用された最初期の model-based 手法の 1 つとし、畳み込みオートエンコーダで画素空間の world model を学ぶと説明している。EMERALD は、画素空間で再構成した軌道から方策を学ぶ系統(Δ-IRIS、DIAMOND)を、SimPLe と同様のやり方と説明している [Burchi TWISTER (2025/03), §2.1] [Burchi EMERALD (2025/07), §2.1・§2.2]。STORM は、系列モデルに LSTM を使う RNN 系の world model として SimPLe を挙げている [Zhang STORM (2023/10), p.2・Figure 1]。

**否定的な位置づけもある。**
- Dreamer(2019-12)は、Atari の実験で SimPLe を比較相手に入れた。付録では、world model だけから学ぶエージェントはまだ競争力がないという記述の根拠に Kaiser らを引いている [Hafner Dreamer (2019/12), p.8 §6・p.16 付録 C]。
- EfficientZero は、Kaiser らや Hafner らのような画像入力の model-based 手法が想像の rollout を使うことについて、モデルを搾取する(model exploitation)危険があると書いている [Ye EfficientZero (2021/10), §2.3]。
- OC-STORM は、SimPLe が Atari へ考えを広げたが効率は限られていた、と書いている(HTML 抽出)[Zhang OC-STORM (2025/01), 関連研究]。
- DreamerV2 の位置づけは §5.4 に書いた。

**Hieros の SimPLe の説明は、SimPLe 論文の記述と食い違っている(本調査の照合)。** Hieros は SimPLe を、環境の画素入力で直接 PPO を学ぶ手法として説明し、world model の想像の軌道で方策を学ぶ TWM・IRIS・DreamerV3 と対比している。SimPLe 論文では、方策は world model の中で行動して更新される(§1.1)[Mattes Hieros (2023/10), §3 p.6] [Kaiser SimPLe v4 (2020/02), Figure 1]。

### 7.3 比較表に載る SimPLe の値の 2 系統

**後続論文の比較表に載る SimPLe の集計値は、平均 0.332 / 中央値 0.134 と、平均 0.443 / 中央値 0.144 の 2 系統に分かれる。** 本項だけは、world model 系に限らず model-free の論文(CURL、DrQ、SPR など)の比較表も対象に含めた。 0.332 の系統は IRIS 以降の world model の論文の大半が使い、0.443 の系統は CURL・DrQ・SPR・EfficientZero・EfficientZero V2 が使っている。

```
系統      平均 HNS  中央値 HNS  IQM    採用している論文
0.332 系  0.332     0.134       0.130  IRIS、TWM、REM、DIAMOND、EDELINE、Δ-IRIS(0.33)、
                                       DreamerV3・STORM・Drama・TWISTER・EMERALD(33% / 13%)
0.443 系  0.443     0.144       —      CURL(44.3% / 14.4%)、DrQ(中央値のみ)、SPR、EfficientZero、EfficientZero V2
その他    Hieros 34 / 11 / 13(25 ゲームで評価)、OTRainbow の中央値 10.17%(独自の人間の基準値)、
          DER(SimPLe v1 のゲーム別の値)
```

[Micheli IRIS (2022/09), Table 1] [Robine TWM (2023/03), Table 1] [Cohen REM (2024/02), Table 1] [Alonso DIAMOND (2024/05), Table 1] [Lee EDELINE (2025/02), Table 1] [Micheli Δ-IRIS (2024/06), Table 8] [Hafner DreamerV3 (2023/01), v1 Table S.1・v2 Table 9] [Zhang STORM (2023/10), Table 2] [Wang Drama (2024/10), Table 1] [Burchi TWISTER (2025/03), Table 2] [Burchi EMERALD (2025/07), Table 11] [Srinivas CURL (2020/04), §6.2] [Kostrikov DrQ (2020/04), Table 5] [Schwarzer SPR (2020/07), Table 1] [Ye EfficientZero (2021/10), Table 1] [Wang EZ-V2 (2024/03), Table 1] [Mattes Hieros (2023/10), Table 1] [Kielak OTRainbow (2020/03), Table 4]

**2 系統の違いは、SimPLe v4 の付録の 2 つの列のどちらを使うかから来ている。** v4 の表 2 には、world model を長く学習させた "SD long" と、既定の学習の "SD" の 2 列がある。表 3 の SimPLe 列は "SD long" と同じ値である(§1.4)。0.332 系のゲーム別の値は "SD long" と一致する。IRIS と TWM は、Agarwal ら(2021)を経由して SimPLe の既報の 5 run を使ったと明記している。0.443 系のゲーム別の値は、26 ゲームすべてで 2 列の大きい方と一致する(角度 C の担当者の照合。下の 4 ゲームは本調査でも v4 の表 2 で確認した)。CURL 自身も、SimPLe と人間の基準値は先行研究(Kielak、van Hasselt ら)で報告の仕方が違うため、ゲームごとに既報の最良値を採ったと書いている [Kaiser SimPLe v4 (2020/02), Table 2・Table 3] [Micheli IRIS (2022/09), §3.2] [Robine TWM (2023/03), Figure 3 の注記] [Srinivas CURL (2020/04), §5.3]。

```
ゲーム     v4 表 2 "SD long"   v4 表 2 "SD"(= v1 の値)   0.332 系の論文   0.443 系の論文   DER
Alien      616.9               405.2                      616.9            616.9            405.2
Freeway    16.7                20.3                       16.7             20.3             20.3
Kangaroo   51.2                323.1                      51.2             323.1            323.1
Krull      2204.8              4539.9                     2204.8           4539.9           4539.9
```

[Kaiser SimPLe v4 (2020/02), Table 2] [vanHasselt DER (2019/06), arXiv v1 付録 E Table 1]

**評価の方法を扱った Agarwal らの再評価には、SimPLe は入っていない。** Agarwal らは Atari 100k の 5 手法(DER・OTR・DrQ・CURL・SPR)を 100 run ずつ評価し直したが、SimPLe は対象に含めていない。SimPLe は performance profile の図に既報の 5 run で載っているだけである [Agarwal Precipice (2021/08), §3 p.4・Figure 7 の注記 p.7]。IRIS と TWM は、SimPLe については Agarwal らを経由して既報の 5 run を使ったと明記している [Micheli IRIS (2022/09), §3.2] [Robine TWM (2023/03), Figure 3 の注記]。

### 7.4 後継を名乗る記述の有無

**調べた論文の中に、自分の研究を「SimPLe の後継」「SimPLe の拡張」と書いた箇所はなかった。** 近い表現は 3 つある。
- DIAMOND: 学習の手順が Kaiser ら(2019)と同様だと書いている [Alonso DIAMOND (2024/05), §3.2]。
- EMERALD: 画素空間の world model の系統を SimPLe と同様だと説明している [Burchi EMERALD (2025/07), §2.2]。
- DER: SimPLe に一部触発されたと書いている。ただし DER は model-free の論文である [vanHasselt DER (2019/06), 導入部]。

「拡張」という語は、EfficientZero V2 が EfficientZero に対して使っているもので、SimPLe に対してではない [Wang EZ-V2 (2024/03), p.2]。

## 8. 第三者の報道

### 8.1 英語

**同時期の英語の報道で確認できたのは Synced(2019-03-19)の 1 本である。** 記事の大半は、FinBrain Technologies 創業者と紹介された Ahmet Salim Bilgin への Q&A である。回答は SimPLe の限界として 2 点に触れている。最終スコアが最良の model-free 手法より低いことと、同じゲームでも実行ごとの差が大きいことである [Synced (2019/03), Q&A]。

**Synced の記事には、手法の分類を取り違えた記述がある。** 記事冒頭は SimPLe を完全に model-based の手法と説明している。一方、Q&A の回答の 1 つは SimPLe を「model-free アルゴリズムへの新しいアプローチ」と表現している [Synced (2019/03), 冒頭段落・Q&A "What impact might this research bring…"]。

**大手の英語テックメディアで、同時期に SimPLe を取り上げた記事は見つからなかった。** 次の記事を開いたが、SimPLe への言及はなかった。
- VentureBeat の関連記事 2 本(2020-02 と 2021-03)
- ニュースレター Import AI の 2019 年 3 月の 4 号(136〜139 号)

MIT Technology Review、The Next Web、ZDNet、MarkTechPost は、検索で該当記事を見つけられなかった [VentureBeat (2020/02)] [VentureBeat (2021/03)] [Import AI (2019/03)]。

### 8.2 中国語

**中国語では、2019 年 3 月に少なくとも 3 媒体が報じた。** 見出しと本文の書き方は媒体によって違い、論文の「約 100K 相互作用=実プレイ約 2 時間」を訓練時間の短さとして見せる書き方もある。
- 量子位(信息化观察网の転載、2019-03-07): 見出しで「訓練に 2 時間もかからない」「前例のない予測能力」を掲げた。論文の「100K 相互作用は実プレイ約 2 時間に相当」を、訓練の所要時間として表現している [量子位(信息化观察网转载) (2019/03), 見出し・本文冒頭] [Kaiser SimPLe v4 (2020/02), Abstract]。
- 新智元(搜狐の転載、2019-03-26): リード文は「従来の最先端手法より効率が 2 倍以上」と書き、本文では「大多数の場合に」という条件を付けている。Google のブログも「多くの場合」に 2 倍以上という書き方である [新智元(搜狐转载) (2019/03), 导读・本文] [Google Research Blog (2019/03), 結果の段落]。
- 雷锋网 AI 科技评论(2019-03-26): Google のブログを編訳したと明記した記事で、2019-05 に腾讯云开发者社区へ転載された [雷锋网 AI 科技评论 (2019/03), 冒頭] [腾讯云开发者社区 (2019/05), 冒頭]。

### 8.3 ポーランド語・日本語

**ポーランドの ITwiz(2020-02-17)は、deepsense.ai の Błażej Osiński(Senior Data Scientist)へのインタビューで、この研究に触れている。** 主題は Volkswagen 向けの sim-to-real である。Atari の研究については、Google Brain と米国の大学との共同研究で、最小限のやり取りで学び、システム内部に世界のシミュレータを作る手法だと説明している [ITwiz (2020/02), Atari プロジェクトに関する回答]。

**日本語の報道記事は見つからなかった。** InfoQ Japan の 2 本と日経Robotics の 1 本を開いたが、言及はなかった [InfoQ Japan (2020/09)] [InfoQ Japan (2024/10)] [日経Robotics (2021/09)]。日本語での流通は、§9 の個人ブログによる翻訳・要約である。

## 9. 技術ブログ・解説

**日本語では、2019 年に 1 件の要約と 2 件のブログ翻訳が出ている。**
- arXivTimes(GitHub issue #1128、2019-03-06、icoxfog417)は一言要約である。Dyna をベースに、CNN の Encoder/Decoder で環境をモデル化し、多くのゲームで PPO や Rainbow より効率が良い、とまとめている [arXivTimes (2019/03), issue 本文]。
- WebBigData(2019-03-28、dahara1)と note(2019-10-23、npaka)は、Google のブログの日本語訳である。訳者のコメントは付いていない(WebBigData で確認)[WebBigData (2019/03)] [note (2019/10)]。

**英語と中国語のブログには、発表の再構成、参考文献としての言及、翻訳ノート、自動要約の転載がある。**
- techsutram(2019-03、Mandar Pise)は、deepsense.ai の発表を出典と明記して再構成した記事である [techsutram (2019/03), 冒頭]。
- BAIR Blog の MBPO の解説(2019-12-12、Michael Janner)は、この論文を参考文献欄に載せている。本文のどこで触れているかは特定できなかった [BAIR Blog (2019/12), References]。
- 博客园(2021-01-08)は論文の中国語訳ノートである [博客园 (2021/01)]。
- DEV Community(2024-04-11)には、自動要約サイトの要約が転載されている [DEV Community (2024/04)]。

**後年のブログでは、van Hasselt らの反論の紹介、Dreamer の説明での参照、ベンチマークの起点としての言及がある。**
- Shagun Sodhani(2020-07-02)は van Hasselt らの論文を要約し、変更を加えた Rainbow DQN が 26 ゲーム中 17 で SimPLe を上回ったと紹介している [Papers I Read (2020/07)]。
- horomary(2022-02-27)は Dreamer を、world model 上でエージェントを訓練する SimPLe 的な手法と、PlaNet の精度の高い world model を組み合わせたものと説明している [どこから見てもメンダコ (2022/02)]。
- Pablo Samuel Castro のブログ(2024-12-02)は、Kaiser らが相互作用を 10 万回に制限したベンチマークを SimPLe とともに提案し、それが後に「Atari 100k」(26 ゲーム)と呼ばれるようになったと書いている。Castro は自サイトで Google DeepMind(モントリオール)の senior staff research scientist と名乗っている。ベンチマークとしての扱いは §6 で詳しく述べる [psc-g.github.io (2024/12), Atari 100k の節]。

**動画の解説は、閲覧できたものがない。** YouTube の解説動画(Hacker News に 2020-07-20 投稿)は、現在は閲覧できない。Two Minute Papers でこの論文を扱った回は見つからなかった [Hacker News (2020/07), 投稿 23895224]。

## 10. SNS・掲示板

本節の反応数は取得時点の値である。X と Hacker News は 2026-09-24、Reddit はアーカイブを取得した時点(ax406x は 2019-06-10)の値である。

### 10.1 X

**2019-03-04 に @hardmaru が論文を紹介し、いいね 611・返信 7 を集めた。** 投稿の要旨は、単純な反復で world model を学ぶだけでデータ効率が最先端に近づき、必要なのは 10 万回(実時間 2 時間)だけ、というものである。3 分後の投稿では、自身の NeurIPS 2018 の講演の終盤でこの研究に触れたと書いている [X (2019/03), @hardmaru 投稿 1102561871620648960・1102562698741665794]。@hardmaru は David Ha のアカウントである。Milken Institute の講演者紹介によれば、過去に Google で Google Brain の日本チームを率いた Research Scientist で、現在は Sakana AI に所属する。投稿時点の所属は、年次が示されておらず確認できていない [Google Scholar David Ha (2026/09)] [Milken Institute (2026/09)]。

**著者本人(Kaiser、Levine、Finn、Erhan、Tucker、Michalewski、Sepassi)の 2019 年 3 月の投稿は確認できなかった(§14)。**

### 10.2 Reddit

**反応が最も大きかったのは r/MachineLearning のスレッド ax406x(2019-03-04)で、score 180・35 コメントである。** 投稿者は題名に、論文の題名にはない「2 時間の実時間プレイの後、多くの Atari ゲームで人間レベルの性能」という主張を付け加えていた [Reddit r/MachineLearning (2019/03), スレッド ax406x]。

**このスレッドでは、付け加えられた「人間レベル」の主張が誇張だと指摘された。** 最初の指摘は gwern(score 18)で、ほかの利用者が、論文中の human-level は DeepMind の先行論文を引いた 1 か所だけだと補足した。gwern はさらに、アブストラクトの「10 万回で competitive」が、10 万回の予算の下での比較だと読み取りにくいとも述べている [Reddit r/MachineLearning (2019/03), コメント ehrk0o4・ehryvxm・ehs0j87]。

**技術的な批判は 3 点に集まった。**
- sticky actions を使っていないこと。
- scheduled sampling が場当たり的だということ。確率的な環境での妥当性を問う声もあった。
- 4 フレームしか見ないことによる不確実性は、確率性ではなく部分観測だということ。

好意的な反応がある一方、Atari での model-based RL はほぼ無意味だとする否定的な声もあった [Reddit r/MachineLearning (2019/03), コメント ehr55e0・ehsccg2・ehra5qu・ehsd1p5・ehvqacd・ei2ru2w・ehr658z・ehs7nw4・ehsmqao]。

**著者の名前と同じか似た利用者名のアカウント(piotr_milos、babaeizadeh、koz4k)が、手法の側に立って説明している。** piotr_milos は、sticky actions と環境モデルの確率性は別の問題だと述べた。koz4k は損失をクリップする理由を、babaeizadeh は確率的 world model の節を示した。投稿の中で身元は明示されておらず、本人かどうかは確認していない [Reddit r/MachineLearning (2019/03), コメント ehsviy2・ehrofyf・ehsp9fl]。

**2020-02 には r/reinforcementlearning に「SimPLe への反証論文はどれか」を尋ねるスレッド(f37b56、score 7・4 コメント)が立った。** gwern が van Hasselt らの論文を挙げ、ほかの利用者がその論文を説得力があると評価した。筆頭格の著者の 1 人を名乗る blazej0 は、両方の論文を興味深い発展と認めた。そのうえで、巨大な観測空間を持つ Atari で model-based RL が可能だと示すのが本論文の主眼だった、という OpenReview での回答を紹介した [Reddit r/reinforcementlearning (2020/02), スレッド f37b56]。

**その他の投稿の反応は小さい。**
- r/reinforcementlearning の論文紹介(ax76wr): score 12
- Synced の記事のリンク: 5 つのサブレディットに投稿され、最大で score 15
- Google のブログのリンク: score 1

2020-05 には、neptune.ai の社員とみられる投稿者が ICLR 2020 の注目論文の強化学習部門にこの論文を挙げ、スレッド全体で score 219 を得ている [Reddit r/reinforcementlearning (2019/03)] [Reddit r/MachineLearning (2019/03), スレッド b5e7b8] [Reddit r/MachineLearning (2020/05), スレッド gn6j0m]。

### 10.3 Hacker News

**Hacker News での反応は小さく、関連する投稿 5 件のうち最大でも 58 points・3 コメントである。**

```
投稿 ID    日付         内容            points  コメント
19310771   2019-03-05   論文             1       0
19486005   2019-03-25   Google ブログ    7       0
20425135   2019-07-12   論文             3       0
20728585   2019-08-18   論文            58       3
23895224   2020-07-20   YouTube 解説     3       0
```

[Hacker News (2019/03)] [Hacker News (2019/07)] [Hacker News (2019/08)] [Hacker News (2020/07)]

**58 points の投稿についた 3 件のコメントは、次のとおりである。**
- 批判: Ben Recht の連載を引き、著者らは功績を主張するのが早すぎ、モデルを持つこと自体の効果と計算資源の効果を切り分けていない。
- 反論: ここでのモデルは人手で作ったものではなく、自己教師で学習したものだ。
- 紹介: van Hasselt らの論文を挙げ、条件を揃えれば Rainbow DQN が上回ったと要約した。

[Hacker News (2019/08), コメント 20728901・20729040・20729359]

### 10.4 後継研究の発表時の言及

**IRIS・DIAMOND・GameNGen・Genie・Genie 2 の Hacker News スレッドでは、SimPLe への言及が見つからなかった。** これらのスレッドには多くの反応が集まっている。
- GameNGen: 1149 points・409 コメント
- Genie 2: 1247 points・410 コメント

全コメントを走査したが、SimPLe、Kaiser、arXiv ID、論文の題名はどれも出てこなかった [Hacker News (2022/09), 投稿 32692404] [Hacker News (2024/10), 投稿 41826402] [Hacker News (2024/08), 投稿 41375548] [Hacker News (2024/02), 投稿 39509937] [Hacker News (2024/12), 投稿 42317903]。

**X では、@hardmaru が 2020-10-14 に離散潜在空間の world model の論文を紹介したとき、SimPLe(2019)に触れている。** 紹介した論文が、はるかに大きい SimPLe のモデルと 10 万回の条件で比べた、という内容である。いいね 292・返信 3 である [X (2020/10), @hardmaru 投稿 1316251053638127616]。

## 11. 産業シグナル

本節では、企業または企業の研究所が主体となって 2019〜2026 年に発表した取り組みを扱う。対象は、動画モデルや world model で環境を模擬し、エージェントの学習・評価や対話的な生成に使うものである。それぞれについて、出典が SimPLe を参照しているかを確かめた。SimPLe との関連は、論文・技術報告の参考文献や本文、公式ブログのリンクで SimPLe を実際に確認できた場合だけを書く。確認方法は 4 通りである。PDF の目視、HTML の抽出、Crossref の参考文献の全件照合、Semantic Scholar の参考文献データでの照合である。Google の Dreamer 系の論文(Dreamer、DreamerV2、DreamerV3)は §5.4 と §7 で扱い、本節では公式ブログのリンクだけを書いた。

### 11.1 SimPLe を参照している企業の取り組み

**NVIDIA の GameGAN(2020-05、CVPR 2020)は、関連研究で SimPLe を「world model で Atari ゲームを生成した同時期の研究」として挙げている。** GameGAN は、プレイ画面とキー入力からゲームを模倣する生成モデルで、学習した模擬環境の中で RL エージェントを訓練して実環境で試す評価もしている。参考文献番号 1 は、著者名が Anonymous の、ICLR 2020 に査読中の "Model based reinforcement learning for atari" である。題名と投稿先は SimPLe の ICLR 版と一致する [Kim GameGAN (2020/05), p.3 §2・p.8 §4.2・p.10 参考文献]。NVIDIA の公式技術ブログは、GameGAN が PAC-MAN を 5 万エピソードから学び、ゲームエンジンなしで遊べる版を生成したと発表した。ブログに SimPLe の語はない [NVIDIA GameGAN blog (2020/05)]。

**NVIDIA の DriveGAN(2021-04、CVPR 2021)は、行動に応じて運転映像を生成するニューラルシミュレータである。** 関連研究の「データ駆動のシミュレーションと model-based RL」の段落で、エージェントが行動の計画に使うダイナミクスモデルの例として Kaiser ら(arXiv:1903.00374)を挙げている [Kim DriveGAN (2021/04), §2・参考文献 25]。

**Google Research の Pathdreamer(2021-05、ICCV 2021)は、序論で SimPLe を 2 つの記述の根拠に引いている。** 1 つは、model-based 手法が深層 RL のサンプル効率を高めること。もう 1 つは、画像を生成する world model がこれまで Atari などの単純な環境に限られてきたことである。参考文献の題名と会場は SimPLe と一致するが、著者の並び順は SimPLe 論文と異なる [Koh Pathdreamer (2021/05), §1・参考文献 62]。Pathdreamer を紹介した Google の公式ブログ(2021-09)には、SimPLe の語もリンクもない [Google Pathdreamer blog (2021/09)]。

**Wayve の GAIA-1(2023-09)は、運転用の生成的 world model である。** 関連研究には、world model が環境のシミュレータとして RL のサンプル効率の問題を解決しうる、と述べる文があり、この文に付いた 7 件の文献の 1 つが Kaiser ら(ICLR 2020)である。同じ文は、それには world model が環境を正確に模していることが前提だとも述べている [Hu GAIA-1 (2023/09), 関連研究・参考文献 85]。Wayve の公式ブログ(2023-10)は GAIA-1 を学習・検証用のデータを生成するシミュレータと位置づけており、SimPLe の語はない [Wayve Scaling GAIA-1 blog (2023/10)]。

**Waabi の Copilot4D(2023-11、ICLR 2024)は、点群の観測を離散拡散で予測する自動運転向けの world model である。** 序論で world model と model-based RL が成果を上げた領域を挙げる箇所で、Atari の例として Kaiser ら(2019)を引いている [Zhang Copilot4D (2023/11), §1]。

**Google の公式ブログでは、Dreamer の紹介記事(2020-03)と DreamerV2 の紹介記事(2021-02)が、SimPLe のブログ記事にリンクしている。** DreamerV2 の記事の位置づけは §3.4 に書いた。論文側の扱いは §7 にまとめた [Google Dreamer blog (2020/03)] [Google DreamerV2 blog (2021/02)]。

### 11.2 SimPLe を参照していない企業の取り組み

**参考文献を確認できた範囲で、2024-02 以降の企業主体の論文・技術報告のうち SimPLe を参照していたのは DreamerV3 の Nature 版(2025-04)だけだった。** それ以外は次の表のとおり、参考文献に SimPLe がない。なお Microsoft Research の研究者が共著に入った DIAMOND(2024-05)は SimPLe を参照しているが、大学主導の研究なので §7 で扱う。

| 取り組み | 組織 | 初出 | 出典が述べる用途 | 参考文献の確認方法 |
|---|---|---|---|---|
| UniSim | Google DeepMind ほか | 2023-10 | 学習したシミュレータで方策を学び実世界へ | Semantic Scholar・HTML |
| Genie | Google DeepMind | 2024-02 | ラベルなしの動画から学ぶ生成的な対話環境 | HTML(85 件)・Semantic Scholar |
| SIMA | Google DeepMind | 2024-03 | 市販ゲームなどで指示に従うエージェント | Semantic Scholar のみ |
| GameNGen | Google Research・Google DeepMind | 2024-08 | DOOM を実時間で模擬するゲームエンジン | HTML(44 件)・Semantic Scholar |
| 1X World Model | 1X | 2024-09・2025-06 | ロボットの方策を評価するシミュレータ | HTML・PDF 目視 |
| Oasis | Decart・Etched | 2024-10 | 人間がプレイする実時間の生成 | HTML(8 件) |
| Navigation World Models | Meta ほか | 2024-12 | 軌道を模擬して計画を立てる | Semantic Scholar・HTML |
| WHAM / Muse | Microsoft Research | 2025-02 | ゲームプレイの発想支援 | Crossref 全件(79 件) |
| GAIA-2 | Wayve | 2025-03 | 運転シナリオのシミュレーション | HTML・Semantic Scholar |
| WHAMM | Microsoft | 2025-04 | Quake II の実時間プレイ体験 | HTML(3 件) |
| MineWorld | Microsoft Research | 2025-04 | Minecraft の実時間の対話型 world model | Semantic Scholar・HTML |
| DreamGen | NVIDIA ほか | 2025-05 | 合成したロボットデータで方策を学ぶ | Semantic Scholar・HTML |
| V-JEPA 2 | Meta | 2025-06 | ゼロショットのロボット計画 | Semantic Scholar のみ(原文は未確認) |
| Hunyuan-GameCraft | Tencent | 2025-06 | 対話型のゲーム動画生成 | Semantic Scholar・HTML |
| Matrix-Game 2.0 | Skywork AI | 2025-08 | 実時間の対話型 world model | Semantic Scholar・HTML |
| Dreamer 4 | Google DeepMind | 2025-09 | world model の中で強化学習 | PDF 目視(84 件) |
| Cosmos-Predict2.5 | NVIDIA | 2025-10 | 合成データ、方策の評価、閉ループの模擬 | Semantic Scholar のみ |
| SIMA 2 | Google DeepMind | 2025-11 | Genie 3 が生成した世界で自己改善 | Semantic Scholar・HTML |
| Veo による方策の評価 | Google DeepMind | 2025-12 | ロボット方策の評価 | Semantic Scholar・HTML |

[Yang UniSim (2023/10)] [Bruce Genie (2024/02)] [SIMA Team (2024/03)] [Valevski GameNGen (2024/08)] [1X World Model blog (2024/09)] [1X World Model 2025 (2025/06)] [Decart Oasis (2024/10)] [Bar NWM (2024/12)] [Kanervisto WHAM (2025/02)] [Russell GAIA-2 (2025/03)] [Microsoft WHAMM (2025/04)] [Guo MineWorld (2025/04)] [NVIDIA DreamGen (2025/05)] [Meta V-JEPA 2 paper (2025/06)] [Tencent Hunyuan-GameCraft (2025/06)] [Skywork Matrix-Game 2.0 (2025/08)] [Hafner Dreamer 4 (2025/09)] [NVIDIA Cosmos-Predict2.5 (2025/10)] [SIMA Team SIMA 2 (2025/12)] [Gemini Robotics Team Veo eval (2025/12)]

**学習した環境モデルの中でエージェントを訓練する研究の系統を関連研究で挙げていても、SimPLe を含めていない例がある。**
- Genie: 行動を条件とする次フレーム予測モデルが実環境の経験なしの方策学習に使える、という説明で挙げているのは Ha & Schmidhuber(2018)、Hafner(2020・2021)、Micheli(2023)、Robine(2023)などである [Bruce Genie (2024/02), 関連研究]。
- GameNGen: 同じ系統として挙げているのは Ha & Schmidhuber(2018)と Hafner(2020)である [Valevski GameNGen (2024/08), §6]。
- Dreamer 4: 学習した環境モデルに基づく行動学習の長い歴史として、Sutton の Dyna、PILCO、E2C を挙げている [Hafner Dreamer 4 (2025/09), p.17]。

**参考文献リストを持たない公式ブログ・プレスリリースは、本文とリンクに SimPLe の語がなかった。** 対象は次のとおりである。
- Genie 2(2024-12)、Genie 3(2025-08)、Project Genie(2026-01)
- Waymo World Model(2026-02)
- GAIA-3(2025-12)、Runway GWM-1(2025-12)
- Muse(2025-02)、V-JEPA 2(2025-06)
- Cosmos と Cosmos 3 のプレスリリース

Genie 2 の記事は「Atari の初期の研究」の箇所で DQN の Nature 論文にリンクしている [DeepMind Genie 2 blog (2024/12)] [DeepMind Genie 3 blog (2025/08)] [Google Project Genie blog (2026/01)] [Waymo World Model blog (2026/02)] [Wayve GAIA-3 blog (2025/12)] [Runway GWM-1 (2025/12)] [Microsoft Muse blog (2025/02)] [Meta V-JEPA 2 blog (2025/06)] [NVIDIA Cosmos press (2025/01)] [NVIDIA Cosmos 3 press (2026/05)]。第三者の報道 3 本(Genie 3、Project Genie、Waymo World Model)にも SimPLe の語はなかった [heise online (2025/08)] [9to5Google (2026/01)] [THE DECODER (2026/02)]。

**参考文献を確認できなかったものが 3 件ある。**
- NVIDIA Cosmos(2025-01)と Cosmos 3(2026-06): PDF が大きく取得できず、HTML も参考文献の手前で切れた。
- OpenAI の Sora 技術報告(2024): openai.com が 403 を返した。

[NVIDIA Cosmos (2025/01)] [NVIDIA Cosmos 3 (2026/06)]

### 11.3 特許

**Google Patents で SimPLe の著者を発明者として検索した範囲では、題名から「動画予測モデルや学習したシミュレータの中で方策を学習する」と読める特許・出願は見つからなかった。** 検索は途中で Google のボット判定のページが返るようになり、止めた。キーワードでの検索と、一部の著者の権利者による絞り込みはできていない [Google Patents search (2026/09)]。

**近い内容の特許は 3 件あるが、確認できた範囲では、どれも動画予測モデルの中での方策学習を請求していない。**
- EP3402633B1(Google LLC、発明者 Levine・Finn・Goodfellow、2020-05 付与): ロボットの試行から、行動を条件として画素の動きの分布を予測するモデルを訓練する方法 [EPO publication server EP3402633B1 (2020/05), 請求項 1・5]。
- US20230239499A1(Google LLC、発明者に Babaeizadeh・Finn・Erhan・Levine ら、2023-07 公開): 過去のフレームから後続のフレームを予測する、畳み込み VAE を用いた動画予測モデル。請求項は確認していない [PubChem US20230239499A1 (2023/07)]。
- US12154212B2(Waymo LLC、発明者に Erhan、2024-11 付与): 実環境の観測から作った地図を使い、模擬環境のセンサデータを生成する方法。要約に方策学習の記載はない [PubChem US12154212B2 (2024/11)]。

## 12. 批判と留保

### 12.1 SimPLe に向けられた批判

**本書で確認した SimPLe への批判は、8 つの論点に分けられる(本調査の整理)。** 各論点の根拠は右端の節にある。

| 論点 | 批判の中身 | 主な出典 | 節 |
|---|---|---|---|
| 比較相手の強さ | 低データ向けに調整した model-free 手法(DER、OTRainbow)は、SimPLe と同等以上の成績を出す | van Hasselt ら、Kielak、SPR、EfficientZero。SimPLe v4 自身も互角と認めた | §5.1・§5.2・§2.2 |
| 計算コストと実時間 | 実データは少ないが、world model の学習と world model 内の 15.2M 回の相互作用に数週間かかる | 査読者 2 名、Kielak、SPR、IRIS、TWM、DreamerV2 | §4.2・§5.3 |
| 低データ域に限った評価 | 500K で model-free の PPO に並び、それ以上では優位が消える | 査読者 1 名、SimPLe §6.2、DreamerV2 | §1.4・§4.2・§5.4 |
| ゲームの選び方 | 26 ゲームは SimPLe か Rainbow で成績の良かったゲームで、選択バイアスがある。より易しい部分集合だという指摘もある | Korkmaz、DreamerV2。BBF は外側の 29 ゲームの方が難しいと報告 | §5.5・§6.3 |
| 評価手順 | 評価の単位、ALE の確率性の設定、停止基準が明らかでない。主な結果は sticky actions なし | Neves ら、Reddit の議論、BBF | §5.5・§6.2・§10.2 |
| 再現性 | 再現できないという issue が未回答のまま残り、コードは TensorFlow 1 系でリポジトリはアーカイブ済み | tensor2tensor の issue、GitHub | §2.3・§5.7 |
| 数値の揺れ | 版によって主張の強さと表の値が変わり、後続論文の表では 2 系統の値が使われている | SimPLe の各版、後続論文の比較表 | §2.2・§7.3 |
| 発信での誇張 | 「人間レベル」「訓練 2 時間」など、論文にない言い回しが広まった | Reddit のスレッドの題名、中国語の報道の見出し | §8.2・§10.2 |

### 12.2 批判の側の留保

**批判の中には、前提や根拠に限定が付くものがある。**
- DER の比較は SimPLe v1 の値に対するものである。SimPLe v4 は、SD long の値なら 13 対 13 だと応じている。DER の付録の表の太字の数(Rainbow 16・SimPLe 10)と本文の「17/26」は合わない(§2.2・§5.1)。
- Kielak の論文はプレプリントで、ICLR 2020 での採否は確認できていない(§5.2)。
- Korkmaz の選択バイアスの批判は 2026 年の単著論文で、2026-09-24 時点の被引用数は 0 である。BBF は、少なくとも BBF について Atari 100k への過適合の証拠はないと報告している(§5.5・§6.3)。
- 後続論文が挙げる「3 週間」「500 時間」は、SimPLe v3 が SD long の学習について書いた「3 週間以上」と一致する。SPR は、この値を v3 に基づくと注記している。既定の短い学習(45K/15K ステップ)の所要時間は、本調査の範囲では確認できなかった(§2.2・§5.3)。

**SimPLe の著者側の説明もある。**
- 査読への回答: 計算コストが model-free より大きいことを認めたうえで、実世界での経験の収集が高価または危険な分野(ロボティクス、自動運転)では利点があるとした(§4.3)。
- Reddit の投稿: 筆頭格の著者を名乗る利用者が、巨大な観測空間を持つ Atari で model-based RL が可能だと示すのが本論文の主眼だった、と説明した。本人かどうかは確認していない(§10.2)。

## 13. 総合評価

本節は、§1〜§12 で確認した事実をもとにした本調査の評価である。個々の事実の根拠は各節にある。

**SimPLe は、学習した画素空間の動画予測モデルの中だけで方策を学ぶ方式が、多数の Atari ゲームで低データの設定でも機能することを示した論文として受け止められた。** 査読者 3 名は全員が掲載に賛成し、決定コメントは結果の重要性が新規性の弱さを上回るとした(§4)。

**一方で、性能の主張は公開から数か月で、低データ向けに調整した model-free の手法に相殺された。** 著者自身も v4 で互角(13 対 13)と認めている(§2.2・§5.1・§5.2)。計算コスト、低データ域に限った評価、ゲームの選び方、評価手順については、査読の段階から 2026 年まで批判が続いている(§12.1)。

**最も長く残った影響は、評価設定としての Atari 100k である。** 2020〜2026 年の model-free と model-based の多数の論文、それに Dopamine の公式実装が、SimPLe を起点と明記してこの設定を使っている(§6.1)。この設定にも、26 ゲームの選び方と評価手順のばらつきへの批判がある(§5.5・§6.2)。

**手法としての SimPLe は、後続の world model 研究の比較表に、古いベースラインとして載り続けている。**
- 載っている値は平均 HNS 0.33 か 0.44 で、後続手法の自己申告値は 1.0〜2.4 に達している(§7.1・§7.3)。
- 自らを SimPLe の後継と名乗る論文は見つからなかった。後続研究は、SimPLe を想像の中で学ぶ初期の例、あるいは画素空間の world model の初期の例として扱っている(§7.2・§7.4)。
- 企業の world model の論文では、2020〜2023 年には参照されていた。2024-02 以降で参考文献を確認できたもののうち、SimPLe を参照していたのは DreamerV3 の Nature 版だけだった。Genie、GameNGen、Dreamer 4 は、学習した環境モデルでエージェントを学ぶ系統として別の文献を挙げている(§11)。

**被引用数と話題性は次のとおりである。**
- 被引用数は、集計元によって 1,016〜1,404 と幅がある。OpenAlex の arXiv 版レコードでは、年別の被引用は 2021 年の 101 件が最多で、2025 年は 16 件である(§2.1)。
- 同時期の報道は、英語では Synced の 1 本、中国語では 3 媒体に限られた。SNS で最大の反応は r/MachineLearning の score 180 のスレッドで、題名に付けられた「人間レベル」の主張は誇張だと指摘された(§8・§10)。

## 14. 調査の限界

**引用の方針**
- 私の運用ルール(著作物の逐語の引用は 1 回答につき 1 か所・15 語未満)に従い、本書の逐語引用は §2.2 の 1 か所だけにした。survey スキルが求める「主張ごとの逐語引用」の形式からは外れている。ほかの根拠は、場所を示したうえで要旨を書いた。

**一次資料に届かなかったもの**
- OpenReview: フォーラムと API がボット検証で開けなかった。査読の内容は、第三者のデータセット(PeerSum、MReD、Paper Copilot)による二次情報である。投稿日、決定日、議論期間中の PDF の改訂、Kielak の OpenReview ページは確認できていない。Wayback Machine のスナップショットも開けなかった。
- 発表区分(Spotlight): ICLR 公式サイトには区分を示す記載が見つからなかった。根拠は第三者のデータと、共著機関 deepsense.ai の告知である。
- 見られなかったサイト: dblp(ボット検証)、Nature の本体ページ(PMC の HTML で代用)、openai.com(403)。
- 被引用数: Google Scholar は WebFetch で 1 回取っただけの値で、変わりうる。DreamerV3 の Nature 版は、Semantic Scholar(605)と OpenAlex(147)で大きく違う。

**版と抽出の精度**
- 論文の版の比較は、arXiv が現在配信している PDF(v1〜v4 は 2024-12 に再生成)の抽出テキストで行った。当初配布された PDF とはページ区切りが違う可能性がある。図の比較は v4 と v5 の間だけで行った。
- 多くの後続論文と企業の論文は、PDF のページではなく HTML の抽出で確認した(§7.1・§11 の表に明記)。WebFetch の要約が誤った例を 2 回確認した。Korkmaz の該当箇所と GAIA-1 の参考文献番号で、どちらも原文の再照合で解消した。
- 「角度 X の担当者の照合」と書いた集計と照合(DER の太字の数、0.443 系のゲーム別の値の一致など)は、本調査では一部しか再確認していない。

**未確認・未解決**
- van Hasselt らの NeurIPS 版の補足資料は確認しておらず、「17/26」と付録の表の食い違いの理由は分からない。
- Agarwal らの arXiv 版 PDF は大きすぎて取得できず、付録は ar5iv の抽出経由でしか見ていない。
- サーベイの雑誌版(Moerland の FnT、Luo の Science China、Plaat の AI Review)は確認しておらず、読んだのは arXiv 版だけである。
- 後続研究: EfficientZero の IQM、Hieros のゲーム別の SimPLe の値、この一覧に入らない 2024〜2026 年の論文(MuDreamer、R2I など)は調べていない。
- 企業: NVIDIA Cosmos、Cosmos 3、OpenAI の Sora 技術報告は、参考文献を確認できなかった。V-JEPA 2、SIMA、Cosmos-Predict2.5 は Semantic Scholar の参考文献データだけで確認した。Tesla、xAI、World Labs などの取り組みは調べていない。
- 特許: Google Patents の検索は途中でボット判定に止められた。キーワードでの検索と、一部の発明者の権利者による絞り込みはできていない。個別の特許ページの多くは開けず、請求項は EP3402633B1 の一部しか確認していない。
- 再現性: tensor2tensor の issue は GitHub の検索 API で探したため、キーワードを含まない issue は漏れている可能性がある。第三者の再実装は 1 件しか見つからなかった。評価時の温度の設定や、エピソード長の上限の食い違いを指摘した第三者の記述は見つからなかった。

**SNS と報道**
- X: 検索が使えなかった。Wayback Machine にアーカイブされた投稿だけを読んだ。
  - 著者本人の 2019 年 3 月の投稿は確認できなかった。@lukaszkaiser の 2019-03-16〜03-25 ごろは、アーカイブの検索が失敗して未確認である。
  - @gwern は非公開で読めなかった。
  - 著者以外の @ylecun、@Miles_Brundage、@jackclarkSF の 2019 年 3 月のアーカイブ済みの投稿には、SimPLe への言及がなかった。
- Reddit: アーカイブの API(pullpush)で読んだ。途中でレート制限がかかったため、2021 年以降の言及は調べていない。
- Hacker News: Algolia の検索は大文字と小文字を区別しない。各クエリの上位 800 件を区別して絞り込んだため、取りこぼしがありうる。
- 開けなかったもの: 知乎、机器之心(JavaScript で描画される)、量子位の元記事、Medium の記事、ShortScience、neptune.ai の記事、YouTube の解説動画。
- 見つからなかったもの: VentureBeat、MIT Technology Review、The Next Web、ZDNet、MarkTechPost の同時期の報道、Two Minute Papers の回、日本語の報道記事。

## 15. 出典リスト

被引用数は、断りがなければ Semantic Scholar の値で、2026-09-24 に取得した。ラベルの年月は初出の年月で、読んだ版が違う場合は各項目に書いた。

### 15.1 SimPLe 本体

[Kaiser SimPLe v1, 2019/03] Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłoś, Błażej Osiński ほか 13 名(Google Brain / UIUC / University of Warsaw / deepsense.ai). "Model Based Reinforcement Learning for Atari." arXiv:1903.00374v1. https://arxiv.org/abs/1903.00374v1

[Kaiser SimPLe v2, 2019/03] 著者は v1 と同じ. "Model Based Reinforcement Learning for Atari." arXiv:1903.00374v2. https://arxiv.org/abs/1903.00374v2

[Kaiser SimPLe v3, 2019/06] v1 の著者に Afroz Mohiuddin(Google Brain)を加えた 14 名. "Model Based Reinforcement Learning for Atari." arXiv:1903.00374v3. https://arxiv.org/abs/1903.00374v3

[Kaiser SimPLe v4, 2020/02] Łukasz Kaiser ほか 14 名(Google Brain / deepsense.ai / Institute of Mathematics of the Polish Academy of Sciences / University of Warsaw / UIUC / Stanford University). "Model Based Reinforcement Learning for Atari." ICLR 2020(arXiv:1903.00374v4、採択版の体裁). https://arxiv.org/abs/1903.00374v4 — 本書の §1 は主にこの版を読んだ。被引用数は Semantic Scholar 1,016(influential 88)、OpenAlex 418(arXiv 版)+ 84(ICLR 版)、Google Scholar 1,404(主なクラスタ)

[Kaiser SimPLe v5, 2024/04] 著者は v4 と同じ. "Model Based Reinforcement Learning for Atari." arXiv:1903.00374v5. https://arxiv.org/abs/1903.00374v5

[arXiv abs 1903.00374, 2024/04] arXiv. "Model-Based Reinforcement Learning for Atari"(抄録ページ、Submission history). arXiv. https://arxiv.org/abs/1903.00374

[arXiv API 1903.00374, 2024/04] arXiv. export API の応答. arXiv. http://export.arxiv.org/api/query?id_list=1903.00374

[SimPLe project page, 日付不明] 作成者の表記は確認できていない. "Model-Based Reinforcement Learning for Atari." Google Sites. https://sites.google.com/view/modelbasedrlatari/home(https://goo.gl/itykP8 から 302 で転送、2026-09-24 確認)

### 15.2 公式発信

[Google Research Blog, 2019/03] Łukasz Kaiser, Dumitru Erhan(Research Scientists, Google AI。掲載時の肩書き). "Simulated Policy Learning in Video Models." Google Research Blog. https://research.google/blog/simulated-policy-learning-in-video-models/

[Google AI (X), 2019/03] @GoogleAI(Google の公式アカウント). 投稿 1110244037305327618. X. https://twitter.com/GoogleAI/status/1110244037305327618

[deepsense.ai, 2019/03] deepsense.ai(組織の署名). "deepsense.ai and Google Brain design artificial imagination for reinforcement learning." deepsense.ai blog. https://deepsense.ai/blog/deepsense-ai-and-google-brain-design-artificial-imagination-for-reinforcement-learning/

[PR Newswire, 2019/03] deepsense.ai(発信元). "deepsense.ai and Google Brain Design Artificial Imagination for Reinforcement Learning." PR Newswire. https://www.prnewswire.com/news-releases/deepsenseai-and-google-brain-design-artificial-imagination-for-reinforcement-learning-300816398.html

[deepsense.ai (X), 2019/03] @deepsense_ai(deepsense.ai の公式アカウント). 投稿 1108742905974394881. X. https://twitter.com/deepsense_ai/status/1108742905974394881

[deepsense.ai, 2020/01] deepsense.ai(組織の署名). "deepsense.ai paper to be featured at ICLR spotlight." deepsense.ai blog. https://deepsense.ai/blog/deepsense-ai-paper-to-be-featured-at-iclr-spotlight/

[Google Dreamer blog, 2020/03] Danijar Hafner(Student Researcher, Google Research). "Introducing Dreamer: Scalable Reinforcement Learning Using World Models." Google Research Blog. https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/

[Google DreamerV2 blog, 2021/02] Danijar Hafner(Student Researcher, Google Research). "Mastering Atari with Discrete World Models." Google Research Blog. https://research.google/blog/mastering-atari-with-discrete-world-models/

[Google Pathdreamer blog, 2021/09] Jing Yu Koh(Research Engineer), Peter Anderson(Senior Research Scientist, Google Research). "Pathdreamer: A World Model for Indoor Navigation." Google Research Blog. https://research.google/blog/pathdreamer-a-world-model-for-indoor-navigation/

### 15.3 査読・データセット・集計

[ICLR 2020 virtual site, 2020/04] ICLR. "Model Based Reinforcement Learning for Atari"(論文ページと papers.json). ICLR 2020. https://iclr.cc/virtual_2020/poster_S1xCPJHtDB.html ・ https://iclr.cc/virtual_2020/papers.json

[OpenReview API, 2026/09] OpenReview. フォーラム S1xCPJHtDB への API 要求の 403 応答(アクセスを試みた記録). OpenReview. https://api.openreview.net/notes?forum=S1xCPJHtDB(フォーラム https://openreview.net/forum?id=S1xCPJHtDB)

[PeerSum, 2023/10] Miao Li(University of Melbourne、学生), Eduard Hovy(University of Melbourne / CMU LTI), Jey Han Lau(University of Melbourne). "Summarizing Multiple Documents with Conversational Structure for Meta-Review Generation." Findings of EMNLP 2023(データセット PeerSum の row 2925 を参照). https://huggingface.co/datasets/oaimli/PeerSum — 被引用数 33

[MReD, 2022/03] Chenhui Shen, Liying Cheng, Ran Zhou, Lidong Bing, Yang You, Luo Si(Alibaba DAMO Academy / NUS ほか). "MReD: A Meta-Review Dataset for Structure-Controllable Text Generation." Findings of ACL 2022(ID 2020-597 を参照). https://github.com/Shen-Chenhui/MReD — 被引用数 53

[Paper Copilot paperlists, 2025/06] Jing Yang, Qiyao Wei, Jiaxin Pei(所属は確認できていない). "Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences." arXiv:2510.13201(データ iclr/iclr2020.json を参照). https://github.com/papercopilot/paperlists — 被引用数 11

[Semantic Scholar API, 2026/09] Semantic Scholar. Graph API. Semantic Scholar. https://api.semanticscholar.org/graph/v1/paper/arXiv:1903.00374

[OpenAlex API, 2026/09] OpenAlex. works API. OpenAlex. https://api.openalex.org/works/W2920362155 ・ https://api.openalex.org/works/W2994714051

[Google Scholar, 2026/09] Google Scholar. 検索結果. Google Scholar. https://scholar.google.com/scholar?q=%22Model-Based+Reinforcement+Learning+for+Atari%22+Kaiser

### 15.4 学術論文: 再評価・評価手法・サーベイ

[vanHasselt DER, 2019/06] Hado van Hasselt, Matteo Hessel, John Aslanides(DeepMind). "When to use parametric models in reinforcement learning?" NeurIPS 2019(arXiv v1 と NeurIPS 版を読んだ). https://arxiv.org/abs/1906.05243 ・ https://proceedings.neurips.cc/paper/2019/file/1b742ae215adf18b75449c6e272fd92d-Paper.pdf — 被引用数 214

[Kielak OTRainbow, 2020/03] Kacper P. Kielak(University of Birmingham, School of Computer Science). "Importance of using appropriate baselines for evaluation of data-efficiency in deep reinforcement learning for Atari"(v1 の題は "Do recent advancements in model-based deep reinforcement learning really improve data efficiency?"). arXiv:2003.10181(ICLR 2020 に投稿、採否は確認できていない). https://arxiv.org/abs/2003.10181 — 被引用数 9

[Srinivas CURL, 2020/04] Aravind Srinivas, Michael Laskin, Pieter Abbeel(UC Berkeley). "CURL: Contrastive Unsupervised Representations for Reinforcement Learning." ICML 2020. https://arxiv.org/abs/2004.04136 — 被引用数 1,342

[Kostrikov DrQ, 2020/04] Ilya Kostrikov, Denis Yarats, Rob Fergus(NYU / Facebook AI Research). "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels." ICLR 2021(arXiv v4 を読んだ). https://arxiv.org/abs/2004.13649 — 被引用数 966

[Schwarzer SPR, 2020/07] Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, Philip Bachman(Mila / Université de Montréal / Microsoft Research ほか). "Data-Efficient Reinforcement Learning with Self-Predictive Representations." ICLR 2021. https://arxiv.org/abs/2007.05929 — 被引用数 445

[Hafner DreamerV2, 2020/10] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba(Google Research / DeepMind / University of Toronto). "Mastering Atari with Discrete World Models." ICLR 2021. https://arxiv.org/abs/2010.02193 — 被引用数 1,363

[Dorner, 2021/02] Florian E. Dorner(ETH Zurich). "Measuring Progress in Deep Reinforcement Learning Sample Efficiency." arXiv:2102.04881. https://arxiv.org/abs/2102.04881 — 被引用数 14

[Plaat Survey2020, 2020/08] Aske Plaat, Walter Kosters, Mike Preuss(Leiden University). "Deep Model-Based Reinforcement Learning for High-Dimensional Problems, a Survey." arXiv:2008.05598(v2 を読んだ). https://arxiv.org/abs/2008.05598 — 被引用数 17

[Moerland Survey, 2020/06] Thomas M. Moerland, Joost Broekens, Aske Plaat, Catholijn M. Jonker(Leiden University / TU Delft). "Model-based Reinforcement Learning: A Survey." Foundations and Trends in Machine Learning(arXiv v4 を読んだ). https://arxiv.org/abs/2006.16712 — 被引用数 Semantic Scholar 66 / OpenAlex 523(FnT 版)

[Plaat Survey2021, 2021/07] Aske Plaat, Walter Kosters, Mike Preuss(Leiden University). "High-Accuracy Model-Based Reinforcement Learning, a Survey." Artificial Intelligence Review(2023。arXiv:2107.08241v1 を読んだ). https://arxiv.org/abs/2107.08241 — 被引用数 59

[Agarwal Precipice, 2021/08] Rishabh Agarwal(Google Research Brain Team / Mila), Max Schwarzer(Mila), Pablo Samuel Castro(Google Research Brain Team), Aaron Courville(Mila), Marc G. Bellemare(Google Research Brain Team). "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021(Outstanding Paper). https://arxiv.org/abs/2108.13264 ・ https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf — 被引用数 1,016

[Neves COMPER, 2021/11] Daniel Eugênio Neves ほか(PUC Minas). "Improving Experience Replay through Modeling of Similar Transitions' Sets." arXiv:2111.06907. https://arxiv.org/abs/2111.06907 — 被引用数 1

[Luo Survey, 2022/06] Fan-Ming Luo ほか(Nanjing University / Shanghai Jiao Tong University / Polixir.ai). "A Survey on Model-based Reinforcement Learning." Science China Information Sciences(arXiv v1 を読んだ). https://arxiv.org/abs/2206.09328 — 被引用数 190

[Schwarzer BBF, 2023/05] Max Schwarzer, Johan Obando-Ceron, Aaron Courville, Marc G. Bellemare, Rishabh Agarwal, Pablo Samuel Castro(Google DeepMind / Mila / Université de Montréal). "Bigger, Better, Faster: Human-level Atari with human-level efficiency." ICML 2023. https://arxiv.org/abs/2305.19452 — 被引用数 176

[Korkmaz AAAI2026, 2026/07] Ezgi Korkmaz(論文に所属の記載はない。本人のサイト https://ezgikorkmaz.github.io/ によれば UCL で PhD を取得し、最近は DeepMind に在籍). "Principled Analysis of Deep Reinforcement Learning Evaluation and Design Paradigms." AAAI 2026(arXiv:2607.07769v1). https://arxiv.org/abs/2607.07769 — 被引用数 0

### 15.5 学術論文: 後続の world model 研究

[Hafner Dreamer, 2019/12] Danijar Hafner(University of Toronto / Google Brain), Timothy Lillicrap(DeepMind), Jimmy Ba(University of Toronto), Mohammad Norouzi(Google Brain). "Dream to Control: Learning Behaviors by Latent Imagination." ICLR 2020. https://arxiv.org/abs/1912.01603 — 被引用数 2,294

[Ye EfficientZero, 2021/10] Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, Yang Gao(Tsinghua University / UC Berkeley / Shanghai Qi Zhi Institute). "Mastering Atari Games with Limited Data." NeurIPS 2021. https://arxiv.org/abs/2111.00210 — 被引用数 340

[Micheli IRIS, 2022/09] Vincent Micheli, Eloi Alonso, François Fleuret(University of Geneva). "Transformers are Sample-Efficient World Models." ICLR 2023. https://arxiv.org/abs/2209.00588 — 被引用数 373

[Hafner DreamerV3, 2023/01] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap(Google DeepMind / University of Toronto). "Mastering Diverse Domains through World Models." arXiv:2301.04104(v1 と v2 を読んだ). https://arxiv.org/abs/2301.04104 — 被引用数 1,415

[Hafner DreamerV3 Nature, 2025/04] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap(Google DeepMind / University of Toronto). "Mastering diverse control tasks through world models." Nature 640, 647–653. https://www.nature.com/articles/s41586-025-08744-2(本文は PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/ で確認)— 被引用数 Semantic Scholar 605 / OpenAlex 147

[Robine TWM, 2023/03] Jan Robine, Marc Höftmann, Tobias Uelwer, Stefan Harmeling(TU Dortmund). "Transformer-based World Models Are Happy With 100k Interactions." ICLR 2023. https://arxiv.org/abs/2303.07109 — 被引用数 167

[Ma HarmonyDream, 2023/09] H. Ma ほか(Tsinghua University / Huawei Noah's Ark Lab / Tianjin University). "HarmonyDream: Task Harmonization Inside World Models." arXiv:2310.00344(HTML 版を読んだ). https://arxiv.org/abs/2310.00344 — 被引用数 28

[Zhang STORM, 2023/10] W. Zhang, G. Wang, J. Sun, Y. Yuan, G. Huang(Beijing Institute of Technology / Tsinghua University). "STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning." NeurIPS 2023. https://arxiv.org/abs/2310.09615 — 被引用数 150

[Mattes Hieros, 2023/10] P. Mattes, R. Schlosser, R. Herbrich(Hasso Plattner Institute, University of Potsdam). "Hieros: Hierarchical Imagination on Structured State Space Sequence World Models." arXiv:2310.05167(v3 を読んだ). https://arxiv.org/abs/2310.05167 — 被引用数 10

[Cohen REM, 2024/02] L. Cohen, K. Wang, B. Kang, S. Mannor(Technion / ByteDance). "Improving Token-Based World Models with Parallel Observation Prediction." ICML 2024. https://arxiv.org/abs/2402.05643 — 被引用数 15

[Wang EZ-V2, 2024/03] S. Wang, S. Liu, W. Ye, J. You, Y. Gao(Tsinghua University IIIS / Shanghai Qi Zhi Institute / Shanghai AI Lab / Texas A&M University). "EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data." ICML 2024. https://arxiv.org/abs/2403.00564 — 被引用数 45

[Alonso DIAMOND, 2024/05] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, François Fleuret(University of Geneva / University of Edinburgh / Microsoft Research). "Diffusion for World Modeling: Visual Details Matter in Atari." NeurIPS 2024. https://arxiv.org/abs/2405.12399 — 被引用数 334

[P. Agarwal DART, 2024/06] P. Agarwal, S. Andrews, S. E. Kahou(ÉTS / Mila / Roblox / University of Calgary). "Learning to Play Atari in a World of Tokens." ICML 2024. https://arxiv.org/abs/2406.01361 — 被引用数 11

[Micheli Δ-IRIS, 2024/06] Vincent Micheli, Eloi Alonso, François Fleuret(University of Geneva). "Efficient World Models with Context-Aware Tokenization." ICML 2024. https://arxiv.org/abs/2406.19320 — 被引用数 36

[Wang Drama, 2024/10] W. Wang, I. Dusparic, Y. Shi, K. Zhang, V. Cahill(Trinity College Dublin). "Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient." ICLR 2025. https://arxiv.org/abs/2410.08893 — 被引用数 8

[Zhang OC-STORM, 2025/01] W. Zhang, A. Jelley, T. McInroe, A. Storkey, G. Wang(Beijing Institute of Technology / University of Edinburgh). "Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning." arXiv:2501.16443(HTML 版 v2 を読んだ). https://arxiv.org/abs/2501.16443 — 被引用数 8

[Lee EDELINE, 2025/02] J.-H. Lee, B.-J. Lin, W.-F. Sun, C.-Y. Lee(National Tsing Hua University / National Taiwan University / NVIDIA). "EDELINE: Enhancing Memory in Diffusion-based World Models via Linear-Time Sequence Modeling." arXiv:2502.00466(v2 を読んだ). https://arxiv.org/abs/2502.00466 — 被引用数 4

[Cohen Simulus, 2025/02] L. Cohen, K. Wang, B. Kang, U. Gadot, S. Mannor(Technion / Microsoft Research / ByteDance Seed). "Simulus: Combining Improvements in Sample-Efficient World Model Agents." arXiv:2502.11537(2026-05 の v4 を読んだ). https://arxiv.org/abs/2502.11537 — 被引用数 4

[Burchi TWISTER, 2025/03] Maxime Burchi, Radu Timofte(University of Würzburg). "Learning Transformer-based World Models with Contrastive Predictive Coding." ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf — 被引用数 30

[Burchi EMERALD, 2025/07] Maxime Burchi, Radu Timofte(University of Würzburg). "Accurate and Efficient World Modeling with Masked Latent Transformers." ICML 2025. https://arxiv.org/abs/2507.04075 — 被引用数 5

[Zhang DyMoDreamer, 2025/09] B. Zhang ほか(Beijing Institute of Technology / Tsinghua University). "DyMoDreamer: World Modeling with Dynamic Modulation." NeurIPS 2025. https://arxiv.org/abs/2509.24804 — 被引用数 2

[Peng EAWM, 2026/01] Z.-H. Peng ほか(Tsinghua University / Zhejiang University). "From Observations to Events: Event-Aware World Model for Reinforcement Learning." ICLR 2026. https://arxiv.org/abs/2601.19336 — 被引用数 1

[Mete OWM, 2026/02] A. Mete ほか(Texas A&M University). "Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning." arXiv:2602.10044. https://arxiv.org/abs/2602.10044 — 被引用数 1

### 15.6 企業の取り組み: 論文・技術報告・公式発表

[Kim GameGAN, 2020/05] Seung Wook Kim(NVIDIA / University of Toronto / Vector Institute)ほか. "Learning to Simulate Dynamic Environments with GameGAN." CVPR 2020. https://arxiv.org/abs/2005.12126 — 被引用数 132

[NVIDIA GameGAN blog, 2020/05] Nefi Alarcon(NVIDIA). "PAC-MAN Recreated with AI by NVIDIA Researchers." NVIDIA Technical Blog. https://developer.nvidia.com/blog/pac-man-recreated-with-ai-by-nvidia-researchers/

[Kim DriveGAN, 2021/04] Seung Wook Kim(NVIDIA ほか)ほか. "DriveGAN: Towards a Controllable High-Quality Neural Simulation." CVPR 2021. https://arxiv.org/abs/2104.15060 — 被引用数 160

[Koh Pathdreamer, 2021/05] Jing Yu Koh(Google Research)ほか. "Pathdreamer: A World Model for Indoor Navigation." ICCV 2021. https://arxiv.org/abs/2105.08756 — 被引用数 148

[Hu GAIA-1, 2023/09] Anthony Hu(Wayve)ほか. "GAIA-1: A Generative World Model for Autonomous Driving." arXiv:2309.17080. https://arxiv.org/abs/2309.17080 — 被引用数 689

[Wayve Scaling GAIA-1 blog, 2023/10] Wayve(署名なし). "Scaling GAIA-1." Wayve. https://wayve.ai/thinking/scaling-gaia-1/

[Yang UniSim, 2023/10] Sherry Yang(UC Berkeley / Google DeepMind)ほか. "Learning Interactive Real-World Simulators." ICLR 2024. https://arxiv.org/abs/2310.06114 — 被引用数 527

[Zhang Copilot4D, 2023/11] Lunjun Zhang(Waabi / University of Toronto)ほか. "Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion." ICLR 2024. https://arxiv.org/abs/2311.01017 — 被引用数 124

[Bruce Genie, 2024/02] Jake Bruce(Google DeepMind)ほか. "Genie: Generative Interactive Environments." ICML 2024. https://arxiv.org/abs/2402.15391 — 被引用数 823

[SIMA Team, 2024/03] SIMA Team(Google DeepMind). "Scaling Instructable Agents Across Many Simulated Worlds." arXiv:2404.10179. https://arxiv.org/abs/2404.10179 — 被引用数 83

[Valevski GameNGen, 2024/08] Dani Valevski(Google Research)ほか. "Diffusion Models Are Real-Time Game Engines." ICLR 2025. https://arxiv.org/abs/2408.14837 — 被引用数 299

[1X World Model blog, 2024/09] Jack Monas, Eric Jang(1X). "1X World Model." 1X. https://www.1x.tech/discover/1x-world-model

[Decart Oasis, 2024/10] Decart, Etched. "Oasis: A Universe in a Transformer." https://oasis-model.github.io/

[DeepMind Genie 2 blog, 2024/12] Jack Parker-Holder ほか(Google DeepMind). "Genie 2: A large-scale foundation world model." Google DeepMind. https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

[Bar NWM, 2024/12] Amir Bar(FAIR at Meta)ほか. "Navigation World Models." CVPR 2025. https://arxiv.org/abs/2412.03572 — 被引用数 315

[NVIDIA Cosmos press, 2025/01] NVIDIA Newsroom. "NVIDIA Launches Cosmos World Foundation Model Platform to Accelerate Physical AI Development." NVIDIA. https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development

[NVIDIA Cosmos, 2025/01] NVIDIA. "Cosmos World Foundation Model Platform for Physical AI." arXiv:2501.03575. https://arxiv.org/abs/2501.03575 — 被引用数 858

[Microsoft Muse blog, 2025/02] Katja Hofmann(Partner Research Manager, Microsoft Research). "Introducing Muse: Our first generative AI model designed for gameplay ideation." Microsoft Research Blog. https://www.microsoft.com/en-us/research/blog/introducing-muse-our-first-generative-ai-model-designed-for-gameplay-ideation/

[Kanervisto WHAM, 2025/02] Anssi Kanervisto(Microsoft Research)ほか. "World and Human Action Models towards gameplay ideation." Nature 638, 656–663. https://doi.org/10.1038/s41586-025-08600-3 — 被引用数 73

[Russell GAIA-2, 2025/03] Lloyd Russell(Wayve)ほか. "GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving." arXiv:2503.20523. https://arxiv.org/abs/2503.20523 — 被引用数 182

[Microsoft WHAMM, 2025/04] Tabish Rashid ほか(Microsoft Research). "WHAMM! Real-time world modelling of interactive environments." Microsoft Research. https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/

[Guo MineWorld, 2025/04] Junliang Guo(Microsoft Research)ほか. "MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft." arXiv:2504.08388. https://arxiv.org/abs/2504.08388 — 被引用数 90

[NVIDIA DreamGen, 2025/05] Joel Jang ほか(NVIDIA ほか). "DreamGen: Unlocking Generalization in Robot Learning through Video World Models." arXiv:2505.12705. https://arxiv.org/abs/2505.12705 — 被引用数 164

[1X World Model 2025, 2025/06] 1X AI Team. "1X World Model: Evaluating Bits, not Atoms"(技術報告)と関連ブログ. 1X. https://www.1x.tech/1x-world-model.pdf ・ https://www.1x.tech/discover/redwood-ai-world-model

[Meta V-JEPA 2 blog, 2025/06] Meta AI. "Introducing the V-JEPA 2 world model and new benchmarks for physical reasoning." Meta AI Blog. https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

[Meta V-JEPA 2 paper, 2025/06] Mido Assran ほか(Meta ほか。著者ごとの所属は確認できていない). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985. https://arxiv.org/abs/2506.09985 — 被引用数 727

[Tencent Hunyuan-GameCraft, 2025/06] Jiaqi Li(Tencent Hunyuan / Huazhong University of Science and Technology)ほか. "Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition." arXiv:2506.17201. https://arxiv.org/abs/2506.17201 — 被引用数 85

[DeepMind Genie 3 blog, 2025/08] Jack Parker-Holder, Shlomi Fruchter(Google DeepMind). "Genie 3: A new frontier for world models." Google DeepMind. https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/

[Skywork Matrix-Game 2.0, 2025/08] Xianglong He ほか(Skywork AI). "Matrix-game 2.0: An open-source real-time and streaming interactive world model." arXiv:2508.13009. https://arxiv.org/abs/2508.13009 — 被引用数 157

[Hafner Dreamer 4, 2025/09] Danijar Hafner, Wilson Yan, Timothy Lillicrap(Google DeepMind). "Training Agents Inside of Scalable World Models." arXiv:2509.24527. https://arxiv.org/abs/2509.24527 — 被引用数 122

[NVIDIA Cosmos-Predict2.5, 2025/10] NVIDIA. "World Simulation with Video Foundation Models for Physical AI." arXiv:2511.00062. https://arxiv.org/abs/2511.00062 — 被引用数 186

[SIMA Team SIMA 2, 2025/12] SIMA Team(Google DeepMind). "SIMA 2: A Generalist Embodied Agent for Virtual Worlds." arXiv:2512.04797. https://arxiv.org/abs/2512.04797 — 被引用数 18

[Wayve GAIA-3 blog, 2025/12] Sofía Dudas ほか(Wayve). "GAIA-3." Wayve. https://wayve.ai/thinking/gaia-3/

[Runway GWM-1, 2025/12] Runway. "Introducing GWM-1." Runway. https://runway.com/research/introducing-runway-gwm-1

[Gemini Robotics Team Veo eval, 2025/12] Gemini Robotics Team(Google DeepMind). "Evaluating Gemini Robotics Policies in a Veo World Simulator." arXiv:2512.10675. https://arxiv.org/abs/2512.10675 — 被引用数 53

[Google Project Genie blog, 2026/01] Diego Rivas(Group Product Manager, Google DeepMind), Elliott Breece(Product Manager, Google Labs), Suz Chambers(Director, Google Creative Lab). "Project Genie." Google Blog. https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/

[Waymo World Model blog, 2026/02] Chiyu Max Jiang, Xander Masotto, Bo Sun(Waymo). "The Waymo World Model: A New Frontier for Autonomous Driving Simulation." Waymo Blog. https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/

[NVIDIA Cosmos 3 press, 2026/05] NVIDIA Newsroom. "NVIDIA Launches Cosmos 3, the Open Frontier Foundation Model for Physical AI." NVIDIA. https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai

[NVIDIA Cosmos 3, 2026/06] NVIDIA. "Cosmos 3: Omnimodal World Models for Physical AI." arXiv:2606.02800. https://arxiv.org/abs/2606.02800 — 被引用数 90

### 15.7 第三者の報道

[Synced, 2019/03] Synced(組織の署名。記事中の回答者 Ahmet Salim Bilgin は FinBrain Technologies の創業者と紹介). "Google Brain SimPLe: Complete Model-Based Reinforcement Learning for Atari." SyncedReview. https://syncedreview.com/2019/03/19/google-brain-simple-complete-model-based-reinforcement-learning-for-atari/

[量子位(信息化观察网转载), 2019/03] 量子位(組織の表記). "谷歌大脑AI飞速解锁雅达利，训练不用两小时：预测能力前所未有." 信息化观察网(転載). https://www.infoobs.com/index.php/article/20190307/30760.html

[新智元(搜狐转载), 2019/03] 肖琴(新智元、編集). "谷歌提出强化学习新算法SimPLe，模拟策略学习效率提高2倍." 搜狐(転載). https://www.sohu.com/a/304019501_473283

[雷锋网 AI 科技评论, 2019/03] skura(雷锋网 AI 科技评论、役職は不明). "谷歌 AI 最新博文：视频模型中的模拟策略学习." 雷锋网. https://m.leiphone.com/category/ai/e6taTb3uRYhOXzzt.html

[腾讯云开发者社区, 2019/05] AI研习社(転載者). "动态 | 谷歌 AI 最新博文：视频模型中的模拟策略学习." 腾讯云开发者社区. https://cloud.tencent.com/developer/article/1422847

[ITwiz, 2020/02] Mikołaj Marszycki(ITwiz、役職は不明)。回答者は Błażej Osiński(deepsense.ai、Senior Data Scientist). "Reinforcement Learning jako wsparcie nauki systemów autonomicznych." ITwiz. https://itwiz.pl/reinforcement-learning-jako-wsparcie-nauki-systemow-autonomicznych/

[VentureBeat, 2020/02] Kyle Wiggers(VentureBeat). "Google Brain and DeepMind researchers attack reinforcement learning efficiency." VentureBeat. https://venturebeat.com/ai/google-brain-deepmind-reinforcement-learning-efficiency(SimPLe への言及なし)

[VentureBeat, 2021/03] Khari Johnson(VentureBeat). "How AI trained to beat Atari games could impact robotics and drug design." VentureBeat. https://venturebeat.com/ai/how-ai-trained-to-beat-atari-games-could-impact-robotics-and-drug-design/(SimPLe への言及なし)

[Import AI, 2019/03] Jack Clark. "Import AI" 136〜139 号. jack-clark.net. https://jack-clark.net/2019/03/(SimPLe への言及なし)

[InfoQ Japan, 2020/09] Anthony Alford(原著)、編集部T(翻訳). "DeepMindのAgent57はすべてのAtari 2600ゲームで人間よりも優れた結果を出している." InfoQ. https://www.infoq.com/jp/news/2020/09/deepmind-ai-atari/(SimPLe への言及なし)

[日経Robotics, 2021/09] 進藤 智則. "グーグルが開発、強化学習を1000倍高速にする技術." 日経クロステック. https://xtech.nikkei.com/atcl/nxt/mag/rob/18/012600001/00084/(公開範囲の本文に SimPLe への言及なし)

[InfoQ Japan, 2024/10] Anthony Alford(原著)、Takashi Kawase(翻訳). "Google、ゲームシミュレーションAI「GameNGen」を発表." InfoQ. https://www.infoq.com/jp/news/2024/10/google-gamengen/(SimPLe への言及なし)

[heise online, 2025/08] Daniel Herbig(heise online). "Genie 3: Google's world model builds interactive environments." heise online. https://www.heise.de/en/news/Genie-3-Google-s-world-model-builds-interactive-environments-10511028.html(SimPLe への言及なし)

[9to5Google, 2026/01] Abner Li(9to5Google). Project Genie の提供開始に関する記事(題名は未確認). 9to5Google. https://9to5google.com/2026/01/29/google-project-genie/(SimPLe への言及なし)

[THE DECODER, 2026/02] Matthias Bastian(THE DECODER). "Waymo taps Google DeepMind's Genie 3 to simulate driving scenarios its cars have never seen." THE DECODER. https://the-decoder.com/waymo-taps-google-deepminds-genie-3-to-simulate-driving-scenarios-its-cars-have-never-seen/(SimPLe への言及なし)

### 15.8 ブログ・解説

[arXivTimes, 2019/03] icoxfog417(GitHub の表示名は Takahiro Kubo。2019 年当時の所属は確認できていない). "Model-Based Reinforcement Learning for Atari #1128." GitHub(arXivTimes). https://github.com/arXivTimes/arXivTimes/issues/1128

[techsutram, 2019/03] Mandar Pise(所属は不明). "Artificial Imagination for Reinforcement Learning Now made Possible by deepsense.ai and Google Brain." techsutram. https://www.techsutram.com/2019/03/artificial-imagination-rl-by-deepsense-ai-google-brain.html

[WebBigData, 2019/03] dahara1(所属は不明). "SimPLe：ビデオモデルを用いてポリシー学習をシミュレート(1/2)(2/2)." webbigdata.jp. https://webbigdata.jp/ai/post-3138 ・ https://webbigdata.jp/post-3142/

[note, 2019/10] npaka(所属は不明。プロフィールはプログラマー・技術書の著者). "SimPLe : Atariゲームのモデルベースの強化学習." note. https://note.com/npaka/n/n2097e77e294c

[BAIR Blog, 2019/12] Michael Janner. "Model-Based Reinforcement Learning: Theory and Practice." BAIR Blog. https://bair.berkeley.edu/blog/2019/12/12/mbpo/

[Papers I Read, 2020/07] Shagun Sodhani(所属は不明). "When to use parametric models in reinforcement learning?" shagunsodhani.com. https://shagunsodhani.com/papers-I-read/When-to-use-parametric-models-in-reinforcement-learning

[博客园, 2021/01] 穷酸秀才大草包(所属は不明). "Model Based Reinforcement Learning for Atari." 博客园. https://www.cnblogs.com/lucifer1997/p/14252465.html

[どこから見てもメンダコ, 2022/02] horomary(所属は不明). "世界モデルベース強化学習①： DreamerV2の実装." はてなブログ. https://horomary.hatenablog.com/entry/2022/02/27/201339

[DEV Community, 2024/04] aimodels-fyi(Mike Young、所属は不明). "Model-Based Reinforcement Learning for Atari." DEV Community. https://dev.to/mikeyoung44/model-based-reinforcement-learning-for-atari-4g29

[psc-g.github.io, 2024/12] Pablo Samuel Castro(本人のサイトの記載では Google DeepMind(モントリオール)の senior staff research scientist). "In Defense of Atari - the ALE is not 'solved'!" psc-g.github.io. https://psc-g.github.io/posts/research/rl/atari_defense/

### 15.9 SNS・掲示板

[X, 2019/03] @hardmaru(David Ha のアカウント). 投稿 1102561871620648960・1102562698741665794. X. https://twitter.com/hardmaru/status/1102561871620648960

[X, 2020/10] @hardmaru(David Ha のアカウント). 投稿 1316251053638127616. X. https://twitter.com/hardmaru/status/1316251053638127616

[Google Scholar David Ha, 2026/09] Google Scholar. David Ha のプロフィール(現在の所属 Sakana AI). Google Scholar. https://scholar.google.com/citations?user=N7X-kbUAAAAJ

[Milken Institute, 2026/09] Milken Institute. David Ha の講演者紹介. Milken Institute. https://milkeninstitute.org/events/asia-summit-2025/speakers/david-ha

[Reddit r/MachineLearning, 2019/03] evc123(所属は不明). "[R] [1903.00374] Model-Based Reinforcement Learning for Atari: Achieving human-level performance on many Atari games after two hours of real-time play." Reddit(スレッド ax406x。api.pullpush.io のアーカイブから取得). https://www.reddit.com/r/MachineLearning/comments/ax406x/ ・ 同月の Google ブログのリンクのスレッド https://www.reddit.com/r/MachineLearning/comments/b5e7b8/

[Reddit r/reinforcementlearning, 2019/03] gwern ほか(所属は不明). "Model-Based Reinforcement Learning for Atari" の紹介(スレッド ax76wr)と Synced の記事のリンク(スレッド b2zoz5 ほか). Reddit. https://www.reddit.com/r/reinforcementlearning/comments/ax76wr/ ・ https://www.reddit.com/r/reinforcementlearning/comments/b2zoz5/

[Reddit r/reinforcementlearning, 2020/02] MasterScrat(所属は不明). "[D] Rebuttal of the SimPLe algorithm ("Model Based Reinforcement Learning for Atari")." Reddit(スレッド f37b56). https://www.reddit.com/r/reinforcementlearning/comments/f37b56/

[Reddit r/MachineLearning, 2020/05] ai_yoda(所属は不明。本文に neptune.ai のブログへのリンク). "[D] What are your favorite papers from ICLR 2020? + a list of our picks by domain." Reddit(スレッド gn6j0m). https://www.reddit.com/r/MachineLearning/comments/gn6j0m/

[Hacker News, 2019/03] lainon、headalgorithm. "Model-Based Reinforcement Learning for Atari" と "Simulated Policy Learning in Video Models". Hacker News. https://news.ycombinator.com/item?id=19310771 ・ https://news.ycombinator.com/item?id=19486005

[Hacker News, 2019/07] ArtWomb. "Model-Based Reinforcement Learning for Atari." Hacker News. https://news.ycombinator.com/item?id=20425135

[Hacker News, 2019/08] henning. "Model-Based Reinforcement Learning for Atari." Hacker News. https://news.ycombinator.com/item?id=20728585

[Hacker News, 2020/07] bitsofdl. "SimPLe: Learning to play Atari with only 2 hours of gameplay – Paper Explained." Hacker News. https://news.ycombinator.com/item?id=23895224

[Hacker News, 2022/09] IRIS の投稿(SimPLe への言及なし). Hacker News. https://news.ycombinator.com/item?id=32692404

[Hacker News, 2024/02] Genie の投稿(SimPLe への言及なし). Hacker News. https://news.ycombinator.com/item?id=39509937

[Hacker News, 2024/08] GameNGen の投稿(SimPLe への言及なし). Hacker News. https://news.ycombinator.com/item?id=41375548

[Hacker News, 2024/10] DIAMOND の投稿(SimPLe への言及なし). Hacker News. https://news.ycombinator.com/item?id=41826402 ・ https://news.ycombinator.com/item?id=40467606

[Hacker News, 2024/12] Genie 2 の投稿(SimPLe への言及なし). Hacker News. https://news.ycombinator.com/item?id=42317903

### 15.10 コード・ドキュメント・issue

[tensor2tensor rl README, 2019/03] Tensor2Tensor の開発者(ルートの README によれば Google Brain チームとユーザーコミュニティ). "Tensor2Tensor Model-Based Reinforcement Learning." GitHub. https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/rl

[tensor2tensor GitHub, 2023/07] tensorflow/tensor2tensor. リポジトリのページ(アーカイブの表示). GitHub. https://github.com/tensorflow/tensor2tensor

[GitHub API tensor2tensor, 2026/09] GitHub. REST API の応答. GitHub. https://api.github.com/repos/tensorflow/tensor2tensor

[MBRL for Atari Drive, 2019/09] 所有者の表示は確認できていない. "MBRL for Atari"(Google Drive のフォルダ。ICLR の Code リンク http://bit.ly/2wjgn1a の転送先). Google Drive. https://drive.google.com/drive/folders/1VQfRxSrLSc-x7hvXfG2ptCW7UOMtJRGF

[tensor2tensor issue #1563, 2019/04] 利用者の投稿. "Looking for parameters used to create RL ALE pretrained results." GitHub. https://github.com/tensorflow/tensor2tensor/issues/1563

[tensor2tensor issue #1579, 2019/05] 利用者の投稿(world model の学習の中断後のエラー). GitHub. https://github.com/tensorflow/tensor2tensor/issues/1579

[tensor2tensor issue #1609, 2019/06] 利用者の投稿. "Tensor2Tensor Model-Based Reinforcement Learning - Rainbow hyperparameters used." GitHub. https://github.com/tensorflow/tensor2tensor/issues/1609

[tensor2tensor issue #1830, 2020/07] 利用者の投稿(推奨バージョンでのモジュールのエラー). GitHub. https://github.com/tensorflow/tensor2tensor/issues/1830

[tensor2tensor issue #1891, 2021/06] 利用者の投稿(エージェント数 256 での AssertionError). GitHub. https://github.com/tensorflow/tensor2tensor/issues/1891

[Dopamine atari_100k, 2026/09] google/dopamine. labs/atari_100k(README と configs の DER.gin・DrQ.gin・DrQ_eps.gin・OTRainbow.gin). GitHub. https://github.com/google/dopamine/tree/master/dopamine/labs/atari_100k

[Schillaci SimPLe-PyTorch, 2020/10] Thomas Schillaci(GitHub のプロフィールでは Machine Learning Engineer、所属企業の記載なし). "SimPLe"(PyTorch による再実装、2020-10〜2022-12). GitHub. https://github.com/thomas-schillaci/SimPLe

### 15.11 特許

[Google Patents search, 2026/09] Google Patents. 発明者名での検索(xhr/query の JSON). Google Patents. https://patents.google.com/

[EPO publication server EP3402633B1, 2020/05] Google LLC(発明者 Sergey Levine, Chelsea Finn, Ian Goodfellow). EP3402633B1. European Patent Office. https://data.epo.org/publication-server/rest/v1.0/publication-dates/20200513/patents/EP3402633NWB1/document.html

[PubChem US20230239499A1, 2023/07] Google LLC(発明者 Babaeizadeh, Finn, Erhan, Saffar, Levine, Nair). US20230239499A1. PubChem. https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/patent/US-2023239499-A1/JSON

[PubChem US12154212B2, 2024/11] Waymo LLC(発明者に Dumitru Erhan を含む 8 名). US12154212B2. PubChem. https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/patent/US-12154212-B2/JSON
