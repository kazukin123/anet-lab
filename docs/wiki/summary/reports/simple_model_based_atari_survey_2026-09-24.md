# Survey: SimPLe(Model-Based Reinforcement Learning for Atari)の受容・批判・後続研究

## 元の材料

[Survey: SimPLe(Model-Based Reinforcement Learning for Atari)の受容・批判・後続研究](../../../../reports/simple_model_based_atari_survey_2026-09-24.md) blob:855dd470e3bcb56ad243ab3defa4bb8160c10515

## 要点

- SimPLe は、学習した動画予測モデルを環境の代わりに使い、その中で方策を学習する model-based の深層強化学習である。
- 評価対象は 26 ゲームである。選定基準は本文と脚注で書き方が違う。
- SimPLe v4 が紹介した低データ向け model-free 手法との比較は、26ゲーム中13ゲームで model-free 側、残る13ゲームで SimPLe が上回った。比較対象の一部は、改良前の SimPLe の値だった。
- 著者は、計算コストが ALE 上で Rainbow などを直接走らせるより大きいこと、SimPLe の実時間が標準的な model-free 学習より長いことを認め、結論と付録 C に書き加えた。
- Kielak は SimPLe を当時のデータ効率の最先端の model-based 手法と位置づけたうえで、100k では明確な勝者がないと報告した。

## 内容

### 1. 論文の概要

半数を超えるゲームで、Rainbow は SimPLe の 2 倍以上のサンプルを要した。

結果は各ゲーム 5 run の平均である。 Bank Heist を除く全ゲームでランダム方策を上回った。5 run の最良値はしばしば平均よりかなり良く、最良値では 6 ゲームで人間の平均スコアを超えた [Kaiser SimPLe v4 (2020/02), Section 6.1]。

実データからのランダム始点をやめて rollout 長を 1000 にすると、Seaquest で大きく悪化した。

### 2. 書誌・版の変遷・コード

採択版の著者は 14 名で、所属は Google Brain、deepsense.ai、ポーランド科学アカデミー数学研究所、ワルシャワ大学、UIUC、スタンフォード大学である。 先頭 4 名(Kaiser、Babaeizadeh、Miłoś、Osiński)は同等の貢献で、並び順はランダムと脚注にある。連絡著者は Osiński で、作業の一部を Google Brain のインターン中に行った [Kaiser SimPLe v4 (2020/02), p.1 著者欄・脚注]。v1・v2 の著者は Afroz Mohiuddin を除く 13 名で、Mohiuddin(Google Brain)は v3 で加わった [Kaiser SimPLe v1 (2019/03), p.1] [Kaiser SimPLe v3 (2019/06), p.1]。

題名は arXiv のメタデータではハイフン付き(Model-Based)、PDF 本体と ICLR ではハイフンなし(Model Based)である。 arXiv の Comments 欄と Journal-ref 欄は空である [arXiv API 1903.00374 (2024/04)] [ICLR 2020 virtual site (2020/04), papers.json]。

会場は ICLR 2020 である。採否の区分は、OpenReview をもとにした第三者データ 3 件がそろって Spotlight としている。 ICLR 公式サイトの論文データにはセッション名("Wednesday: RL and Planning")しかなく、区分を示す項目は見つからなかった [ICLR 2020 virtual site (2020/04), poster_S1xCPJHtDB] [MReD (2022/03)] [PeerSum (2023/10)] [Paper Copilot paperlists (2025/06)]。共著機関の deepsense.ai も、2020-01 に spotlight への選出を告知している(§3.3)[deepsense.ai (2020/01)]。

### 3. 公式発信

Google Research ブログの記事 "Simulated Policy Learning in Video Models" は 2019-03-25 に公開された。著者は Łukasz Kaiser と Dumitru Erhan で、掲載時の肩書きは Google AI の Research Scientist である。 公開日は arXiv の v2(03-05)と v3(06-11)の間にあたる [Google Research Blog (2019/03), 冒頭]。

Google AI の公式 X アカウントは、ブログ公開と同じ 2019-03-25 に SimPLe を紹介した。 投稿は、SimPLe を Atari 向け強化学習のオープンソースのフレームワークとし、約 10 万回の相互作用で競争力のある結果が出ると述べ、コードの入手先を案内している。反応はいいね 357・返信 6 である(2026-09-24 取得時点)[Google AI (X) (2019/03), 投稿 1110244037305327618]。

deepsense.ai は 2020-01-17 に、論文が ICLR 2020 の spotlight に選ばれたと告知した。 告知は、spotlight に選ばれるのは投稿の 5% だけだとし、Piotr Miłoś のコメントを載せている。これは §2.1 の第三者データ(Spotlight)と一致する、共著機関側の一次情報である [deepsense.ai (2020/01), 本文]。

### 4. ICLR 2020 の査読

OpenReview 本体はボット検証のため閲覧できなかった。本節は OpenReview をもとに作られた第三者のデータセット(PeerSum、MReD、Paper Copilot)による二次情報である。 OpenReview の API は 403(Challenge verification required)を返した [OpenReview API (2026/09)]。

公式査読は 3 件で、評点は 6・8・6 である。 ほかに各査読への著者回答が 3 件、一般コメントが 1 件、それへの著者返信が 1 件ある [PeerSum (2023/10), row 2925] [Paper Copilot paperlists (2025/06)]。

評点 6 の査読(H1lwnmopFH)は、実環境の軌跡数は大きく減っても PPO の更新回数の総数はあまり減らないので、強化学習そのもののサンプル効率はあまり進んでいないと指摘した。 ALE は実行が安く並列化しやすい一方、world model には GPU が要るとして、総実時間への影響を尋ねた [PeerSum (2023/10), row 2925 review H1lwnmopFH]。

### 5. 再評価・批判・再現性

Kielak は、近年の手法の改善は新しい仕組みではなく、実データ 1 件あたりの学習更新の回数(比 r)を増やしたことによると主張した。 既定の DQN は r = 1/4 である。SimPLe は実相互作用 6.4k ごとに 800k の模擬ステップを回すので、model-free 部分が DQN なら r = 31.5 になると試算している。新手法にだけ追加の更新を許す比較は不公平だとし、更新回数を増やした Rainbow(OTRainbow)を比較相手に立てた [Kielak OTRainbow (2020/03), §4 p.3–4]。

Kielak は、SimPLe 論文の比較相手の Rainbow について、調整したとされるハイパーパラメータが開示されていないとも指摘した。 v1 の SimPLe はこの Rainbow を時間をかけて調整したと書くだけだった。v4 では調整の詳細が付録 E に加わっている(§1.3)。Kielak がどの版を参照したかは確認できていない [Kielak OTRainbow (2020/03), §4 p.4] [Kaiser SimPLe v1 (2019/03), Section 6] [Kaiser SimPLe v4 (2020/02), Appendix E]。

結論として Kielak は、少なくとも Atari では model-free が依然として最先端であり、OTRainbow のような適切なベースラインを使うべきだとした。 [Kielak OTRainbow (2020/03), §7 p.8]。論文は v2(2020-03-31)で "Importance of using appropriate baselines for evaluation of data-efficiency in deep reinforcement learning for Atari" に改題されている。ICLR 2020 に投稿したとあるが、採否は確認できていない。

### 6. Atari 100k ベンチマークの起点として

Agarwal ら: 既報は主に 3 run か 5 run で、まれに 10 run や 20 run と、run 数がそろっていない。学習中の最大評価値を報告する方式(CURL、SUNRISE)は、最終性能の報告とは比べられないとし、DER を CURL の方式で評価すると CURL の報告値を大きく超えることを示した。DrQ のコードは評価に非標準の ε を使っており、標準に直した DrQ(ε) を別に用意した [Agarwal Precipice (2021/08), §3 p.4–5・Figure 5・脚注 5]。

SPR: 1 ゲームあたり 500,000 フレームだけで評価する方式では、1 エピソードが最長 108,000 フレームあるため完了するエピソードが 4 本しか取れない場合があり、問題だとした。長さに関係なく 100 エピソードで評価することを推奨し、seed を 10 に増やした [Schwarzer SPR (2020/07), §4.1 p.8]。

BBF は、Atari 100K が ALE の 55 ゲームのうち 26 しか使わず、sticky actions もない点を挙げたうえで、含まれない 29 ゲームの方が有意に難しいと報告した。 同時に、BBF が DQN と比べて Atari 100k に過適合した証拠はないとしている [Schwarzer BBF (2023/05), §6 p.8・Figure 11]。

### 7. 後続の world model 研究

表の world model の種類と集計値は、各論文の要旨・本文・表から角度 C の担当者が転記したものである。「HTML 抽出」と書いた項目は、PDF のページではなく arXiv の HTML 版から取った。

後続論文による SimPLe の位置づけは、主に 3 通りに分けられる(本調査の整理)。ベンチマークを作った研究、world model の中で方策を学ぶ初期の例、画素空間の world model の初期の例である。

### 8. 第三者の報道

同時期の英語の報道で確認できたのは Synced(2019-03-19)の 1 本である。 記事の大半は、FinBrain Technologies 創業者と紹介された Ahmet Salim Bilgin への Q&A である。回答は SimPLe の限界として 2 点に触れている。最終スコアが最良の model-free 手法より低いことと、同じゲームでも実行ごとの差が大きいことである [Synced (2019/03), Q&A]。

ポーランドの ITwiz(2020-02-17)は、deepsense.ai の Błażej Osiński(Senior Data Scientist)へのインタビューで、この研究に触れている。 主題は Volkswagen 向けの sim-to-real である。Atari の研究については、Google Brain と米国の大学との共同研究で、最小限のやり取りで学び、システム内部に世界のシミュレータを作る手法だと説明している [ITwiz (2020/02), Atari プロジェクトに関する回答]。

### 9. 技術ブログ・解説

arXivTimes(GitHub issue #1128、2019-03-06、icoxfog417)は一言要約である。Dyna をベースに、CNN の Encoder/Decoder で環境をモデル化し、多くのゲームで PPO や Rainbow より効率が良い、とまとめている [arXivTimes (2019/03), issue 本文]。

WebBigData(2019-03-28、dahara1)と note(2019-10-23、npaka)は、Google のブログの日本語訳である。訳者のコメントは付いていない(WebBigData で確認)[WebBigData (2019/03)] [note (2019/10)]。

### 10. SNS・掲示板

本節の反応数は取得時点の値である。X と Hacker News は 2026-09-24、Reddit はアーカイブを取得した時点(ax406x は 2019-06-10)の値である。

反応が最も大きかったのは r/MachineLearning のスレッド ax406x(2019-03-04)で、score 180・35 コメントである。 投稿者は題名に、論文の題名にはない「2 時間の実時間プレイの後、多くの Atari ゲームで人間レベルの性能」という主張を付け加えていた [Reddit r/MachineLearning (2019/03), スレッド ax406x]。

Hacker News での反応は小さく、関連する投稿 5 件のうち最大でも 58 points・3 コメントである。

### 11. 産業シグナル

NVIDIA の DriveGAN(2021-04、CVPR 2021)は、行動に応じて運転映像を生成するニューラルシミュレータである。 関連研究の「データ駆動のシミュレーションと model-based RL」の段落で、エージェントが行動の計画に使うダイナミクスモデルの例として Kaiser ら(arXiv:1903.00374)を挙げている [Kim DriveGAN (2021/04), §2・参考文献 25]。

Waabi の Copilot4D(2023-11、ICLR 2024)は、点群の観測を離散拡散で予測する自動運転向けの world model である。 序論で world model と model-based RL が成果を上げた領域を挙げる箇所で、Atari の例として Kaiser ら(2019)を引いている [Zhang Copilot4D (2023/11), §1]。

OpenAI の Sora 技術報告(2024): openai.com が 403 を返した。

### 12. 批判と留保

本書で確認した SimPLe への批判は、8 つの論点に分けられる(本調査の整理)。 各論点の根拠は右端の節にある。

DER の比較は SimPLe v1 の値に対するものである。SimPLe v4 は、SD long の値なら 13 対 13 だと応じている。DER の付録の表の太字の数(Rainbow 16・SimPLe 10)と本文の「17/26」は合わない(§2.2・§5.1)。

### 13. 総合評価

本節は、§1〜§12 で確認した事実をもとにした本調査の評価である。個々の事実の根拠は各節にある。

SimPLe は、学習した画素空間の動画予測モデルの中だけで方策を学ぶ方式が、多数の Atari ゲームで低データの設定でも機能することを示した論文として受け止められた。 査読者 3 名は全員が掲載に賛成し、決定コメントは結果の重要性が新規性の弱さを上回るとした(§4)。

### 14. 調査の限界

私の運用ルール(著作物の逐語の引用は 1 回答につき 1 か所・15 語未満)に従い、本書の逐語引用は §2.2 の 1 か所だけにした。survey スキルが求める「主張ごとの逐語引用」の形式からは外れている。ほかの根拠は、場所を示したうえで要旨を書いた。

OpenReview: フォーラムと API がボット検証で開けなかった。査読の内容は、第三者のデータセット(PeerSum、MReD、Paper Copilot)による二次情報である。投稿日、決定日、議論期間中の PDF の改訂、Kielak の OpenReview ページは確認できていない。Wayback Machine のスナップショットも開けなかった。

## 主要な概念

- [Model-based Atariの評価](../../concepts/methods/model_based_evidence.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
