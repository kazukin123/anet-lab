# LLMの外部知識と探索

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

LLM・VLMによる改善は入力・課題・学習予算に依存する。Atariの条件付き改善と、ロボット向けの小型報酬モデルの成功例がある一方、成功率が上がっても対象を取り違える失敗が増える例がある。KungFuMasterのボス攻略やQbert第8面への直接の実証は再調査でも未確認。

## 分かっていること

### anet の知見

- 2026年preprintの古い目標を使う報酬整形式について、本調査では目標切り替え時に方策の順位が逆転する反例を構成し、有理数で検算した。指摘は当該v1の式に限定し、通常のポテンシャル報酬整形を否定するものではない。（出典: [保証と目標切り替え §2–3・§6](../../../../reports/llm_reward_shaping_guarantees_2026-11-07.md)）。
- potentialの差を取ったあとに非線形変換する場合、元の差の相殺は一般に保存されない。本調査は固定potential・共通終端の2step反例を構成し、tanh後の順位逆転を検算した。これは経験的な有効性の否定ではない。（出典: [保証 §7](../../../../reports/llm_reward_shaping_guarantees_2026-11-07.md#7-差を非線形変換すると相殺は残るか)）。
- 報告の数式例では、同じ入力に対する教師の one-hot が二択半々なら、最良の予測でも iLLM 型の平均報酬は 1/4 残る。教師だけが履歴を持つ場合、温度 0 でも head の入力不足によるばらつきは残り得る。これは iLLM 実験の観測値ではない（出典: [探索を補助する LLM に何を渡すか](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§6）。
- 47論文の読み順では11冊を数えるが、公開実装の再現条件と報酬整形の保証を調べた別冊はほかの冊子と対象論文が重なるため、同じ研究を二重に数えていない（出典: [LLMを活用したRL：47論文の読み方と研究上の問い](../../../../reports/llm_rl_reading_guide_2026-11-07.md)、1. 最初に読む場所）。

### 外の事例

- 調査ではAtariの条件付き改善は確認されたが、KungFuMasterのボス攻略やQbert第8面を直接改善する実証は見つからなかった。captionの正しさと目的の出来事の成功を分けて確かめる。（出典: [llm_prior_knowledge_rl_survey_2026-10-18.md](../../../../reports/llm_prior_knowledge_rl_survey_2026-10-18.md)、9. 総合評価と調査の限界）。
- 再調査でも上記2ゲームの特定の攻略を直接改善する証拠は未確認。LLM-Explorerの平均改善、Freewayの有利な条件での実証、TextAtariの言語エージェント評価を同じ効果として扱わない。（出典: [再サーベイ §5・§9](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md#9-到達点と調査の限界)）。
- Robometerは4Bモデルでシミュレーション・実機のRL改善を報告する。ただし事前学習、既存方策、場面のリセットなどの条件があり、長い課題の過去のイベントを記憶する能力には限界がある。（出典: [再サーベイ §3.5](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md#35-robometer)）。
- CriticHackの引き出し課題では、Robometerによる最適化で成功率と対象を取り違える失敗が同時に増えた。全体の成功率の改善だけでは、特定の失敗の増幅を否定できない。（出典: [再サーベイ §7.4](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md#74-critichack)）。
- 計画を文章として解析できることと、環境の完了判定に意味が通じることは別である。ローカルQwen2.5-14BのMiniGrid pilotでは、語彙不一致で目標の進行が0だった。報酬整形は計算費用のため無効で、改善の実証にはなっていない。（出典: [再サーベイ §7.3](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md#73-policy-invariant-reward-shaping-from-llm-feedback)）。
- 公開コードの既定値は論文の実験条件とは限らない。LIMENの反復数・モデル、BIRDの起動時の上書きと方法別の候補数には確認が必要で、ローカルAPIへの接続可能性だけでは小型モデルの効果を証明できない。（出典: [公開実装の再現条件 §3–4・§8](../../../../reports/llm_assisted_rl_reproducibility_2026-11-07.md)）。
- Code to Controlの公開記録では、Atariの3 runは環境seedを42に固定してLLMの生成を変えていた。生成のばらつきと環境seedへの汎化は異なる。ESBMのJAXAtariとPlayTrainのJS cloneも、ゲーム名だけでALEの評価へ合算できない。（出典: [制御と学習環境 §2–3・§6–8](../../../../reports/llm_generated_control_and_environments_2026-11-07.md)）。
- VLM-AR3LはPhi-3.5とMiniCPM-o 2.6で課題依存のRL改善を調べるが、主比較はGeminiである。約20分の1という削減は問い合わせ数で、総学習時間ではない。小型モデルの採点精度、学習後の成績、総費用を別々に確かめる。（出典: [小型モデルと費用 §2・§6–7](../../../../reports/llm_rl_small_models_and_costs_2026-11-07.md)）。
- EARSの主評価は軌跡の順位相関であり、生成報酬で学習した方策の成績は検証対象ではない。DERLの0.5Bは報酬提案器であり、内側の方策学習を繰り返す費用は残る。（出典: [小型モデルと費用 §4–5](../../../../reports/llm_rl_small_models_and_costs_2026-11-07.md)）。
- iLLMの補足資料には280M・4-bit使用とAtariのモデル別比較がある。各stepの履歴表現計算と、rollout後にまとめる行動・要約の問い合わせを区別する。補助headを残して内発報酬を外す比較にも改善があり、効果を補助報酬だけへ帰属させない。（出典: [探索に渡す情報 §2–3](../../../../reports/llm_rl_exploration_information_2026-11-07.md)）。
- IGEの公刊実験はBabyAIなどであり、元のGo-ExploreのAtari成績をIGE自身の実証にはできない。公開BabyAI実装は同じseedでの軌跡再生によって戻り、新規探索だけのstep数と全環境呼び出し数は異なる。（出典: [探索に渡す情報 §4–6](../../../../reports/llm_rl_exploration_information_2026-11-07.md)）。
- LLMが発見した更新則を固定して多数seedで学習する評価は、LLM探索そのものを独立に繰り返す評価とは異なる。How Should We Meta-LearnとMultiagent Learningは、独立探索のばらつきを十分に測った実証ではない。（出典: [更新則発見 §2–4](../../../../reports/llm_rl_algorithm_discovery_2026-11-07.md)）。
- DiscoGenでは、生成に成功した試行だけを集めても、全体では固定した既存手法を上回らなかった。個別の発見例と全体の探索能力を分け、API費用に加えて候補の学習費用を数える。（出典: [更新則発見 §5–7](../../../../reports/llm_rl_algorithm_discovery_2026-11-07.md)）。
- ONIは8Bによる非同期注釈でNetHackの学習を補助するが、LLMとRLを別GPUに置いた測定である。RLang翻訳では語彙にない概念が手順欠落につながり、DECKARDではVPT・inventory・craft操作などの前提がある。言語の品質と、環境へ作用するまでの条件を分ける。（出典: [言葉と環境 §2–6](../../../../reports/llm_rl_language_grounding_2026-11-07.md)）。
- LAGEAはQwen2.5-VL-3Bと外部RLを単一RTX 4090で使い、実時間表の9条件で平均をわずかに短縮するが、一部課題では遅くなる。目標画像・環境の成功ラベル・状態入力方策を用いる。SGRLの同step比較の改善も、PPOより高速という意味ではない。（出典: [小目標と振り返り §3・§5–6](../../../../reports/llm_rl_goals_and_reflection_2026-11-07.md)）。
- LLM Priorsは行動選択だけでなく価値更新の候補も制限する。観測ヒントは無視できる構造を持つが、実証はLlama3-70bによるBabyAIであり、報告の9倍はframe数の効率である。候補のcache、KL付き目的、参考情報としての助言を区別する。（出典: [行動助言 §2–3・§5–7](../../../../reports/llm_rl_action_advice_2026-11-07.md)）。
- LVLM2Pの公開PPO分岐は論文のsoft教師分布をそのまま使わず、学生確率にもsoftmaxを重ねている。PriorZeroの公開BabyAI設定は学習・評価で同じdata_idxのリストを使う。論文の成績、公開コードの既定値、未見配置への汎化をそれぞれ区別する。（出典: [蒸留と先読み §3–4](../../../../reports/llm_rl_distillation_and_planning_2026-11-07.md)）。
- iLLM(hop) は灰色画像と直前行動を埋め込みへ写す。行動 head の報酬は推奨行動の one-hot への二乗距離の半分であり、履歴要約も別の head が予測する。温度 0 で有限の実行可能行動の対数確率を比較し、要約の上限は 64 tokens（出典: [探索を補助する LLM に何を渡すか](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§2〜3）。
- Atari 5 ゲーム・100M training steps・10 seed で Montezuma は iLLM(hop) 2,632±277、RND 456±55。Seaquest では APT が上回った。Montezuma の GPT2 137M は 2,299±256、GPT2 870M は 2,524±312、Llama2 7B は 2,953±269、Llama3 8B は 3,220±312。内的報酬を外して補助学習を残すと 1,452±201、PPO は 11±4。原文の単位は training steps（出典: [探索を補助する LLM に何を渡すか](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§2〜3）。
- IGE は戻る状態・探索行動・保存の要否を選んで成功軌跡を探索する。BabyAI-Visual の OpenDoor 8%、PutNextTo 52% と課題差がある。BabyAI と TextWorld は各 25 seed。当時の平均 API 費用は各課題 2.01 USD と 1.28 USD、最終実験は開発費を除き 2,000 USD 未満（出典: [探索を補助する LLM に何を渡すか](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§4〜5）。
- LLM-Explorer の行動分布は現在状態を入力しない。INSIGHT は学習後の式・座標・勾配を GPT-4 に渡して説明を作る。9 Atari・10M environment steps・3 seed の Qbert の得点は 16,978.6±1,936.1、neural baseline は 17,879.2±1,857.1。生成説明を RL の更新へ戻す実験はない（出典: [探索を補助する LLM に何を渡すか](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§1・7）。
- Motif は NetHack のメッセージの進展度を LLM にオフラインで比べさせ、小型の報酬モデルに移す。報酬はエピソード内の同じメッセージの出現回数の β 乗で割る。ELLM は観測の文章化が必要で、cache なしの 5M step に数千ドルを見込んだ（出典: [内的報酬の手法の全体像](../../../../reports/intrinsic_reward_landscape_survey_2026-11-18.md)、§5.4）。
- Motif では NetHack のメッセージ、ELLM では状態・遷移の caption が必要になる。Crafter の公開コードでは行動の成否も使うため、画素だけに同じ採点器を置く条件とは異なる（出典: [Survey: LLM の事前知識で RL の探索と報酬を補助する手法](../../../../reports/llm_prior_knowledge_rl_survey_2026-10-18.md)、5. 出来事を検出する方法「5.1 物体検出と意味の接続」）。
- Tomato の1 training seed の cache 比較では、呼び出しが2,933回から434回、時間が31.1分から13.8分になった。単一条件での計測で、他の環境や GPU へその比率を移せない（出典: [LLMの行動助言をRLのどこへ接続するか](../../../../reports/llm_rl_action_advice_2026-11-07.md)、2. LLM Priors：候補の中で価値を学ぶ）。
- PriorZero は環境の実状態で木探索を始める根だけで LLM を使う。木の内部で呼び出さないことと、実行時の各環境 step で教師を必要としないことは異なる（出典: [LLMの助言を方策へ蒸留する方法と、世界モデルの先読みに使う方法](../../../../reports/llm_rl_distillation_and_planning_2026-11-07.md)、4. PriorZero：先読みの入口だけを言語で補う）。
- ONI は NetHack で注釈を非同期に処理する。別 GPU を使う条件で RL の基準約32k step/s に対し、注釈を直接参照すると約30k、分類モデルを同時学習すると約26kだった（出典: [LLMの言葉をRLの出来事と行動へ結び付ける](../../../../reports/llm_rl_language_grounding_2026-11-07.md)、2. ONI：採点を待たずに学習を進める）。
- BIRD の配布時既定は生成器と評価器の双方が mock で、tester の通過だけでは実際の LLM と RL を組み合わせた成績を検証できない。（出典: [LLMでRLを補助する公開実装の再現条件](../../../../reports/llm_assisted_rl_reproducibility_2026-11-07.md)、4. BIRD）。
- 公開された ALE Pong の制御器は学習時に +21 で止め、開始状況を変えた3回の評価で全て21対0だった。評価時に LLM を呼ばず生成済み制御器を使った事例である。（出典: [LLMが制御・世界モデル・探索・学習環境を作る研究](../../../../reports/llm_generated_control_and_environments_2026-11-07.md)、4. Memento 3：世界モデルを文章とコードで保持する）。
- MinAtar で損失だけを探索する公開設定は、探索用が Breakout と Freeway、評価用が Asterix と SpaceInvaders だった。生成した損失を ALE の画像 Atari 全体へ移した実証にはならない。（出典: [LLMでRLの更新則を発見する研究](../../../../reports/llm_rl_algorithm_discovery_2026-11-07.md)、6. 公開DiscoGenの評価境界）。
- ELLM の主条件では画像に状態説明を加えており、学習した文章化器の追加試験では人手・合成のラベルを使っても単語の重なりから誤報酬が残った。（出典: [LLMの小目標・行動の制約・失敗の振り返りをRLへ渡す](../../../../reports/llm_rl_goals_and_reflection_2026-11-07.md)、2. ELLM：有用そうな目標を探索の報酬にする）。
- DERL は報酬を提案する小型モデルと、その報酬で学習する内側の方策を持つ。最終方策の学習予算を揃えても事前の報酬探索費用は残る。（出典: [小型モデルと計算費用から見たLLMとRL](../../../../reports/llm_rl_small_models_and_costs_2026-11-07.md)、5. DERL：小さな報酬提案器も学習させる）。

## 横断的知見

- 対立点: iLLM は実行可能な行動候補の対数確率を直接比較する。LLM Priors は生成した自由文を規則で行動へ対応付ける。条件の違い: iLLM の推奨は補助 head の教師、LLM Priors の候補は行動選択と価値更新を制限する（出典: [iLLM の調査](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§2〜3、[行動助言の調査](../../../../reports/llm_rl_action_advice_2026-11-07.md)、§2）。
- 補完: LVLM2P は学習中に助言を蒸留し、学習後の教師を外せる。一方、行動助言の調査で扱った候補制限は、価値更新で比べられる行動集合も変える。両者で必要なのは助言の正しさだけでなく、いつ・どこで使うかである。条件の違い: 蒸留は学習後に教師を外せるが、候補の制限は学習時の価値更新の選択肢にも影響する（出典: [llm_rl_distillation_and_planning](../../../../reports/llm_rl_distillation_and_planning_2026-11-07.md)、1. 助言をいつ使うか。[llm_rl_action_advice](../../../../reports/llm_rl_action_advice_2026-11-07.md)、1. 同じ助言でも接続先が違う）。

## 見立て

なし

## 分かっていないこと

- 小型ローカルモデルで、生成済み制御器の再生だけでなく生成からの再発見と未知ゲームへの適応も成立するか。（出典: [制御と学習環境 §9](../../../../reports/llm_generated_control_and_environments_2026-11-07.md#9-調査の限界と残る問い)）。

- 本冊で扱うAtariの対象課題で、小型ローカルモデルによるRL改善とGPU共有時の総実時間を両立できるか。LAGEAのロボット課題の実証だけでは、この問いへ答えられない。（出典: [公開実装の再現条件 §8](../../../../reports/llm_assisted_rl_reproducibility_2026-11-07.md#8-未確認事項)、[小目標と振り返り §5–8](../../../../reports/llm_rl_goals_and_reflection_2026-11-07.md)）。
- ロボット動画で学んだ報酬モデルをAtariへ移したとき、終端・後退・長い履歴を含む採点は妥当か。（出典: [公開実装の再現条件 §8](../../../../reports/llm_assisted_rl_reproducibility_2026-11-07.md#8-未確認事項)）。
- iLLM で事前学習の意味知識・時系列処理・補助損失の寄与はどこまで分かれるか。調査はランダム重みの対照、公式実装、全 backbone の量子化条件と Atari wrapper を確認できていない（出典: [探索への入力](../../../../reports/llm_rl_exploration_information_2026-11-07.md)、§3・7）。
- 生成した観測・報酬を加えたとき、改善のうちどれだけが補助損失による表現学習で、どれだけが報酬そのものに依存するか。（出典: [LLMを活用したRL：47論文の読み方と研究上の問い](../../../../reports/llm_rl_reading_guide_2026-11-07.md)、3. 今回の証拠から分けられる研究上の問い）。

## 撤回・更新

- 2026-11-07: TextAtariの評価予算を補足。100Kという名称に対し、v2の実験手順は1,000 interaction step・5 seed。RAMラベルの文章化を使い、公開READMEのObscuringも本文の条件と異なる。長期攻略の証拠へ広げない。（出典: [再サーベイ §5.3](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md#53-textatari)）。

- 2026-11-07: GPU共有の未確認事項をAtariの対象課題へ限定。LAGEAの単一RTX 4090による報告を追加し、全方式について単一GPU実証がないという読み方を避ける。（出典: [小目標と振り返り §5–6](../../../../reports/llm_rl_goals_and_reflection_2026-11-07.md)）。

- 2026-11-07: 前回サーベイのiLLMの問い合わせ頻度を訂正。各stepの履歴表現計算は残るが、報酬用の二つの言語課題はrollout後にバッチ化する。4-bitとAtariのbackbone比較を補足資料から追加した。（出典: [探索に渡す情報 §3](../../../../reports/llm_rl_exploration_information_2026-11-07.md#3-illmの補足資料で更新された実行条件)）。
- 2026-11-07: 前回のAtari中心の未確認事項 → 21論文の再調査でも特定攻略の証拠は未確認。ロボットの成功条件、対象取り違え、ローカルモデルの統合失敗、理論の反例を追加した。前回の結果を撤回するものではない。（出典: [再サーベイ §9](../../../../reports/llm_assisted_rl_resurvey_2026-11-07.md#9-到達点と調査の限界)）。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。
- [決定的環境への復帰前提と BabyAI での実装](return_and_explore.md#横断的知見)。
