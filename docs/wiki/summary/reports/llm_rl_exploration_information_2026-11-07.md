# 探索を補助する LLM に何を渡すか

## 元の材料

[探索を補助する LLM に何を渡すか](../../../../reports/llm_rl_exploration_information_2026-11-07.md) blob:8aecdf0e10b1fd384b362c01dc18e742b9c5b408

## 要点

- 探索分布を変える方式、補助報酬・表現を学ぶ方式、制御コードを生成する方式、訪問済み状態に戻る方式で、モデルが受け取る情報も成果の単位も異なる。
- iLLM の報酬は、固定 LLM が示す行動と履歴要約を予測する head の二乗誤差である。履歴表現と補助損失も方策へ作用する。
- iLLM は各 step で履歴を表現し、報酬用の問い合わせを rollout 後にまとめる。Montezuma のモデル比較は 100M training steps・各 10 seed。手元の実時間は未測定。
- Intelligent Go-Explore は復帰先・探索行動・状態の保存を選ぶ。BabyAI 公開実装は同じ seed で環境をリセットし、保存した行動列を再生する。
- 予測誤差には入力だけでは識別できない目標のばらつきも残る。報告の数式例と、iLLM の実験結果を分けて読む。
- Qbert 第 8 面や Kung-Fu Master のボスで LLM が攻略を改善した直接の実験は見つかっていない。INSIGHT の説明生成も、学習への介入とは別の用途である。

## 内容

### 入力と介入箇所

LLM-Explorer は課題説明・間引いた行動・合計報酬を使い、現在の画面を見ない。iLLM は観測と直前行動の履歴を埋め込み、制御コードを作る方式では生成物が直接行動を選ぶ。Intelligent Go-Explore（IGE）は訪問済み状態を探索の起点として選ぶ。画像を与えた基盤モデルについても Freeway と Montezuma の結果は異なり、精密な制御までできるとは限らない。

### iLLM の予測と実行条件

固定 LLM が候補行動に付ける対数確率と、履歴要約の表現を教師にする。方策側の二つの head がこれらを予測し、残った距離から正の内的報酬を作る。推奨行動を実行したかには依存しない。画像版は灰色画像・直前行動を Hopfield 写像で token embedding に変換し、履歴表現も方策へ渡す。

補足資料には温度 0・4-bit の Transfo-XL 280M、cache とバッチ問い合わせ、要約上限 64 tokens が記されている。Atari 5 ゲーム・100M steps・10 seed の Montezuma は iLLM(hop) 2,632±277、RND 456±55。Seaquest は APT が上回った。Montezuma の backbone 比較は GPT2 137M が 2,299±256、Llama3 8B が 3,220±312。内的報酬を除いて補助 head を学習させる条件も 1,452±201 で、PPO は 11±4。事前学習した意味知識だけの寄与を測るランダム重み対照は確認できなかった。

### IGE と公開実装

IGE の評価対象は Game of 24、BabyAI、TextWorld。戻れることを前提に成功軌跡を探す。BabyAI-Visual の OpenDoor は 8%、PutNextTo は 52% だった。当時の平均 API 費用は BabyAI が 1 課題 2.01 USD、TextWorld が 1.28 USD。開発時の試行は含まない。固定コミットの BabyAI-Text は同じ seed から行動を再生して復帰し、その後で探索の step カウンタを初期化した。画像版および Atari 版の入口は確認できなかった。

### 比較条件・限界・書誌

入力、知識の由来、学習に作用する場所、復帰方法、seed の種類、環境操作の総数、最終的に評価するものを区別する。予測教師の行動が同じ入力に対して二択半々なら、この報告の例で最良予測でも iLLM 型報酬 1/4 が残る。これは測定された iLLM の失敗ではない。INSIGHT では GPT-4 が学習後の式・座標・勾配を受けて説明する。Qbert の成績はこの説明を使った再学習の成績ではない。調査は iLLM 本文・補足と IGE 公刊版・固定コードを確認したが、手元の学習や費用測定は行っていない。

## 主要な概念

- [LLM の外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [Env 共通基盤](../../concepts/framework/320_env_core.md)
