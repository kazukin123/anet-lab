# LLMが制御・世界モデル・探索・学習環境を作る研究

## 元の材料

[LLMが制御・世界モデル・探索・学習環境を作る研究](../../../../reports/llm_generated_control_and_environments_2026-11-07.md) blob:64a5175c5c8cd24ff0fd8a8212119303df77834e

## 要点

- Code to Control の Atari では OCAtari の物体表現を入力し、各 Run の最良エピソードを選んだ中央値を比較した。6ゲーム中、20M step の PPO が上回るのは Pong・Breakout・Freeway の3本だった。
- ESBM の JAXAtari 実験は3ゲーム・5 seeds・20,000 raw frames。Kangaroo と Kingkong は最初の面に限定し、同じ wrapper で比べたのは PPO・EvoSymbol・ESBM だった。
- Memento 3 の ALE Pong では frame skip 4 で、+21に達した時点で学習を終了した。得られた制御器は評価時に LLM を呼ばず、異なる開始状況3回とも21対0だった。
- PDE の実機試験で報告した約1時間は、128 rollouts 相当のロボットデータ収集時間である。論文 v4 の本文は3課題の平均を、調査時の著者ページは4課題を記載している。
- Code to Control の約11µsは、あらかじめ収集した物体状態を渡して `policy(state)` を呼び出す時間である。300状態に各5回の測定で、観測抽出、環境 step、LLM による生成の時間は含まない。

## 内容

### 4. Memento 3：世界モデルを文章とコードで保持する

これは世界モデルと制御器をコードとして獲得した事例であり、DQNに補助報酬を追加した実験ではない。この分類は本調査による整理である。

### 7. 公開コードで確かめた再現の境界

以下は静的な確認である。公開資料が「再現できた」と記す箇所を、今回こちらで再現したとは扱わない。

### 8. 比べる前に揃えるもの

根拠はCode to Controlの生成run、ESBMのwrapper区分、PlayTrainの再現資料を参照。

### 9. 調査の限界と残る問い

5論文の実験を再実行していない。効果量は著者報告で、今回の独立検証結果ではない。

本冊で使った追加の一次資料は各節へ直接リンクした。報酬整形の数式点検と報酬生成系の公開実装の確認は別冊を参照。

## 主要な概念

- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [Model-based Atariの評価](../../concepts/methods/model_based_evidence.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [実ロボット課題の選定](../../concepts/subjects/robot_tasks.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
