---
title: 観測加工の定義と強制行動の Actor 入力
page_kind: サマリ
material: "[[docs/memo/0161_observation_pipeline_10rfd.md|0161_observation_pipeline_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# 観測加工の定義と強制行動の Actor 入力

## 元の材料

[RFD0161　観測加工の定義と強制行動の Actor 入力](../../../memo/0161_observation_pipeline_10rfd.md) blob:419aeaecae765166bc245d38868765cd19aaf177

## 要点

- frame stack、正規化、観測の仕様生成が別々の経路にあり、新しい観測キーを足すと食い違いが起きやすかった。
- 行動選択、Replay 保存、Sample からの学習などで使う加工と出力仕様を同じ定義から求める。
- 強制行動を MakeAction の入力とし、Actor が返す行動を実際に環境で実行した行動にそろえる。
- lane とエピソード境界を保ち、行動に付随する Q の情報やスキップ処理も実行した行動と対応させる。
- 既存設定では観測・行動・メトリクスが変わらないことを、複数の環境と手動操作経路で検証する。

## 内容

### 問題

旧ドラフトが所見に挙げた「GPU 推論で `WithAction` の device 検査に掛かって止まる」は、RFD0192 が直した（`94182c99`）。

### ゴールと非ゴール

あわせて、題材の課題 2 件が片付く。

### 確定事項

語を 3 つに分ける。Observation は Env が Agent へ渡すもの。Frame は ReplayBuffer に保存する、観測側の 1 step 分。Perception（知覚） は、Agent が Observation と自分の持つ関連情報から作る、Network へ渡す入力で、taus を含めない。Perception は、渡す先の Network ごとにある（Q の Network、skip の補助 Network、RND の Network は、それぞれ別の入力を受ける）。本 RFD が並びで作るのは、DefaultDQN の Q の Networkへ渡す Perception で、以下、Network の名を添えない Perception はこれを指す。Frame と Perception の定義は互いを参照しない。「Frame をもとに Perception を作る」という依存は、フレームワークの前提にしない。
行動選択側と Sample 側の Perception が往復で一致するかは、Frame にキーを足す部品については RFD0142 で確かめる。本 RFD の受入には、stack と正規化だけの往復のテストを入れる。
RFD は本 RFD の 1 本のままとし、段階を A（強制行動）、C（基盤）、B（Perception の仕組み）の順に分ける。課題の結び付けと、ほかの RFD からのリンクは変えない。題とファイル名も変えない。
共通の口は順方向だけとし、逆方向（入れるとき用）の変換の口は持たない。ReplayBuffer には加工前の Frame を入れ、読む側がどれも同じ並びを順に当てる。見直しの条件は、ReplayBuffer に入れる形を加工する要求か、行動の側を並びで加工する要求が出たとき。

### 実装契約

DQN の Actor は、方策の選択、skip の適用、強制行動の反映、ヒントの計算の順に行う。
Agent の外に共通のカタログ（NN の `net.block` に当たるもの）は置かない。部品が 3 種類で、共有する定義がまだ無いためである。要るようになったら足す。
並びに `FrameStack` が書かれていなければ（並びを書かない場合を含む）、先頭に 1 枚の `FrameStack` を置いたものとして、表を当てる。行を上から順に当て、最初に当たった行で扱う。1〜9 のどれにも当たらず 10 に当たった設定は、WARN の一覧に出たうえで、11 と同じに構築へ進む。
`metrics.jsonl` の `scalar` と `trace` には何も足さないので、等価性の比べ方には影響しない。

### 受入

全環境で、変更前後の両方の起動時に `backend.$=backend.@deterministic` を指定する。Atari では前後の起動時に `E1.game=qbert` も指定し、実効設定の `AtariEnv.game` が `qbert` であることを確かめる。出荷ファイルと Run プロファイルを使い、比較のための指定は起動時に渡す。比較に使う前後両方の Run で、`backend.deterministic_algorithms` と `backend.cudnn_deterministic` の実効値がともに `true` であることを確認し、実装計画に記録する。
段階の前の Run が、比べる中身を持つ。道具に `--output <ファイル>` を付けて書き出した JSON の `old.counts`（標準出力には出ない）で、`scalar:37_agent_qtd/01_td_mean`（学習の TD 誤差）と `scalar:20_eps/10_train_reward`（学習側の episode の return）の件数が、どちらも 1 以上である。満たさなければ、Run の長さを learn の step 10,000 まで伸ばして、前後の両方を取り直す。それでも満たさなければ、受入は不成立とし、実装を止めて開発者に示す。
除くのは、実時間と資源に依る `90_perf/`・`91_resource/`・`92_resource/` で始まる tag である（RFD0097 の比べ方と同じ）。道具の `--allow-lives-values` と `--allow-migrated-edges` は付けない。checkpoint の raw checksum は使わない。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
