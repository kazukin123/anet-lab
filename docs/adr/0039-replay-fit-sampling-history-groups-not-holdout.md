# replay 当てはまり診断は held-out 分割ではなく既存の抽選履歴フラグで群分けし、評価条件を学習条件から独立させる

Breakout の replay ratio 実験（2026-09-08）で、再利用回数の多い腕ほど学習バッチの loss / TD が小さく、成績は低かった。記録の残件は「held-out の TD を測る計器が無い」であり、既存の probe チャネル（ReplayBuffer からの一様抽出）は学習分布そのものなので対照にならない。一方、真の held-out 分割は一部の遷移を学習から除外して学習系列を変えるうえ、知りたいこと（network が抽選された遷移を、まだ抽選されていない遷移より良く当てているか）に直接は答えない。必要なのは学習に触れない観測であって、学習を変える実験設計ではない。

**未抽選群 U / 抽選済み群 S を、測定時点の sampleable range を既存の `sampled_once_` フラグで二分して定義し、両群と実 PER バッチを学習条件から独立した共通条件（FP32・eval mode・固定 midpoint 分位点・現行と同じ target 式）で評価する**ことを決定する。母数と平均年齢は snapshot の 1 回走査で求め、per-entry の抽選回数カウンタや書込時刻キャッシュは持たない。測定は optimizer 更新と target 同期の前に行い、target 組立とサンプル別誤差は learner ごとの 2 関数（NoGrad の `MakeTarget`、純粋な `ComputeElementError`）へ抽出して**学習経路も同じ関数を呼ぶ**。IQN の taus は生成関数を渡し、MakeTarget が従来の位置で呼ぶ。学習側は既存 RNG、診断側は固定 midpoint を返し、行動選択・target forward・Munchausen fresh online forward の順序を維持する。既に評価した current 出力も呼び出し側から渡す。Build は Builder パターンに使うため、ここでは MakeTarget と命名する。hard UQE × IQN の target 行動選択は、policy の `SelectAction` に診断評価指定を渡す入口で固定 midpoint 化し、スコア計算は policy 実装を使う。指定時は policy 内部も FP32 とし、補助 full query・探索 RNG・未要求の方策診断を停止する。ReplayBuffer 抽象には母数・年齢・U 抽出・S 抽出を 1 lock 区間で組み合わせて要求できる `ProbeSamplingHistory` を加え、常に両群を抽出する API にはしない。購読ヒント（ADR 0031）で購読された出力に必要な処理だけを、その出力の cadence で実行する。

## Considered Options

- **真の held-out 分割（一部遷移を学習から除外して TD を測る）**: 学習系列が変わり、OFF との等価性が定義できない。抽選履歴による当てはまりの差という問いにも直接答えない。別の実験設計として再検討する余地は残す。却下
- **per-entry の抽選回数カウンタで抽選回数別に追跡する**: Sample hot path に毎回の加算が乗り、二群の比較には「一度以上か否か」の既存フラグで足りる。却下
- **学習で計算済みの loss / TD を流用する**: train mode・BF16・random taus・IS 重みが混ざり、U 側には値が無い。学習条件と独立した診断条件で全バッチを揃える。却下
- **年齢を揃えた対照群を作る**: U は若い遷移に偏るため比較を鋭くするが、抽出設計が別物になる。群ごとの平均年齢を記録して構成差を追えるようにするに留め、対照群設計は別実験へ送る
- **診断側で forward と target 合成を独自に書き、葉の helper だけ共有する**: hot path 無変更だが Munchausen × TBO × n-step の合成が 3 learner 分重複し drift する。同 seed 等価性ゲートを受入に含めて共有関数化を選ぶ。却下
- **測定バッチを `replay_batch_size` 以下の chunk に固定分割する**: 既存 forward が config の B を前提に assert していることの影響で、共有関数を shape 非依存にすれば不要。契約は件数維持と全件平均だけにする。却下

## Consequences

- U は若い遷移に偏り、RR が低い腕ほど古い遷移も含む。`unsampled_td_ratio` は群平均の比であって記憶ギャップではなく、腕をまたぐ比較は母数・平均年齢・評価条件を併せて読む
- ReplayBuffer 抽象に `ProbeSamplingHistory` が加わり、`DefaultReplayBuffer`・`PrefetchingReplayBuffer`・現用 test double 3 つを同一変更内で更新する。`PrefetchingReplayBuffer` は `SampleUniqueUniform` と同じ FIFO settle 契約を引き継ぐ
- 学習経路の target 組立が共有関数へ書き換わるため、旧実装 / 新実装 OFF の同 seed 等価性を受入条件に含める。式の正本は 1 箇所になる
- policy の `SelectAction` に risk taus の注入口が加わる。非注入の学習側呼び出しは不変で、注入時は RNG を消費しない
- 診断の絶対値は学習ログ（train mode・BF16・random ×8）と一致しない。比較は診断内の群同士と、同じ評価条件の腕同士に限る
- 詳細契約・13 指標・購読依存の実行表・受入条件は [PRD 073](../memo/073_replay_fit_metrics_10prd.md) を正本とする。用語は `CONTEXT.md`（replay 抽選履歴群 / 未抽選群 / 抽選済み群 / 遷移の年齢 / replay 当てはまり診断 / PER 選択比）
