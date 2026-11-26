# EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る RFD

## 元の材料

[EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る RFD](../../../memo/0081_evalpanel_episode_trace_10rfd.md) blob:ed64102dd1075a4a1c031e9a15343db9b0271138

## 要点

- EvalPanel で遊ばせたゲームを後から調べられるよう、評価中の個体を trace の出力先に結び付ける。
- 従来のゲーム終了ログでは、時刻やスコアは読めても任意の scalar や方策スナップショットを追えなかった。
- 通常スケジュールで動かない設定タグでも、対応する EvalPanel のインスタンスがあれば trace をつなぐ。
- trace 行の時刻を手掛かりに、録画の場面とゲームの記録を対応づける。
- 原本の追記では、遷移ごとの step trace 出力も RFD0094 の後に仕様を取り込む対象とした。

## 内容

### 録画と評価記録

Kung-Fu Master の100M Run では、EvalPanel の録画を見ても、そのゲームが特定のスコアの境界を越えたかを対応するメトリクスから確認できなかった。パネルは専用の1 lane の評価 Runner を作るため、通常の評価スロットの記録には入らない。

### 対応づけに使う値

既存の CreateEvalRunner を維持し、設定タグに対応する鏡写しのインスタンスに trace を接続する。表示のために生成されたインスタンスがない場合は診断して読み飛ばす。録画との対応にはゲームの序数の一致を仮定せず、記録時刻を使う。

### 後続の仕様

通常の学習経験数とデモ側の trace の step を混同しない。EvalPanel の各遷移を trace チャネルへ永続出力する仕様は、RFD0094 の履歴表示に続く改訂で具体化する。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
