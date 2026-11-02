# NetworkModel は primary の Network に加えて補助 Network を持ち、補助は primary の公開出力を入力に取る。補助の target は bootstrap に使わず、target を元にする Actor の行動にだけ使う

> 2026-10-28 注記（[PRD103](../memo/103_temporl_reference_alignment_10prd.md) D2・D5）: `feature_gradient` に 3 値目 `split`（合算のまま、Adam のモーメントを loss ごとに分ける）を足した。`split` では、Consequences の「optimizer の param group は Network ごとに持つ」が「Network ごとに持ち、primary は Q 側と skip 側の 2 組のモーメントを持つ」に変わり、checkpoint の optimizer の状態に役割が 1 つ増える。global norm クリップを Network 別に取ること（`split` では loss ごとにも分ける）、補助の target を行動にだけ使うこと、checkpoint の 4 object と補助の有無の印は変わらない。

TempoRL のスキップ価値（[PRD 098_temporl_skip_policy](../memo/098_temporl_skip_policy_10prd.md)）は、primary の特徴量 z（構成によって obs の鍵、行動、taus も）から、選んだ行動を続ける長さごとの価値を出す別の出力で、終点の価値を行動価値で測るため学習の bootstrap に自分の target を要しない。**`NetworkModel` に名前付きの補助 Network（まず skip）を持たせ、補助は primary が `net.body.output` で公開した Body の出力（Network の出力にも残す）を入力に取り、primary と同じ online／target の対と同期を持つが、補助の target は Learner の bootstrap には使わず、target を元にする Actor（eval_target・EvalPanel）の行動にだけ使う**ことを決定する。特徴空間・行動価値・スキップ価値を別 Network にする 3 分割は、特徴空間と行動価値の切れ目が primary の DAG の branch で表せる一方で、checkpoint の object 数と、Network を 1 つ前提にする診断（plasticity・churn・replay fit・GNS・weight norm・attention・Noisy σ）と可視化の経路すべてに波及するため採らない。1 つの Network に skip の branch と射影を足す形は、行動を入力に取る構成で Actor の forward が全 branch で 2 回になり、勾配を止める NN ブロックが要り、`NetworkModel::Load` が module を丸ごと読むため無効の checkpoint から分岐できないので採らない。補助の target を持つのは、target 側の Actor が online の skip を遅れた z に当てる歪みを避け、DropMerge の eval_target と EvalPanel でスキップをそのまま測るためである。MuZero 試作が 3 つの Network を束ねるのは dynamics を再帰的に当てるためで、1 回だけ当てる skip には当たらない。

## Considered Options

- **3 分割（特徴空間 F / 行動価値 Q / スキップ価値 S を別 Network、target は F と Q）**: MuZero 試作の束ね方に沿うが、skip を無効にしても F|Q に分かれたままで既存 checkpoint の継続 Run が読めなくなり、Network を 1 つ前提にする経路が「どの Network か」を持つ。F を単体で差し替える要件が pin されたら、primary を Head 無しの F にし Q を補助にする形で同じ器に足せる。却下。
- **1 つの Network（DAG の branch と合成 Head）**: 新しい器は無いが、包含型で Actor の forward が 2 倍、Detach ブロックと部分 forward の経路が要り、無効の checkpoint から分岐できない。却下。
- **2 つ（primary と補助。採用）**: primary を読む既存の経路はそのまま。包含型の 2 回目の forward は補助だけ。遮断は seam の切り離し 1 つ。
- **補助に target を持たない**: target 側の Actor は online の skip を遅れた z に当てるか、スキップしないかになり、DropMerge の評価と EvalPanel でスキップを測れない。却下。
- **z の取り出し**: Head の pass-through、`net.output` の新設、補助側の名指しからの導出は、記法か経路を増やすため却下し、`net.body.output` の鍵を Network の出力にも残す形を採った。

## Consequences

- skip が有効な Run の checkpoint は primary と補助のそれぞれ online・target の 4 object と、補助の有無の印を持つ。無効なら今と同一。optimizer の param group は Network ごとに持ち、読み込みは group の役割で対応付ける。
- target の同期（hard／soft）と Actor の複製・定期同期は primary と補助に同じ時点で掛ける。
- global norm クリップは Network 別に取る（PRD の D6）。
- 補助 Network の入力 spec は、その bind が参照する鍵だけから導く（未使用入力 WARN を避ける）。
- 詳細は [PRD 098_temporl_skip_policy](../memo/098_temporl_skip_policy_10prd.md) §5.3。
