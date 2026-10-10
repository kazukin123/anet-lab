# NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ

設定 DTO のカタログは ConfigData、選択は nn_forward_key とする。Agent が選択済み搬送型を Resource として持ち、初期化時に PrepareForward を呼ぶ。具象 Module／Head は必要な設定を解釈し、私有の準備済み処理を返す。通常の部品は本体を直接呼ぶ。

## Considered Options

- 毎 forward の解釈・内容別キャッシュ: 速度クリティカルな経路に検索・比較・同期を残すため削除。
- 型別 Map・RTTI・型トークン: 具象設定管理を共通層へ移すため不採用。
- 全 Module のラッパー: 通常の層にも間接呼び出しを増やすため、本体を共通実行 interface の直接呼び先にする。
- 公開 PrepareCall: 二段階操作を増やすため不採用。別文で作る CallInfo を Forward に渡す。

## Consequences

Execution が対象 Network の寿命・準備済み処理・用途専用 State と時計種別を持つ。初期化時の BindNetwork だけが State を共有する別接続を作る。private MakeContext は呼び出し情報を組み合わせるだけで、抽選・State 更新・Tensor 計算をしない。時計を進める判断は Actor／Learner に残す。

[ADR0048](0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) の設定・State を毎回 Context で渡す判断を変更する。ε を Module buffer に置かない判断と箱×層×時計の共有規則、重み・初期化・SN・AMP・RNG・保持・保存は維持する。0048 は当時の記録として保持する。

実行中の設定切替・準備結果の重複排除は延期。State 内部の型管理、AMP 集約、episode 用語は対象外。Config 分離と実行経路変更の二段階で検証する。正本は NN 設計130と DQN 設計200。
