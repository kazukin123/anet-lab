# V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書

## 元の材料

[V2: 可視化層（metrics_image / HeatMap / 状態スイープ）の TensorDict 復旧 実装指示書](../../../memo/done/0010_image_metrics_v2_10rfd.md) blob:cf474586830e898bc781ea6fba0e92d180b9a65f

## 要点

- 観測を TensorDict に移した後、単一 Tensor を前提にしていた画像メトリクスと状態スイープの可視化を復旧する。
- 経験、ReplayBuffer、ネットワークの各入力で、読む観測キーを識別できるようにする。
- ネットワークを評価する関数は TensorDict の入出力に合わせる。
- スイープは入力を変えるキーと描画する出力キーを分け、追跡対象の設定にも明記する。
- 経験の可視化、ReplayBuffer、ネットワークのスイープの順に確認する。

## 内容

### 観測の参照

環境の観測には vector、grid、action_mask などが同時に含まれる。経験を表示するときは対象キーを選ぶ。キーの指定がない従来の設定も、既存の probe 文法で扱う。

### ネットワークのスイープ

入力キーごとに状態を差し替え、戻ってきた TensorDict から表示したい出力を取り出す。スイープ用の関数と通常のネットワークの取得方法をそろえ、呼ぶ側に古い単一 Tensor の前提を残さない。

### 確認の順序

経験を読む画像メトリクス、ReplayBuffer の表示、ネットワークを評価する状態スイープの順に確認する。現用設定には対象の入力・出力キーを明記する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
