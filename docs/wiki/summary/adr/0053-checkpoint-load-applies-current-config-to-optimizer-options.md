# checkpoint 引き継ぎでは optimizer の options を現在の設定で置き換える

## 元の材料

[ADR 0053](../../../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) blob:9e141d47e2c680a802bd28e917f51601b8ae06d5

## 要点

- checkpoint の読み込みは optimizer の moment と step に加えて options も復元する。
- 分岐 Run では新しい設定の学習率が保存値で上書きされる問題があった。
- 読み込み直後に全 param group の options を現在の設定で作り直す。
- moment と step は checkpoint から引き継ぐ。
- 保存値と現設定の差の通知は後に INFO へ変更され、options の扱いは維持された。
- 保存した options を採用する継続モードや、不一致だけで停止するモードはこの決定では設けない。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
