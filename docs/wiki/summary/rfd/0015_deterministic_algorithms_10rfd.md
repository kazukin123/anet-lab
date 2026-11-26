# `backend.deterministic_algorithms` 導入（学習の同 seed 再現性確保）仕様書

## 元の材料

[`backend.deterministic_algorithms` 導入（学習の同 seed 再現性確保）仕様書](../../../memo/done/0015_deterministic_algorithms_10rfd.md) blob:ec4a49268f1ccb193eadd798fc2372448a7ef166

## 要点

- 同じ seed の学習が再現しなくなった原因を、CUDA の SDPA backward の atomic 加算に絞った。
- 決定的なアルゴリズムを要求すると、利用中の実装では勾配計算が決定的な経路に切り替わった。
- backend.deterministic_algorithms でこの設定を制御する。
- 既定を有効にし、決定的な実装がない演算ではエラーを表に出す。
- cuDNN 用の既存フラグと分け、同じ seed で実際の Run の出力を照合する。

## 内容

### Context（背景・目的）

学習 Run の 同 seed → 同結果（再現性）が失われていた。原因は SDPA（`at::scaled_dot_product_attention`）が CUDA で Memory-Efficient backend（cutlass fmha）にディスパッチされ、その backward が gradient を atomic 加算するため非決定だったこと。forward は決定的なので「eval は再現するが train（loss/weight）だけ run ごとに割れる」という症状になる。

### 設計方針

既定 true（`cudnn_deterministic=true` と同じ「決定論的を既定」方針に揃え、SDPA 非決定の穴を塞ぐ）。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
