# ImageCls

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [題材](../index.md#題材)

## 今の結論

Food-101のスクラッチ学習の到達点はResNet18ish_hr＋cosine＋Mixup/CutMixの86.0〜86.4%（seed 1の3本）で、同じ学習条件のConvNeXt Atto・Femto（stem stride 2）は83.1・83.7%だった。evalはFlatten → GAP2D、最終解像度14×14、DropHeadとdroppath、cosineで上がり、weight decay 0.1と、ConvNeXtのBodyの拡大では上がらなかった。

## 分かっていること

- ImageCls（Food-101、101 クラス、224×224）は train が高く eval が低い過学習だった。当時の FoodResNet13 は空間 28×28 のまま Flatten して 100,352 次元にし、最終段の LinearOut（in 100,352 × out 101）に約 1014 万 parameter が 1 枚で集中していた（出典: [023_imagecls_gap_head_10prd.md](../../memo/done/023_imagecls_gap_head_10prd.md)、Context（背景・目的））。
- Flatten → GAP2D だけを変えると、eval は 8.71M で 44.5 → 61.1% に上がり、parameter は 10.91M → 0.79M になった。同じ step の train の正解率も 70.3 → 73.9% と下がらなかった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ2・探索ブロック01。Run: run_20260625-231716_imgcls_022aug / run_20260626-012847_imgcls_023gap）。
- 定数 LR・10M の Body の比較では、ResNet18ish_hr（最終解像度 14×14）が peak 76.5%・final 71.9% で最も高かった。標準の ResNet18（7×7）は 66.5%、ConvNeXt Nano は 57.3%、Hybrid は 56.2〜60.3%、ViT は 36.5〜42.5%（final）だった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ3・探索ブロック02・03。Run: run_20260630-004827_imgcls_ResNet18ish_hr / run_20260629-093157_imgcls_Res18 / run_20260703-015300_imgcls_cn-nano）。
- ResNet18ish に DropHead 0.2 と ResBlock の droppath を入れると、5.07M で 68.7 → 76.1% に上がった。weight decay 0.05 → 0.1 は 7.57M で 69.3 → 64.6% に下げ、標準の ResNet18 に DropHead だけを入れても 7.32M で 66.8 → 66.2% と上がらなかった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、探索ブロック02。Run: run_20260626-025438_imgcls_Res18 / run_20260627-234723_imgcls_Res18_do1 / run_20260627-053928_imgcls_Res18-3）。
- 定数 LR の ResNet18ish_hr は 4.08M で peak 76.5% に達し、10M までに 71.9% に下がった。DropHead と droppath を入れた ResNet 系の 4 本は、どれも 4.1〜4.6M で peak を付けて下がった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ4・探索ブロック02。Run: run_20260630-004827_imgcls_ResNet18ish_hr）。
- ConvNeXt Atto の stem の stride を 4 → 2（最終 7×7 → 14×14）にすると、10M の eval は 67.1 → 72.8% に上がり、10M までの実時間は 2.33h → 5.97h になった。batch 256 → 128 は 10M で 65.7 → 67.1%。同じ条件の Atto（3.41M）と Femto（4.87M）は 10M で 65.7% と 65.3% だった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ5・探索ブロック03。Run: run_20260704-113836_imgcls_cn-atto-b128 / run_20260704-014020_imgcls_cn-atto-b128_s2 / run_20260703-224506_imgcls_cn-atto）。
- LR の cosine（warmup 0.5M のあと 1e-3 → 1e-5）だけを変えた Atto（stem 2）は、15M で 73.6 → 79.2% になった。ResNet18ish_hr は定数 LR の 71.9% から 83.7%（15M）になったが、batch と RRC の下限も同時に変えた（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ6・探索ブロック04。Run: run_20260705-132946_imgcls_CnAttoHR / run_20260704-214337_imgcls_ResNet18ish_hr）。
- Mixup/CutMix を入れて学習予算を 15M → 20M に延ばすと、ResNet18ish_hr は 83.7 → 86.0%、Atto は 79.2 → 83.1% になった。30M の ResNet18ish_hr は 86.4% で、20M との差は同じ設定の 2 本のずれ（0.3〜1.0pt）の中だった。学習側の `target_prob_mix_norm` は 0.80 → 0.84、train の正解率は 56.1 → 61.4% と上がった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ7・8・探索ブロック05。Run: run_20260706-124351_imgcls_ResNet18HRmix / run_20260707-084617_imgcls_ResNet18HR32_30 / run_20260708-090643_imgcls_CnAttoHR）。
- BF16（learner だけ）の ResNet18ish_hr＋Mixup の 20M は 86.4% で、FP32 の 20M（86.0%）より低くならず、実時間は 9.24h → 6.20h になった。この 2 本は deterministic の設定と fused optimizer の設定も違う。backend と fused が同じ FP32 の 30M は 645 exp step/s、BF16 は 896 exp step/s だった（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ9・探索ブロック05。Run: run_20260706-124351_imgcls_ResNet18HRmix / run_20260708-020741_imgcls_ResNet18HR）。
- Mixup を入れた ConvNeXt は、Body だけが違う Atto（3.41M）と Femto（4.87M）で 83.1% と 83.7% だった。ResNet18ish_hr（14.15M）は 86.4%（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、現時点のまとめ10・探索ブロック05。Run: run_20260708-090643_imgcls_CnAttoHR / run_20260708-210559_imgcls_CnFemtoHR）。
- Mixup/CutMix の Run の train の正解率は混ぜる前のラベル 1 つで採点するので 55〜61% に留まり、eval（83〜86%）より低い。収束の判定には使わない（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、この文書の読み方・探索ブロック05）。

## 分かっていないこと

- ResNet18ish の DropHead＋droppath による上昇（5.07M で 7.4pt）は、DropHead と droppath のどちらによるか（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、探索ブロック02 考察）。
- ResNet18ish_hr の 71.9 → 83.7% のうち、cosine の分はいくらか（batch と RRC の下限も同時に変えた）（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、探索ブロック04 考察）。
- Mixup/CutMix だけの効果はいくらか（学習予算も 15M → 20M に延ばした）（出典: [2026-06-25_food101-scratch.md](../../experiments/image-cls/imagecls/2026-06-25_food101-scratch.md)、探索ブロック05 考察）。

## 撤回・更新

なし。

## 関連

- [環境](../../design/120_environments.jp.md)、§2.3（ImageCls は native batch Env）。
- [Agentと学習](../../design/110_agents_and_learning.jp.md)、§2.4（登録済み Agent）。
- [ImageCls 過学習対策①: train 専用データ拡張の PRD](../../memo/done/022_imagecls_augmentation_10prd.md)。
- [ImageCls 過学習対策②: 最終段 Flatten → GAP2D の PRD](../../memo/done/023_imagecls_gap_head_10prd.md)。
- [Mixup / CutMix 導入の PRD](../../memo/done/030_imagecls_mixup_cutmix_10prd.md)。
- [ProfiledValue による learning_rate 制御の PRD](../../memo/done/031_imagecls_cosine_lr_10prd.md)。
- [BF16 と Head 化の PRD](../../memo/done/033_imagecls_bf16_head_10prd.md)。
- [batch 入力の PRD](../../memo/done/034_imagecls_batch_input_10prd.md)。
- [評価と比較の条件](../mechanisms/evaluation.md)。
- [322 Env個別実装の実装の地図](../categories/322_env_individual.md)。
