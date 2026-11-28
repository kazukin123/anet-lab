---
title: Atari の RAM 指標で値の種類数とビット遷移を数える
page_kind: サマリ
material: "[[docs/memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md|0126_atari_ram_metric_distinct_and_bit_reducers_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# Atari の RAM 指標で値の種類数とビット遷移を数える

## 元の材料

[RFD0126　Atari の RAM 指標で値の種類数とビット遷移を数える](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md) blob:a1f1cb8846bf5478f2311f0eaf057e8170248450

## 要点

- 1ゲーム内に観測した RAM 値の種類を数え、訪問した部屋数などを示せるようにする。
- 指定マスクの各ビットについて0から1、1から0への変化を数える。
- 同じフレームで複数の対象ビットが立った場合は、変化したビットの数だけ加える。
- ゲームごとの指標と Run 全体の和集合を区別し、後者の集約は対象から外す。
- RAM 定義と畳み方の解釈を整理し、既存の大小・増減を測る方式も確かめる。

## 内容

### 目的と問題

RAM メトリクスの畳み方は `max_seen`・`min_seen`・`inc_count`・`dec_count`・`reach_count:N` の 5 語だけで、どれも 1 バイトの値の大小か、値の遷移の回数を数える。1 ゲームで観測した値の種類と、ビットごとの変化を数える語が無い。そのため Montezuma's Revenge で、訪れた部屋の数（部屋番号 0x83 の種類数）と、持ち物（0xC1 のビット）を種類ごとに取った回数・使った回数を記録できない。
あわせて、RAM 定義を読むコードが `AtariEnvConfig` のコンストラクタに並び、畳み方の名前と種別が 5 か所に散らばり、env の設定クラスが Run 側の設定の階層（`run.eval.[..].env.`）を名指しで知っている。縮小方式（`resize_mode`）は文字列のまま 3 つの層で検証している。畳み方を足すたびに、この散らばった箇所を直すことになる。

### 起草時点の現状

ADR 0046 は 5 語を閉じた列挙とし、「語を足すのは既存契約を壊さない追加である」と書く。毎フレームの比較は 1 バイトの参照で、コストは無視できるとした。
`room_up` は部屋番号が増えた回数で、行き来でも増える。`mz_nm_rnd_b0p1_s1` で、得点が 400 のまま 1 ゲームで 41 回になったゲームがあった（課題 949e8a、事象）。`room_max` は最大の番号だけで、開始の部屋の左の部屋 0 へ行っても増えない。
今の 5 語はバイト全体を比べるので、`sword_only` は鍵を持ったまま剣を拾った分を数えない。剣で敵を倒した回数は定義を置かず、得点で読んでいる。
畳み方の名前と種別は、種別の列挙・if の連鎖（306 行）・`Begin` の条件（246 行）・`Observe` の switch（253 行）・ERROR の文言に現れる。

### 確定事項

畳み方 `distinct_seen` を足す。1 ゲームで観測した値（`reset_game()` 直後の値を含む）の種類数を返す。値は 1〜256 で、開始の値から動かなかったゲームは 1。
新しい 3 語の観測の時機、集計の単位、NaN の規則、値の出口は今の 5 語と同じにする。ビットの 2 語は 0 から始め、ゲームの開始の時点ですでに 1 のビットは数えない。
Run 全体で見つけた値の和集合は作らない。
課題 9ff3b7 を全部（RAM 定義の読み取りと縮小方式の両方）範囲に入れる。

### 仕様

前のフレームの値を p、今のフレームの値を c、マスクを m、ゲームの開始の値を v0 とする（どれも 0〜255 の 1 バイト）。

### 実装契約

設定キーは今と同じ `AtariEnv.ram_metric.[<game>].[<label>]` で、値の 2 つ目（畳み方）に 3 語が加わる。判定は上から順に当て、最初に当たった行で扱う。
`Atari RAM metric override is unsupported` の ERROR は無くなる。
`GrayscaleResizer` が自分の寸法の前提（正の寸法、`pixel_area` の縮小先が元の短辺以下）を検査し続けるか。縮小方式の文字列の検査は持たない。
互換層は持たない（新しい語の追加と、内側の整理だけで、旧い契約のための別名・分岐は要らない）。過去の Run の記録と ADR は書き換えない。

### 成功の測り方と受入

テストは既定のスイートで確かめる。ROM が要るテスト（`[rom]` のタグ）は、ROM のある環境で通す。
`distinct_seen`: 開始の値 1 で始め、0・1・2・6・2 を順に観測したゲームで 4。開始の値 1 で始めて何も観測しないゲームで 1。開始の値 0 で始め、1〜255 を順に観測したゲームで 256。
3 語の出口（`[atari][ram_metric][config]`、ROM なし）: 3 語を使う定義を番号付けしたゲームで、番号が既知の番号に入り、実効設定に定義の行がそのまま記録される（今の `reports accepted RAM declarations` と同じ確かめ方）。
Run 側の階層の名指しの不在: `git grep -n -F -e 'run.eval' -e 'run\.eval' -- core/envs/atari1/src/AtariEnv.cpp core/envs/atari1/src/AtariEnv.hpp` が 0 件（固定文字列の 2 つで、地の文と正規表現の中の書き方の両方を引く。コメントも数える。テストのファイルは範囲に入れない）。

### Montezuma で使うときの材料

訪れた部屋の数だけでは左右の扉を見分けられない: 部屋 1 で鍵を取り左の扉から部屋 0 へ入ったゲームも、右の扉で部屋 2 へ入ったゲームも 2 になる。実験記録は左右の扉を `room_max` が 1 か 2 以上かで見分けている（Taiga ら寄せの記録、29 行）。
5 番以降の tag の枠が無い: scalar の tag は、群の中の 50〜59 番が番号 1〜4 で埋まっている（Atari.txt:1494-1508）。5 番以降を番号付けすると、trace の列は足せるが、scalar の mean・max は RFD0108 の枠が入るまで出せない。
番号の結び直しと過去の Run: Montezuma の番号 1〜3 は、すでに全ゲーム共通の意味（面クリア・ボス）から外れている。RFD0082 の「既存の番号の意味は変えない」は共通の意味についての決まりである。結び直すと、同じ列の意味が Run の前後で変わるので、過去の Run とはラベル（Run の `config/config_data.txt`）で読み分ける。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
