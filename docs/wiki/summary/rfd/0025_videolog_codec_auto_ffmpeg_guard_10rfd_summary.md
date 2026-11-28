---
title: VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD
page_kind: サマリ
material: "[[docs/memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md|0025_videolog_codec_auto_ffmpeg_guard_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD

## 元の材料

[VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD](../../../memo/done/0025_videolog_codec_auto_ffmpeg_guard_10rfd.md) blob:7bddcbcd11eaf95cde1e37b765cbc9d5bb448856

## 要点

- 128×128 の HeatMap を NVENC に渡すと、調査環境では ffmpeg が終了し、0バイトの MKV が残った。
- VideoLogger で ffmpeg の起動後の異常を検出し、失敗したまま書き込みを続けない。
- auto は動画寸法などの条件から使用するコーデックを決める。
- 小さい動画に対する NVENC の下限は幅160・高さ64を目安に設定する。
- 明示的なコーデック指定も扱い、正常なサイズと失敗したサイズをそれぞれ検証する。

## 内容

### 要件

既定閾値 `kNvencMinWidth=160`, `kNvencMinHeight=64`（NVENC H.264 documented ~145×49 に安全マージン。実測: 128×128 失敗 / 512×512 成功）。定数化して将来のドライバ差異に対応可能に。
`requested == "h264_nvenc"` → 不適格なら `ANET_SYSTEM_ERROR`（「nvenc は WxH 非対応（最小 160×64・偶数）。`auto` か `libx264` を使用」）。適格なら nvenc。勝手にフォールバックしない。

### 設計判断

検出は生 `stream_` に依存しない: UAF の主因は「無効化された `stream_` を触ること」。よって死亡判定は `wxEVT_END_PROCESS`（プロセス由来・stream 非依存）で行い、`WriteFrame` は stream へ触れる前にフラグで門番する。`write_mutex_` を `WriteFrame`/`Close`/終了ハンドラで共有し、`stream_` の無効化を 1 つのロック下に集約。
既定を `auto` に: 何も指定しない場合に最も壊れにくい挙動（=`auto`）を既定とする。既存 config は `video_codec` を明示しているため影響を受けない（§影響・移行）。

### 影響・移行

他の video 経路（`43_agent_img`=512, `45_agent_img`=1024, `metrics.image` 各種）は全て適格サイズのため `auto` で従来通り nvenc が選ばれ、挙動不変。

### 検証方針

回帰: 512/1024 動画は `auto` で nvenc 選択・従来通り出力。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
