# ALE は外部ビルド木を ALE_ROOT で参照し、Atari env はオプショナルビルドにする（GPL 隔離）

## 元の材料

[ADR 0025](../../../adr/0025-ale-external-build-reference-and-license.md) blob:0703a4c883f64d0f542eb85441081ccf76f4d1c4

## 要点

- Atari が使う ALE は本体に同梱せず、外部でビルドした v0.12.0 の木を ALE_ROOT で参照する。
- 通常は外部ビルドの有無で AtariEnv の構成を決める。
- 明示して有効にしたのに必要な構成がなければ configure で失敗する。
- リリースビルドでは Atari を無効にし、本体と ALE を結合したバイナリを配布しない。
- Atari を含む配布が必要になった場合は、環境を別プロセスに置く構成を再検討する。

## 内容

### 参照と配布

対象版は Windows で upstream の install 手順をそのまま使えないため、Debug・Release を含む外部ビルド木を参照する。依存ライブラリを本体の third_party に取り込む構成は選ばなかった。

### 構成の判定

AUTO は外部参照の実在で構成を決め、ON は開発者の意図を明示した指定として不備を報告する。Atari を含む配布用の別方式は、実際に配布が必要になったときの課題とした。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
