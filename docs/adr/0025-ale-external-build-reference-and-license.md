# ALE は外部ビルド木を ALE_ROOT で参照し、Atari env はオプショナルビルドにする（GPL 隔離）

Atari 環境の基盤である ALE（Arcade Learning Environment v0.12.0）は GPL-2.0 であり、Apache-2.0 の本体リポジトリと同梱・無条件リンクすると、バイナリ配布時に結合著作物全体へ GPL-2.0 が及ぶ（Apache-2.0 は GPLv2 と非互換のため「全体を GPLv2 で出す」以外の選択肢がなくなる）。GPL の義務は配布時にのみ発火し、手元での使用・学習・スコア公表には及ばない。また ALE は vcpkg/Conan レジストリに存在せず、upstream の install ルールは `if(UNIX ...)` ガードで Windows 未対応のため、Windows C++ 利用の実質標準はソースビルド木の直接参照である。

**ALE をリポジトリに同梱せず、環境変数 `ALE_ROOT` で外部ビルド木（tag v0.12.0 固定、Debug/Release 両建て）を直接参照する。Atari env モジュールは三値 cache 変数 `ANET_ENABLE_ATARI`（AUTO/ON/OFF、既定 AUTO = ALE_ROOT の存在・実在で自動判定）によるオプショナルビルドとし、リリースパッケージ（PRD 043 系）には含めない（リリースビルドは OFF で行う）。`core/envs/atari1/NOTICE.md` に「ALE とリンクした成果物の配布は GPL-2.0 に従う」旨を明記する**ことを決定する。libtorch の `Torch_DIR` と同型の「環境変数解決＋外部配置」であり、`third_party/` は同梱物専用という既存の区分を保つ。

ROM も同様に非同梱とし、環境変数 `ATARI_ROM_DIR`（config `AtariEnv.rom_dir` が非空なら優先）で自己調達ディレクトリを指す。

## Considered Options

- **`third_party/` へソース vendoring + add_subdirectory（box2d 方式）**: ALE の CMake は vcpkg toolchain を前提に依存解決するため、本体ビルド全体に vcpkg toolchain が混入する。GPL コードのリポジトリ同梱はライセンス境界も曖昧にする。VS ソリューションも重くなる。却下。
- **install prefix + find_package（libtorch 方式そのまま）**: upstream の install ルールが `if(UNIX AND BUILD_CPP_LIB)` ガードで Windows では空振りし、ローカルパッチが必要になる。CI で ALE ビルド木ごとキャッシュする方針なら install 層に意味がない。却下（upstream へのガード除去 PR は別途の貢献候補として残す）。
- **DLL 化による GPL 回避**: GPL-2.0 は静的/動的リンクを区別しない（リンク方式で回避できるのは LGPL の話）。却下。
- **プロセス分離（GPL の env サーバ + IPC）**: GPL 境界として有効であり、Atari 入りバイナリを配布する必要が生じた場合の唯一の道。ただし step 毎の IPC コストと複雑さを常時払うことになる。現時点では不要のため採用せず、将来の選択肢として記録する。
- **外部ビルド木参照 + オプショナルビルド + 配布除外（方針A）**: 配布物に GPL コードが含まれず本体は Apache-2.0 のまま。ALE 無し環境は従来通りビルド可能。採用。

## Consequences

- Atari を使う開発者は ALE を一度自前ビルドし（手順は PRD 051 §3.1 に確定構成を記載）、`ALE_ROOT` と `ATARI_ROM_DIR` を設定する。設定が無い環境では Atari env が自動的に外れ、既存ビルドに影響しない。
- `ANET_ENABLE_ATARI=ON` なのに ALE_ROOT が不備の場合は configure が fail-fast する（CI での意図明示用）。
- リリースパッケージに Atari は入らない。将来 Atari 入り配布が必要になったらプロセス分離を再検討する。
- ALE は tag 固定（v0.12.0）で、vcpkg manifest の builtin-baseline により依存（zlib/SDL2）も固定される。CI キャッシュキーは tag + triplet + configure オプション。
- 詳細設計は `docs/memo/051_atari_ale_env_10prd.md`。
