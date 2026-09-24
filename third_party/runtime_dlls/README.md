# runtime_dlls

実行時ロードされる DLL の置き場。ここに置いた `*.dll` は、ビルド時(MSVC POST_BUILD)に
`AnetRLRunner` の出力ディレクトリ(`apps/runner/bin/<Config>/`)へ `copy_if_different` で配備される。

- 用途例: `SDL2.dll`(AtariEnv の `display_screen` / `sound`。ALE が SDL_DYNLOAD で実行時ロードする)
- **DLL を追加・削除したら CMake の再 configure が必要**(一覧が configure 時の GLOB で確定するため。libtorch DLL コピーと同じ制約)
- DLL はコミットしない(.gitignore がグローバルに `*.dll` を無視)。この README だけがリポジトリに残る
- SDL2.dll の入手先: ale-py wheel 同梱(`.venv/Lib/site-packages/ale_py/SDL2.dll`)または SDL 公式リリース。SDL2 は zlib ライセンス
- コピー先は runner のみ。テスト exe に必要な DLL が出てきたら、該当 CMakeLists に同型ブロックを追加する
