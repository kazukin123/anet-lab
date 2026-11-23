# Viewer の設定表は config_data.txt をフレームワークの読み込み側の規則で読む

MetricsViewer の設定表は、Run の設定を Run フォルダの `config/config_data.txt` から読み、UTF-8 として読めないバイト列と `$include` を失敗とするほかは C++ の `Properties::Load` と同じ行の規則を Java に実装して読む（PRD196）。同じ中身の `json/config_data.json` は Jackson で読めて書式にも強く、キャッシュの `json_lines` は取り込み済みだが、設定ファイルの様式は読み込み側が正本で、CLI と同じフレームワークの本体の形式を読むことを選んだ。代わりに、設定ファイルの行の規則を変えるときは、C++ と Java の両方の読み込みを同じ変更で直す。
