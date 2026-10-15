# checkpoint 引き継ぎでは optimizer の options を現在の設定で置き換える

LibTorch の optimizer 読み込みは moment と step に加えて param group の options（lr、weight_decay、eps、betas）も checkpoint の値で復元するため、`auto_load_file` で分岐した Run で設定ファイルの学習率変更が黙って無効になっていた（DropMerge の長期 Run、2026-10。[PRD 091](../memo/091_checkpoint_load_optimizer_options_10prd.md)）。`dqn::Learner::Load` は読み込み直後に全 options を現在の設定から作り直して各 group へ代入し、checkpoint と違った項目は WARN で示す。checkpoint の値を残す `continue` や不一致で止める `require_match` は、Run 全体の Save / Load（PRD 930）が policy として扱うまで作らない。
