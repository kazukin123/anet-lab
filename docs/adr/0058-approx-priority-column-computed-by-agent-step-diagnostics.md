# step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない

Evaluation Trace の近似優先度の列は、Agent が作る step 部品が、Actor と共有するヘルパーで aux の Q から Actor Q ヒントを作り直し（`actor_approx` の構成で既にヒントがあっても使わない）、replay と同じ n step の組み立てと `DqnInitialPriorityEstimator` で計算する（PER の設定によらず計算し、DQN 系以外の Agent は部品を出さない）。Actor の設定に「ヒントを作るか」のキーを足して eval_panel の Actor にだけ作らせる案は、設定キーが増え、`auto` が自動選択ではなく「今の条件」という意味になり、n step の窓の置き場が Agent の外（GUI 側）に出るので採らない。ADR 0010 の平均 Q 近似と追加 forward を行わない方針（step 部品は eval_panel の Actor のネットワークの Q をそのまま使う）、ADR 0012 の推定器の責務は変えない。
