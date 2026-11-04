# AIハーネスの動線マップ

生成根拠: HEAD `1cbe76d65c3087f7d6b06645cf7243ddb2e2179b` と現在の作業ツリー。状態と順序はissue、PRDは指示文書である。日本語の本文を使い、固定Viewer UIとhtml langは英語にフォールバックする。

## 凡例と図の索引

ハーネスは作業を決める規約、入口は最初に読む指示、正本は意味の基準、動線は参照の連なり、shadowは別スコープの同名skill、実体コピーは同期する2つのproject版である。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
|---|---|---|---|---|
| 1 | 俯瞰 | 入口から規約・設計・検証への主経路 | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_architecture.html) | [HTML](harness_map_architecture.html) / [JSON](harness_map_architecture.archify.json) |
| 2 | 工程 | PRD起票・再グリル・実装・人間の確認 | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [HTML](harness_map_workflow.html) / [JSON](harness_map_workflow.archify.json) |

## 表1 ノード在庫

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
|---|---|---|---|---|---|
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | ビルド・配信・CIを直すとき | 人間・Claude・Codex | なし（死んだ動線） | .github/workflows/gemini_ai_review.yml（実在と参照元をrgで確認） |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | ビルド・配信・CIを直すとき | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:8 | .github/workflows/pages.yml（実在と参照元をrgで確認） |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | ビルド・配信・CIを直すとき | 人間・Claude・Codex | なし（死んだ動線） | .github/workflows/release.yml（実在と参照元をrgで確認） |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | ビルド・配信・CIを直すとき | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:45、docs/design/040_development_environment.en.md:46 | .github/workflows/windows-ci.yml（実在と参照元をrgで確認） |
| 入口 | [AGENTS.md](../../AGENTS.md) | 常時 | 人間・Claude・Codex | CLAUDE.md:4、.agents/skills/anet-audit/SKILL.md:24、.agents/skills/anet-harness-map/SKILL.md:28 | AGENTS.md（実在と参照元をrgで確認） |
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 常時 | 人間・Claude・Codex | AGENTS.md:23、.agents/skills/anet-harness-map/SKILL.md:28、.agents/skills/anet-harness-map/SKILL.md:31 | CLAUDE.md（実在と参照元をrgで確認） |
| 規約・索引 | [CONTEXT.md](../../CONTEXT.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/agents/domain.md:7、docs/agents/domain.md:8、docs/agents/domain.md:19 | CONTEXT.md（実在と参照元をrgで確認） |
| 規約・索引 | [apps/runner/tools/compress_workspace_metrics_test.py](../../apps/runner/tools/compress_workspace_metrics_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:274 | apps/runner/tools/compress_workspace_metrics_test.py（実在と参照元をrgで確認） |
| 規約・索引 | [apps/runner/tools/optuna_metrics_gzip_test.py](../../apps/runner/tools/optuna_metrics_gzip_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:280 | apps/runner/tools/optuna_metrics_gzip_test.py（実在と参照元をrgで確認） |
| 規約・索引 | [apps/runner/tools/optuna_workspace_test.py](../../apps/runner/tools/optuna_workspace_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:279 | apps/runner/tools/optuna_workspace_test.py（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd061/README.md](../../core/anet-core/testdata/prd061/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.en.md:217、docs/design/110_agents_and_learning.jp.md:220 | core/anet-core/testdata/prd061/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd072/README.md](../../core/anet-core/testdata/prd072/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.en.md:367、docs/design/100_runtime_and_configuration.jp.md:376 | core/anet-core/testdata/prd072/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd078/README.md](../../core/anet-core/testdata/prd078/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:370、docs/design/150_replay_buffer.en.md:353 | core/anet-core/testdata/prd078/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd082/README.md](../../core/anet-core/testdata/prd082/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:333、docs/design/220_atari_env.en.md:332 | core/anet-core/testdata/prd082/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd084/README.md](../../core/anet-core/testdata/prd084/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:373、docs/design/130_neural_networks.en.md:370、docs/design/200_dqn_agents.en.md:353 | core/anet-core/testdata/prd084/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd089/README.md](../../core/anet-core/testdata/prd089/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1082、docs/design/210_metrics_viewer.en.md:1082、docs/design/210_metrics_viewer.en.md:1098 | core/anet-core/testdata/prd089/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd090/README.md](../../core/anet-core/testdata/prd090/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1075、docs/design/210_metrics_viewer.en.md:1075 | core/anet-core/testdata/prd090/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd092/README.md](../../core/anet-core/testdata/prd092/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.en.md:534、docs/design/200_dqn_agents.jp.md:580 | core/anet-core/testdata/prd092/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd093/README.md](../../core/anet-core/testdata/prd093/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.en.md:512、docs/design/200_dqn_agents.en.md:536、docs/design/200_dqn_agents.jp.md:553 | core/anet-core/testdata/prd093/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd094/README.md](../../core/anet-core/testdata/prd094/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.en.md:162、docs/design/100_runtime_and_configuration.en.md:358、docs/design/100_runtime_and_configuration.jp.md:171 | core/anet-core/testdata/prd094/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd095/README.md](../../core/anet-core/testdata/prd095/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.en.md:380、docs/design/160_applications_and_tools.jp.md:423 | core/anet-core/testdata/prd095/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd097/README.md](../../core/anet-core/testdata/prd097/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:268、docs/design/120_environments.en.md:256、docs/design/220_atari_env.jp.md:343 | core/anet-core/testdata/prd097/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd098/README.md](../../core/anet-core/testdata/prd098/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:353、docs/design/140_observability.jp.md:356、docs/design/200_dqn_agents.jp.md:565 | core/anet-core/testdata/prd098/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd099/README.md](../../core/anet-core/testdata/prd099/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:331、docs/design/220_atari_env.en.md:330 | core/anet-core/testdata/prd099/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd100/README.md](../../core/anet-core/testdata/prd100/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:270、docs/design/120_environments.en.md:268 | core/anet-core/testdata/prd100/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd101/README.md](../../core/anet-core/testdata/prd101/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:351、docs/design/200_dqn_agents.jp.md:563 | core/anet-core/testdata/prd101/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd102/README.md](../../core/anet-core/testdata/prd102/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:282、docs/design/120_environments.en.md:270 | core/anet-core/testdata/prd102/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd103/README.md](../../core/anet-core/testdata/prd103/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:354、docs/design/200_dqn_agents.en.md:306、docs/design/200_dqn_agents.en.md:518 | core/anet-core/testdata/prd103/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd104/README.md](../../core/anet-core/testdata/prd104/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.en.md:157、docs/design/110_agents_and_learning.jp.md:160、docs/design/150_replay_buffer.jp.md:212 | core/anet-core/testdata/prd104/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd105/README.md](../../core/anet-core/testdata/prd105/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:135、docs/design/160_applications_and_tools.jp.md:421 | core/anet-core/testdata/prd105/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd106/README.md](../../core/anet-core/testdata/prd106/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:557 | core/anet-core/testdata/prd106/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd109/README.md](../../core/anet-core/testdata/prd109/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:419 | core/anet-core/testdata/prd109/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd110/README.md](../../core/anet-core/testdata/prd110/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.en.md:378、docs/design/160_applications_and_tools.jp.md:417 | core/anet-core/testdata/prd110/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd111/README.md](../../core/anet-core/testdata/prd111/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1055、docs/design/210_metrics_viewer.en.md:1055 | core/anet-core/testdata/prd111/README.md（実在と参照元をrgで確認） |
| 検証資材 | [core/anet-core/testdata/prd114/README.md](../../core/anet-core/testdata/prd114/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.en.md:376、docs/design/160_applications_and_tools.jp.md:415 | core/anet-core/testdata/prd114/README.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/agents/README.md](README.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:3、.agents/skills/anet-housekeeping/SKILL.md:37 | docs/agents/README.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/agents/domain.md](domain.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:664、.agents/skills/anet-harness-map/references/harness-map-contract.md:22、.agents/skills/setup-matt-pocock-skills/SKILL.md:75 | docs/agents/domain.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/agents/issue-tracker.md](issue-tracker.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:654、.agents/skills/anet-harness-map/references/harness-map-contract.md:22、.agents/skills/setup-matt-pocock-skills/SKILL.md:75 | docs/agents/issue-tracker.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/agents/triage-labels.md](triage-labels.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:659、docs/agents/issue-tracker.md:46、.agents/skills/anet-harness-map/references/harness-map-contract.md:22 | docs/agents/triage-labels.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/010_framework_overview.jp.md](../design/010_framework_overview.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:12、.agents/skills/anet-translate-docs/SKILL.md:17、.agents/skills/README.md:18 | docs/design/010_framework_overview.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/030_user_guide_analysis.jp.md:447、docs/design/030_user_guide_analysis.jp.md:521、docs/design/020_user_guide_run.en.md:1 | docs/design/020_user_guide_run.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:545、docs/design/040_development_environment.jp.md:27、docs/design/040_development_environment.jp.md:315 | docs/design/030_user_guide_analysis.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/040_development_environment.jp.md](../design/040_development_environment.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/030_user_guide_analysis.jp.md:24、docs/design/030_user_guide_analysis.jp.md:522、docs/design/040_development_environment.en.md:1 | docs/design/040_development_environment.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:287、docs/design/100_runtime_and_configuration.en.md:1、docs/design/110_agents_and_learning.jp.md:154 | docs/design/100_runtime_and_configuration.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:21、docs/design/130_neural_networks.jp.md:380、docs/design/120_environments.jp.md:288 | docs/design/110_agents_and_learning.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/120_environments.jp.md](../design/120_environments.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/120_environments.en.md:1、docs/design/100_runtime_and_configuration.jp.md:385、docs/design/110_agents_and_learning.jp.md:237 | docs/design/120_environments.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/130_neural_networks.jp.md](../design/130_neural_networks.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/130_neural_networks.en.md:1、docs/design/110_agents_and_learning.jp.md:21、docs/design/110_agents_and_learning.jp.md:238 | docs/design/130_neural_networks.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/140_observability.jp.md](../design/140_observability.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:381、docs/design/030_user_guide_analysis.jp.md:523、docs/design/100_runtime_and_configuration.jp.md:21 | docs/design/140_observability.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:65、docs/design/130_neural_networks.jp.md:382、docs/design/120_environments.jp.md:289 | docs/design/150_replay_buffer.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:290、docs/design/030_user_guide_analysis.jp.md:524、docs/design/110_agents_and_learning.jp.md:154 | docs/design/160_applications_and_tools.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:21、docs/design/130_neural_networks.jp.md:383、docs/design/030_user_guide_analysis.jp.md:236 | docs/design/200_dqn_agents.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/210_metrics_viewer.jp.md](../design/210_metrics_viewer.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20、docs/design/160_applications_and_tools.jp.md:443、docs/design/210_metrics_viewer.en.md:1 | docs/design/210_metrics_viewer.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/220_atari_env.jp.md](../design/220_atari_env.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190、docs/design/220_atari_env.en.md:1、docs/design/README.jp.md:75 | docs/design/220_atari_env.jp.md（実在と参照元をrgで確認） |
| 設計文書 | [docs/design/README.jp.md](../design/README.jp.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29 | docs/design/README.jp.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/experiments/README.md](../experiments/README.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29、.agents/skills/anet-harness-map/references/harness-map-contract.md:41、.agents/skills/anet-harness-map/references/harness-map-contract.md:77 | docs/experiments/README.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/memo/README.md](../memo/README.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:653、.agents/skills/anet-create-prd/SKILL.md:151、docs/agents/issue-tracker.md:18 | docs/memo/README.md（実在と参照元をrgで確認） |
| 規約・索引 | [docs/ownership_guideline.md](../ownership_guideline.md) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29、.agents/skills/anet-harness-map/references/harness-map-contract.md:41、docs/design/010_framework_overview.en.md:404 | docs/ownership_guideline.md（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/atari5.py](../../viewers/metrics-tools/atari5.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:570、AGENTS.md:573、AGENTS.md:574 | viewers/metrics-tools/atari5.py（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/inspect_run.py](../../viewers/metrics-tools/inspect_run.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:535、AGENTS.md:548、AGENTS.md:553 | viewers/metrics-tools/inspect_run.py（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/inspect_run_test.py](../../viewers/metrics-tools/inspect_run_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:277 | viewers/metrics-tools/inspect_run_test.py（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/metrics_source_test.py](../../viewers/metrics-tools/metrics_source_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:276 | viewers/metrics-tools/metrics_source_test.py（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/mlflow_bridge.py](../../viewers/metrics-tools/mlflow_bridge.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:484 | viewers/metrics-tools/mlflow_bridge.py（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/mlflow_bridge_test.py](../../viewers/metrics-tools/mlflow_bridge_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:275 | viewers/metrics-tools/mlflow_bridge_test.py（実在と参照元をrgで確認） |
| 規約・索引 | [viewers/metrics-tools/tb_bridge_test.py](../../viewers/metrics-tools/tb_bridge_test.py) | 対象の領域・作業に入る前。AGENTS.mdと入口skillから引く | 人間・Claude・Codex | AGENTS.md:278 | viewers/metrics-tools/tb_bridge_test.py（実在と参照元をrgで確認） |
| 設計の記録 | docs/adr（68ファイル） | 語でファイル名を絞る | 人間・Claude・Codex | AGENTS.md、docs/memo/README.md | 当該フォルダと入口規約 |
| 設計の記録 | docs/memo（318ファイル） | PRDは指示、状態と順序は索引issue。過去の保管場所も検索する | 人間・Claude・Codex | AGENTS.md、docs/memo/README.md | 当該フォルダと入口規約 |
| 課題記録 | docs/management/issues/ | 一覧・起票・判定とPRD起票・実装完了 | 人間・Claude・Codex | docs/process/registry.md、課題skill3本 | docs/management/issues/README.md |

## 表2 作業から読む文書

| 作業 | 読むもの | 読まなくてよいとき | 根拠 |
|---|---|---|---|
| PRDを起こす・改訂する | anet-create-prd、docs/quality/prd.md、wiki目録と対象の地図 | 実装だけの工程では再グリルしない | .agents/skills/anet-create-prd/SKILL.md |
| 実装を始める | PRD、保存済み2ximpl、docs/quality/impl.mdの変更種類の節 | 今回の種類に当たらない節は除く | .agents/skills/implement-prd-with-docs/PRD-IMPLEMENTATION.md |
| C++を書く | impl.mdのC++・名前・コメント・計測・GetScalar・所有権・ログ・失敗 | C++を変えないとき | AGENTS.md「作業ごとに読むもの」 |
| Python・Java・PowerShellを書く | impl.mdのコメントとTODO・ログ | 該当言語を変えないとき | AGENTS.md「作業ごとに読むもの」 |
| 設定を書く | impl.mdの設定ファイル・名前、関連設計文書 | 設定を変えないとき | AGENTS.md「設定値の扱い」 |
| 構造・契約を変える | 関連する設計文書・ADRとwikiの実装の地図 | バグ修正や既存PRDの実装で判断が既定のとき | AGENTS.md「ADRを読む条件」「wikiを引く条件」 |
| 新しい名前を付ける | CONTEXT.md、impl.mdの名前 | 新しい名前を付けないとき | AGENTS.md「用語を引く条件」 |
| 検証する | 領域の設計文書のテスト章→testdata/PRDのREADME、AGENTS.mdのMSVC手順 | 対象外の検証は理由を記録する | AGENTS.md「検証」 |
| レビューする | anet-review→レビューエージェント、protocol.md、registry.md、対象の観点 | 対象が無いときは呼ばない | .agents/skills/anet-review/SKILL.md |
| 課題を扱う | anet-create-issue・anet-list-issues・anet-update-issue、numbering.md | 途中の未決はまず作業記録へ貯める | docs/process/registry.md |
| Runを分析する | 030_user_guide_analysis.jp.md、inspect_run.py | 指標を読み終える前に結論を確定しない | AGENTS.md「Run結果分析ルール」 |
| 工程を終える | 未決と選ばれたissue、実装済みのAI判定、自己点検・レビュー記録 | コミットは人間が行う | 入口skillとdocs/process/registry.md |

## 表3 AGENTS.mdの節の配置

| 節名 | 行数 | 分類 | 移設先候補 | 根拠 |
|---|---|---|---|---|
| Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 常時必要 | 入口に保持 | AGENTS.md:3 |
| プロジェクト概要 | 16 | 常時必要 | 入口に保持 | AGENTS.md:26 |
| 基本方針 | 13 | 常時必要 | 入口に保持 | AGENTS.md:42 |
| 作業ごとに読むもの | 14 | 常時必要 | 入口に保持 | AGENTS.md:55 |
| クリーンブレーク方針 | 23 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:69 |
| 設計判断の記録(ADR)を読む条件 | 21 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:92 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | docs/wiki/README.md（読む条件は入口に残す） | AGENTS.md:113 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:120 |
| 汎用機構と利用側の責任境界 | 41 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:132 |
| Fail-Fast 原則 | 25 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:173 |
| 設定値の扱い | 13 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:198 |
| 改行と文字コード | 9 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:211 |
| ビルド | 32 | 条件付き | docs/design/040_development_environment.jp.md（入口に標準コマンドを残す） | AGENTS.md:220 |
| 検証 | 34 | 条件付き | docs/design/040_development_environment.jp.md（入口に標準コマンドを残す） | AGENTS.md:252 |
| Python 補助ツールの実行 | 22 | 条件付き | docs/design/040_development_environment.jp.md（入口に標準コマンドを残す） | AGENTS.md:286 |
| 人間へ提示するコマンドの書き方 | 30 | 条件付き | docs/quality/impl.md（入口に読む条件を残す） | AGENTS.md:308 |
| 編集しない・慎重に扱う領域 | 12 | 常時必要 | 入口に保持 | AGENTS.md:338 |
| AI エージェントの応答言語ルール | 14 | 常時必要 | 入口に保持 | AGENTS.md:350 |
| AI エージェントの概念説明ルール | 11 | 常時必要 | 入口に保持 | AGENTS.md:364 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 条件付き | .agents/skills/README.md / karpathy-guidelines/SKILL.md | AGENTS.md:375 |
| AI エージェントの作業ルール | 27 | 常時必要 | 入口に保持 | AGENTS.md:386 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | docs/design/040_development_environment.jp.md（入口に標準コマンドを残す） | AGENTS.md:413 |
| Git 操作・コミットメッセージルール | 48 | 条件付き | .agents/skills/anet-prepare-commit/SKILL.md（Gitの禁止事項は入口に残す） | AGENTS.md:425 |
| Run 命名・後片付けルール | 36 | 条件付き | docs/design/030_user_guide_analysis.jp.md / docs/experiments/README.md | AGENTS.md:473 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | docs/design/030_user_guide_analysis.jp.md / docs/experiments/README.md | AGENTS.md:509 |
| AI エージェントのRun結果分析ルール | 54 | 条件付き | docs/design/030_user_guide_analysis.jp.md / docs/experiments/README.md | AGENTS.md:541 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 50 | 条件付き | docs/design/040_development_environment.jp.md（入口に標準コマンドを残す） | AGENTS.md:595 |
| Agent skills | 20 | 条件付き | .agents/skills/README.md / karpathy-guidelines/SKILL.md | AGENTS.md:645 |
| 集計 | 532 | 条件付き20節 | 移設は本作業の対象外 | AGENTS.mdの全##見出し |

## 表4 skill在庫

| 名前 | ファイル | 所在 | コピー同期 | shadow | Codex表示名 | 根拠 |
|---|---|---|---|---|---|---|
| anet-archify-atlas | anet-archify-atlas/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-archify-atlas/agents/openai.yaml |
| anet-archify-atlas | anet-archify-atlas/references/atlas-contract.md | project | 一致 | 個人版に記録 | あり | anet-archify-atlas/references/atlas-contract.md |
| anet-archify-atlas | anet-archify-atlas/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-archify-atlas/SKILL.md |
| anet-audit | anet-audit/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-audit/agents/openai.yaml |
| anet-audit | anet-audit/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-audit/SKILL.md |
| anet-create-issue | anet-create-issue/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-create-issue/agents/openai.yaml |
| anet-create-issue | anet-create-issue/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-create-issue/SKILL.md |
| anet-create-prd | anet-create-prd/ADR-FORMAT.md | project | 一致 | 個人版に記録 | あり | anet-create-prd/ADR-FORMAT.md |
| anet-create-prd | anet-create-prd/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-create-prd/agents/openai.yaml |
| anet-create-prd | anet-create-prd/CONTEXT-FORMAT.md | project | 一致 | 個人版に記録 | あり | anet-create-prd/CONTEXT-FORMAT.md |
| anet-create-prd | anet-create-prd/OTHER-MODEL.md | project | 一致 | 個人版に記録 | あり | anet-create-prd/OTHER-MODEL.md |
| anet-create-prd | anet-create-prd/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-create-prd/SKILL.md |
| anet-harness-map | anet-harness-map/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-harness-map/agents/openai.yaml |
| anet-harness-map | anet-harness-map/references/harness-map-contract.md | project | 一致 | 個人版に記録 | あり | anet-harness-map/references/harness-map-contract.md |
| anet-harness-map | anet-harness-map/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-harness-map/SKILL.md |
| anet-housekeeping | anet-housekeeping/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-housekeeping/agents/openai.yaml |
| anet-housekeeping | anet-housekeeping/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-housekeeping/SKILL.md |
| anet-improve | anet-improve/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-improve/agents/openai.yaml |
| anet-improve | anet-improve/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-improve/SKILL.md |
| anet-ingest-wiki | anet-ingest-wiki/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-ingest-wiki/agents/openai.yaml |
| anet-ingest-wiki | anet-ingest-wiki/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-ingest-wiki/SKILL.md |
| anet-lint-wiki | anet-lint-wiki/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-lint-wiki/agents/openai.yaml |
| anet-lint-wiki | anet-lint-wiki/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-lint-wiki/SKILL.md |
| anet-list-issues | anet-list-issues/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-list-issues/agents/openai.yaml |
| anet-list-issues | anet-list-issues/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-list-issues/SKILL.md |
| anet-prepare-commit | anet-prepare-commit/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-prepare-commit/agents/openai.yaml |
| anet-prepare-commit | anet-prepare-commit/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-prepare-commit/SKILL.md |
| anet-review | anet-review/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-review/agents/openai.yaml |
| anet-review | anet-review/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-review/SKILL.md |
| anet-stats | anet-stats/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-stats/agents/openai.yaml |
| anet-stats | anet-stats/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-stats/SKILL.md |
| anet-survey-queue | anet-survey-queue/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-survey-queue/agents/openai.yaml |
| anet-survey-queue | anet-survey-queue/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-survey-queue/SKILL.md |
| anet-translate-docs | anet-translate-docs/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-translate-docs/agents/openai.yaml |
| anet-translate-docs | anet-translate-docs/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-translate-docs/SKILL.md |
| anet-update-issue | anet-update-issue/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | anet-update-issue/agents/openai.yaml |
| anet-update-issue | anet-update-issue/SKILL.md | project | 一致 | 個人版に記録 | あり | anet-update-issue/SKILL.md |
| grill-me | grill-me/SKILL.md | project | 一致 | 個人版に記録 | なし | grill-me/SKILL.md |
| handoff | handoff/SKILL.md | project | 一致 | 個人版に記録 | なし | handoff/SKILL.md |
| implement-prd-with-docs | implement-prd-with-docs/PRD-IMPLEMENTATION.md | project | 一致 | 個人版に記録 | なし | implement-prd-with-docs/PRD-IMPLEMENTATION.md |
| implement-prd-with-docs | implement-prd-with-docs/SKILL.md | project | 一致 | 個人版に記録 | なし | implement-prd-with-docs/SKILL.md |
| improve-codebase-architecture | improve-codebase-architecture/DEEPENING.md | project | 一致 | 個人版に記録 | なし | improve-codebase-architecture/DEEPENING.md |
| improve-codebase-architecture | improve-codebase-architecture/HTML-REPORT.md | project | 一致 | 個人版に記録 | なし | improve-codebase-architecture/HTML-REPORT.md |
| improve-codebase-architecture | improve-codebase-architecture/INTERFACE-DESIGN.md | project | 一致 | 個人版に記録 | なし | improve-codebase-architecture/INTERFACE-DESIGN.md |
| improve-codebase-architecture | improve-codebase-architecture/LANGUAGE.md | project | 一致 | 個人版に記録 | なし | improve-codebase-architecture/LANGUAGE.md |
| improve-codebase-architecture | improve-codebase-architecture/SKILL.md | project | 一致 | 個人版に記録 | なし | improve-codebase-architecture/SKILL.md |
| karpathy-guidelines | karpathy-guidelines/SKILL.md | project | 一致 | 個人版に記録 | なし | karpathy-guidelines/SKILL.md |
| setup-matt-pocock-skills | setup-matt-pocock-skills/domain.md | project | 一致 | 個人版に記録 | なし | setup-matt-pocock-skills/domain.md |
| setup-matt-pocock-skills | setup-matt-pocock-skills/issue-tracker-github.md | project | 一致 | 個人版に記録 | なし | setup-matt-pocock-skills/issue-tracker-github.md |
| setup-matt-pocock-skills | setup-matt-pocock-skills/issue-tracker-gitlab.md | project | 一致 | 個人版に記録 | なし | setup-matt-pocock-skills/issue-tracker-gitlab.md |
| setup-matt-pocock-skills | setup-matt-pocock-skills/issue-tracker-local.md | project | 一致 | 個人版に記録 | なし | setup-matt-pocock-skills/issue-tracker-local.md |
| setup-matt-pocock-skills | setup-matt-pocock-skills/SKILL.md | project | 一致 | 個人版に記録 | なし | setup-matt-pocock-skills/SKILL.md |
| setup-matt-pocock-skills | setup-matt-pocock-skills/triage-labels.md | project | 一致 | 個人版に記録 | なし | setup-matt-pocock-skills/triage-labels.md |
| tdd | tdd/deep-modules.md | project | 一致 | 個人版に記録 | なし | tdd/deep-modules.md |
| tdd | tdd/interface-design.md | project | 一致 | 個人版に記録 | なし | tdd/interface-design.md |
| tdd | tdd/mocking.md | project | 一致 | 個人版に記録 | なし | tdd/mocking.md |
| tdd | tdd/refactoring.md | project | 一致 | 個人版に記録 | なし | tdd/refactoring.md |
| tdd | tdd/SKILL.md | project | 一致 | 個人版に記録 | なし | tdd/SKILL.md |
| tdd | tdd/tests.md | project | 一致 | 個人版に記録 | なし | tdd/tests.md |
| to-prd | to-prd/SKILL.md | project | 一致 | 個人版に記録 | なし | to-prd/SKILL.md |

## 発見事項

- shadow: {'Claude': 6, 'Codex': 6}件。同名の実体と差分は個人版だけへ記録する。
- 実体コピーのドリフト: 0ファイル。
- 欠落: Codex表示YAMLの無いproject skillは表4に示す。汎用skillにも必須とは判断しない。
- 死んだ動線: 2ファイル（参照元をrgで測定）。.github/workflows/gemini_ai_review.yml、.github/workflows/release.yml。
- 機械化点: skill同期・文字コード・共有コアの検査はdocs/quality/impl.mdを使う。新しい仕組みは足していない。
- evidence gap: ユーザー環境の読み取り状況と詳細は個人版に記録する。
