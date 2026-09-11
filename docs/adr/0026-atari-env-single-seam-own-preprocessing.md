# AtariEnv は SingleDiscreteEnv として登録し、前処理は env 内自前・frame stack は既存 stucker/RB に委ねる

ALE を anet-lab へ接続する seam には、SingleDiscreteEnvFactory（1 instance = 1 `ale::ALEInterface`、core が N lane へ vectorize）と、ALE v0.11 で導入された内蔵ベクトル化環境 AtariVectorEnv の batch-native 接続（BatchEnvFactory）の二択がある。AtariVectorEnv は標準前処理（skip/max-pool/resize/stack）と autoreset を内蔵し EnvPool 系の高スループットを謳うが、フレームワーク側には既に ThreadPoolDiscreteEnv（lane 並列 + auto-reset）と stucker/ReplayBuffer の frame stack 両輪（Actor=DictFrameStacker、RB=stack_count。DropMerge 実運用実績、境界契約は PRD 050 で確定済み）が存在する。

**AtariEnv は SingleDiscreteEnvFactory で登録する。並列化・auto-reset は既存 wrapper、frame stack は既存 stucker/RB に委ね、env は「1 ALE の駆動と前処理（自前 skip ループ → grayscale 2 フレーム max-pool → area resize）で単フレーム uint8 `[1,S,S]` を出す」ことだけを責務とする**ことを決定する。sticky actions は ALE 内部（`repeat_action_probability`）に残す——自前 skip ループでも act() 単位 = エミュレータフレーム単位の判定となり、原義（Machado et al. 2018）と同一の意味論が保たれる。ALE 側の `frame_skip` は 1 固定（max-pool に中間フレームが必要）、`max_num_frames_per_episode` は 0 固定（truncation は env 自前カウントで done と厳密区別）。

決定打は ReplayBuffer のストレージ設計との整合である。RB は env の出力 dtype をそのまま保存し、stack はサンプル時に過去方向 slice で再構成する（単フレーム保存）。AtariVectorEnv の stack 済み出力 `[N,4,84,84]` をそのまま保存すると同一フレームが 4 遷移へ重複保存され、RB メモリが 4 倍になる（84×84 uint8 × 1M 遷移で 7GB → 28GB）。回避するには最新 1 フレームの切り出しが要り、それは内蔵 stack を使う意味の放棄である。

## Considered Options

- **AtariVectorEnv を batch-native 接続**: 前処理実装ゼロ・高スループットだが、①RB 4 倍メモリ（上記）②独自 autoreset（same-step/next-step）と `continue_state` 契約の突き合わせが必要で、episode 境界の意味論ズレはバグの温床 ③lane 別 seed・`train.eval.[tag].env` overlay・per-lane GetScalar を隠蔽 API の外から再構成する配線が必要 ④opencv4 依存が加わり初回ビルド・CI キャッシュが肥大 ⑤stucker/RB と stack 実装が二重化し、境界バグ類を別実装で再び踏むリスク ⑥C++ からの利用実績・ドキュメントが薄い（Python first の若い API）。却下。
- **Single 接続 + ALE 内蔵 frame_skip（max-pool なし）**: env 実装は最小になるが、スプライト点滅対策の 2 フレーム max-pool が標準前処理から欠落し、点滅依存ゲームで事例比較性を失う。却下。
- **Single 接続 + 前処理も agent/RB 側へ移設**: 前処理（resize 等）は Atari 固有の知識であり、agent 側へ出すと env 抽象が漏れる。stack だけは既に agent 側機構（stucker）が汎用部品として存在するため、stack のみ委ね前処理は env が持つ、が責務境界として一貫する。却下。
- **Single 接続 + env 内自前前処理 + stack は stucker/RB（採用）**: 実装は skip ループ・pixelwise max・area resize（`torch::nn::functional::interpolate` の既存流儀）のみで小さく、既存機構との接続コストがゼロ。性能面も、必要スループット（replay ratio 0.25 で数千 steps/s）に対し ThreadPool 並列の見積り余裕が大きく、現在の律速は GPU Learner 側にある。採用。

## Consequences

- RB には Atari 観測が単フレーム uint8 で保存され、ストレージ増なしで stack4 学習が成立する（wrap 境界は ADR 0024 の history margin が守る）。
- Atari の並列度・スレッド配置は既存の `env.worker_type` / `env.worker_threads` 設定がそのまま効く。env 側に並列実装を持たない。
- 前処理は ALE 非依存の free 関数（named namespace）に切り出し、合成入力の golden テストで数値を固定する。
- env 側が将来律速になった場合、variant registry（ADR 0009）により AtariVectorEnv の batch-native 接続を別 class_id で並存追加できる（その際は RB へ最新 1 フレーム切り出しで格納する設計が前提）。
- 詳細設計は `docs/memo/051_atari_ale_env_10prd.md`。

## Follow-up: ALE episode frame を truncation の正本にする

PRD 051 の実装契約確定時に、truncation と `episode_frames` は自前 counter ではなく `ale.getEpisodeFrameNumber()` を正本とすることを追加決定した。これにより hard Reset 中の NOOP/FIRE と life-loss 後の soft-reset NOOP も同じ ALE episode frame に含まれる。前処理を Env が所有する本 ADR の主決定は変更しない。
