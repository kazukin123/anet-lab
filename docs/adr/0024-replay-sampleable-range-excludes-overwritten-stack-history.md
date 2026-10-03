# ReplayBufferのsample可能範囲は、ring上書きで失われたstack履歴を必要とするtransitionを除外する

`DefaultReplayBuffer`はframe stackを保存せず、sample時に開始indexから過去へ`stack_count - 1`件遡って再構成する。リング折り返し後、保持最古付近の開始indexはこの過去frameがすでに新しいデータで上書きされているにもかかわらずsample候補として公開され、extractorが上書き後の未来frameを過去frameとして読む。結果として時間逆転stackが学習minibatchへ静かに混入する。クラッシュも形状異常も起きないため、既存のどのメトリクスでも観測できない。

**範囲概念をready range（未来側確定）とsampleable range（ready + wrap後のhistory margin）へ分離し、sample候補・`Size()`・可視化はsampleable rangeだけを使う**ことを決定する。wrap後は保持最古から`stack_count - 1`件を下限側から除外する（history margin）。wrap前はmargin 0であり、起動直後・実episode境界のpadding契約は変えない。paddingが許されるのはepisode構造に由来する履歴不足だけであり、ring容量不足による履歴喪失はepisode境界ではないためpaddingしてはならない——この区別が本決定の核である。

`InitialPriorityCompleter`とeviction統計はready rangeに残す。前者は初期priority完成に未来側確定しか必要とせず、stack marginを混ぜるとcapacityとhorizonの組合せによりFIFO先頭が永久に完成しない構成が生まれる。後者は、追い出されるslotが定義上必ずhistory margin通過後に追い出されるため、sampleable基準にするとwrap後に統計が構造的に0件となりmetricが死ぬ。ready基準は「margin期間中はsample不可だった」分を過大に数える近似を含むが、margin幅（数step）はcapacity（数千step）に対して十分小さく許容する。

extractorは変更しない。安全な開始indexを渡せばring跨ぎのstack再構成は正しく動くことがテストで実証されており、修正は候補列挙側の中央不変条件に集約するのが最小である。

## Considered Options

- **extractorでpaddingを継続する（現状維持）**: 上書き由来の欠損をepisode境界と同様にpaddingすると、実在しない遷移列を正当なデータとして学習へ供給し続ける。境界判定に使うterminalフラグ自体が上書き後の新世代のもので信頼できない。却下。
- **sample結果へlogical generationを運びextractorでassertする**: 汚染を確実に検出できるが、sampler・PER・`IndexSampleResult`・replay item keyまで影響が広がる。まず候補列挙側の不変条件で解決し、防御の追加は別判断とする。今回は却下。
- **eviction統計もsampleable range基準にする**: 当初案。追い出しslotは常にmargin通過後に追い出されるためwrap後に統計が恒常0件化し、`last_evicted_never_sampled_ratio`が死ぬ。却下し、ready基準とした。
- **候補列挙側でhistory marginを適用する**: 消費者（uniform/PER/`Size()`/可視化）が同一集合を共有する既存構造をそのまま活かせ、変更が`ValidIndexManager`に閉じる。採用。

## Consequences

- wrap後の`Size()`はenv laneごとに`stack_count - 1`件だけ減る（例: 256 env × stack 4で768件、capacity 524,288に対し約0.15%）。`stack_count == 1`とwrap前は挙動不変。
- uniform・PER・可視化accessor・`DumpToLog`が返す集合と件数が常に一致する。
- 「sampleable」の語はstack込みの最終集合だけを指すよう用語を固定し、未来側確定のみの区間は「ready」と呼ぶ（`IsLogicalSampleable`→`IsLogicalReady`等のクリーンブレークrename）。
- 構築時チェックは`max(1, n_step) + 1 + (stack_count - 1)`を下限とし、sample可能transitionが構造的に0件になる構成をfail-fastで拒否する。
- 修正前後の比較Runで学習性能の有意差は出ない想定（汚染率がRun間ブレ幅に埋没する規模のため）。本決定の正当化は契約の正しさに置く。
- 詳細設計は`docs/memo/050_replay_ring_stack_margin_10prd.md`。
