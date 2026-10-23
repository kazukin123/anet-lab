# Qbert

## 今の結論

低εラダーの64 env・容量1Mでは45–50Mの第8面突破率が3 seedで45.6・32.6・0.7%だった。容量だけ・env数だけを変えた腕は1.0%・0.1%に留まり、初突破が早い腕でも突破が定着しない例があった。

## 分かっていること

- RR4でも第8面の壁が残り、低εのlaneほど深い面へ届いた。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ・探索ブロック02。Run: run_20260926-022413_a5_qbert_btr50m）。
- BTR型のεスケジュールだけでは差が埋まらず、初突破の時期はseedで大きくぶれ、早く出ても定着しない例があった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ。Run: run_20260927-173004_a5_qbert_btrsched_s1 / run_20260927-035604_a5_qbert_btr50m_eps005_end0）。

- εラダー1e-4〜0.05、64 env・replay容量1M（lane窓16,384）の素の3 seedでは、45–50Mの第8面突破率が45.6 / 32.6 / 0.7%だった。seed1の45.7M・ε=0評価は平均32,948で102本すべて突破し、BTR公開CSVの45M・33,105と同水準だった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ4・5、探索ブロック03。Run: `run_20260928-065041_a5_qbert_btr50m_eps005_env64`, `run_20260929-025249_a5_qbert_btr50m_eps005_env64_seed2`, `run_20261001-015829_a5_qbert_btr50m_eps005_env64_seed3`）。
- 容量だけを下げた128 env・1Mは45–50Mの突破率1.0%、env数だけを下げた64 env・2Mは0.1%だった。初突破は全腕で18.5〜45.9Mに散らばり、早い初突破だけでは終盤の突破定着に結び付かなかった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ6・7、探索ブロック03。Run: `run_20260928-202307_a5_qbert_btr50m_eps005_cap1m`, `run_20260929-162737_a5_qbert_btr50m_eps005_env64_cap2m`, `run_20260928-134117_a5_qbert_btr50m_eps005_sync25`）。
- 64 env・1MにUQEまたはViTを加えた45–50Mの突破率は45.8% / 27.4%で、素のenv64のseed幅内だった。ViTの実時間は7.56h、素のenv64は6.25〜6.84hだった。（出典: [2026-09-27_btr-gap-qbert-bz.md](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ10、探索ブロック04。Run: `run_20260930-044852_a5_qbert_btr50m_env64_uqe`, `run_20260930-110739_a5_qbert_btr50m_eps005_env64_vit`, `run_20260928-065041_a5_qbert_btr50m_eps005_env64`, `run_20260929-025249_a5_qbert_btr50m_eps005_env64_seed2`, `run_20261001-015829_a5_qbert_btr50m_eps005_env64_seed3`）。
- RAMのround_clear=8には第8面へ入ってすぐ終わるゲームも含まれ、当該比較の突破判定はround_clear=8かつ29,000点以上だった。（出典: [2026-09-25_sample-efficiency-profile-atari5.md](../../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md)、現時点のまとめ13、探索ブロック03。Run: `run_20260926-022413_a5_qbert_btr50m`, `run_20260927-035604_a5_qbert_btr50m_eps005_end0`）。

## 分かっていないこと

- 64 env・容量1Mで突破が増えた効き方の仕組みは何か（出典: [2026-09-27のcampaign](../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、現時点のまとめ6・探索ブロック03「効き方の仕組みは測れていない」）？

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
