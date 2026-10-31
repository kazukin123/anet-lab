"""PRD099のOpenCV 4.8.1 INTER_AREA正解画像を生成する。"""
import json
import hashlib
import os
import platform
import sys

import cv2
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

GAMES = ["Qbert", "Breakout", "BattleZone"]
FRAMES_PER_GAME = 8  # 3ゲーム×8枚で、入力と期待画像は計975,744 byte。
STEP_GAP = 37  # 何 step おきに 1 枚取るか（似た画面ばかりにならないように）
SEED = 20261023


def pooled_frames(game, count, rng):
    # ALE自身のskipを止め、下の4フレーム反復と最後2枚のmax-poolを分けて管理する。
    env = gym.make(f"ALE/{game}-v5", frameskip=1, repeat_action_probability=0.0)
    env.reset(seed=SEED)
    ale = env.unwrapped.ale
    buf = [np.empty((210, 160), np.uint8), np.empty((210, 160), np.uint8)]
    out = []
    step = 0
    while len(out) < count:
        action = int(rng.integers(0, env.action_space.n))
        ended = False
        # 同じ行動を4回進め、3・4枚目の灰色画面だけを採取候補へ保存する。
        for t in range(4):
            _, _, term, trunc, _ = env.step(action)
            if t == 2:
                ale.getScreenGrayscale(buf[1])
            elif t == 3:
                ale.getScreenGrayscale(buf[0])
            if term or trunc:
                ended = True
                break
        # 途中終端で4枚揃わない候補は除外する。完了したstepだけを採取間隔へ数える。
        if ended:
            env.reset()
            continue
        step += 1
        # 完了stepを37回隔てて採取し、続く似た画面ばかりに偏らないようにする。
        if step % STEP_GAP == 0:
            out.append(np.maximum(buf[0], buf[1]).copy())
    env.close()
    return out


def main():
    if cv2.__version__ != "4.8.1":
        raise RuntimeError("Expected OpenCV 4.8.1, got " + cv2.__version__)
    out_dir = sys.argv[1]
    # 固定した正解データは、再実行で黙って置き換えない。
    for name in ("frames_210x160.u8", "expected_84x84.u8", "meta.json"):
        if os.path.exists(os.path.join(out_dir, name)):
            raise RuntimeError("Golden file already exists: " + name)
    os.makedirs(out_dir, exist_ok=True)
    # ゲーム順と乱数列を固定し、各ゲームで同じ条件の8枚を集める。
    rng = np.random.default_rng(SEED)
    frames, per_game = [], {}
    for game in GAMES:
        got = pooled_frames(game, FRAMES_PER_GAME, rng)
        frames.extend(got)
        per_game[game] = len(got)
    # 同じ入力から参照出力を作り、変換を加えないuint8配列として対で保存する。
    raw = np.stack(frames).astype(np.uint8)
    expected = np.stack([cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.uint8)
    raw.tofile(os.path.join(out_dir, "frames_210x160.u8"))
    expected.tofile(os.path.join(out_dir, "expected_84x84.u8"))
    # 再生成時に入力の同一性と版を確認できるよう、環境とchecksumを記録する。
    meta = {
        "count": int(raw.shape[0]),
        "frame_shape": [210, 160],
        "expected_shape": [84, 84],
        "per_game": per_game,
        "seed": SEED,
        "step_gap": STEP_GAP,
        "ale_py": ale_py.__version__,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        "opencv": cv2.__version__,
        "resize": "cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)",
        "dtype": "uint8",
        "order": "C",
        "sha256": {"frames_210x160.u8": hashlib.sha256(raw.tobytes()).hexdigest(),
                   "expected_84x84.u8": hashlib.sha256(expected.tobytes()).hexdigest()},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
