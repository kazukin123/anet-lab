"""サーベイの目標切替と非線形変換の反例を独立に検算する。"""
from fractions import Fraction
import json
from math import tanh, isclose


def evaluate(action, use_next_goal):
    # 最初の遷移だけに報酬があり、以後は同じ拡張状態へ自己遷移する。
    gamma = Fraction(9, 10)
    state, goal = "start", "g0"
    potential = {("left", "g0"): Fraction(1)}
    next_state = "left" if action == "A" else "right"
    next_goal = "g1" if action == "A" else "g0"
    reward = Fraction(0) if action == "A" else Fraction(1, 10)
    scoring_goal = next_goal if use_next_goal else goal
    shaping = gamma * potential.get((next_state, scoring_goal), 0) - potential.get((state, goal), 0)

    # 吸収後は正しい現在の目標を保持するため、無限個の後続項はすべて0。
    absorbing_phi = potential.get((next_state, next_goal), 0)
    tail_per_step = (gamma - 1) * absorbing_phi
    tail_return = gamma * tail_per_step / (1 - gamma)
    return reward, reward + shaping + tail_return


if __name__ == "__main__":
    base = {a: evaluate(a, False)[0] for a in ("A", "B")}
    old_goal = {a: evaluate(a, False)[1] for a in ("A", "B")}
    next_goal = {a: evaluate(a, True)[1] for a in ("A", "B")}
    assert max(base, key=base.get) == "B"
    assert max(old_goal, key=old_goal.get) == "A"
    assert next_goal == base
    print(json.dumps({"base": {a: str(v) for a, v in base.items()},
                      "old_goal_shaping": {a: str(v) for a, v in old_goal.items()},
                      "next_goal_shaping": {a: str(v) for a, v in next_goal.items()},
                      "result": "counterexample confirmed"}, indent=2))

    # 同じ終端へ到達する2経路。中間状態のpotentialだけを変える。
    gamma = 0.9
    potentials = {"A": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 0.0)}
    base_return = {"A": 0.0, "B": 0.01}
    linear_return, squashed_return = {}, {}
    for action, phi in potentials.items():
        delta = [gamma * phi[t + 1] - phi[t] for t in range(2)]
        linear_return[action] = base_return[action] + sum(gamma ** t * f for t, f in enumerate(delta))
        squashed_return[action] = base_return[action] + sum(gamma ** t * tanh(f) for t, f in enumerate(delta))
    assert all(isclose(linear_return[a], base_return[a], abs_tol=1e-12) for a in potentials)
    assert max(base_return, key=base_return.get) == "B"
    assert max(squashed_return, key=squashed_return.get) == "A"
    print(json.dumps({"linear_return": linear_return,
                      "tanh_return": squashed_return,
                      "result": "nonlinear counterexample confirmed"}, indent=2))
