"""行動候補と事前分布の説明用の数値例を検算する。学習実験ではない。"""
from fractions import Fraction
from math import exp, isclose, log


def main():
    p = Fraction(1, 10)
    for k in (1, 5, 10, 20):
        coverage = 1 - (1 - p) ** k
        print(f"k={k}, coverage={float(coverage):.12f}")
    assert 1 - (1 - p) ** 5 == Fraction(40951, 100000)
    assert 1 - (1 - p) ** 10 == Fraction(6513215599, 10000000000)
    q_good = 0.01 * exp(1) / (0.01 * exp(1) + 0.99)
    assert isclose(q_good, 0.026723630989395224)
    assert q_good < 0.03
    assert 1 - (1 - p) ** 50 > 1 - (1 - p) ** 5
    print(f"q_good={q_good:.12f}")
    for p_left in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        assert isclose(p_left / 2 + (1 - p_left) / 2, 0.5)
        mean_kl = -0.5 * log(p_left) - 0.5 * log(1 - p_left)
        assert mean_kl >= log(2) - 1e-12
    assert isclose(-0.5 * log(0.5) - 0.5 * log(0.5), log(2))
    print(f"aliased_observation_best_mean_kl={log(2):.12f}")
    print(f"double_softmax_max_three_actions={exp(1) / (exp(1) + 2):.12f}")
    for prediction in ((Fraction(0), Fraction(1)),
                       (Fraction(1, 2), Fraction(1, 2)),
                       (Fraction(1), Fraction(0))):
        expected_bonus = sum(
            sum((target[i] - prediction[i]) ** 2 for i in range(2)) / 4
            for target in ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
        )
        assert expected_bonus >= Fraction(1, 4)
        if prediction == (Fraction(1, 2), Fraction(1, 2)):
            assert expected_bonus == Fraction(1, 4)
    print("unpredictable_teacher_minimum_expected_bonus=1/4")
    print("PASS: finite candidate coverage and regularized two-action example")


if __name__ == "__main__":
    main()
