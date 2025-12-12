import numpy as np

def calc_penalty(utilization, epsilon=0.05, eta=1000.0):
    violation = max(0, utilization - 1.0 + epsilon)
    return eta * (violation ** 2)

def calc_total_penalty(utilizations, epsilon=0.05, eta=1000.0):
    total = 0.0
    for util in utilizations:
        total += calc_penalty(util, epsilon, eta)
    return total


if __name__ == "__main__":
    print("testing penalty module")

    util_safe = 0.7
    penalty = calc_penalty(util_safe)
    print(f"utilization {util_safe}: penalty={penalty:.2f}")
    assert penalty == 0.0
    print("safe utilization test passed")

    util_high = 0.96
    epsilon = 0.05
    penalty_high = calc_penalty(util_high, epsilon)
    expected = 1000.0 * (0.96 - 1.0 + 0.05) ** 2
    print(f"utilization {util_high}: penalty={penalty_high:.2f}, expected={expected:.2f}")
    assert np.isclose(penalty_high, expected)
    print("high utilization test passed")

    util_unstable = 1.1
    penalty_unstable = calc_penalty(util_unstable, epsilon)
    expected_unstable = 1000.0 * (1.1 - 1.0 + 0.05) ** 2
    print(f"utilization {util_unstable}: penalty={penalty_unstable:.2f}, expected={expected_unstable:.2f}")
    assert np.isclose(penalty_unstable, expected_unstable)
    print("unstable utilization test passed")

    util_boundary = 0.95
    penalty_boundary = calc_penalty(util_boundary, epsilon)
    print(f"utilization {util_boundary}: penalty={penalty_boundary:.2f}")
    assert penalty_boundary == 0.0
    print("boundary test passed")

    utilizations = np.array([0.5, 0.7, 0.96, 1.1])
    total_penalty = calc_total_penalty(utilizations, epsilon)
    expected_total = sum(calc_penalty(u, epsilon) for u in utilizations)
    print(f"\ntotal penalty for {utilizations}: {total_penalty:.2f}")
    assert np.isclose(total_penalty, expected_total)
    print("total penalty test passed")

    print("\npenalty with different epsilon values:")
    util_test = 0.98
    for eps in [0.01, 0.05, 0.1]:
        pen = calc_penalty(util_test, eps)
        print(f"  epsilon={eps}: penalty={pen:.2f}")

    print("\npenalty with different eta values:")
    for eta_val in [100.0, 1000.0, 10000.0]:
        pen = calc_penalty(0.98, 0.05, eta_val)
        print(f"  eta={eta_val}: penalty={pen:.2f}")

    print("\nall tests passed")
