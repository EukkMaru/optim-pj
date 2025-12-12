import numpy as np

def calc_dynamic_power(alpha, freq, gamma):
    return alpha * (freq ** gamma)

def calc_total_power(p_idle, alpha, freq, gamma, utilization):
    return p_idle + utilization * calc_dynamic_power(alpha, freq, gamma)


if __name__ == "__main__":
    print("testing power module")

    alpha = 5e-10
    freq = 2e9
    gamma = 2.5
    p_dyn = calc_dynamic_power(alpha, freq, gamma)
    expected = 5e-10 * (2e9 ** 2.5)
    print(f"dynamic power: {p_dyn:.4f}W, expected: {expected:.4f}W")
    assert np.isclose(p_dyn, expected)
    print("calc_dynamic_power test passed")

    p_idle = 10.0
    utilization = 0.7
    p_total = calc_total_power(p_idle, alpha, freq, gamma, utilization)
    expected_total = 10.0 + 0.7 * p_dyn
    print(f"total power: {p_total:.4f}W, expected: {expected_total:.4f}W")
    assert np.isclose(p_total, expected_total)
    print("calc_total_power test passed")

    for test_gamma in [2.0, 2.5, 3.0]:
        p_dyn = calc_dynamic_power(alpha, freq, test_gamma)
        print(f"dynamic power with gamma={test_gamma}: {p_dyn:.4f}W")

    # test edge cases
    p_total_zero = calc_total_power(p_idle, alpha, freq, gamma, 0.0)
    assert np.isclose(p_total_zero, p_idle)
    print(f"zero utilization power: {p_total_zero:.4f}W (should be {p_idle:.4f}W)")

    p_dyn_full = calc_dynamic_power(alpha, freq, gamma)
    p_total_full = calc_total_power(p_idle, alpha, freq, gamma, 1.0)
    expected_full = p_idle + p_dyn_full
    assert np.isclose(p_total_full, expected_full, rtol=1e-5)
    print(f"full utilization power: {p_total_full:.4f}W")

    print("all tests passed")
