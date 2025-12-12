import numpy as np

def calc_queueing_delay(utilization, capacity):
    if utilization >= 1.0:
        return float('inf')  # unstable queue
    if capacity == 0:
        return float('inf')
    return utilization / (capacity * (1.0 - utilization))

def calc_service_time(cpu_cycles, capacity):
    if capacity == 0:
        return float('inf')
    return cpu_cycles / capacity


if __name__ == "__main__":
    print("testing queueing module")

    utilization = 0.7
    capacity = 1e9  # 1 GHz capacity
    delay = calc_queueing_delay(utilization, capacity)
    expected = 0.7 / (1e9 * (1.0 - 0.7))
    print(f"queueing delay: {delay:.6e}s, expected: {expected:.6e}s")
    assert np.isclose(delay, expected)
    print("calc_queueing_delay test passed")

    # test edge case: high utilization
    high_util = 0.99
    high_delay = calc_queueing_delay(high_util, capacity)
    print(f"high utilization ({high_util}) delay: {high_delay:.6e}s")
    assert high_delay > delay  # should be higher
    print("high utilization test passed")

    # test edge case: unstable system
    unstable_util = 1.0
    unstable_delay = calc_queueing_delay(unstable_util, capacity)
    assert unstable_delay == float('inf')
    print(f"unstable utilization delay: {unstable_delay}")
    print("unstable system test passed")

    cpu_cycles = 1e8
    service_time = calc_service_time(cpu_cycles, capacity)
    expected_service = 1e8 / 1e9
    print(f"service time: {service_time:.6f}s, expected: {expected_service:.6f}s")
    assert np.isclose(service_time, expected_service)
    print("calc_service_time test passed")

    for cap in [5e8, 1e9, 2e9]:
        st = calc_service_time(cpu_cycles, cap)
        print(f"service time at {cap:.2e} capacity: {st:.6f}s")

    print("all tests passed")
