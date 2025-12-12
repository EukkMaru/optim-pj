import numpy as np
from .power import calc_total_power

def calc_energy(power, time_horizon):
    return power * time_horizon

def calc_server_energy(server, utilization, gamma, time_horizon):
    power = calc_total_power(server.p_idle, server.alpha, server.freq, gamma, utilization)
    return calc_energy(power, time_horizon)

def calc_total_energy(servers, utilizations, gamma, time_horizon):
    total = 0.0
    for i, server in enumerate(servers):
        total += calc_server_energy(server, utilizations[i], gamma, time_horizon)
    return total


if __name__ == "__main__":
    print("testing energy module")

    from server import EdgeServer

    power = 100.0
    time_horizon = 3600.0
    energy = calc_energy(power, time_horizon)
    expected = 100.0 * 3600.0
    print(f"energy: {energy:.2f}J, expected: {expected:.2f}J")
    assert np.isclose(energy, expected)
    print("calc_energy test passed")

    server = EdgeServer(0, 1e9, 3e9, 1.0, 5e-10, 10.0)
    server.set_freq(2e9)
    utilization = 0.7
    gamma = 2.5
    server_energy = calc_server_energy(server, utilization, gamma, time_horizon)
    print(f"server energy: {server_energy:.2e}J")
    assert server_energy > 0
    print("calc_server_energy test passed")

    servers = [
        EdgeServer(0, 1e9, 3e9, 1.0, 5e-10, 10.0),
        EdgeServer(1, 1e9, 3e9, 1.5, 3e-10, 12.0)
    ]
    servers[0].set_freq(2e9)
    servers[1].set_freq(1.5e9)
    utilizations = np.array([0.6, 0.8])

    total_energy = calc_total_energy(servers, utilizations, gamma, time_horizon)
    print(f"total energy: {total_energy:.2e}J")
    assert total_energy > 0
    print("calc_total_energy test passed")

    energy0 = calc_server_energy(servers[0], utilizations[0], gamma, time_horizon)
    energy1 = calc_server_energy(servers[1], utilizations[1], gamma, time_horizon)
    expected_total = energy0 + energy1
    assert np.isclose(total_energy, expected_total)
    print(f"verified total = {total_energy:.2e}J = {energy0:.2e} + {energy1:.2e}J")

    print("all tests passed")
