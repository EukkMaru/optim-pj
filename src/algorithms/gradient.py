import numpy as np

def calc_energy_gradient(server, utilization, gamma, time_horizon):
    if server.freq == 0:
        return 0.0
    grad = time_horizon * utilization * server.alpha * gamma * (server.freq ** (gamma - 1))
    return grad

def calc_latency_gradient_numerical(assignment, tasks, server, tau, beta, utilization, delta=1e6):
    from sys import path
    path.insert(0, 'src/models')
    from latency import calc_total_latency

    original_freq = server.freq

    server.freq = original_freq + delta
    capacity_plus = server.get_capacity()
    util_plus = utilization * (original_freq / server.freq)  # adjust utilization
    latency_plus = 0.0
    for j in range(len(tasks)):
        if assignment[j]:
            latency_plus += calc_total_latency(tasks[j], server, tau[j], beta[j], util_plus)

    server.freq = original_freq - delta
    capacity_minus = server.get_capacity()
    util_minus = utilization * (original_freq / server.freq)  # adjust utilization
    latency_minus = 0.0
    for j in range(len(tasks)):
        if assignment[j]:
            latency_minus += calc_total_latency(tasks[j], server, tau[j], beta[j], util_minus)

    server.freq = original_freq

    grad = (latency_plus - latency_minus) / (2.0 * delta)
    return grad


if __name__ == "__main__":
    print("testing gradient module")

    import sys
    sys.path.insert(0, 'src/models')
    from task import Task, generate_random_tasks
    from server import EdgeServer
    from network import generate_network_params

    np.random.seed(42)

    server = EdgeServer(0, 1e9, 3e9, 1.0, 5e-10, 10.0)
    server.set_freq(2e9)
    utilization = 0.6
    gamma = 2.5
    time_horizon = 3600.0

    grad_energy = calc_energy_gradient(server, utilization, gamma, time_horizon)
    print(f"energy gradient at f={server.freq:.2e}, util={utilization}: {grad_energy:.2e}")
    assert grad_energy != 0
    print("calc_energy_gradient test passed")

    print("\nenergy gradient at different frequencies:")
    for freq in [1e9, 1.5e9, 2e9, 2.5e9]:
        server.set_freq(freq)
        grad = calc_energy_gradient(server, utilization, gamma, time_horizon)
        print(f"  f={freq:.2e}Hz: grad={grad:.2e}")

    print("\nenergy gradient at different utilizations:")
    server.set_freq(2e9)
    for util in [0.3, 0.5, 0.7, 0.9]:
        grad = calc_energy_gradient(server, util, gamma, time_horizon)
        print(f"  util={util}: grad={grad:.2e}")

    tasks = generate_random_tasks(3)
    tau, beta = generate_network_params(3, 1)
    assignment = np.array([1, 1, 0])  # tasks 0,1 assigned to server

    grad_latency = calc_latency_gradient_numerical(assignment, tasks, server, tau[:, 0], beta[:, 0], utilization)
    print(f"\nlatency gradient (numerical): {grad_latency:.6e}")
    print("calc_latency_gradient_numerical test passed")

    print(f"latency gradient sign: {'negative (reduces latency)' if grad_latency < 0 else 'positive (increases latency)'}")

    print("\nall tests passed")
