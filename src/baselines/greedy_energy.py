import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective
from models.utilization import calc_utilization
from models.power import calc_total_power
from algorithms.freq_optimizer import FrequencyOptimizer

def greedy_energy_aware(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, epsilon=0.05):
    num_tasks = len(tasks)
    num_servers = len(servers)

    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)
    current_utilizations = np.zeros(num_servers)

    for j, task in enumerate(tasks):
        min_energy_increase = float('inf')
        best_server = 0

        for i in range(num_servers):
            temp_assignment = assignment_matrix[:, i].copy()
            temp_assignment[j] = 1

            new_util = calc_utilization(temp_assignment, tasks, servers[i])

            # avoid overload
            if new_util >= (1.0 - epsilon):
                continue

            old_power = calc_total_power(servers[i].p_idle, servers[i].alpha, servers[i].freq, gamma, current_utilizations[i])
            new_power = calc_total_power(servers[i].p_idle, servers[i].alpha, servers[i].freq, gamma, new_util)
            energy_increase = (new_power - old_power) * time_horizon

            if energy_increase < min_energy_increase:
                min_energy_increase = energy_increase
                best_server = i

        assignment_matrix[j, best_server] = 1
        current_utilizations[best_server] = calc_utilization(assignment_matrix[:, best_server], tasks, servers[best_server])

    freq_opt = FrequencyOptimizer(tasks, servers, assignment_matrix, tau, beta, gamma, lambda_weight, time_horizon, epsilon)
    freq_opt.optimize()

    objective, energy, latency = calc_objective(assignment_matrix, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    return assignment_matrix, objective, energy, latency


if __name__ == "__main__":
    print("testing greedy_energy module")

    from task import generate_random_tasks
    from server import generate_random_servers
    from network import generate_network_params

    np.random.seed(42)

    num_tasks = 10
    num_servers = 3
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)
    tau, beta = generate_network_params(num_tasks, num_servers)

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    print("\nrunning greedy energy aware...")
    assignment, obj, energy, latency = greedy_energy_aware(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    print(f"\nassignment matrix:\n{assignment}")
    print(f"objective: {obj:.2e}")
    print(f"energy: {energy:.2e}J")
    print(f"latency: {latency:.6f}s")

    assert assignment.shape == (num_tasks, num_servers)
    assert np.all(np.sum(assignment, axis=1) == 1)
    print("\nassignment validity test passed")

    assert obj > 0
    assert energy > 0
    assert latency > 0
    print("objective values test passed")

    for i in range(num_servers):
        count = np.sum(assignment[:, i])
        print(f"server {i}: {count} tasks")

    print("\nall tests passed")
