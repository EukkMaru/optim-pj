import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective
from models.latency import calc_total_latency
from algorithms.freq_optimizer import FrequencyOptimizer

def greedy_minimum_latency(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, epsilon=0.05):
    num_tasks = len(tasks)
    num_servers = len(servers)

    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)

    for j, task in enumerate(tasks):
        min_latency = float('inf')
        best_server = 0

        for i in range(num_servers):
            net_latency = tau[j, i] + beta[j, i] * task.data_size

            if net_latency < min_latency:
                min_latency = net_latency
                best_server = i

        assignment_matrix[j, best_server] = 1

    freq_opt = FrequencyOptimizer(tasks, servers, assignment_matrix, tau, beta, gamma, lambda_weight, time_horizon, epsilon)
    freq_opt.optimize()

    objective, energy, latency = calc_objective(assignment_matrix, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    return assignment_matrix, objective, energy, latency


if __name__ == "__main__":
    print("testing greedy_latency module")

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

    print("\nrunning greedy minimum latency...")
    assignment, obj, energy, latency = greedy_minimum_latency(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

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

    task_assignments = np.argmax(assignment, axis=1)
    print(f"\ntask assignments: {task_assignments}")
    assert len(task_assignments) == num_tasks
    print("all tasks assigned test passed")

    for i in range(num_servers):
        count = np.sum(assignment[:, i])
        print(f"server {i}: {count} tasks")

    print("\nall tests passed")
