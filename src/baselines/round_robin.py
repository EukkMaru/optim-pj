import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective
from algorithms.freq_optimizer import FrequencyOptimizer

def round_robin(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, epsilon=0.05):
    num_tasks = len(tasks)
    num_servers = len(servers)

    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)

    for j in range(num_tasks):
        server_idx = j % num_servers
        assignment_matrix[j, server_idx] = 1

    freq_opt = FrequencyOptimizer(tasks, servers, assignment_matrix, tau, beta, gamma, lambda_weight, time_horizon, epsilon)
    freq_opt.optimize()

    objective, energy, latency = calc_objective(assignment_matrix, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    return assignment_matrix, objective, energy, latency


if __name__ == "__main__":
    print("testing round_robin module")

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

    print("\nrunning round robin...")
    assignment, obj, energy, latency = round_robin(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

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
    expected = [j % num_servers for j in range(num_tasks)]
    assert np.array_equal(task_assignments, expected)
    print("round robin distribution test passed")

    for i in range(num_servers):
        count = np.sum(assignment[:, i])
        print(f"server {i}: {count} tasks")
        expected_count = len([j for j in range(num_tasks) if j % num_servers == i])
        assert count == expected_count

    print("\nall tests passed")
