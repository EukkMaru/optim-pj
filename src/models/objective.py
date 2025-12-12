import numpy as np
from .energy import calc_total_energy
from .latency import calc_total_latency
from .utilization import calc_all_utilizations

def calc_objective(assignment_matrix, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon):
    num_tasks = len(tasks)
    num_servers = len(servers)

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)

    total_energy = calc_total_energy(servers, utilizations, gamma, time_horizon)

    total_latency = 0.0
    for j in range(num_tasks):
        for i in range(num_servers):
            if assignment_matrix[j, i] == 1:
                latency = calc_total_latency(tasks[j], servers[i], tau[j, i], beta[j, i], utilizations[i])
                total_latency += latency

    objective = total_energy + lambda_weight * total_latency
    return objective, total_energy, total_latency


if __name__ == "__main__":
    print("testing objective module")

    from task import generate_random_tasks
    from server import generate_random_servers
    from network import generate_network_params

    np.random.seed(42)

    num_tasks = 5
    num_servers = 2
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)
    tau, beta = generate_network_params(num_tasks, num_servers)

    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)
    for j in range(num_tasks):
        assignment_matrix[j, j % num_servers] = 1

    print(f"assignment matrix:\n{assignment_matrix}")

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    obj, energy, latency = calc_objective(assignment_matrix, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)
    print(f"\nobjective: {obj:.2e}")
    print(f"energy: {energy:.2e}J")
    print(f"latency: {latency:.6f}s")

    assert obj > 0
    assert energy > 0
    assert latency > 0
    print("\ncalc_objective test passed")

    print("\nobjective with different lambda weights:")
    for lw in [0.1, 1.0, 10.0, 100.0]:
        obj_lw, _, _ = calc_objective(assignment_matrix, tasks, servers, tau, beta, gamma, lw, time_horizon)
        print(f"  lambda={lw}: objective={obj_lw:.2e}")

    assignment_single = np.zeros((num_tasks, num_servers), dtype=int)
    assignment_single[:, 0] = 1
    obj_single, energy_single, latency_single = calc_objective(assignment_single, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)
    print(f"\nall tasks on server 0:")
    print(f"  objective: {obj_single:.2e}")
    print(f"  energy: {energy_single:.2e}J")
    print(f"  latency: {latency_single:.6f}s")

    print("\nall tests passed")
