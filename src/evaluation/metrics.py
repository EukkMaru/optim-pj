import numpy as np

def compute_energy_metrics(assignment_matrix, tasks, servers, gamma, time_horizon):
    import sys
    sys.path.insert(0, 'src')
    from models.utilization import calc_all_utilizations
    from models.energy import calc_total_energy, calc_server_energy

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)
    total_energy = calc_total_energy(servers, utilizations, gamma, time_horizon)

    server_energies = [calc_server_energy(servers[i], utilizations[i], gamma, time_horizon) for i in range(len(servers))]

    return {
        'total_energy': total_energy,
        'server_energies': server_energies,
        'mean_server_energy': np.mean(server_energies),
        'std_server_energy': np.std(server_energies)
    }

def compute_latency_metrics(assignment_matrix, tasks, servers, tau, beta):
    import sys
    sys.path.insert(0, 'src')
    from models.latency import calc_total_latency
    from models.utilization import calc_all_utilizations

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)
    latencies = []

    for j, task in enumerate(tasks):
        server_idx = np.argmax(assignment_matrix[j, :])
        latency = calc_total_latency(task, servers[server_idx], tau[j, server_idx], beta[j, server_idx], utilizations[server_idx])
        latencies.append(latency)

    return {
        'mean_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99),
        'all_latencies': latencies
    }

def compute_utilization_metrics(assignment_matrix, tasks, servers):
    import sys
    sys.path.insert(0, 'src')
    from models.utilization import calc_all_utilizations

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)

    return {
        'utilizations': utilizations,
        'mean_utilization': np.mean(utilizations),
        'max_utilization': np.max(utilizations),
        'min_utilization': np.min(utilizations),
        'std_utilization': np.std(utilizations)
    }


if __name__ == "__main__":
    print("testing metrics module")

    import sys
    sys.path.insert(0, 'src')
    from models.task import generate_random_tasks
    from models.server import generate_random_servers
    from models.network import generate_network_params

    np.random.seed(42)

    num_tasks = 10
    num_servers = 3
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)
    tau, beta = generate_network_params(num_tasks, num_servers)

    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)
    for j in range(num_tasks):
        assignment_matrix[j, j % num_servers] = 1

    gamma = 2.5
    time_horizon = 3600.0

    print("\ntesting energy metrics...")
    energy_metrics = compute_energy_metrics(assignment_matrix, tasks, servers, gamma, time_horizon)
    print(f"total energy: {energy_metrics['total_energy']:.2e}J")
    print(f"mean server energy: {energy_metrics['mean_server_energy']:.2e}J")
    assert energy_metrics['total_energy'] > 0
    assert len(energy_metrics['server_energies']) == num_servers
    print("energy metrics test passed")

    print("\ntesting latency metrics...")
    latency_metrics = compute_latency_metrics(assignment_matrix, tasks, servers, tau, beta)
    print(f"mean latency: {latency_metrics['mean_latency']:.6f}s")
    print(f"median latency: {latency_metrics['median_latency']:.6f}s")
    print(f"p95 latency: {latency_metrics['p95_latency']:.6f}s")
    assert latency_metrics['mean_latency'] > 0
    assert len(latency_metrics['all_latencies']) == num_tasks
    print("latency metrics test passed")

    print("\ntesting utilization metrics...")
    util_metrics = compute_utilization_metrics(assignment_matrix, tasks, servers)
    print(f"mean utilization: {util_metrics['mean_utilization']:.6e}")
    print(f"max utilization: {util_metrics['max_utilization']:.6e}")
    assert len(util_metrics['utilizations']) == num_servers
    assert all(u >= 0 for u in util_metrics['utilizations'])
    print("utilization metrics test passed")

    print("\nall tests passed")
