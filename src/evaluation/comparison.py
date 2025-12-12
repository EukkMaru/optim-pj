import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective

def compare_algorithms(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, epsilon=0.05):
    results = {}

    print("running greedy latency...")
    from baselines.greedy_latency import greedy_minimum_latency
    assignment_gl, obj_gl, energy_gl, latency_gl = greedy_minimum_latency(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, epsilon)
    results['greedy_latency'] = {
        'assignment': assignment_gl,
        'objective': obj_gl,
        'energy': energy_gl,
        'latency': latency_gl
    }


    print("running round robin...")
    from baselines.round_robin import round_robin
    assignment_rr, obj_rr, energy_rr, latency_rr = round_robin(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, epsilon)
    results['round_robin'] = {
        'assignment': assignment_rr,
        'objective': obj_rr,
        'energy': energy_rr,
        'latency': latency_rr
    }

    return results

def print_comparison(results):
    print("\nresults comparison:")
    for alg_name, alg_results in results.items():
        print(f"{alg_name}: objective={alg_results['objective']:.2e}, energy={alg_results['energy']:.2e}J, latency={alg_results['latency']:.6f}s")

    best_alg = min(results.items(), key=lambda x: x[1]['objective'])
    print(f"best algorithm: {best_alg[0]} (objective: {best_alg[1]['objective']:.2e})")


if __name__ == "__main__":
    print("testing comparison module")

    from models.task import generate_random_tasks
    from models.server import generate_random_servers
    from models.network import generate_network_params

    np.random.seed(42)

    num_tasks = 10
    num_servers = 3
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)
    tau, beta = generate_network_params(num_tasks, num_servers)

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    print("\nrunning algorithm comparison...")
    results = compare_algorithms(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    assert 'greedy_latency' in results
    assert 'round_robin' in results
    print("\nresults structure test passed")

    for alg_name, alg_results in results.items():
        assert 'objective' in alg_results
        assert 'energy' in alg_results
        assert 'latency' in alg_results
        assert alg_results['objective'] > 0
        print(f"{alg_name} validity test passed")

    print_comparison(results)

    print("\nall tests passed")
