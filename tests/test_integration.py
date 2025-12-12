import numpy as np
import sys
sys.path.insert(0, 'src')

def test_end_to_end_small_problem():
    # test complete workflow with small problem
    print("testing end-to-end small problem...")

    from data.synthetic_generator import generate_synthetic_workload
    from baselines.round_robin import round_robin
    from evaluation.validator import validate_solution
    from evaluation.metrics import compute_energy_metrics, compute_latency_metrics

    # generate small problem
    tasks, servers, tau, beta = generate_synthetic_workload(num_tasks=5, num_servers=2, seed=42)

    # run algorithm
    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    assignment, obj, energy, latency = round_robin(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    # validate
    results = validate_solution(assignment, tasks, servers)
    assert results['valid'], f"solution not valid: {results['errors']}"

    # compute metrics
    energy_metrics = compute_energy_metrics(assignment, tasks, servers, gamma, time_horizon)
    latency_metrics = compute_latency_metrics(assignment, tasks, servers, tau, beta)

    assert energy_metrics['total_energy'] > 0
    assert latency_metrics['mean_latency'] > 0

    print(f"  objective: {obj:.2e}")
    print(f"  total energy: {energy_metrics['total_energy']:.2e}J")
    print(f"  mean latency: {latency_metrics['mean_latency']:.6f}s")
    print("end-to-end small problem test passed")

def test_baseline_comparison():
    # test baseline algorithm comparison
    print("\ntesting baseline comparison...")

    from data.synthetic_generator import generate_synthetic_workload
    from evaluation.comparison import compare_algorithms, print_comparison

    # generate problem
    tasks, servers, tau, beta = generate_synthetic_workload(num_tasks=10, num_servers=3, seed=42)

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    # run comparison
    results = compare_algorithms(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)

    # verify all algorithms ran
    assert 'greedy_latency' in results
    assert 'round_robin' in results

    # print results
    print_comparison(results)

    print("baseline comparison test passed")

def test_tabu_search_integration():
    # test tabu search with full workflow
    print("\ntesting tabu search integration...")

    from data.synthetic_generator import generate_synthetic_workload
    from algorithms.tabu_search import TabuSearch
    from evaluation.validator import validate_solution

    # generate problem
    tasks, servers, tau, beta = generate_synthetic_workload(num_tasks=8, num_servers=3, seed=42)

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    # initial assignment (round robin)
    initial_assignment = np.zeros((len(tasks), len(servers)), dtype=int)
    for j in range(len(tasks)):
        initial_assignment[j, j % len(servers)] = 1

    # run tabu search
    tabu = TabuSearch(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, tenure=5, max_iter=5)
    best_solution, best_obj, history = tabu.optimize(initial_assignment)

    # validate solution
    results = validate_solution(best_solution, tasks, servers)
    assert results['valid'], f"tabu search solution not valid: {results['errors']}"

    print(f"  best objective: {best_obj:.2e}")
    print(f"  iterations: {len(history)}")
    print("tabu search integration test passed")

def test_pso_integration():
    # test PSO with full workflow
    print("\ntesting PSO integration...")

    from data.synthetic_generator import generate_synthetic_workload
    from algorithms.pso import PSO
    from evaluation.validator import validate_solution

    # generate problem
    tasks, servers, tau, beta = generate_synthetic_workload(num_tasks=8, num_servers=3, seed=42)

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    # run PSO
    pso = PSO(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, swarm_size=5, max_iter=3)
    best_solution, best_obj, history = pso.optimize()

    # validate solution
    results = validate_solution(best_solution, tasks, servers)
    assert results['valid'], f"PSO solution not valid: {results['errors']}"

    print(f"  best objective: {best_obj:.2e}")
    print(f"  iterations: {len(history)}")
    print("PSO integration test passed")


if __name__ == "__main__":
    print("running integration tests")

    test_end_to_end_small_problem()
    test_baseline_comparison()
    test_tabu_search_integration()
    test_pso_integration()

    print("\nall integration tests passed")
