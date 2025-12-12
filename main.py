#!/usr/bin/env python3
import argparse
import numpy as np
import sys
sys.path.insert(0, 'src')

from data.synthetic_generator import generate_synthetic_workload
from data.google_loader import load_machine_events, machines_to_servers, generate_tasks_from_trace
from algorithms.tabu_search import TabuSearch
from algorithms.pso import PSO
from baselines.greedy_latency import greedy_minimum_latency
from baselines.greedy_energy import greedy_energy_aware
from baselines.round_robin import round_robin
from evaluation.comparison import print_comparison
from evaluation.validator import validate_solution
from evaluation.metrics import compute_energy_metrics, compute_latency_metrics, compute_utilization_metrics

def run_experiment(args):
    print("edge computing energy-latency optimization")

    if args.dataset == 'synthetic':
        print(f"\ngenerating synthetic workload: {args.num_tasks} tasks, {args.num_servers} servers")
        tasks, servers, tau, beta = generate_synthetic_workload(
            num_tasks=args.num_tasks,
            num_servers=args.num_servers,
            seed=args.seed
        )
    elif args.dataset == 'google':
        print(f"\nloading google cluster trace data...")
        machines = load_machine_events("data/raw/machine_events-000000000000.json.gz", max_machines=args.num_servers)
        servers = machines_to_servers(machines)
        tasks = generate_tasks_from_trace(args.num_tasks, seed=args.seed)
        from models.network import generate_network_params
        tau, beta = generate_network_params(len(tasks), len(servers))
    else:
        print(f"unknown dataset: {args.dataset}")
        return

    print(f"  tasks: {len(tasks)}")
    print(f"  servers: {len(servers)}")

    gamma = args.gamma
    lambda_weight = args.lambda_weight
    time_horizon = args.time_horizon

    print(f"\nparameters:")
    print(f"  gamma: {gamma}")
    print(f"  lambda_weight: {lambda_weight}")
    print(f"  time_horizon: {time_horizon}s")

    results = {}

    if args.algorithm in ['all', 'baseline']:
        print("\nrunning baseline algorithms...")

        if args.algorithm in ['greedy_latency', 'baseline', 'all']:
            print("\ngreedy minimum latency...")
            assignment_gl, obj_gl, energy_gl, latency_gl = greedy_minimum_latency(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)
            results['greedy_latency'] = {
                'assignment': assignment_gl,
                'objective': obj_gl,
                'energy': energy_gl,
                'latency': latency_gl
            }
            print(f"  objective: {obj_gl:.2e}")


        if args.algorithm in ['round_robin', 'baseline', 'all']:
            print("\nround robin...")
            assignment_rr, obj_rr, energy_rr, latency_rr = round_robin(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon)
            results['round_robin'] = {
                'assignment': assignment_rr,
                'objective': obj_rr,
                'energy': energy_rr,
                'latency': latency_rr
            }
            print(f"  objective: {obj_rr:.2e}")

    if args.algorithm in ['all', 'tabu']:
        print("\nrunning tabu search...")

        initial_assignment_tabu = np.zeros((len(tasks), len(servers)), dtype=int)
        for j in range(len(tasks)):
            server = np.random.randint(0, len(servers))
            initial_assignment_tabu[j, server] = 1

        tabu = TabuSearch(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon,
                         tenure=args.tabu_tenure, max_iter=args.tabu_iter)
        best_solution, best_obj, history = tabu.optimize(initial_assignment_tabu)

        from evaluation.metrics import compute_energy_metrics, compute_latency_metrics
        energy_metrics_tabu = compute_energy_metrics(best_solution, tasks, servers, gamma, time_horizon)
        latency_metrics_tabu = compute_latency_metrics(best_solution, tasks, servers, tau, beta)

        results['tabu_search'] = {
            'assignment': best_solution,
            'objective': best_obj,
            'energy': energy_metrics_tabu['total_energy'],
            'latency': latency_metrics_tabu['mean_latency'],
            'history': history
        }
        print(f"  final objective: {best_obj:.2e}")

    if args.algorithm in ['all', 'pso']:
        print("\nrunning particle swarm optimization...")

        pso = PSO(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon,
                 swarm_size=args.pso_swarm, max_iter=args.pso_iter)
        best_solution, best_obj, history = pso.optimize()

        energy_metrics_pso = compute_energy_metrics(best_solution, tasks, servers, gamma, time_horizon)
        latency_metrics_pso = compute_latency_metrics(best_solution, tasks, servers, tau, beta)

        results['pso'] = {
            'assignment': best_solution,
            'objective': best_obj,
            'energy': energy_metrics_pso['total_energy'],
            'latency': latency_metrics_pso['mean_latency'],
            'history': history
        }
        print(f"  final objective: {best_obj:.2e}")

    if len(results) > 1:
        print_comparison(results)

    best_alg = min(results.items(), key=lambda x: x[1]['objective'])
    print(f"\ndetailed metrics for best algorithm: {best_alg[0]}")

    best_assignment = best_alg[1]['assignment']

    validation = validate_solution(best_assignment, tasks, servers)
    print(f"\nvalid: {validation['valid']}")
    if validation['errors']:
        print(f"errors: {validation['errors']}")
    if validation['warnings']:
        print(f"warnings: {validation['warnings']}")

    energy_metrics = compute_energy_metrics(best_assignment, tasks, servers, gamma, time_horizon)
    latency_metrics = compute_latency_metrics(best_assignment, tasks, servers, tau, beta)
    util_metrics = compute_utilization_metrics(best_assignment, tasks, servers)

    print(f"\nenergy metrics:")
    print(f"  total energy: {energy_metrics['total_energy']:.2e}J")
    print(f"  mean server energy: {energy_metrics['mean_server_energy']:.2e}J")

    print(f"\nlatency metrics:")
    print(f"  mean latency: {latency_metrics['mean_latency']:.6f}s")
    print(f"  median latency: {latency_metrics['median_latency']:.6f}s")
    print(f"  p95 latency: {latency_metrics['p95_latency']:.6f}s")
    print(f"  p99 latency: {latency_metrics['p99_latency']:.6f}s")

    print(f"\nutilization metrics:")
    print(f"  mean utilization: {util_metrics['mean_utilization']:.4e}")
    print(f"  max utilization: {util_metrics['max_utilization']:.4e}")
    print(f"  per-server: {[f'{u:.4e}' for u in util_metrics['utilizations']]}")

    print("\nexperiment completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="edge computing energy-latency optimization")

    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'google'],
                       help='dataset to use')
    parser.add_argument('--num_tasks', type=int, default=20, help='number of tasks')
    parser.add_argument('--num_servers', type=int, default=5, help='number of servers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--gamma', type=float, default=2.5, help='power exponent')
    parser.add_argument('--lambda_weight', type=float, default=1.0, help='latency weight')
    parser.add_argument('--time_horizon', type=float, default=3600.0, help='time horizon (seconds)')

    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['all', 'baseline', 'tabu', 'pso', 'greedy_latency', 'round_robin'],
                       help='algorithm to run')

    parser.add_argument('--tabu_tenure', type=int, default=7, help='tabu tenure')
    parser.add_argument('--tabu_iter', type=int, default=50, help='tabu search iterations')

    parser.add_argument('--pso_swarm', type=int, default=20, help='PSO swarm size')
    parser.add_argument('--pso_iter', type=int, default=50, help='PSO iterations')

    args = parser.parse_args()

    run_experiment(args)
