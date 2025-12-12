import numpy as np
import sys
sys.path.insert(0, 'src')
from models.feasibility import check_feasibility
from models.utilization import calc_all_utilizations

def validate_solution(assignment_matrix, tasks, servers, epsilon=0.05):
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    feasible, msg = check_feasibility(assignment_matrix, tasks, servers, epsilon)
    if not feasible:
        results['valid'] = False
        results['errors'].append(f"feasibility check failed: {msg}")

    num_tasks = len(tasks)
    num_servers = len(servers)
    if assignment_matrix.shape != (num_tasks, num_servers):
        results['valid'] = False
        results['errors'].append(f"wrong shape: expected ({num_tasks}, {num_servers}), got {assignment_matrix.shape}")

    row_sums = np.sum(assignment_matrix, axis=1)
    if not np.all(row_sums == 1):
        results['valid'] = False
        results['errors'].append("some tasks not assigned exactly once")

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)
    for i, util in enumerate(utilizations):
        if util >= 1.0:
            results['valid'] = False
            results['errors'].append(f"server {i} utilization {util:.3f} >= 1.0")
        elif util >= (1.0 - epsilon):
            results['warnings'].append(f"server {i} utilization {util:.3f} near threshold")

    load_std = np.std(utilizations)
    if load_std > 0.3:
        results['warnings'].append(f"high load imbalance: std={load_std:.3f}")

    return results


if __name__ == "__main__":
    print("testing validator module")

    from models.task import generate_random_tasks
    from models.server import generate_random_servers

    np.random.seed(42)

    num_tasks = 10
    num_servers = 3
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)

    print("\ntesting valid solution...")
    assignment = np.zeros((num_tasks, num_servers), dtype=int)
    for j in range(num_tasks):
        assignment[j, j % num_servers] = 1

    results = validate_solution(assignment, tasks, servers)
    print(f"valid: {results['valid']}")
    print(f"errors: {results['errors']}")
    print(f"warnings: {results['warnings']}")
    assert results['valid']
    print("valid solution test passed")

    print("\ntesting invalid solution (unassigned task)...")
    invalid_assignment = assignment.copy()
    invalid_assignment[0, :] = 0

    results = validate_solution(invalid_assignment, tasks, servers)
    print(f"valid: {results['valid']}")
    print(f"errors: {results['errors']}")
    assert not results['valid']
    assert len(results['errors']) > 0
    print("invalid solution test passed")

    print("\ntesting invalid solution (multiple assignments)...")
    invalid_assignment2 = assignment.copy()
    invalid_assignment2[0, :] = 1

    results = validate_solution(invalid_assignment2, tasks, servers)
    print(f"valid: {results['valid']}")
    print(f"errors: {results['errors']}")
    assert not results['valid']
    print("multiple assignments test passed")

    print("\nall tests passed")
