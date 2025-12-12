import numpy as np

def calc_utilization(assignment, tasks, server):
    total_arrival_rate = sum(tasks[j].arrival_rate for j in range(len(tasks)) if assignment[j])
    capacity = server.get_capacity()
    if capacity == 0:
        return float('inf')
    return total_arrival_rate / capacity

def calc_all_utilizations(assignment_matrix, tasks, servers):
    num_servers = len(servers)
    utilizations = np.zeros(num_servers)
    for i in range(num_servers):
        assignment = assignment_matrix[:, i]
        utilizations[i] = calc_utilization(assignment, tasks, servers[i])
    return utilizations


if __name__ == "__main__":
    print("testing utilization module")

    from task import Task
    from server import EdgeServer

    tasks = [
        Task(0, 2.0, 1e8, 1e4),
        Task(1, 3.0, 1e8, 1e4),
        Task(2, 1.5, 1e8, 1e4)
    ]

    server = EdgeServer(0, 1e9, 3e9, 1.0, 5e-10, 10.0)
    server.set_freq(2e9)

    assignment = np.array([1, 1, 0])  # tasks 0 and 1 assigned to server
    util = calc_utilization(assignment, tasks, server)
    expected = (2.0 + 3.0) / (1.0 * 2e9)
    print(f"utilization: {util:.6e}, expected: {expected:.6e}")
    assert np.isclose(util, expected)
    print("calc_utilization test passed")

    assignment_all = np.array([1, 1, 1])
    util_all = calc_utilization(assignment_all, tasks, server)
    expected_all = (2.0 + 3.0 + 1.5) / (1.0 * 2e9)
    print(f"utilization (all tasks): {util_all:.6e}, expected: {expected_all:.6e}")
    assert np.isclose(util_all, expected_all)
    print("calc_utilization with all tasks passed")

    servers = [
        EdgeServer(0, 1e9, 3e9, 1.0, 5e-10, 10.0),
        EdgeServer(1, 1e9, 3e9, 1.5, 3e-10, 12.0)
    ]
    servers[0].set_freq(2e9)
    servers[1].set_freq(1.5e9)

    assignment_matrix = np.array([
        [1, 0],
        [0, 1],
        [0, 1]
    ])

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)
    expected_utils = np.array([
        2.0 / (1.0 * 2e9),
        (3.0 + 1.5) / (1.5 * 1.5e9)
    ])
    print(f"utilizations: {utilizations}")
    print(f"expected: {expected_utils}")
    assert np.allclose(utilizations, expected_utils)
    print("calc_all_utilizations test passed")

    print("all tests passed")
