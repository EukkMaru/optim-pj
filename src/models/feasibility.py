import numpy as np
from .utilization import calc_all_utilizations

def check_feasibility(assignment_matrix, tasks, servers, epsilon=0.05):

    num_tasks = len(tasks)
    num_servers = len(servers)

    for j in range(num_tasks):
        if np.sum(assignment_matrix[j, :]) != 1:
            return False, f"task {j} not assigned to exactly one server"

    utilizations = calc_all_utilizations(assignment_matrix, tasks, servers)
    for i in range(num_servers):
        if utilizations[i] >= (1.0 - epsilon):
            return False, f"server {i} utilization {utilizations[i]:.3f} >= {1.0-epsilon:.3f}"

    for i, server in enumerate(servers):
        if server.freq < server.freq_min or server.freq > server.freq_max:
            return False, f"server {i} freq {server.freq} out of bounds [{server.freq_min}, {server.freq_max}]"

    return True, "feasible"


if __name__ == "__main__":
    print("testing feasibility module")

    from task import generate_random_tasks
    from server import generate_random_servers

    np.random.seed(42)

    num_tasks = 5
    num_servers = 2
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)

    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)
    for j in range(num_tasks):
        assignment_matrix[j, j % num_servers] = 1

    feasible, msg = check_feasibility(assignment_matrix, tasks, servers)
    print(f"feasible assignment: {feasible}, msg: {msg}")
    assert feasible
    print("feasible assignment test passed")

    assignment_bad1 = assignment_matrix.copy()
    assignment_bad1[0, :] = 0
    feasible, msg = check_feasibility(assignment_bad1, tasks, servers)
    print(f"task not assigned: {feasible}, msg: {msg}")
    assert not feasible
    print("task not assigned test passed")

    assignment_bad2 = assignment_matrix.copy()
    assignment_bad2[0, :] = 1
    feasible, msg = check_feasibility(assignment_bad2, tasks, servers)
    print(f"task assigned to multiple servers: {feasible}, msg: {msg}")
    assert not feasible
    print("multiple assignment test passed")

    servers[0].freq = 5e9  # exceeds max
    feasible, msg = check_feasibility(assignment_matrix, tasks, servers)
    print(f"freq out of bounds: {feasible}, msg: {msg}")
    assert not feasible
    servers[0].freq = 2e9  # reset
    print("freq out of bounds test passed")

    assignment_bad3 = np.zeros((num_tasks, num_servers), dtype=int)
    assignment_bad3[:, 0] = 1
    servers[0].set_freq(1e9)  # low frequency to increase utilization
    feasible, msg = check_feasibility(assignment_bad3, tasks, servers, epsilon=0.05)
    print(f"high utilization: {feasible}, msg: {msg}")

    print("\nall tests passed")
