import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective
from models.utilization import calc_all_utilizations
from algorithms.penalty import calc_total_penalty
from algorithms.lbfgsb import minimize_lbfgsb

class FrequencyOptimizer:
    def __init__(self, tasks, servers, assignment_matrix, tau, beta, gamma, lambda_weight, time_horizon, epsilon=0.05, penalty_factor=1000.0):
        self.tasks = tasks
        self.servers = servers
        self.assignment_matrix = assignment_matrix
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.lambda_weight = lambda_weight
        self.time_horizon = time_horizon
        self.epsilon = epsilon
        self.penalty_factor = penalty_factor

        self.cache_server_aggregates()

    def cache_server_aggregates(self):
        num_servers = len(self.servers)
        self.server_arrival_rates = np.zeros(num_servers)
        for i in range(num_servers):
            for j in range(len(self.tasks)):
                if self.assignment_matrix[j, i] == 1:
                    self.server_arrival_rates[i] += self.tasks[j].arrival_rate

    def objective_function(self, freqs):
        for i, server in enumerate(self.servers):
            server.set_freq(freqs[i])

        obj, energy, latency = calc_objective(self.assignment_matrix, self.tasks, self.servers, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon)

        utilizations = calc_all_utilizations(self.assignment_matrix, self.tasks, self.servers)
        penalty = calc_total_penalty(utilizations, self.epsilon, self.penalty_factor)

        return obj + penalty

    def gradient_function(self, freqs):
        eps = 1e-8
        grad = np.zeros_like(freqs)
        f0 = self.objective_function(freqs)

        for i in range(len(freqs)):
            freqs_plus = freqs.copy()
            freqs_plus[i] += eps
            f_plus = self.objective_function(freqs_plus)
            grad[i] = (f_plus - f0) / eps

        return grad

    def optimize(self):
        initial_freqs = np.array([server.freq for server in self.servers])

        bounds = [(server.freq_min, server.freq_max) for server in self.servers]

        result = minimize_lbfgsb(
            self.objective_function,
            initial_freqs,
            self.gradient_function,
            bounds,
            max_iter=100,
            tol=1e-6
        )

        if result['success']:
            for i, server in enumerate(self.servers):
                server.set_freq(result['x'][i])

        return result['x'], result['fun'], result['success']


if __name__ == "__main__":
    print("testing freq_optimizer module")

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

    optimizer = FrequencyOptimizer(tasks, servers, assignment_matrix, tau, beta, gamma, lambda_weight, time_horizon)

    print(f"\ninitial frequencies: {[s.freq for s in servers]}")

    print(f"cached server arrival rates: {optimizer.server_arrival_rates}")
    expected_rates = np.array([
        sum(tasks[j].arrival_rate for j in range(num_tasks) if assignment_matrix[j, 0] == 1),
        sum(tasks[j].arrival_rate for j in range(num_tasks) if assignment_matrix[j, 1] == 1)
    ])
    assert np.allclose(optimizer.server_arrival_rates, expected_rates)
    print("cache test passed")

    initial_obj = optimizer.objective_function([s.freq for s in servers])
    print(f"initial objective: {initial_obj:.2e}")
    assert initial_obj > 0
    print("objective function test passed")

    optimal_freqs, optimal_obj, success = optimizer.optimize()
    print(f"\noptimal frequencies: {optimal_freqs}")
    print(f"optimal objective: {optimal_obj:.2e}")
    print(f"optimization success: {success}")
    assert success
    assert optimal_obj <= initial_obj  # should improve or stay same
    print("optimize test passed")

    for i, server in enumerate(servers):
        assert server.freq >= server.freq_min
        assert server.freq <= server.freq_max
    print("bounds test passed")

    print("\noptimization with different lambda weights:")
    for lw in [0.1, 1.0, 10.0]:
        for server in servers:
            server.set_freq(server.freq_min + (server.freq_max - server.freq_min) / 2)

        opt = FrequencyOptimizer(tasks, servers, assignment_matrix, tau, beta, gamma, lw, time_horizon)
        freqs, obj, succ = opt.optimize()
        print(f"  lambda={lw}: obj={obj:.2e}, freqs={freqs}")

    print("\nall tests passed")
