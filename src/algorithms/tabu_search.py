import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective
from models.feasibility import check_feasibility
from algorithms.freq_optimizer import FrequencyOptimizer
from algorithms.neighbors import generate_random_neighbor

class TabuSearch:
    def __init__(self, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, tenure=5, max_iter=100, epsilon=0.05, temperature=1e8):
        self.tasks = tasks
        self.servers = servers
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.lambda_weight = lambda_weight
        self.time_horizon = time_horizon
        self.tenure = tenure
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.initial_temperature = temperature
        self.temperature = temperature

        self.tabu_list = []
        self.best_solution = None
        self.best_objective = float('inf')
        self.convergence_history = []
        self.iteration = 0

    def is_tabu(self, move):
        return move in self.tabu_list

    def aspiration_criterion(self, objective):
        return objective < self.best_objective

    def accept_worse_solution(self, current_obj, neighbor_obj):
        if neighbor_obj < current_obj:
            return True
        delta = neighbor_obj - current_obj
        probability = np.exp(-delta / self.temperature)
        return np.random.rand() < probability

    def add_to_tabu(self, move):
        self.tabu_list.append(move)
        if len(self.tabu_list) > self.tenure:
            self.tabu_list.pop(0)

    def diversify(self, current_assignment):
        if self.iteration > 0 and self.iteration % 50 == 0:
            num_tasks, num_servers = current_assignment.shape
            new_assignment = current_assignment.copy()
            
            num_moves = min(3, num_tasks // 3)
            for _ in range(num_moves):
                task = np.random.randint(0, num_tasks)
                current_server = np.argmax(new_assignment[task, :])
                new_server = np.random.randint(0, num_servers)
                if new_server != current_server:
                    new_assignment[task, :] = 0
                    new_assignment[task, new_server] = 1
            return new_assignment
        return current_assignment

    def optimize(self, initial_assignment):
        current_assignment = initial_assignment.copy()
        
        freq_opt = FrequencyOptimizer(self.tasks, self.servers, current_assignment, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon, self.epsilon)
        freq_opt.optimize()
        
        current_obj, current_energy, current_latency = calc_objective(current_assignment, self.tasks, self.servers, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon)

        self.best_solution = current_assignment.copy()
        self.best_objective = current_obj
        self.convergence_history.append(current_obj)

        print(f"initial objective: {current_obj:.2e}")

        for iteration in range(self.max_iter):
            self.iteration = iteration
            
            self.temperature = self.initial_temperature * (1.0 - iteration / self.max_iter)
            
            current_assignment = self.diversify(current_assignment)

            best_neighbor = None
            best_neighbor_obj = float('inf')
            best_neighbor_move = None

            for _ in range(50):
                neighbor, move = generate_random_neighbor(current_assignment)
                if neighbor is None:
                    continue

                feasible, _ = check_feasibility(neighbor, self.tasks, self.servers, self.epsilon)
                if not feasible:
                    continue

                freq_opt_neighbor = FrequencyOptimizer(self.tasks, self.servers, neighbor, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon, self.epsilon)
                freq_opt_neighbor.optimize()

                neighbor_obj, _, _ = calc_objective(neighbor, self.tasks, self.servers, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon)
                
                if neighbor_obj < best_neighbor_obj:
                    if not self.is_tabu(move) or self.aspiration_criterion(neighbor_obj) or self.accept_worse_solution(current_obj, neighbor_obj):
                        best_neighbor = neighbor
                        best_neighbor_obj = neighbor_obj
                        best_neighbor_move = move
            
            if best_neighbor is not None:
                current_assignment = best_neighbor
                current_obj = best_neighbor_obj

                self.add_to_tabu(best_neighbor_move)

                if current_obj < self.best_objective:
                    self.best_solution = current_assignment.copy()
                    self.best_objective = current_obj
                    print(f"iteration {iteration}: new best objective: {self.best_objective:.2e}")

            self.convergence_history.append(self.best_objective)
            
            if len(self.convergence_history) > 40:
                recent_improvement = abs(self.convergence_history[-1] - self.convergence_history[-40])
                if recent_improvement < 1e-8 * self.best_objective:
                    print(f"converged at iteration {iteration}")
                    break

        return self.best_solution, self.best_objective, self.convergence_history


if __name__ == "__main__":
    print("testing tabu_search module")

    from task import generate_random_tasks
    from server import generate_random_servers
    from network import generate_network_params

    np.random.seed(42)

    
    num_tasks = 5
    num_servers = 2
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)
    tau, beta = generate_network_params(num_tasks, num_servers)

    
    initial_assignment = np.zeros((num_tasks, num_servers), dtype=int)
    for j in range(num_tasks):
        initial_assignment[j, j % num_servers] = 1

    print(f"initial assignment:\n{initial_assignment}")

    
    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    
    tabu_search = TabuSearch(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, tenure=5, max_iter=10)

    
    print("\ntesting tabu list:")
    test_move = ('move', 0, 1, 2)
    assert not tabu_search.is_tabu(test_move)
    tabu_search.add_to_tabu(test_move)
    assert tabu_search.is_tabu(test_move)
    print("tabu list test passed")

    
    print("\ntesting aspiration criterion:")
    tabu_search.best_objective = 100.0
    assert tabu_search.aspiration_criterion(50.0)
    assert not tabu_search.aspiration_criterion(150.0)
    print("aspiration criterion test passed")

    
    print("\nrunning tabu search optimization:")
    best_solution, best_obj, history = tabu_search.optimize(initial_assignment)

    print(f"\nbest solution:\n{best_solution}")
    print(f"best objective: {best_obj:.2e}")
    print(f"convergence history length: {len(history)}")
    print(f"improvement: {(history[0] - history[-1]) / history[0] * 100:.2f}%")

    assert best_solution is not None
    assert best_obj > 0
    assert len(history) > 0
    print("\noptimization test passed")

    
    feasible, msg = check_feasibility(best_solution, tasks, servers)
    print(f"best solution feasible: {feasible}, msg: {msg}")

    print("\nall tests passed")
