import numpy as np
import sys
sys.path.insert(0, 'src')
from models.objective import calc_objective
from models.feasibility import check_feasibility
from algorithms.freq_optimizer import FrequencyOptimizer

class Particle:
    def __init__(self, num_tasks, num_servers):
        self.num_tasks = num_tasks
        self.num_servers = num_servers
        self.position = np.random.rand(num_tasks, num_servers)
        self.velocity = np.random.rand(num_tasks, num_servers) * 0.1
        self.pbest_position = self.position.copy()
        self.pbest_objective = float('inf')

    def get_discrete_assignment(self):
        assignment = np.zeros((self.num_tasks, self.num_servers), dtype=int)
        for j in range(self.num_tasks):
            server = np.argmax(self.position[j, :])
            assignment[j, server] = 1
        return assignment

    def update_velocity(self, gbest_position, inertia, cognitive, social):
        r1 = np.random.rand(self.num_tasks, self.num_servers)
        r2 = np.random.rand(self.num_tasks, self.num_servers)

        self.velocity = (inertia * self.velocity +
                        cognitive * r1 * (self.pbest_position - self.position) +
                        social * r2 * (gbest_position - self.position))

        max_velocity = 0.5
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def update_position(self):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, 0.0, 1.0)

    def update_pbest(self, objective):
        if objective < self.pbest_objective:
            self.pbest_objective = objective
            self.pbest_position = self.position.copy()


class PSO:
    def __init__(self, tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, swarm_size=20, max_iter=100, inertia=0.9, cognitive=2.0, social=2.0, epsilon=0.05):
        self.tasks = tasks
        self.servers = servers
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.lambda_weight = lambda_weight
        self.time_horizon = time_horizon
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.epsilon = epsilon

        self.num_tasks = len(tasks)
        self.num_servers = len(servers)

        self.particles = [Particle(self.num_tasks, self.num_servers) for _ in range(swarm_size)]

        self.gbest_position = None
        self.gbest_assignment = None
        self.gbest_objective = float('inf')

        self.convergence_history = []

    def evaluate_particle(self, particle):
        assignment = particle.get_discrete_assignment()

        feasible, _ = check_feasibility(assignment, self.tasks, self.servers, self.epsilon)
        if not feasible:
            return float('inf'), assignment

        freq_opt = FrequencyOptimizer(self.tasks, self.servers, assignment, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon, self.epsilon)
        freq_opt.optimize()

        objective, _, _ = calc_objective(assignment, self.tasks, self.servers, self.tau, self.beta, self.gamma, self.lambda_weight, self.time_horizon)

        return objective, assignment

    def seed_particle_with_assignment(self, particle, assignment):
        for j in range(self.num_tasks):
            server = np.argmax(assignment[j, :])
            particle.position[j, :] = np.random.rand(self.num_servers) * 0.3
            particle.position[j, server] = 0.7 + np.random.rand() * 0.3

    def optimize(self, initial_assignment=None):
        print("initializing swarm...")

        if initial_assignment is not None:
            self.seed_particle_with_assignment(self.particles[0], initial_assignment)

        for i, particle in enumerate(self.particles):
            objective, assignment = self.evaluate_particle(particle)
            particle.update_pbest(objective)

            if objective < self.gbest_objective:
                self.gbest_objective = objective
                self.gbest_position = particle.position.copy()
                self.gbest_assignment = assignment.copy()

        self.convergence_history.append(self.gbest_objective)
        print(f"initial gbest: {self.gbest_objective:.2e}")

        for iteration in range(self.max_iter):
            for particle in self.particles:
                particle.update_velocity(self.gbest_position, self.inertia, self.cognitive, self.social)
                particle.update_position()

                objective, assignment = self.evaluate_particle(particle)
                particle.update_pbest(objective)

                if objective < self.gbest_objective:
                    self.gbest_objective = objective
                    self.gbest_position = particle.position.copy()
                    self.gbest_assignment = assignment.copy()
                    print(f"iteration {iteration}: new gbest: {self.gbest_objective:.2e}")

            self.convergence_history.append(self.gbest_objective)

            if len(self.convergence_history) > 30:
                recent_improvement = abs(self.convergence_history[-1] - self.convergence_history[-30])
                if recent_improvement < 1e-8 * self.gbest_objective:
                    print(f"converged at iteration {iteration}")
                    break

        return self.gbest_assignment, self.gbest_objective, self.convergence_history


if __name__ == "__main__":
    print("testing pso module")

    from task import generate_random_tasks
    from server import generate_random_servers
    from network import generate_network_params

    np.random.seed(42)

    print("\ntesting particle class:")
    num_tasks = 5
    num_servers = 2
    particle = Particle(num_tasks, num_servers)

    print(f"position shape: {particle.position.shape}")
    print(f"velocity shape: {particle.velocity.shape}")
    assert particle.position.shape == (num_tasks, num_servers)
    assert particle.velocity.shape == (num_tasks, num_servers)
    print("particle initialization test passed")

    assignment = particle.get_discrete_assignment()
    print(f"discrete assignment shape: {assignment.shape}")
    print(f"assignment:\n{assignment}")
    assert assignment.shape == (num_tasks, num_servers)
    assert np.all(np.sum(assignment, axis=1) == 1)
    print("discretization test passed")

    gbest_position = np.random.rand(num_tasks, num_servers)
    particle.update_velocity(gbest_position, 0.7, 1.5, 1.5)
    print("velocity update test passed")

    particle.update_position()
    assert np.all(particle.position >= 0.0) and np.all(particle.position <= 1.0)
    print("position update test passed")

    particle.update_pbest(100.0)
    assert particle.pbest_objective == 100.0
    particle.update_pbest(50.0)
    assert particle.pbest_objective == 50.0
    particle.update_pbest(75.0)
    assert particle.pbest_objective == 50.0
    print("pbest update test passed")

    print("\ntesting PSO class:")
    tasks = generate_random_tasks(num_tasks)
    servers = generate_random_servers(num_servers)
    tau, beta = generate_network_params(num_tasks, num_servers)

    gamma = 2.5
    lambda_weight = 1.0
    time_horizon = 3600.0

    pso = PSO(tasks, servers, tau, beta, gamma, lambda_weight, time_horizon, swarm_size=5, max_iter=3)

    print(f"swarm size: {len(pso.particles)}")
    assert len(pso.particles) == 5
    print("PSO initialization test passed")

    print("\nrunning PSO optimization:")
    best_solution, best_obj, history = pso.optimize()

    print(f"\nbest solution:\n{best_solution}")
    print(f"best objective: {best_obj:.2e}")
    print(f"convergence history: {[f'{h:.2e}' for h in history]}")

    assert best_solution is not None
    assert best_obj > 0
    assert len(history) > 0
    print("optimization test passed")

    feasible, msg = check_feasibility(best_solution, tasks, servers)
    print(f"best solution feasible: {feasible}, msg: {msg}")

    print("\nall tests passed")
