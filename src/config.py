import numpy as np

class Config:
    def __init__(self):
        self.num_tasks = 10
        self.num_servers = 3

        self.gamma = 2.5  # power exponent (typically 2-3)

        self.time_horizon = 3600.0  # seconds

        self.lambda_weight = 1.0  # trade-off weight for latency
        self.penalty_factor = 1000.0  # penalty for infeasibility
        self.epsilon = 0.05  # safety margin for utilization

        self.tabu_tenure = 7
        self.tabu_max_iter = 100
        self.pso_swarm_size = 30
        self.pso_max_iter = 100
        self.pso_inertia = 0.7
        self.pso_cognitive = 1.5
        self.pso_social = 1.5

        self.random_seed = 42

    def set_problem_size(self, num_tasks, num_servers):
        self.num_tasks = num_tasks
        self.num_servers = num_servers

    def set_lambda_weight(self, lambda_weight):
        self.lambda_weight = lambda_weight

    def validate(self):
        assert self.num_tasks > 0, "num_tasks must be positive"
        assert self.num_servers > 0, "num_servers must be positive"
        assert self.gamma >= 2.0 and self.gamma <= 3.0, "gamma must be in [2, 3]"
        assert self.time_horizon > 0, "time_horizon must be positive"
        assert self.lambda_weight >= 0, "lambda_weight must be non-negative"
        assert self.epsilon > 0 and self.epsilon < 1, "epsilon must be in (0, 1)"
        return True


if __name__ == "__main__":
    print("testing config module")

    config = Config()
    print(f"default config: tasks={config.num_tasks}, servers={config.num_servers}, gamma={config.gamma}")

    try:
        config.validate()
        print("validation passed")
    except AssertionError as e:
        print(f"validation failed: {e}")

    config.set_problem_size(20, 5)
    print(f"updated problem size: tasks={config.num_tasks}, servers={config.num_servers}")

    config.set_lambda_weight(2.0)
    print(f"updated lambda_weight: {config.lambda_weight}")

    config.gamma = 5.0
    try:
        config.validate()
        print("validation should have failed")
    except AssertionError as e:
        print(f"expected failure: {e}")

    print("all tests passed")
