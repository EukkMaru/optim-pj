# Edge Computing Energy-Latency Optimization

Scheduling with Nonlinear Energy-Latency Trade-offs in Edge Computing for the Energy Sector

## Overview

This project implements optimization algorithms for task scheduling in edge computing systems, minimizing the combined objective of energy consumption and task latency. The problem is formulated as a Mixed-Integer Nonlinear Program (MINLP) with:
- Binary assignment variables (which server processes which task)
- Continuous frequency variables (CPU frequency per server)
- Nonlinear energy model (power ∝ f^γ where γ ∈ [2,3])
- Nonlinear latency model (M/M/1 queueing dynamics)

## Project Structure

```
pj/
├── src/
│   ├── models/          # Mathematical models
│   │   ├── task.py
│   │   ├── server.py
│   │   ├── network.py
│   │   ├── power.py
│   │   ├── energy.py
│   │   ├── queueing.py
│   │   ├── latency.py
│   │   ├── objective.py
│   │   └── feasibility.py
│   ├── algorithms/      # Optimization algorithms
│   │   ├── penalty.py
│   │   ├── gradient.py
│   │   ├── freq_optimizer.py
│   │   ├── neighbors.py
│   │   ├── tabu_search.py
│   │   └── pso.py
│   ├── baselines/       # Baseline algorithms
│   │   ├── greedy_latency.py
│   │   ├── greedy_energy.py
│   │   └── round_robin.py
│   ├── data/            # Data loaders
│   │   ├── google_loader.py
│   │   ├── synthetic_generator.py
│   │   ├── preprocessor.py
│   │   └── statistics.py
│   ├── evaluation/      # Evaluation framework
│   │   ├── metrics.py
│   │   ├── validator.py
│   │   └── comparison.py
│   └── visualization/   # Plotting utilities
│       └── plots.py
├── tests/               # Test suite
│   └── test_integration.py
├── data/                # Datasets
│   ├── raw/
│   └── README.md
├── results/             # Experimental results
├── main.py              # Main entry point
└── requirements.txt     # Python dependencies

```

## Installation

```bash
# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run with default synthetic workload

```bash
python main.py
```

### Run baseline algorithms only

```bash
python main.py --algorithm baseline --num_tasks 20 --num_servers 5
```

### Run Tabu Search

```bash
python main.py --algorithm tabu --num_tasks 20 --num_servers 5 --tabu_iter 100
```

### Run Particle Swarm Optimization

```bash
python main.py --algorithm pso --num_tasks 20 --num_servers 5 --pso_swarm 30 --pso_iter 100
```

### Run all algorithms for comparison

```bash
python main.py --algorithm all --num_tasks 20 --num_servers 5
```

### Use Google Cluster Trace data

```bash
python main.py --dataset google --num_tasks 20 --num_servers 10
```

## Command Line Arguments

### Problem Parameters
- `--dataset`: Dataset to use (`synthetic` or `google`, default: `synthetic`)
- `--num_tasks`: Number of tasks (default: 20)
- `--num_servers`: Number of edge servers (default: 5)
- `--seed`: Random seed (default: 42)

### Model Parameters
- `--gamma`: Power exponent γ ∈ [2,3] (default: 2.5)
- `--lambda_weight`: Latency weight in objective (default: 1.0)
- `--time_horizon`: Time horizon in seconds (default: 3600)

### Algorithm Selection
- `--algorithm`: Algorithm to run
  - `all`: Run all algorithms
  - `baseline`: Run all baseline algorithms
  - `tabu`: Tabu Search only
  - `pso`: PSO only
  - `greedy_latency`: Greedy Minimum Latency only
  - `greedy_energy`: Greedy Energy Aware only
  - `round_robin`: Round Robin only

### Tabu Search Parameters
- `--tabu_tenure`: Tabu list tenure (default: 7)
- `--tabu_iter`: Maximum iterations (default: 50)

### PSO Parameters
- `--pso_swarm`: Swarm size (default: 20)
- `--pso_iter`: Maximum iterations (default: 50)

## Testing

### Run integration tests
```bash
python tests/test_integration.py
```

### Test individual modules
```bash
# test models
python src/models/task.py
python src/models/server.py
python src/models/power.py

# test algorithms
python src/algorithms/penalty.py
python src/algorithms/tabu_search.py
python src/algorithms/pso.py

# test baselines
python src/baselines/greedy_latency.py
python src/baselines/round_robin.py
```

All modules include `if __name__ == "__main__"` test blocks.

## Algorithms

### Core Optimization Algorithms (Implemented from Scratch)

**Three main algorithms demonstrating optimization concepts:**

1. **L-BFGS-B**: Limited-memory quasi-Newton method with box constraints
   - Two-loop recursion for Hessian approximation
   - Backtracking line search with Armijo condition
   - Box constraint projection

2. **Tabu Search**: Local search with memory structure
   - Tabu list with tenure mechanism
   - Aspiration criterion
   - Neighborhood exploration for discrete assignments

3. **Particle Swarm Optimization**: Population-based metaheuristic
   - Continuous position and velocity updates
   - Social and cognitive components
   - Discretization for binary assignments

**Toy problem demonstrations** (see [`DEMO_README.md`](DEMO_README.md)):
- Static visualizations: `python3 demo_algorithms.py`
- Animated visualizations: `python3 demo_animations.py`
- Proves algorithms work correctly on standard benchmarks

### Baseline Algorithms
1. **Greedy Minimum Latency**: Assigns each task to the server with minimum network latency
2. **Greedy Energy Aware**: Assigns each task to minimize energy increase
3. **Round Robin**: Distributes tasks evenly across servers

### Algorithm Architecture

**Bi-level optimization:**
- **Outer layer**: Tabu Search or PSO for discrete task assignments
- **Inner layer**: L-BFGS-B for continuous frequency optimization
- **Constraint handling**: Penalty methods for utilization constraints

## Mathematical Model

### Objective Function
```
minimize J(x, f) = Σ E_i + λ Σ x_ij L_ij
```

### Energy Model
```
E_i = P_i * T
P_i = P_i0 + ρ_i * α_i * f_i^γ
```

### Latency Model
```
L_ij = L_net_ij + W_q_i + S_ij
W_q_i = ρ_i / (μ_i * (1 - ρ_i))  // M/M/1 queueing
S_ij = c_j / (κ_i * f_i)
```

### Constraints
- Each task assigned to exactly one server: Σ x_ij = 1 ∀j
- Server utilization: ρ_i < 1 ∀i
- Frequency bounds: f_min ≤ f_i ≤ f_max ∀i

## Results and Datasets

**For comprehensive documentation of datasets and experimental results, see [`DATASETS_AND_RESULTS.md`](DATASETS_AND_RESULTS.md)**

This includes:
- Detailed dataset descriptions (Synthetic and Google Cluster Trace)
- Experimental results across multiple configurations
- Algorithm performance comparisons
- Convergence analysis
- Key findings and insights

### Quick Results Summary

Results from synthetic datasets:

**Small (20 tasks, 5 servers):**
- Tabu Search: 49% improvement over baseline
- PSO: 45% improvement over baseline

**Medium (50 tasks, 10 servers):**
- Tabu Search: 59% improvement over baseline
- PSO: 41% improvement over baseline

The comparison framework evaluates:
- **Energy**: Total energy consumption (Joules)
- **Latency**: Mean, median, 95th/99th percentile (seconds)
- **Utilization**: Per-server utilization
- **Convergence**: Objective value over iterations

See `results_summary.csv` for tabulated results.

## Citation

```bibtex
@project{ryu2025edge,
  title={Scheduling with Nonlinear Energy-Latency Trade-offs in Edge Computing},
  author={Ryu, Sunoh},
  year={2025}
}
```

## License

This project is for academic and research purposes.

## Contact

For questions or issues, please open an issue on the project repository.
