#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dx1 = 200 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

def sphere(x):
    return np.sum(x**2)

def assignment_cost(assignment, cost_matrix):
    return np.sum(assignment * cost_matrix)

def demo_lbfgsb():
    print("="*60)
    print("demo: l-bfgs-b on rosenbrock function")
    print("="*60)
    print("\nproblem: minimize f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print("global minimum: (1, 1) with f = 0")

    from algorithms.lbfgsb import LBFGSB

    x0 = np.array([-1.2, 1.0])
    bounds = np.array([[-2, 2], [-2, 2]])

    print(f"\ninitial point: ({x0[0]:.3f}, {x0[1]:.3f})")
    print(f"initial value: f = {rosenbrock(x0):.6e}")

    history = {'iteration': [], 'x': [], 'y': [], 'f_val': []}

    def track_func(x):
        f = rosenbrock(x)
        history['x'].append(x[0])
        history['y'].append(x[1])
        history['f_val'].append(f)
        return f

    def track_grad(x):
        return rosenbrock_grad(x)

    optimizer = LBFGSB(track_func, track_grad, bounds, x0, m=5, max_iter=50, tol=1e-8)
    x_final, f_final, success = optimizer.optimize()

    history['iteration'] = list(range(len(history['f_val'])))

    print(f"\nconvergence:")
    print("-" * 60)
    for i in range(0, len(history['iteration']), max(1, len(history['iteration'])//10)):
        it = history['iteration'][i]
        x_val = history['x'][i]
        y_val = history['y'][i]
        f_val = history['f_val'][i]
        print(f"  iter {it:3d}: x=({x_val:7.4f}, {y_val:7.4f})  f={f_val:.6e}")

    final_idx = len(history['iteration']) - 1
    print(f"\nfinal solution:")
    print(f"  x = ({history['x'][final_idx]:.6f}, {history['y'][final_idx]:.6f})")
    print(f"  f = {history['f_val'][final_idx]:.6e}")
    print(f"  iterations = {len(history['iteration'])}")
    print(f"  success = {success}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['iteration'], history['f_val'], 'b-', linewidth=2)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('objective value')
    ax1.set_title('l-bfgs-b convergence')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    ax2.plot(history['x'], history['y'], 'r.-', linewidth=1.5, markersize=4, label='trajectory')
    ax2.plot(history['x'][0], history['y'][0], 'go', markersize=8, label='start')
    ax2.plot(history['x'][-1], history['y'][-1], 'r*', markersize=12, label='end')
    ax2.plot(1, 1, 'bs', markersize=10, label='optimum')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('search trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_lbfgsb.png', dpi=150)
    print(f"\nvisualization saved to demo_lbfgsb.png")
    plt.close()

    return history

def demo_pso():
    print("="*60)
    print("demo: particle swarm optimization on sphere function")
    print("="*60)
    print("\nproblem: minimize f(x,y) = x^2 + y^2")
    print("global minimum: (0, 0) with f = 0")

    dim = 2
    swarm_size = 20
    max_iter = 50
    bounds = np.array([[-5, 5]] * dim)

    np.random.seed(42)
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (swarm_size, dim))
    velocities = np.random.uniform(-0.5, 0.5, (swarm_size, dim))
    pbest_positions = positions.copy()
    pbest_scores = np.array([sphere(p) for p in positions])
    gbest_position = pbest_positions[np.argmin(pbest_scores)].copy()
    gbest_score = np.min(pbest_scores)

    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social

    print(f"\nswarm size: {swarm_size}")
    print(f"dimensions: {dim}")
    print(f"initial best: f = {gbest_score:.6e}")

    history = {
        'iteration': [],
        'gbest_score': [],
        'mean_score': [],
        'positions': [],
        'gbest_position': []
    }

    print(f"\nconvergence:")
    print("-" * 60)

    for iteration in range(max_iter):
        scores = np.array([sphere(p) for p in positions])

        better_mask = scores < pbest_scores
        pbest_scores[better_mask] = scores[better_mask]
        pbest_positions[better_mask] = positions[better_mask]

        min_idx = np.argmin(pbest_scores)
        if pbest_scores[min_idx] < gbest_score:
            gbest_score = pbest_scores[min_idx]
            gbest_position = pbest_positions[min_idx].copy()

        if iteration % 5 == 0 or iteration == max_iter - 1:
            history['iteration'].append(iteration)
            history['gbest_score'].append(gbest_score)
            history['mean_score'].append(np.mean(scores))
            history['positions'].append(positions.copy())
            history['gbest_position'].append(gbest_position.copy())

        if iteration % 5 == 0 or iteration == max_iter - 1:
            print(f"  iter {iteration:3d}: gbest={gbest_score:.6e}  mean={np.mean(scores):.6e}")

        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities +
                     c1 * r1 * (pbest_positions - positions) +
                     c2 * r2 * (gbest_position - positions))

        max_velocity = 1.0
        velocities = np.clip(velocities, -max_velocity, max_velocity)

        positions = positions + velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

    print(f"\nfinal solution:")
    print(f"  x = ({gbest_position[0]:.6f}, {gbest_position[1]:.6f})")
    print(f"  f = {gbest_score:.6e}")
    print(f"  iterations = {max_iter}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['iteration'], history['gbest_score'], 'b-', linewidth=2, label='global best')
    ax1.plot(history['iteration'], history['mean_score'], 'r--', linewidth=1.5, label='swarm mean')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('objective value')
    ax1.set_title('pso convergence on sphere function')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2

    ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

    num_display = 5
    for particle_idx in range(min(num_display, swarm_size)):
        trajectory_x = [pos[particle_idx, 0] for pos in history['positions']]
        trajectory_y = [pos[particle_idx, 1] for pos in history['positions']]
        ax2.plot(trajectory_x, trajectory_y, 'o-', alpha=0.4, markersize=3, linewidth=0.8)

    final_positions = history['positions'][-1]
    ax2.scatter(final_positions[:, 0], final_positions[:, 1],
               c='red', s=30, alpha=0.6, marker='o', label='final swarm')

    ax2.plot(gbest_position[0], gbest_position[1], 'g*', markersize=15, label='global best')
    ax2.plot(0, 0, 'bs', markersize=10, label='optimum')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('swarm movement on sphere function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_pso.png', dpi=150)
    print(f"\nvisualization saved to demo_pso.png")
    plt.close()

    return history

def demo_tabu():
    print("="*60)
    print("demo: tabu search on assignment problem")
    print("="*60)
    print("\nproblem: assign 8 tasks to 4 servers minimizing cost")
    print("binary assignment matrix x[i,j] = 1 if task i -> server j")

    np.random.seed(42)
    num_tasks = 8
    num_servers = 4

    cost_matrix = np.random.uniform(10, 100, (num_tasks, num_servers))

    for i in range(num_tasks):
        good_server = i % num_servers
        cost_matrix[i, good_server] = np.random.uniform(1, 10)

    print(f"\ntasks: {num_tasks}")
    print(f"servers: {num_servers}")
    print(f"\ncost matrix:")
    for i in range(num_tasks):
        print(f"  task {i}: [{', '.join([f'{c:5.1f}' for c in cost_matrix[i]])}]")

    assignment = np.zeros((num_tasks, num_servers), dtype=int)
    for i in range(num_tasks):
        j = np.random.randint(num_servers)
        assignment[i, j] = 1

    current_cost = assignment_cost(assignment, cost_matrix)
    best_assignment = assignment.copy()
    best_cost = current_cost

    print(f"\ninitial assignment cost: {current_cost:.2f}")

    max_iter = 100
    tenure = 7
    tabu_list = []

    history = {
        'iteration': [],
        'current_cost': [],
        'best_cost': [],
        'assignments': []
    }

    print(f"\nconvergence:")
    print("-" * 60)

    for iteration in range(max_iter):
        history['iteration'].append(iteration)
        history['current_cost'].append(current_cost)
        history['best_cost'].append(best_cost)
        if iteration % 10 == 0 or iteration == max_iter - 1:
            history['assignments'].append(assignment.copy())

        if iteration % 10 == 0 or iteration == max_iter - 1:
            print(f"  iter {iteration:3d}: current={current_cost:.2f}  best={best_cost:.2f}")

        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None

        for task_idx in range(num_tasks):
            current_server = np.argmax(assignment[task_idx])

            for new_server in range(num_servers):
                if new_server == current_server:
                    continue

                move = (task_idx, current_server, new_server)

                neighbor = assignment.copy()
                neighbor[task_idx, current_server] = 0
                neighbor[task_idx, new_server] = 1
                neighbor_cost = assignment_cost(neighbor, cost_matrix)

                is_tabu = move in tabu_list
                aspiration = neighbor_cost < best_cost

                if(not is_tabu or aspiration):
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor_cost
                        best_move = move

        if best_neighbor is None:
            break

        assignment = best_neighbor
        current_cost = best_neighbor_cost

        tabu_list.append(best_move)
        if len(tabu_list) > tenure:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()

    print(f"\nfinal solution:")
    print(f"  best cost = {best_cost:.2f}")
    print(f"  iterations = {len(history['iteration'])}")
    print(f"\nbest assignment:")
    for i in range(num_tasks):
        server = np.argmax(best_assignment[i])
        print(f"  task {i} -> server {server} (cost: {cost_matrix[i, server]:.2f})")

    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.plot(history['iteration'], history['current_cost'], 'g-', linewidth=1, alpha=0.6, label='current solution')
    ax1.plot(history['iteration'], history['best_cost'], 'b-', linewidth=2, label='best found')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('assignment cost')
    ax1.set_title('tabu search convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    im = ax2.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('server')
    ax2.set_ylabel('task')
    ax2.set_title('cost matrix')
    ax2.set_xticks(range(num_servers))
    ax2.set_yticks(range(num_tasks))

    for i in range(num_tasks):
        j = np.argmax(best_assignment[i])
        ax2.plot(j, i, 'go', markersize=10, markerfacecolor='none', markeredgewidth=2)

    plt.colorbar(im, ax=ax2, label='cost')

    assignment_iter = [0, len(history['assignments'])//2, len(history['assignments'])-1]
    assignment_labels = ['initial', 'middle', 'final']

    for idx, (iter_idx, label) in enumerate(zip(assignment_iter, assignment_labels)):
        if iter_idx < len(history['assignments']):
            assn = history['assignments'][iter_idx]
            task_to_server = [np.argmax(assn[i]) for i in range(num_tasks)]
            y_pos = idx * (num_tasks + 1)
            for task_idx, server_idx in enumerate(task_to_server):
                color = plt.cm.tab10(server_idx / num_servers)
                ax3.barh(y_pos - task_idx, 1, left=server_idx, height=0.8, color=color)
                ax3.text(server_idx + 0.5, y_pos - task_idx, f'T{task_idx}',
                        ha='center', va='center', fontsize=7)

    ax3.set_xlabel('server')
    ax3.set_ylabel('snapshot')
    ax3.set_title('assignment evolution')
    ax3.set_yticks([0, -(num_tasks+1), -2*(num_tasks+1)])
    ax3.set_yticklabels(assignment_labels)
    ax3.set_xlim(-0.5, num_servers)
    ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('demo_tabu.png', dpi=150)
    print(f"\nvisualization saved to demo_tabu.png")
    plt.close()

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demonstration of optimization algorithms")
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['all', 'lbfgsb', 'pso', 'tabu'],
                       help='algorithm to demonstrate')

    args = parser.parse_args()

    print("\noptimization algorithm demonstrations")
    print("toy problems showing convergence behavior\n")

    if args.algorithm in ['all', 'lbfgsb']:
        demo_lbfgsb()
        print("\n")

    if args.algorithm in ['all', 'pso']:
        demo_pso()
        print("\n")

    if args.algorithm in ['all', 'tabu']:
        demo_tabu()
        print("\n")

    print("demonstrations completed")
