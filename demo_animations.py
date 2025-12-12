#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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

def animate_lbfgsb():
    print("generating l-bfgs-b animation...")
    print("this will show gradient direction and line search in action")

    from algorithms.lbfgsb import LBFGSB

    x0 = np.array([-1.2, 1.0])
    bounds = np.array([[-2, 2], [-2, 2]])

    history = {
        'x': [],
        'f': [],
        'gradient': [],
        'direction': []
    }

    def track_func(x):
        f = rosenbrock(x)
        history['x'].append(x.copy())
        history['f'].append(f)
        return f

    def track_grad(x):
        g = rosenbrock_grad(x)
        history['gradient'].append(g.copy())
        return g

    optimizer = LBFGSB(track_func, track_grad, bounds, x0, m=5, max_iter=50, tol=1e-8)
    x_final, f_final, success = optimizer.optimize()

    for i in range(len(history['x']) - 1):
        d = history['x'][i+1] - history['x'][i]
        history['direction'].append(d)
    history['direction'].append(history['direction'][-1])

    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    fig = plt.figure(figsize=(14, 5))
    ax_traj = fig.add_subplot(1, 2, 1)
    ax_conv = fig.add_subplot(1, 2, 2)

    contour = ax_traj.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 25), cmap='viridis', alpha=0.4)
    ax_traj.plot(1, 1, 'r*', markersize=15, label='optimum', zorder=10)

    point, = ax_traj.plot([], [], 'bo', markersize=10, label='current position', zorder=20)
    trail, = ax_traj.plot([], [], 'g-', linewidth=1, alpha=0.5, zorder=5)

    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')
    ax_traj.set_title('search trajectory')
    ax_traj.legend(loc='upper left')
    ax_traj.set_xlim(-2, 2)
    ax_traj.set_ylim(-1, 3)

    conv_line, = ax_conv.plot([], [], 'b-', linewidth=2)
    ax_conv.set_xlabel('iteration')
    ax_conv.set_ylabel('objective value')
    ax_conv.set_title('convergence')
    ax_conv.set_xlim(0, len(history['x']))
    ax_conv.set_ylim(min(history['f']) * 0.5, max(history['f']) * 1.1)
    ax_conv.set_yscale('log')
    ax_conv.grid(True, alpha=0.3)

    trail_x, trail_y = [], []
    conv_iters, conv_vals = [], []
    arrows = []

    def init():
        point.set_data([], [])
        trail.set_data([], [])
        conv_line.set_data([], [])
        return point, trail, conv_line

    def animate(frame):
        nonlocal arrows

        for arrow in arrows:
            arrow.remove()
        arrows = []

        if frame >= len(history['x']):
            return point, trail, conv_line

        x_curr = history['x'][frame]
        f_curr = history['f'][frame]

        point.set_data([x_curr[0]], [x_curr[1]])

        trail_x.append(x_curr[0])
        trail_y.append(x_curr[1])
        trail.set_data(trail_x, trail_y)

        conv_iters.append(frame)
        conv_vals.append(f_curr)
        conv_line.set_data(conv_iters, conv_vals)

        if frame < len(history['gradient']):
            grad = history['gradient'][frame]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-6:
                grad_scaled = -grad / grad_norm * 0.3
                gradient_arrow = ax_traj.arrow(x_curr[0], x_curr[1],
                                         grad_scaled[0], grad_scaled[1],
                                         head_width=0.08, head_length=0.06,
                                         fc='red', ec='red', linewidth=2,
                                         alpha=0.7, zorder=15)
                arrows.append(gradient_arrow)

        if frame < len(history['direction']):
            direction = history['direction'][frame]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-6:
                dir_scaled = direction / max(dir_norm, 0.3) * 0.3
                direction_arrow = ax_traj.arrow(x_curr[0], x_curr[1],
                                          dir_scaled[0], dir_scaled[1],
                                          head_width=0.08, head_length=0.06,
                                          fc='blue', ec='blue', linewidth=2,
                                          alpha=0.7, zorder=15)
                arrows.append(direction_arrow)

        fig.suptitle(f'l-bfgs-b: iteration {frame}/{len(history["x"])-1} | f={f_curr:.2e}', fontsize=12)

        return point, trail, conv_line

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(history['x']), interval=200, blit=False)

    writer = PillowWriter(fps=5)
    anim.save('demo_lbfgsb_animated.gif', writer=writer, dpi=100)
    plt.close()

    print(f"saved demo_lbfgsb_animated.gif ({len(history['x'])} frames)")

def animate_pso():
    print("generating pso animation...")
    print("this will show swarm dynamics and velocity vectors")

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

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    history = {
        'positions': [],
        'velocities': [],
        'gbest_position': [],
        'gbest_score': []
    }

    for iteration in range(max_iter):
        history['positions'].append(positions.copy())
        history['velocities'].append(velocities.copy())
        history['gbest_position'].append(gbest_position.copy())
        history['gbest_score'].append(gbest_score)

        scores = np.array([sphere(p) for p in positions])
        better_mask = scores < pbest_scores
        pbest_scores[better_mask] = scores[better_mask]
        pbest_positions[better_mask] = positions[better_mask]

        min_idx = np.argmin(pbest_scores)
        if pbest_scores[min_idx] < gbest_score:
            gbest_score = pbest_scores[min_idx]
            gbest_position = pbest_positions[min_idx].copy()

        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities +
                     c1 * r1 * (pbest_positions - positions) +
                     c2 * r2 * (gbest_position - positions))
        velocities = np.clip(velocities, -1.0, 1.0)
        positions = positions + velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2

    fig, ax = plt.subplots(figsize=(8, 8))

    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)

    ax.plot(0, 0, 'r*', markersize=20, label='optimum', zorder=30)

    particles = ax.scatter([], [], c='blue', s=50, alpha=0.6, zorder=20)
    gbest_marker, = ax.plot([], [], 'go', markersize=15,
                            markerfacecolor='yellow', markeredgewidth=2,
                            label='global best', zorder=25)

    arrows = []

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    def init():
        particles.set_offsets(np.empty((0, 2)))
        gbest_marker.set_data([], [])
        return particles, gbest_marker

    def animate(frame):
        nonlocal arrows

        for arrow in arrows:
            arrow.remove()
        arrows = []

        if frame >= len(history['positions']):
            return particles, gbest_marker

        pos = history['positions'][frame]
        vel = history['velocities'][frame]
        gbest = history['gbest_position'][frame]
        gbest_score = history['gbest_score'][frame]

        particles.set_offsets(pos)

        gbest_marker.set_data([gbest[0]], [gbest[1]])

        for i in range(0, swarm_size, 3):
            vel_norm = np.linalg.norm(vel[i])
            if vel_norm > 0.01:
                scale = 0.5  # scale factor for visibility
                arrow = ax.arrow(pos[i, 0], pos[i, 1],
                               vel[i, 0] * scale, vel[i, 1] * scale,
                               head_width=0.2, head_length=0.15,
                               fc='red', ec='red', alpha=0.5,
                               linewidth=1, zorder=15)
                arrows.append(arrow)

        ax.set_title(f'pso: iteration {frame}/{max_iter-1} | gbest={gbest_score:.4f}')

        return particles, gbest_marker

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(history['positions']), interval=100, blit=False)

    writer = PillowWriter(fps=10)
    anim.save('demo_pso_animated.gif', writer=writer, dpi=100)
    plt.close()

    print(f"saved demo_pso_animated.gif ({len(history['positions'])} frames)")

def animate_tabu():
    print("generating tabu search animation...")
    print("this will show assignment changes and tabu list dynamics")

    np.random.seed(42)
    num_tasks = 8
    num_servers = 4

    cost_matrix = np.random.uniform(10, 100, (num_tasks, num_servers))
    for i in range(num_tasks):
        good_server = i % num_servers
        cost_matrix[i, good_server] = np.random.uniform(1, 10)

    assignment = np.zeros((num_tasks, num_servers), dtype=int)
    for i in range(num_tasks):
        j = np.random.randint(num_servers)
        assignment[i, j] = 1

    current_cost = assignment_cost(assignment, cost_matrix)
    best_assignment = assignment.copy()
    best_cost = current_cost

    max_iter = 50  # shorter for animation
    tenure = 7
    tabu_list = []

    history = {
        'assignments': [],
        'current_cost': [],
        'best_cost': [],
        'tabu_list': [],
        'last_move': []
    }

    for iteration in range(max_iter):
        history['assignments'].append(assignment.copy())
        history['current_cost'].append(current_cost)
        history['best_cost'].append(best_cost)
        history['tabu_list'].append(list(tabu_list))

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
            history['last_move'].append(None)
            break

        history['last_move'].append(best_move)
        assignment = best_neighbor
        current_cost = best_neighbor_cost

        tabu_list.append(best_move)
        if len(tabu_list) > tenure:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

    ax_assign = fig.add_subplot(gs[0])
    ax_cost = fig.add_subplot(gs[1])
    ax_tabu = fig.add_subplot(gs[2])

    im = ax_assign.imshow(cost_matrix, cmap='YlOrRd', aspect='auto', alpha=0.3)
    ax_assign.set_xlabel('server')
    ax_assign.set_ylabel('task')
    ax_assign.set_title('current assignment')
    ax_assign.set_xticks(range(num_servers))
    ax_assign.set_yticks(range(num_tasks))

    ax_cost.set_xlabel('iteration')
    ax_cost.set_ylabel('cost')
    ax_cost.set_title('convergence')
    ax_cost.set_xlim(0, max_iter)
    ax_cost.set_ylim(0, max(history['current_cost']))
    ax_cost.grid(True, alpha=0.3)

    ax_tabu.set_xlim(-0.5, tenure + 0.5)
    ax_tabu.set_ylim(-0.5, num_tasks - 0.5)
    ax_tabu.set_xlabel('tabu slot')
    ax_tabu.set_ylabel('task')
    ax_tabu.set_title('tabu list')
    ax_tabu.grid(True, alpha=0.3)

    assignment_markers = []
    cost_current_line, = ax_cost.plot([], [], 'g-', linewidth=2, label='current')
    cost_best_line, = ax_cost.plot([], [], 'b-', linewidth=2, label='best')
    tabu_markers = []

    ax_cost.legend()

    def init():
        cost_current_line.set_data([], [])
        cost_best_line.set_data([], [])
        return cost_current_line, cost_best_line

    def animate(frame):
        nonlocal assignment_markers, tabu_markers

        if frame >= len(history['assignments']):
            return cost_current_line, cost_best_line

        for marker in assignment_markers:
            marker.remove()
        assignment_markers = []

        for marker in tabu_markers:
            marker.remove()
        tabu_markers = []

        assn = history['assignments'][frame]
        for i in range(num_tasks):
            j = np.argmax(assn[i])
            color = 'green' if frame == len(history['assignments']) - 1 and np.array_equal(assn, best_assignment) else 'blue'
            marker = ax_assign.plot(j, i, 'o', markersize=12, color=color,
                                   markerfacecolor='none', markeredgewidth=2)[0]
            assignment_markers.append(marker)

            if frame > 0 and history['last_move'][frame-1] is not None:
                task_idx, old_server, new_server = history['last_move'][frame-1]
                if task_idx == i:
                    arrow = ax_assign.annotate('', xy=(new_server, task_idx),
                                              xytext=(old_server, task_idx),
                                              arrowprops=dict(arrowstyle='->',
                                                            color='red', lw=2))
                    assignment_markers.append(arrow)

        iters = list(range(frame + 1))
        cost_current_line.set_data(iters, history['current_cost'][:frame+1])
        cost_best_line.set_data(iters, history['best_cost'][:frame+1])

        tabu = history['tabu_list'][frame]
        for slot_idx, move in enumerate(tabu):
            if move is not None:
                task_idx, old_server, new_server = move
                marker = ax_tabu.plot(slot_idx, task_idx, 's', markersize=15,
                                     color='red', alpha=0.7)[0]
                tabu_markers.append(marker)
                text = ax_tabu.text(slot_idx, task_idx, f'{old_server}→{new_server}',
                                   ha='center', va='center', fontsize=7, color='white')
                tabu_markers.append(text)

        fig.suptitle(f'tabu search: iteration {frame}/{max_iter-1} | ' +
                    f'cost={history["current_cost"][frame]:.1f}', fontsize=12)

        return cost_current_line, cost_best_line

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(history['assignments']), interval=200, blit=False)

    writer = PillowWriter(fps=5)
    anim.save('demo_tabu_animated.gif', writer=writer, dpi=100)
    plt.close()

    print(f"saved demo_tabu_animated.gif ({len(history['assignments'])} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="animated demonstrations of optimization algorithms")
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['all', 'lbfgsb', 'pso', 'tabu'],
                       help='algorithm to animate')

    args = parser.parse_args()

    print("\noptimization algorithm animated demonstrations")
    print("generating gif animations (this may take a minute)...\n")

    if args.algorithm in ['all', 'lbfgsb']:
        animate_lbfgsb()
        print()

    if args.algorithm in ['all', 'pso']:
        animate_pso()
        print()

    if args.algorithm in ['all', 'tabu']:
        animate_tabu()
        print()

    print("animations completed!")
