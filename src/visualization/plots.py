import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history, title="convergence", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('objective value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved convergence plot to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_utilization(utilizations, title="server utilization", save_path=None):
    plt.figure(figsize=(10, 6))
    servers = [f"server {i}" for i in range(len(utilizations))]
    colors = ['green' if u < 0.7 else 'orange' if u < 0.9 else 'red' for u in utilizations]

    plt.bar(servers, utilizations, color=colors, alpha=0.7)
    plt.axhline(y=0.95, color='r', linestyle='--', label='threshold (0.95)')
    plt.xlabel('server', fontsize=12)
    plt.ylabel('utilization', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim([0, 1.0])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved utilization plot to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_comparison(results, save_path=None):
    algorithms = list(results.keys())
    objectives = [results[alg]['objective'] for alg in algorithms]
    energies = [results[alg]['energy'] for alg in algorithms]
    latencies = [results[alg]['latency'] for alg in algorithms]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.bar(algorithms, objectives, alpha=0.7, color='blue')
    ax1.set_ylabel('objective value', fontsize=12)
    ax1.set_title('objective comparison', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(algorithms, energies, alpha=0.7, color='green')
    ax2.set_ylabel('energy (J)', fontsize=12)
    ax2.set_title('energy comparison', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3.bar(algorithms, latencies, alpha=0.7, color='orange')
    ax3.set_ylabel('latency (s)', fontsize=12)
    ax3.set_title('latency comparison', fontsize=14)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("testing plots module")

    print("\ntesting convergence plot...")
    history = [1000, 900, 850, 820, 800, 790, 785, 783, 782, 782]
    plot_convergence(history, title="test convergence")
    print("convergence plot test passed")

    print("\ntesting utilization plot...")
    utilizations = np.array([0.5, 0.75, 0.95, 0.3])
    plot_utilization(utilizations, title="test utilization")
    print("utilization plot test passed")

    print("\ntesting comparison plot...")
    results = {
        'algorithm_a': {'objective': 1000, 'energy': 500, 'latency': 0.5},
        'algorithm_b': {'objective': 1100, 'energy': 450, 'latency': 0.65},
        'algorithm_c': {'objective': 950, 'energy': 520, 'latency': 0.43}
    }
    plot_comparison(results)
    print("comparison plot test passed")

    print("\nall tests passed")
    print("note: plots should have been displayed (close windows to continue)")
