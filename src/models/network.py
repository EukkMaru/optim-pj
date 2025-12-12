import numpy as np

def calc_network_delay(tau, beta, data_size):
    return tau + beta * data_size

def generate_network_params(num_tasks, num_servers, tau_range=(0.001, 0.01), beta_range=(1e-9, 1e-8)):
    tau = np.random.uniform(*tau_range, size=(num_tasks, num_servers))
    beta = np.random.uniform(*beta_range, size=(num_tasks, num_servers))
    return tau, beta


if __name__ == "__main__":
    print("testing network module")

    tau = 0.005
    beta = 5e-9
    data_size = 1e5
    delay = calc_network_delay(tau, beta, data_size)
    expected = 0.005 + 5e-9 * 1e5
    print(f"network delay: {delay:.6f}s, expected: {expected:.6f}s")
    assert np.isclose(delay, expected)
    print("calc_network_delay test passed")

    np.random.seed(42)
    num_tasks = 5
    num_servers = 3
    tau, beta = generate_network_params(num_tasks, num_servers)
    print(f"tau shape: {tau.shape}, beta shape: {beta.shape}")
    assert tau.shape == (num_tasks, num_servers)
    assert beta.shape == (num_tasks, num_servers)
    assert np.all(tau >= 0.001) and np.all(tau <= 0.01)
    assert np.all(beta >= 1e-9) and np.all(beta <= 1e-8)
    print("generate_network_params test passed")

    print(f"sample tau matrix:\n{tau}")
    print(f"sample beta matrix:\n{beta}")

    data_sizes = np.array([1e4, 5e4, 1e5, 5e5, 1e6])
    print("sample network delays for task 0:")
    for server_id in range(num_servers):
        delay = calc_network_delay(tau[0, server_id], beta[0, server_id], data_sizes[0])
        print(f"  task 0 to server {server_id}: {delay:.6f}s")

    print("all tests passed")
