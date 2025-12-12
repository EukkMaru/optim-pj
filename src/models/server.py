import numpy as np

class EdgeServer:
    def __init__(self, server_id, freq_min, freq_max, capacity_coef, alpha, p_idle):
        self.server_id = server_id
        self.freq_min = freq_min  # minimum frequency (Hz)
        self.freq_max = freq_max  # maximum frequency (Hz)
        self.capacity_coef = capacity_coef  # kappa_i (cycles per Hz)
        self.alpha = alpha  # power coefficient alpha_i
        self.p_idle = p_idle  # idle power P_i0

        self.freq = freq_min  # current frequency
        self.utilization = 0.0  # current utilization rho_i

    def set_freq(self, freq):
        if freq < self.freq_min or freq > self.freq_max:
            raise ValueError(f"freq {freq} out of bounds [{self.freq_min}, {self.freq_max}]")
        self.freq = freq

    def get_capacity(self):
        return self.capacity_coef * self.freq

    def __repr__(self):
        return f"EdgeServer(id={self.server_id}, freq={self.freq:.2e}, cap={self.get_capacity():.2e})"


def generate_random_servers(num_servers, freq_min=1e9, freq_max=3e9, capacity_coef_range=(1.0, 2.0), alpha_range=(1e-10, 1e-9), p_idle_range=(5.0, 15.0)):
    servers = []
    for i in range(num_servers):
        capacity_coef = np.random.uniform(*capacity_coef_range)
        alpha = np.random.uniform(*alpha_range)
        p_idle = np.random.uniform(*p_idle_range)
        servers.append(EdgeServer(i, freq_min, freq_max, capacity_coef, alpha, p_idle))
    return servers


if __name__ == "__main__":
    print("testing server module")

    server = EdgeServer(0, 1e9, 3e9, 1.5, 5e-10, 10.0)
    print(f"created server: {server}")
    assert server.server_id == 0
    assert server.freq_min == 1e9
    assert server.freq_max == 3e9
    assert server.capacity_coef == 1.5
    assert server.alpha == 5e-10
    assert server.p_idle == 10.0
    assert server.freq == 1e9
    print("server creation test passed")

    server.set_freq(2e9)
    assert server.freq == 2e9
    print(f"freq set to {server.freq:.2e}")
    print("set_freq test passed")

    capacity = server.get_capacity()
    expected_capacity = 1.5 * 2e9
    assert capacity == expected_capacity
    print(f"capacity: {capacity:.2e}, expected: {expected_capacity:.2e}")
    print("get_capacity test passed")

    try:
        server.set_freq(5e9)
        print("should have raised error")
    except ValueError as e:
        print(f"expected error: {e}")

    np.random.seed(42)
    servers = generate_random_servers(3)
    print(f"generated {len(servers)} random servers")
    for server in servers:
        print(f"  {server}")
    assert len(servers) == 3
    assert all(isinstance(s, EdgeServer) for s in servers)
    assert all(s.freq >= s.freq_min and s.freq <= s.freq_max for s in servers)
    print("generate_random_servers test passed")

    print("all tests passed")
