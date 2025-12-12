import numpy as np
from .network import calc_network_delay
from .queueing import calc_queueing_delay, calc_service_time

def calc_total_latency(task, server, tau, beta, utilization):
    network_delay = calc_network_delay(tau, beta, task.data_size)
    capacity = server.get_capacity()
    queueing_delay = calc_queueing_delay(utilization, capacity)
    service_time = calc_service_time(task.cpu_cycles, capacity)
    return network_delay + queueing_delay + service_time


if __name__ == "__main__":
    print("testing latency module")

    from task import Task
    from server import EdgeServer

    task = Task(0, 2.0, 1e8, 5e4)
    server = EdgeServer(0, 1e9, 3e9, 1.0, 5e-10, 10.0)
    server.set_freq(2e9)

    tau = 0.005
    beta = 5e-9
    utilization = 0.6

    latency = calc_total_latency(task, server, tau, beta, utilization)
    print(f"total latency: {latency:.6f}s")

    net_delay = calc_network_delay(tau, beta, task.data_size)
    capacity = server.get_capacity()
    queue_delay = calc_queueing_delay(utilization, capacity)
    serv_time = calc_service_time(task.cpu_cycles, capacity)
    expected = net_delay + queue_delay + serv_time

    print(f"  network delay: {net_delay:.6f}s")
    print(f"  queueing delay: {queue_delay:.9f}s")
    print(f"  service time: {serv_time:.6f}s")
    print(f"  total (expected): {expected:.6f}s")

    assert np.isclose(latency, expected)
    print("calc_total_latency test passed")

    print("\nlatency at different utilizations:")
    for util in [0.3, 0.5, 0.7, 0.9]:
        lat = calc_total_latency(task, server, tau, beta, util)
        print(f"  util={util}: latency={lat:.6f}s")

    print("\nlatency at different frequencies:")
    for freq in [1e9, 1.5e9, 2e9, 2.5e9]:
        server.set_freq(freq)
        lat = calc_total_latency(task, server, tau, beta, 0.6)
        print(f"  freq={freq:.2e}Hz: latency={lat:.6f}s")

    print("\nall tests passed")
