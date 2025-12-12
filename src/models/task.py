import numpy as np

class Task:
    def __init__(self, task_id, arrival_rate, cpu_cycles, data_size):
        self.task_id = task_id
        self.arrival_rate = arrival_rate  # lambda_j
        self.cpu_cycles = cpu_cycles  # c_j
        self.data_size = data_size  # d_j

    def __repr__(self):
        return f"Task(id={self.task_id}, lambda={self.arrival_rate:.2f}, cycles={self.cpu_cycles}, data={self.data_size})"


def generate_random_tasks(num_tasks, arrival_rate_range=(0.1, 5.0), cpu_cycles_range=(1e6, 1e9), data_size_range=(1e3, 1e6)):
    tasks = []
    for i in range(num_tasks):
        arrival_rate = np.random.uniform(*arrival_rate_range)
        cpu_cycles = np.random.uniform(*cpu_cycles_range)
        data_size = np.random.uniform(*data_size_range)
        tasks.append(Task(i, arrival_rate, cpu_cycles, data_size))
    return tasks


if __name__ == "__main__":
    print("testing task module")

    task = Task(0, 2.5, 1e8, 5e4)
    print(f"created task: {task}")
    assert task.task_id == 0
    assert task.arrival_rate == 2.5
    assert task.cpu_cycles == 1e8
    assert task.data_size == 5e4
    print("task creation test passed")

    np.random.seed(42)
    tasks = generate_random_tasks(5)
    print(f"generated {len(tasks)} random tasks")
    for task in tasks:
        print(f"  {task}")
    assert len(tasks) == 5
    assert all(isinstance(t, Task) for t in tasks)
    assert all(t.arrival_rate > 0 for t in tasks)
    assert all(t.cpu_cycles > 0 for t in tasks)
    assert all(t.data_size > 0 for t in tasks)
    print("generate_random_tasks test passed")

    assert all(tasks[i].task_id == i for i in range(len(tasks)))
    print("task id sequence test passed")

    print("all tests passed")
