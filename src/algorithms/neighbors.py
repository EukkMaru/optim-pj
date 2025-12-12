import numpy as np

def generate_swap_neighbor(assignment_matrix):
    num_tasks, num_servers = assignment_matrix.shape

    task_assignments = np.argmax(assignment_matrix, axis=1)

    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        task1 = np.random.randint(0, num_tasks)
        task2 = np.random.randint(0, num_tasks)
        if task1 != task2 and task_assignments[task1] != task_assignments[task2]:
            break
        attempts += 1

    if attempts == max_attempts:
        return None, None

    new_assignment = assignment_matrix.copy()
    server1 = task_assignments[task1]
    server2 = task_assignments[task2]

    new_assignment[task1, :] = 0
    new_assignment[task2, :] = 0
    new_assignment[task1, server2] = 1
    new_assignment[task2, server1] = 1

    move = ('swap', task1, task2, server1, server2)
    return new_assignment, move

def generate_move_neighbor(assignment_matrix):
    num_tasks, num_servers = assignment_matrix.shape

    if num_servers < 2:
        return None, None

    task = np.random.randint(0, num_tasks)
    current_server = np.argmax(assignment_matrix[task, :])

    new_server = np.random.randint(0, num_servers)
    attempts = 0
    while new_server == current_server and attempts < 10:
        new_server = np.random.randint(0, num_servers)
        attempts += 1

    if new_server == current_server:
        return None, None

    new_assignment = assignment_matrix.copy()
    new_assignment[task, :] = 0
    new_assignment[task, new_server] = 1

    move = ('move', task, current_server, new_server)
    return new_assignment, move

def generate_random_neighbor(assignment_matrix):
    if np.random.rand() < 0.5:
        return generate_swap_neighbor(assignment_matrix)
    else:
        return generate_move_neighbor(assignment_matrix)


if __name__ == "__main__":
    print("testing neighbors module")

    np.random.seed(42)

    num_tasks = 5
    num_servers = 2
    assignment_matrix = np.zeros((num_tasks, num_servers), dtype=int)
    for j in range(num_tasks):
        assignment_matrix[j, j % num_servers] = 1

    print("original assignment:")
    print(assignment_matrix)
    print(f"task assignments: {np.argmax(assignment_matrix, axis=1)}")

    print("\ntesting swap neighbor:")
    new_assignment, move = generate_swap_neighbor(assignment_matrix)
    if new_assignment is not None:
        print(f"move: {move}")
        print("new assignment:")
        print(new_assignment)
        print(f"task assignments: {np.argmax(new_assignment, axis=1)}")

        assert np.all(np.sum(new_assignment, axis=1) == 1), "invalid assignment"
        assert new_assignment.shape == assignment_matrix.shape
        print("swap neighbor test passed")
    else:
        print("could not generate swap neighbor")

    print("\ntesting move neighbor:")
    new_assignment2, move2 = generate_move_neighbor(assignment_matrix)
    if new_assignment2 is not None:
        print(f"move: {move2}")
        print("new assignment:")
        print(new_assignment2)
        print(f"task assignments: {np.argmax(new_assignment2, axis=1)}")

        assert np.all(np.sum(new_assignment2, axis=1) == 1), "invalid assignment"
        assert new_assignment2.shape == assignment_matrix.shape
        print("move neighbor test passed")
    else:
        print("could not generate move neighbor")

    print("\ntesting random neighbor generation:")
    for i in range(5):
        neighbor, move = generate_random_neighbor(assignment_matrix)
        if neighbor is not None:
            print(f"  iteration {i}: {move[0]} move")
            assert np.all(np.sum(neighbor, axis=1) == 1), "invalid assignment"
    print("random neighbor test passed")

    print("\ntesting with single server:")
    single_server_assignment = np.ones((num_tasks, 1), dtype=int)
    neighbor, move = generate_move_neighbor(single_server_assignment)
    assert neighbor is None, "should not generate neighbor with single server"
    print("single server test passed")

    print("\nall tests passed")
