import numpy as np
import time
from functools import wraps

def set_random_seed(seed):
    np.random.seed(seed)

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

def validate_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return True

def validate_in_range(value, name, min_val, max_val):
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
    return True

def validate_matrix_shape(matrix, expected_shape, name):
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {matrix.shape}")
    return True

def save_results(results, filepath):
    np.savez(filepath, **results)
    print(f"results saved to {filepath}")

def load_results(filepath):
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


if __name__ == "__main__":
    print("testing utils module")

    set_random_seed(42)
    r1 = np.random.rand(3)
    set_random_seed(42)
    r2 = np.random.rand(3)
    assert np.allclose(r1, r2), "random seed not working"
    print(f"random seed test passed: {r1}")

    @timer
    def test_func():
        time.sleep(0.1)
        return 42

    result = test_func()
    assert result == 42, "timer decorator broke function"
    print("timer decorator test passed")

    try:
        validate_positive(5, "test_value")
        print("validate_positive test passed")
    except ValueError as e:
        print(f"unexpected error: {e}")

    try:
        validate_positive(-5, "test_value")
        print("validate_positive should have failed")
    except ValueError as e:
        print(f"expected error: {e}")

    try:
        validate_in_range(2.5, "test_value", 2.0, 3.0)
        print("validate_in_range test passed")
    except ValueError as e:
        print(f"unexpected error: {e}")

    try:
        validate_in_range(5.0, "test_value", 2.0, 3.0)
        print("validate_in_range should have failed")
    except ValueError as e:
        print(f"expected error: {e}")

    matrix = np.ones((3, 4))
    try:
        validate_matrix_shape(matrix, (3, 4), "test_matrix")
        print("validate_matrix_shape test passed")
    except ValueError as e:
        print(f"unexpected error: {e}")

    try:
        validate_matrix_shape(matrix, (3, 5), "test_matrix")
        print("validate_matrix_shape should have failed")
    except ValueError as e:
        print(f"expected error: {e}")

    print("all tests passed")
