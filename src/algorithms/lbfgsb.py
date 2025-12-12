import numpy as np

class LBFGSB:
    def __init__(self, func, grad, bounds, x0, m=10, max_iter=100, tol=1e-6, ls_max_iter=20):
        self.func = func
        self.grad = grad
        self.bounds = np.array(bounds)
        self.x = np.array(x0, dtype=float)
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.ls_max_iter = ls_max_iter

        self.n = len(x0)
        self.s_history = []
        self.y_history = []
        self.rho_history = []

        self.f_val = None
        self.g_val = None
        self.iteration = 0
        self.success = False

    def project_bounds(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def compute_direction(self, g):
        q = g.copy()
        alpha = []

        for i in range(len(self.s_history) - 1, -1, -1):
            a = self.rho_history[i] * np.dot(self.s_history[i], q)
            alpha.append(a)
            q = q - a * self.y_history[i]

        alpha.reverse()

        if len(self.s_history) > 0:
            s_last = self.s_history[-1]
            y_last = self.y_history[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            r = gamma * q
        else:
            r = q

        for i in range(len(self.s_history)):
            beta = self.rho_history[i] * np.dot(self.y_history[i], r)
            r = r + self.s_history[i] * (alpha[i] - beta)

        return -r

    def line_search(self, x, d, f, g, c1=1e-4, alpha_init=1.0):
        alpha = alpha_init
        f_new = None
        x_new = None

        d_proj = self.project_bounds(x + d) - x
        directional_derivative = np.dot(g, d_proj)

        if directional_derivative >= 0:
            d_proj = -g
            directional_derivative = np.dot(g, d_proj)

        for _ in range(self.ls_max_iter):
            x_new = self.project_bounds(x + alpha * d_proj)
            f_new = self.func(x_new)

            if f_new <= f + c1 * alpha * directional_derivative:
                break

            alpha *= 0.5

        return x_new, f_new, alpha

    def optimize(self):
        self.f_val = self.func(self.x)
        self.g_val = self.grad(self.x)

        for iteration in range(self.max_iter):
            self.iteration = iteration

            grad_norm = np.linalg.norm(self.g_val)
            if grad_norm < self.tol:
                self.success = True
                break

            d = self.compute_direction(self.g_val)
            x_new, f_new, alpha = self.line_search(self.x, d, self.f_val, self.g_val)
            g_new = self.grad(x_new)

            s = x_new - self.x
            y = g_new - self.g_val

            ys = np.dot(y, s)
            if ys > 1e-10:
                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(1.0 / ys)

                if len(self.s_history) > self.m:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                    self.rho_history.pop(0)

            self.x = x_new
            self.f_val = f_new
            self.g_val = g_new

        return self.x, self.f_val, self.success


def minimize_lbfgsb(func, x0, grad, bounds, max_iter=100, tol=1e-6):
    optimizer = LBFGSB(func, grad, bounds, x0, max_iter=max_iter, tol=tol)
    x_opt, f_opt, success = optimizer.optimize()

    result = {
        'x': x_opt,
        'fun': f_opt,
        'success': success,
        'nit': optimizer.iteration,
        'message': 'optimization converged' if success else 'max iterations reached'
    }

    return result


if __name__ == "__main__":
    print("testing lbfgsb module")

    print("\ntest 1: quadratic function")
    def quad_func(x):
        return (x[0] - 1.5)**2 + (x[1] - 2.0)**2

    def quad_grad(x):
        return np.array([2*(x[0] - 1.5), 2*(x[1] - 2.0)])

    bounds = [[0.0, 3.0], [0.0, 3.0]]
    x0 = np.array([0.5, 0.5])

    result = minimize_lbfgsb(quad_func, x0, quad_grad, bounds)
    print(f"  optimal x: {result['x']}")
    print(f"  optimal f: {result['fun']:.6f}")
    print(f"  expected: [1.5, 2.0], f=0.0")
    print(f"  success: {result['success']}")
    assert np.allclose(result['x'], [1.5, 2.0], atol=1e-4)
    print("test 1 passed")

    print("\ntest 2: rosenbrock function")
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def rosenbrock_grad(x):
        dfdx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dfdx1 = 200*(x[1] - x[0]**2)
        return np.array([dfdx0, dfdx1])

    bounds = [[0.0, 2.0], [0.0, 2.0]]
    x0 = np.array([0.0, 0.0])

    result = minimize_lbfgsb(rosenbrock, x0, rosenbrock_grad, bounds, max_iter=200)
    print(f"  optimal x: {result['x']}")
    print(f"  optimal f: {result['fun']:.6f}")
    print(f"  expected: [1.0, 1.0], f=0.0")
    print(f"  success: {result['success']}")
    assert result['fun'] < 1e-4
    print("test 2 passed")

    print("\ntest 3: constrained minimum")
    def constrained_func(x):
        return (x[0] - 5.0)**2 + (x[1] - 5.0)**2

    def constrained_grad(x):
        return np.array([2*(x[0] - 5.0), 2*(x[1] - 5.0)])

    bounds = [[0.0, 2.0], [0.0, 2.0]]
    x0 = np.array([0.5, 0.5])

    result = minimize_lbfgsb(constrained_func, x0, constrained_grad, bounds)
    print(f"  optimal x: {result['x']}")
    print(f"  optimal f: {result['fun']:.6f}")
    print(f"  expected: [2.0, 2.0] (boundary)")
    print(f"  success: {result['success']}")
    assert np.allclose(result['x'], [2.0, 2.0], atol=1e-4)
    print("test 3 passed")

    print("\nall tests passed")
