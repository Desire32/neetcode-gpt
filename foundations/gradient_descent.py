class Solution:
    def step_rule(self, init: int, learning_rate: float, derivative) -> float:
        return init - learning_rate * derivative
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        if iterations == 0: return init
        # θ := θ * α * gradient
        # func is f(x) = x^2
        x_new = 0
        for _ in range(iterations):
            derivative = 2 * init
            x_new = self.step_rule(init, learning_rate, derivative)
            init = x_new
            print(init)
        return round(init, 5)