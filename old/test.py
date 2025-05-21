import numpy as np

def F(a, b):
    T = 1 - a + b  # same as 1-a+b = 1-b+a
    # Compute the denominator for F:
    denom = 1 - 0.5 * T * (1 - np.sqrt(1 - (4 * b) / (T**2)))
    return a / denom

def G(a, b):
    T = 1 - b + a  # same as 1-b+a = 1-a+b
    return 0.5 * T * (1 - np.sqrt(1 - (4 * a) / (T**2)))

# Number of random sample points
n_points = 1000

rng = np.random.default_rng(seed=42)
errors = []

for _ in range(n_points):
    # generate a and b in (0,1) until the condition sqrt(a)+sqrt(b)<=1 is met
    while True:
        a = rng.random()
        b = rng.random()
        if np.sqrt(a) + np.sqrt(b) <= 1:
            break
    val_F = F(a, b)
    val_G = G(a, b)
    errors.append(abs(val_F - val_G))

max_error = np.max(errors)
mean_error = np.mean(errors)

print("Maximum absolute error between F(a,b) and G(a,b):", max_error)
print("Mean absolute error:", mean_error)