from lib.dataset import generate_data

X, y, beta  = generate_data(n=20, m=5, sigma=0.3, density=0.2, seed=123456)

print(X.shape)