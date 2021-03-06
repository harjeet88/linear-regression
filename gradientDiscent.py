from LRcomputeCost import compute_cost
import numpy as np

def gradient_descent(X, y, theta, alpha, num_iters):
	m = y.size
	J_history = np.zeros(shape=(num_iters, 1))
 
	for i in range(num_iters):
 		predictions = X.dot(theta).flatten()
		errors_x1 = (predictions - y) * X[:, 0]
		errors_x2 = (predictions - y) * X[:, 1]
		theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
		theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
	
		J_history[i, 0] = compute_cost(X, y, theta)
 
	return theta, J_history
 
