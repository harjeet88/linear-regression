
def compute_cost(X, y, theta):
	m = y.size
	predictions = X.dot(theta).flatten()
	sqErrors = (predictions - y) ** 2
 
	J = (1.0 / (2 * m)) * sqErrors.sum()
 
	return J
 
