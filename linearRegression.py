import numpy as np
import pylab as pl
from LRcomputeCost import compute_cost
from gradientDiscent import gradient_descent

#Load the dataset
data = np.loadtxt('ex1data1.txt', delimiter=',')
 
#Plot the data
pl.scatter(data[:, 0], data[:, 1], marker='o', c='b')
pl.title('Profits distribution')
pl.xlabel('Population of City in 10,000s')
pl.ylabel('Profit in $10,000s')

X = data[:, 0]
y = data[:, 1]
 
 
#number of training samples
m = y.size
 
#Create a mX2 numpy matrix of ones 
#and replace all ones in second column by X values
#first column stays one : remember x subscript zero is all ones. just to make it mathematically convinient 
x = np.ones(shape=(m, 2))
x[:, 1] = X
 
#we need to intialize theta with some dummy values, lets assume its all zeros
theta = np.zeros(shape=(2, 1))
 
#set number of iterations for updating gradient discent
iterations = 50000

#set learning rate
alpha = 0.01
 
#display initial cost
print compute_cost(x, y, theta)
 
theta, J_history = gradient_descent(x, y, theta, alpha, iterations)
 
print theta
#Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta).flatten()
print 'For population = 35,000, we predict a profit of %f' % (predict1 * 10000)
predict2 = np.array([1, 5.0]).dot(theta).flatten()
print 'For population = 50,000, we predict a profit of %f' % (predict2 * 10000)
 
#Plot the results
result = x.dot(theta).flatten()
pl.plot(data[:, 0], result)
pl.show()
 
 
#Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
 
 
#initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))
 
#Fill out J_vals
for t1, element in enumerate(theta0_vals):
	for t2, element2 in enumerate(theta1_vals):
		thetaT = np.zeros(shape=(2, 1))
		thetaT[0][0] = element
		thetaT[1][0] = element2
		J_vals[t1, t2] = compute_cost(x, y, thetaT)
 
#Contour plot
J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
pl.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
pl.xlabel('theta_0')
pl.ylabel('theta_1')
pl.scatter(theta[0][0], theta[1][0])
pl.show()
