import os
import numpy as np

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2*len(X))

def gradientDescent(X, y, theta, alpha, iter_count):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iter_count)
    parameters = int(theta.ravel().shape[1])
    for i in range(iter_count):
        error = X*theta.T - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha/len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

def formatData(data, matrix):
    for line in data:
        line = line.strip("\n")
        l1 = [float(x) for x in line.split(",")]
        l1.insert(0, 1)
        matrix.append(l1)
    return matrix

path = os.getcwd() + "\\linear_data.txt"
with open(path, "r") as csvfile:
    data = csvfile.readlines()
matrix = []
matrix = formatData(data, matrix)
    # print(l1)
# print(matrix)
cols = len(matrix[0])
matrix = np.matrix(matrix)
X = matrix[:, 0:cols-1]
y = matrix[:, cols-1:cols]
theta = np.matrix(np.array([0,0]))
# print(matrix)
# print(X)
# print(y)
iter_count = 1000
alpha = 0.01
g, cost = gradientDescent(X, y, theta, alpha, iter_count)
print(g)
