import os
import math
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# def hypothesis(X, theta):
#     h = sigmoid(X * theta.T)
#     # for elem in h:
#     #     elem = sigmoid(elem)
#     return h

def computeCost(X, y, theta):
    pos = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    neg = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T)))
    inner = pos - neg
    return np.sum(inner)/ (len(X))

def gradientDescent(X, y, theta, alpha, iter_count):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iter_count)
    parameters = int(theta.ravel().shape[1])
    for i in range(iter_count):
        error = sigmoid(X * theta.T) - y
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

path = os.getcwd() + "\\logistic_data.txt"
with open(path, "r") as csvfile:
    data = csvfile.readlines()
# matrix = []
matrix = formatData(data, [])
# print(matrix)

cols = len(matrix[0])
matrix = np.matrix(matrix)
X = matrix[:, 0:cols-1]
y = matrix[:, cols-1:cols]
theta = np.matrix(np.zeros(3)) #np.matrix(np.array([0,0,0]))
print(computeCost(X, y, theta))
# print(X, y, theta, sep = "\n\n")
# print(X.shape, y.shape, theta.shape,)

iter_count = 20000
alpha = 0.001
g, cost = gradientDescent(X, y, theta, alpha, iter_count)
print(g)
print(computeCost(X, y, g))