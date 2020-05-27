import numpy as np

def pageRankScore(A, alpha):
	print('A: \n', A, '\n')
	n = len(A)

	# Initialize x with indegrees and normalize it
	previous_x = A.sum(axis=0) / np.sum(A)
	print('Initialization of x:', '\n', np.transpose(previous_x))

	# Creation of P the probability transition matrix
	P = np.zeros((n, n))
	for i in range(n):
		P[i] = A[i] / np.sum(A[i])
	print('probability transition matrix:\n', P, '\n')

	# Creation of google matrix, vector v and e
	v = np.full((n, 1), 1.0/n)
	e = np.full((n, 1), 1.0)
	G = alpha * P + (1 - alpha) * np.dot(e, np.transpose(v))
	print('Google matrix:\n', G, '\n')

	epsilon = 0.00000001
	iteration = 1
	# transpose x to get x^T and iterating x^T * G
	previous_x = np.transpose(previous_x)

	# keep iterating while x isn't stationary
	while (True):
		x = np.dot(previous_x, G)
		if (abs(x - previous_x < epsilon)).all():
			return x
		if iteration <= 3:
			print('Iteration', iteration, '\n', x, '\n')
			iteration += 1
		previous_x = x

def main ():
	# (csv data) −> numpy array
	myGraph = np.genfromtxt('myGraph.csv', delimiter=', ')
	x = pageRankScore(myGraph, 0.7)
	print(x)

main()
