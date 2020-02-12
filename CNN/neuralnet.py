from __future__ import print_function
import sys
import string
import math
import numpy as np
import random

def update(m, lr, g_m):
	matrix = m
	for i in range(0, np.size(m, 0)):
		matrix[i] = matrix[i] - lr * g_m[i]
	return matrix


def LinearForward(x, alpha):
	#print("z.shape", x.shape)
	#print("alpha.shape", alpha.shape)
	a = np.matmul(x,alpha.T)
	#print("alpha shape", alpha.shape)
	#print("x shape", x.size)
	#print("a", a)
	return a

def SigmoidForward(a):
	return 1/(1+np.exp(-a))

def SoftmaxForward(b):
	den = 0
	for j in range(0, len(b)):
		den += np.exp(b[j])

	y_hat_mat = list()
	for j in range(0, len(b)):
		y_hat = np.exp(b[j]) / den
		y_hat_mat.append(y_hat)

	return np.array(y_hat_mat)

def CrossEntropyForward(y, y_hat):
	#print("y", y)
	#print("yhat", y_hat)
	#N = len(y)
	K = 10
	J = 0
	for k in range(0, K):
		J += y[k] * math.log(y_hat[k])
	#J = -1/N * J
	return J

# Forward Computation
def NNForward(x, y, alpha, beta):

	#print("xxx", x)
	a = LinearForward(x, alpha)
	# adding the bias
	#print("a", a)
	#print(SigmoidForward(a))
	z = list(SigmoidForward(a))
	# Adding a bias
	zz = [1] + z
	#print("zz", zz)
	#z = np.hstack((1, SigmoidForward(a)))
	b = LinearForward(np.array(zz), beta)

	y_hat = SoftmaxForward(b)
	#print(y_hat)
	#J = CrossEntropyForward(y, y_hat)
	o = [x, a, np.array(zz), b, y_hat]

	#print("a", a)
	#print("b", b)
	#print(y)
	return o

def SoftmaxBackward(y_hat, y):
	gb = y_hat - y
	return gb

def LinearBackward_beta(z, beta, g_b):
	#print("beta", beta)
	beta_star = np.delete(beta, 0, 1)
	#print("beta_star", beta_star)
	#print("g_b shape", g_b.shape)
	#print("z shape", z.shape)
	g_beta = np.outer(g_b, np.transpose(z))

	g_z = np.dot(np.transpose(beta_star), g_b)

	return g_beta, g_z

def LinearBackward_alpha(x, ga):
	g_alpha = np.outer(ga, np.transpose(x))
	return g_alpha

def SigmoidBackward(z, g_z):
	z_l = list(z)
	z = np.array([z_l[1:]])
	zz = np.multiply(z, (np.ones(z.shape)-z))
	#zz = z - np.multiply(z, z)
	zzz = np.transpose(zz)
	g_z_n = [list(g_z)]
	g_a = np.multiply(np.transpose(g_z_n), zzz)
	return g_a

# Backpropagation
def NNBackward(x, y, alpha, beta, o):
	# Place intermediate quantities x, a z, b, y_hat, J in o in scope
	# g_J = dJdJ = 1
	#g_yhat = CrossEntropyBackward(y, y_hat, J, g_J)
	#o = object(x, a, b, z, y_hat)

	y_hat = o[4]
	z = o[2]

	#print("y_hat", y_hat)
	#print("z", z)

	g_b = SoftmaxBackward(y_hat, y)
	#print("g_b", g_b)
	g_beta, g_z = LinearBackward_beta(z, beta, g_b)
	#print("g_beta", g_beta)
	#print("g_z", g_z)

	g_a = SigmoidBackward(z, g_z)
	g_alpha = LinearBackward_alpha(x, g_a)
	return g_alpha, g_beta


# D: unlabeled train or test dataset
def predict(x, y, alpha, beta):

	y_hat_max = list()
	for i in range(0, len(x)):
		# Compute neural network prediction y_hat = h(x)
		o = NNForward(x[i], y[i], alpha, beta)
		y_hat_prob = o[4]
		maxim = 0
		prediction = -1
		# Predict the label with highest probability l = argmax_k y_hat_k
		for i in range(0, len(y_hat_prob)):
			if y_hat_prob[i] > maxim:
				maxim = y_hat_prob[i]
				prediction = i
		y_hat_max.append(prediction)
	return np.array(y_hat_max)

def error(y, y_hat):
	total = len(y)
	count = 0 
	for i in range(0, len(y)):
		if (y[i] != y_hat[i]):
			count += 1
	return count/total

# save_error(train_entropy, test_entropy, train_error, test_error, metrics_out)

def save_error(train_ce, test_ce, train_e, test_e, path):

	file = open(path, mode = "w")

	with file as outfile: 
		print(" crossentropy(train): " + "\n")
		for i in range(0, len(train_ce)):
			string1 = str(train_ce[i]) + "\n"
			outfile.write(string1)
		print(" crossentropy(test): " + "\n")
		for i in range(0, len(train_ce)):
			string2 = str(test_ce[i]) + "\n"
			outfile.write(string2)
		string3 = "error(train): " + str(train_e) + "\n"
		string4 = "error(test): " + str(test_e) + "\n"
		outfile.write(string3)
		outfile.write(string4)

	file.close()

def oneHot(y):
	result = list()
	for j in range(0, len(y)):
		y_j = list(np.zeros(10))
		y_j[y[j]] = 1
		#print(y_j)
		result.append( y_j)
	return np.array(result)

def data_prep(path):
	y = list()
	x = list()
	file = open(path, "r")

	for line in file:
		
		line_comp = line.split(",")
		#print(line_comp)
		y_label = int(float(line_comp[0]))
		y.append(y_label)

		image = []
		#image.append(1)
		line_data = line_comp[1:]
		for pixel in line_data:
			image.append(int(float(pixel)))
		xxx = [1] + image
		x.append(xxx)


	x_n = np.array(x)	
	y_n = oneHot(y)

	#print ("x_n", len(x_n[1]))
	return x_n, y_n, np.array(y)

def initialize(init_flag, D):
	alpha = np.array([])
	beta = np.array([])
	
	# Initialize bias term to zero
	if int(init_flag) == 1:
    	# initialize weights randomly from a uniformly distribution over [-0.1, 0.1]
		alpha_star = np.random.uniform(-0.1, 0.1, (D, 128))
		beta_star = np.random.uniform(-0.1, 0.1, (10, D))
    	# Bias column
		alpha_b = np.zeros((D, 1))
		beta_b = np.zeros((10, 1))

		alpha = np.hstack((alpha_b, alpha_star))
		beta = np.hstack((beta_b, beta_star))
	elif int(init_flag) == 2:
    	# initialize all weights to zero
		alpha = np.zeros((D, 129))
		beta = np.zeros((10, D+1))

	return alpha, beta

def SGD(trainx, trainy, testx, testy, epoch, lr, D):
	# Initialize parameters
	alpha, beta = initialize(init_flag, D)
	train_entropy = []
	test_entropy = []
	for j in range(0, epoch):
	# for e in {1, 2, ... E} do
	# 	for (x, y) in D do
	#   for x, y in D (trainx, trainy):
		for i in range(0, len(trainx)):
	# 		Computer neural network layers:
	# 		o = object(x, a, b, z, y_hat) = NNForward(x, y, alpha, beta)
			o = NNForward(trainx[i], trainy[i], alpha, beta)

	# 		Compute gradient via backprop:
			g_alpha, g_beta = NNBackward(trainx[i], trainy[i], alpha, beta, o)
	# 		g_alpha = grad_alpha J
	# 		g_beta = grad_beta J

	# 		Update parameters
			alpha = update(alpha, lr, g_alpha)
			beta = update(beta, lr, g_beta)

		# 	Evaluate training mean cross-entropy JD(alpha, beta)

		o = 0
		
		N_train = len(trainx)
		train_ce_list = list()
		for i in range(0, N_train):
			o = NNForward(trainx[i], trainy[i], alpha, beta)
			train_ce = CrossEntropyForward(trainy[i], o[4])
			train_ce_list.append(train_ce)
		#train_ce = 0
			
		train_entropy.append(-sum(train_ce_list)/N_train)

		test_ce_list = list()
		N_test = len(testx)
		# 	Evaluate test mean cross-entropy JDt(alpha, beta)
		for i in range(0, N_test):
			o = NNForward(testx[i], testy[i], alpha, beta)
			test_ce = CrossEntropyForward(testy[i], o[4])
			test_ce_list.append(test_ce)
		#test_ce = 0
		test_entropy.append(-sum(test_ce_list)/N_test)

	# return parameters alpha, beta
	return alpha, beta, train_entropy, test_entropy

def save_labels(predict, path):
	file = open(path, mode = "w")
	with file as outfile:
		for i in range(0, len(predict)):
			outfile.write(str(predict[i]) + "\n")
	file.close()
	return 

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input  = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = sys.argv[6]
    hidden_units = sys.argv[7]
    init_flag = sys.argv[8]
    learning_rate = sys.argv[9]

    # x: np 2D array, y: np 1D array
    x_train, y_train_tru, y_train_notHot = data_prep(train_input)
    x_test, y_test_tru, y_test_notHot = data_prep(test_input)

    D = int(hidden_units)
    epoch = int(num_epoch)
    lr = float(learning_rate)

    alpha, beta, train_entropy, test_entropy = SGD(x_train, y_train_tru, x_test, y_test_tru, epoch, lr, D)

    #print(beta)

    train_y_pred = predict(x_train, y_train_tru, alpha, beta)
    test_y_pred = predict(x_test, y_test_tru, alpha, beta)

    save_labels(train_y_pred, train_out)
    save_labels(test_y_pred, test_out)

    train_error = error(train_y_pred, y_train_notHot)
    test_error = error(test_y_pred, y_test_notHot)

    #avg_train_entropy = [sum(train_entropy)/epoch]
    #avg_test_entropy =  [sum(test_entropy)/epoch]
    save_error(train_entropy, test_entropy, train_error, test_error, metrics_out)

'''
    # Compute the gradient by backpropogation
    grad_bp = backprop(x, y, theta)

    # Approximate the gradient by the centered finite difference method
    grad_fd = finite_diff(x, y, theta)

    # Check that the gradients are (nearly) the same
    diff = grad_bp - grad_fd # element-wise difference of two vectors
    print 12_norm(diff) # this value should be small



def finite_diff(x, y, theta):
	epsilon = 1e-5
	grad = zero_vector(theta.length)
	for m in [1, ....., theta.length]:
		d = zero_vector(theta.length)
		d[m] = 1
		v = forward(x, y, theta + epsilon * d)
		v -= forward(x, y, theta - epsilon * d)
		v /= 2*epsilon
		grad[m] = v
'''
