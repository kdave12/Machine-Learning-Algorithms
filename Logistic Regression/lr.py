from __future__ import print_function
import sys
import string
import math
import numpy as np

# takes in a set of features, labels and model parameters 
# outputs the negative log-likelihood of the regression model


def input_2_dict(datafile):
    file1 = open(datafile, "r")
    
    dictionary = dict()
    for word in file1:
        values = word.split()
        wrd = values[0]
        indx = values[1]
        dictionary[wrd] = indx
    return dictionary

# update bias term
# no shuffling

def data_prep(data, dict_len):
	file1 = open(data, "r")

	x_lst = list()
	y_lst = list()
	for line in file1:

		line_data_dict = dict()
		line_comp = line.split()
		y_label = int(line_comp[0])
		y_lst.append(y_label)
		line_data = line_comp[1:]

		for index in line_data:
			# x - index:1
			x = index.split(":")
			ind = int(x[0])
			one = x[1]
			line_data_dict[ind] = 1

 		# Add an index (dict_len:1 at the end of each dict for the bias term)
		line_data_dict[dict_len] = 1
		x_lst.append(line_data_dict)

	#print(data_lst)
	return x_lst, y_lst

def lr(x_lst, y_lst, dic_len, epoch):
	learn_rate = 0.1
	theta = [0]*(dic_len+1) # initialized to zero
	n = len(x_lst)
	y = y_lst
	for iter in range(0, epoch):
		for i in range(0, n):
			dot_prod = sparse_dot_prod(theta, x_lst[i]) # what are the dimensions of this?
			x_dict, constant =  x_lst[i], math.exp(dot_prod)/(1+math.exp(dot_prod))
			for j, v in x_dict.items():
				#print("jvalue", j)
				theta[j] += learn_rate*(y[i] - constant)
	return theta

def sparse_dot_prod(theta, x): # What to transpse?
	#print(theta)
	product = 0.0
	for i, v in x.items():  # i is the index : v is 1 initially?
		product += theta[i] # what do I multiply theta[i] with? v or i?
	return product

def sigmoid(a):
    return 1/(1+math.exp(-a))

def predict(theta, x):  #???
	y_pred = list()

	for i in range(0, len(x)):
		a = sparse_dot_prod(theta, x[i])
		sig_val = sigmoid(a)
		if sig_val >= 0.5:
			y_pred.append(1)
		else: 
			y_pred.append(0)
	return y_pred


# takes in a set of features, labels and model parameters 
# outputs the error % of labels incorrectly predicted
def error(true, pred):
	count = 0
	total = len(true)

	for i in range(0, len(pred)):
		if pred[i] != true[i]:
			count += 1

	return count/total


def save_error(error_v, error_t, path): # ???
	file = open(path, mode = "w")

	with file as outfile: 
		string1 = "error(train): " + str(error_v) + "\n"
		string2 = "error(test): " + str(error_t) + "\n"
		outfile.write(string1)
		outfile.write(string2)
	file.close()


def save_y_labels(lst, path):
	file = open(path, mode = "w")
	with file as outfile: 
		for i in range(0, len(lst)):
			outfile.write(str(lst[i]) + "\n")
	file.close()
	return

if __name__ == '__main__':
    formatted_train = sys.argv[1]
    formatted_valid = sys.argv[2]
    formatted_test = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]

    # Extracting info that will be input in the training model
    epoch = int(num_epoch)
    dictionary = input_2_dict(dict_input)
    dic_len = len(dictionary)

    # Training the logistic regression during x-train and y-train
    x_lst, y_train_true = data_prep(formatted_train, dic_len)
    theta = lr(x_lst, y_train_true, dic_len, epoch)     # The output of the train
    #print(theta)
    y_train_pred = predict(theta, x_lst)
    save_y_labels(y_train_pred, train_out)
    #print(y_train_pred)
    #print(y_train_true)
    train_error = error(y_train_pred, y_train_true)
    #print(train_error)
    x_test, y_test_true = data_prep(formatted_test, dic_len)
    y_test_pred = predict(theta, x_test)
    save_y_labels(y_test_pred, test_out)
    test_error = error(y_test_pred, y_test_true)
    #print(test_error)
    save_error(train_error, test_error, metrics_out)
    '''
	# Validation 
	x_valid, y_valid_true = extract_xy(formatted_valid)
 	y_valid_pred = predict(theta, formatted_valid)
 	valid_error = error(y_valid_true, y_valid_pred)

 	# Test
	

 	# Save error in metrics
 	save_error(valid_error, test_error, metrics_out)
'''












