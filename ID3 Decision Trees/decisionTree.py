''' This program implements a Decision Tree learner. 

    This file learns a decision tree with a specified maximum depth, 
    prints the decision tree in a specified format, predicts the labels 
    and calculates the training and testing errors.

    Krishna Dave (kdave)
'''

from __future__ import print_function
import sys
import string
import math
import numpy as np


class Node: 
	def __init__(self, key, depth):
		self.left = None
		self.right = None
		self.val = key
		self.depth = depth
		self.y_values = None
		self.leftval = None
		self.rightval = None


# Description: helper to calculate the entropy of the y-label
# Inputs: takes in the entire train input array
# Returns: the entropy value H(Y)
def H_y(data): 
	# H(Y)
	unique_y, counts_y = np.unique(data[1:, (len(data[0]) - 1)], return_counts = True)
	y_values = dict(zip(unique_y, counts_y))
	
	#print(y_values)
	# Counting the total number of labels
	sum_count = 0
	for each_val in y_values: 
		sum_count += y_values[each_val]

	# Calculate the entropy
	entropy = 0
	for each_value in y_values:
		#print(each_value)
		prob = y_values[each_value]/sum_count
		entropy = entropy + (prob * math.log2(prob))
	entropy = (-1) * entropy

	return entropy

# Description: calculates the Specific conditional entropy
# Inputs: train input array, x column name, x feature uniq value
# Returns: the entropy value H(Y)
def H_specific_cond(data, x_name, x_value): 

	# calculating unique y_values
	y_col_i = (len(data[0]) - 1)
	unique_y, counts_y = np.unique(data[1:, y_col_i], return_counts = True)
	y_values = dict(zip(unique_y, counts_y))
	#print("y_values", y_values)
	# calculate the column index
	xcol_ind = np.where(data[0] == x_name)[0][0]

	# calculating unique x_values
	unique_x, counts_x = np.unique(data[1:, xcol_ind], return_counts = True)
	x_values = dict(zip(unique_x, counts_x))

	#print("x_values: ", x_values)

	entropy_sc = 0

	for each_y in y_values:
		#print("each_y", each_y)
		count = 0
		for each_row in data[1:]: 
			#print(each_row)
			if ((each_row[xcol_ind] == x_value) and (each_row[y_col_i] == each_y)):
				#print("counting!")
				count += 1
		#print(count, each_y)
		prob = count/x_values[x_value]
		if (prob != 0):
			entropy_sc = entropy_sc + prob * math.log2(prob)

	entropy_sc = (-1) * entropy_sc

	return entropy_sc

# Description: calculates the conditional entropy
# Helper functions: H_specific_cond
# Inputs: takes in the entire train input array, and x feature name
# Returns: the entropy value of H
def H_cond(data, x_name):

	#print("x_name", x_name)
	xcol_ind = np.where(data[0] == x_name)[0][0]
	#print("column index", xcol_ind)
	unique_x, counts_x = np.unique(data[1:, xcol_ind], return_counts = True)
	x_values = dict(zip(unique_x, counts_x))
	#print("x_values", x_values)

	# Counting the total number of labels
	sum_count = 0
	for each_val in x_values: 
		sum_count += x_values[each_val]
	#print("sumcount", sum_count)

	# Calculate the conditional entropy
	c_entropy = 0
	for each_val in x_values:
		#print("x_value: ", each_val)
		prob = x_values[each_val]/sum_count
		#print("x_name in H_specific_cond", x_name)
		if (prob != 0):
			c_entropy = c_entropy + prob * H_specific_cond(data, x_name, each_val)
		#print("c_entropy: ", c_entropy)

	return c_entropy

# Description: recursively trains a tree, keeps track of depth
# ** Edge Case: specificied max depth > total # of attributes
# ** Edge Case: max depth == 0
# Inputs:
# Returns: trained trees

def max_vote_classifier(data):
	unique_y, counts_y = np.unique(data[1:, (len(data[0]) - 1)], return_counts = True)
	#y_values = list(unique_y)

	y_values = dict(zip(unique_y, counts_y))

	y_val_list = list()
		
	for each_val in y_values:
		y_val_list.append(each_val)

	# Finding the majority binary label

	print(y_values)

	# Finding the majority label
	max_label = None
	max_label_count = 0
	for label in y_values:
		if ((max_label == None) or (y_values[label] > max_label_count)): 
			max_label = label
			max_label_count = y_values[label]

	return y_values

# Description: returns a new truncated data array with rows with the x_label value 
			# and the x_label column eliminated 
def new_data_array(data, x_label, x_value):

	if (data.size != 0):
		new = data[0] 
		xcol_ind = np.where(data[0] == x_label)[0][0]
		unique_y, counts_y = np.unique(data[1:, xcol_ind], return_counts = True)
		y_values = dict(zip(unique_y, counts_y))
		y_val_list = list()
		for each_val in y_values:
			y_val_list.append(each_val)

		#print(y_val_list)

		for row in data[1:]:
			if (row[xcol_ind] == x_value):
				new = np.vstack([new, row])
	new = np.delete(new, xcol_ind, axis = 1)
	return new


def train_tree(data, tree_node, max_depth):

	# train a stump, tree with only one level

	# Base Case: Max depth is zero
	if (max_depth == 0):
		# ******** #
		# max vote classifier
		tree_node.y_values = max_vote_classifier(data)
		# return tree_node

	# Base Case: Tree depth >= max_depth
	#print(type(max_depth))
	#print(type(tree_node.depth))
	attributes_left = np.size(data, 1);

	#if (data.size)

	if (max_depth <= tree_node.depth):
		return tree_node

	#print("att left", attributes_left)
	if (max_depth > tree_node.depth and attributes_left <= 1):
		return tree_node
	#print('data', data)

	# Calculate the information gain for all x features I(Y; X1), I(Y; X2), I(Y; X3) ...
	mutual_infos = dict()
	y_col_i = (len(data[0]) - 1)
	x_names = data[0][0: y_col_i]

	for x_name in x_names:
		mutual_infos[x_name] = I(data, x_name)

	# If all mutual infos are less than zero, return tree_node
	# Return tree if none of the mutual_info is greater than zero:
	zero = False
	for each in mutual_infos:
		if (mutual_infos[each] <= 0):
			zero = True

	if (zero): return tree_node

	# recursively creates sub-trees for non-leaf node
	else:
		#calculate the max mutual info x

		max_mutual_info_label = ""
		max_mutual_info_val = 0
		for each in mutual_infos: 
			if (mutual_infos[each] > max_mutual_info_val):
				max_mutual_info_label = each
				max_mutual_info_val = mutual_infos[each]

		tree_node.val = max_mutual_info_label
		
		#=========================================================================================#
		# SPLITTING THE ARRAY IN LEFT AND RIGHT
		#=========================================================================================#

		# create left subtree (one value)
		left_node = Node("", tree_node.depth + 1)
		
		xcol_ind = np.where(data[0] == max_mutual_info_label)[0][0]
		unique_y, counts_y = np.unique(data[1:, xcol_ind], return_counts = True)
		#y_values = list(unique_y)
		y_values = dict(zip(unique_y, counts_y))
		y_val_list = list()
		
		for each_val in y_values:
			y_val_list.append(each_val)
		
		y_value1 = y_val_list[0]
		y_value2 = y_val_list[1]

		tree_node.leftval = y_value1
		tree_node.rightval = y_value2

		new_left_data = new_data_array(data, max_mutual_info_label, y_value1)

		tree_node.left = train_tree(new_left_data, left_node, max_depth)

		unique_l_y, counts_l_y = np.unique(new_left_data[1:, len(new_left_data[0]) -1 ], return_counts = True)
		l_y_values = dict(zip(unique_l_y, counts_l_y))
		#print(tree_node.val, y_value1, l_y_values)
		tree_node.left.y_values = l_y_values

		# create right subtree (another value)
		right_node = Node("", tree_node.depth + 1)

		#new_right_data = new 2D data array for N of max mutual_info node
		new_right_data = new_data_array(data, max_mutual_info_label, y_value2)
		tree_node.right = train_tree(new_right_data, right_node, max_depth)
		
		# Printing the decision tree
		unique_r_y, counts_r_y = np.unique(new_right_data[1:, len(new_right_data[0]) -1 ], return_counts = True)
		r_y_values = dict(zip(unique_r_y, counts_r_y))
		#print(tree_node.val, y_value2, r_y_values)
		tree_node.right.y_values = r_y_values

		return tree_node

# Description: Calculate error of the predicted labels 
				# with respect to the given labels (ground truth)
def error(right_lst, lst):
	

	s = right_lst.size

	### Training Error Rate ###
	error_count = 0
	for i in range(0, s):
		#print(lst[i], right_lst[i])
		if (right_lst[i] != lst[i]):
			error_count+=1

	return (error_count/s)

#
def max_yval(dic):
	max_count = None
	max_label = None
	#print(dic)
	for each_label in dic:
		#print("yes")
		if ((max_label == None) or (dic[each_label] > max_count)):
			max_label = each_label
			max_count = dic[each_label]
	return max_label


# Takes in the learned tree and a 1D row array
# returns the prediction based on 
def row_pred(l_tree, label_array, row_array):
	#print(l_tree.val)
	# Base Case
	#print("tree value", l_tree.val)
	if (l_tree.val == ""):
		y_values = l_tree.y_values
		#print(y_values)
		#print("max", max_yval(l_tree.y_values))
		return max_yval(l_tree.y_values)
	# Recursive Case
	else:
		# i: index in the row that is equal to the l_tree.val
		i = np.where(label_array == l_tree.val)[0][0]

		x_label_value = row_array[i]

		if (x_label_value == l_tree.leftval):
			return row_pred(l_tree.left, label_array, row_array)
		elif (x_label_value == l_tree.rightval):
			return row_pred(l_tree.right, label_array, row_array)

# Description: generates predicted labels based on a tree
# Inputs: learned decision tree and data as inputs
# Returns: generates predicted labels
def generate_pred(l_tree, data_in):
	#print(data_in)
	output_list = list()
	label_array = data_in[0]
	for each_row in data_in[1:]:
		prediction = row_pred(l_tree, label_array, each_row)
		output_list.append(prediction)
	#print(output_list)
	return output_list

def print_tree(trained_tree):
	print(trained_tree.y_values)
	printPreorder(trained_tree)
	return

# Description: A function to do preorder tree traversal
def printPreorder(trained_tree):

	slashes = trained_tree.depth + 1
	for i in range(0, slashes): 
		print("| ", end = '')
	print(trained_tree.val, "=", trained_tree.rightval, ":", trained_tree.right.y_values)

	#print(trained_tree.right.val == None)
	if (trained_tree.right.val != ""):
		printPreorder(trained_tree.right)

	slashes = trained_tree.depth + 1
	for i in range(0, slashes): 
		print("| ", end = '')
	print(trained_tree.val, "=", trained_tree.leftval, ":", trained_tree.left.y_values)

	if (trained_tree.left.val != ""):
		#print(trained_tree.left.depth)
		printPreorder(trained_tree.left)

def save_labels(lst, file_name):
	file = open(file_name, mode = "w")
	#write = file.writelines(lst)
	with file as outfile:
		for s in lst:
			outfile.write("%s\n" %s)
	file.close()

def save_metrics(train_error, test_error, file_name):
	file = open(file_name, mode = "w")
	value1 = "error(train): " + str(train_error)
	value2 = "error(test): " + str(test_error)
	lst = list()
	lst.append(value1)
	lst.append(value2)
	with file as outfile:
		for s in lst:
			outfile.write("%s\n" %s)
	file.close()

# Description: calculates the mutual information
# Helper functions: H_y, H_cond
# Inputs: data and the x feature name
# Returns: mutual information value of I(X feature;Y)
def I(data, x_name):
	#print("H_cond", H_cond(data, x_name))
	I = H_y(data) - H_cond(data, x_name)
	return I

if __name__ == '__main__':
	train_input = sys.argv[1]
	test_input = sys.argv[2]
	max_depth = sys.argv[3]
	train_out = sys.argv[4]
	test_out = sys.argv[5]
	metrics_out = sys.argv[6]

	# num of columns = x_features + y label = len(data[0])

	# Processing the input train file
	train_i = open(train_input, "r")
	train_in = np.genfromtxt(fname = train_input, delimiter="\t", dtype = str)

	# TRAINING THE TRAIN INPUTS
	unique_y, counts_y = np.unique(train_in[1:, (len(train_in[0]) - 1)], return_counts = True)
	y_values = dict(zip(unique_y, counts_y))

	tree_node = Node("", 0)
	tree_node.y_values = y_values
	trained_tree = train_tree(train_in, tree_node, int(max_depth))

	#trainedd = np.array([ ['Anti_satellite_test_ban', 'Export', 'Party'], 
	#					  ['y', 'n', 'republican']])
	
	print_tree(trained_tree)

	train_out_list = generate_pred(trained_tree, train_in)
	save_labels(train_out_list, train_out)
	#print(train_out_list)
	#print(trained_tree.right.leftval)

	# Processing the input TRAIN files
	trow, tcol = train_in.shape
	train_right_labels = train_in[:, tcol-1][1:]

	
	# Processing the  TEST files
	test_i = open(test_input, "r")
	test_in = np.genfromtxt(fname = test_input, delimiter="\t", dtype = str)
	test_out_list = generate_pred(trained_tree, test_in)
	save_labels(test_out_list, test_out)
	#print(test_out_list)

	test_r, test_col = test_in.shape
	test_right_labels = test_in[:, test_col-1][1:]
	

	### Testing Error Rate ###

	# Calculate errorr for train and test data
	train_error = error(train_right_labels, train_out_list)
	test_error = error(test_right_labels, test_out_list)
	print("train error: ", train_error)
	print("test error: ", test_error)
	save_metrics(train_error, test_error, metrics_out)
















	