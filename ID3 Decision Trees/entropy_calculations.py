''' This program will calculate the label entropy at the root
 and error rate of classifying using a majority vote. '''

from __future__ import print_function
import sys
import string
import math
import numpy as np

if __name__ == '__main__':
	datafile = sys.argv[1]
	outfile = sys.argv[2]

	file1 = open(datafile, "r")
	data = np.genfromtxt(fname = datafile, delimiter="\t", 
							dtype = str)

	#print(data)
	#print(len(data[0]))
	#usecols = [0, 1, 2]
	# array[row, column]
	# row = 0 has 3 columns 0, 1, 2 
	# column 2 is the label
	#print(data[0, 1])

	# Isolates the unique values of y
	# num of columns = x_features + y label = len(data[0])
	unique_y, counts_y = np.unique(data[1:, (len(data[0]) - 1)], return_counts = True)
	y_values = dict(zip(unique_y, counts_y))

	#print(y_values)

	# Counting the total number of labels
	sum_count = 0
	for each_val in y_values: 
		sum_count += y_values[each_val]

	#print(sum_count)

	# Finding the minority label
	min_value = 0
	for each_val in y_values:
		if (min_value == 0): 
			min_value = y_values[each_val]
		elif y_values[each_val] < min_value:
			min_value = y_values[each_val]
	#print(min_value)

	# Calculate the entropy
	entropy = 0
	for each_value in y_values:
		#print(each_value)
		prob = y_values[each_value]/sum_count
		entropy = entropy + prob * math.log2(prob)

	#error is just the probability of the minority class
	error = min_value/sum_count

	#print("entropy: ", -entropy)
	#print("error: ", error)
	# write the answers (entropy and error) to the output file

	file2 = open(outfile, "w")
	string = "entropy: " + str(-entropy) + "\n" + "error: " + str(error) + "\n"
	write = file2.writelines(string)
	file1.close()
	file2.close()