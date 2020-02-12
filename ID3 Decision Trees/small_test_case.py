from __future__ import print_function
import sys
import string
import math
import numpy as np

data1 = np.array([["y"]])

#print(data1.size)

data = np.array([ [ "b", "c", "d"],
			  [ 1, 0, 1], 
             [0, 1, 1], 
             [1, 1, 1] ])

row, col = data.shape
print(row)
print(col)
#data = np.array([])
#data = np.array([ ["a", "b", "c", "d"]])
data = np.array([ ["c", "d"],
			      [0,     1] ])
 
if data.size != 0:
	xcol_ind = np.where(data[0] == "c")[0][0]
	unique_y = np.unique(data[1:, xcol_ind], return_counts = False)
	y_values = list(unique_y)

	#y_value1 = y_values[0]
	#y_value2 = y_values[1]
	print("yvalues", y_values)
	new = data[0]

	print("new", new)

	for row in data[1:]:
		if (row[xcol_ind] == y_values[0]):
			#print("new", new)
			print("row", row)
			new = np.vstack([new, row])

	new = np.delete(new, xcol_ind, axis = 1)
	print(new)