import numpy as np
import math
import random
import sys
################
# Formatting
################
def trim_elems(lst, D):
    if D == 1: 
        new_lst = list()
        for elem in lst: 
            new_lst.append(elem.replace('\n', ''))
        return new_lst
    elif D == 2: 
        new_list = list()
        for line in lst:
            line_lst = list()
            for word in line: 
                line_lst.append(word.replace('\n', ''))
            new_list.append(line_lst)
        return new_list

def tag_list(data_file):
    file1 = open(data_file, "r")
    tag_list = list()
    for tag in file1:
        tag_list.append(tag)
    tag_list_t = trim_elems(tag_list, 1)
    return tag_list_t

def hmmprior_list(data_file):
	file1 = open(data_file, "r")
	tag_list = list()
	for tag in file1:
		tag_list.append(tag)
	tag_list_t = trim_elems(tag_list, 1)
	num_lst = list()
	for value in tag_list_t:
		num_lst.append(float(value))
	return num_lst

def two_D_int_array(data_file):
    file3 = open(data_file, "r")
    two_D_array = list()
    for line in file3:
        #line_lst = list()
        words_list = line.split()
        two_D_array.append(words_list)
    two_D_array_t = trim_elems(two_D_array, 2)

    new_arr = list()
    for val in two_D_array_t:
    	temp = []
    	for line in val:
    		temp.append(float(line))
    	new_arr.append(temp)
    return new_arr

def word_list(data_file):
    file2 = open(data_file, "r")
    word_list = list()
    for word in file2:
        word_list.append(word)
    word_list_t = trim_elems(word_list, 1)
    return word_list_t

def two_D_array(data_file):
    file3 = open(data_file, "r")
    two_D_array = list()
    for line in file3:
        #line_lst = list()
        words_list = line.split()
        two_D_array.append(words_list)
    two_D_array_t = trim_elems(two_D_array, 2)
    return two_D_array_t

def empty_array(test_input):
	empty = []
	for line in test_input:
		ln = []
		#for wrd in line:
			#ln.append("")
		empty.append(ln)
	return empty
##################
# Helper Functions 
##################
def argmax(probs):
	# Probabilities are tuples with (tag_name, prob_value) returns the tag with highest prob
	max_tag_far = None
	max_prob_far = 0
	for i in range(0, len(probs)):
		tag, prob = probs[i]
		if max_tag_far == None: 
			max_tag_far = tag
			max_prob_far = prob
		else: 
			if (max_prob_far < prob):
				max_tag_far = tag
				max_prob_far = prob
	#print(max_prob_far, max_tag_far)
	return max_tag_far 

def max_prob(probs):
	# Probabilities are tuples with (tag_name, prob_value) returns the tag with highest prob
	max_tag_far = None
	max_prob_far = 0
	for i in range(0, len(probs)):
		tag, prob = probs[i]
		if max_tag_far == None: 
			max_tag_far = tag
			max_prob_far = prob
		else: 
			if (max_prob_far < prob):
				max_tag_far = tag
				max_prob_far = prob
	#print(max_prob_far, max_tag_far)
	return max_prob_far

def parse_line(line, word_lst):
	words = list()
	for word in line: 
		word_split = word.split("_")
		wrd = word_split[0]
		word_index = word_lst.index(wrd)
		words.append(word_index)
	return words

# Initializes matrix of size total number of states * length of line
def init_matrix(line, tags):
	mat = list()
	line_len = len(line)
	tags_len = len(tags)
	for tag in tags: 
		mat.append([])
	return mat

# Takes in a list of tuples
def find_max_arg(lst):

	return

# Takes in a list of tuples
def find_max_value(lst):
	
	max_tag = None
	max_val = None

	for i in range(len(lst)):
		tag_str, summ = lst[i]
		if (max_val == None):
			max_val = summ
			max_tag = tag_str
		elif (max_val < summ): 
			max_val = summ
			max_tag = tag_str

	return max_tag, max_val

def max_prob(lst):
	max_val = None
	max_ind = None
	for i in range(0, len(lst)):
		if max_val == None: 
			max_val = lst[i]
			max_ind = i
		elif max_val < lst[i]: 
			max_val = lst[i]
			max_ind = i 
	# returns the index of the tag and the associated probability value
	return max_val, max_ind



#################
# Predictions
#################
def predict(test_input_lst, tag_lst, word_lst, hmmprior_lst, hmmemit_lst, hmmtrans_lst):

	predictions = list() # 2d array

	for line in test_input_lst:
		line_v = parse_line(line, word_lst)
		# change the line index_to_tag a vector with the corresponding word indices

		# Initialize the lw matrix and b matrix
		lw = init_matrix(line, tag_lst)
		b = init_matrix(line, tag_lst)
		# lw stores the path log probabilities
		# b stores the backpointers

		for word_i in range(len(line_v)):

			for tag_j in range(len(tag_lst)):

				if (word_i == 0): 
					
					# Compute the initial probabilities (prior * emission)
					prior = hmmprior_lst[tag_j]
					emission = hmmemit_lst[tag_j][line_v[word_i]]

					# Store the prob in the lw matrix 
					lw[tag_j].append(math.log(prior) + math.log(emission))

					# In the b matrix, store in the original order
					b[tag_j].append(tag_lst[tag_j])

				elif (word_i > 0): 
					# Compute the other transition stuff probs ws

					# Compute the emission prob (word given tag_j)
					emission = hmmemit_lst[tag_j][line_v[word_i]]

					# Calculate the max of trans + w(i-1) and store the argmax (tag)
					store_prod = list()

					for tag_k in range(len(tag_lst)): 
						# For each tag_k compute P(tag_j/tag_k) * w(i-1)(tag_k) -> from the table
						transition = hmmtrans_lst[tag_k][tag_j]
						w_i_minus_one = lw[tag_k][word_i - 1]

						#print(transition, w_i_minus_one)
						sm = math.log(transition) + w_i_minus_one
						store_prod.append((tag_lst[tag_k], sm))
					
					# take the max of that value 
					max_tag, max_value = find_max_value(store_prod)

					# Store the argmax (tag_k which gives the max) in the b matrix!
					lw[tag_j].append(math.log(emission) + max_value)
					b[tag_j].append(max_tag)

		# Backpointing
		predict_line_tag = [0] * len(line_v)

		# Look at the the last column of lw and assign the prediction of
		# the state (row) that has the highest value to the last word in the line_v
		lw_last_col_index = len(lw[0]) - 1
		
		assert(line[lw_last_col_index] == line[-1])

		word_last_col_ex = line[lw_last_col_index]
		word_last_col, wrd_tag = word_last_col_ex.split("_")
	
		# to find the highest value for the last word in the line_v based on lw_last_col_index
		all_prob_last_col = list()
		for tag_i in range(len(tag_lst)):
			all_prob_last_col.append(lw[tag_i][lw_last_col_index])

		max_prob_val, max_prob_tag_ind = max_prob(all_prob_last_col)

		tag_value = tag_lst[max_prob_tag_ind]
		#print(tag_value)
		string1 = word_last_col + "_" + tag_value
		predict_line_tag[-1] = string1

		pointer = max_prob_tag_ind
		for p in range(len(line_v)-1, 0, -1):
			# Now, look at the assignment in the b matrix at the last word, at the state predicted
			# for the second last word predict that state, and point to until the end
			back_tag_value = b[pointer][p]
			wrd_exp = line[p-1]
			
			word_val, wrd_tag = wrd_exp.split("_")
			string = word_val + "_" + back_tag_value
			predict_line_tag[p-1] = string
			pointer = tag_lst.index(back_tag_value)

		predictions.append(predict_line_tag)

	return predictions
	

#################
# Output Stuff
#################
def output_pred(path, lst):
	file = open(path, mode = "w")
	with file as outfile: 
		for i in range(0, len(lst)):
			line = lst[i]
			for j in range(0, len(line)): 
				word = lst[i][j]
				if (len(line) - 1 == j): 
					outfile.write(word + "\n")
				else: 
					outfile.write(word + " ")
	file.close()

def output_metrics(path, acc):
	file = open(path, mode = "w")
	with file as outfile: 
		string = "Accuracy: " + str(acc)
		outfile.write(string)
	file.close()

def create_metrics(pred, input):
	total = 0
	correct = 0
	for i in range(0, len(pred)):
		for j in range(0, len(pred[i])):
			total += 1
			if (pred[i][j] == input[i][j]):
				correct += 1
	return correct/total

if __name__ == '__main__':
    test_input = sys.argv[1]
    index_to_word  = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    test_input_lst = two_D_array(test_input)

   	# Put index to tag in a list
    tag_lst = tag_list(index_to_tag)

    # Put index to word in a list
    word_lst = word_list(index_to_word)

    # Extract hmm prior, hmm emit, hmm trans
    hmmprior_lst = hmmprior_list(hmmprior)
    #hmmprior_lst[0]
    hmmemit_lst = two_D_int_array(hmmemit)
    hmmtrans_lst = two_D_int_array(hmmtrans)

   # print(test_input_lst)
    predictions = predict(test_input_lst, tag_lst, word_lst, 
    					hmmprior_lst, hmmemit_lst, hmmtrans_lst)
    
    #print(predictions)

    accuracy = create_metrics(predictions, test_input_lst)
    print("accuracy", accuracy)


    output_pred(predicted_file, predictions)
    output_metrics(metric_file, accuracy)









