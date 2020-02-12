from __future__ import print_function
import sys
import string
import math
import numpy as np

def input_2_dict(datafile):
	file1 = open(datafile, "r")
	
	dictionary = dict()
	for word in file1:
		values = word.split()
		wrd = values[0]
		indx = values[1]
		dictionary[wrd] = indx
	return dictionary

def model1(datafile, dic, outfile):

	file1 = open(datafile, "r")
	index = 0
	data_list = list()

	for line in file1:
		index += 1
		values = line.split("\t")
		y_label = values[0]
		review = values[1]

		line_list = list()
		line_list.append(y_label)

		words = review.split()
		line_dict = dict()

		for word in words:
			word_index = dic.get(word, -1)
			if (word_index != -1):
				if (line_dict.get(word_index, -2) == -2):
					line_dict[word_index] = 1

		line_list.append(line_dict)
		data_list.append(line_list)

	save_file(data_list, outfile)
	return data_list

def model2(datafile, dic, outfile):
	file1 = open(datafile, "r")
	index = 0
	data_list = list()

	for line in file1:
		index += 1
		values = line.split("\t")
		y_label = values[0]
		review = values[1]

		line_list = list()
		line_list.append(y_label)
		words = review.split()
		line_dict = dict()

		for word in words:
			word_index = dic.get(word, -1)
			if (word_index != -1):
				if (line_dict.get(word_index, -2) == -2):
					line_dict[word_index] = 1
				else:
					line_dict[word_index] += 1
		line_list.append(line_dict)
		data_list.append(line_list)
	
	# Trimming

	trim_list = list()

	for review in data_list:
		y_label = review[0]
		word_count = review[1]
		trim_dict = dict()
		trim_line_lst = list()
		trim_line_lst.append(y_label)

		for word_index in word_count:
			count = word_count[word_index]
			if count < 4:
				if (trim_dict.get(word_index, -3) == -3):
					trim_dict[word_index] = 1

		trim_line_lst.append(trim_dict)
		trim_list.append(trim_line_lst)

	save_file(trim_list, outfile)
	return trim_list

def save_file(lst, file_name):

	file = open(file_name, mode = "w")

	with file as outfile:
		for i in range(0, len(lst)):
			review = lst[i]
			y_label = review[0]
			words = review[1]
				
			review_str = y_label + "\t"
			for each_word in words:
				word_str = each_word + ":" + str(words[each_word]) + "\t"
				review_str += word_str
			
			outfile.write(review_str + "\n")

	file.close()

if __name__ == '__main__':
	train_input = sys.argv[1]
	validation_input = sys.argv[2]
	test_input = sys.argv[3]
	dict_input = sys.argv[4]
	formatted_train_out = sys.argv[5]
	formatted_valid_out = sys.argv[6]
	formatted_test_out = sys.argv[7]
	feature_flag = sys.argv[8]
	
	dictionary = input_2_dict(dict_input)

	if feature_flag == '1': 
		train_out_lst = model1(train_input, dictionary, formatted_train_out)
		validation_out_lst = model1(validation_input, dictionary, formatted_valid_out)
		test_out_lst = model1(test_input, dictionary, formatted_test_out)

	if feature_flag == '2':
		train_out_lst = model2(train_input, dictionary, formatted_train_out)
		validation_out_lst = model2(validation_input, dictionary, formatted_valid_out)
		test_out_lst = model2(test_input, dictionary, formatted_test_out)

