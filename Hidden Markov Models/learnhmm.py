import math
import numpy as np
import random
import sys
import copy 

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

def sum_list(counts, D, row_i): 
    if D == 1: 
        total = 0
        for elem in counts:
            total += elem
        return total
    elif D == 2: 
        total = 0
        for elem in counts[row_i]:
            total += elem
        return total

def divide_by(total, lst):
    new_list = list()
    for elem in lst:
        new_list.append(elem/total)
    return new_list

def create_hmmprior(tag_lst, train_in_arr):
    hmmprior_count = [1] * len(tag_lst)

    for line in train_in_arr:
        first_word = line[0]
        word_split = first_word.split("_")
        ind = tag_lst.index(word_split[-1])
        hmmprior_count[ind] += 1

    total = sum_list(hmmprior_count, 1, 0)
    hmmprior = divide_by(total, hmmprior_count)

    return hmmprior

def empty_hmmemit(row, col):
    lst = list()
    for tag in range(0, row):
        lst_tag = list()
        for word in range(0, col):
            lst_tag.append(1)
        lst.append(lst_tag)
    return lst  

def create_hmmemit(tag_lst, word_lst, train_in_arr):
    hmmemit = copy.copy(empty_hmmemit(len(tag_lst), len(word_lst)))
    for line in train_in_arr: 
        for word in line: 
            word_split = word.split("_")
            wrd = word_split[0]
            tag = word_split[1]
            wrd_index = word_lst.index(wrd)
            tag_index = tag_lst.index(tag)
            hmmemit[tag_index][wrd_index] += 1
    new_emit = list()
    for i in range(0, len(hmmemit)): 
        row_total = sum_list(hmmemit, 2, i)
        new_emit.append(divide_by(row_total, hmmemit[i]))
    return new_emit

def empty_trans(row):
    lst = list()
    for tag in range(0, row):
        lst_tag = list()
        for tag in range(0, row):
            lst_tag.append(1)
        lst.append(lst_tag)
    return lst

def create_hmmtrans(tag_lst, word_lst, train_in_arr):
    hmmtrans = empty_trans(len(tag_lst))
    for line in train_in_arr: 
        prev_tag = ''
        for i in range(0, len(line)): 
            if i == 0: 
                word_split = line[i].split("_")
                tag = word_split[1]
                prev_tag = tag
                print("i", i, "prev", prev_tag)
            elif i != 0: 
                print("i", i, "prev", prev_tag)
                word_split = line[i].split("_")
                curr_tag = word_split[1]
                row_index = tag_lst.index(prev_tag)
                column_index = tag_lst.index(curr_tag)
                hmmtrans[row_index][column_index] += 1
                prev_tag = curr_tag

    new_trans = list()
    for j in range(0, len(hmmtrans)):
        row_total = sum_list(hmmtrans, 2, j)
        new_trans.append(divide_by(row_total, hmmtrans[j]))
    return new_trans

def save_1D(lst, path):
    file = open(path, mode = "w")

    with file as outfile: 
        for val in lst: 
            string  = str(val) + "\n"
            outfile.write(string)
    file.close()
    return 

def save_2D(lst, path):
    file = open(path, mode = "w")

    with file as outfile: 
        for row in lst: 
            for val in row: 
                string = str(val) + " "
                outfile.write(string)
            outfile.write("\n")
    file.close()
    return

if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    # Put index to tag in a list
    tag_lst = tag_list(index_to_tag)

    # Put index to word in a list
    word_lst = word_list(index_to_word)

    # Put train input in a 2D array
    train_in_arr = two_D_array(train_input)

    #print(tag_lst)
    #print(word_lst)
    #print(train_in_arr)

    # Create hmmprior
    hmmprior_lst = create_hmmprior(tag_lst, train_in_arr)
    #print(hmmprior_lst)

    # Create hmmemmit
    hmmemit_lst = create_hmmemit(tag_lst, word_lst, train_in_arr)
    #print(hmmemit_lst)
    #emit_np = np.array(hmmemit)

    # Create hmmtrans
    hmmtrans_lst = create_hmmtrans(tag_lst, word_lst, train_in_arr)
    #print(hmmtrans_lst)

    save_1D(hmmprior_lst, hmmprior)
    save_2D(hmmemit_lst, hmmemit)
    save_2D(hmmtrans_lst, hmmtrans)










