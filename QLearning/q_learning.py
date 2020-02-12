from environment import MountainCar
import numpy as np
import math
import random 
import sys

np.random.seed()

def update_w(w, s, a, constant):

	keys = list(s.keys())

	w1 = np.copy(w)

	for val in keys:

		w1[val][a] = w1[val][a] - constant*s[val]

	return w1

def update_b(b, s, a, constant):
	
	b1 = b - constant

	return b1

def sparse_dot_prod(s, wa): 

	product = 0.0
	for i, v in s.items():  
		product += v * wa[i]
	return product

def print_labels(lst, path):
	file = open(path, mode = "w")
	with file as outfile: 
		for val in lst: 
			string1 = str(val) + "\n"
			outfile.write(string1)
	file.close()

def q(s, a, w, b):
	prod = sparse_dot_prod(s, w[:, a]) + b
	return prod

def epsilon_greedy(epsilon, a_qvals):

	action = -1

	x = random.uniform(0, 1)
	
	if (x >= epsilon):

		aq = np.array(a_qvals)
		action = np.argmax(aq)

	elif (x < epsilon): 
		action = random.randint(0, 2)

	#print("action", action)
	return action

def combine_w_b(bias, weight):
	
	lst = []
	lst.append(bias)
	row, col = weight.shape

	for i in range(0, row):
		for j in range(0, col):
			lst.append(weight[i][j])
	return lst

def main(args):
	# parse input
	mode = args[1]
	weight_out = args[2]
	returns_out = args[3]
	episodes = int(args[4])
	max_iterations = int(args[5])
	epsilon = float(args[6])
	gamma = float(args[7])
	learning_rate = float(args[8])

	# Initialize a mountain car, parameters?
	new  = MountainCar(mode = mode)

	weight = np.array([]) 
	if mode == 'tile':
		weight = np.zeros((2048, 3))
	elif mode == 'raw':
		weight = np.zeros((2, 3))

	b = 0
	total_rewards = []

	# episode < episodes:          
	for i in range(0, episodes):

		s = new.reset()

		#print("s beginning", s)
		reward_sum = 0
	# 	iter < max_ite:
		#print(weight)
		for j in range(0, max_iterations):
			
			#print("state", s)
			q1 = q(s, 0, weight, b)
			q2 = q(s, 1, weight, b)
			q3 = q(s, 2, weight, b)
			a_qvals = [q1, q2, q3]
			#print(a_qvals)
			#print("a_qvals", a_qvals)
			o_action = np.argmax(a_qvals)

			#a = epsilon_greedy(epsilon, a_qvals)
			actions = np.array([0, 1, 2])
			rand_action = np.random.choice(actions)
			bounds = [epsilon, 1-epsilon]
			a = np.random.choice([rand_action, o_action], p = bounds)
			a = int(a)
			#print("action: ", a)

			q_value = q(s, a, weight, b)

			(s_p, r, done) = new.step(a)

			q1 = q(s_p, 0, weight, b)
			q2 = q(s_p, 1, weight, b)
			q3 = q(s_p, 2, weight, b)
			a_qvals = [q1, q2, q3]
			max_q = np.amax(a_qvals)
				
			constant = learning_rate * (q_value - (r + gamma*max_q))

			weight_new = update_w(weight, s, a, constant)
			b_new = update_b(b, s, a, constant)

			weight = weight_new
			b = b_new
			s = s_p
			reward_sum += r

			if done:

				break

		total_rewards.append(reward_sum)

	weight_bias = combine_w_b(b, weight)

	print_labels(weight_bias, weight_out)
	print_labels(total_rewards, returns_out)

if __name__ == "__main__":
    main(sys.argv)