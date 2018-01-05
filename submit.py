import random
import pickle as pkl
import argparse
import csv
import numpy as np
import sys
import copy
from scipy import stats

'''
TreeNode represents a node in your decision tree
TreeNode can be:
	- A non-leaf node: 
		- data: contains the feature number this node is using to split the data
		- children[0]-children[4]: Each correspond to one of the values that the feature can take

	- A leaf node:
		- data: 'T' or 'F' 
		- children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''


# DO NOT CHANGE THIS CLASS
class TreeNode():
	def __init__(self, data='T', children=[-1] * 5):
		self.nodes = list(children)
		self.data = data

	def save_tree(self, filename):
		obj = open(filename, 'w')
		pkl.dump(self, obj)


# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [], [], []
	with open(ftrain, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			rw = map(int, row[0].split())
			Xtrain.append(rw)

	with open(ftest, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			rw = map(int, row[0].split())
			Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			rw = int(row[0])
			Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


num_feats = 274

#calculate target entropy
def calc_entropy(S):
	values = np.unique(S)
	total_len = len(S)

	if total_len == 0 or len(values) == 0:
		return 0

	probs = {}
	entropy = 0
	for value in values:
		probs[value] = float(S.count(value)) / total_len;

	for key, prob in probs.iteritems():
		if prob == 0:
			entropy -= 0
		else:
			entropy -= prob * np.log2(prob);
	return entropy

def select_best_feature(visited, train_data, labels, target_entropy):
	max_gain = 0
	best_feature = None

	for feature in range(0, num_feats):
		if feature in visited:
			continue
		gain = calc_info_gain(train_data[:, feature], labels, target_entropy)
		if (gain > max_gain):
			max_gain = gain
			best_feature = feature
	return best_feature

def make_leaf(labels):
	if labels.count(0) > labels.count(1):
		leaf = 'F'
	else:
		leaf = 'T'
	node = TreeNode(leaf, [])
	return node

def calc_info_gain(train_column, labels, target_entropy):
	values = np.unique(train_column)
	total_len = len(train_column)
	entropy = 0
	for value in values:
		temp = []
		for idx, item in enumerate(train_column):
			if (item == value):
				temp.append(labels[idx])
		entropy += ((float(len(temp)) / total_len) * calc_entropy(temp))
	return target_entropy - entropy

def create_tree(train_data, labels, processed, depth = 0):
	target_entropy = calc_entropy(labels)
	if target_entropy <= 0:
		return make_leaf(labels)

	best_feature = select_best_feature(processed, train_data, labels, target_entropy)

	if(best_feature is None):
		return make_leaf(labels)

	node = TreeNode(best_feature, [])

	processed.append(best_feature)
	best_feature_column = train_data[:, best_feature]

	#for chi square evaluation
	temp_train_data = list()
	temp_train_labels = list()

	for domain in range(1, 6):
		new_labels = list()
		new_train_data = list()
		for key, value in np.ndenumerate(best_feature_column):
			if(value == domain):
				new_labels.append(labels[key[0]])
				new_train_data.append(train_data[key[0]])
		new_train_data = np.array(new_train_data)
		temp_train_data.append(new_train_data)
		temp_train_labels.append(new_labels)

	p_value = calc_pval(temp_train_labels)
	if(p_value < threshold):
		node = make_leaf(labels)
	else:
		for i in range(0, 5):
			child = create_tree(temp_train_data[i], temp_train_labels[i], processed, depth + 1)
			node.nodes.append(child)
	return node

def calc_pval(labels):
	observed_values = list()
	total_length = 0
	for i in range(len(labels)):
		observed_values.append(labels[i].count(1))
		observed_values.append(labels[i].count(0))
		total_length += len(labels[i])

	expected_values = list()
	for i in range(len(labels)):
		expected_values.append(len(labels[i]) * labels[i].count(1)/total_length)
		expected_values.append(len(labels[i]) * labels[i].count(0)/total_length)
	chi, p = stats.chisquare(observed_values, expected_values)
	return p

# A random tree construction for illustration, do not use this in your code!
def create_random_tree(depth):
	if (depth >= 7):
		if (random.randint(0, 1) == 0):
			return TreeNode('T', [])
		else:
			return TreeNode('F', [])

	feat = random.randint(0, 273)
	root = TreeNode(data=str(feat))

	for i in range(5):
		root.nodes[i] = create_random_tree(depth + 1)

	return root


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[
				  0] + '_labels.csv'  # labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']
threshold = pval

tree_name = args['t']

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")
s = create_tree(np.array(Xtrain), Ytrain, [])
s.save_tree(tree_name)
def test(testrow, node):
	if node.data == "T":
		return 1
	elif node.data == "F":
		return 0
	val = testrow[node.data]
	if not node.nodes:
		return None
	if node.nodes[val - 1] == None or node.nodes[val - 1] == -1:
		return None
	else:
		return test(testrow, node.nodes[val - 1])

print("Testing...")
Ypredict = []
tree_cpy = copy.copy(s)
for i in range(0, len(Xtest)):
	Ypredict.append([test(Xtest[i], s)])
Ytest_predict_name = "output.csv"

with open(Ytest_predict_name, "wb") as f:
	writer = csv.writer(f)
	writer.writerows(Ypredict)

print("Output files generated")

print("Output files generated")