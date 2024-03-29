import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################	
			branches = np.array(branches)
			prob = branches / np.sum(branches, axis=0)
			etp = np.array([[-i * np.log2(i) if i > 0 else 0 for i in x] for x in prob])
			w = np.sum(branches, axis=0) / np.sum(branches)
			
			return np.sum(np.sum(etp, axis=0)*w)
			
		features = np.array(self.features)
		entropy = []
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		#		split the node, add child nodes
		############################################################
			feature = features[:,idx_dim]
			branches = np.zeros((self.num_cls, len(np.unique(feature))))
			for i, cls in enumerate(np.unique(feature)):
				label = np.array(self.labels)[np.where(feature == cls)]
				for j in label:
					branches[j, i] += 1
			entropy.append(conditional_entropy(np.array(branches)))

		
		self.dim_split = np.argmin(entropy)
		feature = features[:,self.dim_split]
		self.feature_uniq_split = np.unique(feature).tolist()
		if len(np.unique(feature)) > 1:
			for i in self.feature_uniq_split:
				index = np.where(feature==i)
				child = TreeNode(features[index].tolist(),np.array(self.labels)[index].tolist(), self.num_cls)
				self.children.append(child)
		else:
			self.splittable = False

		
		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return


	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			# feature = feature[:self.dim_split] + feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max


