import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		h = np.zeros(len(features))
		for i in range(self.T):
			h += np.array(self.clfs_picked[i].predict(features))*self.betas[i]
		h = np.sign(h)
		return h.tolist()

		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		features = np.array(features)
		N = features.shape[0]
		d = np.full(N, 1/N)
		for i in range(self.T):
			errors = []
			for i in range(len(list(self.clfs))):
				errors.append(np.dot(d, np.not_equal(labels,list(self.clfs)[i].predict(features))))
			e_t = min(errors)
			h = errors.index(e_t)
			self.clfs_picked.append(list(self.clfs)[h])
			preds = [int(x) for x in np.not_equal(labels, list(self.clfs)[h].predict(features))]
			preds = [x if x==1 else -1 for x in preds]
			beta = 0.5 * np.log((1-e_t)/e_t)
			self.betas.append(beta)

			d = np.multiply(d, np.exp([x*beta for x in preds]))
			d /= np.sum(d)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	