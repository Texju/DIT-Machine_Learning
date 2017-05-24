from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

class FeatureSelection(object):
	def __init__(self, data, target):
		self.data = data
		self.target = target 

	def selection_from_model_l1(self):
		lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(self.data, self.target)
		model = SelectFromModel(lsvc, prefit=True)
		return model.transform(self.data)
	
	def selection_chi2(self, number_node):
		""" Return a new dataset with the number of node 
			It's for classification
		"""
		return SelectKBest(chi2, k=number_node).fit_transform(self.data, self.target)
	
	def selection_classif(self):
		return f_classif(self.data, self.target)
		