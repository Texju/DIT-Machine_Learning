# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:00:15 2017

@author: Julien Couillard & Jean Thevenet
"""

import matplotlib.pyplot as plt 
import sklearn
import pydotplus
from sklearn.calibration import calibration_curve

# TODO : http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py 
# TODO : http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html#sphx-glr-auto-examples-ensemble-plot-forest-iris-py



def disp_confusion_mat(conf_mat):
	"""Draw the confusion matrix
	Show confusion matrix in a separate window
	"""
	print("Show confusion matrix")
	plt.matshow(conf_mat)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	
def visualize_tree(tree, pdf = True):
	"""Create dot or pdf file to draw the tree
	You need to put in input DecisionTree  
	"""
	print("Creating out for decision tree")
	path = "../Results/tree"
	if tree.Type == "DecisionTreeEntropy" or tree.Type == "DecisionTreeGini" : 
		if pdf : 
			dot_data = sklearn.tree.export_graphviz(tree, out_file=None)
			graph = pydotplus.graph_from_dot_data(dot_data) 
			graph.write_pdf(path+".pdf")
			print("pdf is here : "+path)
		else :
			dot_data = sklearn.tree.export_graphviz(tree, out_file=path+".dot")
	else :
		print("Impossible you don't use a DecisionTree sorry")
		print('Alright')

def comparasion_result(dict_classifier, data):
	"""Display a comparaison of differents classifiers
	"""
	plt.figure(figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	ax2 = plt.subplot2grid((3, 1), (2, 0))

	ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
	for name, clf in dict_classifier.items():
		instances_test, target_test = clf[0].prepareData(data.Testing)
		if hasattr(clf[0].Tree, "predict_proba"):
			prob_pos = clf[0].Tree.predict_proba(instances_test)[:, 1]
		else:  # use decision function
			prob_pos = clf[0].Tree.decision_function(instances_test)
			prob_pos = \
				(prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
		fraction_of_positives, mean_predicted_value = \
				calibration_curve(target_test, prob_pos, n_bins=10)

		ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
					label=name)

		ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
					histtype="step", lw=2)

	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title('Calibration plots  (reliability curve)')

	ax2.set_xlabel("Mean predicted value")
	ax2.set_ylabel("Count")
	ax2.legend(loc="upper center", ncol=2)

	plt.tight_layout()
	plt.show()