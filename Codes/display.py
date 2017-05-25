import matplotlib.pyplot as plt 
import sklearn
import pydotplus



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
	path = "../Results/tree.pdf"
	if tree.Type == "DecisionTreeEntropy" or tree.Type == "DecisionTreeGini" : 
		if pdf : 
			dot_data = sklearn.tree.export_graphviz(tree, out_file=None)
			graph = pydotplus.graph_from_dot_data(dot_data) 
			graph.write_pdf(path)
			print("pdf is here : "+path)
		else :
			dot_data = sklearn.tree.export_graphviz(tree, out_file="tree.dot")
	else :
		print("Impossible you don't use a DecisionTree sorry")
		print('Alright')

def comparasion_result():
	# http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py
	pass