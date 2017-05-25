import matplotlib.pyplot as plt 
import sklearn
import pydotplus

def disp_confusion_mat(conf_mat):
	"""Draw the confusion matrix
	Show confusion matrix in a separate window
	"""
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
	if pdf : 
		dot_data = sklearn.tree.export_graphviz(tree, out_file=None)
		graph = pydotplus.graph_from_dot_data(dot_data) 
		graph.write_pdf("tree.pdf")
	else :
		dot_data = sklearn.tree.export_graphviz(tree, out_file="tree.dot")