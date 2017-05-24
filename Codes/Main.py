from Data import MLData
from Tree import MLTree
import sklearn

"""
Data usage:
    Raw: raw dataframe
    Training: training dataset
    Validation: validation dataset
    Testing: testing dataset
    
    SplitSize([40, 30]) change split size for training, validation & testing
                            distribution (example: 40%, 30%, 30%).
    shuffle() Shuffle the data
    unshuffle() Restore original data (not shuffled)
"""

# Load data
data = MLData('../Data/DataSet.csv')

# Create tree
tree = MLTree()
tree.setTrainingData(data.Training)
tree.Type="DecisionTree"
tree.ignored_features = ["machin"]

print("Creating tree")
tree.learn()
print("Creating tree visualization")
dot_data = sklearn.tree.export_graphviz(tree.Tree, out_file="out.dot")
#print("Creating dot file")
#graph = pydotplus.graph_from_dot_data(dot_data) 
#print("Creating pdf")
#graph.write_pdf("tree.pdf")

#print(data.Training)