from Data import MLData
from Tree import MLTree
import feature_selection
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

# http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
"""
for each model created your program should run a 5 fold-crossvalidation and output the accuracy score for each fold
"""

# Load data
data = MLData('../Data/DataSet.csv')

# Create tree
tree = MLTree()
tree.setTrainingData(data)
tree.Type="DecisionTree"
tree.ignored_features = [
        "industry code",    # duplicate of "major industry code"
        "occupation code",  # duplicate of "major occupation code"
        "detailed household summary in household", # duplicate of "detailed household and family stat" with less info
        ]

print("Creating tree")
train_dfs = tree.prepareData()
tree_select = feature_selection.FeatureSelection(train_dfs, tree.Target)
tree_select = tree_select.select_threshold(0.1)
print(tree_select.dtype.names)
#print("Creating tree visualization")
#dot_data = sklearn.tree.export_graphviz(tree.Tree, out_file="out.dot")
#print("Creating dot file")
#graph = pydotplus.graph_from_dot_data(dot_data) 
#print("Creating pdf")
#graph.write_pdf("tree.pdf")

#print(data.Training)