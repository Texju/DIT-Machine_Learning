from Data import MLData
from Tree import MLTree
from validation import MLValidation
import display

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
tree.Type="DecisionTreeEntropy"
tree.ignored_features = [
        "industry code",    # duplicate of "major industry code"
        "occupation code",  # duplicate of "major occupation code"
        "detailed household summary in household", # duplicate of "detailed household and family stat" with less info
        ]

print("Creating tree")

#--------------------------------------------
# Hold-out Test Set + Confusion Matrix
#--------------------------------------------
print("Learn ")
decTree = tree.learn()
print("Test ")
#Split the data: 60% training : 40% test set
#instances_train, instances_test, target_train, target_test = train_test_split(train_dfs, tree.Target, test_size=0.4, random_state=0)
#fit the model using just the test set

display.visualize_tree(decTree, True)

validation = MLValidation(tree)

validation.test(data)

print("Accuracy : " + str(validation.accuracy()))
print(validation.confusionMatrix())
display.disp_confusion_mat(validation.confusionMatrix())


"""
tree_select = feature_selection.FeatureSelection(train_dfs, tree.Target)
tree_select = tree_select.select_threshold(0.1)
print(tree_select.dtype.names)
"""

#print("testing")
#validation = MLValidation(tree)

#validation.test(data)