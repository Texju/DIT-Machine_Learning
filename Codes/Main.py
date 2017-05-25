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

print("Load data")
print("______________________________")
# Load data
data = MLData('../Data/DataSet.csv')

# Create tree
tree = MLTree()
tree.setTrainingData(data)
tree.Type="RandomForest"
tree.ignored_features = [ # see notebook for uncommented reasons
        "wage per jour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "instance weight",
        "class of worker",
        "industry code",    # duplicate of "major industry code"
        "occupation code",  # duplicate of "major occupation code"
        "enrolled in edu inst last wk",
        "major industry code",
        "major occupation code",
        "member of a labor union",
        "reason for unemployment",
        "region of previous residence",
        "state of previous residence",
        "detailed household summary in household", # duplicate of "detailed household and family stat" with less info
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "migration prev res in sunbelt",
        "family members under 18",
        "fill inc questionnaire for veteran's admin"
        ]

"""
# DecisionTreeEntropy DecisionTreeGini RandomForest GaussianNB OCSVM SVC LinearRegression
#  , "OCSVM", "SVC", "LinearRegression"
list_classifier = ["DecisionTreeEntropy", "DecisionTreeGini", "RandomForest", "GaussianNB"]
dict_classifier = {}
for classifier in list_classifier :
    print("Creating tree"+ classifier)
    # Create tree 
    dict_classifier[classifier] = list()
    dict_classifier[classifier].append(MLTree())
    dict_classifier[classifier][0].setTrainingData(data)
    dict_classifier[classifier][0].Type=classifier
    dict_classifier[classifier][0].ignored_features = [
            "industry code",    # duplicate of "major industry code"
            "occupation code",  # duplicate of "major occupation code"
            "detailed household summary in household", # duplicate of "detailed household and family stat" with less info
            ]
    print("Learn "+classifier)
    dict_classifier[classifier][0].learn()
    print("Test "+classifier)
    dict_classifier[classifier].append(MLValidation(dict_classifier[classifier][0]))
    dict_classifier[classifier][1].test(data)
    print("______________________________")

display.comparasion_result(dict_classifier, data)


"""

#--------------------------------------------
# Hold-out Test Set + Confusion Matrix
#--------------------------------------------
print("Learn ")
tree.learn()
print("Test ")
#Split the data: 60% training : 40% test set
#instances_train, instances_test, target_train, target_test = train_test_split(train_dfs, tree.Target, test_size=0.4, random_state=0)
#fit the model using just the test set

#display.visualize_tree(tree, True)

validation = MLValidation(tree)

validation.test(data)

print("Accuracy : " + str(validation.accuracy()))
print("F1 score : " + str(validation.accuracy_harmonic()))
print(validation.confusionMatrix())
display.disp_confusion_mat(validation.confusionMatrix())
"""
""" prendre autant de 0 que de 1 en prédiction """ 

"""
tree_select = feature_selection.FeatureSelection(train_dfs, tree.Target)
tree_select = tree_select.select_threshold(0.1)
print(tree_select.dtype.names)
"""

#print("testing")
#validation = MLValidation(tree)

#validation.test(data)