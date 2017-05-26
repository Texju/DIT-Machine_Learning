from Data import MLData
from Tree import MLTree
from validation import MLValidation
import display

class MLMain:
    def __init__(self, treeType = "DecisionTreeEntropy"):
        print("Load data")
        print("______________________________")
        # Load data
        self.__data = MLData('../Data/DataSet.csv')

        # Create tree
        tree = MLTree()
    
        tree.Type = treeType
        tree.ignored_features = [ # see notebook for uncommented reasons
                "wage per hour",
                "capital gains",
                "capital losses",
                "divdends from stocks",
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
        
        tree.setTrainingData(self.__data)
        
        self.__tree = tree
    
    def simple_test(self):
        #--------------------------------------------
        # Hold-out Test Set + Confusion Matrix
        #--------------------------------------------
        print("Learn ")
        self.__tree.learn()
        print("Test ")

        validation = MLValidation(self.__tree)
        
        validation.test(self.__data)
        
        print("Accuracy : " + str(validation.accuracy()))
        print("F1 score : " + str(validation.accuracy_harmonic()))
        print(validation.confusionMatrix())
        display.disp_confusion_mat(validation.confusionMatrix())
    def simple_test_naive(self):
        self.__tree.learn()
        validation = MLValidation(self.__tree)
        validation.testNaiveAlwaysYes(self.__data)
        
        print("Accuracy : " + str(validation.accuracy()))
        print("F1 score : " + str(validation.accuracy_harmonic()))
        print(validation.confusionMatrix())
        display.disp_confusion_mat(validation.confusionMatrix())
    def display_fancy_graphics(self):
        # DecisionTreeEntropy DecisionTreeGini RandomForest GaussianNB OCSVM SVC LinearRegression
        #  , "OCSVM", "SVC", "LinearRegression"
        list_classifier = ["DecisionTreeEntropy", "DecisionTreeGini", "RandomForestEntropy","RandomForestGini" , "GaussianNB"]
        dict_classifier = {}
        for classifier in list_classifier :
            print("Creating tree"+ classifier)
            # Create tree 
            dict_classifier[classifier] = list()
            dict_classifier[classifier].append(MLTree())
            dict_classifier[classifier][0].Type=classifier
            dict_classifier[classifier][0].ignored_features = self.__tree.ignored_features
            dict_classifier[classifier][0].setTrainingData(self.__data)
            print("Learn "+classifier)
            dict_classifier[classifier][0].learn()
            print("Test "+classifier)
            dict_classifier[classifier].append(MLValidation(dict_classifier[classifier][0]))
            dict_classifier[classifier][1].test(self.__data)
            print("______________________________")
        
        display.comparasion_result(dict_classifier, self.__data)

    def display_decisision_tree(self, pdf = True):
        if self.__tree.Type == "DecisionTreeEntropy" or self.__tree.Type == "DecisionTreeGini" :
            print("Learn ")
            self.__tree.learn()
            print("Test ")
            validation = MLValidation(self.__tree)
            validation.test(self.__data)
            print("Accuracy : " + str(validation.accuracy()))
            print("F1 score : " + str(validation.accuracy_harmonic()))
            display.visualize_tree(self.__tree, pdf)

        else : 
            print("Sorry, bad tree type")


    def fiveFoldValidation(self):
        self.__tree.learn(True)
        validation = MLValidation(self.__tree)
        
        scores = validation.test_KFoldCrossValidation(self.__data, 5)
        print("Score by fold: " + str(scores))
        print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
