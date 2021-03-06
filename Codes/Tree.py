# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:00:15 2017

@author: Julien Couillard & Jean Thevenet
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy
from sklearn import tree

class MLTree:
    """This class create a tree from the data"""
    def __init__(self):
        self.__ignore = set([])
        self.__treeType = "DecisionTree"
        self.__data = []
        self.__tree = None
        self.__target = None
        self.__train_dfs = []
        self.__vectorizer = DictVectorizer( sparse = False )
        
    @property
    def ignored_features(self):
        return self.__ignore;
    
    @ignored_features.setter
    def ignored_features(self,features):
        self.__ignore = set(features)
        
    def setTrainingData(self, data, debug=False):
        self.__data = [data.Training.copy(), data.Validation.copy(), data.Testing.copy()]
        
        categorical_dfs = data.Raw.select_dtypes(include=['O'])
        categorical_dfs = categorical_dfs.T.to_dict().values()
        self.__vectorizer.fit(categorical_dfs)

        self.__data[0].drop(self.__ignore, 1)
        self.__data[1].drop(self.__ignore, 1)
        self.__data[2].drop(self.__ignore, 1)

    @property
    def Tree(self):
        return self.__tree

    @property
    def Target(self):
        return self.__target
    @property
    def Train_dfs(self):
        return self.__train_dfs

    @Train_dfs.setter
    def Train_dfs(self, Train_dfs):
        self.__train_dfs = Train_dfs
        
    @property
    def Type(self):
        return self.__treeType
    
    @Type.setter
    def Type(self, treeType):
        if treeType == "DecisionTreeEntropy":
            self.__treeType = treeType
        elif treeType == "DecisionTreeGini":
            self.__treeType = treeType
        elif treeType == "RandomForestEntropy":
            self.__treeType = treeType
        elif treeType == "RandomForestGini":
            self.__treeType = treeType
        elif treeType == "GaussianNB":
            self.__treeType = treeType
        elif treeType == "MLP":
            self.__treeType = treeType
        else:
            raise ValueError('Tree type not implemented (yet?)')
            
    def prepareData(self, data):
        if len(data) != 0:
            # Extract Target Feature
            target = data['target']
            
            # List of features with numerical values but must be considered as categorical features
            list_wrong_categories = ["year", "industry code", "occupation code", "own business or self employed", "veterans benefits" ]

            # Extract numeric feature list
            numeric_features = list(data.select_dtypes(exclude=['O']))
            #print("Numeric features num: " + str(len(numeric_features)))
            
            # Delete from numeric feature list the features which must be considered as categorical features
            for cat in list_wrong_categories:
                numeric_features.remove(cat)

            # Extract Categorical Descriptive Features
            categorical_dfs = data.drop(numeric_features + ['target'],axis=1)
            #print("Categorical features num: " + str(categorical_dfs.count()))
            
            # Extract Numeric Descriptive Features
            numeric_dfs = data[numeric_features]

            # Remove missing values and apply one-hot encoding
            categorical_dfs.replace('?','NA')
            categorical_dfs.replace('Not in universe','NA')
            categorical_dfs.replace('Do not know','NA')
            categorical_dfs.replace('Not in universe or children','NA')
            categorical_dfs.fillna( 'NA', inplace = True )

            #transpose into array of dictionaries (one dict per instance) of feature:level pairs
            categorical_dfs = categorical_dfs.T.to_dict().values()
            
            #convert to numeric encoding
            vec_cat_dfs = self.__vectorizer.transform(categorical_dfs) 
            #print("Categorical features num: " + str(len(vec_cat_dfs[0])))
            # Merge Categorical and Numeric Descriptive Features
            train_dfs = numpy.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))
            #print("final" + str(len(train_dfs[0])))
            return train_dfs, target
        else:
            raise ValueError('Data not set.')

    def learn(self, dontFit = False):
        self.__train_dfs, self.__target = self.prepareData(self.__data[0])
        
        if len(self.__train_dfs) != 0:
            #---------------------------------------------------------------
            #   Create and train a tree model using sklearn api
            #---------------------------------------------------------------
            #create an instance of a tree model.
            if self.__treeType == "DecisionTreeEntropy":
                self.__tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 10)
            elif self.__treeType == "DecisionTreeGini":
                self.__tree = tree.DecisionTreeClassifier(criterion='gini', max_depth = 10)
            elif self.__treeType == "RandomForestEntropy":
                self.__tree  = RandomForestClassifier(criterion='entropy', n_estimators=25, max_depth = 10)
            elif self.__treeType == "RandomForestGini":
                self.__tree  = RandomForestClassifier(criterion='gini', n_estimators=25, max_depth = 10)
            elif self.__treeType == "GaussianNB":
                self.__tree = GaussianNB()
            elif self.__treeType == "MLP":
                self.__tree = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,40), random_state=1)

            if dontFit == False:
                #fit the model using the numeric representations of the training data
                self.__tree.fit(self.__train_dfs, self.__target)

            return self.__tree

        else:
            raise ValueError('Training data not prepared.')