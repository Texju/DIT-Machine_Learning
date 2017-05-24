# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:00:15 2017

@author: Julien Couillard & Jean Thevenet
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# TODO : 
"""
creates 3 different prediction models from the sci-kit library; for example, a decision tree, nearest neighbor, naive bayes models, random forest.

"""
class MLTree:
    """This class create a tree from the data"""
    def __init__(self):
        self.__ignore = set([])
        self.__treeType = "DecisionTree"
        self.__data = []
        self.__tree = None
        self.__target = None
        self.__train_dfs = []
        
    @property
    def ignored_features(self):
        return self.__ignore;
    
    @ignored_features.setter
    def ignored_features(self,features):
        self.__ignore = set(features)
        
    def setTrainingData(self, data):
        self.__data = [data.Training, data.Validation, data.Testing]
        for feature in self.__ignore:
            data[0].drop(self.__ignore)

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
        if treeType == "DecisionTree":
            self.__treeType = treeType
        elif self.__treeType == "RandomForest":
            self.__treeType = treeType
        elif self.__treeType == "GaussianNB":
            self.__treeType = treeType
        else:
            raise ValueError('Tree type not implemented (yet?)')
            
    def prepareData(self):
        if len(self.__data) != 0:
            # Extract Target Feature
            self.__target = self.__data[0]['target']
            
            # List of features with numerical values but must be considered as categorical features
            list_wrong_categories = ["year", "industry code", "occupation code", "own business or self employed", "veterans benefits" ]

            # Extract numeric feature list
            numeric_features = list(self.__data[0].select_dtypes(exclude=['O']))

            # Delete from numeric feature list the features which must be considered as categorical features
            for cat in list_wrong_categories:
                numeric_features.remove(cat)

            # Extract Categorical Descriptive Features
            categorical_dfs = self.__data[0].drop(numeric_features + ['target'],axis=1)

            # Extract Numeric Descriptive Features
            numeric_dfs = self.__data[0][numeric_features]

            # Remove missing values and apply one-hot encoding
            categorical_dfs.replace('?','NA')
            categorical_dfs.replace('Not in universe','NA')
            categorical_dfs.replace('Do not know','NA')
            categorical_dfs.replace('Not in universe or children','NA')
            categorical_dfs.fillna( 'NA', inplace = True )

            #transpose into array of dictionaries (one dict per instance) of feature:level pairs
            categorical_dfs = categorical_dfs.T.to_dict().values()
            
            #convert to numeric encoding
            vectorizer = DictVectorizer( sparse = False )
            vec_cat_dfs = vectorizer.fit_transform(categorical_dfs) 
            # Merge Categorical and Numeric Descriptive Features
            self.__train_dfs = numpy.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))
            
            return self.__train_dfs
        else:
            raise ValueError('Training data not set.')


    def learn(self):
        if len(self.__train_dfs) != 0:
            #---------------------------------------------------------------
            #   Create and train a decision tree model using sklearn api
            #---------------------------------------------------------------
            #create an instance of a decision tree model.

            if self.__treeType == "DecisionTree":
                decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')
                #fit the model using the numeric representations of the training data
                decTreeModel.fit(self.__train_dfs, self.__target)
                self.__tree = decTreeModel
                return self.__tree
                
            #---------------------------------------------------------------
            #   Create and train a decision tree model using sklearn api
            #---------------------------------------------------------------
            #create an instance of a Random Forest tree model.
            
            elif self.__treeType == "RandomForest":
                # Train random forest classifier, calibrate on validation data and evaluate
                # on test data
                clf = RandomForestClassifier(n_estimators=25)
                clf.fit(self.__data[0], self.__target)
                clf_probs = clf.predict_proba(self.__data[2])
                sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
                # /!\ self.__target == y_valid
                sig_clf.fit(self.__data[1], self.__target)
                sig_clf_probs = sig_clf.predict_proba(self.__data[2])
                sig_score = log_loss(y_test, sig_clf_probs)

            elif self.__treeType == "GaussianNB":
                gnb = GaussianNB()
                y_pred = gnb.fit(iris.data, iris.target).predict(self.__data[0])
                """
                print("Number of mislabeled points out of a total %d points : %d"
                ...       % (iris.data.shape[0],(iris.target != y_pred).sum()))
                """ 
        else:
            raise ValueError('Training data not prepared.')