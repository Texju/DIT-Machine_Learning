# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:00:15 2017

@author: Julien Couillard & Jean Thevenet
"""

from sklearn.feature_extraction import DictVectorizer
import numpy
from sklearn import tree

class MLTree:
    """This class create a tree from the data"""
    def __init__(self):
        self.__ignore = set([])
        self.__treeType = "DecisionTree"
        self.__data = []
        self.__tree = None
        
    @property
    def ignored_features(self):
        return self.__ignore;
    
    @ignored_features.setter
    def ignored_features(self,features):
        self.__ignore = set(features)
        
    def setTrainingData(self, data):
        for feature in self.__ignore:
            data.drop(self.__ignore)
            
        self.__data = data
    
    @property
    def Tree(self):
        return self.__tree
    
    @property
    def Type(self):
        return self.__treeType
    
    @Type.setter
    def Type(self, treeType):
        if treeType == "DecisionTree":
            self.__treeType = treeType
        else:
            raise ValueError('Tree type not implemented (yet?)')
            
    def learn(self):
        if len(self.__data) != 0:
            # Extract Target Feature
            targetLabels = self.__data['target']
            
            # List of wrong feature in numerical 
            list_wrong_categories = ["year", "industry code", "occupation code", "own business or self employed", "veterans benefits" ]

            # Extract feature numeric 
            numeric_features = list(self.__data.select_dtypes(exclude=['O']))

            # Delete from feature numeric wrong feature
            for cat in list_wrong_categories:
                numeric_features.remove(cat)


            # Extract feature categorical feature 
            categorical_dfs = self.__data.drop(numeric_features + ['target'],axis=1)


            # Extract Numeric Descriptive Features
            numeric_dfs = self.__data[numeric_features]
            
            # Extract Categorical Descriptive Features
            #categorical_features = list(self.__data.select_dtypes(include=['O']))
            
            
            

            

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
            train_dfs = numpy.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))
    
            if self.__treeType == "DecisionTree":
                decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')
                
            #---------------------------------------------------------------
            #   Create and train a decision tree model using sklearn api
            #---------------------------------------------------------------
            #create an instance of a decision tree model.
            
            #fit the model using the numeric representations of the training data
            decTreeModel.fit(train_dfs, targetLabels)
            
            self.__tree = decTreeModel
        else:
            raise ValueError('Training data not set.')
        