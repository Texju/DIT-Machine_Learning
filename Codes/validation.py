# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:00:15 2017

@author: Julien Couillard & Jean Thevenet
"""
from sklearn import cross_validation
from sklearn import metrics
import numpy

class MLValidation:
    """This class calculates the preciseness of our tree against a set of data"""
    
    def __init__(self, tree):
        self.__tree = tree
        self.__targets = []
        self.__predictions = []
        
    def test(self, data):
        """ Testing the model """
        # Train our model
        instances_train, target_train = self.__tree.prepareData(data.Training)
        self.__tree.Tree.fit(instances_train, target_train)
        
        # Test the model
        instances_test, target_test = self.__tree.prepareData(data.Testing)

        self.__targets = target_test

        #Use the model to make predictions for the test set queries
        self.__predictions = self.__tree.Tree.predict(instances_test)

    def test_KFoldCrossValidation(self, data, k):
        instances_train, target_train = self.__tree.prepareData(data.Raw)
        scores=cross_validation.cross_val_score(self.__tree.Tree, instances_train, target_train, cv=k)
        
        return scores

    def testNaiveAlwaysYes(self, data):
        """ Test our targets against a matrix that always return - 50000"""
        self.test(data)
        self.__predictions[:] = " - 50000."
            
    def confusionMatrix(self):
        if len(self.__predictions) != 0:
            return metrics.confusion_matrix(self.__targets, self.__predictions)
        
    def accuracy(self):
        return metrics.accuracy_score(self.__targets, self.__predictions, normalize=True)
    
    def accuracy_harmonic(self):
        t = self.__targets.replace(" - 50000.","yes")
        t = t.replace(" 50000+.","no")
        
        p = numpy.copy(self.__predictions)
        p[p == " - 50000."] = "yes"
        p[p == " 50000+."] = "no"
        
        return metrics.f1_score(t, p, pos_label="yes")