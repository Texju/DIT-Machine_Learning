from sklearn import metrics
import sys

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

    def testNaiveAlwaysYes(self, data):
        """ Test our targets against a matrix that always return - 50000"""
        self.test(data)
        self.__predictions[:] = " - 50000."
            
    def confusionMatrix(self):
        if len(self.__predictions) != 0:
            return metrics.confusion_matrix(self.__targets, self.__predictions)
        
    def accuracy(self):
        return metrics.accuracy_score(self.__targets, self.__predictions, normalize=True)
    