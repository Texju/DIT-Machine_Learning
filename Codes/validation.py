
class MLValidation:
    """This class calculates the preciseness of our tree against a set of data"""
    
    def __init__(self, tree):
        self.__tree = tree
        
    def test(self, data):
        """treeForConversion = MLTree()
        
        treeForConversion.ignored_features = self.__tree.ignored_features
        treeForConversion.setTrainingData(data, True)
        treeForConversion.prepareData(dataType)
        """
        dta, res = self.__tree.prepareData(data.Testing)
        
        predictions = self.__tree.Tree.predict(dta)
        
        print(predictions)