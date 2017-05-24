from Data import MLData

data = MLData('../Data/DataSet.csv')

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

