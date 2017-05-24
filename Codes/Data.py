# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:22:41 2017

@author: Jean Thevenet
"""

import pandas
import numpy

class MLData:
    """This class loads & prepare the data"""
    
    def __init__(self, csvPath):
        self.__CSV = pandas.read_csv(csvPath,delimiter=',')
        self.__Data = self.__CSV
        self.__SplitSizes = [.5, .2] # Default split: 50%, 20%, 30%
        self.__splitData()
        
    def shuffle(self):
        self.__Data = self.__CSV.sample(frac=1)
        self.__splitData()

    def unshuffle(self):
        self.__Data = self.__CSV
        self.__splitData()

    def __splitData(self):
        splitIndexes = []
        splitIndexes.append(int(self.__SplitSizes[0]*len(self.__Data)))
        splitIndexes.append(splitIndexes[0] + int(self.__SplitSizes[1]*len(self.__Data)))

        [self.__Training, self.__Validate, self.__Test] = numpy.split(self.__Data, splitIndexes)
            
    @property
    def Raw(self):
        return self.__Data
    
    @property
    def Training(self):
        return self.__Training
    
    @property
    def Validation(self):
        return self.__Validate
    
    @property
    def Testing(self):
        return self.__Test
    
    @property
    def SplitSize(self):
        return self.__SplitSizes
    
    @SplitSize.setter
    def SplitSize(self, SplitSize):
        if len(SplitSize) == 2:
            if SplitSize[0] > 0 and SplitSize[1] > 0:
                if (SplitSize[0] + SplitSize [1]) < 1:
                    self.__SplitSizes = SplitSize
                elif (SplitSize[0] + SplitSize [1]) < 100:
                    self.__SplitSizes = [SplitSize[0]/100, SplitSize[1]/100]
    