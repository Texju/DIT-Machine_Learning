# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:25:20 2017

@author: jthev001
"""
import sys

from Main import MLMain

if len(sys.argv) >= 2:
    valid_input = True

    if len(sys.argv) >= 3:
        t = sys.argv[2]

        if not (t == "DecisionTreeEntropy" or t == "DecisionTreeGini" or t == "RandomForest" or t=="GaussianNB"):
            valid_input = False
            print("This tree type is not implemented. Please use \"python launch.py help\".")
    elif not (sys.argv[1] == "fancy_graphics" or sys.argv[1] == "help" or sys.argv[1] == "test_naive"):
        print("You must specify a tree type. Please use \"python launch.py help\".")
        valid_input = False
    
    if valid_input == True:
        if sys.argv[1] == "test_accuracy":
            main = MLMain(sys.argv[2])
            main.simple_test()
        elif sys.argv[1] == "test_naive":
            main = MLMain()
            main.simple_test_naive()
        elif sys.argv[1] == "fancy_graphics":
            main = MLMain()
            main.display_fancy_graphics()
        elif sys.argv[1] == "5_fold_validation":
            main = MLMain(sys.argv[2])
            main.fiveFoldValidation()
        elif sys.argv[1]=="help":
            print("Command list:")
            print("---")
            print("- test_accuracy [tree type]: Test the accuracy of a model using standard accuracy, F1 and confusion matrix.")
            print("- test_naive: Test the accuracy of a model that return always yes using standard accuracy, F1 and confusion matrix.")
            print("- 5_fold_validation [tree type]: Test the accuracy of a model using five-fold validation")
            print("- fancy_graphics: compare the accuracies of all available models using fancy graphics")
            print("---")
            print("Possible values for [tree type]:")
            print("---")
            print("- DecisionTreeEntropy: A entropy-based decision tree")
            print("- DecisionTreeGini: A Gini-based decision tree")
            print("- RandomForest: Random forest")
            print("- GaussianNB: Naive gaussian tree")
        else:
            print("Unknown argument. Please use \"python launch.py help\".")
else:
    print("Wrong number of input arguments. Expected 1 (use \"python launch.py help\" to display a list of commands))")