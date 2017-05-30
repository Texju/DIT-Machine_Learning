# -*- coding: utf-8 -*-
##############################
# Julien Couillard IMR 2 ENSSAT 
# DIT Machine Learning 
# 22 / 05 / 2017 
##############################

import os
import plotly.offline as py
import plotly.graph_objs as go
import collections
import pandas
import numpy
import scipy
import operator 
import copy
import Information_Based_Learning_Part_1

def dir_ok():
	print("Check configuration")
	folders = os.listdir('../')
	if "Results" not in folders:
		print("Create folder 'Results'")
		os.mkdir('../Results')
	else :
		print("Configuration ok")
	
def count_occurences(values):
	occurences = dict()
	for value in values:
		value = str(value).strip()
		if not value in occurences:
			occurences[value] = 0
		occurences[value] += 1
	return occurences
	
def show(data):
	print("Generate graphics")
	for cat in data.columns:
		if data[cat].dtypes == 'int64' and data[cat].value_counts().size >= 10:
			result = [go.Histogram(x=data[cat])]
		else:
			result = [go.Bar(x=data[cat].value_counts().keys(),y=data[cat].value_counts().values)]
		py.plot(result, filename="../Results/"+cat+".html")

def percent_miss(list_cat):
	percent = 0 
	nb_elem = 0
	for elem in list_cat:
		nb_elem = nb_elem + 1
		elem = str(elem)
		if elem == " ?" :
			percent = percent + 1
		elif elem.find("Not in universe") != -1 :
			percent = percent + 1
		elif elem.find("Do not know") != -1 :
			percent = percent + 1
		elif elem.find("NA") != -1: 
			percent = percent + 1
	percent = (percent ) /  nb_elem
	percent = round(percent,4)
	return percent

def traitement_continuous(list_continuous, data): 
	print('Traitement for continous table')
	dict_continuous =  collections.OrderedDict()
	list_column = ["Feature", "Count", "Miss", "Card", "Min", "1 Qrt","Mean", "Median", "2 Qrt","Max", "Std Dev"]
	"""
	for name in list_column : 
		dict_continuous[name] = list()
	"""

	dict_continuous["Feature"] = list_continuous
	dict_continuous["Count"] = []
	dict_continuous["Miss"] = []
	dict_continuous["Card"] = []
	dict_continuous["Min"] = []
	dict_continuous["1 Qrt"] = []
	dict_continuous["Mean"] = []
	dict_continuous["Median"] = []
	dict_continuous["2 Qrt"] = []
	dict_continuous["Max"] = []
	dict_continuous["Std Dev"] = []
	
	for cat in dict_continuous["Feature"] :
		print("Traitement : "+cat)
		current = data[cat]
		dict_continuous["Count"].append(len(current))
		dict_continuous["Miss"].append(percent_miss(current))
		dict_continuous["Card"].append(len(set(current)))
		dict_continuous["Min"].append(min(current))
		dict_continuous["1 Qrt"].append(numpy.percentile(current, 25))
		dict_continuous["Mean"].append(round(numpy.mean(current),4))
		dict_continuous["Median"].append(numpy.median(current))
		dict_continuous["2 Qrt"].append(numpy.percentile(current, 75))
		dict_continuous["Max"].append(max(current))
		dict_continuous["Std Dev"].append(round(current.std(),4 ))

	table = pandas.DataFrame.from_dict(dict_continuous)
	return table
	

def traitement_categorical(list_categorical, data): 
	print('Traitement for categorical table')
	dict_categorical = collections.OrderedDict()
	dict_categorical["Feature"] = list_categorical
	dict_categorical["Count"] = []
	dict_categorical["Miss"] = []
	dict_categorical["Card"] = []
	dict_categorical["Mode"] = []
	dict_categorical["Mode Freq"] = []
	dict_categorical["Mode %"] = []
	dict_categorical["2nd Mode"] = []
	dict_categorical["2nd Mode Freq"] = []
	dict_categorical["2nd Mode %"] = []

	for cat in dict_categorical["Feature"] :
		print("Traitement : "+cat)
		current = data[cat]
		mode_occurences = count_occurences(current)
		first_mode = max(iter(mode_occurences.items()), key=operator.itemgetter(1))[0]
		second_mode = copy.copy(mode_occurences)
		second_mode.pop(first_mode, None)
		mode_second = max(iter(second_mode.items()), key=operator.itemgetter(1))[0]
		dict_categorical["Count"].append(len(current))
		dict_categorical["Miss"].append(percent_miss(current))
		dict_categorical["Card"].append(len(set(current)))
		dict_categorical["Mode"].append(first_mode)
		dict_categorical["Mode Freq"].append((mode_occurences[first_mode]))
		dict_categorical["Mode %"].append(round(((mode_occurences[first_mode] / len(data))),4))
		dict_categorical["2nd Mode"].append((mode_second))
		dict_categorical["2nd Mode Freq"].append((mode_occurences[mode_second]))
		dict_categorical["2nd Mode %"].append(round(((mode_occurences[mode_second] / len(data))),4))
		
	table = pandas.DataFrame.from_dict(dict_categorical)
	return table

def traitement_file_in(path):
	print("Traitement of the file")
	continuous = "../Results/DQR-ContinuousFeatures.csv.csv"
	categorical = "../Results/DQR-CategoricalFeatures.csv"
	data = pandas.read_csv(path)
	categories = data.columns

	list_wrong_categories = ["year", "industry code", "occupation code", "own business or self employed", "veterans benefits" ]
	list_continuous = list(data.select_dtypes(exclude=['O']))
	list_categorical = list(data.select_dtypes(include=['O']))
	for cat in list_wrong_categories:
		list_continuous.remove(cat)
		list_categorical.append(cat)
	
	
	table_continuous = traitement_continuous(list_continuous, data)
	table_categorical = traitement_categorical(list_categorical, data)
	
	table_categorical.to_csv(categorical)
	table_continuous.to_csv(continuous)
	
if __name__ == "__main__":
	print("Launch of moddeling")
	path_entre = "../Data/DataSet.csv"
	dir_ok()
	traitement_file_in(path_entre)
	#data = pandas.read_csv(path_entre)
	#show(data)