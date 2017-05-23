import collections
import pandas
import numpy
import scipy
import Information_Based_Learning_Part_1 as base1

from sklearn.feature_extraction import DictVectorizer
#create the vectorizor object
mlVectorizer = DictVectorizer(sparse=False)

# Set your dataset path below:
path_dataset = "../Data/DataSet.csv"

# Read the data and separate continuous & categorical values:
data = pandas.read_csv(path_dataset)

targetValues = data["target"]

data.drop(["target"], axis=1)

feature_dict = dict()
list_continuous = list(data.select_dtypes(exclude=['O']))
list_categorical = list(data.select_dtypes(include=['O']))
feature_dict["continuous"] = list_continuous
feature_dict["categorical"] = list_categorical

feature_dict["categorical_new"] = dict()
dict_categorical = dict()

def dictToList(dictionnary):
	result = [[], {}]
	i = 0
	for name, value in dictionnary.items():
		result[0].append(list(value))
		result[1][name] = i
		
		i=i+1
	return result

def reject_to_misses(feature_list, data):
	list_delete = list()
	## DEBUG
	print(len(feature_list))
	for feature in feature_list :
		if GenerateDQR.percent_miss(data[feature]) >= float(25) :
			list_delete.append(feature)
	
	for feature in list_delete:
		if feature in feature_list:
			feature_list.remove(feature)	
	## DEBUG
	print(len(feature_list))
	return feature_list

def toVectorizerCompatible(data, feature_list):
	cat_list = []
	first = True
	
	for feature in feature_list:
		for i, v in data[feature].items():
			if first == True:
				cat_list.append({})
			
			cat_list[i][feature] = v
		first = False
		
	return cat_list

#pass in the data into the vectorizer object so that it can learn an encoding for the feature values, this function 
#also returns the data after the encoding as been applied (but we will ignore this for now)
X=mlVectorizer.fit_transform(toVectorizerCompatible(data, feature_dict["categorical"]))
print(X)
#print(reject_to_misses(feature_dict["categorical"], data))

train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))