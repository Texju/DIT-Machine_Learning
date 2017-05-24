
# coding: utf-8


from pandas import DataFrame
from sklearn import preprocessing
from sklearn import tree
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn
import numpy as np
import pandas as pd
import GenerateDQR
import Information_Based_Learning_Part_1 as base1
import itertools
import subprocess

fromUrl = False


def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        sklearn.tree.export_graphviz(tree, out_file=f,
                        feature_names=feature_names)
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def reject_to_misses(feature_list, data):
	list_delete = list()
	## DEBUG
	print(len(feature_list))
	for feature in feature_list :
		if GenerateDQR.percent_miss(data[feature]) >= float(25) :
			list_delete.append(feature)
	print(list_delete)
	for feature in list_delete:
		if feature in feature_list:
			feature_list.remove(feature)	
	## DEBUG
	print(len(feature_list))
	return feature_list, list_delete

if (fromUrl):
	#Reading the dataset from an online repository:
	#-----------------------------------------------
	fileUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
	#define the list of column headings for the dataset. This list is based on the documentation
	#for the dataset available at: 
	#https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
	columnHeadings=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','annualincome']
	# we can directly use read_csv to download the file
	data = pd.read_csv(fileUrl,header=None,names=columnHeadings,index_col=False,na_values=['?'],nrows=32560)
	# save the file locally 
	data.to_csv('../Datasets/censusDataRaw.csv',index=False)
else:
	#Reading the dataset from a local file
	#---------------------------------------------
	#censusData = pd.read_csv("../Datasets/censusDataRaw.csv",index_col=False,na_values=['?'],nrows=32560)
	# Set your dataset path below:
	path_dataset = "../Data/DataSet.csv"
	# Read the data and separate continuous & categorical values:
	data = pd.read_csv(path_dataset)


# Extract Target Feature
targetLabels = data['target']
# Extract Numeric Descriptive Features
numeric_features = list(data.select_dtypes(exclude=['O']))
numeric_features, numeric_features_drop = reject_to_misses(numeric_features, data)
numeric_dfs = data[numeric_features]
# Extract Categorical Descriptive Features
list_categorical = list(data.select_dtypes(include=['O']))
list_categorical, list_categorical_drop = reject_to_misses(list_categorical, data)
cat_dfs = data.drop(numeric_features + ['target'] + list_categorical_drop,axis=1)
# Remove missing values and apply one-hot encoding
cat_dfs.replace('?','NA')
cat_dfs.fillna( 'NA', inplace = True )
#transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()
#convert to numeric encoding
vectorizer = DictVectorizer( sparse = False )
vec_cat_dfs = vectorizer.fit_transform(cat_dfs) 
# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))


#---------------------------------------------------------------
#   Create and train a decision tree model using sklearn api
#---------------------------------------------------------------
#create an instance of a decision tree model.
decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')
#fit the model using the numeric representations of the training data
decTreeModel.fit(train_dfs, targetLabels)


#---------------------------------------------------------------
#   Define 2 Queries, Make Predictions, Map Predictions to Levels
#---------------------------------------------------------------
#print(numeric_features)
#print(list_categorical_drop)
# ['age', 'industry code', 'occupation code', 'wage per hour', 'capital gains', 'capital losses', 'divdends from stocks', 'instance weight', 'num persons worked for employer', 'own business or self employed', 'veterans benefits', 'weeks worked in year', 'year'] 

# ['class of worker', 'education', 'enrolled in edu inst last wk', 'marital status', 'major industry code', 'major occupation code', 'race', 'hispanic Origin', 'sex', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'tax filer status', 'region of previous residence', 'state of previous residence', 'detailed household and family stat', 'detailed household summary in household', 'live in this house 1 year ago', 'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', "fill inc questionnaire for veteran's admin", 'target']

### TODO : FAIRE UNE QUERY

q = {'age':[39,50],'workclass':['State-gov','Self-emp-not-inc'],'fnlwgt':[77516,83311],'education':['Bachelors','Bachelors'],'education-num':[13,13],'marital-status':['Never-married','Married-civ-spouse'],'occupation':['Adm-clerical','Exec-managerial'],'relationhip':['Not-in-family','Husband'],'race':['White','White'],'sex':['Male','Male'],'capital-gain':[2174,0],'capital-loss':[0,0],'hours-per-week':[40,13],'native_country':['United-States','United-States']}


col_names_tmp = itertools.chain(numeric_features,list_categorical)
col_names = list()

for i in col_names_tmp:
	col_names.append(i)
	print(i)

#visualize_tree(decTreeModel, col_names)
"""
for i in range(0, decTreeModel.n_classes_):
	print(decTreeModel.tree_.best_error[i])
"""

"""
qdf = pd.DataFrame.from_dict(q,orient="columns")
#extract the numeric features
q_num = qdf[numeric_features].as_matrix() 
#convert the categorical features
q_cat = qdf.drop(numeric_features,axis=1)
q_cat_dfs = q_cat.T.to_dict().values()
q_vec_dfs = vectorizer.transform(q_cat_dfs) 
#merge the numeric and categorical features
query = np.hstack((q_num, q_vec_dfs ))
#Use the model to make predictions for the 2 queries
predictions = decTreeModel.predict([query[0],query[1]])
print("Predictions!")
print("------------------------------")
print(predictions)
"""



#--------------------------------------------
# Hold-out Test Set + Confusion Matrix
#--------------------------------------------
"""
#define a decision tree model using entropy based information gain
decTreeModel2 = tree.DecisionTreeClassifier(criterion='entropy')
#Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)
#fit the model using just the test set
decTreeModel2.fit(instances_train, target_train)
#Use the model to make predictions for the test set queries
predictions = decTreeModel2.predict(instances_test)
#Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))
#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

#Draw the confusion matrix
import matplotlib.pyplot as plt
# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
#plt.plot(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#--------------------------------------------
# Cross-validation to Compare to Models
#--------------------------------------------

#run a 10 fold cross validation on this model using the full census data
scores=cross_validation.cross_val_score(decTreeModel2, instances_train, target_train, cv=10)
#the cross validaton function returns an accuracy score for each fold
print("Entropy based Model:")
print("Score by fold: " + str(scores))
#we can output the mean accuracy score and standard deviation as follows:
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")
#for a comparison we will do the same experiment using a decision tree that uses the Gini impurity metric
decTreeModel3 = tree.DecisionTreeClassifier(criterion='gini')
scores=cross_validation.cross_val_score(decTreeModel3, instances_train, target_train, cv=10)
print("Gini based Model:")
print("Score by fold: " + str(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""

