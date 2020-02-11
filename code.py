# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the train data stored in path variable
train_data = pd.read_csv(path)
print(train_data.tail())

# Load the test data stored in path1 variable
test_data = pd.read_csv(path1)
print(test_data.tail())

# necessary to remove rows with incorrect labels in test dataset
test_data.drop(index= test_data[test_data['Target'].isna()==True].index, inplace=True)

# clean the target variable of train and test dataset
train_data['Target'] = train_data['Target'].str.strip()
test_data['Target'] = test_data['Target'].str.strip()
test_data['Target'] = test_data['Target'].str.replace(".","")

# check the distribution of target variable
print('Train data Target before encoding',train_data['Target'].value_counts())
print('Test data target before encoding', test_data['Target'].value_counts())

# encode target variable as integer
le = LabelEncoder()
train_data['Target'] = le.fit_transform(train_data['Target'])
test_data['Target'] = le.transform(test_data['Target'])

# check the distribution of target variable after encoding
print('Train data Target before encoding',train_data['Target'].value_counts())
print('Test data target before encoding', test_data['Target'].value_counts())

# identify the categorical and numerical features
num_features = train_data.select_dtypes(include=[np.number])
cat_features = train_data.select_dtypes(include='object')

# Plot the distribution of each feature
fig = plt.figure(figsize=(10,12))
nrows = 5
ncols = 3

for i, col in enumerate(train_data.columns):
    ax = fig.add_subplot(nrows, ncols, i+1)
    if col in num_features.columns:
        train_data[col].hist()
    elif col in cat_features.columns:
        train_data[col].value_counts().plot(kind='bar')
plt.show()

# Analysing the dtypes of train and test set
print('train data type', train_data.dtypes)
print('test data type', test_data.dtypes)

# convert the data type of Age column in the test data to int type, we see that the dtypes of numerical 
# features are float i.e. different than the train set
test_data['Age'] = test_data['Age'].astype('int64')

# cast all float features to int type to keep types consistent between our train and test data
for col in num_features:
    test_data[col] = test_data[col].astype('int64')


# choose categorical and continuous features from data and print them
print('categorial features are:', cat_features)
print('continuous features are:', num_features)

# fill missing data for catgorical columns with mode
for col in cat_features.columns:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    test_data[col].fillna(test_data[col].mode()[0], inplace = True)


# fill missing data for numerical columns with median
for col in num_features.columns:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    test_data[col].fillna(test_data[col].median(), inplace = True)

# Label Encoding Categorical features
le_l = LabelEncoder()

for col in cat_features:
    train_data[[col]] = le_l.fit_transform(train_data[[col]])
    test_data[[col]] = le_l.transform(test_data[[col]])


# Dummy code Categoricol features in train and test set
train_data = pd.get_dummies(train_data , columns = cat_features.columns)
test_data = pd.get_dummies(test_data , columns = cat_features.columns)

# Check for Column which is not present in test data
col_not_present = [col for col in train_data.columns if col not in test_data.columns]
print('Column present in train data but absent in test data is:',col_not_present)

# New Zero valued feature in test data for Holand
test_data['Country_14'] = 0

#check for the same dim in train and test set
print('Shape of train data:',train_data.shape)
print('Shape of test data:', test_data.shape)

# Split train and test data into X_train ,y_train,X_test and y_test data
X_train = train_data.drop(columns=['Target'], axis=1)
y_train = train_data['Target']
X_test = test_data.drop(columns=['Target'], axis=1)
y_test = test_data['Target']

# train a decision tree model then predict our test data and compute the accuracy
tree = DecisionTreeClassifier(max_depth=3, random_state=17)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the Decision Tree with 3 as max depth is:', accuracy_score)

# Decision tree with parameter tuning
tree_params = {'max_depth': range(2,11)}

tree_cv = DecisionTreeClassifier(random_state = 17)

grid_cv = GridSearchCV(tree_cv, tree_params , cv = 5)

grid_cv.fit(X_train, y_train)

# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_

print('Optimal Maximum depth of the tree is:', grid_cv.best_params_)
print('Best score of the tree is:', grid_cv.best_score_)

#train a decision tree model with best parameter then predict our test data and compute the accuracy
# using the Optimal depth of 9 and building the tree
tree_optimal = DecisionTreeClassifier(max_depth = 9,random_state=17)

tree_optimal.fit(X_train, y_train)

y_pred = tree_optimal.predict(X_test)

score_best = tree_optimal.score(X_test, y_test)

print('Best Accuracy of the decision tree classifier is:', score_best)



