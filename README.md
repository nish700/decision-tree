### Project Overview

 **About the Project:**

To achieve the objective of classifying people using their demographic data , following steps have been performed:

- Load the dataset

- Analyse the dataset

- Clean the dataset

- Encode the Target Label 

- Visualize the features : Bar plot for numerical featurea and Histogram for categorical features

- Impute the missing values : by replacing the missing categorical values with mode and numerical values with median

- Label Encode the categorical features

- One Hot encode the categorical features

- Split the data into Train and Test set

- Model a Decision Tree Classifier and fit to the train set.

- Use GridSearchCV to find the optimun depth of the tree

-  Use the Optimal depth to fit Decision tree on test data and calculate the Accuracy.



**Dataset description:**

Dataset UCI Adult:  classify people using demographical data - whether they earn more than $50,000 per year or not.

 **Feature descriptions:**

**Age **– continuous feature

**Workclass** – continuous feature

**fnlwgt** – final weight of object, continuous feature

**Education** – categorical feature

**Education_Num** – number of years of education, continuous feature

**Martial_Status** – categorical feature

**Occupation** – categorical feature

**Relationship** – categorical feature

**Race** – categorical feature

**Sex** – categorical feature

**Capital_Gain** – continuous feature

**Capital_Loss** – continuous feature

**Hours_per_week** – continuous feature

**Country** – categorical feature

**Target** – earnings level, categorical (binary) feature.



