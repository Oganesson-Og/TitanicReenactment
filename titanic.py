import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option("display.max_columns", None)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from xgboost import XGBClassifier
import pickle

data_set = pd.read_csv("C:/Users/Admin/Documents/titanic data science/datasets_11657_16098_train.csv")

print(data_set.head())
data_set_2 = data_set.copy()
data_set_2 = np.where(data_set_2.isnull(), 1, 0)
data_set_2 = pd.DataFrame(data_set_2, columns=data_set.columns)

sns.heatmap(data_set_2, cmap='plasma')
plt.show()
# We can see that there are far too many null values in Cabin section. We can drop that section
# On the other hand we can fill up the null values of the age and embarked sections as they have a
# reasonably low number of null values.

# Since we are predicting the chances of survival during the fateful accident, certain features are inconsequential
print(data_set.shape)
print(len(data_set['Name'].unique()))
print(len(data_set['PassengerId'].unique()))
print(data_set.info())

droppables = ['PassengerId', 'Name', 'Cabin', 'Ticket']
X = data_set.drop(droppables, axis=1)

gender_ser = X["Sex"].factorize()
gender_ser = pd.Series(gender_ser[0], name='Sex')
embarked = X["Embarked"].factorize()
embarked = pd.Series(embarked[0], name='Embarked')
X = X.drop(['Embarked', 'Sex'], axis=1)
X = pd.concat([X, embarked, gender_ser], axis=1)
print(X.info())

for feature in X.columns:
    if feature == 'Age':
        pass
    else:
        sns.boxplot(X[feature], X['Age'])
        plt.show()
        # We can see a discernible correlation between passenger class and age 37, 29 and 23'''

for i in range(X.shape[0]):
    if X.iloc[i, 1] == 1:
        X['Age'].fillna(37, inplace=True)
    elif X.iloc[i, 1] == 2:
        X['Age'].fillna(29, inplace=True)
    else:
        X['Age'].fillna(23, inplace=True)

#print(X.info())

sns.heatmap(X.corr(), cmap='plasma', annot=True)
plt.show()
# There seems to be the greatest relationship between Sex and survival chances. Even more than passenger class or age

y = X.pop('Survived')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# BASE MODEL (LOGISTIC REGRESSION)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
first_result = classification_report(y_test, y_pred)
print(first_result)

# ADA BOOST CLASSIFIER
ada_boost = AdaBoostClassifier()
ada_boost.fit(X_train, y_train)
y_pred = ada_boost.predict(X_test)
sec_result = classification_report(y_test, y_pred)
print(sec_result)

# GRADIENT BOOSTING CLASSIFIER
grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train, y_train)
y_pred = grad_boost.predict(X_test)
third_result = classification_report(y_test, y_pred)
print(third_result)

#DECISION TREE CLASSIFIER
dec_tree = tree.DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
y_pred = dec_tree.predict(X_test)
fourth_result = classification_report(y_test, y_pred)
print(fourth_result)
print(X_train.head())
# XG BOOST CLASSIFIER
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
fifth_result = classification_report(y_test, y_pred)
print(fifth_result)

pickle.dump(grad_boost, open('titanic_model.pkl', 'wb'))