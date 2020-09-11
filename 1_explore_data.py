import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def acquire_data():
    train_df = pd.read_csv('./titanic/train.csv')
    test_df = pd.read_csv('./titanic/test.csv')
    combine = [train_df, test_df]
    return train_df, test_df, combine


def print_test():
    train_df, test_df, combine = acquire_data()
    print(train_df.columns)
    print('_'*40)
    print(train_df.head())
    print('_'*40)
    print(train_df.tail())
    print('_'*40)
    print(train_df.info())
    print('_'*40)
    print(test_df.info())
    print('_'*40)
    print(train_df.describe())
    print('_'*40)
    print(train_df.describe(include=['O']))


def pivot_ana():
    train_df = pd.read_csv('./titanic/train.csv')
    pclass = train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by ='Survived', ascending = False)
    sex = train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    Sibsp = train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        


# Note:
# Question List:
#
# Which features are available in the dataset?
#
# Data type:
# Which features are categorical?
# Which features are numerical?
# Which features are mixed data types?
#
# Data May Need to Clean:
# Which features may contain errors or typos?
#   Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.
# Which features contain blank, null or empty values?
# What are the data types for various features?
# 
# What is the distribution of numerical feature values across the samples?