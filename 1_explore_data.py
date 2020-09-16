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

def acq_train():
    train_df = pd.read_csv('./titanic/train.csv')
    return train_df

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
    parch = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by = 'Survived', ascending=False)
    print(pclass)
    print('_'*40)
    print(sex)
    print('_'*40)
    print(Sibsp)
    print('_'*40)
    print(parch)

def age_dis():
    train_df = acq_train()
    g = sns.FacetGrid(train_df, col = 'Survived')
    g.map(plt.hist, 'Age', bins = 20)
    plt.show()


def pclass_dis():
    train_df = acq_train()
    grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size = 2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5,bins=20)
    grid.add_legend()
    plt.show()

def embark_dis():
    train_df =acq_train()
    grid = sns.FacetGrid(train_df, row ='Embarked', size = 2.2, aspect = 1.6)
    grid.map(sns.pointplot, 'Pclass','Survived', 'Sex', palette='deep')
    grid.add_legend()
    plt.show()


def fare_dis():
    train_df =acq_train()
    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()
    plt.show()

def drop_TicCabin():
    train_df, test_df, combine = acquire_data()
    # print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape )
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    # print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
    return train_df, test_df, combine

# Name Anslysis:
def convert_name():
    train_df, test_df, combine = drop_TicCabin()
    # print("Check", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # print(pd.crosstab(train_df['Title'], train_df['Sex']))
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                    'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
    title_mapping = {"Mr": 1,"Miss": 2,"Mrs": 3,"Master":4,"Rare":5}
    for dataset in combine:
        dataset['Title']=dataset['Title'].map(title_mapping)
        dataset['Title']=dataset['Title'].fillna(0)
    # print(train_df.head())
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    # print(train_df.shape, test_df.shape)
    return train_df, test_df, combine

def convert_sex():
    train_df, test_df, combine = convert_name()
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    # print(train_df.head())
    # print(test_df.head())
    return train_df, test_df, combine












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
# What is the distribution of numerical feature values across the samples?~