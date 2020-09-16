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


def guess_age():
    train_df, test_df, combine = convert_sex()
    guess_ages = np.zeros((2,3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
                guess_ages[i,j]=guess_df.median()
                # Convert random age float to nearest .5 age
                # guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
        # print(guess_df)        
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
        dataset['Age'] = dataset['Age'].astype(int)
    # print(train_df.head())
    return train_df, test_df, combine

def convert_age():
    train_df, test_df, combine = guess_age()
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    # print(train_df.head())
    return train_df, test_df, combine


def convert_family_size():
    train_df, test_df, combine = convert_age()
    # Add family size
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    # Add IsAlone
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
    # Drop Parch Sibsp and Family Size
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]
    # print(train_df.head())
    return train_df, test_df, combine


def add_AgePclass():
    train_df, test_df, combine = convert_family_size()
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    # print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
    return train_df, test_df, combine


def convert_Embarked():
    train_df, test_df, combine = add_AgePclass()
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    # print(train_df.head())
    return train_df, test_df, combine

def convert_fare():
    train_df, test_df, combine = convert_Embarked()
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    # train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]
    # print(train_df.head(10))
    # print(test_df.head(10))
    return train_df, test_df, combine


def xy_split_train():
    train_df, test_df, combine = convert_fare()
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    # print(X_train.shape, Y_train.shape)
    return X_train, Y_train


def xy_split_test():
    train_df, test_df, combine = convert_fare()
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    return X_test


def log_reg():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    # print(acc_log)
    return acc_log


def svc():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train)*100, 2)
    # print(acc_svc)
    return acc_svc

def knn():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train)*100,2)
    # print(acc_knn)
    return acc_knn


def gaussian():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)
    # print(acc_gaussian)
    return acc_gaussian


def perceptron():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    # print(acc_perceptron)
    return acc_perceptron


def linear_SVC():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    # print(acc_linear_svc)
    return acc_linear_svc


def sgd():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    # print(acc_sgd)
    return acc_sgd


def decision_tree():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print(acc_decision_tree)
    # return acc_decision_tree
    return Y_pred


def random_forest():
    X_train, Y_train = xy_split_train()
    X_test = xy_split_test()
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)
    # return acc_random_forest
    return Y_pred

def evaluation():
    models = pd.DataFrame({
        'Model':['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
        'Score':[svc(), knn(), log_reg(),random_forest(),
                gaussian(), perceptron(), sgd(), linear_SVC(),
                decision_tree()]
    })
    models = models.sort_values(by = 'Score', ascending = False)
    print(models)
    print(models.dtypes)
    return models


def submission():
    Y_pred = decision_tree()
    train_df, test_df, combine = convert_fare()
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    print(submission.head())
    submission.to_csv('./output/submission_DT.csv', index=False)

submission()
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