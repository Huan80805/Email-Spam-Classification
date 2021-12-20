import numpy as np
import pandas as pd
import argparse
import sys
import random
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from genetic_selection import GeneticSelectionCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def feat_select(x, y, feature_selection=None):

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.25)
    selected_train_x, selected_test_x = None, FileNotFoundError
    if feature_selection == "GA":
        GA = GeneticSelectionCV(
            estimator= MultinomialNB(alpha=1.9),
            scoring="accuracy",
            cv=5,
            verbose=2,
            max_features=100,
            n_population=100,
            n_generations=40,
        )
        GA.fit(train_x, train_y)
        features = GA.support_
        selected_train_x = train_x.values[:, features]
        selected_test_x = test_x.values[:, features]

    elif feature_selection == "tree-based":
        model = ExtraTreesClassifier(n_estimators=50)
        model = model.fit(train_x, train_y)
        model = SelectFromModel(model, prefit=True)
        selected_train_x = model.transform(train_x.values)
        selected_test_x = model.transform(test_x.values)

    elif feature_selection == 'l1-based':
        lsvc = LinearSVC(C=0.01, penalty="l1", tol=1e-4, max_iter=10000,
                            dual=False).fit(train_x, train_y)
        model = SelectFromModel(lsvc, prefit=True)
        selected_train_x = model.transform(train_x.values)
        selected_test_x = model.transform(test_x.values)
    elif feature_selection == 'uni-var':
        selector = SelectKBest(chi2, k=500)
        selector.fit_transform(train_x, train_y)
        feature_selected = selector.get_support(True)
        selected_train_x = train_x.values[:, feature_selected]
        selected_test_x = test_x.values[:, feature_selected]
    elif feature_selection == None:
        return train_x, test_x, train_y, test_y
    return selected_train_x, selected_test_x, train_y, test_y

def predict(clf, model_type, selected_train_x, selected_test_x, train_y, test_y, feature_selection):
    clf.fit(selected_train_x, train_y)
    predict = clf.predict(selected_test_x)
    print(f'{model_type} with feature selction by {feature_selection} acc: ',
            100.*(np.mean(predict == test_y)))

parser = argparse.ArgumentParser()
parser.add_argument('--feature_selection', default=None, type=str,
                    choices=['GA', 'tree-based', 'l1-based', 'uni-var'], help='enable feature selection')
parser.add_argument('--NB', default=False,
                    action='store_true', help='Naive Bayes')
parser.add_argument('--KNN', default=False,
                    action='store_true', help='k nearest neighbor')
parser.add_argument('--SVM', default=False,
                    action='store_true', help='support vector machine')
parser.add_argument('--RF', default=False,
                    action='store_true', help='random forest')
parser.add_argument('--all', default=False,
                    action='store_true', help='run all classifier')                   
        
args = parser.parse_args()
def predict(clf, model_type, selected_train_x, selected_test_x, train_y, test_y, feature_selection):
    clf.fit(selected_train_x, train_y)
    predict = clf.predict(selected_test_x)
    print(f'{model_type} with feature selction by {feature_selection} acc: ',
            100.*(np.mean(predict == test_y)))

seed = 7
np.random.seed(seed)
random.seed(seed)
df = pd.read_csv("archive/emails.csv")
x = df.iloc[:, 1:3001]
y = df.iloc[:, -1].values
train_x, test_x, train_y, test_y = feat_select(x, y, args.feature_selection)
if args.NB:
    model_type = "NB"
    clf = MultinomialNB(alpha=1.9)
    predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)
elif args.KNN:
    model_type = "KNN"
    neighbors_param = [1, 3, 5, 7, 9]
    for neighbors in neighbors_param:
        clf = KNN(n_neighbors=neighbors)  # tune:weights, p
        print(f'{neighbors}-NN ')
        predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)

elif args.SVM:
    choices = ['linear', 'sigmoid', 'rbf']
    for kernel_func in choices:
        clf = svm.SVC(kernel=kernel_func)
        model_type = f"SVM kernel is {kernel_func}"
        predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)

elif args.RF:

    model_type = "RandomForest"
    estimator_param = [50, 100, 150]
    depth_param = [5, 10, 15, 20, 25, 30, 40]

    for estimator in estimator_param:
        for depth in depth_param:
            clf = RF(n_estimators=estimator, random_state=3,
                     max_depth=depth, criterion="entropy")
            print(f'# of trees: {estimator}\ndepth: {depth}')
            predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)
elif args.all:
    model_type = "NB"
    clf = MultinomialNB(alpha=1.9)
    predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)
    model_type = "KNN"
    neighbors_param = [1, 3, 5, 7, 9]
    for neighbors in neighbors_param:
        clf = KNN(n_neighbors=neighbors)  # tune:weights, p
        print(f'{neighbors}-NN ')
        predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)
    choices = ['linear', 'sigmoid', 'rbf']
    for kernel_func in choices:
        clf = svm.SVC(kernel=kernel_func)
        model_type = f"SVM kernel is {kernel_func}"
        predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)
    model_type = "RandomForest"
    estimator_param = [50, 100, 150]
    depth_param = [5, 10, 15, 20, 25, 30, 40]

    for estimator in estimator_param:
        for depth in depth_param:
            clf = RF(n_estimators=estimator, random_state=3,
                     max_depth=depth, criterion="entropy")
            print(f'# of trees: {estimator}\ndepth: {depth}')
            predict(clf, model_type, train_x, test_x, train_y, test_y, args.feature_selection)

    
else:
    sys.exit('must specify classifier')







