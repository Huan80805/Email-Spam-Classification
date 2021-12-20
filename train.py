import numpy as np 
import pandas as pd
import argparse 
import sys
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn_genetic import GAFeatureSelectionCV 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DCT
parser = argparse.ArgumentParser()
#feature selection
parser.add_argument('--feature_selection', default=False, action='store_true', help='enable feature selection')
parser.add_argument('--GA', default=False, action='store_true', help='Genetic Algorithm')
#classifier
parser.add_argument('--NB', default=False, action='store_true', help='Naive Bayes')
parser.add_argument('--alpha', default=1.0, type=float, help='naive bayes smoothing')
parser.add_argument('--KNN', default=False, action='store_true', help='k nearest neighbor')
parser.add_argument('--p', default=2, type=int, help='type of distance')
parser.add_argument('--SVM', default=False, action='store_true', help='support vector machine')
parser.add_argument('--kernel', default='linear', type=str, choices=['linear', 'sigmoid', 'rbf', 'poly'])
parser.add_argument('--RF', default=False, action='store_true', help='random forest') 
parser.add_argument('--DCT', default=False, action='store_true', help='decision tree') 
args = parser.parse_args()
seed = 7
np.random.seed(7)
random.seed(seed)
df = pd.read_csv("archive/emails.csv")
pd.set_option('display.max_columns', None)
x = df.iloc[:, 1:3001]
y = df.iloc[:, -1].values
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.25)
if args.NB:
    clf = MultinomialNB(alpha=args.alpha)
elif args.KNN:
    clf = KNN(n_neighbors=5, p=args.p, n_jobs=-1) #tune:weights, p
elif args.SVM:
    clf = svm.SVC(kernel=args.kernel, verbose=True)
elif args.RF:
    clf = RF(random_state=5, n_jobs=-1)
elif args.DCT:
    clf = DCT(random_state=5)
else:
    sys.exit('must specify classifier')
#feature selection
if args.feature_selection:
    if args.GA :
        # use multinomialNB to evaluate genetic performance
        GA = GAFeatureSelectionCV(
            estimator=clf,
            scoring="accuracy",
            cv=5,
            population_size=100,
            generations=50,
            n_jobs=-1)
        GA.fit(train_x, train_y)
        features = GA.best_features_
        print('Features:', train_x.columns[features])
        predict = GA.predict(test_x.values[:, features])
        print('test Acc:', 100.*(np.mean(predict==test_y)))
        predict = GA.predict(train_x.values[:, features])
        print('train Acc:', 100.*(np.mean(predict==train_y)))
else: 
    clf.fit(train_x, train_y)
    predict = clf.predict(test_x)
    print('test Acc:', 100.*(np.mean(predict==test_y)))
    predict = clf.predict(train_x)
    print('train Acc:', 100.*(np.mean(predict==train_y)))
    if args.DCT or args.RF:
        if args.RF:
            clf = clf.estimators_[0]
        plt.figure(figsize=(50,50))
        sort = np.argsort(clf.feature_importances_)
        print('important features:',train_x.columns[sort][:50])
        # tree.plot_tree(clf, 
        #             fontsize=1, 
        #             feature_names=train_x.columns,
        #             filled=True)
        # plt.savefig('tree.png', dpi=500)






