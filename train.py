import numpy as np 
import pandas as pd
import argparse 
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn_genetic import GAFeatureSelectionCV 
from sklearn.neighbors import KNeighborsClassifier as KNN
parser = argparse.ArgumentParser()
parser.add_argument('--feature_selection', default=False, action='store_true', help='enable feature selection')
parser.add_argument('--GA', default=False, action='store_true', help='Genetic Algorithm')
parser.add_argument('--NB', default=False, action='store_true', help='Naive Bayes')
parser.add_argument('--KNN', default=False, action='store_true', help='k nearest neighbor')
parser.add_argument('--SVM', default=False, action='store_true', help='support vector machine')
args = parser.parse_args()
df = pd.read_csv("archive/emails.csv")
x = df.iloc[:, 1:3001]
y = df.iloc[:, -1].values
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.25)
#feature selection
if args.feature_selection:
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    if args.GA :
        # use multinomialNB to evaluate genetic performance
        clf = MultinomialNB(alpha=1.9)
        GA = GAFeatureSelectionCV(
            estimator=clf,
            scoring="accuracy",
            cv=5,
            population_size=100,
            generations=20,
            n_jobs=-1)
        GA.fit(train_x, train_y)
        features = GA.best_features_
        print('Features:', train_x.columns[features])
        predict = GA.predict(test_x.values[:, features])
        print('Acc:', 100.*(np.mean(predict==test_y)))
if args.NB:
    clf = MultinomialNB(alpha=1.9)
elif args.KNN:
    clf = KNN(n_neighbors=5) #tune:weights, p
elif args.SVM:
    clf = svm.SVC(kernel='linear') #tune:rbf kernel, sigmoid
else:
    sys.exit('must specify classifier')
clf.fit(train_x, train_y)
predict = clf.predict(test_x)
print('Acc:', 100.*(np.mean(predict==test_y)))






