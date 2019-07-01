



from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn import neighbors
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import importlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
linear_model.LogisticRegression(solver='lbfgs')


def runModel(train_features, train_labels, test_features, test_labels):
    # create an instance based learner
    clf = KNeighborsClassifier()
    clf = clf.fit(train_features, train_labels)

    # predict the class for an unseen example
    results = clf.predict(test_features)
    score = metrics.accuracy_score(results, test_labels)
    return metrics.accuracy_score(results, test_labels)


def main():

    p_data = pd.read_csv("processed_data/cleaned_ordered_gtd.csv")
    # print(p_data)
    # print(p_data.isnull().sum())
    # p_data.info()

    X = p_data.iloc[:10000, 0:-1]
    y = p_data.iloc[:10000, -1]

    # scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)



    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    cf_mat = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('Confusion matrix:\n', cf_mat)


    # param_grid = [{'n_neighbors': list(range(1, 3)), 'p': [1, 2, 3, 4, 5]}]
    # clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)

    knn = KNeighborsClassifier()
    scores = model_selection.cross_val_score(knn, X, y, cv=10)
    print(scores.mean())
    param_grid = [{'n_neighbors': list(range(1, 3)), 'p': [1, 2, 3, 4, 5]}]
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
    clf.fit(X, y)
    print("\n Best parameters set found on development set:")
    print(clf.best_params_, "with a score of ", clf.best_score_)
    scores = model_selection.cross_val_score(clf.best_estimator_,X, y, cv=10)
    print(scores.mean())

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std, param))


main()