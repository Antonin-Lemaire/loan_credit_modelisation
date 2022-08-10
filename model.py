import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from Kernel import get_data
import pickle


def custom_metric(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = mat.ravel()
    eval_result = (tp - fn) / (tp + fp + fn)
    eval_name = 'penalized_accuracy'
    is_higher_better = True
    return eval_name, eval_result, is_higher_better


def search(data):
    scaler = StandardScaler()
    model = LGBMClassifier(objective='binary')
    rebalance = SMOTE()
    pipe = Pipeline([('scaler', scaler), ("balance", rebalance), ('model', model)])

    param_grid = {"model__learning_rate": [0.2],
                  "model__num_iterations": [5000],
                  "model__n_estimators": [100]}

    clf = GridSearchCV(pipe, param_grid, cv=5, n_jobs=2)

    feats = [f for f in data.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X = data[feats]
    y = data['TARGET']
    selector = SelectKBest(f_classif, k=20)
    X_new = selector.fit_transform(X, y)
    print(selector.get_feature_names_out())

    clf.fit(X_new, y)

    print(clf.best_params_)
    selection = clf.best_estimator_

    y_pred = selection.predict(X_new)
    confu = confusion_matrix(y, y_pred)
    print(confu)
    _, score_value, __ = custom_metric(y, y_pred)
    print('penalized score : {}'.format(score_value))
    return clf


if __name__ == '__main__':
    training_set = pd.read_csv('train.csv')

    grid = search(training_set)
    filename = 'placeholder_mcdoctorate.sav'
    pickle.dump(grid.best_estimator_, open(filename, 'wb'))

