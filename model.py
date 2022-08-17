import lightgbm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle


def custom_loss(y_true, y_pred):

    gain = 0.1
    loss = -1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    actual_gain = (fn * loss + tn * gain)
    min_gain = (fn+tp) * loss
    max_gain = (tn+fp) * gain
    eval_result = (actual_gain - min_gain) / (max_gain - min_gain)
    eval_name = 'penalized_loss'
    is_higher_better = True
    # return eval_name, eval_result, is_higher_better
    return eval_result


def custom_metric(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = mat.ravel()
    eval_result = (tp - fn) / (tp + fp + fn)
    eval_name = 'penalized_accuracy'
    is_higher_better = True
    return eval_name, eval_result, is_higher_better


def search(data):
    scaler = StandardScaler()
    model = LGBMClassifier(learning_rate=0.2,
                           num_iterations=5000,
                           n_estimators=100,
                           objective='binary')
    rebalance = SMOTE()
    pipe = Pipeline([('scaler', scaler), ("balance", rebalance), ('model', model)])

    param_grid = {"model__learning_rate": [0.2],
                  "model__num_iterations": [5000],
                  "model__n_estimators": [100]}

    clf = GridSearchCV(pipe, param_grid, cv=5, n_jobs=2, scoring=make_scorer(custom_loss))

    feats = [f for f in data.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X = data[feats]
    y = data['TARGET']
    selector = SelectKBest(f_classif, k=20)
    X_new = selector.fit_transform(X, y)
    print(selector.get_feature_names_out())
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_new, y)
    clf.fit(Xtrain, ytrain)

    print(clf.best_params_)
    selection = clf.best_estimator_

    y_pred = selection.predict(Xtest)
    confu = confusion_matrix(ytest, y_pred)
    print(confu)
    score_value= custom_loss(ytest, y_pred)
    print('penalized score : {}'.format(score_value))
    return clf


if __name__ == '__main__':
    training_set = pd.read_csv('train.csv')
    feats = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT',
             'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
             'EXT_SOURCE_3', 'NAME_INCOME_TYPE_Working',
             'NAME_EDUCATION_TYPE_Higher education', 'BURO_DAYS_CREDIT_MIN',
             'BURO_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_UPDATE_MEAN',
             'BURO_CREDIT_ACTIVE_Active_MEAN', 'BURO_CREDIT_ACTIVE_Closed_MEAN',
             'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',
             'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
             'PREV_CODE_REJECT_REASON_XAP_MEAN', 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
             'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'CC_CNT_DRAWINGS_CURRENT_MAX']
    grid = search(training_set)
    filename = 'placeholder_mcdoctorate.sav'
    pickle.dump(grid.best_estimator_, open(filename, 'wb'))

