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


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result


def custom_loss(z, data):
    t = data
    y = sigmoid(z)
    grad = y - t
    hess = y * (1 - y)
    return grad, hess


def custom_metric(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = mat.ravel()
    eval_result = (tp - fn) / (tp + fp + fn)
    eval_name = 'penalized_accuracy'
    is_higher_better = True
    return eval_name, eval_result, is_higher_better


def score_cost(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True )
    cost_score = (9*report['0.0']['precision']+ report['0.0']['recall']+report['1.0']['precision']+9*report['1.0']['recall'])/20
    return cost_score


def custom_asymmetric_train(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual<0, -2*10.0*residual, -2*residual)
    hess = np.where(residual<0, 2*10.0, 2.0)
    return grad, hess


def search(data):
    scaler = StandardScaler()
    model = LGBMClassifier(objective=custom_loss)
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

