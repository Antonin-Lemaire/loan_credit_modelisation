import numpy as np
import pandas as pd
import os
import glob
from sklearn.impute import SimpleImputer
import zipfile
import urllib.request


def download_data():
    url = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/"\
          "Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring"\
          "/Projet+Mise+en+prod+-+home-credit-default-risk.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    zip_file_object.extractall('Inputs')
    print('Data downloaded')


def one_hot_encoder(df):
    original_columns = list(df.columns)
    cat_col = [var for var in df.columns if df[var].dtype == 'object']
    df = pd.get_dummies(df, columns=cat_col, dummy_na=True)
    new_columns = [v for v in df.columns if v not in original_columns]
    return df, new_columns


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None):
    bureau = pd.read_csv('Inputs/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('Inputs/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb)
    bureau, bureau_cat = one_hot_encoder(bureau)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None):
    prev = pd.read_csv('Inputs/previous_application.csv', nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    return prev_agg


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('Inputs/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None):
    ins = pd.read_csv('Inputs/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    return ins_agg


def credit_card_balance(num_rows=None):
    cc = pd.read_csv('Inputs/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    return cc_agg


def get_train_test(debug=False):
    """
    loads and cleans data to feed Model.py
    Parameters
    ----------
    debug : bool
        if true, returns a smaller dataset for testing purposes
    Returns
    -------
    df : DataFrame
        Cleaned up dataset, nan free, with feature TARGET for modeling
    """

    if debug:
        n_rows = 10
    else:
        n_rows = None

    file_list = glob.glob(os.path.join(os.getcwd(), "Inputs", "*.csv"))
    df = pd.read_csv('Inputs/application_train.csv', nrows=n_rows)
    test_df = pd.read_csv('Inputs/application_test.csv', nrows=n_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()

    # remove the 4 np.nan in gender as non-representative
    df = df[df['CODE_GENDER'] != 'XNA']

    # encode binary features
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # encode categorical features using above function
    df, cat_cols = one_hot_encoder(df)

    # weird values (1000 years employed, that's a new record)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # TODO : appending other datasets / add normalisation / mode debug
    return df


def nan_rate(var, df):
    size = df.shape[0]
    rate = 1 - (df[var].count() / size)
    return rate


def get_data(debug=False):
    if debug:
        num_rows = 10
    else:
        num_rows = None
    if not os.path.exists('Inputs'):
        download_data()

    df = get_train_test(debug)
    bureau = bureau_and_balance()
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau

    prev = previous_applications(num_rows)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev

    pos = pos_cash(num_rows)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos

    ins = installments_payments(num_rows)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins

    cc = credit_card_balance(num_rows)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc

    # delete columns lacking in completion, because completion is not correlated to the target here
    low_value = [var for var in df.columns if nan_rate(var, df) > 0.8]
    df = df.drop(columns=low_value)

    #
    for v in [var for var in df.columns if var not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]:
        df[v] = df[v].fillna(df[v].median())

    valuable_features = [val for val in df.columns if df[val].nunique() > 1]
    df = df[valuable_features]

    df.replace([np.inf, -np.inf], 0, inplace=True)

    test_set = df[df['TARGET'].isna() == True]
    train_set = df[df['TARGET'].isna() == False]
    return train_set, test_set


if __name__ == "__main__":
    data_train, data_test = get_data()
    data_train.to_csv('train.csv')
    data_test.to_csv('test.csv')
    print(data_test.shape)
