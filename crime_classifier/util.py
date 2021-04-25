import pandas as pd
import numpy as np
import importlib

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

def get_max_prediction(Y):
    return np.argmax(Y)

def convert_dates(df):
    df["dates_dt"] = pd.to_datetime(df.Dates)
    # date proceessing
    df["month"] = df.dates_dt.dt.month
    df["dow"] = df.dates_dt.dt.dayofweek
    df["year"] = df.dates_dt.dt.year
    # time of day
    df["hour"] = df.dates_dt.dt.hour
    df["minute"] = df.dates_dt.dt.minute

def impute_bad_xy(df, x_imp=None, y_imp=None):
    # we identified oultiers/missing data frorm EDA, set them to null and usse mean imputation
    x_badval = -120.5
    y_badval = 90
    missing_xy_ix = (df.X==x_badval) & (df.Y==y_badval)
    df.loc[missing_xy_ix, "X"] = np.nan
    df.loc[missing_xy_ix, "Y"] = np.nan
    if not x_imp:
        x_imp = SimpleImputer().fit([[x] for x in df["X"]]) #.reshape(-1,1)
    df.X = x_imp.transform([[x] for x in df["X"]])
    df.X.isnull().sum()
    if not y_imp:
        y_imp = SimpleImputer().fit([[x] for x in df["X"]]) #.reshape(-1,1)
    df.Y = x_imp.transform([[x] for x in df["X"]])
    df.Y.isnull().sum()
    return x_imp, y_imp


# tr_df.hour tr_df.minutee
def tod_circular(hour, minute):
    TOT_MIN = 24*60
    point_in_day = (hour*60 + minute )/ TOT_MIN
    # 0 -> 24 hours "x": 0 to 0 (sin), "y": 1 to 1 (cosine)
    # circular radians = 0 to 2pi
    x = np.sin(2*np.pi * point_in_day)
    y = np.cos(2*np.pi * point_in_day)
    return x, y

def set_tod_xy(df):
    TOT_MIN = 24*60
    df['tod_x'] = np.sin(2*np.pi* (df.hour*60 + df.minute )/ TOT_MIN)
    df['tod_y'] = np.cos(2*np.pi* (df.hour*60 + df.minute )/ TOT_MIN)

def set_intersection(df):
    df['address_is_intersection'] = [' / ' in str(x) for x in df.Address]
    #tr_df.address_is_intersection.value_counts()

def categorize_dates(df, date_enc=None):
    onehot_cols = 'month,dow,year'.split(',')
    if not date_enc:
        date_enc = OneHotEncoder()
        date_enc.fit(df[onehot_cols])
    X_date = date_enc.transform(df[onehot_cols])
    X_date = csr_matrix.todense(X_date)
    return X_date, date_enc

def build_features(df_orig, fit, fit_objs=None):
    df = pd.DataFrame(df_orig)
    convert_dates(df)
    if not fit:
        x_imp, y_imp = impute_bad_xy(df, fit_objs["x_imp"], fit_objs["y_imp"])
        X_date, date_enc = categorize_dates(df, fit_objs["date_enc"])
    else:
        x_imp, y_imp = impute_bad_xy(df)
        X_date, date_enc = categorize_dates(df)
    set_tod_xy(df)
    # encode categorical to one hoot
    set_intersection(df)
    cols_to_keep = "X,Y,tod_x,tod_y,address_is_intersection".split(',')
    X = df[cols_to_keep]
    #tr_df[cols_to_keep].head()
    X = np.hstack([X, X_date])
    artifacts = {"x_imp": x_imp, "y_imp":y_imp, "date_enc":date_enc}
    return X, artifacts