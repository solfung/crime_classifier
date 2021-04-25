import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from crime_classifier.util import *

model_path = "/Users/sfung/dev/crime_classifier/outputs/model.pkl"
model = pickle.load(open(model_path, 'rb'))

artifacts_path = "/Users/sfung/dev/crime_classifier/outputs/artifacts.pkl"
artifacts = pickle.load(open(artifacts_path, 'rb'))

app = Flask(__name__)

'''
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
Dates	DayOfWeek	PdDistrict	Address	X	Y
    
'''
import pandas as pd
def process_data(data, artifacts):
    #x_imp = artifacts["x_imp"]
    #y_imp = artifacts["y_imp"]
    #date_enc = artifacts["date_enc"]
    # build the df
    df = pd.DataFrame(data)
    print(df.columns)
    X, _ = build_features(df, False, artifacts)
    return X

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    #data["PdDistrict"]
    # assume data is a list of dict
    # each dict has Dates, Address, X, Y
    X = process_data(data, artifacts)
    Y_pred = model.predict_prob_a(X)
    Y_out = [[float(x) for x in row] for row in Y_pred]
    #pred_class = get_max_prediction(Y_pred)
    #categories = labelenc.classes_
    Y_out, categories, pred_class = 1,2,3
    output = {"probabilities": Y_out, "categories": categories, "predictions": pred_class}
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
