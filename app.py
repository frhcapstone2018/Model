from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

def runcheck(df):
    if all(df['Attending Physician'] == 'ALOKA, FERAS'):
        df.drop('Attending Physician',axis=1,inplace=True)
    if all(df['MS DRG Description'] == 'ACUTE ISCHEMIC STROKE W USE OF THROMBOLYTIC AGENT'):
        df.drop('MS DRG Description',axis=1,inplace=True)
    final_df = pd.get_dummies(data=df)
    return final_df
@app.route('/')
def index():
    return '<h1> Hello World </h1>'

@app.route('/', methods=['POST'])
def predict():
    AgeP = request.get_json()['Age']
    DateP = request.get_json()['Admit Date']
    PhysicianName = request.get_json()['Attending Physician']
    DRG = request.get_json()['DRG']
    LOS =  request.get_json()['LOS']
     
    final_dataset = pd.DataFrame([[AgeP, DateP, PhysicianName, DRG, LOS]],columns=["Age", "Admit Date", "Attending Physician","MS DRG Description","LOS"])
    epoch_0 = datetime.datetime(1970,1,1)
    final_dataset['Admit Date']=(pd.to_datetime(final_dataset['Admit Date'])-epoch_0) / np.timedelta64(1,'D')

    input_df = runcheck(final_dataset)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]
     
    X = input_df.iloc[:,:].values
    prediction = clf.predict(X)
    return jsonify({'prediction': list(prediction)})

if __name__=="__main__":
    clf = joblib.load('linear_regression_model_for_charges.pkl')
    model_columns = joblib.load('model_columns.pkl')
    app.run()