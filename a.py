from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

def runcheck(df):
    if all(df[2] == 'ALOKA, FERAS'):
        df.drop(2,axis=1,inplace=True)
    if all(df[3] == 'ACUTE ISCHEMIC STROKE W USE OF THROMBOLYTIC AGENT '):
        df.drop(3,axis=1,inplace=True)
    final_df = pd.get_dummies(data=df)
    return final_df

@app.route('/', methods=['POST'])
def predict():
     AgeP = request.get_json()['Age']
     DateP = request.get_json()['Admit Date']
     PhysicianName = request.get_json()['Attending Physician']
     DRG = request.get_json()['DRG']
     LOS =  request.get_json()['LOS']
     
     final_dataset = pd.DataFrame([[AgeP, DateP, PhysicianName, DRG, LOS]])
     epoch_0 = datetime.datetime(1970,1,1)
     final_dataset[1]=(pd.to_datetime(final_dataset[1])-epoch_0) / np.timedelta64(1,'D')
     input_df = runcheck(final_dataset)

     for col in model_columns:
          if col not in input_df.columns:
               input_df[col] = 0
     
     X = input_df.iloc[:,:].values
     
     # json_ = request.get_json()
     # #query_df = pd.DataFrame(json_)
     # query_df = pd.DataFrame({'Age': 20,'Admit Date': 190,'Attending Physician': 2,'MS DRG Description': 3,'LOS': 3}, index=[0])
     # query = pd.get_dummies(query_df)
     print(X, "x---")
     prediction = clf.predict(X)
     return jsonify({'prediction': list(prediction)})

if __name__=="__main__":
    clf = joblib.load('linear_regression_model_for_charges.pkl')
    model_columns = joblib.load('model_columns.pkl')
    app.run()