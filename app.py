from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import datetime
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
def runcheck(df):
    if all(df['Attending Physician'] == 'ADIB, KEENAN'):
        df.drop('Attending Physician',axis=1,inplace=True)
    if all(df['MS DRG Description'] == 'ACUTE ISCHEMIC STROKE W USE OF THROMBOLYTIC AGENT'):
        df.drop('MS DRG Description',axis=1,inplace=True)
    final_df = pd.get_dummies(data=df)
    return final_df

def runcheckLOS(df):
    if all(df['MS DRG Description'] == 'ACUTE ISCHEMIC STROKE W USE OF THROMBOLYTIC AGENT'):
        df.drop('MS DRG Description',axis=1,inplace=True)
    else:
	    final_df = pd.get_dummies(data=df)
    return final_df

@app.route('/Costs', methods=['POST'])
@cross_origin()
def predict():
    losp = joblib.load('perfect_predictor_averageLOS.pkl')
    model_columns_los = joblib.load('model_columns_los.pkl')
    clf = joblib.load('linear_regression_model_for_total_direct_variable.pkl')
    olf = joblib.load('linear_regression_model_for_Total_Other.pkl')
    elf = joblib.load('linear_regression_model_for_charges.pkl')
    model_columns = joblib.load('model_columns_new.pkl')

    DRG = request.get_json()['DRG'] 
    diabeticFactor = request.get_json()['D_Factor'] 
    DateP = request.get_json()['Admit Date']
    PhysicianName = request.get_json()['Attending Physician']

    final_dataset_1 = pd.DataFrame([DRG],columns=["MS DRG Description"])
    input_df_1 = runcheckLOS(final_dataset_1)
    for col in model_columns_los:
        if col not in input_df_1.columns:
            input_df_1[col] = 0
    input_df_1 = input_df_1[model_columns_los]
    X = input_df_1.iloc[:,:].values
    LOS = losp.predict(X) * diabeticFactor
    LOS = float(LOS[0])
    final_dataset = pd.DataFrame([[DateP, PhysicianName, DRG, LOS]],columns=["Admit Date", "Attending Physician","MS DRG Description","LOS"])
    epoch_0 = datetime.datetime(1970,1,1)
    final_dataset['Admit Date']=(pd.to_datetime(final_dataset['Admit Date'])-epoch_0) / np.timedelta64(1,'D')

    input_df = runcheck(final_dataset)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
	
	#added below line for correct column ordering
    input_df = input_df[model_columns] 
    X = input_df.iloc[:,:].values
    totalDirectVariable = clf.predict(X)
    totalOther = olf.predict(X)
    totalCharges = elf.predict(X)
    return jsonify({'total_direct_variable': list(totalDirectVariable)},{'total_Charges':list(totalCharges)},{'total_other':list(totalOther)},{'LOS':LOS})

if __name__=="__main__":
    losp = joblib.load('perfect_predictor_averageLOS.pkl')
    clf = joblib.load('linear_regression_model_for_total_direct_variable.pkl')
    olf = joblib.load('linear_regression_model_for_Total_Other.pkl')
    elf = joblib.load('linear_regression_model_for_charges.pkl')
    model_columns = joblib.load('model_columns_new.pkl')
    model_columns_los = joblib.load('model_columns_los.pkl')
    app.run()