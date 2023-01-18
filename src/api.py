from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd
import util
import data_pipeline as data_pipeline
import preprocessing as preprocessing
import modelling as modelling

config_data = util.load_config()

ohe_data = util.pickle_load(config_data["ohe_path"])
le_data = util.pickle_load(config_data["le_path"])
le_label = util.pickle_load(config_data["le_label_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    policy_bind_date : str
    incident_date : str
    months_as_customer : int
    age : int
    policy_number : int
    policy_annual_premium : int
    insured_zip : int
    capital_gains : int
    capital_loss : int
    incident_hour_of_the_day : int
    total_claim_amount : int
    injury_claim : int
    property_claim : int
    vehicle_claim : int
    policy_deductable : str
    umbrella_limit : str
    number_of_vehicles_involved : str
    bodily_injuries : str
    witnesses : str
    auto_year : str
    policy_state : str
    policy_csl : str
    insured_sex : str
    insured_hobbies : str
    incident_type : str  
    collision_type : str
    incident_severity : str
    authorities_contacted : str
    incident_state : str
    incident_city : str
    property_damage : str
    police_report_available : str
    auto_make : str
    auto_model : str

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):
    # 0. Load config
    config_data = util.load_config()
    
    # 1. Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    # 2. Convert Dtype
    data = data_pipeline.type_data(data)
    data.columns = config_data['api_predictor']

    # 3. Check range data
    try:
        data_pipeline.check_data(data, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # 4. Split data into predictor and label
    data = data[config_data["predictor"]].copy()

    # 5. Split data into numerical and categorical for handling each type of data
    data_num, data_cat = preprocessing.splitNumCat(data)

    # 6. Imputed numerical data for any missing value
    data_num_imputed, imputer_num = preprocessing.imputerNum(data = data_num)

    # 7. Imputed Categorical data for any missing value
    data_cat_imputed, imputer_cat = preprocessing.imputerCat(data = data_cat)

    # 8. Encoding data categorical using OHE for nominal data and LE for ordinal data
    data_cat_ohe, encoder_ohe_col, encoder_ohe = preprocessing.OHEcat(data = data_cat_imputed)
    data_cat_le, encoder_le = preprocessing.LEcat(data = data_cat_imputed)

    # 9. Concatenate ohe and le encoded data
    data_cat_concat = pd.concat([data_cat_ohe,data_cat_le], axis = 1)

    # 10. Concatenate numerical data and categorical data
    data_concat = pd.concat([data_num_imputed, data_cat_concat], axis=1)

    # 11. Standardize value of train data
    data_clean, scaler = preprocessing.standardizeData(data = data_concat)
    
    # 12. Load x_train variable to equalize new data len columns with model fit len columns
    x_train, y_train = modelling.load_train_clean(config_data)
    
    # 13. Equalize the columns since OHE create 131 columns, with non existing value must have value = 0
    
    if len(data_clean.columns) != 131:
        d_col = set(data_clean.columns).symmetric_difference(set(x_train['nonbalance'].columns))
        
        for col in d_col:
            data_clean[col] = 0

    # 13. Predict the data
    y_pred = model_data["model_data"]["model_object"].predict(data_clean)

    if y_pred[0] == 0:
        y_pred = "TIDAK FRAUD."
    else:
        y_pred = "FRAUD."

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)