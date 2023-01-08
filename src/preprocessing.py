import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util as util
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # return 3 set of data
    return train_set, valid_set, test_set

def remove_outlier(set_data):
    set_data = set_data.copy()
    list_of_set_data = list()

    # set_data = set_data.drop(['umbrella_limit'], axis = 1)
    # config_data_num = config_data['int32_col'].copy()
    # config_data_num = [x for x in config_data_num if x != 'umbrella_limit']

    for col in set_data[config_data['int32_col']]:
        q1 = set_data[col].quantile(0.25)
        q3 = set_data[col].quantile(0.75)
        iqr = q3 - q1

        set_data_cleaned = set_data[~((set_data[col] < (q1 - 1.5*iqr)) |
                                    (set_data[col] > (q3 + 1.5*iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())

    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == len(config_data['int32_col'])].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned

def balancing(data):
    x_data = data.drop(columns = config_data['label'])
    y_data = data[config_data['label']]

    x_over, y_over = RandomOverSampler(random_state=42).fit_resample(x_data, y_data)
    x_smote, y_smote = SMOTE(random_state=42).fit_resample(x_data, y_data)

    train_set_smote = pd.concat([x_smote, y_smote], axis = 1)
    train_set_over = pd.concat([x_over, y_over], axis = 1)

    return x_smote, y_smote, x_over, y_over

def splitxy(set_data):
    x_data = set_data.drop(columns = config_data['label'], axis = 1)
    y_data = set_data[config_data['label']]

    return x_data, y_data

def splitNumCat(set_data):
    config_data = util.load_config()
    
    numerical_col = config_data['int32_col']
    categorical_col = config_data['object_predictor']

    x_train_num = set_data[numerical_col]
    x_train_cat = set_data[categorical_col]

    return  x_train_num, x_train_cat

def imputerNum(data, imputer = None):
    if imputer == None:
        # Create imputer based on median value
        imputer = SimpleImputer(missing_values = np.nan,
                                strategy = "median")
        imputer.fit(data)

    # Transform data dengan imputer
    # else:
    data_imputed = pd.DataFrame(imputer.transform(data),
                                index = data.index,
                                columns = data.columns)
    
    # Convert data_imputed to int32
    data_imputed = data_imputed.astype('int32')
    
    return data_imputed, imputer

def imputerCat(data, imputer = None):
    data.umbrella_limit = data.umbrella_limit.replace('-1000000','1000000')

    for col in ['collision_type','property_damage','police_report_available']:
        data[col] = data[col].replace('?', 'UNKNOWN')
        
    if imputer == None:
        # Create Imputer
        imputer = SimpleImputer(missing_values = np.nan,
                                strategy = 'constant',
                                fill_value = 'UNKNOWN')
        imputer.fit(data)

    # Transform data with imputer
    data_imputed = imputer.transform(data)
    data_imputed = pd.DataFrame(data_imputed,
                                index = data.index,
                                columns = data.columns)

    return data_imputed, imputer

def OHEcat(data, encoder_col = None, encoder = None) -> pd.DataFrame:
    
    nominal = ['policy_state','policy_csl','policy_deductable','insured_sex','insured_hobbies','collision_type',
                'authorities_contacted','incident_state','incident_city','property_damage','police_report_available',
                'auto_make','auto_model']

    data_ohe = data[nominal]

    if encoder == None:
        # Create Object
        encoder = OneHotEncoder(handle_unknown = 'ignore',
                                drop = 'if_binary')
        encoder.fit(data_ohe)
        encoder_col = encoder.get_feature_names_out(data_ohe.columns)
    
    
    # Transform the data
    data_encoded = encoder.transform(data_ohe).toarray()
    data_encoded = pd.DataFrame(data_encoded,
                                index = data_ohe.index,
                                columns = encoder_col)

    return data_encoded, encoder_col, encoder

def LEcat(data, encoder = None) -> pd.DataFrame:
    
    ordinal = ['incident_type','witnesses','incident_severity','auto_year','umbrella_limit','bodily_injuries',
            'number_of_vehicles_involved']
    
    data_le = data[ordinal]

    bodily_injuries = ['0','1','2']
    witnesses = ['0','1','2','3']
    umbrella_limit = ['0', '1000000', '2000000', '3000000', '4000000', '5000000', '6000000',
                      '7000000','8000000','9000000','10000000']
    incident_severity = ['Trivial Damage','Minor Damage','Major Damage','Total Loss']
    incident_type = ['Parked Car','Single Vehicle Collision','Multi-vehicle Collision','Vehicle Theft']
    auto_year = sorted(data_le.auto_year.unique())
    number_of_vehicles_involved = ['1','2','3','4']

    if encoder == None:
        # Create object
        encoder = OrdinalEncoder(categories=[incident_type,witnesses,incident_severity,auto_year,
                                   umbrella_limit,bodily_injuries,number_of_vehicles_involved])
        encoder.fit(data_le)

    ## Transform the data
    data_encoded = encoder.transform(data_le)
    data_encoded = pd.DataFrame(data_encoded,
                                index = data_le.index,
                                columns = data_le.columns)

    return data_encoded, encoder

def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()

    # Fit le
    le_encoder.fit(data_tobe_fitted)

    # Save le object
    util.pickle_dump(le_encoder, le_path)

    # Return trained le
    return le_encoder

def concat_numcat(data_num, data_cat_ohe, data_cat_le):
    data_cat = pd.concat([data_cat_ohe, data_cat_le], axis=1)
    data_concat = pd.concat([data_num, data_cat], axis=1)

    return data_concat

def standardizeData(data, scaler =None):
    if scaler == None:
        # Create Fit Scaler
        scaler = StandardScaler()
        scaler.fit(data)

    # Transform data
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled,
                                index = data.index,
                                columns = data.columns)
    
    return data_scaled, scaler

# Handling valid data using previous function
## Splitting into num & cat, imputer num & cat, ohe & le, standarization
def handlingData(set_data):
    # Load config
    config_data = util.load_config()
    
    # Split data into x_data, y_data
    x_data, y_data = splitxy(set_data)

    # Split x_data into numerical and categorical
    x_data_num, x_data_cat = splitNumCat(x_data)

    # Encoding categorical data by separating it into OHE and Ordinal..
    nominal = ['policy_state','policy_csl','policy_deductable','insured_sex','insured_hobbies','collision_type',
                'authorities_contacted','incident_state','incident_city','property_damage','police_report_available',
                'auto_make','auto_model']
    ordinal = ['incident_type','witnesses','incident_severity','auto_year','umbrella_limit','bodily_injuries',
            'number_of_vehicles_involved']

    # Impute num data
    x_data_num_imputed, imputer_num_ = imputerNum(data = x_data_num, imputer = imputer_num)

    # Impute cat data
    x_data_cat_imputed, imputer_cat_ = imputerCat(data = x_data_cat, imputer = imputer_cat)

    x_data_cat_ohe, encoder_col_, encoder_ = OHEcat(x_data_cat_imputed, encoder_ohe_col,
                                                            encoder_ohe)
    x_data_cat_le, encoder_ = LEcat(x_data_cat_imputed, encoder_le)

    x_data_cat_concat = pd.concat([x_data_cat_ohe, x_data_cat_le], axis=1)
        
    # Concatenate data numeric and categorical
    x_data_concat = pd.concat([x_data_num_imputed, x_data_cat_concat], axis = 1)

    # Standardize data using standarscaler
    x_data_clean, scaler_ = standardizeData(x_data_concat, scaler)

    y_data_clean = y_data.map(dict(Y=1, N=0))

    train_set_clean = pd.concat([x_data_clean, y_data_clean], axis=1)

    x_smote_set, y_smote_set, x_over_set, y_over_set = balancing(train_set_clean)

    return x_data_clean, y_data_clean, x_smote_set, y_smote_set, x_over_set, y_over_set

if __name__ == "__main__":
    # 1. load Configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Split data train into x and y
    x_train, y_train = splitxy(train_set)

    # 4. Split data into numerical and categorical for handling each type of data
    x_train_num, x_train_cat = splitNumCat(x_train)

    # 5. Imputed numerical data for any missing value
    x_train_num_imputed, imputer_num = imputerNum(data = x_train_num)

    # 6. Imputed Categorical data for any missing value
    x_train_cat_imputed, imputer_cat = imputerCat(data = x_train_cat)

    # 7. Encoding data categorical using OHE for nominal data and LE for ordinal data
    x_train_cat_ohe, encoder_ohe_col, encoder_ohe = OHEcat(data = x_train_cat_imputed)
    x_train_cat_le, encoder_le = LEcat(data = x_train_cat_imputed)

    # 8. Concatenate ohe and le encoded data
    x_train_cat_concat = pd.concat([x_train_cat_ohe,x_train_cat_le], axis = 1)

    # 9. Concatenate numerical data and categorical data
    x_train_concat = pd.concat([x_train_num_imputed, x_train_cat_concat], axis=1)

    # 10. Standardize value of train data
    x_train_clean, scaler = standardizeData(data = x_train_concat)

    # 11. Change class label with Y into 1 and N into 0
    y_train_clean = y_train.map(dict(Y=1, N=0))

    le_fit(config_data["label_categories"], config_data["le_label_path"])

    # 12. Concatenate x_data and y_data into train_set
    train_set_clean = pd.concat([x_train_clean, y_train_clean], axis=1)

    # 13. Balancing train data using SMOTE and Oversampling
    x_smote, y_smote, x_over, y_over = balancing(train_set_clean)

    # 14. Handling valid and test data
    x_valid_clean, y_valid_clean, \
    x_valid_smote_clean, y_valid_smote_clean, \
    x_valid_over_clean, y_valid_over_clean  = handlingData(valid_set)

    x_test_clean, y_test_clean, \
    x_test_smote_clean, y_test_smote_clean, \
    x_test_over_clean, y_test_over_clean = handlingData(test_set)

    # 15. Dump training set
    x_train_final = {
        "nonbalance" : x_train_clean,
        "smote" : x_smote,
        "oversampling" : x_over
    }

    y_train_final = {
        "nonbalance" : y_train_clean,
        "smote" : y_smote,
        "oversampling" : y_over
    }

    # Save dataset
    util.pickle_dump(x_train_final, config_data['train_set_clean'][0])
    util.pickle_dump(y_train_final, config_data['train_set_clean'][1])

    util.pickle_dump(x_valid_clean, config_data['valid_set_clean'][0])
    util.pickle_dump(y_valid_clean, config_data['valid_set_clean'][1])

    util.pickle_dump(x_test_clean, config_data['test_set_clean'][0])
    util.pickle_dump(y_test_clean, config_data['test_set_clean'][1])


