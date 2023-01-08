from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import util as util
import os
import copy

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw Dataset Dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add csv files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input_data, params, api = False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        assert input_data.select_dtypes("datetime").columns.to_list() == \
            params["datetime_columns"], "an error occurs in datetime column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            params["int32_col"], "an error occurs in int32 column(s)."
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_predictor"], "an error occurs in object column(s)."

        assert set(input_data.policy_deductable).issubset(set(params["policy_deductable"])), \
        "an error occurs in policy_deductable range."
        assert set(input_data.umbrella_limit).issubset(set(params["umbrella_limit"])), \
        "an error occurs in umbrella_limit range."
        assert set(input_data.number_of_vehicles_involved).issubset(set(params["number_of_vehicles_involved"])), \
        "an error occurs in number_of_vehicles_involved range."
        assert set(input_data.bodily_injuries).issubset(set(params["bodily_injuries"])), \
        "an error occurs in bodily_injuries range."
        assert set(input_data.witnesses).issubset(set(params["witnesses"])), \
        "an error occurs in witnesses range."
        assert set(input_data.auto_year).issubset(set(params["auto_year"])), \
        "an error occurs in auto_year range."
        assert set(input_data.policy_state).issubset(set(params["policy_state"])), \
        "an error occurs in policy_state range."
        assert set(input_data.policy_csl).issubset(set(params["policy_csl"])), \
        "an error occurs in policy_csl range."
        assert set(input_data.insured_sex).issubset(set(params["insured_sex"])), \
        "an error occurs in insured_sex range."
        assert set(input_data.insured_hobbies).issubset(set(params["insured_hobbies"])), \
        "an error occurs in insured_hobbies range."
        assert set(input_data.incident_type).issubset(set(params["incident_type"])), \
        "an error occurs in incident_type range."
        assert set(input_data.collision_type).issubset(set(params["collision_type"])), \
        "an error occurs in collision_type range."
        assert set(input_data.incident_severity).issubset(set(params["incident_severity"])), \
        "an error occurs in incident_severity range."
        assert set(input_data.authorities_contacted).issubset(set(params["authorities_contacted"])), \
        "an error occurs in authorities_contacted range."
        assert set(input_data.incident_state).issubset(set(params["incident_state"])), \
        "an error occurs in incident_state range."
        assert set(input_data.incident_city).issubset(set(params["incident_city"])), \
        "an error occurs in incident_city range."
        assert set(input_data.property_damage).issubset(set(params["property_damage"])), \
        "an error occurs in property_damage range."
        assert set(input_data.police_report_available).issubset(set(params["police_report_available"])), \
        "an error occurs in police_report_available range."
        assert set(input_data.auto_make).issubset(set(params["auto_make"])), \
        "an error occurs in auto_make range."
        assert set(input_data.auto_model).issubset(set(params["auto_model"])), \
        "an error occurs in auto_model range."

        assert input_data.months_as_customer.between(params["months_as_customer"][0], params["months_as_customer"][1]).sum() == \
            len(input_data), "an error occurs in months_as_customer range."
        assert input_data.age.between(params["age"][0], params["age"][1]).sum() == \
            len(input_data), "an error occurs in age range."
        assert input_data.policy_number.between(params["policy_number"][0], params["policy_number"][1]).sum() == \
            len(input_data), "an error occurs in policy_number range."
        assert input_data.policy_annual_premium.between(params["policy_annual_premium"][0], params["policy_annual_premium"][1]).sum() == \
            len(input_data), "an error occurs in policy_annual_premium range."
        assert input_data.insured_zip.between(params["insured_zip"][0], params["insured_zip"][1]).sum() == \
            len(input_data), "an error occurs in insured_zip range."
        assert input_data["capital_gains"].between(params["capital_gains"][0], params["capital_gains"][1]).sum() == \
            len(input_data), "an error occurs in capital_gains range."
        assert input_data["capital_loss"].between(params["capital_loss"][0], params["capital_loss"][1]).sum() == \
            len(input_data), "an error occurs in capital_loss range."
        assert input_data.incident_hour_of_the_day.between(params["incident_hour_of_the_day"][0], params["incident_hour_of_the_day"][1]).sum() == \
            len(input_data), "an error occurs in incident_hour_of_the_day range."
        assert input_data.total_claim_amount.between(params["total_claim_amount"][0], params["total_claim_amount"][1]).sum() == \
            len(input_data), "an error occurs in total_claim_amount range."
        assert input_data.injury_claim.between(params["injury_claim"][0], params["injury_claim"][1]).sum() == \
            len(input_data), "an error occurs in injury_claim range."
        assert input_data.property_claim.between(params["property_claim"][0], params["property_claim"][1]).sum() == \
            len(input_data), "an error occurs in property_claim range."
        assert input_data.vehicle_claim.between(params["vehicle_claim"][0], params["vehicle_claim"][1]).sum() == \
            len(input_data), "an error occurs in vehicle_claim range."
   
    else:
        # In case checking data from api
        object_columns = params["object_predictor"]

        # Max column not used as predictor
        int_columns = params["int32_col"]

        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            int_columns, "an error occurs in int32 column(s)."

def type_data(input_data):
    """Change raw dataset type.
    Change raw dataset datetime, float into int, int64 to int32 and convert few int columns into object
    """
    # 1. Load Config data
    config_data = util.load_config()
    
    # 2. Change datetime object
    for col in config_data["datetime_columns"]:
        input_data[col] = pd.to_datetime(input_data[col])

    # 3. change float columns type into int32
    input_data = input_data.astype({col: 'int32' for col in input_data.select_dtypes('float64').columns})

    # 4. Chane int64 col into int32 col
    input_data = input_data.astype({col: 'int32' for col in input_data.select_dtypes('int64').columns})

    # 5. define data into datetime, int32 and object format
    raw_dataset_date = input_data[config_data['datetime_columns']]
    raw_dataset_num = input_data[config_data['int32_col']]
    raw_dataset_num = raw_dataset_num.astype('int32')
    raw_dataset_cat = input_data[config_data['object_predictor']]
    raw_dataset_cat = raw_dataset_cat.astype(str)

    # 6 Concatenate data type into one dataset
    raw_dataset = pd.concat([raw_dataset_date, raw_dataset_num, raw_dataset_cat], axis = 1)

    return raw_dataset

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset Index
    raw_dataset.reset_index(inplace=True, drop=True)

    # 4. Save raw dataset
    util.pickle_dump(raw_dataset, config_data["raw_dataset_path"])

    # 5. Handling data type
    raw_dataset = type_data(raw_dataset)

    # 6. Change ? data into UNKNOWN
    raw_dataset.collision_type = raw_dataset.collision_type.replace("?","UNKNOWN")
    raw_dataset.property_damage = raw_dataset.property_damage.replace("?", "UNKNOWN")
    raw_dataset.police_report_available = raw_dataset.police_report_available.replace("?", "UNKNOWN")

    # 7. Check data definition
    check_data(raw_dataset, config_data)

    # 8. Splitting data
    X = raw_dataset[config_data["predictor"]].copy()
    y = raw_dataset[config_data["label"]].copy()

    # 9. splitting train and test set with ratio 0.7:0.3 and do stratify splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3, 
                                                        random_state= 42, 
                                                        stratify= y)
    
    # 10. Splitting test and valid set with ratio 0.5:0.5 and do stratify splitting
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, 
                                                        y_test, 
                                                        test_size = 0.5, 
                                                        random_state= 42, 
                                                        stratify= y_test)

    # 11. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])

    