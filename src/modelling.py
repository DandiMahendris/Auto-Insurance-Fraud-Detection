## Load Required Library 
import util as util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import hashlib
import json

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

## Load Train data set
def load_train_clean(params: dict) -> pd.DataFrame:
    x_train = util.pickle_load(params["train_set_clean"][0])
    y_train = util.pickle_load(params["train_set_clean"][1])

    return x_train, y_train

## Load Valid data set
def load_valid_clean(params:dict) -> pd.DataFrame:
    x_valid = util.pickle_load(params["valid_set_clean"][0])
    y_valid = util.pickle_load(params["valid_set_clean"][1])

    return x_valid, y_valid

def load_test_clean(params: dict) -> pd.DataFrame:
    x_test = util.pickle_load(params["test_set_clean"][0])
    y_test = util.pickle_load(params["test_set_clean"][1])

    return x_test, y_test

## Load Test data set
def Baseline_model():
    dummy_clf = DummyClassifier(strategy='stratified')

    dummy_clf.fit(x_train["nonbalance"], y_train["nonbalance"])

    y_pred = dummy_clf.predict(x_train["nonbalance"])

    report = classification_report(y_true = y_train["nonbalance"],
                                    y_pred = y_pred,
                                    output_dict = True)
    report = pd.DataFrame(report)
    return report

## Create Weight for unbalanced dataset
def add_weight():
    sklearn_weight = compute_class_weight(class_weight = 'balanced', 
                                            classes = np.unique(y_train["nonbalance"]), 
                                            y = y_train["nonbalance"])
    sklearn_weight = dict(zip(np.unique(y_train["nonbalance"]), sklearn_weight))

    util.pickle_dump(sklearn_weight, config_data['model_params_path'][0])

    return sklearn_weight

## Crete training log template
def training_log_template() -> dict:
    # Debug message
    util.print_debug("creating training log template.")

    # Template of training log
    logger = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "performance" : [],
        "f1_score_avg" : [],
         "data_configurations" : [],
    }

    # Debug message
    util.print_debug("Training log template created.")

    # Return training log template
    return logger

# Create training log updater
def training_log_updater(current_log: dict, params: dict) -> list:
    # Create copy of current log
    current_log = copy.deepcopy(current_log)

    # Path for training log file
    log_path = params["training_log_path"]

    # Try to load training log file
    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()

    # If file not found, create a new one
    except FileNotFoundError as fe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()

        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    # Add current log to previous log
    last_log.append(current_log)

    # Save updated log
    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    # Return log
    return last_log

def create_model_object(params: dict) -> list:
    # Debug message
    util.print_debug("Creating model objects.")

    # Create model objects
    baseline_knn = KNeighborsClassifier()
    baseline_dt = DecisionTreeClassifier()
    baseline_logreg = LogisticRegression()
    baseline_svm = SVC()
    baseline_rforest = RandomForestClassifier()
    baseline_ada = AdaBoostClassifier()
    baseline_grad = GradientBoostingClassifier()
    baseline_xgb = XGBClassifier()
    baseline_qda = QuadraticDiscriminantAnalysis()
    baseline_gnb = GaussianNB()

    # Create list of model
    list_of_model = [
        { "model_name": baseline_knn.__class__.__name__, "model_object": baseline_knn, "model_uid": ""},
        { "model_name": baseline_dt.__class__.__name__, "model_object": baseline_dt, "model_uid": ""},
        { "model_name": baseline_logreg.__class__.__name__, "model_object": baseline_logreg, "model_uid": ""},
        { "model_name": baseline_svm.__class__.__name__, "model_object": baseline_svm, "model_uid": ""},
        { "model_name": baseline_rforest.__class__.__name__, "model_object": baseline_rforest, "model_uid": ""},
        { "model_name": baseline_ada.__class__.__name__, "model_object": baseline_ada, "model_uid": ""},
        { "model_name": baseline_grad.__class__.__name__, "model_object": baseline_grad, "model_uid": ""},
        { "model_name": baseline_xgb.__class__.__name__, "model_object": baseline_xgb, "model_uid": ""},
        { "model_name": baseline_qda.__class__.__name__, "model_object": baseline_qda, "model_uid": ""},
        { "model_name": baseline_gnb.__class__.__name__, "model_object": baseline_gnb, "model_uid": ""},
    ]

    # Debug message
    util.print_debug("Model objects created.")

    # Return the list of model
    return list_of_model

def train_eval(configuration_model: str, params: dict, hyperparams_model: list = None):
    # Load dataset

    # Variabel to store trained models
    list_of_trained_model = dict()

    # Create log template
    training_log = training_log_template()

    # Training for every data configuration
    for config_data in x_train:
        # Debug message
        util.print_debug("Training model based on configuration data: {}".format(config_data))

        # Create model objects
        if hyperparams_model == None:
            list_of_model = create_model_object(params)
        else:
            list_of_model = copy.deepcopy(hyperparams_model)

        # Variabel to store tained model
        trained_model = list()

        # Load train data based on its configuration
        x_train_data = x_train[config_data]
        y_train_data = y_train[config_data]

        # Train each model by current dataset configuration
        for model in list_of_model:
            # Debug message
            util.print_debug("Training model: {}".format(model["model_name"]))

            # Training
            training_time = util.time_stamp()
            model["model_object"].fit(x_train_data, y_train_data)
            training_time = (util.time_stamp() - training_time).total_seconds()

            # Debug message
            util.print_debug("Evalutaing model: {}".format(model["model_name"]))

            # Evaluation
            y_predict = model["model_object"].predict(x_valid)
            performance = classification_report(y_valid, y_predict, output_dict = True)

            # Debug message
            util.print_debug("Logging: {}".format(model["model_name"]))

            # Create UID
            uid = hashlib.md5(str(training_time).encode()).hexdigest()

            # Assign model's UID
            model["model_uid"] = uid

            # Create training log data
            training_log["model_name"].append("{}-{}-{}".format(configuration_model, config_data, model["model_name"]))
            training_log["model_uid"].append(uid)
            training_log["training_time"].append(training_time)
            training_log["training_date"].append(util.time_stamp())
            training_log["performance"].append(performance)
            training_log["f1_score_avg"].append(performance["weighted avg"]["f1-score"])
            training_log["data_configurations"].append(config_data)

            # Collect current trained model
            trained_model.append(copy.deepcopy(model))

            # Debug message
            util.print_debug("Model {} has been trained for configuration data {}.".format(model["model_name"], config_data))
        
        # Collect current trained list of model
        list_of_trained_model[config_data] = copy.deepcopy(trained_model)
    
    # Debug message
    util.print_debug("All combination models and configuration data has been trained.")
    
    # Return list trained model
    return list_of_trained_model, training_log

def train_eval_test(configuration_model: str, params: dict, hyperparams_model: list = None):
    # Variabel to store trained models
    list_of_trained_model = dict()

    # Create log template
    training_log = training_log_template()

    # Training for every data configuration
    for config_data in x_train:
        # Debug message
        util.print_debug("Training model based on configuration data: {}".format(config_data))

        # Create model objects
        if hyperparams_model == None:
            list_of_model = create_model_object(params)
        else:
            list_of_model = copy.deepcopy(hyperparams_model)

        # Variabel to store tained model
        trained_model = list()

        # Load train data based on its configuration
        x_train_data = x_train[config_data]
        y_train_data = y_train[config_data]

        # Train each model by current dataset configuration
        for model in list_of_model:
            # Debug message
            util.print_debug("Training model: {}".format(model["model_name"]))

            # Training
            training_time = util.time_stamp()
            model["model_object"].fit(x_train_data, y_train_data)
            training_time = (util.time_stamp() - training_time).total_seconds()

            # Debug message
            util.print_debug("Evalutaing model: {}".format(model["model_name"]))

            # Evaluation
            y_predict = model["model_object"].predict(x_test)
            performance = classification_report(y_test, y_predict, output_dict = True)

            # Debug message
            util.print_debug("Logging: {}".format(model["model_name"]))

            # Create UID
            uid = hashlib.md5(str(training_time).encode()).hexdigest()

            # Assign model's UID
            model["model_uid"] = uid

            # Create training log data
            training_log["model_name"].append("{}-{}-{}".format(configuration_model, config_data, model["model_name"]))
            training_log["model_uid"].append(uid)
            training_log["training_time"].append(training_time)
            training_log["training_date"].append(util.time_stamp())
            training_log["performance"].append(performance)
            training_log["f1_score_avg"].append(performance["weighted avg"]["recall"])
            training_log["data_configurations"].append(config_data)

            # Collect current trained model
            trained_model.append(copy.deepcopy(model))

            # Debug message
            util.print_debug("Model {} has been trained for configuration data {}.".format(model["model_name"], config_data))
        
    # Collect current trained list of model
    list_of_trained_model[config_data] = copy.deepcopy(trained_model)
    
    
    # Debug message
    util.print_debug("All combination models and configuration data has been trained.")
    
    # Return list trained model
    return list_of_trained_model, training_log

def get_production_model(list_of_model, training_log, params):
    # Create copy list of model
    list_of_model = copy.deepcopy(list_of_model)
    
    # Debug message
    util.print_debug("Choosing model by metrics score.")

    # Create required predefined variabel
    curr_production_model = None
    prev_production_model = None
    production_model_log = None

    # Debug message
    util.print_debug("Converting training log type of data from dict to dataframe.")

    # Convert dictionary to pandas for easy operation
    training_log = pd.DataFrame(copy.deepcopy(training_log))

    # Debug message
    util.print_debug("Trying to load previous production model.")

    # Check if there is a previous production model
    try:
        prev_production_model = util.pickle_load(params["production_model_path"])
        util.print_debug("Previous production model loaded.")

    except FileNotFoundError as fe:
        util.print_debug("No previous production model detected, choosing best model only from current trained model.")

    # If previous production model detected:
    if prev_production_model != None:
        # Debug message
        util.print_debug("Loading validation data.")
        x_valid, y_valid
        
        # Debug message
        util.print_debug("Checking compatibilty previous production model's input with current train data's features.")

        # Check list features of previous production model and current dataset
        production_model_features = set(prev_production_model["model_data"]["model_object"].feature_names_in_)
        current_dataset_features = set(x_valid.columns)
        number_of_different_features = len((production_model_features - current_dataset_features) | (current_dataset_features - production_model_features))

        # If feature matched:
        if number_of_different_features == 0:
            # Debug message
            util.print_debug("Features compatible.")

            # Debug message
            util.print_debug("Reassesing previous model performance using current validation data.")

            # Re-predict previous production model to provide valid metrics compared to other current models
            y_pred = prev_production_model["model_data"]["model_object"].predict(x_valid)

            # Re-asses prediction result
            eval_res = classification_report(y_valid, y_pred, output_dict = True)

            # Debug message
            util.print_debug("Assessing complete.")

            # Debug message
            util.print_debug("Storing new metrics data to previous model structure.")

            # Update their performance log
            prev_production_model["model_log"]["performance"] = eval_res
            prev_production_model["model_log"]["f1_score_avg"] = eval_res["macro avg"]["f1-score"]

            # Debug message
            util.print_debug("Adding previous model data to current training log and list of model")

            # Added previous production model log to current logs to compere who has the greatest f1 score
            training_log = pd.concat([training_log, pd.DataFrame([prev_production_model["model_log"]])])

            # Added previous production model to current list of models to choose from if it has the greatest f1 score
            list_of_model["prev_production_model"] = [copy.deepcopy(prev_production_model["model_data"])]
        else:
            # To indicate that we are not using previous production model
            prev_production_model = None

            # Debug message
            util.print_debug("Different features between production model with current dataset is detected, ignoring production dataset.")

    # Debug message
    util.print_debug("Sorting training log by f1 macro avg and training time.")

    # Sort training log by f1 score macro avg and trining time
    best_model_log = training_log.sort_values(["f1_score_avg", "training_time"], ascending = [False, True]).iloc[0]
    
    # Debug message
    util.print_debug("Searching model data based on sorted training log.")

    # Get model object with greatest f1 score macro avg by using UID
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_log["model_uid"]:
                curr_production_model = dict()
                curr_production_model["model_data"] = copy.deepcopy(model_data)
                curr_production_model["model_log"] = copy.deepcopy(best_model_log.to_dict())
                curr_production_model["model_log"]["model_name"] = "Production-{}".format(curr_production_model["model_data"]["model_name"])
                curr_production_model["model_log"]["training_date"] = str(curr_production_model["model_log"]["training_date"])
                production_model_log = training_log_updater(curr_production_model["model_log"], params)
                break
    
    # In case UID not found
    if curr_production_model == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    # Debug message
    util.print_debug("Model chosen.")

    # Dump chosen production model
    util.pickle_dump(curr_production_model, params["production_model_path"])
    
    # Return current chosen production model, log of production models and current training log
    return curr_production_model, production_model_log, training_log

def create_dist_params(model_name: str) -> dict:
    # Define models paramteres   
    dist_params_xgb = {
        "n_estimators" : [7],
        "learning_rate" : [0.7],
        "max_depth" : [4],
        "alpha" : [5],
        "scale_pos_weight" : [0.4],
    }
    dist_params_gb = {
        "max_depth" : [2],
        "n_estimators" : [5],
        "learning_rate" : [0.5],
    }
    dist_params_ada = {
        "base_estimator" : [DecisionTreeClassifier(max_depth = 2)],
        "n_estimators" : [3],
        "learning_rate" : [0.7],
    }
    dist_params_dt = {
        "max_depth" : [5],
    }

    # Make all models parameters in to one
    dist_params = {
        "XGBClassifier": dist_params_xgb,
        "GradientBoostingClassifier": dist_params_gb,
        "AdaBoostClassifier": dist_params_ada,
        "DecisionTreeClassifier": dist_params_dt
    }

    # Return distribution of model parameters
    return dist_params[model_name]

def hyper_params_tuning(model: list) -> list:
    # Create copy of current best baseline model
    model_list = []
    trained_model = [XGBClassifier(),GradientBoostingClassifier(),AdaBoostClassifier(),DecisionTreeClassifier()]
    
    # Create model object
    for col, mod in list(zip(model, trained_model)):
        dist_params = create_dist_params(col)
        model_rsc = GridSearchCV(mod, dist_params, n_jobs = -1)
        model_ = {
                        "model_name": col,
                        "model_object": model_rsc,
                        "model_uid": ""
                    }
            
        model_list.append(model_.copy())
    
    # Return model object
    return model_list

if __name__ == "__main__":
    # 1. Load Configuration File
    config_data = util.load_config()
    
    # 2. Load dataset
    x_train, y_train = load_train_clean(config_data)
    x_valid, y_valid = load_valid_clean(config_data)
    x_test, y_test = load_test_clean(config_data)
    
    # 3. Create class weight for training non balancing data
    sklearn_weight = add_weight()
    
    # 4. Training and evaluate model on validation data
    list_of_trained_model, training_log = train_eval("", config_data)
    
    # 5. Choose the best model on validation data
    model, production_model_log, training_logs = get_production_model(list_of_trained_model, training_log, config_data)
    
    # 6. Try to Optimize data with hyperparamter tuning
    list_of_trained_model_test, training_log_test = train_eval_test("Hyperparams", 
                                                                   config_data, 
                                                                    hyper_params_tuning(['XGBClassifier',
                                                                                         'GradientBoostingClassifier', 
                                                                                         'AdaBoostClassifier',
                                                                                         'DecisionTreeClassifier']))
    
    # 7. Choose the best model on testing data
    model_test, production_model_log_test, training_logs_test = get_production_model(list_of_trained_model_test, 
                                                                                      training_log_test, 
                                                                                      config_data)
    
    
    
    

