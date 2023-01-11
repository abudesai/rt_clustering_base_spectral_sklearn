import os, shutil
import sys
import time
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import adjusted_mutual_info_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


sys.path.insert(0, './../app')
import algorithm.utils as utils
import algorithm.model_trainer as model_trainer
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.clustering as clustering


inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data", "clusteringBaseMainInput")

model_path = "./ml_vol/model/"
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_outputs")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

test_results_path = "test_results"
if not os.path.exists(test_results_path): os.mkdir(test_results_path)


# change this to whereever you placed your local testing datasets
local_datapath = "./../../datasets" 


'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os or python version-related issues, so beware. 
'''

model_name = clustering.MODEL_NAME

prediction_col = 'prediction'   # this is a hard_coded target field mandated by our specs
target_field = "__target__"  # this is a hard_coded internal field 

def create_ml_vol():
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "clusteringBaseMainInput": None
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            },

            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,
            }
        }
    }

    def create_dir(curr_path, dir_dict):
        for k in dir_dict:
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None:
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)


def copy_example_files(dataset_name):
    # data schema
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # data
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}.csv", os.path.join(data_path, f"{dataset_name}_test.csv"))
    

def train_and_predict():
    # Read hyperparameters
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)
    # Read data
    data = utils.get_data(data_path)
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)
    # set global variables specific to the dataset
    set_dataset_variables(data_schema=data_schema)
    # get trained preprocessor, model, training history
    preprocessor, model, predictions = model_trainer.get_trained_model(data, data_schema, hyper_parameters)
    # Save the processing pipeline
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model
    clustering.save_model(model, model_artifacts_path)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    print("done with training and predictions")
    # score the results
    results, chart_data = score(data, predictions, data_schema)
    return results, chart_data


def set_dataset_variables(data_schema):
    global id_field, target_field, test_answers
    id_field = data_schema["inputDatasets"]["clusteringBaseMainInput"]["idField"]   
    test_answers = pd.read_csv(f"{local_datapath}/{dataset_name}/{dataset_name}_test_key.csv")
    

def score(test_data, predictions, data_schema):
    X_cols = [c for c in test_data.columns if c not in [id_field, target_field]]    
    predictions = predictions.merge(test_answers[[id_field, target_field]], on=id_field)

    # external validation metrics
    purity = purity_score(predictions[target_field], predictions[prediction_col])
    ami = adjusted_mutual_info_score(predictions[target_field], predictions[prediction_col])
    
    # for internal validation
    test_data_with_pred_clusters = test_data.merge(predictions, on=id_field)    
    # standardize the data before doing internal validation    
    scaled_test_data = standardize_data(test_data_with_pred_clusters[X_cols], data_schema)
    # internal validation metrics
    n_clusters = len(set(test_data_with_pred_clusters[prediction_col]))
    if n_clusters == 1: 
        dbi = chi = silhouette = np.float("inf")
    else: 
        dbi = davies_bouldin_score(scaled_test_data, test_data_with_pred_clusters[prediction_col])
        chi = calinski_harabasz_score(scaled_test_data, test_data_with_pred_clusters[prediction_col])
        silhouette = silhouette_score(scaled_test_data, test_data_with_pred_clusters[prediction_col])    
    
    
    dbi_best = davies_bouldin_score(scaled_test_data, test_data_with_pred_clusters[target_field])
    chi_best = calinski_harabasz_score(scaled_test_data, test_data_with_pred_clusters[target_field])
    silhouette_best = silhouette_score(scaled_test_data, test_data_with_pred_clusters[target_field])

    # print(test_data_with_pred_clusters.head())
    y = test_data_with_pred_clusters[target_field].factorize()[0]
    y_hat = test_data_with_pred_clusters[prediction_col]

    results = {
        "purity": np.round(purity, 4),
        "ami": np.round(ami, 4),
        
        "dbi": np.round(dbi, 4),
        "chi": np.round(chi, 4),
        "silhouette": np.round(silhouette, 4),
        
        "dbi_best": np.round(dbi_best, 4),
        "chi_best": np.round(chi_best, 4),
        "silhouette_best": np.round(silhouette_best, 4),
    }
    chart_data = {
        "X": scaled_test_data,
        "y": y,
        "y_hat": y_hat,
    }
    return results, chart_data


def save_scoring_outputs(results, dataset_name, run_hpt = False, chart_data=None):
    df = pd.DataFrame(results) if dataset_name is None else pd.DataFrame([results])
    df = df[["model", "dataset_name", "run_hpt", "num_hpt_trials",
             "purity", "ami", 
             "dbi", "chi", "silhouette", 
             "dbi_best", "chi_best", "silhouette_best", 
             "elapsed_time_in_minutes"]]
    print(df)
    file_path_and_name = get_file_path_and_name(dataset_name, run_hpt)
    df.to_csv(file_path_and_name, index=False)

    if chart_data is not None:
        fig, axs = plt.subplots(1, 2)
        fig.set_figwidth(10)
        X, y, y_hat = chart_data["X"], chart_data["y"], chart_data["y_hat"]
        if X.shape[1] > 2:  X = reduce_dims(X)
        axs[0].scatter(X[:, 0], X[:, 1], alpha=0.3, c=y)
        axs[0].set_title('actual clusters')
        axs[1].scatter(X[:, 0], X[:, 1], alpha=0.3, c=y_hat)
        axs[1].set_title('predicted clusters')
        file_path_and_name = get_file_path_and_name(dataset_name, run_hpt, file_type="scatter")
        plt.suptitle(f"clusters on tsne-reduced data for {dataset_name} dataset with {run_hpt=}")
        plt.savefig(file_path_and_name)
        plt.clf()


def get_file_path_and_name(dataset_name, run_hpt, file_type="scores" ):
    if file_type == 'scores':
        if dataset_name is None:
            fname = f"_{model_name}_results_with_hpt.csv" if run_hpt else f"_{model_name}_results_no_hpt.csv"
        else:
            fname = f"{model_name}_{dataset_name}_results_with_hpt.csv" if run_hpt else f"{model_name}_{dataset_name}_results_no_hpt.csv"
    elif file_type == 'scatter':
        if dataset_name is None:
            raise Exception("Cant do this.")
        else:
            fname = f"{model_name}_{dataset_name}_scatter_hpt_{run_hpt}.png"
    else:
        raise Exception(f"Invalid file_type for scatter plot: {file_type}")
    full_path = os.path.join(test_results_path, fname)
    return full_path



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def get_input_vars_lists(data_schema):      
    input_vars = []
    attributes = data_schema["inputDatasets"]["clusteringBaseMainInput"]["inputFields"]   
    for attribute in attributes: 
        input_vars.append(attribute["fieldName"])
    return input_vars


def standardize_data(data, data_schema):
    input_vars = get_input_vars_lists(data_schema)    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[input_vars])
    return scaled_data


def reduce_dims(X): 
    reducer = TSNE(n_components=2)
    # reducer = PCA(n_components=2)       
    reduced_dim_data = reducer.fit_transform(X)
    return reduced_dim_data
    

def run_train_and_test(dataset_name, run_hpt=False, num_hpt_trials=None):
    start = time.time()

    create_ml_vol()  # create the directory which imitates the bind mount on container
    copy_example_files(dataset_name)  # copy the required files for model training
    results, chart_data = train_and_predict()  # train and predict

    end = time.time()
    elapsed_time_in_minutes = np.round((end - start) / 60.0, 2)

    results = {**results,
               "model": model_name,
               "dataset_name": dataset_name,
               "run_hpt": False, 
               "num_hpt_trials": None, 
               "elapsed_time_in_minutes": elapsed_time_in_minutes
               }

    return results, chart_data


if __name__ == "__main__":
    
    num_hpt_trials = None
    run_hpt_list = [False]

    datasets = [
        "concentric_circles",
        "gesture_phase2",
        "iris",
        "landsat_satellite2",
        "page_blocks2",
        "penguins",
        "spam2",
        "steel_plate_fault2",
        "unequal_variance_blobs",
        "vehicle_silhouettes2",
    ]
    datasets = ["iris"]


    for run_hpt in run_hpt_list:    
        all_results = []    
        for dataset_name in datasets:
            print("-" * 60)
            print(f"Running dataset {dataset_name}")
            results, chart_data = run_train_and_test(dataset_name, run_hpt, num_hpt_trials)
            save_scoring_outputs(results, dataset_name, run_hpt, chart_data)
            all_results.append(results)
            print("-" * 60)

        save_scoring_outputs(all_results, dataset_name=None, run_hpt=run_hpt)