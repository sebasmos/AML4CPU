import pandas as pd
import os
import joblib
import numpy as np
import pickle

def save_predictions(args, model_name, seed, y_test, y_pred, training_time, inference_time, model_memory, model, window_size, metrics):
    """
    Save predicted and true values as CSV files in a specified folder.

    Args:
    - pred_folder (str): The folder path where the files will be saved.
    - model_name (str): The name of the model.
    - seed (int): The seed used for randomization.
    - y_test (numpy array): True values.
    - y_pred (numpy array): Predicted values.
    """
    pred_folder = os.path.join(f"./outputs/{args.output_folder}", args.output_file, f"testbed_{seed}", model_name)

    os.makedirs(pred_folder, exist_ok=True)

    np.savetxt(os.path.join(pred_folder, 'y_test.csv'), y_test, delimiter=',', header="y_test", comments='')
    np.savetxt(os.path.join(pred_folder, 'y_pred.csv'), y_pred, delimiter=',', header="y_pred", comments='')
    # t*

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).T

    metrics_df.index = [0]


    metrics_df.to_csv(os.path.join(pred_folder, 'model_data.csv'), index=False)
    
    store_pickle_model(model, model_name, window_size, pred_folder)
    
def store_pickle_model(model, model_name, window_size, model_path):
    """
    Store a trained model as a pickle file.

    Args:
    - model: The trained model object to be stored.
    - model_name (str): The name of the model.
    - window_size (int): The window size used in the model.
    - model_path (str): The directory path where the model will be stored.
    """
    os.makedirs(model_path, exist_ok=True)

    model_name = f"{model_name}_{window_size}"

    model_filename = os.path.join(model_path, model_name + '.pkl')

    print("Model saved in:", model_filename)
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Failed to pickle the model due to: {e}")
        
def save_timestamps(x, args):
    data_folder = f"./outputs/{args.output_folder}"
    os.makedirs(data_folder, exist_ok = True)
    samples_to_visualize  = args.samples_to_visualize
    timestamps_plotting = pd.to_datetime(x.Date)
    timestamps = [timestamp.strftime('%Y-%m-%d %H:%M') for timestamp in timestamps_plotting]
    timestamps_to_plot = timestamps[-samples_to_visualize:]
    joblib.dump(timestamps_to_plot, os.path.join(data_folder, f'{args.output_folder}_timestamp.joblib'))
    return timestamps_to_plot