import argparse
import datetime
import json
import numpy as np
import os
import joblib
from pympler import asizeof
from pathlib import Path
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import csv
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.base import BaseEstimator
import time
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from scipy.signal import savgol_filter
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from river import forest,tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import copy
from river import forest,tree, neural_net, ensemble, evaluate, metrics, preprocessing, stream
from pandas import DataFrame, concat
from xgboost import XGBRegressor
from src.utils import save_timestamps, store_pickle_model, save_predictions
import src.misc as misc
from src.misc import NativeScalerWithGradNormCount as NativeScaler
from src.datasets import load_orangepi_data, normalize_data, series_to_supervised, structure, denormalize_predictions,load_data
from src.metrics import symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, calculate_metrics
from src.train import train_xgb_model, test_xgb_model_update, train_river_model, test_river_model, train_sklearn_model, test_sklearn_model, train_sklearn_partial_fit,test_sklearn_partial_fit
from src.memory import bytes_to_mb
import sys
sys.setrecursionlimit(100000) #setting recursion to avoid RecursionError: maximum recursion depth exceeded in comparison

def get_args_parser():
    
    parser = argparse.ArgumentParser('CPU regression', add_help=False)
    # General params
    parser.add_argument('--num_seeds', default=20, type=int,
                        help='number of seeds to test experiment')
    parser.add_argument('--small_sample', action='store_true', help='if true then use only 800 samples for testing purposes')
    
    
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--window_sizes', default=[6,8,12,20,32,64], type=int)

    parser.add_argument('--epochs', default=50, type=int)
    
    parser.add_argument('--output_folder', default='exp1', type=str, help='Name of experiment folder')
    
    parser.add_argument('--output_file', default='exp1', type=str, help='Name of plot')
    
    parser.add_argument('--samples_to_visualize', default=10, type=int, help='Number of samples to visualize in plots')
    
    parser.add_argument('--use_smoothed_res', default=False, type=bool, help='Set to smooth the graphics')
    
    parser.add_argument('--normalization', default=1, type=int, help='0) Without normalization, 1) use scaler, 3) use min-max normalization ')
    
    parser.add_argument('--autoregressive', default=True, type=bool, help=' set to True to set autoregressiveness')
    
    parser.add_argument('--eval', action='store_true', help='eval over 20 seeds')

    # Model parameters
    parser.add_argument('--model_path', default='./models', type=str, metavar='MODEL',
                        help='Name of model_path to save')
    
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    # Dataset parameters
    parser.add_argument('--num', default=4000, type=int,
                        help='define the total number of samples to use from the dataset - by default its 40k for the CPU dataset')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    
    parser.add_argument('--dist_on_itp', action='store_true')
    
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def plot_predictions(data_train = None, 
                     data_test  = None, 
                     window_sizes = None,
                     output_file  = None,
                     use_smoothed_res = False,
                     NORMALIZATION   = None,
                     AUTOREGRESSIVE  = True,
                     eval = False, 
                     seed = 1, 
                     models = None,
                     timestamps_to_plot = 0,
                     model_path = "",
                     ):
    os.makedirs("./outputs", exist_ok=True)
    pdf = PdfPages(os.path.join(f"./outputs/{args.output_folder}", args.output_file, output_file+".pdf"))
    
    fig, axes = plt.subplots(len(window_sizes), len(models), figsize=(20, 12))
    
    plt.xticks(rotation=45, fontsize=5) 
    
    results = []

    for i, window_size in enumerate(window_sizes):

        supervised_train_data = series_to_supervised(data_train[["target"]], n_in=window_size, n_out=1, autoregressive=AUTOREGRESSIVE )
        supervised_test_data = series_to_supervised(data_test[["target"]], n_in=window_size, n_out=1, autoregressive=AUTOREGRESSIVE )
        
        
        X_train = supervised_train_data.iloc[:,:-1] 
        y_train = supervised_train_data.iloc[:,-1] 
        X_test = supervised_test_data.iloc[:,:-1] 
        y_test = supervised_test_data.iloc[:,-1]
        x_min = X_train.min()
        x_max = X_train.max()
        y_min = y_train.min()
        y_max = y_train.max()
        
        for j, (model_name_original, model) in enumerate(models.items()):
            
            model_cp = copy.deepcopy(model)
            
            model_name = model_name_original+"_"+"ws_"+str(window_size)
            
            print(f"Training {model_name}".center(60, "-"))
            
            if len(models)>1:
            
                ax = axes[i,j]
                
            else:
                
                ax = axes[i]
            
            #  PRE-PROCESSING STAGE 
            
            if NORMALIZATION == 0:
                
                X_train_normalized = X_train
                y_train_normalized = y_train
                X_test_normalized = X_test
                y_test_normalized = y_test

            elif NORMALIZATION == 1:
                
                X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y = normalize_data(X_train, X_test, y_train, y_test)

            elif NORMALIZATION == 2:
                X_train_normalized = mimax_norm(X_train, x_min, x_max)
                y_train_normalized = mimax_norm(y_train, y_min, y_max)
                X_test_normalized = mimax_norm(X_test, x_min, x_max)
                y_test_normalized = mimax_norm(y_test, y_min, y_max)
            
            print(X_train_normalized.shape, y_train_normalized.shape)
            print(X_test_normalized.shape, y_test_normalized.shape)
            
            if isinstance(model, XGBRegressor): 
                # If XGBoost model is selected
                trained_model, training_time = train_xgb_model(model, X_train_normalized, y_train_normalized)
                # change to online approach
                y_pred_normalized, inference_time = test_xgb_model_update(trained_model, X_train_normalized,  X_test_normalized, y_train_normalized, y_test_normalized)
                # Model memory for Xgboost model
                model_memory = bytes_to_mb(asizeof.asizeof(model))
                
            elif model_name_original in (  'AdaptiveRandomForest', 
                                           'HoeffdingTreeRegressor', 
                                           'HoeffdingAdaptiveTreeRegressor',
                                           'SRPRegressor'):
                print(f"training {model_name_original} model")
                trained_model, training_time = train_river_model(model, X_train_normalized, y_train_normalized)
                y_pred_normalized, inference_time = test_river_model(trained_model, X_test_normalized, y_test_normalized)
                model_memory = bytes_to_mb(asizeof.asizeof(trained_model))
             
            elif model_name_original in ('SGDRegressor', 'PassiveAggressive','MLP_partialfit'):
         
                trained_model, training_time = train_sklearn_partial_fit(model_cp, X_train_normalized, y_train_normalized)
                y_pred_normalized, inference_time = test_sklearn_partial_fit(trained_model, X_test_normalized,y_test_normalized)
                model_memory = bytes_to_mb(asizeof.asizeof(trained_model))
                
            elif model_name_original in ('LinearRegression', 'KNeighborsRegressor', 'RandomForestRegressor', 'MLPRegressor', 'SVR', 'DecisionTreeRegressor', 'AdaBoostRegressor'):

                trained_model, training_time = train_sklearn_model(model, X_train_normalized, y_train_normalized)
                y_pred_normalized, inference_time = test_sklearn_model(trained_model, X_test_normalized)
                model_memory = bytes_to_mb(asizeof.asizeof(trained_model))
            
            # DENORMALIZE PREDICTIONS BEFORE CALCULATING METRICS
            y_pred = denormalize_predictions(y_pred_normalized, NORMALIZATION, scaler_y, y_min, y_max)
            
            # CALCULATE METRICS
            metrics = calculate_metrics(y_test, y_pred)
          
            # VISUALIZATION STAGE
            if use_smoothed_res:
                # Smooth the y_pred using the Savitzky-Golay filter
                window_length = 61  # Smaller window_length value
                polyorder = 3  # Order of the polynomial fitting
                smooth_y_pred = savgol_filter(y_pred, window_length, polyorder, mode='interp')
                # Plot the smooth version of y_pred along with the ground truth
                ax.plot(range(len(y_test)), y_test, label='Ground Truth')
                ax.plot(range(len(y_test)), y_pred, label='Predicted', alpha=0.7)  # Original y_pred with alpha for transparency
                ax.plot(range(len(y_test)), smooth_y_pred, label='Smoothed Predicted', alpha=0.7)  # Smoothed version with alpha    
                ax.set_title(f'{model_name}\n'
                            f'Window Size: {window_size}\n'
                            f'Proc time: {training_time+inference_time:.2f}\n'
                            f'MAE: {metrics["mae"]:.2f}  | SMAPE: {metrics["smape"]:.2f} \n | R2: {metrics["r2"]:.2f} \n | RMSE: {metrics["rmse"]:.7f} | \n MASE: {metrics["mase"]:.2f}',
                            fontsize=10)
                ax.legend()
            else: 
                ax.set_ylabel('CPU')
                ax.plot(timestamps_to_plot, y_test[-len(timestamps_to_plot):], label='Ground Truth')
                ax.plot(timestamps_to_plot, y_pred[-len(timestamps_to_plot):], label='Predicted', alpha=0.7)  # Original y_pred with alpha for transparency

                ax.legend()
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.set_xlabel('Timestamp')
                ax.set_title(f'{model_name}\n'
                            f'Window Size: {window_size}\n'
                            f'Proc time: {training_time+inference_time:.2f}\n'
                            f'MAE: {metrics["mae"]:.2f}  | SMAPE: {metrics["smape"]:.2f} \n | R2: {metrics["r2"]:.2f} \n | RMSE: {metrics["rmse"]:.7f} | \n MASE: {metrics["mase"]:.2f}',
                            fontsize=10)

            result = {
                'Model': model_name,
                'Window Size': window_size,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'SMAPE': metrics['smape'],
                'r2': metrics['r2'],
                'MASE': metrics['mase'],
                'Training_time': training_time,
                'Inference_time': inference_time,
                'Model memory (MB)': model_memory
            }
            # SAVE PREDICTIONS   
            save_predictions(args, model_name, seed, y_test, y_pred, training_time,inference_time,model_memory, trained_model, window_size, result)
          
            results.append(result)

    plt.xlabel('Timestamp')
    plt.tight_layout()
    pdf.savefig()
    pdf.close()

    return results


def train_and_evaluate(seed, args, models):
    
    data_train, data_test = load_data(args.data_path, small_sample=args.small_sample)
    
    data_test_t = data_test.copy()
    
    data_test_t.index.name = 'Date'
    
    data_test_t['Date'] = data_test_t.index
    
    timestamps_to_plot = save_timestamps(data_test_t, args)
    
    print("=" * 30)
    print(f"|     NEW DATA Data Information for seed {seed}      |")
    print("=" * 30)
    print("|      Training Data:        |")
    print("=" * 30)
    print(f"Shape {data_train.shape} - value range: ({data_train.target.min()} - {data_train.target.max()})")
    print("-" * 30)
    print("|      Testing Data:         |")
    print("=" * 30)
    print(f"Shape {data_test.shape} - value range: ({data_train.target.min()} - {data_train.target.max()})")
    print("-" * 30)


    return plot_predictions(data_train            = data_train, 
                                 data_test        = data_test, 
                                 window_sizes     = args.window_sizes, 
                                 output_file      = args.output_file, 
                                 use_smoothed_res = args.use_smoothed_res,
                                 NORMALIZATION    = args.normalization,
                                 AUTOREGRESSIVE   = args.autoregressive,
                                 eval             = args.eval,
                                 models           = models,
                                 timestamps_to_plot = timestamps_to_plot,
                                 model_path       = args.model_path,
                                 seed             = seed)

    
def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    device = torch.device(args.device)

    # Initialize the models
    base_model = tree.HoeffdingTreeRegressor(grace_period=50)

    models_base = {
                  'SGDRegressor': SGDRegressor(max_iter=1000, tol=1e-3, loss="squared_error", random_state=0),
                  'PassiveAggressive':PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3),
                  'MLP_partialfit':MLPRegressor(random_state=0, max_iter=500),
                # 'XGBRegressor': XGBRegressor(random_state=0),
                'AdaptiveRandomForest': (forest.ARFRegressor(seed=0)),

                'HoeffdingTreeRegressor': (tree.HoeffdingTreeRegressor(grace_period=100,
                                                                        model_selector_decay=0.9,)),

                'HoeffdingAdaptiveTreeRegressor': tree.HoeffdingAdaptiveTreeRegressor(
                                        grace_period=50,
                                        model_selector_decay=0.3,
                                        seed=0),
                    #https://riverml.xyz/0.21.0/api/ensemble/SRPRegressor/
                'SRPRegressor': ensemble.SRPRegressor(
                        model=base_model,
                        training_method="patches",
                        n_models=3,
                        seed=0
                    )
                 



            }
    
    models_cp = copy.deepcopy(models_base)
    # fix the seed for reproducibility: on this section we ensure that the seed will be variable per seed-cycle
    if args.eval:
        data_list = []
        
        for seed in range(args.num_seeds):
            
            for model_name in models_cp:
                if model_name == 'XGBRegressor':
                    #subsample will define the % of training data to be used for each tree:hard to replicate same seeds each time
                    random = np.random.uniform(0.5, 1)
                    models_cp[model_name]  = XGBRegressor(random_state=seed, subsample=random)
                
                if isinstance(models_cp.get(model_name), BaseEstimator):
                    print(f"The value for key '{model_name}' is a scikit-learn model instance.")
                    np.random.seed(seed)
                    models_cp[model_name].set_params(random_state=seed)

                else:
                    models_cp[model_name].seed = seed

            metrics = train_and_evaluate(seed, args, models_cp)
            
            print(f"Seed {seed+1} and metrics: {metrics}")
            
            data_list.append(pd.DataFrame(metrics))

        combined_df = pd.concat(data_list)
        
        summary = combined_df.groupby(['Model', 'Window Size']).agg({
                                                                     'MAE'  :['mean', 'std'],
                                                                     'RMSE' :['mean', 'std'],
                                                                     'SMAPE':['mean', 'std'],
                                                                     'r2'   :['mean', 'std'],
                                                                     'MASE' :['mean', 'std'],
                                                                     'Training_time' :['mean'],
                                                                     'Inference_time' :['mean'],
                                                                     'Model memory (MB)' :['mean'],
                                                                      }).reset_index()
        
        summary = summary.round(3)

    else:# Quick evaluation of model for generating plottings
        
        seed = args.seed + misc.get_rank()
        
        summary = pd.DataFrame(train_and_evaluate(seed, args, models_cp))
        
        summary = summary.round(3)
    
    print(summary)
    
    summary.to_csv(f'./outputs/{args.output_folder}/{args.output_file}/model_metrics_{args.output_file}.csv', index=False)
    
        
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)