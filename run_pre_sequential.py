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

    parser.add_argument('--window_sizes', default=[6,32,64], type=int)

    parser.add_argument('--epochs', default=50, type=int)
    
    parser.add_argument('--output_folder', default='exp1', type=str, help='Name of experiment folder')
    
    parser.add_argument('--output_file', default='exp1', type=str, help='Name of plot')
    
    parser.add_argument('--autoregressive', default=True, type=bool, help=' set to True to set autoregressiveness')
    
    parser.add_argument('--eval', action='store_true', help='eval over 20 seeds')
    
    parser.add_argument('--samples_to_visualize', default=10, type=int, help='Number of samples to visualize in plots')
    
    # Model parameters
    parser.add_argument('--model_path', default='./models', type=str, metavar='MODEL',
                        help='Name of model_path to save')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--normalization', default=1, type=int, help='0) Without normalization, 1) use scaler, 3) use min-max normalization ')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    

    return parser


def plot_predictions(window_size = None,
                     output_file  = None,
                     NORMALIZATION   = 1,
                     AUTOREGRESSIVE  = True,
                     eval = False, 
                     seed = 1, 
                     models = None,
                     timestamps_to_plot = 0,
                     model_path = ""):
    os.makedirs("./outputs", exist_ok=True)
    
    data_train, data_test = load_data(args.data_path, small_sample=args.small_sample)
    
    data_test_t = data_test.copy()
    
    data_test_t.index.name = 'Date'
    
    data_test_t['Date'] = data_test_t.index
    
    timestamps_to_plot = save_timestamps(data_test_t, args)

    results = []

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
            
            X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y = normalize_data(X_train, X_test, y_train, y_test)
            
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
                # print(f"training {model_name_original} model")
                trained_model, training_time = train_river_model(model_cp, X_train_normalized, y_train_normalized)
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
            
            y_pred = denormalize_predictions(y_pred_normalized, NORMALIZATION, scaler_y, y_min, y_max)
            

            metrics = calculate_metrics(y_test, y_pred)

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

            save_predictions(args, model_name, seed, y_test, y_pred, training_time,inference_time,model_memory, trained_model, window_size, result)
          
            results.append(result)

    return results



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_models(seed):
    base_model = tree.HoeffdingTreeRegressor(grace_period=200)
    return {
        #SGDRegressor, ARF, HoeffdingTreeRegressor, MLP_partialfit,MLP_partialfit
                  # 'SGDRegressor': SGDRegressor(random_state=seed),
                  # 'PassiveAggressive':PassiveAggressiveRegressor(random_state=seed),
                  # 'MLP_partialfit':MLPRegressor(random_state=seed),
                  # 'XGBRegressor': XGBRegressor(random_state=seed),
                  'AdaptiveRandomForest': (forest.ARFRegressor(seed=seed)),

                  'HoeffdingTreeRegressor': (tree.HoeffdingTreeRegressor(grace_period=200)),

                  'HoeffdingAdaptiveTreeRegressor': tree.HoeffdingAdaptiveTreeRegressor(
                                        seed=seed),
                    #https://riverml.xyz/0.21.0/api/ensemble/SRPRegressor/
                  'SRPRegressor': ensemble.SRPRegressor(
                        model=base_model,
                        training_method="patches",
                        n_models=10,
                        seed=seed
                    )
            }

def main(args):
    set_random_seed(args.seed) 
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    device = torch.device(args.device)
    if args.eval:
            data_list = []
            for seed in range(args.num_seeds):# run each seed 20 times
                for i, winsize in enumerate(args.window_sizes):# for each seed, run all defined windows
                    models_init = get_models(seed)  # for each window, new model
                    metrics = plot_predictions(
                                 window_size     = winsize, 
                                 output_file      = args.output_file, 
                                 AUTOREGRESSIVE   = args.autoregressive,
                                 eval             = args.eval,
                                 models           = models_init,
                                 model_path       = args.model_path,
                                 seed             = seed)
                    print(f"Seed {seed+1} and metrics: {metrics} and window: {winsize}")
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
        
        models_init = get_models()
        
        summary = pd.DataFrame(train_and_evaluate(seed, args, models_init))
        
        summary = summary.round(3)
    
    print(summary)
    
    summary.to_csv(f'./outputs/{args.output_folder}/{args.output_file}/model_metrics_{args.output_file}.csv', index=False)
    
        
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)