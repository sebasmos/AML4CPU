import argparse
import datetime
import json
import numpy as np
import os
import joblib

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import csv
from datetime import datetime
#from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import time
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from scipy.signal import savgol_filter
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from river import forest,tree, neural_net, ensemble, evaluate, metrics, preprocessing, stream

from sklearn.preprocessing import StandardScaler
import copy
from sklearn.model_selection import train_test_split
from pandas import DataFrame, concat
from xgboost import XGBRegressor
from src.utils import save_timestamps, store_pickle_model, save_predictions
import src.misc as misc
from src.misc import NativeScalerWithGradNormCount as NativeScaler
from src.datasets import load_orangepi_data, normalize_data, ts_supervised_structure, structure, denormalize_predictions,load_data
from src.metrics import symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, calculate_metrics
from src.train import train_xgb_model, test_xgb_model, train_river_model, test_river_model, train_sklearn_model, test_sklearn_model, train_pytorch_model, test_pytorch_model, train_pytorch_model_LR, test_pytorch_model_LR
from src.memory import bytes_to_mb, model_memory_usage_alternative
from src.model import LSTMModel, GRUModel, BiLSTMModel, LSTMModelWithAttention, LinearRegression
import sys
from pympler import asizeof
sys.setrecursionlimit(100000) #setting recursion to avoid RecursionError: maximum recursion depth exceeded in comparison


def get_args_parser():
    
    parser = argparse.ArgumentParser('CPU regression', add_help=False)
    # General params
    parser.add_argument('--small_sample', action='store_true', help='if true then use only 800 samples for testing purposes')
    
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--window_sizes', default=[6,9,12,20,32,64], type=int)

    parser.add_argument('--epochs', default=50, type=int)
    
    parser.add_argument('--output_folder', default='exp1', type=str, help='Name of experiment folder')
    
    parser.add_argument('--output_file', default='exp1', type=str, help='Name of plot')
    
    parser.add_argument('--samples_to_visualize', default=50, type=int, help='Number of samples to visualize in plots')
    
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
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_seeds', default=20, type=int,
                        help='number of seeds to test experiment')
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


def plot_predictions(window_size = None,
                     output_file  = None,
                     NORMALIZATION   = 1,
                     AUTOREGRESSIVE  = True,
                     eval = False, 
                     seed = 1, 
                     model = None,
                     model_name = None,
                     timestamps_to_plot = 0,
                     model_path = ""):
    
    os.makedirs("./outputs", exist_ok=True)

    data_train, data_test = load_data(args.data_path, small_sample=args.small_sample)
    
    data_test_t = data_test.copy()
    
    data_test_t.index.name = 'Date'
    
    data_test_t['Date'] = data_test_t.index
    
    timestamps_to_plot = save_timestamps(data_test_t, args)
    
    results = []

    supervised_train_data = ts_supervised_structure(data_train[["target"]], n_in=window_size, n_out=1, autoregressive=AUTOREGRESSIVE)
    supervised_test_data = ts_supervised_structure(data_test[["target"]], n_in=window_size, n_out=1, autoregressive=AUTOREGRESSIVE )
        
    X_train = supervised_train_data.iloc[:,:-1] 
    y_train = supervised_train_data.iloc[:,-1] 
    X_test = supervised_test_data.iloc[:,:-1] 
    y_test = supervised_test_data.iloc[:,-1]
    x_min = X_train.min()
    x_max = X_train.max()
    y_min = y_train.min()
    y_max = y_train.max()

    # for j, (model_name_original, model) in enumerate(models.items()):
            
    model_cp = copy.deepcopy(model)
            
    model_name = model_name+"_"+"ws_"+str(window_size)

    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y = normalize_data(X_train, X_test, y_train, y_test)
            
    if isinstance(model, XGBRegressor): 
        # If XGBoost model is selected
        trained_model, training_time = train_xgb_model(model, X_train_normalized, y_train_normalized)
        y_pred_normalized, inference_time = test_xgb_model(trained_model, X_test_normalized)
        model_memory = bytes_to_mb(asizeof.asizeof(model))
                
    elif model_name in ('RandomForestRegressor',
                        'DecisionTreeRegressor', 
                        'AdaBoostRegressor', 
                        'SGDRegressor', 
                        'PassiveAggressiveRegressor'):
        # Assuming it's a scikit-learn model
        trained_model, training_time = train_sklearn_model(model, X_train_normalized, y_train_normalized)
        y_pred_normalized, inference_time = test_sklearn_model(trained_model, X_test_normalized)
                
        model_memory = bytes_to_mb(asizeof.asizeof(trained_model))
            
    else:
        print("Training Pytorch model")
        #define pytorch model
        torch.backends.cudnn.benchmark =  True
        torch.backends.cudnn.enabled =  True
        device = "cpu"
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
        print(model_name)
        model_memory = 0
        print(f"X_train_normalized shape: {X_train_normalized.shape}")
        print(f"y_train_normalized shape: {y_train_normalized.shape}")
        print(f"X_test_normalized shape: {X_test_normalized.shape}")
        torch.manual_seed(seed)
        attention = False
        if model_name in ('LinearRegression', 'SVR'):
            print('Inside Linear Pytorch model')
            if 'SVR' in model_name:
                print("SVR model uses hingeloss")
                loss = True
            print(f"X_train_normalized shape: {X_train_normalized.shape}")
            model_cp = LinearRegression(input_dim=window_size)
            print(f"Model: {model_cp}")
            trained_model, training_time = train_pytorch_model_LR(model_cp, X_train_normalized, y_train_normalized, loss=False)
            y_pred_normalized, inference_time = test_pytorch_model_LR(trained_model, X_test_normalized, attention)

        else:
            if 'LSTM_ATTN' in model_name:
                print('Inside LSTM ATTN')
                attention = True
                model_cp = LSTMModelWithAttention(input_size=window_size, num_layers=2, hidden_size=64, output_size=1)
                    
            model_cp.to(device)
            print(f"Model: {model_cp}")
            trained_model, training_time = train_pytorch_model(model_cp, X_train_normalized, y_train_normalized, attention, device)
            y_pred_normalized, inference_time = test_pytorch_model(trained_model, X_test_normalized, attention, device)
                
        model_memory = model_memory_usage_alternative(trained_model)
        print(f"Model Memory: {model_memory}")
                
    # DENORMALIZE PREDICTIONS BEFORE CALCULATING METRICS
    y_pred = denormalize_predictions(y_pred_normalized, NORMALIZATION, scaler_y, y_min, y_max)

            
    # CALCULATE METRICS
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
    # SAVE PREDICTIONS   
    save_predictions(args, model_name, seed, y_test, y_pred, training_time,inference_time,model_memory,model, window_size, result)
          
    results.append(result)


    return results


    
def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    # Initialize the models

    models_cp = {
                # 'XGBRegressor': XGBRegressor(random_state=0),
                # 'LinearRegression': LinearRegression(input_dim=6),
                # 'RandomForestRegressor': RandomForestRegressor(random_state=0),
                # 'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
                # 'AdaBoostRegressor': AdaBoostRegressor(random_state=0),
                # 'SGDRegressor': SGDRegressor(random_state=0),
                # 'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=0),
                # 'SVR': LinearRegression(input_dim=6),
                # 'LSTM': LSTMModel(input_size=1, num_layers=2, hidden_size=64, output_size=1),
                # 'GRU': GRUModel(input_size=1, hidden_size=64),
                # 'BI-LSTM': BiLSTMModel(input_size=1, hidden_size=64),
                'LSTM_ATTN': LSTMModelWithAttention(input_size=6, num_layers=2, hidden_size=64, output_size=1),
            }
    # fix the seed for reproducibility: on this section we ensure that the seed will be variable per seed-cycle
    if args.eval:
        
        data_list = []
        
        for seed in range(args.num_seeds):
            
            for model_name in models_cp:# for each seed, run all defined windows
                for i, winsize in enumerate(args.window_sizes):
                    if model_name == 'XGBRegressor':
                        #subsample will define the % of training data to be used for each tree:hard to replicate same seeds each time
                        random = np.random.uniform(0.5, 1)
                        models_cp[model_name] = XGBRegressor(random_state=seed, subsample=random)
                    # Scikit-learn models
                    if isinstance(models_cp.get(model_name), BaseEstimator):
                        print(f"The value for key '{model_name}' is a scikit-learn model instance.")
                        np.random.seed(seed)
                        if model_name in ('RandomForestRegressor','DecisionTreeRegressor', 'AdaBoostRegressor', 'PassiveAggressiveRegressor', 'SGDRegressor'):
                            print(model_name)
                            models_cp[model_name].set_params(random_state=seed)
                    # Pytorch models
                    if isinstance(models_cp[model_name],nn.Module):
                        print(model_name)
                        torch.manual_seed(seed)
                        if model_name == 'LSTM':
                            models_cp[model_name] = LSTMModel(input_size=1, num_layers=2, hidden_size=64, output_size=1)
                        elif model_name == 'GRU':
                            models_cp[model_name] = GRUModel(input_size=1, hidden_size=64)
                        elif model_name == 'BI-LSTM':
                            models_cp[model_name] = BiLSTMModel(input_size=1, hidden_size=64)
                        elif model_name == 'LSTM_ATTN':
                            models_cp[model_name] = LSTMModelWithAttention(input_size=winsize, num_layers=2, hidden_size=64, output_size=1)
                        elif model_name == 'LinearRegression':
                            models_cp[model_name] = LinearRegression(input_dim=winsize)
                        elif model_name == ('SVR'):
                            models_cp[model_name] = LinearRegression(input_dim=winsize)
    
                    else:
                        np.random.seed(seed)
                        models_cp[model_name].seed = seed # river models

                    metrics = plot_predictions(
                                             window_size     = winsize, 
                                             output_file      = args.output_file, 
                                             AUTOREGRESSIVE   = args.autoregressive,
                                             eval             = args.eval,
                                             model           = models_cp[model_name],
                                             model_name      = model_name,
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
        
        summary = pd.DataFrame(train_and_evaluate(seed, args, models_cp))
        
        summary = summary.round(3)
    
    print(summary)
    
    summary.to_csv(f'./outputs/{args.output_folder}/{args.output_file}/model_metrics_{args.output_file}.csv', index=False)
    
        
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
