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
from tqdm import tqdm
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

from pandas import DataFrame, concat
from xgboost import XGBRegressor
from src.utils import save_timestamps, store_pickle_model, save_predictions
import src.misc as misc
from src.misc import NativeScalerWithGradNormCount as NativeScaler
from src.datasets import load_orangepi_data, normalize_data, ts_supervised_structure, structure, denormalize_predictions,load_data
from src.metrics import symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, calculate_metrics
from src.train import train_xgb_model, test_xgb_model_update, train_river_model, test_river_model, train_sklearn_model, test_sklearn_model, train_sklearn_partial_fit,test_sklearn_partial_fit
from src.memory import bytes_to_mb,model_memory_usage_alternative
import sys
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from src.metrics import calculate_metrics
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=Warning, module="gluonts")

sys.setrecursionlimit(100000) #setting recursion to avoid RecursionError: maximum recursion depth exceeded in comparison

def get_args_parser():
    
    parser = argparse.ArgumentParser('CPU regression', add_help=False)

    # General params

    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')

    parser.add_argument('--device', default='cuda:0',
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
    
    parser.add_argument('--small_sample', action='store_true', help='if true then use only 800 samples for testing purposes')
    
    parser.add_argument('--finetune', action='store_true', help='finetune')
    
    parser.add_argument('--zero_shot', action='store_true', help='zero_shot')
    
    parser.add_argument('--context_length', default=32, type=int,
                        help='context_length')
    parser.add_argument('--context_length_test', default=32, type=int,
                        help='context_length')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_seeds', default=20, type=int,
                        help='number of seeds to test experiment')
    parser.add_argument('--window_sizes', default=[6, 9, 12, 20, 32], type=int)

    parser.add_argument('--epochs', default=50, type=int)
    
    parser.add_argument('--output_folder', default='exp1', type=str, help='Name of experiment folder')
    
    parser.add_argument('--output_file', default='exp1', type=str, help='Name of plot')
    
    parser.add_argument('--samples_to_visualize', default=10, type=int, help='Number of samples to visualize in plots')
    
    parser.add_argument('--use_smoothed_res', default=False, type=bool, help='Set to smooth the graphics')
    
    parser.add_argument('--normalization', default=1, type=int, help='0) Without normalization, 1) use scaler, 3) use min-max normalization ')
    
    parser.add_argument('--autoregressive', default=True, type=bool, help=' set to True to set autoregressiveness')
    
    parser.add_argument('--eval_single', action='store_true', help='eval over 20 seeds single model')

    # Model parameters
  
    parser.add_argument('--max_epochs', default=1, type=int, help='Number of samples per prediction')
    
    parser.add_argument('--eval_multiple', action='store_true', help='eval over 20 seeds')
    
    parser.add_argument('--eval_multiple_CL', action='store_true', help='eval over 20 seeds using windowed approach based on lags')
    parser.add_argument('--eval_multiple_zero_shot', action='store_true', help='eval over 20 seeds using windowed approach based on lags')
    
    
    parser.add_argument('--use_rope_scaling', action='store_true', help='use_rope_scaling: call this if want to use it')
    
    parser.add_argument('--num_samples', default=100, type=int, help='Number of samples per prediction')
    
    parser.add_argument('--model_path', default='./lag-llama.ckpt', type=str, metavar='MODEL',
                        help='Name of model_path to save')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    
    parser.add_argument('--dist_on_itp', action='store_true')
    
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def train_model(backtrain_dataset=None,
                device=torch.device("cuda:0"), 
                context_length=32,
                num_samples=100,
                prediction_length=1,
                model_path="../../models/lag_llama_models/lag-llama.ckpt",
                use_rope_scaling=False,
                max_epochs=1):
    print("train_model() function..")
    ckpt = torch.load(model_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    
    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }
    print([print("using rope..") if use_rope_scaling else print("not using rope")])

    estimator = LagLlamaEstimator(
            ckpt_path=model_path,
            prediction_length=prediction_length,
            context_length=context_length,

            # distr_output="neg_bin",
            # scaling="mean",
            nonnegative_pred_samples=True,
            aug_prob=0,
            lr=5e-4,

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },

            batch_size=64,
            num_parallel_samples=num_samples,
            trainer_kwargs = {"max_epochs": max_epochs,}, # <- lightning trainer arguments
        )
    predictor = estimator.train(backtrain_dataset, cache_data=False, shuffle_buffer_length=1000)
    return predictor


def test_data_per_CL(
                     trained_model=None,
                     data_test=None,
                     context_length=32,
                     prediction_length=1,
                     device = torch.device("cuda:0"), 
                     num_samples = 100,
                     model_path="../../models/lag_llama_models/lag-llama.ckpt",
                    ):
    all_forecasts = []
    all_tss = []
    all_metrics = []

    for idx in range(0, len(data_test) - context_length + 1):
        # print(f"sample {idx} of {len(data_test)}")
        X_test = data_test.iloc[idx:idx + context_length]
        backtest_dataset = PandasDataset(X_test, target="target", freq="T")
        forecasts, tss = make_evaluation_predictions(
                            dataset=backtest_dataset,
                            predictor=trained_model,
                            num_samples=num_samples)

        forecasts_ctx = list(forecasts)
        tss_ctx = list(tss)
        y_test = data_test.iloc[idx + context_length - 1]  # Index of the last value

        y_pred = forecasts_ctx[0].mean
        all_forecasts.append(y_pred)
        all_tss.append(y_test)
    y_pred_np = np.array(all_forecasts)
    y_test_np = np.array(all_tss)

    return calculate_metrics(y_pred_np, y_test_np), y_pred_np, y_test_np


def train_model_training_with_context_length(data_train=None,
                device=torch.device("cuda:0"), 
                context_length=32,
                num_samples=100,
                prediction_length=1,
                model_path="../../models/lag_llama_models/lag-llama.ckpt",
                use_rope_scaling=False,
                max_epochs=1):
    print("train_model_training_with_context_length..")

    for idx in range(0, len(data_train) - context_length + 1):
            # print(f"sample {idx} of {len(data_train)}")
            X_train = data_train.iloc[idx:idx + context_length]
            backtrain_dataset = PandasDataset(X_train, target="target", freq="T")
            ckpt = torch.load(model_path, map_location=device)
            estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
            
            rope_scaling_arguments = {
                                    "type": "linear",
                                    "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
                                }
            estimator = LagLlamaEstimator(
                    ckpt_path=model_path,
                    prediction_length=prediction_length,
                    context_length=context_length,

                    # distr_output="neg_bin",
                    # scaling="mean",
                    nonnegative_pred_samples=True,
                    aug_prob=0,
                    lr=5e-4,

                    # estimator args
                    input_size=estimator_args["input_size"],
                    n_layer=estimator_args["n_layer"],
                    n_embd_per_head=estimator_args["n_embd_per_head"],
                    n_head=estimator_args["n_head"],
                    time_feat=estimator_args["time_feat"],

                    rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

                    batch_size=64,
                    num_parallel_samples=num_samples,
                    trainer_kwargs = {"max_epochs": max_epochs,}, # <- lightning trainer arguments
                )
            predictor = estimator.train(backtrain_dataset, cache_data=False, shuffle_buffer_length=1000)
    return predictor

def get_lag_llama_predictions_with_checkpoint(dataset, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100, model_checkpoint = None):
    ckpt = torch.load(model_checkpoint, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=model_checkpoint,
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss

def prepare_data_gluon(data):
    data.reset_index(inplace=True)
    data.columns = ['timestamp', 'target']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    for col in data.columns:
        if data[col].dtype != 'object' and not pd.api.types.is_string_dtype(data[col]):
            data[col] = data[col].astype('float32')
    data.index.rename(None, inplace=True)
    return data

def train_and_evaluate(seed, args):
    
    data_train, data_test = load_data(args.data_path, small_sample=args.small_sample)
    data_train = prepare_data_gluon(data_train)
    data_test = prepare_data_gluon(data_test)

    print("=" * 30)
    print("|      Training Data:        |")
    print("=" * 30)
    print(f"Shape {data_train.shape} - value range: ({data_train.target.min()} - {data_train.target.max()})")
    print("-" * 30)
    print("|      Testing Data:         |")
    print("=" * 30)
    print(f"Shape {data_test.shape} - value range: ({data_test.target.min()} - {data_test.target.max()})")
    print("-" * 30)
    
    prediction_length = 1 
        
    context_length = args.context_length
    
    context_length_test = args.context_length_test

    seeds = args.num_seeds

    device = torch.device(args.device)
        
    num_samples = args.num_samples
        
    use_rope_scaling = args.use_rope_scaling
    
    max_epochs = args.max_epochs
    
    model_path = args.model_path
    
    output_folder = args.output_folder
    
    output_folder = f"./outputs/Exp3/{args.output_folder}"
        
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = args.output_file

    backtrain_dataset = PandasDataset(data_train, target="target")
        
    backtest_dataset = PandasDataset(data_test, target="target")
    
    # preparing for time series forecasting
    
    AUTOREGRESSIVE=True

    supervised_train_data = ts_supervised_structure(data_train[["target"]], n_in=context_length, n_out=1, autoregressive=AUTOREGRESSIVE)
    supervised_test_data = ts_supervised_structure(data_test[["target"]], n_in=context_length, n_out=1, autoregressive=AUTOREGRESSIVE)
    X_train = supervised_train_data.iloc[:,:-1] 
    y_train = supervised_train_data.iloc[:,-1] 
    X_test = supervised_test_data.iloc[:,:-1] 
    y_test = supervised_test_data.iloc[:,-1]
    X_test.shape, y_test.shape

    #alternative
    supervised_test_data.rename(columns={'var1(t)': 'target'}, inplace=True)
    
    if args.zero_shot: 
        
        all_forecasts = []
        all_tss = []
        all_metrics = []

        for idx in tqdm(range(0, len(data_test) - context_length + 1), desc="Processing samples"):
            print(f"sample {idx} of {len(data_test)}")

            X_test = data_test.iloc[idx:idx + context_length]
            backtest_dataset = PandasDataset(X_test, target="target", freq="T")
            forecasts, tss = get_lag_llama_predictions_with_checkpoint(
                        dataset=backtest_dataset,
                        prediction_length=prediction_length,
                        device=device,
                        context_length=context_length,
                        use_rope_scaling=use_rope_scaling,
                        num_samples=num_samples,
                        model_checkpoint = model_path
                    )

            forecasts_ctx = list(forecasts)
            
            y_test = data_test.iloc[idx + context_length - 1]  # Index of the last value

            y_pred = forecasts[0].mean
            
            print(f"y_test: {y_test} vs y_pred: {y_pred}")
            all_forecasts.append(y_pred)
            all_tss.append(y_test)

        y_pred_np = np.array(all_forecasts)
        y_test_np = np.array(all_tss)
        
        metrics = calculate_metrics(y_pred_np, y_test_np)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test_np, label='Ground Truth')
        ax.plot(y_pred_np, label='Predicted', alpha=0.7)
        ax.set_ylabel('CPU')
        ax.set_xlabel('Timestamp')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_folder, f"prediction_plot_{args.output_file}.png")
        plt.savefig(fig_path)
        plt.close()

        # Save metrics
        result_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        metrics_path = os.path.join(output_folder, f"combined_metrics_{args.output_file}.csv")
        result_df.to_csv(metrics_path, index=False)

    #Evaluate over 20 seeds individual model
    if args.eval_single:
        data_list = []
        for seed in range(seeds): # lets get the metrics at each seed
            backtrain_dataset = PandasDataset(data_train, target="target")
            if args.eval_multiple_CL:
                trained_model = train_model_training_with_context_length(data_train,
                                                            device, 
                                                            args.context_length,
                                                            num_samples,
                                                            prediction_length,
                                                            args.model_path,
                                                            args.use_rope_scaling,
                                                            args.max_epochs)# ensure here we store the y_pred, y_test and metrics and model
            else: 
                trained_model = train_model(backtrain_dataset,
                                                            device, 
                                                            args.context_length,
                                                            num_samples,
                                                            prediction_length,
                                                            model_path,
                                                            args.use_rope_scaling,
                                                            args.max_epochs)# ensure here we store the y_pred, y_test and metrics and model

            metrics,y_pred_np, y_test_np = test_data_per_CL(
                             trained_model=trained_model,# we need to input model here
                             data_test=data_test,
                             context_length=args.context_length_test,
                             prediction_length=prediction_length,
                             device = device, 
                             num_samples = num_samples,
                             model_path = model_path)
            metrics = pd.DataFrame([metrics])
            data_list.append(metrics)
            temp_folder = f"{output_folder}/seed_{seed}"
            os.makedirs(temp_folder, exist_ok=True)
            # torch.save(trained_model, os.path.join(temp_folder, f'model_seed_{seed}.pth'))
            pd.DataFrame(y_pred_np).to_csv(f"{temp_folder}/y_pred_np.csv", index=False)
            pd.DataFrame(y_test_np).to_csv(f"{temp_folder}/y_test_np.csv", index=False)          
            torch.save(trained_model, os.path.join(temp_folder, f'model.pth'))
            metrics.to_csv(f"{temp_folder}/model_metrics_seed_{seed}.csv", index=False)
            print(f"Seed {seed+1} and metrics: {metrics}")
        combined_df = pd.concat(data_list)
        exp_name = f"p1_{args.context_length}CL_tested_on_{context_length_test}CL"
        combined_df["Experiment_Name"] = exp_name
        summary = combined_df.groupby(['Experiment_Name']).agg({
                                                                'mae' :['mean', 'std'],
                                                                'mape':['mean', 'std'],
                                                                'rmse'   :['mean', 'std'],
                                                                'r2' :['mean', 'std'],
                                                                'smape' :['mean', 'std'],
                                                                'mase' :['mean', 'std']
                                                                }).reset_index()
        
        summary.to_csv(f"{output_folder}/model_metrics_{output_file}.csv", index=False)

    if args.finetune:
        
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
                ckpt_path=model_path,
                prediction_length=prediction_length,
                context_length=context_length,
                
                nonnegative_pred_samples=True,
                aug_prob=0,
                lr=args.lr,

                # estimator args
                input_size=estimator_args["input_size"],
                n_layer=estimator_args["n_layer"],
                n_embd_per_head=estimator_args["n_embd_per_head"],
                n_head=estimator_args["n_head"],
                time_feat=estimator_args["time_feat"],
                batch_size=args.batch_size,
                num_parallel_samples=num_samples,
                trainer_kwargs = {"max_epochs": 50,}, # <- lightning trainer arguments
            )
        predictor = estimator.train(backtrain_dataset, cache_data=True, shuffle_buffer_length=1000)
        print("finished training.. ")
        
    #Evaluate over 20 seeds individual model
    if args.eval_multiple_zero_shot:
        """
        the main difference with this experiment is that we do not train the model
        """
        

        list_context_length = [32, 64, 128, 256]
        RoPE_list = [True, False]
        models_avg_seed = []
        overall_metrics = []


        # Initialize an empty dataframe to store results
        summary_df = pd.DataFrame()

        for cl in list_context_length:
            for rope_temp in RoPE_list:
                data_list = []  # Reset data_list for each combination of cl and rope_temp
                for seed in range(seeds):  # Get metrics for each seed
                    print(f"Seed {seed} ")
                    all_forecasts = []
                    all_tss = []
                    all_metrics = []

                    # Inference
                    start_inference = time.time()    
                    for idx in tqdm(range(0, len(data_test) - cl + 1), desc="Processing samples"):
                        X_test = data_test.iloc[idx:idx + cl]
                        backtest_dataset = PandasDataset(X_test, target="target", freq="T")
                        forecasts, tss = get_lag_llama_predictions_with_checkpoint(
                            dataset=backtest_dataset,
                            prediction_length=prediction_length,
                            device=device,
                            context_length=cl,
                            use_rope_scaling=rope_temp,  # Use rope_temp here
                            num_samples=num_samples,
                            model_checkpoint=args.model_path
                        )

                        forecasts_ctx = list(forecasts)

                        y_test = data_test.iloc[idx + cl - 1]  # Index of the last value

                        y_pred = forecasts[0].mean

                        all_forecasts.append(y_pred)
                        all_tss.append(y_test)

                    # End of Inference
                    end_inference = time.time()

                    inference_time = end_inference - start_inference

                    y_pred_np = np.array(all_forecasts)
                    y_test_np = np.array(all_tss)

                    # Calculating metrics
                    metrics = calculate_metrics(y_pred_np, y_test_np)
                    metrics['Inference Time'] = inference_time

                    temp_folder = f"{output_folder}/testbed_{seed}"
                    model_folder = f"{temp_folder}/model_CL_{cl}_RoPE_{rope_temp}"
                    os.makedirs(model_folder, exist_ok=True)

                    # Saving metrics and predictions for each seed
                    pd.DataFrame(y_pred_np, columns=["y_pred"]).to_csv(
                        f"{model_folder}/y_pred_np__seed_{seed}.csv", index=False)
                    pd.DataFrame(y_test_np, columns=["y_test"]).to_csv(
                        f"{model_folder}/y_test_np__seed_{seed}.csv", index=False)
                    pd.DataFrame([metrics]).to_csv(f'{model_folder}/model_metrics_CL_{cl}_RoPE_{rope_temp}.csv', index=False)

                    data_list.append(pd.DataFrame([metrics]))

                    print(f"Seed {seed} and metrics: {metrics}")

                # Get means + std for each individual model
                combined_df = pd.concat(data_list)
                combined_df["Experiment_Name"] = f"zs_CL_tested_on_{cl}_CL_{rope_temp}"
                summary = combined_df.groupby(['Experiment_Name']).agg({
                    'mae': ['mean', 'std'],
                    'rmse': ['mean', 'std'],
                    'r2': ['mean', 'std'],
                    'smape': ['mean', 'std'],
                    'mase': ['mean', 'std'],
                    'Inference Time': ['mean'],
                }).reset_index()

                # Append summary statistics to the main dataframe
                summary_df = pd.concat([summary_df, summary])

        # Save the combined summary to a single CSV file
        summary_df.to_csv(f'{output_folder}/summary.csv', index=False)

    #Evaluate over 20 seeds individual model
    if args.eval_multiple:
        """
        combine all trainable windows with all combinations of test windows and RoPe
        """

        list_context_length = [32, 64, 128, 256]
        RoPE_list = [True, False]
        models_avg_seed = []
        data_list = []
        overall_metrics = []

        os.makedirs(output_folder, exist_ok=True)

        for cl in list_context_length:
            for rope_temp in RoPE_list:
                combined_metrics = []
                for seed in range(seeds):
                    data_list_temp = []
                    for new_CL in list_context_length:
                        indv_metrics = []
                        print(f"Combination: {cl} vs {new_CL} with RoPE: {rope_temp}".center(60, "-"))
                        torch.manual_seed(seed)

                        # Training
                        start_training = time.time()
                        backtrain_dataset = PandasDataset(data_train, target="target")  # assuming this is defined
                        trained_model = train_model(backtrain_dataset, device, cl, num_samples, prediction_length, model_path, rope_temp, max_epochs)
                        end_training = time.time()
                        training_time = end_training - start_training
                        model_memory = bytes_to_mb(asizeof.asizeof(trained_model))

                        # Inference
                        start_inference = time.time()
                        metrics, y_pred_np, y_test_np = test_data_per_CL(trained_model=trained_model, data_test=data_test, context_length=new_CL, prediction_length=prediction_length, device=device, num_samples=num_samples, model_path=model_path)
                        end_inference = time.time()
                        inference_time = end_inference - start_inference

                        # Metrics calculation
                        metrics['Training Time'] = training_time
                        metrics['Inference Time'] = inference_time
                        metrics['Model memory (MB)'] = model_memory

                        # save locally
                        
                        model_folder = f"{output_folder}/testbed_{seed}/Finetuned_{cl}CL_{rope_temp}_RoPE_tested_on_{new_CL}CL"
                        os.makedirs(model_folder, exist_ok=True)
                        pd.DataFrame([metrics]).to_csv(f'{model_folder}/model_metrics_CL_{cl}_RoPE_{rope_temp}.csv', index=False)
                        torch.save(trained_model, os.path.join(model_folder, f'model_CL_{cl}_RoPE_{rope_temp}_tested_on_{new_CL}CL_seed_{seed}.pth'))
                        pd.DataFrame(y_test_np,columns=["y_test"]).to_csv(f'{model_folder}/y_test_np_CL_{cl}_RoPE_{rope_temp}_tested_on_{new_CL}CL_seed_{seed}.csv', index=False)
                        pd.DataFrame(y_pred_np,columns=["y_pred"]).to_csv(f'{model_folder}/y_pred_np_CL_{cl}_RoPE_{rope_temp}_tested_on_{new_CL}CL_seed_{seed}.csv', index=False)
                        
                        # create name per metric to ensure correct files and add seed for security
                        metrics['Model'] = f'Finetuned_{cl}CL_{rope_temp}_RoPE_tested_on_{new_CL}CL'
                        metrics['Seed'] = seed 
                        # save metrics per seed, per new_CL model
                        combined_metrics.append(metrics)
                # now for this model, we calculate the means and std over those 20 seeds
                model_summary = pd.DataFrame(combined_metrics).groupby('Model').agg({
                            'mae': ['mean', 'std'],
                            'rmse': ['mean', 'std'],
                            'r2': ['mean', 'std'],
                            'smape': ['mean', 'std'],
                            'mase': ['mean', 'std'],
                            'Training Time': ['mean'],
                            'Inference Time': ['mean'],
                            'Model memory (MB)': ['mean']
                        }).reset_index()
                
                # we save this single row into a list
                overall_metrics.append(model_summary)

                print(overall_metrics)
        # Here we should have all possible combinations into a single dataframe, containing the mean + stds
        summary_df = pd.concat(overall_metrics)

        # Save all metrics in a single CSV outside the seed folders
        summary_df.to_csv(f"{output_folder}/summary.csv", index=False)
        
        
def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True
    
    train_and_evaluate(0, args)
    
    print("Finished..")
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
