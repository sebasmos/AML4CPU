import os
import pandas as pd
import argparse
from src.datasets import load_orangepi_data

def get_args_parser():
    
    parser = argparse.ArgumentParser('CPU regression', add_help=False)
    parser.add_argument('--num', default=40000, type=int,
                        help='define the total number of samples to use from the dataset - by default its 40k for the CPU dataset')
    parser.add_argument('--data_path', default='./data/cpu_data_custom_adapted.csv', type=str,
                        help='dataset path')
    return parser

def main(args):
    df = load_orangepi_data(args.data_path)
    print(f"full size dataset: ", len(df))
    """
    1500 samples for 1 day
    10k for 1 week
    """

    df = df[["Date", "TARGET"]]
    df.columns = ['timestamp', 'target']
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.set_index('timestamp', inplace=True)

    for col in df.columns:
            if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
                df[col] = df[col].astype('float32')
    df.index.rename(None, inplace=True)

    max_end = df.index[-1]
    beginning = df.index[0]

    new_index = pd.date_range(beginning, end=max_end, freq="1T")

    new_index = new_index[:len(df)]

    df.index = new_index

    size = int(len(df) * 0.8)

    data_train, data_test = df[0:size], df[size:len(df)]
    
    data_folder = "./data"
    os.makedirs(data_folder, exist_ok=True)
    data_train.to_csv(os.path.join(data_folder, 'train_data.csv'))
    data_test.to_csv(os.path.join(data_folder, 'test_data.csv'))

    print("=" * 30)
    print(f"|     NEW DATA Data Information      |")
    print("=" * 30)
    print("|      Training Data:        |")
    print("=" * 30)
    print(f"Shape {data_train.shape} - value range: ({data_train.target.min()} - {data_train.target.max()})")
    print("-" * 30)
    print("|      Testing Data:         |")
    print("=" * 30)
    print(f"Shape {data_test.shape} - value range: ({data_train.target.min()} - {data_train.target.max()})")
    print("-" * 30)


    return print("Done!")

def main_op(args):
    df = load_orangepi_data(args.data_path)
    df = df[-args.num:]

    df = df[["Date", "TARGET"]]

    df.columns = ['timestamp', 'TARGET']

    df = df[['timestamp', 'TARGET']]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.set_index('timestamp', inplace=True)

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')

    df.index.name = None

    max_end = df.index[-1]
    beginning = df.index[0]
    new_index = pd.date_range(beginning, end=max_end, freq="1T")
    new_index = new_index[:len(df)]
    df.index = new_index

    size = int(len(df) * 0.8)
    data_train, data_test = df[0:size], df[size:len(df)]

    data_folder = "./data"
    os.makedirs(data_folder, exist_ok=True)
    data_train.to_csv(os.path.join(data_folder, 'train_data.csv'))
    data_test.to_csv(os.path.join(data_folder, 'test_data.csv'))

    print("=" * 30)
    print(f"|     NEW DATA Data Information      |")
    print("=" * 30)
    print("|      Training Data:        |")
    print("=" * 30)
    print(f"Shape {data_train.shape} - value range: ({data_train.TARGET.min()} - {data_train.TARGET.max()})")
    print("-" * 30)
    print("|      Testing Data:         |")
    print("=" * 30)
    print(f"Shape {data_test.shape} - value range: ({data_train.TARGET.min()} - {data_train.TARGET.max()})")
    print("-" * 30)

    return data_train, data_test

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
