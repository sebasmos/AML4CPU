import csv
import psutil
from datetime import datetime
import time
import os
import argparse

def collect_cpu_data():
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_percent_all = psutil.cpu_percent(interval=1, percpu=False)
    memory_percent = psutil.virtual_memory().percent
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time, cpu_percent, cpu_percent_all, memory_percent

def write_to_csv(file_path, data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='System Data Collection')
    parser.add_argument('--file', type=str, default='system_data.csv', help='CSV file path')

    args = parser.parse_args()

    csv_file = args.file

    is_new_file = not os.path.isfile(csv_file)
    if is_new_file:
        headers = ['Date']
        headers += [f'CPU Core {i} Usage (%)' for i in range(psutil.cpu_count())]
        headers += ['CPU Usage (All Cores) (%)', 'Memory Usage (%)']
        write_to_csv(csv_file, headers)

    while True:
        system_data = collect_cpu_data()
        write_to_csv(csv_file, system_data)
        print(f"System data written to {csv_file}.")
        time.sleep(60)
