import time
import random
import subprocess

def run():
    num_cores = 8
    timeout = "60m"
    cpu_perc = random.randint(0,100)
    print(f"Running cpu utilization at {cpu_perc}%")
    subprocess.call(["stress-ng","-c",str(num_cores),"-l",str(cpu_perc),"--timeout",str(timeout)])

if __name__ == '__main__':
    while True:
        print("Starting stress-ng test")
        run()
        print("Stress-ng test completed")
        time.sleep(60)
        print("Sleeping for 60 seconds")
