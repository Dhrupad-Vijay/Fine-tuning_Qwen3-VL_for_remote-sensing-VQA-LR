import subprocess
import os
from pathlib import Path
import time
import argparse
import torch

PTH = Path(__file__).resolve().parent.parent

argp = argparse.ArgumentParser()
argp.add_argument("--path", required=True)
argp.add_argument("--model", default="4B")
argp.add_argument("--split", default="val")
args = argp.parse_args()

ft_4b = PTH / args.path
# ft_4b = Path("Sat_trial/results/runs/fine_tune_4B")
print(f"the path is - {ft_4b}")
list_of_pts = []
list_step = [step for step in os.listdir(ft_4b) if step.startswith("step")]
print(len(list_step))
current_env = os.environ.copy()

model = args.model
split = args.split
avaliable_devices = torch.cuda.device_count()

for i, step in enumerate(list_step):
    adapter_pth = ft_4b / step
    list_of_pts.append([step, adapter_pth])
    
    if len(list_of_pts) == avaliable_devices or (i+1) == len(list_step):
        active_processes = []
        for c, (name, adpt_pth) in enumerate(list_of_pts):
            print(f"name - {name} and pth - {adpt_pth}")
            gpu_env = current_env.copy()
            gpu_env["CUDA_VISIBLE_DEVICES"] = str(c)
            command = ["python", "scripts/evaluate.py", "--model", model, "--split", split, "--out", name, "--adapter", adpt_pth]

            p = subprocess.Popen(command, env=gpu_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            active_processes.append((name, p))

        print(f"Waiting for {len(active_processes)} process to complete....")

        for name, p in active_processes:
            stdout, stderr = p.communicate()
            print(f"\n=== Output for {name} ===")
            print(stdout)
            if p.returncode != 0:
                print(f"Error in {name}:")
                print(stderr)

            
        print(list_of_pts)
        list_of_pts = []
        time.sleep(10)
    