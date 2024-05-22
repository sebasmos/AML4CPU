# AML4CPU: Pytorch, River & ScikitLearn implementation.


### Catalog

- [x] Hold-out script - experiment 1 : `run_holdout.py`
- [x] Pre-sequential script - experiment 2 : `run_pre_sequential.py`
- [x] Zero-shot, finetuning Lag-Llamma: `run_finetune.py`

## Setting Up Your Environment

Let's get started by setting up your environment. 

1. **Create a Conda Environment:**
   ```bash
   conda create -n my_env python=3.10.0 -y
   conda activate my_env
   ```
2. **Clone the Repository and Install Requirements:**
   ```bash
   git clone *
   cd *
   pip install -r requirement.txt
   ```
Pytorch `
pip install clean-fid numba numpy torch==2.0.0+cu118 torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118`


## Experiments

###  *Experiment 1*: Holdout evaluation.


`python run_holdout.py --output_file 'exp1' --output_folder Exp1 --num_seeds 20`

### *Experiment 2 - pre-sequential evaluations"*: 

`python run_pre_sequential.py --output_file 'exp2' --eval --output_folder Exp2 --num_seeds 20`


### *Experiment 3*: Zero-shot and with fine-tuning with lag-llama

- Test zero-shot over different context lenghts (32, 64, 128, 256) with and without RoPE: 

`python run_finetune.py --output_file zs --output_folder zs  --model_path ./models/lag_llama_models/lag-llama.ckpt --eval_multiple_zero_shot --max_epochs 50 --num_seeds 20`


- Finetune and test lag-llama over different context lenghts (32, 64, 128, 256) with and without RoPE:

`python run_finetune.py --output_file exp3_REAL_parallel --output_folder Exp3  --model_path ./models/lag_llama_models/lag-llama.ckpt --max_epochs 50 --num_seeds 20 --eval_multiple `


### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.