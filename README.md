# AML4CPU: Implementation with PyTorch, River, and ScikitLearn

### Contents

- [x] Hold-out Script - Experiment 1: `run_holdout.py`
- [x] Pre-sequential Script - Experiment 2: `run_pre_sequential.py`
- [x] Zero-shot and Fine-tuning with Lag-Llama: `run_finetune.py`

## Setting Up Your Environment

Let's start by setting up your environment:

1. **Create a Conda Environment:**
   ```bash
   conda create -n AML4CPU python=3.10.12 -y
   conda activate AML4CPU
   ```
2. **Clone the Repository and Install Requirements:**
   ```bash
   git clone https://github.com/sebasmos/AML4CPU.git
   cd AML4CPU
   pip install -r requirements.txt
   ```

3. **Install PyTorch and Other Dependencies:**
   ```bash
   pip install clean-fid numba numpy torch==2.0.0+cu118 torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118
   ```

## Experiments

### Experiment 1: Holdout Evaluation

Run the holdout evaluation script:
```bash
python run_holdout.py --output_file 'exp1' --output_folder Exp1 --num_seeds 20
```

### Experiment 2: Pre-sequential Evaluations

Run the pre-sequential evaluation script:
```bash
python run_pre_sequential.py --output_file 'exp2' --eval --output_folder Exp2 --num_seeds 20
```

### Experiment 3: Zero-shot and Fine-tuning with Lag-Llama

#### Zero-shot Testing

Test zero-shot over different context lengths (32, 64, 128, 256) with and without RoPE:
```bash
python run_finetune.py --output_file zs --output_folder zs --model_path ./models/lag_llama_models/lag-llama.ckpt --eval_multiple_zero_shot --max_epochs 50 --num_seeds 20
```

#### Fine-tuning and Testing

Finetune and test Lag-Llama over different context lengths (32, 64, 128, 256) with and without RoPE:
```bash
python run_finetune.py --output_file exp3_REAL_parallel --output_folder Exp3 --model_path ./models/lag_llama_models/lag-llama.ckpt --max_epochs 50 --num_seeds 20 --eval_multiple
```

![exp3](https://github.com/sebasmos/AML4CPU/blob/main/data/exp3.png)



### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
