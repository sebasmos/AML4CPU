# AML4CPU: Implementation of Adaptive Machine Learning for Resource-Constrained Environments

Official PyTorch and River implementation of *Adaptive Machine Learning for Resource-Constrained Environments* presented at DELTA 2024, ACM SIGKDD KDD 2024, Barcelona, Spain.

Please cite the following paper when using AML4CPU:
```
S. A. Cajas, J. Samanta, A. L. Suárez-Cetrulo, and R. S. Carbajo, "Adaptive Machine Learning for Resource-Constrained Environments," in Discovering Drift Phenomena in Evolving Landscape (DELTA 2024) Workshop at ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2024), Barcelona, Catalonia, Spain, Aug. 26, 2024.
```

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

![exp1](https://github.com/sebasmos/AML4CPU/blob/main/data/exp1.png)

### Experiment 2: Pre-sequential Evaluations

Run the pre-sequential evaluation script:
```bash
python run_pre_sequential.py --output_file 'exp2' --eval --output_folder Exp2 --num_seeds 20
```

![exp2](https://github.com/sebasmos/AML4CPU/blob/main/data/exp2.png)



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

### Acknoledgements

We are grateful to our colleagues at the EU Horizon project ICOS and Ireland’s Centre for Applied AI for helping to start and shape this research effort. Our advancement has been made possible by funding from the European Union’s HORIZON research and innovation program (Grant No. 101070177).

Please cite as:

```
@inproceedings{Cajas2024,
  author    = {Sebastián Andrés Cajas and
               Jaydeep Samanta and
               Andrés L. Suárez-Cetrulo and
               Ricardo Simón Carbajo},
  title     = {Adaptive Machine Learning for Resource-Constrained Environments},
  booktitle = {Discovering Drift Phenomena in Evolving Landscape (DELTA 2024) Workshop at ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2024)},
  address   = {Barcelona, Catalonia, Spain},
  year      = {2024},
  month     = {August 26},
}
```
