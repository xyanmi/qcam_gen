# QCAM: Quantum Circuit Autoregressive Model

This is a repository for QCAM experiments. 

## Repository layout

- `qcam/`: QCAM package for `QCAM(L, n, m=1)`, where one qubit is measured.
- `qcam_seq/`: sequence-model package for `QCAM(L, n)`, where all qubits are measured.
- `longtext.py`: context-window experiment entry point.
- `config.toml`: dataset configuration for `qcam`.
- `config_seq.toml`: dataset configuration for `qcam_seq`.
- `data/`: datasets used by the examples.
- `results/` and `results_seq/`: generated training outputs.

## Quick start

### Train on the bars-and-stripes dataset

```bash
python -m qcam --trainer --data bs4 --n 1 --depth 2 --gate rxry --proj cnot --epochs 2000 --device 0
```

Sample from a trained model:

```bash
python -m qcam --sampler --path "./results/bs4/n_1_depth2_reset(False)_proj(cnot)_gate(rxry)_lr0.01_epochs2000" --bs 100
```

### Train sequence models

```bash
python -m qcam_seq --trainer --data genseq --lr 0.001 --bs 1000 --show 1 --gate rxry --proj cnot --device 0 --test 10 --epochs 2500 --depth 5 --mask
python -m qcam_seq --trainer --data easyseq --lr 0.001 --show 1 --gate rxry --proj cnot --device 0 --epochs 2500 --depth 5 --mask
```

Sample from a trained sequence model:

```bash
python -m qcam_seq --sampler --path "./results_seq/aiseq128/depth5_reset(False)_proj(cnot)_gate(rxry)_lr0.001_epochs5000" --bs 100
```

### Train with a context window

```bash
python longtext.py --trainer --model eqcam --lr 0.005 --show 1 --device 0 --max_iters 10000 --checkpoint 1000 --depth 10 --bs 1024 --block_size 32
```

## Smoke tests

The following commands can be used to check whether the command-line entry points are available:

```bash
python -m qcam --help
python -m qcam_seq --help
python longtext.py --help
```

Small training smoke tests can be run by reducing the training length, for example:

```bash
python -m qcam --trainer --data bs4 --n 1 --depth 2 --gate rxry --proj cnot --epochs 1 --device 0 --checkpoint 0 --show 1
python -m qcam_seq --trainer --data easyseq --lr 0.001 --show 1 --gate rxry --proj cnot --device cpu --epochs 1 --depth 5 --mask --checkpoint 0
```

## `qcam` usage

`qcam` trains and samples QCAM models with one measured qubit.

### Training

```bash
python -m qcam --trainer --data DATASET_NAME [--lr 0.005] [--epochs 2000]
```

### Sampling

```bash
python -m qcam --sampler --path "PATH_TO_SAVED_MODEL" [--bs 100]
```

### Options

```text
usage: __main__.py [-h] [--seed SEED] [--device DEVICE] [--path PATH]
                   [--bs BS] [--trainer] [--sampler] [--data DATA]
                   [--epochs EPOCHS] [--lr LR] [--test TEST]
                   [--checkpoint CHECKPOINT] [--test_bs TEST_BS] [--n N]
                   [--depth DEPTH] [--reset] [--proj {cnot,cz,swap}]
                   [--gate {rx,ry,rxry}] [--show SHOW] [--interval INTERVAL]

Quantum Circuit Model Trainer and Sampler

options:
  -h, --help            show this help message and exit
  --seed SEED           Random seed for reproducibility.
  --device DEVICE       CUDA device index to use, for example '0' or '1'.
  --path PATH           Path to a saved model directory for sampling.
  --bs BS               Batch size for training. If 0, the entire dataset is used as one batch.
  --trainer             Enable this flag to train a new model.
  --sampler             Enable this flag to sample from a trained model specified by --path.
  --data DATA           Name of the dataset configuration in 'config.toml'.
  --epochs EPOCHS       Number of training epochs.
  --lr LR               Learning rate for the Adam optimizer.
  --test TEST           Whether to use a test set for validation. 0 disables validation; a positive integer controls the train/test split.
  --checkpoint CHECKPOINT
                        Checkpoint saving interval. 0 disables checkpointing.
  --test_bs TEST_BS     Batch size for the test set. If 0, use the full test set.
  --n N                 Number of qubits in the circuit.
  --depth DEPTH         Depth of each circuit block.
  --reset               Reset data qubits to the |0> state after each block.
  --proj {cnot,cz,swap}
                        Projection gate to use.
  --gate {rx,ry,rxry}   Single-qubit gate to use.
  --show SHOW           Interval for displaying loss during training, in epochs.
  --interval INTERVAL   Minimum time interval, in seconds, for updating the tqdm progress bar.
```

## `qcam_seq` usage

`qcam_seq` trains and samples QCAM sequence models with all qubits measured.

### Training

```bash
python -m qcam_seq --trainer --data aiseq128 --depth 6 --lr 0.001 --show 1 --gate rxry --proj cnot --device 0 --epochs 5000
```

### Sampling

```bash
python -m qcam_seq --sampler --path "./results_seq/aiseq128/depth5_reset(False)_proj(cnot)_gate(rxry)_lr0.001_epochs5000" --bs 100
```

### Options

```text
usage: __main__.py [-h] [--seed SEED] [--device DEVICE] [--path PATH]
                   [--bs BS] [--trainer] [--sampler] [--interact]
                   [--epochs EPOCHS] [--lr LR] [--data DATA] [--depth DEPTH]
                   [--show SHOW] [--interval INTERVAL] [--reset]
                   [--proj {cnot,cz,swap}] [--gate {rx,ry,rxry}]
                   [--checkpoint CHECKPOINT] [--test TEST] [--test_bs TEST_BS]
                   [--mask] [--prompt PROMPT]

Quantum Circuit Model Trainer and Sampler

options:
  -h, --help            show this help message and exit

General Settings:
  --seed SEED           Random seed for reproducibility.
  --device DEVICE       Device to use, for example '0' for cuda:0 or 'cpu'.
  --path PATH           Path to the saved model for sampling.
  --bs BS               Batch size. If 0, use the full dataset for training or 100 for sampling.

Mode Selection:
  --trainer             Activate training mode.
  --sampler             Activate sampling mode.
  --interact            Activate interactive console mode.

Training Settings:
  --epochs EPOCHS       Number of training epochs.
  --lr LR               Learning rate for the Adam optimizer.
  --data DATA           TOML variable key for the dataset configuration.
  --depth DEPTH         Depth of the circuit block.
  --show SHOW           Interval for displaying loss during training.
  --interval INTERVAL   Update interval for the tqdm progress bar, in seconds.
  --reset               Whether to reset data qubits after each step.
  --proj {cnot,cz,swap}
                        Projection gate to use.
  --gate {rx,ry,rxry}   Single-qubit gate to use.
  --checkpoint CHECKPOINT
                        Checkpoint saving interval. 0 disables checkpointing.
  --test TEST           Whether to use a test set for validation. 0 disables validation; a positive integer controls the train/test split.
  --test_bs TEST_BS     Batch size for the test set. If 0, use the full test set.
  --mask                Use the mask in the loss function.

Sampling Settings:
  --prompt PROMPT       Prompt for text generation. The value should be a JSON file name.
```

