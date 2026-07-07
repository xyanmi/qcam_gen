# QCAM

This repository contains the code for the QCAM experiments.  The basic dependencies are:

```bash
pip install torch numpy tqdm
```

The commands below use GPU `0`.  Change `--device 0` to another GPU index if needed.

## Quick Start

### 1. Bars and stripes (`bs4`)

```bash
python -m qcam --trainer --data bs4 --n 1 --depth 2 --gate rxry --proj cnot --epochs 2000 --device 0 
python -m qcam_seq --trainer --data sequence128 --lr 0.001 --bs 1000 --show 1 --gate rxry --proj cnot --device 0 --test 10 --epochs 5000 --depth 5 --mask
python longtext.py --trainer --model eqcam --lr 0.005 --show 1 --device 0 --max_iters 10000 --checkpoint 1000 --depth 10 --bs 1024 --block_size 32
```

## Sampling

After training, sampling only needs the saved run directory.

```bash
python -m qcam --sampler --path "results/bs4/n_1_depth2_reset(False)_proj(cnot)_gate(rxry)_lr0.01_epochs2000" --bs 100 --device 0
python -m qcam_seq --sampler --path "results_seq/sequence128/depth5_reset(False)_proj(cnot)_gate(rxry)_lr0.001_epochs5000" --bs 100 --device 0
python longtext.py --sampler --path "LongtextResults/eqcam/bs1024_block32_lr0.005_depth10_seed42" --max_tokens 800 --device 0
```

For prompt completion with a sequence model, pass a JSON file containing a list
of prompt strings:

```bash
python -m qcam_seq --sampler --path "results_seq/sequence128/depth5_reset(False)_proj(cnot)_gate(rxry)_lr0.001_epochs5000" --prompt prompts.json --device 0
```

Generated samples are saved back into the same run directory.

## Main Arguments

- `--data`: dataset key from `config.toml` or `config_seq.toml`.
- `--device`: GPU index.  For example, `--device 0` uses `cuda:0`.
- `--depth`: number of circuit layers in each autoregressive block.
- `--gate`: single-qubit rotation gate.  The default choice here is `rxry`.
- `--proj`: entangling/projection gate.  The commands above use `cnot`.
- `--epochs`: training epochs for `qcam` and `qcam_seq`.
- `--max_iters`: training iterations for `longtext.py`.
- `--bs`: batch size.  In sampling mode, it is the number of generated samples.
- `--lr`: learning rate.
- `--test`: train/test split control for `qcam_seq`; `--test 10` keeps a small
  held-out part for validation.
- `--mask`: use the padding mask stored in the sequence dataset.
- `--block_size`: context length used by `longtext.py`.
- `--checkpoint`: save intermediate checkpoints every given number of
  iterations or epochs.  Use `0` to disable it.

For the full command-line options:

```bash
python -m qcam --help
python -m qcam_seq --help
python longtext.py --help
```
