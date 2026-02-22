# [NeurIPS'25] A Pre-training Framework for Relational Data with Information-theoretic Principles

## Package Installation

This project uses `Python 3.10.14`, `Pytorch 2.1.2`, and `Pytorch Geometric 2.4.0` on `CUDA 12.1`. Also, please install `Pytorch Frame 0.2.3` and `RelBench 1.1.0`. Please follow the below steps for installation.

```markdown
conda create --name rldl python=3.10.14 -y
conda activate rldl
conda install pytorch==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pyg=2.4.0=\*cu\* -c pyg -y
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install pytorch_frame[full]
pip install relbench[full]
pip install -r requirements.txt
```

Please download the zip file containing the training tables in this Google Drive: `https://drive.google.com/file/d/1X5UBgrCpiL2086IF076xysy5lh9n4TIj/view?usp=sharing`.
Please unzip it in the root directory of this repository.

## WandB

Our repository uses [WandB](https://wandb.ai/) to keep track of research experiments and hyperparameter tuning. Please follow the instruction on WandB official website to register and sign in via WandB CLI. Our codes automatically update research results to your WandB server.

## Experiments

### Pre-training

We can pre-train the model by executing one of the following line, where we replace `<config-file>` with the path to the config file in the following directories: `rel-amazon/item-ssl/`, `rel-amazon/item-tve-1-hop`, `rel-amazon/item-tve-2-hop`, `rel-amazon/user-ssl`, `rel-amazon/user-tve-1-hop`, `rel-amazon/user-tve-2-hop`, `rel-hm/item-ssl/`, `rel-hm/item-tve-1-hop`, `rel-hm/item-tve-2-hop`, `rel-hm/user-ssl`, `rel-hm/user-tve-1-hop`, and `rel-hm/user-tve-2-hop`. Details of TVE construction are in the directory `visualization`. We also provide pre-train tables already, so no need to run the scripts in that folder.

TVE pre-training:

```markdown
bash scripts/tve-dist-train.sh <config-file>
```

TVE-MAE pre-training:

```markdown
bash scripts/tve_mae-dist-train.sh <config-file>
```

TVE-CTR pre-training:

```markdown
bash scripts/tve_contrastive-dist-train.sh <config-file>
```

### Fine-tuning

We can run experiments from scratch (w/o any pre-training by executing one of the following line). The following scripts take `<config-file>` as the first argument, and `<cfg/default.yaml>` is the sample default config file. If you set the second argument to a directory, it will go over all the config files in the directory and run the experiments.

### Single GPU

```markdown
bash scripts/baseline-train.sh <config-file> cfg/default.yaml
```

### Multi GPUs

```markdown
bash scripts/baseline-dist-train.sh <config-file> cfg/default.yaml
```

If you want to run the experiments with pre-trained models, just add `--checkpoint <path-to-checkpoint>` to the command.
