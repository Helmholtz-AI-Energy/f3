# Feed-Forward Optimization With Delayed Feedback for Neural Networks

**F**eed-**F**orward with delayed **F**eedback (F³) is a novel, biologically-inspired algorithm to train neural networks without backpropagation.
It solves the weight transport and update locking problems, two of backpropagation's core issues prohibiting biological plausibility.

This repository implements training and evaluating F³ for fully-connected neural networks on different classification and regression datasets.

## Installation

This project can be installed using the prodivded `setup.py`, e.g. via
```
pip install <path to project root>
```
Alternatively, the exact dependencies used for our results can be installed using the provided `requirements.txt` via
```
pip install -r requirements.txt
```

## Usage

### Prepare the data

The script `prepare_data.py` is used to download the data from sklearn and the UCI repository, apply transformations like standardization, and save it in the format expected by the training script.
This script should be called once before training on any dataset except for MNIST and CIFAR-10.  
Which datasets are prepared can be selected via command line parameters.
See `python prepare_data.py -h` for a detailed description of the available parameters.

### Training

The script `start_training.py` is used to start the actual training.
It supports different training algorithms and datasets for fully-connected neural networks.
The results are collected and saved as a CSV file in the output directory (default `results/`).  
See `python start_training.py -h` for a detailed description of the available parameters.

### Reproduce our results

To reproduce our results, first download and prepare the data as described above.
Then start the training using `start_training.py`.

We used the following hyperparameters:
- seeds: `--seed <seed>` with seeds from 1 to 10
- datasets
  - MNIST: `--dataset mnist --batch_size 50 --epochs 100`
  - SGEMM: `--dataset sgemm --batch_size 512 --epochs 200`
  - CIFAR-10: `--dataset cifar10 --batch_size 100 --epochs 200`
  - Wine Quality: `--dataset wine_quality_regression --batch_size 50 --epochs 500`
- algorithms
  - Backpropagation: `--mode bp`
  - Shallow Learning: `--mode llo`
  - DFA: `--mode f3 --error_info current_error`
  - DRTP: `--mode f3 --error_info one_hot_target`
  - F³: `--mode f3 --error_info <error information>` with error information being one of: `delayed_loss`, `delayed_error`, `delayed_loss_one_hot`, `delayed_error_one_hot`, `delayed_loss_softmax`, `delayed_error_softmax`
- initializations
  - Kaiming: `--initialization_method kaiming_uniform`
  - Trinomial: `--initialization_method discrete_uniform --scalar 1 --discrete_values -1 0 1`
  - Binomial: `--initialization_method discrete_uniform --scalar 1 --discrete_values 0 1`
  - ±I: `--initialization_method identity_repeat_pm`
- model: `--depth <depth> --width <width>` where the depth is the number of hidden layers in a fully-connected neural network and the width is the number of neurons in these hidden layers
- learning rate: `--lr <learning rate>` with the learning rates given in the following table

| Dataset      |      BP |     LLO |     DFA |    DRTP |      F3 |
|:-------------|--------:|--------:|--------:|--------:|--------:|
| MNIST        | 1.5e-04 | 1.5e-02 | 1.5e-04 | 1.5e-04 | 1.5e-04 |
| Wine Quality | 1.0e-04 | 1.0e-04 | 1.0e-04 | 1.0e-04 | 1.0e-04 |
| SGEMM d=1    | 1.0e-02 | 1.0e-02 | 1.0e-04 | 1.0e-04 | 1.0e-04 |
| SGEMM d=2    | 1.0e-03 | 1.0e-03 | 1.0e-05 | 1.0e-05 | 1.0e-05 |
| SGEMM d≥10   | 1.0e-04 | 1.0e-04 | 1.0e-05 | 1.0e-05 | 1.0e-05 |

For example, to reproduce training a network with one hidden layer with 500 neurons on MNIST with F³-Error and trinomial initialization of the feedback weights with seed 0 you would call
```
python start_training.py --seed 0 --depth 1 --width 500 --epochs 100 --dataset mnist --batch_size 50 --mode f3 --error_info delayed_error_one_hot --initialization_method discrete_uniform --scalar 1 --discrete_values -1 0 1 --lr 0.001
```

#### Transformer Proof-of-Concept
To run the transformer training, use
```
python transformer.py <seed>
```