# Petastorm Pytorch Example

## Setup
```bash
PYTHONPATH=~/dev/petastorm  # replace with your petastorm install path
```

## Generating a Petastorm Dataset from MNIST Data

This creates both a `train` and `test` petastorm datasets in `/tmp/mnist`:

```bash
python generate_petastorm_mnist.py
```

## Pytorch training using the Petastormed MNIST Dataset

This will invoke a 10-epoch training run using MNIST data in petastorm form,
stored by default in `/tmp/mnist`, and show accuracy against the test set:

```bash
python main.py
```

```
usage: main.py [-h] [--dataset-url S] [--batch-size N] [--test-batch-size N]
               [--epochs N] [--all-epochs] [--lr LR] [--momentum M]
               [--no-cuda] [--seed S] [--log-interval N]

Petastorm MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --dataset-url S      hdfs:// or file:/// URL to the MNIST petastorm dataset
                       (default: file:///tmp/mnist)
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 10)
  --all-epochs         train all epochs before testing accuracy/loss
  --lr LR              learning rate (default: 0.01)
  --momentum M         SGD momentum (default: 0.5)
  --no-cuda            disables CUDA training
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
```
