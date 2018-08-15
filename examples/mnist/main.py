#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###
# Adapted to petastorm dataset using original contents from
# pytorch/examples/mnist/main.py .
###
from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torchvision import transforms

from petastorm.reader import Reader
from petastorm.tf_utils import tf_tensors
from petastorm.workers_pool.thread_pool import ThreadPool


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

_image_transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

class BatchMaker(object):
    def __init__(self, reader, batch_size, total_size):
        self.reader = reader
        self.batch_size = batch_size
        self.total_size = total_size

    def __len__(self):
        return (self.total_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        for mnist in self.reader:
            batch.append((_image_transform(mnist.image), mnist.digit))
            if len(batch) == self.batch_size:
                yield default_collate(batch)
                batch = []
        if len(batch) > 0:
            yield default_collate(batch)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_loader.total_size,
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader.total_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_loader.total_size,
        100. * correct / test_loader.total_size))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Petastorm MNIST Example')
    parser.add_argument('--dataset-url', type=str,
                        default='file:///{}'.format(os.path.abspath(os.path.join(
                            os.path.dirname(sys.argv[0]), '..', '..', '..'))),
                        metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST petastorm dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # Handle multiple epochs by reinstantiating reader per epoch
        with Reader('{}/train'.format(args.dataset_url), shuffle=True, reader_pool=ThreadPool(1), num_epochs=1) as reader:
            train(args, model, device, BatchMaker(reader, args.batch_size, 60000), optimizer, epoch)

        with Reader('{}/test'.format(args.dataset_url), shuffle=True, reader_pool=ThreadPool(1), num_epochs=1) as reader:
            test(args, model, device, BatchMaker(reader, args.batch_size, 10000))


if __name__ == '__main__':
    main()
