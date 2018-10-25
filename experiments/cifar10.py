import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import argparse
import random
import numpy as np

import torch
from torch import optim
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_

from macow.data import load_datasets, iterate_minibatches, get_batch, preprocess, postprocess
from macow.model import FlowGenModel
from macow.utils import exponentialMovingAverage


parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 524287)')
parser.add_argument('--n_bits', type=int, default=8, metavar='N', help='number of bits per pixel.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--opt', choices=['adam', 'adamax'], help='optimization method', default='adam')
parser.add_argument('--polyak', type=float, default=0.999, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--model_path', help='path for saving model file.', required=True)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')

dataset = 'cifar10'
imageSize = 32
nc = 3
nx = imageSize * imageSize * nc
n_bits = args.n_bits
n_bins = 2. ** n_bits

model_path = args.model_path
model_name = os.path.join(model_path, 'model.pt')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

train_data, test_data, n_val = load_datasets(dataset)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
val_index = train_index[-n_val:]
train_index = train_index[:-n_val]

test_index = np.arange(len(test_data))
np.random.shuffle(test_index)

print(len(train_index))
print(len(val_index))
print(len(test_index))

polyak_decay = args.polyak
params = json.load(open(args.config, 'r'))
json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
fgen = FlowGenModel.from_params(params).to(device)
# initialize
init_batch_size = 512
init_index = np.random.choice(train_index, init_batch_size, replace=False)
init_data, _ = get_batch(train_data, init_index)
init_data = preprocess(init_data, n_bits, False).to(device)
fgen.eval()
fgen.init(init_data, init_scale=1.0)
# create shadow mae for ema
params = json.load(open(args.config, 'r'))
fgen_shadow = FlowGenModel.from_params(params).to(device)
exponentialMovingAverage(fgen, fgen_shadow, polyak_decay, init=True)
print(args)

def get_optimizer(learning_rate, parameters):
    if opt == 'adam':
        return optim.Adam(parameters, lr=learning_rate, betas=betas, eps=eps)
    elif opt == 'adamax':
        return optim.Adamax(parameters, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise ValueError('unknown optimization method: %s' % opt)


opt = args.opt
betas = (0.9, polyak_decay)
eps = 1e-8

if opt == 'adam':
    lr = 1e-3
elif opt == 'adamax':
    lr = 2e-3
else:
    raise ValueError('unknown optimization method: %s' % opt)

optimizer = get_optimizer(lr, fgen.parameters())
step_decay = 0.999995
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)
lr_min = lr / 20

patient = 0


def train(epoch):
    print('Epoch: %d (lr=%.6f (%s), patient=%d' % (epoch, lr, opt, patient))
    fgen.train()
    nll = 0
    num_insts = 0
    num_batches = 0

    num_back = 0
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(iterate_minibatches(train_data, train_index, args.batch_size, True)):
        data = preprocess(data, n_bits, True).to(device)

        batch_size = len(data)
        optimizer.zero_grad()
        log_probs = fgen.log_probability(data)
        loss = log_probs.mean() * -1.0
        loss.backward()
        clip_grad_norm_(fgen.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        exponentialMovingAverage(fgen, fgen_shadow, polyak_decay)

        with torch.no_grad():
            num_insts += batch_size
            num_batches += 1
            nll -= log_probs.sum()

        if batch_idx % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            train_nll = nll / num_insts - np.log(n_bins / 2.) * nx
            bits_per_pixel = train_nll / (nx * np.log(2.0))
            log_info = '[{}/{} ({:.0f}%)] NLL: {:.2f}, BPD: {:.2f}'.format(
                batch_idx * batch_size, len(train_index), 100. * num_insts / len(train_index), train_nll, bits_per_pixel)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    train_nll = nll / num_insts - np.log(n_bins / 2.) * nx
    bits_per_pixel = train_nll / (nx * np.log(2.0))
    print('Average NLL: {:.2f}, BPD: {:.2f}, time: {:.1f}s'.format(train_nll, bits_per_pixel, time.time() - start_time))


def eval(eval_data, eval_index):
    fgen_shadow.eval()
    test_nll = 0
    num_insts = 0
    for i, (data, _) in enumerate(iterate_minibatches(eval_data, eval_index, 100, False)):
        data = preprocess(data, n_bits, False).to(device)

        batch_size = len(data)
        log_probs = fgen.log_probability(data)

        num_insts += batch_size
        test_nll -= log_probs.sum()

    test_nll = test_nll / num_insts - np.log(n_bins / 2.) * nx
    bits_per_pixel = test_nll / (nx * np.log(2.0))

    print('NLL : {:.2f}, BPD: {:.2f}'.format(test_nll, bits_per_pixel))
    return test_nll, bits_per_pixel


best_epoch = 0
best_nll = 1e12
best_bpd = 1e12
for epoch in range(1, args.epochs + 1):
    train(epoch)
    lr = scheduler.get_lr()[0]
    print('----------------------------------------------------------------------------------------------------------------------------')
    with torch.no_grad():
        nll, bits_per_pixel = eval(train_data, val_index)
    if nll < best_nll:
        patient = 0
        torch.save(fgen_shadow.state_dict(), model_name)

        best_epoch = epoch
        best_nll = nll
        best_bpd = bits_per_pixel
    else:
        patient += 1

    print('Best: {:.2f}, BPD: {:.2f}, epoch: {}'.format(best_nll, best_bpd, best_epoch))
    print('============================================================================================================================')

    if lr < lr_min:
        break

fgen_shadow.load_state_dict(torch.load(model_name))
with torch.no_grad():
    print('Final val:')
    eval(train_data, val_index)
    print('Final test:')
    eval(test_data, test_index)
    print('----------------------------------------------------------------------------------------------------------------------------')
