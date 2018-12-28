import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import argparse
import random
import math
import numpy as np

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_

from macow.data import load_datasets, get_batch, preprocess, postprocess
from macow.model import FlowGenModel
from macow.utils import exponentialMovingAverage, total_grad_norm, logsumexp

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')
parser.add_argument('--batch-steps', type=int, default=1, metavar='N', help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
parser.add_argument('--image-size', type=int, default=64, metavar='N', help='input image size(default: 64)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', type=int, default=50000, metavar='N', help='number of epochs to train')
parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N', help='number of epochs to warm up (default: 1)')
parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 524287)')
parser.add_argument('--n_bits', type=int, default=8, metavar='N', help='number of bits per pixel.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--opt', choices=['adam', 'adamax'], help='optimization method', default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--polyak', type=float, default=0.999, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
parser.add_argument('--model_path', help='path for saving model file.', required=True)
parser.add_argument('--data_path', help='path for data file.', default=None)
parser.add_argument('--recover', action='store_true', help='recover the model from disk.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.backends.cudnn.benchmark = True

imageSize = args.image_size
assert imageSize in [32, 64]
dataset = 'imagenet'
nc = 3
nx = imageSize * imageSize * nc
n_bits = args.n_bits
n_bins = 2. ** n_bits
test_k = 5

model_path = args.model_path
model_name = os.path.join(model_path, 'model.pt')
checkpoint_name = os.path.join(model_path, 'checkpoint.tar')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

data_path = os.path.join(args.data_path, 'imagenet{}x{}'.format(imageSize, imageSize))
train_data, test_data = load_datasets(dataset, data_path)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
test_index = np.arange(len(test_data))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False, num_workers=args.workers, pin_memory=True)
batch_steps = args.batch_steps

print(len(train_index))
print(len(test_index))


def get_optimizer(learning_rate, parameters):
    if opt == 'adam':
        return optim.Adam(parameters, lr=learning_rate, betas=betas, eps=eps)
    elif opt == 'adamax':
        return optim.Adamax(parameters, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise ValueError('unknown optimization method: %s' % opt)


def train(epoch):
    print('Epoch: %d (lr=%.6f (%s), patient=%d' % (epoch, lr, opt, patient))
    fgen.train()
    nll = 0
    num_insts = 0
    num_back = 0
    num_nans = 0
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = preprocess(data.to(device, non_blocking=True), n_bits, True)
        batch_size = len(data)
        optimizer.zero_grad()
        nll_batch = 0
        data_list = [data, ] if batch_steps == 1 else data.chunk(batch_steps, dim=0)
        for data in data_list:
            log_probs = fgen.log_probability(data)
            loss = log_probs.mean() * (-1.0 / batch_steps)
            loss.backward()
            with torch.no_grad():
                nll_batch -= log_probs.sum().item()

        if grad_clip > 0:
            grad_norm = clip_grad_norm_(fgen.parameters(), grad_clip)
        else:
            grad_norm = total_grad_norm(fgen.parameters())

        if math.isnan(grad_norm):
            num_nans += 1
        else:
            optimizer.step()
            scheduler.step()
            # exponentialMovingAverage(fgen, fgen_shadow, polyak_decay)
            num_insts += batch_size
            nll += nll_batch

        if batch_idx % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            train_nll = nll / num_insts + np.log(n_bins / 2.) * nx
            bits_per_pixel = train_nll / (nx * np.log(2.0))
            log_info = '[{}/{} ({:.0f}%) {}] NLL: {:.2f}, BPD: {:.4f}'.format(
                batch_idx * batch_size, len(train_index), 100. * batch_idx * batch_size / len(train_index), num_nans,
                train_nll, bits_per_pixel)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    train_nll = nll / num_insts + np.log(n_bins / 2.) * nx
    bits_per_pixel = train_nll / (nx * np.log(2.0))
    print('Average NLL: {:.2f}, BPD: {:.4f}, time: {:.1f}s'.format(train_nll, bits_per_pixel, time.time() - start_time))


def eval(data_loader, k):
    fgen.eval()
    nll_mc = 0
    nll_iw = 0
    num_insts = 0
    for i, (data, _) in enumerate(data_loader):
        # [batch, channels, H, W]
        data = preprocess(data.to(device, non_blocking=True), n_bits, False)
        batch, c, h, w = data.size()
        # [batch, k, channels, H, W]
        data = data.unsqueeze(1) + data.new_empty(batch, k, c, h, w).uniform_(-1. / n_bins, 1. / n_bins)

        # [batch * k, channels, H, W] -> [batch * k] -> [batch, k]
        log_probs = fgen.log_probability(data.view(batch * k, c, h, w)).view(batch, k)

        num_insts += batch
        nll_mc -= log_probs.mean(dim=1).sum().item()
        nll_iw -= (logsumexp(log_probs, dim=1) - math.log(k)).sum().item()

    nll_mc = nll_mc / num_insts + np.log(n_bins / 2.) * nx
    bpd_mc = nll_mc / (nx * np.log(2.0))
    nll_iw = nll_iw / num_insts + np.log(n_bins / 2.) * nx
    bpd_iw = nll_iw / (nx * np.log(2.0))

    print('Avg  NLL: {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}'.format(nll_mc, nll_iw, bpd_mc, bpd_iw))
    return nll_mc, nll_iw, bpd_mc, bpd_iw


def reconstruct(epoch):
    print('reconstruct')
    fgen.eval()
    n = 64
    np.random.shuffle(test_index)
    img, _ = get_batch(test_data, test_index[:n])
    img = preprocess(img.to(device), n_bits, False)

    z, _ = fgen.encode(img)
    img_recon, _ = fgen.decode(z)

    abs_err = img_recon.add(img * -1).abs()
    print('Err: {:.4f}, {:.4f}'.format(abs_err.max().item(), abs_err.mean().item()))

    img = postprocess(img, n_bits)
    img_recon = postprocess(img_recon, n_bits)
    comparison = torch.cat([img, img_recon], dim=0).cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(2)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct{}.png'.format(epoch)
    save_image(comparison, os.path.join(result_path, image_file), nrow=16)


def sample(epoch):
    print('sampling')
    fgen.eval()
    n = 256
    z = torch.randn(n, 3, imageSize, imageSize).to(device)
    img, _ = fgen.decode(z)
    img = postprocess(img, n_bits)
    image_file = 'sample{}.png'.format(epoch)
    save_image(img, os.path.join(result_path, image_file), nrow=16)


print(args)
polyak_decay = args.polyak
opt = args.opt
betas = (0.9, polyak_decay)
eps = 1e-8
lr = args.lr
warmups = args.warmup_epochs
step_decay = 0.999998
grad_clip = args.grad_clip

if args.recover:
    params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    fgen = FlowGenModel.from_params(params).to(device)
    optimizer = get_optimizer(lr, fgen.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay, last_epoch=-1)

    checkpoint = torch.load(checkpoint_name)
    start_epoch = checkpoint['epoch']
    patient = checkpoint['patient']
    best_epoch = checkpoint['best_epoch']
    best_nll_mc = checkpoint['best_nll']
    best_bpd_mc = checkpoint['best_bpd']
    fgen.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    with torch.no_grad():
        _, best_nll_iw, _, best_bpd_iw = eval(test_loader, test_k)
else:
    params = json.load(open(args.config, 'r'))
    json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
    fgen = FlowGenModel.from_params(params).to(device)
    # initialize
    fgen.eval()
    init_batch_size = 2048 if imageSize == 32 else 1024
    init_iter = 1
    print('init: {} instances with {} iterations'.format(init_batch_size, init_iter))
    for _ in range(init_iter):
        init_index = np.random.choice(train_index, init_batch_size, replace=False)
        init_data, _ = get_batch(train_data, init_index)
        init_data = preprocess(init_data, n_bits, True).to(device)
        fgen.init(init_data, init_scale=1.0)
    # create shadow mae for ema
    # params = json.load(open(args.config, 'r'))
    # fgen_shadow = FlowGenModel.from_params(params).to(device)
    # exponentialMovingAverage(fgen, fgen_shadow, polyak_decay, init=True)

    optimizer = get_optimizer(lr, fgen.parameters())
    lmbda = lambda step: min(1., step / (len(train_index) * float(warmups) / args.batch_size))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)
    scheduler.step()

    start_epoch = 1
    patient = 0
    best_epoch = 0
    best_nll_mc = 1e12
    best_bpd_mc = 1e12
    best_nll_iw = 1e12
    best_bpd_iw = 1e12

# number of parameters
print('# of Parameters: %d' % (sum([param.numel() for param in fgen.parameters()])))
lr_min = lr / 100
lr = scheduler.get_lr()[0]
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    print('-' * 50)
    with torch.no_grad():
        nll_mc, nll_iw, bpd_mc, bpd_iw = eval(test_loader, test_k)

    if nll_iw < best_nll_iw:
        patient = 0
        torch.save(fgen.state_dict(), model_name)

        best_epoch = epoch
        best_nll_mc = nll_mc
        best_nll_iw = nll_iw
        best_bpd_mc = bpd_mc
        best_bpd_iw = bpd_iw

        with torch.no_grad():
            reconstruct(epoch)
            sample(epoch)
    else:
        patient += 1

    print('Best NLL: {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, epoch: {}'.format(best_nll_mc, best_nll_iw, best_bpd_mc, best_bpd_iw, best_epoch))
    print('=' * 50)

    if epoch == warmups:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay, last_epoch=0)

    lr = scheduler.get_lr()[0]

    if epoch >= warmups:
        checkpoint = {'epoch': epoch + 1,
                      'model': fgen.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'best_epoch': best_epoch,
                      'best_nll_mc': best_nll_mc,
                      'best_bpd_mc': best_bpd_mc,
                      'best_nll_iw': best_nll_iw,
                      'best_bpd_iw': best_bpd_iw,
                      'patient': patient}
        torch.save(checkpoint, checkpoint_name)

    if lr < lr_min:
        break

fgen.load_state_dict(torch.load(model_name))
final_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
with torch.no_grad():
    print('Final test:')
    eval(final_loader, 512)
    print('-' * 50)
    reconstruct('final')
    sample('final')
