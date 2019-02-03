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
from macow.models import FlowGenModel, VDeQuantFlowGenModel
from macow.utils import exponentialMovingAverage, total_grad_norm

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--batch-size', type=int, default=40, metavar='N', help='input batch size for training (default: 512)')
parser.add_argument('--batch-steps', type=int, default=1, metavar='N', help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
parser.add_argument('--image-size', type=int, default=256, metavar='N', help='input image size(default: 64)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N', help='number of epochs to train')
parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N', help='number of epochs to warm up (default: 1)')
parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 524287)')
parser.add_argument('--n_bits', type=int, default=8, metavar='N', help='number of bits per pixel.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--opt', choices=['adam', 'adamax'], help='optimization method', default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--polyak', type=float, default=0.999, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
parser.add_argument('--dequant', choices=['uniform', 'variational'], help='dequantization method', default='uniform')
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
assert imageSize <= 1024
dataset = 'celeba' + str(imageSize)
nc = 3
nx = imageSize * imageSize * nc
n_bits = args.n_bits
n_bins = 2. ** n_bits
test_k = 4

model_path = args.model_path
model_name = os.path.join(model_path, 'model.pt')
checkpoint_name = os.path.join(model_path, 'checkpoint.tar')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

train_data, test_data = load_datasets(dataset, args.data_path)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
test_index = np.arange(len(test_data))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=25, shuffle=False, num_workers=args.workers, pin_memory=True)
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
    print('Epoch: %d (lr=%.6f (%s), patient=%d)' % (epoch, lr, opt, patient))
    fgen.train()
    nll = 0
    nent = 0
    num_insts = 0
    num_back = 0
    num_nans = 0
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        batch_size = len(data)
        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        nll_batch = 0
        nent_batch = 0
        data_list = [data, ] if batch_steps == 1 else data.chunk(batch_steps, dim=0)
        for data in data_list:
            x = preprocess(data, n_bits)
            # [batch, 1]
            noise, log_probs_posterior = fgen.dequantize(x)
            # [batch] -> [1]
            log_probs_posterior = log_probs_posterior.mean(dim=1).sum()
            data = preprocess(data, n_bits, noise).squeeze(1)
            log_probs = fgen.log_probability(data).sum()
            loss = (log_probs_posterior - log_probs) / batch_size
            loss.backward()
            with torch.no_grad():
                nll_batch -= log_probs.item()
                nent_batch += log_probs_posterior.item()

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
            nent += nent_batch

        if batch_idx % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            train_nent = nent / num_insts
            train_nll = nll / num_insts + train_nent + np.log(n_bins / 2.) * nx
            bits_per_pixel = train_nll / (nx * np.log(2.0))
            nent_per_pixel = train_nent / (nx * np.log(2.0))
            log_info = '[{}/{} ({:.0f}%) {}] NLL: {:.2f}, BPD: {:.4f}, NENT: {:.2f}, NEPD: {:.4f}'.format(
                batch_idx * batch_size, len(train_index), 100. * batch_idx * batch_size / len(train_index), num_nans,
                train_nll, bits_per_pixel, train_nent, nent_per_pixel)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    train_nent = nent / num_insts
    train_nll = nll / num_insts + train_nent + np.log(n_bins / 2.) * nx
    bits_per_pixel = train_nll / (nx * np.log(2.0))
    nent_per_pixel = train_nent / (nx * np.log(2.0))
    print('Average NLL: {:.2f}, BPD: {:.4f}, NENT: {:.2f}, NEPD: {:.4f}, time: {:.1f}s'.format(
        train_nll, bits_per_pixel, train_nent, nent_per_pixel, time.time() - start_time))


def eval(data_loader, k):
    fgen.eval()
    nent = 0
    nll_mc = 0
    nll_iw = 0
    num_insts = 0
    for i, (data, _) in enumerate(data_loader):
        data = data.to(device, non_blocking=True)
        # [batch, channels, H, W]
        batch, c, h, w = data.size()
        x = preprocess(data, n_bits)
        # [batch, k]
        noise, log_probs_posterior = fgen.dequantize(x, nsamples=k)
        # [batch, k, channels, H, W]
        data = preprocess(data, n_bits, noise)
        # [batch * k, channels, H, W] -> [batch * k] -> [batch, k]
        log_probs = fgen.log_probability(data.view(batch * k, c, h, w)).view(batch, k)
        # [batch, k]
        log_iw = log_probs - log_probs_posterior

        num_insts += batch
        nent += log_probs_posterior.mean(dim=1).sum().item()
        nll_mc -= log_iw.mean(dim=1).sum().item()
        nll_iw += (math.log(k) - torch.logsumexp(log_iw, dim=1)).sum().item()

    nent = nent / num_insts
    nepd = nent / (nx * np.log(2.0))
    nll_mc = nll_mc / num_insts + np.log(n_bins / 2.) * nx
    bpd_mc = nll_mc / (nx * np.log(2.0))
    nll_iw = nll_iw / num_insts + np.log(n_bins / 2.) * nx
    bpd_iw = nll_iw / (nx * np.log(2.0))

    print('Avg  NLL: {:.2f}, NENT: {:.2f}, IW: {:.2f}, BPD: {:.4f}, NEPD: {:.4f}, BPD_IW: {:.4f}'.format(
        nll_mc, nent, nll_iw, bpd_mc, nepd, bpd_iw))
    return nll_mc, nent, nll_iw, bpd_mc, nepd, bpd_iw


def reconstruct(epoch):
    print('reconstruct')
    fgen.eval()
    n = 16
    np.random.shuffle(test_index)
    img, _ = get_batch(test_data, test_index[:n])
    img = preprocess(img.to(device), n_bits)

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
    save_image(comparison, os.path.join(result_path, image_file), nrow=4)


def sample(epoch):
    print('sampling')
    fgen.eval()
    n = 64
    taus = [0.7, 0.8, 0.9, 1.0]
    for t in taus:
        z = torch.randn(n, 3, imageSize, imageSize).to(device)
        z = z * t
        img, _ = fgen.decode(z)
        img = postprocess(img, n_bits)
        image_file = 'sample{}.t{:.1f}.png'.format(epoch, t)
        save_image(img, os.path.join(result_path, image_file), nrow=8)


print(args)
polyak_decay = args.polyak
opt = args.opt
betas = (0.9, polyak_decay)
eps = 1e-8
lr = args.lr
warmups = args.warmup_epochs
step_decay = 0.999998
grad_clip = args.grad_clip
dequant = args.dequant

if args.recover:
    params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    if dequant == 'uniform':
        fgen = FlowGenModel.from_params(params).to_device(device)
    elif dequant == 'variational':
        fgen = VDeQuantFlowGenModel.from_params(params).to_device(device)
    else:
        raise ValueError('unknown dequantization method: %s' % dequant)

    optimizer = get_optimizer(lr, fgen.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay, last_epoch=-1)

    checkpoint = torch.load(checkpoint_name)
    start_epoch = checkpoint['epoch']
    patient = checkpoint['patient']
    best_epoch = checkpoint['best_epoch']
    best_nll_mc = checkpoint['best_nll_mc']
    best_bpd_mc = checkpoint['best_bpd_mc']
    best_nll_iw = checkpoint['best_nll_iw']
    best_bpd_iw = checkpoint['best_bpd_iw']
    best_nent = checkpoint['best_nent']
    best_nepd = checkpoint['best_nepd']
    fgen.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    del checkpoint

    with torch.no_grad():
        eval(test_loader, test_k)
else:
    params = json.load(open(args.config, 'r'))
    json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
    if dequant == 'uniform':
        fgen = FlowGenModel.from_params(params).to(device)
    elif dequant == 'variational':
        fgen = VDeQuantFlowGenModel.from_params(params).to(device)
    else:
        raise ValueError('unknown dequantization method: %s' % dequant)
    # initialize
    fgen.eval()
    init_batch_size = 32
    init_iter = 1
    print('init: {} instances with {} iterations'.format(init_batch_size, init_iter))
    for _ in range(init_iter):
        init_index = np.random.choice(train_index, init_batch_size, replace=False)
        init_data, _ = get_batch(train_data, init_index)
        init_data = preprocess(init_data.to(device), n_bits)
        fgen.init(init_data, init_scale=1.0)
    # create shadow mae for ema
    # params = json.load(open(args.config, 'r'))
    # fgen_shadow = FlowGenModel.from_params(params).to(device)
    # exponentialMovingAverage(fgen, fgen_shadow, polyak_decay, init=True)

    fgen.to_device(device)
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
    best_nent = 1e12
    best_nepd = 1e12

# number of parameters
print('# of Parameters: %d' % (sum([param.numel() for param in fgen.parameters()])))
lr_min = lr / 100
lr = scheduler.get_lr()[0]
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    print('-' * 100)
    with torch.no_grad():
        nll_mc, nent, nll_iw, bpd_mc, nepd, bpd_iw = eval(test_loader, test_k)

    if nll_mc < best_nll_mc:
        patient = 0
        torch.save(fgen.state_dict(), model_name)

        best_epoch = epoch
        best_nll_mc = nll_mc
        best_nll_iw = nll_iw
        best_bpd_mc = bpd_mc
        best_bpd_iw = bpd_iw
        best_nent = nent
        best_nepd = nepd

        with torch.no_grad():
            reconstruct(epoch)
            sample(epoch)
    else:
        patient += 1

    print('Best NLL: {:.2f}, NENT: {:.2f}, IW: {:.2f}, BPD: {:.4f}, NEPD: {:.4f}, BPD_IW: {:.4f}, epoch: {}'.format(
        best_nll_mc, best_nent, best_nll_iw, best_bpd_mc, best_nepd, best_bpd_iw, best_epoch))
    print('=' * 100)

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
                      'best_nent': best_nent,
                      'best_nepd': best_nepd,
                      'patient': patient}
        torch.save(checkpoint, checkpoint_name)

    if lr < lr_min:
        break

fgen.load_state_dict(torch.load(model_name))
final_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
with torch.no_grad():
    print('Final test:')
    eval(final_loader, 512)
    print('-' * 100)
    reconstruct('final')
    sample('final')
