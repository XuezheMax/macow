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

from macow.data import load_datasets, get_batch, preprocess_full, postprocess
from macow.model import DepthScaleFlowModel
from macow.utils import exponentialMovingAverage, total_grad_norm

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')
parser.add_argument('--batch-steps', type=int, default=1, metavar='N', help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
# parser.add_argument('--depth-batch-steps', type=int, default=1, metavar='N', help='number of steps for each depth scale batch (the batch size of each step is depth-batch-size / steps (default: 1)')
parser.add_argument('--epochs', type=int, default=50000, metavar='N', help='number of epochs to train')
parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N', help='number of epochs to warm up (default: 1)')
parser.add_argument('--valid_epochs', type=int, default=50, metavar='N', help='number of epochs to validate model (default: 50)')
parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=6700417, metavar='S', help='random seed (default: 6700417)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--opt', choices=['adam', 'adamax'], help='optimization method', default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--eta', type=float, default=1.0, help='l2 loss weight')
parser.add_argument('--polyak', type=float, default=0.999, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
parser.add_argument('--model_path', help='path for saving model file.', required=True)
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

dataset = 'cifar10'
imageSize = 32
nc = 3
nx = imageSize * imageSize * nc
n_bins_8bits = 2. ** 8
n_bins_5bits = 2. ** 5
n_bins_3bits = 2. ** 3

model_path = args.model_path
model_name = os.path.join(model_path, 'model.pt')
checkpoint_name = os.path.join(model_path, 'checkpoint.tar')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

train_data, test_data = load_datasets(dataset)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
test_index = np.arange(len(test_data))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False, num_workers=args.workers, pin_memory=True)
batch_steps = args.batch_steps
# depth_batch_steps = args.depth_batch_steps

eta = args.eta
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
    def depthscale_step():
        # img8bits_list = [img8bits,] if depth_batch_steps == 1 else img8bits.chunk(depth_batch_steps, dim=0)
        # img5bits_list = [img5bits,] if depth_batch_steps == 1 else img5bits.chunk(depth_batch_steps, dim=0)
        # img3bits_list = [img3bits,] if depth_batch_steps == 1 else img3bits.chunk(depth_batch_steps, dim=0)
        # l2_loss_3bits = l2_loss_5bits = 0
        # for img8bits, img5bits, img3bits in zip(img8bits_list, img5bits_list, img3bits_list):
        x_8bits, logdet_8bits, internal_5bits = dsgen.depth_downscale_8bits(img8bits)
        loss_5bits = (img5bits - internal_5bits).pow(2).sum()
        loss = loss_5bits.div(batch_size / eta)
        loss.backward()

        x_5bits, logdet_5bits, internal_3bits = dsgen.depth_downscale_5bits(img5bits)
        loss_3bits = (img3bits - internal_3bits).pow(2).sum()
        loss = loss_3bits.div(batch_size / eta)
        loss.backward()

        x_3bits, logdet_3bits = dsgen.depth_downscale_3bits(img3bits)

        with torch.no_grad():
            l2_loss_5bits = loss_5bits.item()
            l2_loss_3bits = loss_3bits.item()
        return (x_8bits, logdet_8bits), (x_5bits, logdet_5bits), (x_3bits, logdet_3bits), (l2_loss_5bits, l2_loss_3bits)

    def flow_step(x, logdet_bottom):
        xs = [x, ] if batch_steps == 1 else x.chunk(batch_steps, dim=0)
        logdet_bottoms = [logdet_bottom, ] if batch_steps == 1 else logdet_bottom.chunk(batch_steps, dim=0)
        nll_batch = 0
        for x, logdet_bottom in zip(xs, logdet_bottoms):
            log_probs = dsgen.log_probability(x, logdet_bottom)
            loss = log_probs.mean() * (-1.0 / batch_steps)
            loss.backward()
            with torch.no_grad():
                nll_batch -= log_probs.sum().item()
        return nll_batch

    print('Epoch: %d (lr=%.6f (%s), patient=%d' % (epoch, lr, opt, patient))
    dsgen.train()
    nll_8bits = 0
    nll_5bits = 0
    nll_3bits = 0
    num_insts = 0
    num_back = 0
    num_nans = 0
    l2_5bits = 0
    l2_3bits = 0
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        img8bits, img5bits, img3bits = preprocess_full(data.to(device, non_blocking=True), True)
        batch_size = len(img3bits)
        # depth scale step
        p_8bits, p_5bits, p_3bits, (l2_loss_5bits, l2_loss_3bits) = depthscale_step()
        l2_5bits += l2_loss_5bits
        l2_3bits += l2_loss_3bits
        # flow step
        nll_batch_8bits = flow_step(*p_8bits)
        nll_batch_5bits = flow_step(*p_5bits)
        nll_batch_3bits = flow_step(*p_3bits)

        if grad_clip > 0:
            grad_norm = clip_grad_norm_(dsgen.parameters(), grad_clip)
        else:
            grad_norm = total_grad_norm(dsgen.parameters())

        if math.isnan(grad_norm):
            num_nans += 1
        else:
            optimizer.step()
            scheduler.step()
            # exponentialMovingAverage(fgen, fgen_shadow, polyak_decay)
            num_insts += batch_size
            nll_8bits += nll_batch_8bits
            nll_5bits += nll_batch_5bits
            nll_3bits += nll_batch_3bits

        if batch_idx % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            train_nll_8bits = nll_8bits / num_insts + np.log(n_bins_8bits / 2.) * nx
            train_nll_5bits = nll_5bits / num_insts + np.log(n_bins_5bits / 2.) * nx
            train_nll_3bits = nll_3bits / num_insts + np.log(n_bins_3bits / 2.) * nx
            bpd_8bits = train_nll_8bits / (nx * np.log(2.0))
            bpd_5bits = train_nll_5bits / (nx * np.log(2.0))
            bpd_3bits = train_nll_3bits / (nx * np.log(2.0))
            train_l2_5bits = l2_5bits / num_insts
            train_l2_3bits = l2_3bits / num_insts
            log_info = '[{}/{} ({:.0f}%) {}] NLL: {:.2f}, {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, {:.4f}, L2: {:.2f}, {:.2f}'.format(
                batch_idx * batch_size, len(train_index), 100. * batch_idx * batch_size / len(train_index), num_nans,
                train_nll_8bits, train_nll_5bits, train_nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, train_l2_5bits, train_l2_3bits)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    train_nll_8bits = nll_8bits / num_insts + np.log(n_bins_8bits / 2.) * nx
    train_nll_5bits = nll_5bits / num_insts + np.log(n_bins_5bits / 2.) * nx
    train_nll_3bits = nll_3bits / num_insts + np.log(n_bins_3bits / 2.) * nx
    bpd_8bits = train_nll_8bits / (nx * np.log(2.0))
    bpd_5bits = train_nll_5bits / (nx * np.log(2.0))
    bpd_3bits = train_nll_3bits / (nx * np.log(2.0))
    train_l2_5bits = l2_5bits / num_insts
    train_l2_3bits = l2_3bits / num_insts
    print('Average NLL: {:.2f}, {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, {:.4f}, L2: {:.2f}, {:.2f}, time: {:.1f}s'.format(
        train_nll_8bits, train_nll_5bits, train_nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, train_l2_5bits, train_l2_3bits, time.time() - start_time))


def eval(data_loader):
    dsgen.eval()
    test_nll_8bits = 0
    test_nll_5bits = 0
    test_nll_3bits = 0
    l2_5bits = 0
    l2_3bits = 0
    num_insts = 0
    for i, (data, _) in enumerate(data_loader):
        img8bits, img5bits, img3bits = preprocess_full(data.to(device, non_blocking=True), True)
        batch_size = len(img3bits)
        num_insts += batch_size

        x, logdet_bottom, x_5bits = dsgen.depth_downscale_8bits(img8bits)
        log_probs_8bits = dsgen.log_probability(x, logdet_bottom)
        test_nll_8bits -= log_probs_8bits.sum().item()

        x, logdet_bottom, x_3bits = dsgen.depth_downscale_5bits(img5bits)
        log_probs_5bits = dsgen.log_probability(x, logdet_bottom)
        test_nll_5bits -= log_probs_5bits.sum().item()

        x, logdet_bottom = dsgen.depth_downscale_3bits(img3bits)
        log_probs_3bits = dsgen.log_probability(x, logdet_bottom)
        test_nll_3bits -= log_probs_3bits.sum().item()

        l2_5bits += (img5bits - x_5bits).pow(2).sum().item()
        l2_3bits += (img3bits - x_3bits).pow(2).sum().item()

    test_nll_8bits = test_nll_8bits / num_insts + np.log(n_bins_8bits / 2.) * nx
    test_nll_5bits = test_nll_5bits / num_insts + np.log(n_bins_5bits / 2.) * nx
    test_nll_3bits = test_nll_3bits / num_insts + np.log(n_bins_3bits / 2.) * nx
    bpd_8bits = test_nll_8bits / (nx * np.log(2.0))
    bpd_5bits = test_nll_5bits / (nx * np.log(2.0))
    bpd_3bits = test_nll_3bits / (nx * np.log(2.0))
    l2_5bits = l2_5bits / num_insts
    l2_3bits = l2_3bits / num_insts

    print('NLL: {:.2f}, {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, {:.4f}, L2: {:.2f}, {:.2f}'.format(
        test_nll_8bits, test_nll_5bits, test_nll_3bits,
        bpd_8bits, bpd_5bits, bpd_3bits,
        l2_5bits, l2_3bits))
    return test_nll_8bits, test_nll_5bits, test_nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, l2_5bits, l2_3bits


def reconstruct(epoch):
    print('reconstruct')
    dsgen.eval()
    n = 128
    np.random.shuffle(test_index)
    img, _ = get_batch(test_data, test_index[:n])
    img8bits, img5bits, img3bits = preprocess_full(img.to(device), False)
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(2)] for i in range(n)])).view(-1)
    # 8 bits
    x, _, _ = dsgen.depth_downscale_8bits(img8bits)
    z, _ = dsgen.encode(x)
    x_recon, _ = dsgen.decode(z)
    img_recon, _ = dsgen.depth_upscale_8bits(x_recon)
    abs_err = img_recon.add(img8bits * -1).abs()
    print('Err (8bits): {:.4f}, {:.4f}'.format(abs_err.max().item(), abs_err.mean().item()))
    img8bits = postprocess(img8bits, 8)
    img_recon = postprocess(img_recon, 8)
    comparison = torch.cat([img8bits, img_recon], dim=0).cpu()
    comparison = comparison[reorder_index]
    image_file = 'reconstruct{}.8bits.png'.format(epoch)
    save_image(comparison, os.path.join(result_path, image_file), nrow=16)

    # 5 bits
    x, _, _ = dsgen.depth_downscale_5bits(img5bits)
    z, _ = dsgen.encode(x)
    x_recon, _ = dsgen.decode(z)
    img_recon, _ = dsgen.depth_upscale_5bits(x_recon)
    abs_err = img_recon.add(img5bits * -1).abs()
    print('Err (5bits): {:.4f}, {:.4f}'.format(abs_err.max().item(), abs_err.mean().item()))
    img5bits = postprocess(img5bits, 5)
    img_recon = postprocess(img_recon, 5)
    comparison = torch.cat([img5bits, img_recon], dim=0).cpu()
    comparison = comparison[reorder_index]
    image_file = 'reconstruct{}.5bits.png'.format(epoch)
    save_image(comparison, os.path.join(result_path, image_file), nrow=16)

    # 8 bits
    x, _ = dsgen.depth_downscale_3bits(img3bits)
    z, _ = dsgen.encode(x)
    x_recon, _ = dsgen.decode(z)
    img_recon, _ = dsgen.depth_upscale_3bits(x_recon)
    abs_err = img_recon.add(img3bits * -1).abs()
    print('Err (3bits): {:.4f}, {:.4f}'.format(abs_err.max().item(), abs_err.mean().item()))
    img3bits = postprocess(img3bits, 3)
    img_recon = postprocess(img_recon, 3)
    comparison = torch.cat([img3bits, img_recon], dim=0).cpu()
    comparison = comparison[reorder_index]
    image_file = 'reconstruct{}.3bits.png'.format(epoch)
    save_image(comparison, os.path.join(result_path, image_file), nrow=16)


def sample(epoch):
    print('sampling')
    dsgen.eval()
    n = 256
    z = torch.randn(n, 3, imageSize, imageSize).to(device)
    x, _ = dsgen.decode(z)
    img8bits, img5bits, img3bits, _ = dsgen.depth_upscale(x)

    img8bits = postprocess(img8bits, 8)
    image_file = 'sample{}.8bits.png'.format(epoch)
    save_image(img8bits, os.path.join(result_path, image_file), nrow=16)
    img5bits = postprocess(img5bits, 5)
    image_file = 'sample{}.5bits.png'.format(epoch)
    save_image(img5bits, os.path.join(result_path, image_file), nrow=16)
    img3bits = postprocess(img3bits, 3)
    image_file = 'sample{}.3bits.png'.format(epoch)
    save_image(img3bits, os.path.join(result_path, image_file), nrow=16)


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
    dsgen = DepthScaleFlowModel.from_params(params).to(device)
    optimizer = get_optimizer(lr, dsgen.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay, last_epoch=-1)

    checkpoint = torch.load(checkpoint_name)
    start_epoch = checkpoint['epoch']
    patient = checkpoint['patient']
    best_epoch = checkpoint['best_epoch']
    best_nll_8bits = checkpoint['best_nll_8bits']
    best_nll_5bits = checkpoint['best_nll_5bits']
    best_nll_3bits = checkpoint['best_nll_3bits']
    best_bpd_8bits = checkpoint['best_bpd_8bits']
    best_bpd_5bits = checkpoint['best_bpd_5bits']
    best_bpd_3bits = checkpoint['best_bpd_3bits']
    best_l2_5bits = checkpoint['best_l2_5bits']
    best_l2_3bits = checkpoint['best_l2_3bits']
    dsgen.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    with torch.no_grad():
        test_itr = 5
        nlls_8bits = []
        nlls_5bits = []
        nlls_3bits = []
        bpds_8bits = []
        bpds_5bits = []
        bpds_3bits = []
        l2s_5bits = []
        l2s_3bits = []
        for _ in range(test_itr):
            nll_8bits, nll_5bits, nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, l2_5bits, l2_3bits = eval(test_loader)
            nlls_8bits.append(nll_8bits)
            nlls_5bits.append(nll_5bits)
            nlls_3bits.append(nll_3bits)
            bpds_8bits.append(bpd_8bits)
            bpds_5bits.append(bpd_5bits)
            bpds_3bits.append(bpd_3bits)
            l2s_5bits.append(l2_5bits)
            l2s_3bits.append(l2_3bits)
        nll_8bits = sum(nlls_8bits) / test_itr
        nll_5bits = sum(nlls_5bits) / test_itr
        nll_3bits = sum(nlls_3bits) / test_itr
        bpd_8bits = sum(bpds_8bits) / test_itr
        bpd_5bits = sum(bpds_5bits) / test_itr
        bpd_3bits = sum(bpds_3bits) / test_itr
        l2_5bits = sum(l2s_5bits) / test_itr
        l2_3bits = sum(l2s_3bits) / test_itr
        print('Avg  {:.2f}, {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, {:.4f}, L2: {:.2f}, {:.2f}'.format(
            nll_8bits, nll_5bits, nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, l2_5bits, l2_3bits))
else:
    params = json.load(open(args.config, 'r'))
    json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
    dsgen = DepthScaleFlowModel.from_params(params).to(device)
    # initialize
    init_batch_size = 2048
    init_index = np.random.choice(train_index, init_batch_size, replace=False)
    init_data, _ = get_batch(train_data, init_index)
    init_data = preprocess_full(init_data.to(device), True)
    dsgen.eval()
    dsgen.init(*init_data, init_scale=1.0)
    # create shadow mae for ema
    # params = json.load(open(args.config, 'r'))
    # fgen_shadow = FlowGenModel.from_params(params).to(device)
    # exponentialMovingAverage(fgen, fgen_shadow, polyak_decay, init=True)

    optimizer = get_optimizer(lr, dsgen.parameters())
    lmbda = lambda step: min(1., step / (len(train_index) * float(warmups) / args.batch_size))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)
    scheduler.step()

    start_epoch = 1
    patient = 0
    best_epoch = 0
    best_nll_8bits = 1e12
    best_nll_5bits = 1e12
    best_nll_3bits = 1e12
    best_bpd_8bits = 1e12
    best_bpd_5bits = 1e12
    best_bpd_3bits = 1e12
    best_l2_5bits = 1e12
    best_l2_3bits = 1e12

# number of parameters
print('# of Parameters: %d' % (sum([param.numel() for param in dsgen.parameters()])))
lr_min = lr / 100
lr = scheduler.get_lr()[0]
checkpoint_epochs = 5
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    print('-' * 100)
    if epoch < 11 or (epoch < 5000 and epoch % 10 == 0) or epoch % args.valid_epochs == 0:
        with torch.no_grad():
            test_itr = 5
            nlls_8bits = []
            nlls_5bits = []
            nlls_3bits = []
            bpds_8bits = []
            bpds_5bits = []
            bpds_3bits = []
            l2s_5bits = []
            l2s_3bits = []
            for _ in range(test_itr):
                nll_8bits, nll_5bits, nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, l2_5bits, l2_3bits = eval(test_loader)
                nlls_8bits.append(nll_8bits)
                nlls_5bits.append(nll_5bits)
                nlls_3bits.append(nll_3bits)
                bpds_8bits.append(bpd_8bits)
                bpds_5bits.append(bpd_5bits)
                bpds_3bits.append(bpd_3bits)
                l2s_5bits.append(l2_5bits)
                l2s_3bits.append(l2_3bits)
            nll_8bits = sum(nlls_8bits) / test_itr
            nll_5bits = sum(nlls_5bits) / test_itr
            nll_3bits = sum(nlls_3bits) / test_itr
            bpd_8bits = sum(bpds_8bits) / test_itr
            bpd_5bits = sum(bpds_5bits) / test_itr
            bpd_3bits = sum(bpds_3bits) / test_itr
            l2_5bits = sum(l2s_5bits) / test_itr
            l2_3bits = sum(l2s_3bits) / test_itr
            print('Avg  {:.2f}, {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, {:.4f}, L2: {:.2f}, {:.2f}'.format(
                nll_8bits, nll_5bits, nll_3bits, bpd_8bits, bpd_5bits, bpd_3bits, l2_5bits, l2_3bits))

        if nll_8bits < best_nll_8bits:
            patient = 0
            torch.save(dsgen.state_dict(), model_name)

            best_epoch = epoch
            best_nll_8bits = nll_8bits
            best_nll_5bits = nll_5bits
            best_nll_3bits = nll_3bits
            best_bpd_8bits = bpd_8bits
            best_bpd_5bits = bpd_5bits
            best_bpd_3bits = bpd_3bits
            best_l2_5bits = l2_5bits
            best_l2_3bits = l2_3bits

            with torch.no_grad():
                reconstruct(epoch)
                sample(epoch)
        else:
            patient += 1

    print('Best NLL: {:.2f}, {:.2f}, {:.2f}, BPD: {:.4f}, {:.4f}, {:.4f}, L2: {:.2f}, {:.2f}, epoch: {}'.format(
        best_nll_8bits, best_nll_5bits, best_nll_3bits, best_bpd_8bits, best_bpd_5bits, best_bpd_3bits, best_l2_5bits, best_l2_3bits, best_epoch))
    print('=' * 100)

    if epoch == warmups:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay, last_epoch=0)

    lr = scheduler.get_lr()[0]

    if epoch % checkpoint_epochs == 0 or epoch >= warmups and patient == 0:
        checkpoint = {'epoch': epoch + 1,
                      'model': dsgen.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'best_epoch': best_epoch,
                      'best_nll_8bits': best_nll_8bits,
                      'best_nll_5bits': best_nll_5bits,
                      'best_nll_3bits': best_nll_3bits,
                      'best_bpd_8bits': best_bpd_8bits,
                      'best_bpd_5bits': best_bpd_5bits,
                      'best_bpd_3bits': best_bpd_3bits,
                      'best_l2_5bits': best_l2_5bits,
                      'best_l2_3bits': best_l2_3bits,
                      'patient': patient}
        torch.save(checkpoint, checkpoint_name)

    if lr < lr_min:
        break

dsgen.load_state_dict(torch.load(model_name))
with torch.no_grad():
    print('Final test:')
    eval(test_loader)
    print('-' * 100)
    reconstruct('final')
    sample('final')
