import os
import scipy.io
import numpy as np

import torch
from torchvision import datasets, transforms


def load_datasets(dataset, data_path=None):
    if dataset == 'omniglot':
        return load_omniglot()
    elif dataset == 'mnist':
        return load_mnist()
    elif dataset.startswith('lsun64'):
        category = dataset[7:]
        return load_lsun64(data_path, category)
    elif dataset.startswith('lsun128'):
        category = dataset[8:]
        return load_lsun128(data_path, category)
    elif dataset == 'cifar10':
        return load_cifar10()
    elif dataset == 'imagenet':
        return load_imagenet(data_path)
    else:
        raise ValueError('unknown data set %s' % dataset)


def load_omniglot():
    def reshape_data(data):
        return data.T.reshape((-1, 1, 28, 28))

    omni_raw = scipy.io.loadmat('data/omniglot/chardata.mat')

    train_data = reshape_data(omni_raw['data']).astype(np.float32)
    train_label = omni_raw['target'].argmax(axis=0)
    test_data = reshape_data(omni_raw['testdata']).astype(np.float32)
    test_label = omni_raw['testtarget'].argmax(axis=0)

    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).long()

    return [(train_data[i], train_label[i]) for i in range(len(train_data))], \
           [(test_data[i], test_label[i]) for i in range(len(test_data))]


def load_mnist():
    train_data, train_label = torch.load('data/mnist/processed/training.pt')
    test_data, test_label = torch.load('data/mnist/processed/test.pt')

    train_data = train_data.float().div(256).unsqueeze(1)
    test_data = test_data.float().div(256).unsqueeze(1)

    return [(train_data[i], train_label[i]) for i in range(len(train_data))], \
           [(test_data[i], test_label[i]) for i in range(len(test_data))]


def load_lsun64(data_path, category):
    imageSize = 64
    train_data = datasets.LSUN(data_path, classes=[category + '_train'],
                               transform=transforms.Compose([
                                   transforms.Resize(96),
                                   transforms.RandomCrop(imageSize),
                                   transforms.ToTensor(),
                               ]))

    val_data = datasets.LSUN(data_path, classes=[category + '_val'],
                             transform=transforms.Compose([
                                 transforms.Resize(96),
                                 transforms.RandomCrop(imageSize),
                                 transforms.ToTensor(),
                             ]))
    return train_data, val_data


def load_lsun128(data_path, category):
    imageSize = 128
    train_data = datasets.LSUN(data_path, classes=[category + '_train'],
                               transform=transforms.Compose([
                                   transforms.Resize((imageSize, imageSize)),
                                   transforms.ToTensor(),
                               ]))

    val_data = datasets.LSUN(data_path, classes=[category + '_val'],
                             transform=transforms.Compose([
                                 transforms.Resize((imageSize, imageSize)),
                                 transforms.ToTensor(),
                             ]))
    return train_data, val_data


def load_cifar10():
    imageSize = 32
    train_data = datasets.CIFAR10('data/cifar10', train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(4, padding_mode='reflect'),
                                      transforms.RandomCrop(imageSize),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor()
                                  ]))
    test_data = datasets.CIFAR10('data/cifar10', train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    return train_data, test_data


def load_imagenet(data_path):
    train_data = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
    val_data = datasets.ImageFolder(os.path.join(data_path, 'val'),
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
    return train_data, val_data


def get_batch(data, indices):
    imgs = []
    labels = []
    for index in indices:
        img, label = data[index]
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs, dim=0), torch.LongTensor(labels)


def iterate_minibatches(data, indices, batch_size, shuffle):
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield get_batch(data, excerpt)


def binarize_image(img):
    return torch.rand(img.size()).type_as(img).le(img).float()


def binarize_data(data):
    return [(binarize_image(img), label) for img, label in data]


def preprocess(img, n_bits, beta, nsamples=1):
    n_bins = 2. ** n_bits
    # rescale to 255
    img = img.mul(255)
    if n_bits < 8:
        img = torch.floor(img.div(256. / n_bins))

    if beta is not None:
        # add noise (-0.5, 0.5)
        if nsamples == 1:
            u = beta.rsample(img.size()).type_as(img) - 0.5
            img = img + u
        else:
            batch, c, h, w = img.size()
            u = beta.rsample((batch, nsamples, c, h, w)).type_as(img) - 0.5
            img = img.unsqueeze(1) + u
    # normalize
    img = img.div(n_bins)
    img = (img - 0.5).div(0.5)
    return img


def preprocess_full(img, noisy):
    img8bits = preprocess(img, 8, False)
    img5bits = preprocess(img, 5, False)
    img3bits = preprocess(img, 3, False)
    # add noise
    if noisy:
        eps = img.new_empty(img.size()).uniform_(-1., 1.)
        img8bits = img8bits + eps.div(256.)
        img5bits = img5bits + eps.div(32.)
        img3bits = img3bits + eps.div(8.)
    return img8bits, img5bits, img3bits


def postprocess(img, n_bits):
    n_bins = 2. ** n_bits
    # re-normalize
    img = img.mul(0.5) + 0.5
    img = img.mul(n_bins)
    # scale
    img = torch.floor(img) * (256. / n_bins)
    img = img.clamp(0, 255).div(255)
    return img
