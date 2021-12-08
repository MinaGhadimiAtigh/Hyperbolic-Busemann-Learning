# codebase from hyperspherical prototype networks, Pascal Mettes, NeurIPS2019
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms


################################################################################
# General helpers.
################################################################################

#
# Count the number of learnable parameters in a model.
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#
# Get the desired optimizer.
#
def get_optimizer(optimname, params, learning_rate, momentum, decay):
    if optimname == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    elif optimname == "adadelta":
        optimizer = optim.Adadelta(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "adamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "rmsprop":
        optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay, momentum=momentum)
    elif optimname == "asgd":
        optimizer = optim.ASGD(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "adamax":
        optimizer = optim.Adamax(params, lr=learning_rate, weight_decay=decay)
    else:
        print('Your option for the optimizer is not available, I am loading SGD.')
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)

    return optimizer


################################################################################
# Standard dataset loaders.
################################################################################
def load_dataset(dataset_name, basedir, batch_size, kwargs):
    if dataset_name == 'cifar100':
        return load_cifar100(basedir, batch_size, kwargs)
    elif dataset_name == 'cifar10':
        return load_cifar10(basedir, batch_size, kwargs)
    elif dataset_name == 'cub':
        return load_cub(basedir, batch_size, kwargs)
    else:
        raise Exception('Selected dataset is not available.')


def load_cifar100(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=basedir + 'cifar100/', train=True,
                          transform=transforms.Compose([
                              transforms.RandomCrop(32, 4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize,
                          ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=basedir + 'cifar100/', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize,
                          ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def load_cifar10(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=basedir + 'cifar10/', train=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, 4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=basedir + 'cifar10/', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,
                         ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def load_cub(basedir, batch_size, kwargs):
    # Correct basedir.
    basedir += "cub/"

    # Normalization.
    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Train loader.
    train_data = datasets.ImageFolder(basedir + "train/", transform=transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    # Test loader.
    test_data = datasets.ImageFolder(basedir + "test/", transform=transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize]))

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    return trainloader, testloader
