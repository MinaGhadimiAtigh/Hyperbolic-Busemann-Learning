# codebase from hyperspherical prototype networks, Pascal Mettes, NeurIPS2019
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical/hyperbolic prototypes")
    parser.add_argument('-c', dest="classes", default=100, type=int)
    parser.add_argument('-d', dest="dims", default=100, type=int)
    parser.add_argument('-l', dest="learning_rate", default=0.1, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=1000, type=int, )
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-r', dest="resdir", default="prototypes", type=str)
    parser.add_argument('-w', dest="wtvfile", default="", type=str)
    parser.add_argument('-n', dest="nn", default=2, type=int)
    args = parser.parse_args()
    return args


# in dimension d > 2 prototypes can be placed by minimizing their mutual cosine similarities, 
# as proposed in HPN(NeurIPS2019). Since the peBu-loss is a decreasing function of the cosine similarity
# between directional coordinate and prototype, the arguments outlined in HPN(NeurIPS2019) for
# hyperspherical prototypes equally apply to hyperbolic prototypes.

def prototype_loss(prototype):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototype, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]

    return loss.mean(), product.max()


def prototype_unify(num_classes):
    single_angle = 2 * math.pi / num_classes
    help_list = np.array(range(0, num_classes))
    angles = (help_list * single_angle).reshape(-1, 1)

    sin_points = np.sin(angles)
    cos_points = np.cos(angles)

    set_prototypes = torch.tensor(np.concatenate((cos_points, sin_points), axis=1))
    return set_prototypes


#
# Compute the semantic relation loss.
#
def prototype_loss_sem(prototypes, triplets):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss1 = -product[triplets[:, 0], triplets[:, 1]]
    loss2 = product[triplets[:, 2], triplets[:, 3]]
    # return loss1.mean() + loss2.mean(), product.max()
    return loss1.mean() + loss2.mean()


#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 64, 'pin_memory': True}

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Add word2vec here.
    # Initialize prototypes and optimizer.
    if os.path.exists(args.wtvfile):
        use_wtv = True
        wtvv = np.load(args.wtvfile)
        for i in range(wtvv.shape[0]):
            wtvv[i] /= np.linalg.norm(wtvv[i])
        # wtvv = torch.from_numpy(wtvvectors)
        wtvv = torch.from_numpy(wtvv)

        wtvsim = torch.matmul(wtvv, wtvv.t()).float()

        # Precompute triplets.
        nns, others = [], []
        for i in range(wtvv.shape[0]):
            sorder = np.argsort(wtvsim[i, :]).numpy()[::-1]
            nns.append(sorder[:args.nn])
            others.append(sorder[args.nn:-1])
        triplets = []
        for i in range(wtvv.shape[0]):
            for j in range(len(nns[i])):
                for k in range(len(others[i])):
                    triplets.append([i, j, i, k])
        triplets = np.array(triplets).astype(int)
    else:
        use_wtv = False
        triplets = None
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # First Randomly Initialize Prototypes.
    prototypes = torch.randn(args.classes, args.dims)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    print('initializing finished.')

    optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)

    if args.dims > 2:
        # Optimize for separation.
        for i in range(args.epochs):
            # Compute loss.
            loss1, _ = prototype_loss(prototypes)
            if use_wtv:
                loss2 = prototype_loss_sem(prototypes, triplets)
                loss = loss1 + loss2
            else:
                loss = loss1
            # Update.
            loss.backward()
            optimizer.step()
            # Normalize prototypes again
            prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
            optimizer = optim.SGD([prototypes], lr=args.learning_rate,
                                  momentum=args.momentum)

    elif args.dims == 2:
        prototypes = prototype_unify(args.classes)
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))

    else:
        raise Exception('Dimension is not correct.')

    # Store result.
    if not os.path.exists(args.resdir):
        os.mkdir(args.resdir)
    np.save(os.path.join(args.resdir, "prototypes-%dd-%dc.npy" % (args.dims, args.classes)),
            prototypes.data.numpy())
