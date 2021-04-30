import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms
from PIL import Image, ImageShow
from itertools import chain
from matplotlib import pyplot as plt
import os
import math
import argparse
from PIL import Image, ImageDraw
import numpy as np

class ReConvNet(nn.Module):
    def __init__(self, in_dim, hid_dim, in_kernel, hid_kernel, parse=False):
        super().__init__()
        #self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        #self.outconv = nn.Conv2d(hid_dim, in_dim, kernel_size, 1, 0)
        #self.reconv = nn.Conv2d(hid_dim, hid_dim, hid_kernel, 1, hid_kernel//2)
        self.reconv = nn.Conv2d(hid_dim, hid_dim, hid_kernel, 1, 0)

        self.parse = parse
        if parse:
            parse_depth = 2
            n_parse_feature_per_spatial_dim = 2
            self.parse_depth = parse_depth
            parse_in_dim = hid_dim + hid_dim*parse_depth*n_parse_feature_per_spatial_dim**2
            print("parse in dim", parse_in_dim)
            self.parser = nn.Sequential(
                nn.Linear(parse_in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
                #nn.Softmax()
            )
            self.adapool = nn.AdaptiveMaxPool2d(n_parse_feature_per_spatial_dim)

        self.pool = nn.MaxPool2d(2)
        self.hid_kernel = hid_kernel
        self.pad = hid_kernel//2
        self.acti = nn.LeakyReLU()


    def forward(self, X):
        i = 0
        #print(i, "raw", X.shape)
        X = self.inconv(X)
        #print(i, "init", X.shape)
        X = self.acti(X)
        #X = self.pool(X)
        #print(i, "initpool", X.shape)
        i += 1
        #for i in range(int(math.log2(X.shape[2]))):
        acti_list = []
        while X.shape[-1]>self.hid_kernel:
            X = F.pad(X, (self.pad, self.pad, self.pad, self.pad))
            #print(i, "midpad", X.shape)
            X = self.reconv(X)
            #print(i, "midconv", X.shape)
            X = self.acti(X)
            X = self.pool(X)
            acti_list.append(X)
            #print(i, "midpool", X.shape)
            i += 1

        #print(i, "prefinalpad", X.shape)
        pad_a = math.ceil((self.hid_kernel - X.shape[-1])/2)
        pad_b = math.floor((self.hid_kernel - X.shape[-1])/2)+1
        thresh = 0.5
        if pad_a!=pad_b:
            if T.rand(1)>thresh:
                if T.rand(1)>thresh:
                    X = F.pad(X, (pad_b, pad_a, pad_b, pad_a))
                else:
                    X = F.pad(X, (pad_b, pad_a, pad_a, pad_b))
            else:
                if T.rand(1)>thresh:
                    X = F.pad(X, (pad_a, pad_b, pad_b, pad_a))
                else:
                    X = F.pad(X, (pad_a, pad_b, pad_a, pad_b))
        else:
            X = F.pad(X, (pad_a, pad_b, pad_a, pad_b))
                
        #print(i, "finalpad", X.shape)
        X = self.reconv(X)
        #print(i, "finalconv", X.shape)
        X = self.acti(X)
        X = self.pool(X)
        #print(i, "finalpooled", X.shape)

        if self.parse:
            actis = [X.squeeze()]
            #print(actis[0].shape)
            acti_list = acti_list[-self.parse_depth:][::-1]
            for acti in acti_list:
                #print("before", acti.shape)
                acti = self.adapool(acti).flatten(1)
                #print("after", acti.shape)
                actis.append(acti)
            actis = T.cat(actis, dim=1)
            #print("actual parse in dim", actis.shape)
            pred = self.parser(actis)
            
            return pred, X

        else:
            return X.squeeze()[:,:10], X


def eval_model(model, loader):
    hits = []
    for batch_n, (X,Y) in enumerate(loader):
        with T.no_grad():
            X = X.to(device)
            Y = Y.to(device)
            pred, _ = model(X)

        pred = pred.cpu()
        Y = Y.cpu()
        #X = X.cpu()
        hits.append(T.argmax(pred.detach(), dim=-1)==Y)
        if batch_n != 10:
            acc = hits[-1].sum()/len(hits[-1])
            print("running eval:", batch_n, acc.numpy().round(3), end="\r")
    
    print()
    hits = T.cat(hits, dim=0)
    acc = hits.sum()/hits.shape[0]
    print("EVAL FINISHED - accuracy:", acc.item())


def train_model(model, train_loader, test_loader=None):
    # LOADING MODEL
    if args.load:
        load_model(model, args.model_name)

    # SETUP OPTIMIZER
    opti = T.optim.Adam(chain(model.parameters()), lr=1e-3)
    #sched = T.optim.lr_scheduler.CosineAnnealingLR

    # TRAINING
    for epoch in range(50):
        for batch_n, (X,Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            pred, top_layer = model(X)

            #print("pred", pred.shape)
            #print(pred.requires_grad)

            loss = F.cross_entropy(pred, Y)

            opti.zero_grad()
            loss.backward()
            opti.step()

            if not batch_n%100:
                pred = pred.cpu()
                Y = Y.cpu()
                #X = X.cpu()
                acc = (T.argmax(pred.detach(), dim=-1)==Y).sum()/Y.shape[0]
                print(epoch, batch_n, round(loss.item(), 3), acc.numpy().round(3), 
                    T.argmax(pred.detach().flatten(1)[:,:10], dim=-1)[:5])

            if not batch_n and not epoch:
                #maps = T.stack((maps, maps, maps), dim=-1)
                X = X.cpu()
                pred = pred.cpu()
                X = X.permute(0,2,3,1)
                if args.random:
                    X = X*T.tensor((0.229, 0.224, 0.225))+T.tensor((0.485, 0.456, 0.406))
                together = T.cat((X,), dim=1)
                pics = T.cat(list(together), dim=1)
                print("saving results for"+f" {epoch}_{batch_n}", pics.shape)
                #print(maps.shape, pics.shape, together.shape, pics.shape)
                plt.imsave(save_path+f"{epoch}-{batch_n}.png", pics.squeeze().numpy())
        
        # EVAL MODEL
        if test_loader is not None:
            eval_model(model, test_loader)

        # SAVING MODEL
        save_model(model, args.model_name)


def load_model(model, name):
    print("loading model", name)
    model.load_state_dict(T.load("saves/"+name+".pt"))
    return model


def save_model(model, name):
    print("saving model", name)
    T.save(model.state_dict(), "saves/"+name+".pt")


def incept_model(model, name):
    model = load_model(model, name)
    batchsize = 10
    path = f"results/incept/{name}-3/"
    os.makedirs(path, exist_ok=True)

    for label_idx in range(args.dims):
        X = T.rand(batchsize, 1, 28, 28, requires_grad=True)
        opti = T.optim.SGD([X], lr=1, weight_decay=0.005)
        results = []

        for iteration_idx in range(1000):
            pred, acti = model(X)
            acti = acti.squeeze()
            
            loss = F.cross_entropy(acti, T.ones(batchsize, dtype=T.long)*label_idx)


            opti.zero_grad()
            loss.backward()
            opti.step()

            X.data[X.data>1] = 1

            if not iteration_idx%100:
                print(label_idx, iteration_idx, "grad sum:", X.grad.abs().sum())
                results.append(np.concatenate(F.pad(X.detach(), (2,2,2,2)).numpy(), axis=2).squeeze())

        plt.imsave(path+f"index-{label_idx}.png", np.concatenate(results, axis=0))
        #plt.show()


if __name__ == "__main__":
    save_path = f"results/workspace/"
    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if T.cuda.is_available() else "cpu"
    print("device:", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("-mnist", action="store_true")
    parser.add_argument("-parse", action="store_true")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-incept", action="store_true")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("--random", type=float, default=0)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--model-name", type=str, default="model")
    args = parser.parse_args()
    print(args)

    v = 0
    a = 0.95


    # SETUP MODEL AND OPTI
    model = ReConvNet(1 if args.mnist else 3,
        args.dims, 
        args.ks, 
        args.ks, 
        parse=args.parse
    ).to(device)


    # SETUP DATA TRANSFORMS
    if args.random:
        r = args.random
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            #transforms.RandomApply([
            #    transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            #], p=0.2),
            transforms.RandomApply([
                transforms.Grayscale(num_output_channels=3)
            ], p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=r, contrast=r, saturation=r, hue=r)
            ]),  
            transforms.RandomApply([
                transforms.RandomAffine(r*10, shear=r*10)
            ]),
            transforms.RandomResizedCrop((32,32), scale=(1-r, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        train_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()


    # LOADING DATA
    if args.mnist:
        train_loader = T.utils.data.DataLoader(MNIST("data/", download=True, train=True, 
           transform=train_transforms), batch_size=64)    
        test_loader = T.utils.data.DataLoader(MNIST("data/", download=True, train=False, 
           transform=test_transforms), batch_size=64)    
    else:
        train_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=True, 
            transform=train_transforms), batch_size=64)
        test_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=False, 
            transform=test_transforms), batch_size=64)
    #print(len(train_loader), len(test_loader))

    if args.train:
        train_model(model, train_loader, test_loader=test_loader)

    if args.incept:
        incept_model(model, args.model_name)
    