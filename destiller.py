import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
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
import copy
from minerldata import get_minerl_dataset


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


class Destiller(nn.Module):
    def __init__(self, in_dim, hid_dim, in_kernel, hid_kernel, parse=False, radius=3):
        super().__init__()
        self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        self.outconv = nn.Conv2d(hid_dim, in_dim, in_kernel, 1, in_kernel//2)
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
        if radius == 5:
            sobelx = T.FloatTensor([[1,  2, 0,  -2, -1],
                                        [4,  8, 0,  -8, -4],
                                        [6, 12, 0, -12, -6],
                                        [4,  8, 0,  -8, -4],
                                        [1,  2, 0,  -2, -1]]) * -1
        if radius == 3:
            sobelx = T.FloatTensor([
                [1,0,-1],
                [2,0,-2],
                [1,0,-1]
            ])
        sobelx = sobelx.float()
        sobely = sobelx.permute(1,0)
        self.sobelx_kernel = 0+sobelx.expand(hid_dim, hid_dim, radius, radius)
        self.sobely_kernel = 0+sobely.expand(hid_dim, hid_dim, radius, radius)

        update_conv_func = lambda: nn.Sequential(
            nn.Conv2d(hid_dim*4, hid_dim, 1, 1, 0),
            nn.Tanh()
        )

        self.concentrate = update_conv_func()

        self.relax = update_conv_func()

        self.pool = nn.MaxPool2d(2)
        self.hid_kernel = hid_kernel
        self.pad = hid_kernel//2
        self.acti = nn.LeakyReLU()


    def excite(self, X):
        i = 0
        #print(i, "raw", X.shape)
        X = self.inconv(X)
        #print(i, "init", X.shape)
        X = self.acti(X)
        #X = self.pool(X)
        #print(i, "initpool", X.shape)
        i += 1
        #for i in range(int(math.log2(X.shape[2]))):
        acti_list = [X]
        while X.shape[-1]>self.hid_kernel:
            X = F.pad(X, (self.pad, self.pad, self.pad, self.pad))
            #print(i, "midpad", X.shape)
            X = self.reconv(X)
            #print(i, "midconv", X.shape)
            X = self.acti(X)
            X = self.pool(X)
            acti_list.append(X)
            #print(i, "midpool", X.shape)
            #print("excite", X)
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
        acti_list.append(X)
        #print(i, "finalpooled", X.shape)
        return acti_list


    def parse(self, X):
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


    def iterate(self, actis, n_iter=5, destillate=True):
        n = len(actis)
        up_mode = "nearest"
        down_mode = "nearest"
        actis = [e.clone() for e in actis]
        #acti_backup = copy.deepcopy([e.detach() for e in actis])
        #print("init", [e.sum() for e in actis])

        for iter_idx in range(n_iter):
            top_down = []
            bottom_up = []
            xgrad = []
            ygrad = []
            
            for i, X in enumerate(actis):
                bottom_up.append(T.zeros_like(X) if not i else F.interpolate(actis[i-1], X.shape[-2:], mode=up_mode))
                top_down.append(T.zeros_like(X) if i==n-1 else F.interpolate(actis[i+1], X.shape[-2:], mode=down_mode))
                #print(T.ones_like(self.sobelx_kernel).type())
                xgrad.append(F.conv2d(X, self.sobelx_kernel, stride=1, padding=1))
                ygrad.append(F.conv2d(X, self.sobely_kernel, stride=1, padding=1))
            
            for i in range(n):
                influence = (top_down[i], bottom_up[i], xgrad[i], ygrad[i])
                influence = T.cat(influence, dim=1)
                if destillate:
                    change = self.concentrate(influence)
                    #print("change", change.sum())
                    actis[i] = actis[i] + change
                    #print("destillation...")
                else:
                    #old_acti = actis[i]
                    actis[i] = actis[i] + self.relax(influence)
                    #print("local acti diff", (old_acti-actis[i]).abs().mean().item())
                #print("post", i, actis[i].sum())

        # INJECT OPTIONS
        if destillate:
            if args.max_inject:
                for i in range(n):
                    actis[i] = actis[i] * ((actis[i].max(dim=1).values.unsqueeze(1)==actis[i]) if i else actis[i]*0)
                    
            if args.softmax:
                for i in range(n):
                    actis[i] = T.softmax(actis[i], dim=1)
        
        #print("all acti diffs:", [(actis[i]-acti_backup[i]).abs().mean().item() for i in range(len(actis))])
        return actis


    def forward(self, X):
        excitation = self.excite(X)
        destillation = self.iterate(excitation, n_iter=5, destillate=True)
        #print("destill diff:", [(destillation[i]-excitation[i]).abs().mean().item() for i in range(len(excitation))])
        #activations = T.cat([layer.flatten(0) for layer in destillation], dim=0)
        relaxation = self.iterate(destillation, n_iter=5, destillate=False)
        #print("relax diff:", [(destillation[i]-relaxation[i]).abs().mean().item() for i in range(len(excitation))])
        reconstruction = T.sigmoid(self.outconv(relaxation[0]))

        for e in excitation:
            #print(e.shape)
            pass
        
        return reconstruction, excitation, destillation, relaxation

       
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
    # TRAINING
    for epoch in range(50):
        for batch_n, X in enumerate(train_loader):
            if isinstance(X, list):
                X, Y = X
                Y = Y.to(device).float()
            X = X.to(device).float()
            print(X.shape)
            recon, exci, desti, relax = model(X)
            flat_desti = T.cat([layer.flatten(0) for layer in desti], dim=0)


            #print("pred", pred.shape)
            #print(pred.requires_grad)
            #loss = F.cross_entropy(pred, Y)
            #print(recon)

            # LOSSES
            orig_recon_loss = F.mse_loss(recon, X)
            loss = orig_recon_loss
            losses = {"orig recon": orig_recon_loss.detach().numpy().round(3)}

            if args.L1:
                L1_loss = flat_desti.abs().mean()
                loss = loss + args.L1* L1_loss
                losses["L1"] = L1_loss.detach().numpy().round(3)

            if args.acti_recon:
                #print(len(exci), len(relax))
                acti_recon_loss_list = [F.mse_loss(relax[i], exci[i]) for i in range(len(exci))]
                #print([e.item() for e in acti_recon_loss_list])
                acti_recon_loss = T.stack(acti_recon_loss_list).mean()
                loss = loss + args.acti_recon * acti_recon_loss
                losses["acti recon"] = acti_recon_loss.detach().numpy().round(3)

            opti.zero_grad()
            loss.backward()
            opti.step()

            if not batch_n%2:
                #Y = Y.cpu()
                flat_desti = flat_desti.detach()
                comp = T.isclose(flat_desti, T.zeros_like(flat_desti), atol=0.001)
                #X = X.cpu()
                #acc = (T.argmax(pred.detach(), dim=-1)==Y).sum()/Y.shape[0]
                print(epoch, batch_n, losses, "0-%:", comp.float().mean().numpy().round(3))

                if args.vis and not batch_n%6:
                    #maps = T.stack((maps, maps, maps), dim=-1)
                    X = X.cpu()
                    X = X.permute(0,2,3,1)
                    Z = recon.detach().permute(0,2,3,1)
                    if args.random:
                        X = X*T.tensor((0.229, 0.224, 0.225))+T.tensor((0.485, 0.456, 0.406))
                    together = T.cat((X,Z), dim=1)
                    pics = T.cat(list(together), dim=1)
                    print("saving results for"+f" {epoch}_{batch_n}", pics.shape)
                    #print(maps.shape, pics.shape, together.shape, pics.shape)
                    plt.imsave(save_path+f"{epoch}-{batch_n}.png", pics.squeeze().numpy())
        
        if test_loader is not None:
            eval_model(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-parse", action="store_true")
    parser.add_argument("-vis", action="store_true")
    parser.add_argument("-softmax", action="store_true")
    parser.add_argument("-max-inject", action="store_true")

    parser.add_argument("--acti-recon", type=float, default=0.0)
    parser.add_argument("--L1", type=float, default=0.0)
    parser.add_argument("--random", type=float, default=0)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="minerl")
    args = parser.parse_args()

    save_path = f"results/destill/{args.name}/"
    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if T.cuda.is_available() else "cpu"
    print("device:", device)

    v = 0
    a = 0.95

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
    dataname = args.dataset
    if dataname=="mnist":
        train_loader = T.utils.data.DataLoader(MNIST("data/", download=True, train=True, 
           transform=train_transforms), batch_size=64)    
        test_loader = T.utils.data.DataLoader(MNIST("data/", download=True, train=False, 
           transform=test_transforms), batch_size=64)    
    if dataname=="cifar10":
        train_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=True, 
            transform=train_transforms), batch_size=64)
        test_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=False, 
            transform=test_transforms), batch_size=64)
    if dataname=="minerl":
        data = get_minerl_dataset()
        #train, test = data, data[-2000:]
        train_loader = T.utils.data.DataLoader(data, batch_size=32)
        test_loader = None #T.utils.data.DataLoader(test, batch_size=32)
    #print(len(train_loader), len(test_loader))

    # SETUP MODEL AND OPTI
    #model = ReConvNet(1 if args.mnist else 3, args.dims, args.ks, args.ks, parse=args.parse).to(device)
    model = Destiller(1 if dataname=="mnist" else 3, args.dims, args.ks, args.ks, parse=args.parse).to(device)
    opti = T.optim.Adam(chain(model.parameters()), lr=1e-3)
    #sched = T.optim.lr_scheduler.CosineAnnealingLR

    train_model(model, train_loader, test_loader=test_loader)
    