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


class NCA(nn.Module):
    def __init__(self, in_dim, hid_dim, in_kernel, hid_kernel, parse=False, radius=3, width=64):
        super().__init__()
        self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        self.outconv = nn.Conv2d(hid_dim, in_dim, in_kernel, 1, in_kernel//2)
        self.reconv = nn.Conv2d(hid_dim, hid_dim, hid_kernel, 1, 0)

        self.ldim = hid_dim
        self.width = width
        self.depth = int(math.log2(self.width))+1
        self.resolutions = [2**i for i in range(self.depth) if i!=1][::-1]
        print("resolutions:", self.resolutions)

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
        self.sobelx_kernel = nn.Parameter(0+sobelx.expand(hid_dim, hid_dim, radius, radius), requires_grad=False)
        self.sobely_kernel = nn.Parameter(0+sobely.expand(hid_dim, hid_dim, radius, radius), requires_grad=False)

        update_conv_func = lambda: nn.Sequential(
            nn.Conv2d(hid_dim*5, hid_dim*2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hid_dim*2, hid_dim, 1, 1, 0)
        )

        up_conv = nn.Conv2d(hid_dim, hid_dim, 3, 1, 1) 
        self.upconvkernel = nn.Parameter(up_conv.weight, requires_grad=True)
        self.upconvbias = nn.Parameter(up_conv.bias, requires_grad=True)

        self.down_conv = nn.Conv2d(hid_dim, hid_dim, 3, 1, 1),

        self.encode = update_conv_func()
        self.decode = update_conv_func()
        self.rules = update_conv_func()

        self.pool = nn.MaxPool2d(2)
        self.hid_kernel = hid_kernel
        self.pad = hid_kernel//2
        self.acti = nn.LeakyReLU()
        self.nonlin = nn.LeakyReLU()


    def excite_old(self, X):
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


    def excite(self, X):
        actis = []

        # PROJECTING
        X = self.inconv(X)
        X = self.nonlin(X)
        actis.append(X)

        # CONVOLVING
        while X.shape[-1]!=1:
            if X.shape[-1]!=4:
                X = F.pad(X, (1,1,1,1))
            X = self.reconv(X)
            X = self.nonlin(X)
            X = self.pool(X)
            actis.append(X)

        return actis


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


    def propagate(self, modes=[1,-1,1,-1,1,-1], stimulus=None, actis=None, 
        up_mode="conv", down_mode="bilinear", side_mode="grad"):
        global args
        # Check if stimulus OR actis are given
        assert (stimulus is not None) or (actis is not None)

        # Get parameters from class
        width = self.width
        ldim = self.ldim

        # Init actis if None
        if actis is None:
            batchsize = len(stimulus)
            #print("resolutions", resolutions)
            actis = [T.zeros(batchsize, ldim, res, res, device=stimulus.device) for res in self.resolutions]
        else:
            batchsize = len(actis[0])
            actis = [e.clone() for e in actis]


        #print("acti shapes", [tuple(a.shape) for a in actis])
        depth = len(actis)

        # Repetition Loop
        for mode_idx, mode in enumerate(modes):
            
            # Iterating over Layers
            for layer_idx in range(depth)[::mode or 1]:
                current_layer = actis[layer_idx]
                #print(mode_idx, layer_idx, current_layer.shape)

                # BOTTOM UP
                if not layer_idx:  # if first layer
                    if stimulus is not None:
                        bottom_up = self.inconv(stimulus)
                    else:
                        bottom_up = T.zeros(batchsize, ldim, width, width).to(actis[0].device)
                else:  # if not first layer
                    lower_layer = actis[layer_idx-1]
                    if up_mode=="conv":
                        pad = 0 if lower_layer.shape[-1]==4 else 1
                        bottom_up = F.conv2d(lower_layer, self.upconvkernel, self.upconvbias, stride=1, padding=pad)
                        bottom_up = self.nonlin(bottom_up)
                        bottom_up = self.pool(bottom_up)
                    elif up_mode=="max":
                        bottom_up = F.max_pool2d(lower_layer, kernel_size=2, stride=1)
                    elif up_mode=="avg":
                        bottom_up = F.avg_pool2d(lower_layer, kernel_size=2, stride=1)
                    else:
                        assert False, f"can't recognize up mode {up_mode}"

                # TOP DOWN
                if layer_idx==depth-1:  # top layer
                    top_down = T.zeros_like(current_layer) 
                else:  # not top layer
                    upper_layer = actis[layer_idx+1]
                    if down_mode in ["nearest", "bilinear"]:
                        top_down = F.interpolate(upper_layer, current_layer.shape[-2:], mode=down_mode)
                    elif down_mode == "conv":
                        top_down = F.interpolate(upper_layer, current_layer.shape[-2:], mode="bilinear", align_corners=False)
                        top_down = self.down_conv(top_down)
                        top_down = self.nonlin(top_down)
                    else:
                        assert False, f"can't recognize down mode {down_mode}"

                # SIDE SIDE
                if side_mode=="grad":
                    pad = 1
                    xgrad = F.conv2d(current_layer, self.sobelx_kernel, stride=1, padding=pad)
                    ygrad = F.conv2d(current_layer, self.sobely_kernel, stride=1, padding=pad)
                    if not pad:
                        xgrad = F.pad(xgrad, [0,0,0,0])
                        ygrad = F.pad(ygrad, [0,0,0,0])
                    side_side = T.cat((xgrad, ygrad), dim=1)
                elif side_mode=="full":
                    h, w = current_layer.shape[-2:]
                    padded = F.pad(current_layer, [1,1,1,1])
                    neighb_idxs = [(yi,xi) for xi in [-1,0,1] for yi in [-1,0,1] if xi*yi]
                    neighbs = [padded[:, :, 1+yi:1+yi+h, 1+xi:1+xi+w] for (yi,xi) in neighb_idxs]
                    side_side = T.cat(neighbs, dim=1)
                else:
                    assert False, f"can't recognize side mode {side_mode}"

                # UPDATE
                #print(mode_idx, layer_idx, bottom_up.shape, top_down.shape, side_side.shape)
                influence = T.cat((current_layer, bottom_up, top_down, side_side), dim=1)
                actis[layer_idx] = current_layer + self.rules(influence)
                zeros = T.isclose(current_layer.detach().flatten(), T.zeros_like(current_layer.detach().flatten()), atol=0.01)
                #print("zeros", mode_idx, layer_idx, zeros.float().mean().numpy().round(5))
                if args.stimulus_mode=="residual":
                    stimulus = stimulus + self.outconv(actis[0])
                elif args.stimulus_mode=="direct":
                    stimulus = self.outconv(actis[0])
               

        """
        # PRUNING OPTIONS
        if mode=="encode":
            if args.max:
                for i in range(depth):
                    if i>=args.pruning:
                        actis[i] = actis[i] * (actis[i].max(dim=1).values.unsqueeze(1)==actis[i]) 
                    else:
                        actis[i] = T.zeros_like(actis[i])
                    
            if args.softmax:
                for i in range(depth):
                    if i>=args.pruning:
                        actis[i] = T.softmax(actis[i], dim=1)
                    else:
                        actis[i] = T.T.zeros_like(actis[i])
        """

        if args.stimulus_mode!="end-residual":
            stimulus = stimulus + self.outconv(actis[0])
        if args.stimulus_mode!="end-direct":
            stimulus = self.outconv(actis[0])

        #print("all acti diffs:", [(actis[i]-acti_backup[i]).abs().mean().item() for i in range(len(actis))])
        return stimulus, actis


    def forward(self, X, excite=False):
        excitation = self.excite(X) if excite else None
        #print("exci", [e.shape for e in excitation])
        reconstruction, actis = self.propagate(actis=excitation) if excite else self.propagate(stimulus=X)

        return reconstruction, actis

       
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
    for epoch in range(args.epochs):
        for batch_n, X in enumerate(train_loader):
            if isinstance(X, list):
                X, Y = X
                Y = Y.to(device).float()
            X = X.to(device).float()
            recon, actis = model(X, excite=args.excite)
            flat_acti = T.cat([layer.flatten(0) for layer in actis], dim=0)


            #print("pred", pred.shape)
            #print(pred.requires_grad)
            #loss = F.cross_entropy(pred, Y)
            #print(recon)

            # LOSSES
            orig_recon_loss = F.mse_loss(recon, X)
            loss = orig_recon_loss
            losses = {"orig recon": orig_recon_loss.cpu().detach().numpy().round(3)}

            if args.L1:
                L1_loss = flat_acti.abs().mean()
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

            if not batch_n%args.vis:
                #Y = Y.cpu()
                flat_acti = flat_acti.detach()
                zeros = T.isclose(flat_acti, T.zeros_like(flat_acti), atol=0.001)
                #X = X.cpu()
                #acc = (T.argmax(pred.detach(), dim=-1)==Y).sum()/Y.shape[0]
                print(epoch, batch_n, losses)#, "zero%:", zeros.float().mean().cpu().numpy().round(3))

                X = X.cpu()
                X = X.permute(0,2,3,1)
                Z = recon.detach().permute(0,2,3,1).cpu()
                if args.random:
                    X = X*T.tensor((0.229, 0.224, 0.225))+T.tensor((0.485, 0.456, 0.406))
                together = T.cat((X,Z), dim=1)
                pics = T.cat(list(together), dim=1)
                print("saving results for"+f" {epoch}_{batch_n}", pics.shape)
                #print(maps.shape, pics.shape, together.shape, pics.shape)
                plt.imsave(save_path+f"{epoch}-{batch_n}.png", pics.squeeze().clamp(0,1).numpy())
        
        if test_loader is not None:
            #eval_model(model, test_loader)
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-parse", action="store_true")
    parser.add_argument("-softmax", action="store_true")
    parser.add_argument("-max", action="store_true")
    parser.add_argument("-excite", action="store_true")

    parser.add_argument("--vis", type=int, default=10)
    parser.add_argument("--datasize", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pruning", type=int, default=0)
    parser.add_argument("--stimulus-mode", type=str, default="end-direct")
    parser.add_argument("--acti-recon", type=float, default=0.0)
    parser.add_argument("--L1", type=float, default=0.0)
    parser.add_argument("--random", type=float, default=0)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="minerl")
    args = parser.parse_args()

    save_path = f"results/nca/{args.name}/"
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
        data = get_minerl_dataset(size=args.datasize)
        #train, test = data, data[-2000:]
        train_loader = T.utils.data.DataLoader(data, batch_size=32, shuffle=True)
        test_loader = None #T.utils.data.DataLoader(test, batch_size=32)
    #print(len(train_loader), len(test_loader))

    # SETUP MODEL AND OPTI
    #model = ReConvNet(1 if args.mnist else 3, args.dims, args.ks, args.ks, parse=args.parse).to(device)
    model = NCA(1 if dataname=="mnist" else 3, args.dims, args.ks, args.ks, parse=args.parse, width=64 if dataname=="minerl" else 32).to(device)
    opti = T.optim.Adam(chain(model.parameters()), lr=1e-3)
    #sched = T.optim.lr_scheduler.CosineAnnealingLR

    train_model(model, train_loader, test_loader=test_loader)
    