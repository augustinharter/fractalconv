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
from PIL import Image, ImageDraw


class ReConvNet(nn.Module):
    def __init__(self, in_dim, hid_dim, in_kernel, hid_kernel):
        super().__init__()
        #self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        #self.outconv = nn.Conv2d(hid_dim, in_dim, kernel_size, 1, 0)
        #self.reconv = nn.Conv2d(hid_dim, hid_dim, hid_kernel, 1, hid_kernel//2)
        self.reconv = nn.Conv2d(hid_dim, hid_dim, hid_kernel, 1, 0)
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

        self.pool = nn.MaxPool2d(2)
        self.adapool = nn.AdaptiveMaxPool2d(n_parse_feature_per_spatial_dim)
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
        
        return pred


def eval_model(model, loader):
    hits = []
    for batch_n, (X,Y) in enumerate(loader):
        X = X.to(device)
        Y = Y.to(device)
        enc = model(X)
        #print("enc", enc.shape)
        #print(enc.requires_grad)

        #loss = F.cross_entropy(enc.flatten(1)[:,:10], Y)
        loss = F.cross_entropy(enc, Y)

        opti.zero_grad()
        loss.backward()
        opti.step()

        enc = enc.cpu()
        Y = Y.cpu()
        #X = X.cpu()
        hits.append(T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)==Y)
        if batch_n != 10:
            acc = hits[-1].sum()/len(hits[-1])
            print("running eval:", batch_n, acc.numpy().round(3), end="\r")
        
        if batch_n>=10:
            break

    
    print()
    hits = T.cat(hits, dim=0)
    acc = hits.sum()/hits.shape[0]
    print("eval finished - accuracy:", acc.item())


if __name__ == "__main__":
    save_path = f"results/workspace/"
    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if T.cuda.is_available() else "cpu"
    print("device:", device)

    v = 0
    a = 0.95
    model = ReConvNet(3, 128, 5, 5).to(device)
    acc = 0

    #train_loader = T.utils.data.DataLoader(MNIST("data/mnist", download=True, train=True, 
    #    transform=transforms.ToTensor()), batch_size=64)
    train_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=True, 
        transform=transforms.ToTensor()), batch_size=64)
    test_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=False, 
        transform=transforms.ToTensor()), batch_size=64)
    opti = T.optim.Adam(chain(model.parameters()), lr=1e-3)
    #sched = T.optim.lr_scheduler.CosineAnnealingLR

    for epoch in range(100):
        for batch_n, (X,Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            enc = model(X)
            #print("enc", enc.shape)
            #print(enc.requires_grad)

            #loss = F.cross_entropy(enc.flatten(1)[:,:10], Y)
            loss = F.cross_entropy(enc, Y)

            opti.zero_grad()
            loss.backward()
            opti.step()

            if not batch_n%100:
                enc = enc.cpu()
                Y = Y.cpu()
                #X = X.cpu()
                acc = 0.9*acc + 0.1*(T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)==Y).sum()/Y.shape[0]
                print(epoch, batch_n, round(loss.item(), 3), acc.numpy().round(3), 
                    T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)[:5])

            if False:#not batch_n%100:
                #maps = T.stack((maps, maps, maps), dim=-1)
                X = X.cpu()
                enc = enc.cpu()
                X = X.permute(0,2,3,1)
                together = T.cat((X,), dim=1)
                pics = T.cat(list(together), dim=1)
                print("saving results for"+f" {epoch}_{batch_n}", pics.shape)
                #print(maps.shape, pics.shape, together.shape, pics.shape)
                plt.imsave(save_path+f"{epoch}-{batch_n}.png", pics.squeeze().numpy())
        
        eval_model(model, test_loader)
