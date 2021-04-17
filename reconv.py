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
        while X.shape[-1]>self.hid_kernel:
            X = F.pad(X, (self.pad, self.pad, self.pad, self.pad))
            #print(i, "midpad", X.shape)
            X = self.reconv(X)
            #print(i, "midconv", X.shape)
            X = self.acti(X)
            X = self.pool(X)
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
        
        return X


if __name__ == "__main__":
    save_path = f"results/reconv/00/"
    os.makedirs(save_path, exist_ok=True)

    v = 0
    a = 0.95
    model = ReConvNet(1, 32, 3, 3)

    train_loader = T.utils.data.DataLoader(MNIST("data/mnist", download=True, train=True, 
        transform=transforms.ToTensor()), batch_size=64)
    #train_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=True, 
    #    transform=transforms.ToTensor()), batch_size=64)
    opti = T.optim.Adam(chain(model.parameters()), lr=1e-3)
    #sched = T.optim.lr_scheduler.CosineAnnealingLR

    for epoch in range(10):
        for batch_n, (X,Y) in enumerate(train_loader):

            enc = model(X)
            #print(enc.shape)
            #print(enc.requires_grad)

            loss = F.cross_entropy(enc.flatten(1)[:,:10], Y)

            opti.zero_grad()
            loss.backward()
            opti.step()

            if not batch_n%10:
                acc = (T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)==Y).sum()/Y.shape[0]
                print(epoch, batch_n, round(loss.item(), 3), acc.numpy().round(3), 
                    T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)[:5])

            if False:#not batch_n%100:
                #maps = T.stack((maps, maps, maps), dim=-1)
                X = X.permute(0,2,3,1)
                together = T.cat((X,), dim=1)
                pics = T.cat(list(together), dim=1)
                print("saving results for"+f" {epoch}_{batch_n}", pics.shape)
                #print(maps.shape, pics.shape, together.shape, pics.shape)
                plt.imsave(save_path+f"{epoch}-{batch_n}.png", pics.squeeze().numpy())