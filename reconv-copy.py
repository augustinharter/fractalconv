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
        self.inconv = nn.Conv2d(in_dim, hid_dim, in_kernel, 1, in_kernel//2)
        #self.outconv = nn.Conv2d(hid_dim, in_dim, kernel_size, 1, 0)
        self.reconv = nn.Conv2d(hid_dim, hid_dim, hid_kernel, 1, hid_kernel//2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, X):
        X = self.inconv(X)
        X = F.relu(X)
        for i in range(math.ceil(math.log2(X.shape[2]))-1):
            X = self.reconv(X)
            #print(i, X.shape)
            X = F.relu(X)
            X = self.pool(X)
        #print("last", X.shape)
        return X


if __name__ == "__main__":
    save_path = f"results/reconv/00/"
    os.makedirs(save_path, exist_ok=True)

    v = 0
    a = 0.95
    acc = 0
    model = ReConvNet(1, 32, 5, 5)

    train_loader = T.utils.data.DataLoader(MNIST("data/mnist", download=True, train=True, 
       transform=transforms.ToTensor()), batch_size=64)
    #train_loader = T.utils.data.DataLoader(CIFAR10("data/cifar10", download=True, train=True, 
    #    transform=transforms.ToTensor()), batch_size=64)
    opti = T.optim.Adam(chain(model.parameters()))

    for epoch in range(10):
        for batch_n, (X,Y) in enumerate(train_loader):

            enc = model(X)
            #print(enc.shape)

            loss = F.cross_entropy(enc.flatten(1)[:,:10], Y)

            opti.zero_grad()
            loss.backward()
            opti.step()

            if not batch_n%10:
                acc = (T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)==Y).sum()/Y.shape[0]
                print(batch_n, round(loss.item(), 3), acc.numpy().round(3), 
                    T.argmax(enc.detach().flatten(1)[:,:10], dim=-1)[:5])

            if not batch_n%100:
                print("saving results for"+f" {epoch}_{batch_n}")
                #maps = T.stack((maps, maps, maps), dim=-1)
                X = X.permute(0,2,3,1)
                together = T.cat((X,), dim=1)
                pics = T.cat(list(together), dim=1)
                print(pics.shape)
                #print(maps.shape, pics.shape, together.shape, pics.shape)
                plt.imsave(save_path+f"{epoch}-{batch_n}.png", pics.squeeze().numpy())