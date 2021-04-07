import torch as T
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import torchvision.models as visionmodels
from torchvision import transforms
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np


class EncoderDecoder(nn.Module):
    def __init__(self, width, enc_dim, colorchs, activation=nn.Tanh):
        super().__init__()
        self.width = width
        self.enc = nn.Sequential(
            nn.Linear(colorchs*width*width, 32),
            activation(),
            nn.Linear(32, 16),
            activation(),
            nn.Linear(16, enc_dim),
            activation()
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, 16),
            activation(),
            nn.Linear(16, 32),
            activation(),
            nn.Linear(32, colorchs*width*width)
        )
        self.opti = T.optim.Adam(chain(self.enc.parameters(), self.dec.parameters()), 1e-3)
        self.sched = T.optim.lr_scheduler.StepLR(self.opti, 1, 0.5)

    def forward(self, x:T.Tensor):
        shape = x.shape
        x = x.flatten(start_dim=1)
        enc = self.enc(x)
        x = self.dec(enc)
        x = x.view(shape)
        return x, enc

    def convert_enc(self, enc):
        return (enc+1)/2

    def train_batch(self, batch):
        res, enc = self.forward(batch) # Forward pass

        loss = F.binary_cross_entropy_with_logits(res, batch) # Loss
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

        return loss.item(), T.sigmoid(res.detach()), enc.detach()

    def test_batch(self, batch):
        with T.no_grad():
            res, enc = self.forward(batch) # Forward pass()

        return T.sigmoid(res.detach()), enc.detach()

class ConditionedEncoderDecoder32x32(nn.Module):
    def __init__(self, edims = [8,8,16,16], ddims = [8,8,16,16], indim=3, outdim=8, cdim=8, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2,2))
        self.enc = [
            nn.Conv2d(indim,edims[0],3,2,1),
            self.activation(),
            self.pool(),
            nn.Conv2d(edims[0],edims[1],3,2,1),
            self.activation(),
            self.pool(),
            nn.Conv2d(edims[1],edims[2],3,2,1),
            self.activation(),
            self.pool(),
            nn.Conv2d(edims[2],edims[3],4),
            T.tanh()
        ]
        self.dec = [
            nn.ConvTranspose2d(ddims[3]+cdim,ddims[2],4),
            self.activation(),
            self.ups(),
            nn.Conv2d(edims[2]+ddims[2],ddims[1],3,2,1),
            self.activation(),
            self.ups(),
            nn.Conv2d(edims[1]+ddims[1],ddims[0],3,2,1),
            self.activation(),
            self.ups(),
            nn.Conv2d(edims[0]+ddims[0],outdim,3,2,1),
            T.tanh()
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X, C):
        encoding = self.enc_model(X)
        conditioned = T.cat((encoding, C.reshape(*encoding.shape[:-1], C.shape[-1])), dim=-1)
        return self.dec_model(conditioned)

class Printer(nn.Module):
    def __init__(self, shape=True, values=False):
        super().__init__()
        self.shape = shape
        self.values = values
    
    def forward(self, X):
        if self.shape:
            print(X.shape)
        if self.values:
            print(X)
        return X

class Encoder32x32(nn.Module):
    def __init__(self, channels = [8,8,16,16], indim=3, activation=nn.ReLU, end=nn.Tanh):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.enc = [
            nn.Conv2d(indim,channels[0],3,2,1),
            #Printer(),
            self.acti,
            #self.pool,
            nn.Conv2d(channels[0],channels[1],3,2,1),
            #Printer(),
            self.acti,
            #self.pool,
            nn.Conv2d(channels[1],channels[2],3,2,1),
            #Printer(),
            self.acti,
            #self.pool,
            nn.Conv2d(channels[2],channels[3],4),
            #Printer(),
            end()
        ]
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        return self.enc_model(X)

class Decoder32x32(nn.Module):
    def __init__(self, channels = [16,16,8,8], outdim=8, cdim=8, activation=nn.ReLU, end=nn.Sigmoid):
        super().__init__()
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2,2))
        self.dec = [
            nn.ConvTranspose2d(channels[3]+cdim,channels[2],4),
            #Printer(),
            self.acti,
            self.ups,
            #Printer(),
            nn.Conv2d(channels[2],channels[1],3,1,1),
            #Printer(),
            self.acti,
            self.ups,
            #Printer(),
            nn.Conv2d(channels[1],channels[0],3,1,1),
            #Printer(),
            self.acti,
            self.ups,
            #Printer(),
            nn.Conv2d(channels[0],outdim,3,1,1),
            #Printer(),
            end()
        ]
        self.dec_model = nn.Sequential(*self.dec)

    def forward(self, X):
        return self.dec_model(X)

class DecoderFlat(nn.Module):
    def __init__(self, channels = [1,8,8,8], cdim=8, activation=nn.ReLU, end=nn.Sigmoid):
        super().__init__()
        self.acti = activation()
        self.dec = [
            nn.Linear(channels[-1]+cdim, channels[-2]),
            self.acti,
            nn.Linear(channels[-2], channels[-3]),
            self.acti,
            nn.Linear(channels[-3], channels[-4]),
            end()
        ]
        self.dec_model = nn.Sequential(*self.dec)

    def forward(self, X):
        return self.dec_model(X)

class VAE(nn.Module):
    def __init__(self, width, enc_dim, colorchs, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.enc = nn.Sequential(
            nn.Linear(colorchs*width*width, 32),
            activation(),
            nn.Linear(32, 16),
            activation(),
            nn.Lineanetsr(16, enc_dim*2)
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, 16),
            activation(),
            nn.Linear(16, 32),
            activation(),
            nn.Linear(32, colorchs*width*width)
        )
        self.opti = T.optim.Adam(chain(self.enc.parameters(), self.dec.parameters()), lr=1e-3)
        self.sched = T.optim.lr_scheduler.StepLR(self.opti, 1, 0.5)

    def forward(self, x:T.Tensor):
        shape = x.shape
        x = x.flatten(start_dim=1)
        enc = self.enc(x)

        mean = enc[:, :enc.shape[-1]//2]
        log_std = enc[:, enc.shape[-1]//2:]
        #std = self.convert_enc(std)
        dist = T.distributions.Normal(mean, log_std.exp())
        sample = dist.rsample()

        x = self.dec(sample)
        x = x.view(shape)
        return x, mean, log_std, dist

    def convert_enc(self, enc):
        return (enc+1)/2

    def train_batch(self, batch):
        res, mean, log_std, dist = self.forward(batch) # Forward pass

        recon_loss = F.binary_cross_entropy_with_logits(res, batch, reduction="sum") # Loss
        #regul_loss = T.distributions.kl_divergence(dist, T.distributions.Normal(T.zeros_like(mean), T.ones_like(mean)))
        kl_loss = -0.5 * T.sum(1 + log_std - mean.pow(2) - log_std.exp())

        #print( recon_loss, regul_loss)
        #print(kl_loss)
        loss = recon_loss + 1*kl_loss
        #print(mean.mean(dim=0), log_std.mean(dim=0))

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

        return loss.item(), T.sigmoid(res.detach()), mean.detach()

    def test_batch(self, batch):
        with T.no_grad():
            res, mean, _, _ = self.forward(batch) # Forward pass()

        return T.sigmoid(res.detach()), mean.detach()

class MnistVAE(nn.Module):
    def __init__(self, channels=[1,4,8,16,32], activation=nn.ReLU):
        super().__init__()
        self.enc = nn.Sequential(
                nn.Conv2d(channels[0],channels[1],3,1,1),   # in 1x28x28
                nn.ReLU(),
                nn.MaxPool2d(2),        # out 4x14x14
                nn.Conv2d(channels[1],channels[2],3,1,2),   # in 4x16x16
                nn.ReLU(),
                nn.MaxPool2d(2),        # out 8x8x8
                nn.Conv2d(channels[2],channels[3],3,1,1),  # in 8x8x8
                nn.ReLU(),
                nn.MaxPool2d(2),        # out 16x4x4
                nn.Conv2d(channels[3],channels[4],4,1,0),  # out 16
                nn.Tanh()

        )
        
        self.dec = nn.Sequential(
                nn.ConvTranspose2d(channels[4]//2,channels[3],4,1,0),  # out 16
                nn.ReLU(),        # out 8x8x8
                nn.Upsample(scale_factor=(2,2)),
                nn.Conv2d(channels[3],channels[2],3,1,1),  # in 8x8x8
                nn.ReLU(),        # out 4x14x14
                nn.Upsample(scale_factor=(2,2)),
                nn.Conv2d(channels[2],channels[1],3,1,0),   # in 4x16x16
                nn.ReLU(),        # out 16x4x4
                nn.Upsample(scale_factor=(2,2)),
                nn.Conv2d(channels[1],channels[0],3,1,1),   # in 1x28x28
        )

        self.opti = T.optim.Adam(chain(self.enc.parameters(), self.dec.parameters()), lr=1e-3)
        self.sched = T.optim.lr_scheduler.StepLR(self.opti, 1, 0.5)

    def forward(self, x:T.Tensor):
        shape = x.shape
        #x = x.flatten(start_dim=1)
        enc = self.enc(x)

        mean = enc[:, :enc.shape[1]//2].squeeze()
        log_std = enc[:, enc.shape[1]//2:].squeeze()
        #std = self.convert_enc(std)
        dist = T.distributions.Normal(mean, log_std.exp())
        #sample = dist.rsample().view(enc.shape[0], enc.shape[1]//2, *enc.shape[2:])
        sample = dist.rsample()[:,:,None,None]

        x = self.dec(sample)
        #x = x.view(shape)
        return x, mean, log_std, dist

    def convert_enc(self, enc):
        return (enc+1)/2

    def train_batch(self, batch):
        res, mean, log_std, dist = self.forward(batch) # Forward pass

        recon_loss = F.binary_cross_entropy_with_logits(res, batch, reduction="sum") # Loss
        #regul_loss = T.distributions.kl_divergence(dist, T.distributions.Normal(T.zeros_like(mean), T.ones_like(mean)))
        kl_loss = -0.5 * T.sum(1 + log_std - mean.pow(2) - log_std.exp())

        #print( recon_loss, regul_loss)
        #print(kl_loss)
        loss = recon_loss + 1*kl_loss
        #print(mean.mean(dim=0), log_std.mean(dim=0))

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

        return loss.item(), T.sigmoid(res.detach()), mean.detach()

    def test_batch(self, batch):
        with T.no_grad():
            res, mean, _, _ = self.forward(batch) # Forward pass()

        return T.sigmoid(res.detach()), mean.detach()

class Critic(nn.Module):
    def __init__(self, width=64, enc_dim=1, colorchs=3, activation=nn.ReLU, end=[]):
        super().__init__()
        self.width = width
        modules = [nn.Conv2d(colorchs,8,3,1,1),
            activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,3,1,1),
            activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,3,1,1),
            activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,1,1),
            activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,1,4)]
        modules.extend(end)
        self.enc = nn.Sequential(*modules)

    def forward(self, X):
        return self.enc(X)

class SmallCritic(nn.Module):
    def __init__(self, width=16, enc_dim=1, colorchs=3, chfak=1, activation=nn.ReLU, end=[]):
        super().__init__()
        self.width = width
        modules = [nn.Conv2d(colorchs,8*chfack,3,1,2),
            activation(),
            #nn.MaxPool2d(2),
            nn.Conv2d(8*chfack,16*chfack,3,1,2),
            activation(),
            #nn.MaxPool2d(2),
            nn.Conv2d(16*chfack,1,4)]
        modules.extend(end)
        self.enc = nn.Sequential(*modules)

    def forward(self, X):
        return self.enc(X)

class Unet64x64(nn.Module):
    def __init__(self, width=64, edims = [8,8,8,16,32], ddims = [8,8,8,16,32], colorchs=3, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2,2))
        self.enc = [
            nn.Conv2d(colorchs,edims[0],3,1,1),
            nn.Conv2d(edims[0],edims[1],3,1,1),
            nn.Conv2d(edims[1],edims[2],3,1,1),
            nn.Conv2d(edims[2],edims[3],3,1,1),
            nn.Conv2d(edims[3],edims[4],4)
        ]
        self.dec = [
            nn.Conv2d(edims[0]+ddims[0],1,3,1,1),
            nn.Conv2d(edims[1]+ddims[1],ddims[0],3,1,1),
            nn.Conv2d(edims[2]+ddims[2],ddims[1],3,1,1),
            nn.Conv2d(edims[3]+ddims[3],ddims[2],3,1,1),
            nn.ConvTranspose2d(ddims[4],ddims[3],4,1,0)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        enc = self.enc
        dec = self.dec
        x0 = acti(enc[0](X))
        #print(x0.shape)
        x1 = acti(enc[1](pool(x0)))
        #print(x1.shape)
        x2 = acti(enc[2](pool(x1)))
        #print(x2.shape)
        x3 = acti(enc[3](pool(x2)))
        #print(x3.shape)
        x4 = acti(enc[4](pool(x3)))
        #print(x4.shape)
        u3 = acti(dec[4](x4))
        #print(u3.shape)
        u2 = acti(dec[3](T.cat((ups(u3), x3), dim=1)))
        #print(u2.shape)
        u1 = acti(dec[2](T.cat((ups(u2), x2), dim=1)))
        #print(u1.shape)
        u0 = acti(dec[1](T.cat((ups(u1), x1), dim=1)))
        #print(u0.shape)
        y = T.sigmoid(dec[0](T.cat((ups(u0), x0), dim=1)))
        #print(y.shape)
        
        return y

class Unet32x32(nn.Module):
    def __init__(self, edims = [8,8,16,16], ddims = [8,8,16,16], indim=3, outdim=8, activation=nn.ReLU):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2,2))
        self.enc = [
            nn.Conv2d(indim,edims[0],3,1,1),
            nn.Conv2d(edims[0],edims[1],3,1,1),
            nn.Conv2d(edims[1],edims[2],3,1,1),
            nn.Conv2d(edims[2],edims[3],4)
        ]
        self.dec = [
            nn.Conv2d(edims[0]+ddims[0],outdim,3,1,1),
            nn.Conv2d(edims[1]+ddims[1],ddims[0],3,1,1),
            nn.Conv2d(edims[2]+ddims[2],ddims[1],3,1,1),
            nn.ConvTranspose2d(ddims[3],ddims[2],4)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        enc = self.enc
        dec = self.dec
        x0 = acti(enc[0](X))
        #print(x0.shape)
        x1 = acti(enc[1](pool(x0)))
        #print(x1.shape)
        x2 = acti(enc[2](pool(x1)))
        #print(x2.shape)
        x3 = acti(enc[3](pool(x2)))
        #print(x3.shape)
        u2 = acti(dec[3](x3))
        #print(u2.shape)
        u1 = acti(dec[2](T.cat((ups(u2), x2), dim=1)))
        #print(u1.shape)
        u0 = acti(dec[1](T.cat((ups(u1), x1), dim=1)))
        #print(u0.shape)
        y = T.sigmoid(dec[0](T.cat((ups(u0), x0), dim=1)))
        #print(y.shape)
        
        return y

class ConditionalUnet32x32(nn.Module):
    def __init__(self, edims = [8,8,16,16], ddims = [8,8,16,16], indim=3, outdim=8, cdim=8, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2,2))
        self.enc = [
            nn.Conv2d(indim,edims[0],3,1,1),
            nn.Conv2d(edims[0],edims[1],3,1,1),
            nn.Conv2d(edims[1],edims[2],3,1,1),
            nn.Conv2d(edims[2],edims[3],4)
        ]
        self.dec = [
            nn.Conv2d(edims[0]+ddims[0],outdim,3,1,1),
            nn.Conv2d(edims[1]+ddims[1],ddims[0],3,1,1),
            nn.Conv2d(edims[2]+ddims[2],ddims[1],3,1,1),
            nn.ConvTranspose2d(ddims[3],ddims[2],4)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X, C):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        enc = self.enc
        dec = self.dec
        x0 = acti(enc[0](X))
        #print(x0.shape)
        x1 = acti(enc[1](pool(x0)))
        #print(x1.shape)
        x2 = acti(enc[2](pool(x1)))
        #print(x2.shape)
        x3 = acti(enc[3](pool(x2)))
        #print(x3.shape)
        conditioned = T.cat((x3, C.reshape(*x3.shape[:-1], C.shape[-1])), dim=-1)
        #print(conditioned.shape)
        u2 = acti(dec[3](x3))
        #print(u2.shape)
        u1 = acti(dec[2](T.cat((ups(u2), x2), dim=1)))
        #print(u1.shape)
        u0 = acti(dec[1](T.cat((ups(u1), x1), dim=1)))
        #print(u0.shape)
        y = T.sigmoid(dec[0](T.cat((ups(u0), x0), dim=1)))
        #print(y.shape)
        
        return y

class Unet16x16(nn.Module):
    def __init__(self, width=64, edims=[8, 8, 16], ddims=[8, 8, 16], colorchs=3, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2, 2))
        self.enc = [
            nn.Conv2d(colorchs, edims[0], 3, 1, 1),
            nn.Conv2d(edims[0], edims[1], 3, 1, 1),
            nn.Conv2d(edims[1], edims[2], 4, 1, 0)
        ]
        self.dec = [
            nn.Conv2d(edims[0] + ddims[0], 1, 3, 1, 1),
            nn.Conv2d(edims[1] + ddims[1], ddims[0], 3, 1, 1),
            nn.ConvTranspose2d(ddims[2], ddims[1], 4, 1, 0)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        x0 = self.acti(self.enc[0](X))
        # print(x0.shape)
        x1 = self.acti(self.enc[1](self.pool(x0)))
        # print(x1.shape)
        x2 = self.acti(self.enc[2](self.pool(x1)))
        # print(x2.shape)
        u1 = self.acti(self.dec[2](x2))
        #print(u1.shape, x1.shape)
        u0 = self.acti(self.dec[1](T.cat((self.ups(u1), x1), dim=1)))
        # print(u0.shape)
        y = T.sigmoid(self.dec[0](T.cat((self.ups(u0), x0), dim=1)))
        # print(y.shape)
        return y

class GroundedUnet(nn.Module):
    def __init__(self, width=64, edims = [8,8,8,16,32], ddims = [8,8,8,16,32], colorchs=3, activation=nn.ReLU):
        super().__init__()
        self.width = width
        self.pool = nn.MaxPool2d(2)
        self.acti = activation()
        self.ups = nn.Upsample(scale_factor=(2,2))
        self.down = lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        self.enc = [
            nn.Conv2d(colorchs,edims[0],3,1,1),
            nn.Conv2d(3+edims[0],edims[1],3,1,1),
            nn.Conv2d(3+edims[1],edims[2],3,1,1),
            nn.Conv2d(3+edims[2],edims[3],3,1,1),
            nn.Conv2d(edims[3],edims[4],4)
        ]
        self.dec = [
            nn.Conv2d(edims[0]+ddims[0],1,3,1,1),
            nn.Conv2d(edims[1]+ddims[1],ddims[0],3,1,1),
            nn.Conv2d(edims[2]+ddims[2],ddims[1],3,1,1),
            nn.Conv2d(edims[3]+ddims[3],ddims[2],3,1,1),
            nn.ConvTranspose2d(ddims[4],ddims[3],4,1,0)
        ]
        self.dec_model = nn.Sequential(*self.dec)
        self.enc_model = nn.Sequential(*self.enc)

    def forward(self, X):
        pool = self.pool
        ups = self.ups
        acti = self.acti
        enc = self.enc
        dec = self.dec
        x0 = acti(enc[0](X))
        #print(x0.shape)
        d1 = self.down(X)
        x1 = acti(enc[1](T.cat((pool(x0), d1), dim=1)))
        #print(x1.shape)
        d2 = self.down(d1)
        x2 = acti(enc[2](T.cat((pool(x1), d2), dim=1)))
        #print(x2.shape)
        d3 = self.down(d2)
        x3 = acti(enc[3](T.cat((pool(x2), d3), dim=1)))
        #print(x3.shape)
        x4 = acti(enc[4](pool(x3)))
        #print(x4.shape)
        u3 = acti(dec[4](x4))
        #print(u3.shape)
        u2 = acti(dec[3](T.cat((ups(u3), x3), dim=1)))
        #print(u2.shape)
        u1 = acti(dec[2](T.cat((ups(u2), x2), dim=1)))
        #print(u1.shape)
        u0 = acti(dec[1](T.cat((ups(u1), x1), dim=1)))
        #print(u0.shape)
        y = T.sigmoid(dec[0](T.cat((ups(u0), x0), dim=1)))
        #print(y.shape)
        
        return y

class ResNetCritic(nn.Module):
    def __init__(self, HSV=True):
        super().__init__()
        self.HSV = HSV
        self.norm = get_normalizer()
        self.mean = T.tensor([0.485, 0.456, 0.406])
        self.std = T.tensor([0.229, 0.224, 0.225])
        self.resnet = get_resnet18_features()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),    
        )

    def forward(self, X):
        device = X.device
        if X.max()>1:
            X = X/255.0
        X = X.permute(0,2,3,1)
        if device != self.mean.device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        X = (X-self.mean)/self.std
        X = X.permute(0,3,1,2)
        features = self.resnet(X)
        return self.head(features)

class VGGEmbedder(nn.Module):
    def __init__(self):
        super(VGGEmbedder, self).__init__()
        self.model = get_vgg_features()
        self.normer = get_normalizer()

    def forward(self, X, hsv=False, permute=False, norm=False, uint=False):
        if hsv:
            X = T.from_numpy(hsv_to_rgb(X))
        if norm:
            X = self.normer(X)
        if permute:
            X = X.permute(0,3,1,2)
        if uint:
            X = X.float()

        h, w = X.shape[-2:]
        embeds = [X]
        for layer in self.model.children():
            X = layer(X)
            if (isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh)):
                embeds.append(X)

        return (X, embeds)


def get_resnet18_features():
    resnet18 = visionmodels.resnet18(pretrained=True)
    features = nn.Sequential(*(list(resnet18.children())[0:8]))
    return features


def get_vgg_features():
    vgg = visionmodels.vgg11(pretrained=True).features
    return vgg


def get_mobilenet_features():
    net = visionmodels.mobilenet_v2(pretrained=True).features
    tet = T.randn(8,3,64,64)
    out = net(tet)
    print(net)
    print(out.shape)


def get_inceptionv3_features():
    net = visionmodels.inception_v3(pretrained=True)
    features = nn.Sequential(*(list(net.children())[0:8]))
    return features

def get_normalizer():
    #return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return lambda x: (x-T.tensor([0.485, 0.456, 0.406]))/T.tensor([0.229, 0.224, 0.225])


if __name__ == "__main__":
    #print((list(get_vgg_features().children())))
    get_mobilenet_features()
    exit()