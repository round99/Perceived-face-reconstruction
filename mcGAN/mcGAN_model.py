import torch.nn as nn
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# weights initialization
def weight_init(m:nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Generator(nn.Module):
    """
    input: [bs, 100, 1, 1]
    output: [bs, nc, 136, 100]
    """
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(nz, ngf * 8 * 9 * 7),
            Reshape(ngf*8, 9, 7),
            nn.BatchNorm2d(ngf * 8, momentum=0.9),
            nn.ReLU(True),
            # # state shape: [bs, ngf*8, 9, 7]

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.9),
            nn.ReLU(True),
            # state shape: [bs, ngf*4, 17, 13]

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, output_padding=(1, 0), bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.9),
            nn.ReLU(True),
            # state shape: [bs, ngf*2, 34, 25]

            nn.ConvTranspose2d(ngf * 2, ngf, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.9),
            nn.ReLU(True),
            # state shape: [bs, ngf, 68, 50]

            nn.ConvTranspose2d(ngf, nc, 5, 2, 2, output_padding=1,  bias=False),
            nn.Tanh()
            # state shape: [bs, nc, 136, 100]
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input shape: [bs, nc, 136, 100]
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf, 68, 50]

            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf*2, 34, 25]

            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf*4, 17, 13]

            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf*8, 9, 7]

        )

        self.net1 = nn.Sequential(
            Reshape(ndf * 8 * 9 * 7),
            nn.Linear(ndf*8*9*7, 1),
            nn.Sigmoid()
        # state shape: [bs, 1, 1, 1]
        )



        self.net2 = nn.Sequential(
            Reshape(ndf * 8 * 9 * 7),
            nn.Linear(ndf * 8 * 9 * 7, 107),
            nn.Sigmoid()
        # state shape: [bs, 128, 1, 107]
        )


        self.net3 = nn.Sequential(
            Reshape(ndf * 8 * 9 * 7),
            nn.Linear(ndf * 8 * 9 * 7, 7),
            nn.Sigmoid()

            # state shape: [bs, 128, 1, 7]
        )


        self.net4 = nn.Sequential(
            Reshape(ndf * 8 * 9 * 7),
            nn.Linear(ndf * 8 * 9 * 7, 2),
            nn.Sigmoid()

            # state shape: [bs, 128, 1, 2]
        )

    def forward(self, x):
        x = self.net(x)
        out_1 = self.net1(x)
        out_107 = self.net2(x)
        out_7 = self.net3(x)
        out_2 = self.net4(x)
        return out_1, out_107, out_7, out_2


def get_GAN(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    G = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    D = Discriminator(ndf=args.ndf, nc=args.nc).to(device)
    G.apply(weight_init)
    D.apply(weight_init)

    return G, D


