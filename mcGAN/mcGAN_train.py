from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')

import argparse
import os
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from mcGAN_model import get_GAN
from mcGAN_dataloader import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/xxx/xxx/xxx/xxx/') #summary address

data_root = '/xxx/xxx/xxx/xxx/'
id_label_root = '/xxx/xxx/xxx/xxx/'
img_size = (136, 100)

lamda1 = 20
lamda2 = 20
lamda3 = 20

def fix_noise():
    path = '/xxx/xxx/xxx/xxx/'
    imgs = os.listdir(path)
    imgs.sort()

    id_npy = []
    emo_npy = []
    gen_npy = []

    for img in imgs:
        id_npy.append(np.load(path + img))
        emo_npy.append(np.load('/xxx/xxx/xxx/xxx/'))
        gen_npy.append(np.load('/xxx/xxx/xxx/xxx/'))

    id_npy = np.array(id_npy)
    emo_npy = np.array(emo_npy)
    gen_npy = np.array(gen_npy)

    fixed_noise = np.concatenate((id_npy, emo_npy, gen_npy), axis=2)
    fixed_noise = torch.from_numpy(fixed_noise).float()

    return fixed_noise

def train(args):

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    datasets = dataset(img_size, data_root=data_root, id_label_root=id_label_root, transform=True)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size,
                                             shuffle=True, num_workers=0, drop_last=True)

    # set models
    G, D = get_GAN(args)

    # criterion
    criterion = nn.BCELoss()

    # set real label and fake label
    real_label = 1
    fake_label = 0

    # set optimizer

    optimizerD = optim.Adam(D.parameters(), lr=args.lr1, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=args.lr1, betas=(0.5, 0.999))


    fixed_noise = fix_noise().to(device)



    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.num_epochs):
        loss_D_sum, loss_G_sum = 0, 0
        for i, data in enumerate(dataloader):

            ########################################################
            # 1. Update D network:
            ########################################################
            # 1.1 Train with all-real batch
                D.zero_grad()
                real_data = data[0].to(device)
                b_size = real_data.size(0)  # 128
                label = torch.full((b_size,), real_label, device=device)
                output, id_out_real, emo_out_real, gen_out_real = D(real_data)
                output = output.view(-1)
                label = label.to(torch.float32)
                loss_D_real = criterion(output, label)
                data[1] = (data[1])[:, 0, :].float().to(device)
                data[2] = (data[2])[:, 0, :].float().to(device)
                data[3] = (data[3])[:, 0, :].float().to(device)
                loss_D_real_id = criterion(id_out_real, data[1])
                loss_D_real_emo = criterion(emo_out_real, data[2])
                loss_D_real_gen = criterion(gen_out_real, data[3])
                loss_D_real = loss_D_real + lamda3 * loss_D_real_id + lamda1 * loss_D_real_emo + lamda1 * loss_D_real_gen
                loss_D_real.backward()
                D_x = output.mean().item()

                # 1.2 Train with all-fake batch
                noise = data[4].to(device).float().to(device)
                fake = G(noise)
                label.fill_(fake_label)

                output, id_out_fake, emo_out_fake, gen_out_fake  = D(fake.detach())
                output = output.view(-1)
                loss_D_fake = criterion(output, label)
                loss_D_fake_id = criterion(id_out_fake, data[1])
                loss_D_fake_emo = criterion(emo_out_fake, data[2])
                loss_D_fake_gen = criterion(gen_out_fake, data[3])
                loss_D_fake = loss_D_fake + lamda3 * loss_D_fake_id + lamda1 * loss_D_fake_emo + lamda1 * loss_D_fake_gen
                loss_D_fake.backward()
                D_G_z1 = output.mean().item()
                loss_D = (loss_D_real + loss_D_fake)
            # update D
                optimizerD.step()
            ########################################################
            # 2. Update G network:
            ########################################################

                optimizerG.zero_grad()
                label.fill_(real_label)
                output, id_out_fake, emo_out_fake, gen_out_fake  = D(fake)
                output = output.view(-1)
                loss_G = criterion(output, label)
                loss_G_dist = torch.mean(abs(real_data - fake))
                loss_G = loss_G + lamda2 * loss_G_dist
                loss_G.backward()
                optimizerG.step()
                iters += 1
                loss_D_sum = loss_D_sum + loss_D
                loss_G_sum = loss_G_sum + loss_G

        print("epoch[{}/{}]\t Loss_D:{}\tLoss_G:{}\t".format(
            epoch, args.num_epochs, loss_D_sum / len(dataloader), loss_G_sum / len(dataloader), ))

        writer.add_scalar("d_loss", loss_D_sum / len(dataloader), epoch)
        writer.add_scalar("g_loss", loss_G_sum / len(dataloader), epoch)


        with torch.no_grad():

            fake = G(fixed_noise).detach().to(device)

        g_out = vutils.make_grid(fake, nrow=7,padding=2, normalize=True)
        vutils.save_image(g_out, "/xxx/xxx/xxx/xxx/xxx.jpg".format(epoch, iters)) #generated images save address
        torch.save(G, '/xxx/xxx/xxx/xxx.pth'.format(epoch)) #Model weight save address





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN train')
    parser.add_argument('--num_epochs', default=2000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of train')
    parser.add_argument('--lr1', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--nc', default=1, type=int, help='channel of input')
    parser.add_argument('--nz', default=116, type=int, help='len of noise')
    parser.add_argument('--ngf', default=64, type=int, help='The size of the feature map in the generator')
    parser.add_argument('--ndf', default=64, type=int, help='The size of the feature map in the discriminator')
    args = parser.parse_args()
    train(args)