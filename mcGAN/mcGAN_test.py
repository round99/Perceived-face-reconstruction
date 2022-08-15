import warnings
warnings.filterwarnings("ignore")

import torch
import os
import numpy as np
import torchvision.utils as vutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pthfile = '/xxx/xxx/xxx/xxx.pth' #Model weight address
G = torch.load(pthfile).to(device)
id_path = '/xxx/xxx/xxx/xxx/' #Id label address
emo_path = '/xxx/xxx/xxx/xxx/' #Expression label address
gen_path = '/xxx/xxx/xxx/xxx/' #Gender label address
id_npys = os.listdir(id_path)
id_npys.sort()
for id_npy in id_npys:
    id_input = np.load(id_path + id_npy)
    emo_input = np.load(emo_path + id_npy[:7] +'.npy')
    gen_input = np.load(gen_path + id_npy[:7] +'.npy')
    all_condition = np.concatenate((id_input, emo_input, gen_input), axis=1)
    all_condition = torch.from_numpy(all_condition).float().to(device)
    fake = G(all_condition)
    g_out = vutils.make_grid(fake, padding=0, normalize=True)
    vutils.save_image(g_out, '/xxx/xxx/xxx/xxx.jpg') #Image save address
