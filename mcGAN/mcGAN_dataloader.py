from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class dataset(Dataset):
    def __init__(self, image_size, data_root, id_label_root, transform = None):
        emo_label_root = id_label_root[:-3] + 'emo/'
        gen_label_root = id_label_root[:-3] + 'gen/'
        self.imgs = [os.path.join(data_root, img) for img in sorted(os.listdir(data_root))]
        self.label_id = [os.path.join(id_label_root, id) for id in sorted(os.listdir(id_label_root))]
        self.label_emo = [os.path.join(emo_label_root, emo) for emo in sorted(os.listdir(emo_label_root))]
        self.label_gen = [os.path.join(gen_label_root, gen) for gen in sorted(os.listdir(gen_label_root))]

        if transform:
            self.transforms = transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(0.5, 0.5),

            ])

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img)
        img = self.transforms(img)
        label_id = self.label_id[index]
        label_id = np.load(label_id)
        label_emo = self.label_emo[index]
        label_emo = np.load(label_emo)
        label_gen = self.label_gen[index]
        label_gen = np.load(label_gen)
        label_all = np.concatenate((label_id, label_emo, label_gen), axis=1)
        return img, label_id, label_emo, label_gen, label_all

    def __len__(self):
        return len(self.imgs)