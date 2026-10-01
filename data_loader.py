import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt


class CelebAMaskHQ(Dataset):
    def __init__(self, img_path, label_path, image_size, is_train=False):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size  # 必须在这里接收 imsize
        self.is_train = is_train
        self.dataset = []

        self.flip_map = {
            4: 5, 5: 4,  # eyebrow l r
            6: 7, 7: 6,  # eye l r
            8: 9, 9: 8  # ear l r
        }
        self.preprocess()

    def preprocess(self):
        filenames = [f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')]
        for name in filenames:
            img_path = os.path.join(self.img_path, name).replace('\\', '/')
            label_name = name.replace('.jpg', '.png')
            label_path = os.path.join(self.label_path, label_name).replace('\\', '/')
            if os.path.exists(label_path):
                self.dataset.append([img_path, label_path])
        # print(f'Finished preprocessing. Total images: {len(self.dataset)}')

    def __getitem__(self, index):
        img_path, label_path = self.dataset[index]
        img = Image.open(img_path)
        label = Image.open(label_path)

        if self.is_train:
            # 1. 预先确定几何变换参数，确保 img 和 label 完全一致
            angle = random.uniform(-10, 10)

            flip = random.random() > 0.5

            # 2. 对 img 进行变换
            img = transforms.functional.resize(img, (self.imsize, self.imsize))
            if flip:
                img = transforms.functional.hflip(img)
            img = transforms.functional.rotate(img, angle)
            # 颜色抖动 (仅限图像)
            img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)(img)

            # 3. 对 label 进行变换 (必须使用 NEAREST)
            label = transforms.functional.resize(label, (self.imsize, self.imsize),
                                                 interpolation=transforms.InterpolationMode.NEAREST)
            if flip:
                label = transforms.functional.hflip(label)
                # flip labels with left and right again
                label_np = np.array(label, copy=True)
                new_label_np = label_np.copy()
                for left_idx, right_idx in self.flip_map.items():
                    mask_left = (label_np == left_idx)
                    mask_right = (label_np == right_idx)
                    new_label_np[mask_left] = right_idx
                    new_label_np[mask_right] = left_idx
                label = Image.fromarray(new_label_np)

            label = transforms.functional.rotate(label, angle,interpolation=transforms.InterpolationMode.NEAREST)

            # 4. 转为 Tensor
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
            # 直接转 Long，避免乘以 255 的浮点误差
            label = torch.from_numpy(np.array(label, copy=True)).long()
        else:
            # 验证集逻辑
            img = transforms.functional.resize(img, (self.imsize, self.imsize))
            label = transforms.functional.resize(label, (self.imsize, self.imsize),
                                                 interpolation=transforms.InterpolationMode.NEAREST)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
            label = torch.from_numpy(np.array(label, copy=True)).long()

        return img, label

    def __len__(self):
        return len(self.dataset)

class Data_Loader():
    def __init__(self, img_path, label_path, sample_path ,image_size, batch_size, rare_list=None, mode='train'):
        self.img_path = img_path
        self.sample_path=sample_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.rare_list = rare_list
        self.mode = mode  # 'train' or 'test'

    def check_transforms(self, train_loader, val_loader):
        plt.switch_backend('Agg')
        train_imgs, train_labels = next(iter(train_loader))
        val_imgs, val_labels = next(iter(val_loader))
        train_labels = (train_labels.squeeze(1) * 255.0).long()
        val_labels = (val_labels.squeeze(1) * 255.0).long()
        def denorm(x):
            return x * 0.5 + 0.5
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].set_title("Train Image (Augmented)")
        axes[0, 0].imshow(denorm(train_imgs[0]).permute(1, 2, 0).cpu().numpy())
        axes[1, 0].set_title("Train Label (Synchronized)")
        axes[1, 0].imshow(train_labels[0].cpu().numpy(), cmap='tab20')
        axes[0, 1].set_title("Val Image (Clean)")
        axes[0, 1].imshow(denorm(val_imgs[0]).permute(1, 2, 0).cpu().numpy())
        axes[1, 1].set_title("Val Label (Clean)")
        axes[1, 1].imshow(val_labels[0].cpu().numpy(), cmap='tab20')
        # 5. 保存结果
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        save_path = os.path.join(self.sample_path, 'data_check_full.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"sampled data saved at: {save_path}")

    def loader(self):
        if self.mode == 'train':
            train_set = CelebAMaskHQ(self.img_path, self.label_path,self.imsize, is_train=True)
            val_set = CelebAMaskHQ(self.img_path, self.label_path, self.imsize, is_train=False)

            # 8:2 train val
            num_data = len(train_set)
            indices = torch.arange(num_data)
            train_idx, val_idx = random_split(indices, [900, 100], generator=torch.Generator().manual_seed(42))

            # over sampler

            train_weights = []
            if self.rare_list!=None:
                rare_set = set(self.rare_list)  # 转为 set 提高查找速度
            else:
                rare_set=()

            for idx in train_idx:
                img_path, _ = train_set.dataset[idx]
                if img_path in rare_set:
                    train_weights.append(2.0)  # 稀有样本权重设为 2
                else:
                    train_weights.append(1.0)  # 普通样本权重设为 1

            sampler = WeightedRandomSampler(
                weights=train_weights,
                num_samples=len(train_idx),
                replacement=True
            )

            # 3. use subset to get the correct transform
            train_loader = DataLoader(Subset(train_set, train_idx), batch_size=self.batch, sampler=sampler, drop_last=True)
            val_loader = DataLoader(Subset(val_set, val_idx), batch_size=self.batch, shuffle=False)

            # just for debug, do not enable this during training! it may cause OOM
            # self.check_transforms(train_loader, val_loader)

            return train_loader, val_loader

        else:
            # test set
            test_set = CelebAMaskHQ(self.img_path, self.label_path, is_train=False)
            test_loader = DataLoader(test_set, batch_size=self.batch, shuffle=False)
            return test_loader