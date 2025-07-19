import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip_rgb(image, label):
    """RGB画像対応の回転・反転"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))  # RGB対応
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate_rgb(image, label):
    """RGB画像対応の回転"""
    angle = np.random.randint(-20, 20)
    # RGB各チャンネルを個別に回転
    rotated_image = np.zeros_like(image)
    for c in range(image.shape[2]):
        rotated_image[:, :, c] = ndimage.rotate(image[:, :, c], angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return rotated_image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # PIL ImageからNumPy arrayに変換
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(label, Image.Image):
            label = np.array(label)
        
        # データ拡張
        if random.random() > 0.5:
            image, label = random_rot_flip_rgb(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate_rgb(image, label)
        
        # リサイズ
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            # RGB画像のリサイズ
            resized_image = np.zeros((self.output_size[0], self.output_size[1], 3))
            for c in range(3):
                resized_image[:, :, c] = zoom(
                    image[:, :, c], 
                    (self.output_size[0] / h, self.output_size[1] / w), 
                    order=3
                )
            image = resized_image
            
            # ラベルのリサイズ
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        
        # 正規化とテンソル変換
        if image.max() > 1.0:
            image = image / 255.0
        
        # PyTorchテンソル形式 (C, H, W)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class CellMixDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        
        # ファイルリストの読み込み
        list_path = os.path.join(list_dir, self.split + '.txt')
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        
        # ✅ 修正：splitに応じてディレクトリを切り替え
        if self.split == 'train':
            # トレーニング用
            image_path = os.path.join(self.base_dir, 'imagesTr', f'{case_name}_0000.png')
            label_path = os.path.join(self.base_dir, 'labelsTr', f'{case_name}.png')
        elif self.split == 'test':
            # ✅ テスト用：imagesTs と labelsTs を使用
            image_path = os.path.join(self.base_dir, 'imagesTs', f'{case_name}_0000.png')
            label_path = os.path.join(self.base_dir, 'labelsTs', f'{case_name}.png')
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # ファイル存在確認
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")
        
        # 画像読み込み
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        sample['case_name'] = case_name
        return sample