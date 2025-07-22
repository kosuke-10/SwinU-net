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
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate_rgb(image, label):
    """RGB画像対応の回転 (角度範囲拡大)"""
    angle = np.random.randint(-30, 30)
    rotated_image = np.zeros_like(image)
    for c in range(image.shape[2]):
        rotated_image[:, :, c] = ndimage.rotate(image[:, :, c], angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return rotated_image, label


def random_scale_rgb(image, label):
    """スケール変換 (完全修正版) - PIL使用"""
    from PIL import Image as PILImage
    
    # スケール倍率を生成
    scale = np.random.uniform(0.8, 1.25)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    try:
        # PILでリサイズ（より安定）
        image_pil = PILImage.fromarray((image * 255).astype(np.uint8))
        label_pil = PILImage.fromarray(label.astype(np.uint8))
        
        # リサイズ実行
        image_resized = image_pil.resize((new_w, new_h), PILImage.BILINEAR)
        label_resized = label_pil.resize((new_w, new_h), PILImage.NEAREST)
        
        # NumPy配列に戻す
        image_array = np.array(image_resized) / 255.0
        label_array = np.array(label_resized)
        
        return image_array, label_array
        
    except Exception as e:
        # エラー時は元画像を返す
        print(f"Warning: Scale transformation failed: {e}")
        return image, label


def random_gamma_correction(image):
    """ガンマ補正 (nnU-Net準拠)"""
    gamma = np.random.uniform(0.8, 1.25)
    image_clipped = np.clip(image, 0.001, 1.0)
    return np.power(image_clipped, gamma)


def random_gaussian_noise(image):
    """ガウシアンノイズ"""
    noise_std = np.random.uniform(0, 0.03)
    noise = np.random.normal(0, noise_std, image.shape)
    return np.clip(image + noise, 0, 1)


def random_brightness_contrast(image):
    """明度・コントラスト調整"""
    # 明度調整
    brightness = np.random.uniform(-0.05, 0.05)
    image = image + brightness
    
    # コントラスト調整  
    contrast = np.random.uniform(0.95, 1.05)
    image = (image - 0.5) * contrast + 0.5
    
    return np.clip(image, 0, 1)


def random_gaussian_blur(image):
    """ガウシアンぼかし"""
    from scipy.ndimage import gaussian_filter
    
    sigma = np.random.uniform(0.5, 1.5)
    blurred = np.zeros_like(image)
    for c in range(3):
        blurred[:, :, c] = gaussian_filter(image[:, :, c], sigma)
    return blurred


def random_elastic_deformation(image, label):
    """弾性変形 (簡単版)"""
    try:
        from scipy.ndimage import map_coordinates, gaussian_filter
        
        # 変形強度を控えめに設定
        alpha = np.random.uniform(0, 30)  # さらに小さく
        sigma = np.random.uniform(8, 12)
        
        shape = image.shape[:2]
        
        # 変位フィールド生成
        dx = gaussian_filter((np.random.random(shape) - 0.5), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.random(shape) - 0.5), sigma, mode="constant", cval=0) * alpha
        
        # 座標グリッド作成
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # 変形適用
        deformed_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            deformed_image[:, :, c] = map_coordinates(
                image[:, :, c], indices, order=1, mode='reflect'
            ).reshape(shape)
        
        deformed_label = map_coordinates(
            label, indices, order=0, mode='reflect'
        ).reshape(shape)
        
        return deformed_image, deformed_label
        
    except Exception as e:
        print(f"Warning: Elastic deformation failed: {e}")
        return image, label


class RandomGenerator(object):
    def __init__(self, output_size, augmentation_mode='standard'):
        """
        Args:
            output_size: 出力サイズ [H, W]
            augmentation_mode: 'standard' | 'enhanced' | 'nnunet'
        """
        self.output_size = output_size
        self.augmentation_mode = augmentation_mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # PIL ImageからNumPy arrayに変換
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(label, Image.Image):
            label = np.array(label)
        
        # 正規化 (0-1範囲)
        if image.max() > 1.0:
            image = image / 255.0
        
        # データ拡張の適用（エラーハンドリング付き）
        try:
            if self.augmentation_mode == 'standard':
                image, label = self._apply_standard_augmentation(image, label)
            elif self.augmentation_mode == 'enhanced':
                image, label = self._apply_enhanced_augmentation(image, label)
            elif self.augmentation_mode == 'nnunet':
                image, label = self._apply_nnunet_augmentation(image, label)
        except Exception as e:
            print(f"Warning: Augmentation failed, using standard mode: {e}")
            image, label = self._apply_standard_augmentation(image, label)
        
        # リサイズ処理
        image, label = self._resize_safe(image, label)
        
        # テンソル変換
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample
    
    def _apply_standard_augmentation(self, image, label):
        """従来のSwin-Unet拡張"""
        if random.random() > 0.5:
            image, label = random_rot_flip_rgb(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate_rgb(image, label)
        return image, label
    
    def _apply_enhanced_augmentation(self, image, label):
        """改良版拡張 (中程度)"""
        # 基本的な幾何学変換
        if random.random() > 0.4:
            if random.random() > 0.5:
                image, label = random_rot_flip_rgb(image, label)
            else:
                image, label = random_rotate_rgb(image, label)
        
        # スケール変換（確率を下げる）
        if random.random() > 0.9:  # 10%の確率
            image, label = random_scale_rgb(image, label)
        
        # 画像品質変換
        if random.random() > 0.7:
            image = random_gamma_correction(image)
        if random.random() > 0.8:
            image = random_brightness_contrast(image)
        if random.random() > 0.9:
            image = random_gaussian_noise(image)
        
        return image, label
    
    def _apply_nnunet_augmentation(self, image, label):
        """nnU-Net準拠拡張 (安全版)"""
        if random.random() > 0.2:  # 80%の確率で変換適用
            # 幾何学的変換
            if random.random() > 0.3:
                image, label = random_rot_flip_rgb(image, label)
            if random.random() > 0.5:
                image, label = random_rotate_rgb(image, label)
            
            # スケール変換（確率をさらに下げる）
            if random.random() > 0.9:  # 10%の確率
                image, label = random_scale_rgb(image, label)
            
            # 弾性変形（確率を大幅に下げる）
            if random.random() > 0.95:  # 5%の確率
                image, label = random_elastic_deformation(image, label)
            
            # 画像品質変換
            if random.random() > 0.4:
                image = random_gamma_correction(image)
            if random.random() > 0.6:
                image = random_gaussian_noise(image)
            if random.random() > 0.6:
                image = random_brightness_contrast(image)
            if random.random() > 0.8:
                image = random_gaussian_blur(image)
        
        return image, label
    
    def _resize_safe(self, image, label):
        """安全なリサイズ処理（PIL使用）"""
        from PIL import Image as PILImage
        
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            try:
                # PILでリサイズ（最も安定）
                image_pil = PILImage.fromarray((image * 255).astype(np.uint8))
                label_pil = PILImage.fromarray(label.astype(np.uint8))
                
                # リサイズ実行
                image_resized = image_pil.resize((self.output_size[1], self.output_size[0]), PILImage.BILINEAR)
                label_resized = label_pil.resize((self.output_size[1], self.output_size[0]), PILImage.NEAREST)
                
                # NumPy配列に戻す
                image = np.array(image_resized) / 255.0
                label = np.array(label_resized)
                
            except Exception as e:
                print(f"Warning: Resize failed: {e}")
                # フォールバック: 中央クロップまたはゼロパディング
                image, label = self._fallback_resize(image, label)
        
        return image, label
    
    def _fallback_resize(self, image, label):
        """フォールバックリサイズ（中央クロップ/ゼロパディング）"""
        h, w = image.shape[:2]
        target_h, target_w = self.output_size
        
        # 新しい配列を初期化
        new_image = np.zeros((target_h, target_w, 3))
        new_label = np.zeros((target_h, target_w))
        
        # コピー領域を計算
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        
        # 中央配置
        start_h = (target_h - copy_h) // 2
        start_w = (target_w - copy_w) // 2
        
        img_start_h = (h - copy_h) // 2
        img_start_w = (w - copy_w) // 2
        
        # データをコピー
        new_image[start_h:start_h+copy_h, start_w:start_w+copy_w] = \
            image[img_start_h:img_start_h+copy_h, img_start_w:img_start_w+copy_w]
        new_label[start_h:start_h+copy_h, start_w:start_w+copy_w] = \
            label[img_start_h:img_start_h+copy_h, img_start_w:img_start_w+copy_w]
        
        return new_image, new_label


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
        
        # splitに応じてディレクトリを切り替え
        if self.split == 'train':
            image_path = os.path.join(self.base_dir, 'imagesTr', f'{case_name}_0000.png')
            label_path = os.path.join(self.base_dir, 'labelsTr', f'{case_name}.png')
        elif self.split == 'val':
            image_path = os.path.join(self.base_dir, 'imagesTr', f'{case_name}_0000.png')
            label_path = os.path.join(self.base_dir, 'labelsTr', f'{case_name}.png')
        elif self.split == 'test':
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