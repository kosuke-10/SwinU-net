import os
from PIL import Image
import numpy as np

def analyze_cellmix_dataset():
    # パス設定
    image_dir = '../datasets/CellMix/imagesTr'
    label_dir = '../datasets/CellMix/labelsTr'
    
    # ファイル名の対応確認
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    
    print(f'=== ファイル名対応分析 ===')
    print(f'Total images: {len(image_files)}')
    print(f'Total labels: {len(label_files)}')
    print(f'Image example: {image_files[0]}')
    print(f'Label example: {label_files[0]}')
    
    # 対応関係の確認
    if image_files and label_files:
        sample_image_file = image_files[0]  # cell_001_0000.png
        sample_label_file = label_files[0]  # cell_001.png
        
        img = Image.open(os.path.join(image_dir, sample_image_file))
        lbl = Image.open(os.path.join(label_dir, sample_label_file))
        
        print(f'\n=== 画像情報 ===')
        print(f'Image size: {img.size}')
        print(f'Image mode: {img.mode}')
        print(f'Label size: {lbl.size}')
        print(f'Label mode: {lbl.mode}')
        
        # ラベルの値の範囲確認
        lbl_array = np.array(lbl)
        unique_values = np.unique(lbl_array)
        print(f'\n=== ラベル値分析 ===')
        print(f'Label shape: {lbl_array.shape}')
        print(f'Label values: min={lbl_array.min()}, max={lbl_array.max()}')
        print(f'Unique label values: {unique_values}')
        print(f'Number of classes: {len(unique_values)}')
        
        # クラス分布の確認
        print(f'\n=== クラス分布 ===')
        for val in unique_values:
            count = np.sum(lbl_array == val)
            percentage = (count / lbl_array.size) * 100
            print(f'Class {val}: {count} pixels ({percentage:.2f}%)')
        
        # ファイル名対応の確認
        print(f'\n=== ファイル名マッピング確認 ===')
        for i in range(min(10, len(image_files))):
            img_name = image_files[i]
            expected_label = img_name.replace('_0000.png', '.png')
            actual_label = label_files[i] if i < len(label_files) else 'NOT FOUND'
            match = expected_label == actual_label
            print(f'{img_name} -> {expected_label} | Actual: {actual_label} | Match: {match}')
        
        # 複数サンプルのラベル値確認
        print(f'\n=== 複数サンプルのラベル値確認 ===')
        for i in range(min(5, len(label_files))):
            lbl_file = label_files[i]
            lbl_path = os.path.join(label_dir, lbl_file)
            lbl_img = Image.open(lbl_path)
            lbl_arr = np.array(lbl_img)
            unique_vals = np.unique(lbl_arr)
            print(f'{lbl_file}: {unique_vals}')

if __name__ == '__main__':
    analyze_cellmix_dataset()