import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def create_cellmix_kfold_split(k_folds=5):
    """CellMix K-fold交差検証用分割"""
    
    # データ収集
    data_dir = '../datasets/CellMix/imagesTr'
    output_base_dir = '../lists/CellMix_kfold'
    
    # 有効なケースの収集
    case_names = []
    for file in os.listdir(data_dir):
        if file.endswith('_0000.png'):
            case_name = file.replace('_0000.png', '')
            # ラベルファイルの存在確認
            label_path = os.path.join('../datasets/CellMix/labelsTr', f'{case_name}.png')
            if os.path.exists(label_path):
                case_names.append(case_name)
    
    print(f"Total valid cases: {len(case_names)}")
    
    # 全体の80%を交差検証用、20%をテスト用に分割
    train_val_cases, test_cases = train_test_split(
        case_names, test_size=0.2, random_state=1234, shuffle=True
    )
    
    print(f"Train/Val cases: {len(train_val_cases)}")
    print(f"Test cases: {len(test_cases)}")
    
    # K-fold分割
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=1234)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_cases)):
        print(f"\n=== Fold {fold + 1} ===")
        
        train_cases = [train_val_cases[i] for i in train_idx]
        val_cases = [train_val_cases[i] for i in val_idx]
        
        print(f"Train: {len(train_cases)}")
        print(f"Val: {len(val_cases)}")
        
        # フォルダ作成
        fold_dir = os.path.join(output_base_dir, f'fold_{fold + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # ファイル保存
        with open(os.path.join(fold_dir, 'train.txt'), 'w') as f:
            for case in train_cases:
                f.write(f"{case}\n")
        
        with open(os.path.join(fold_dir, 'val.txt'), 'w') as f:
            for case in val_cases:
                f.write(f"{case}\n")
        
        print(f"Saved to: {fold_dir}")
    
    # テストセット保存
    test_dir = os.path.join(output_base_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    with open(os.path.join(test_dir, 'test.txt'), 'w') as f:
        for case in test_cases:
            f.write(f"{case}\n")
    
    print(f"\nTest set saved to: {test_dir}")
    print(f"Summary: {k_folds} folds with {len(train_val_cases)} train/val + {len(test_cases)} test")

if __name__ == '__main__':
    create_cellmix_kfold_split(k_folds=5)