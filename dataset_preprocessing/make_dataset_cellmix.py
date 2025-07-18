import os
from sklearn.model_selection import train_test_split

def create_cellmix_simple_split():
    """CellMix用単純分割（動作確認用）"""
    
    # データ収集
    data_dir = '../datasets/CellMix/imagesTr'
    output_dir = '../lists/CellMix'
    
    # ファイル名からcase_nameを抽出
    case_names = []
    for file in os.listdir(data_dir):
        if file.endswith('_0000.png'):
            case_name = file.replace('_0000.png', '')
            case_names.append(case_name)
    
    print(f"Total cases: {len(case_names)}")
    print(f"Sample cases: {case_names[:5]}")
    
    # ラベルファイルの存在確認
    label_dir = '../datasets/CellMix/labelsTr'
    valid_cases = []
    for case in case_names:
        label_path = os.path.join(label_dir, f'{case}.png')
        if os.path.exists(label_path):
            valid_cases.append(case)
    
    print(f"Valid cases: {len(valid_cases)}")
    
    # Train/Val分割 (80%/20%)
    train_cases, val_cases = train_test_split(
        valid_cases, test_size=0.2, random_state=1234, shuffle=True
    )
    
    print(f"Train: {len(train_cases)}")
    print(f"Val: {len(val_cases)}")
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル保存
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for case in train_cases:
            f.write(f"{case}\n")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for case in val_cases:
            f.write(f"{case}\n")
    
    print(f"Files saved to: {output_dir}")

if __name__ == '__main__':
    create_cellmix_simple_split()