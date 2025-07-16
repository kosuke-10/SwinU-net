import os
from sklearn.model_selection import train_test_split

def create_synapse_split_slice_based():
    """スライスベース分割（論文再現用）"""
    # 参照用ディレクトリから読み込み
    input_file = '../lists/references/Synapse/train.txt'
    output_dir = '../lists/Synapse_slice_based/'
    
    print(f"Reading from: {input_file}")
    
    # ファイル存在確認
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    with open(input_file, 'r') as f:
        all_samples = [line.strip() for line in f.readlines()]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"First 5 samples: {all_samples[:5]}")
    
    # スライスベースで直接分割（論文と同様）
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=0.2,
        random_state=1234,  # 論文と同じシードを使用
        shuffle=True
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # 患者分布の確認
    train_patients = set([sample.split('_slice')[0] for sample in train_samples])
    val_patients = set([sample.split('_slice')[0] for sample in val_samples])
    overlap_patients = train_patients & val_patients
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Overlapping patients: {len(overlap_patients)} ({overlap_patients})")
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # train.txt保存
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for sample in train_samples:
            f.write(f"{sample}\n")
    
    # val.txt保存
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for sample in val_samples:
            f.write(f"{sample}\n")
    
    print(f"Files saved to: {output_dir}")
    print(f"Train file: {os.path.join(output_dir, 'train.txt')} ({len(train_samples)} lines)")
    print(f"Val file: {os.path.join(output_dir, 'val.txt')} ({len(val_samples)} lines)")

def create_synapse_split_patient_based():
    """患者ベース分割（現在の方式、より厳密）"""
    # 既存のコード...

if __name__ == '__main__':
    print("=== スライスベース分割（論文再現用）===")
    create_synapse_split_slice_based()
    
    print("\n=== 患者ベース分割（厳密評価用）===")
    create_synapse_split_patient_based()