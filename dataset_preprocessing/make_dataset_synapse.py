import os
from sklearn.model_selection import train_test_split

def create_synapse_split():
    # 参照用ディレクトリから読み込み（修正）
    input_file = '../lists/references/Synapse/train.txt'
    output_dir = '../lists/Synapse/'
    
    print(f"Reading from: {input_file}")
    
    # ファイル存在確認
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print(f"Please check if the file exists at: {os.path.abspath(input_file)}")
        return
    
    with open(input_file, 'r') as f:
        all_samples = [line.strip() for line in f.readlines()]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"First 5 samples: {all_samples[:5]}")
    
    # train/val分割（患者ベース分割を推奨）
    patients = {}
    for sample in all_samples:
        patient_id = sample.split('_slice')[0]  # case0031_slice000 → case0031
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(sample)
    
    patient_ids = list(patients.keys())
    print(f"Total patients: {len(patient_ids)}")
    print(f"Patients: {patient_ids}")
    
    # 患者ベースで分割（同一患者のスライスが学習と検証の両方に含まれることを防ぐ）
    train_patients, val_patients = train_test_split(
        patient_ids, 
        test_size=0.2, 
        random_state=1234, 
        shuffle=True
    )
    
    # スライス単位で再構築
    train_samples = []
    val_samples = []
    
    for patient in train_patients:
        train_samples.extend(patients[patient])
    
    for patient in val_patients:
        val_samples.extend(patients[patient])
    
    print(f"Train patients: {len(train_patients)} ({train_patients})")
    print(f"Val patients: {len(val_patients)} ({val_patients})")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # train.txt保存（スライス名のみ）
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for sample in train_samples:
            f.write(f"{sample}\n")
    
    # val.txt保存（スライス名のみ）
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for sample in val_samples:
            f.write(f"{sample}\n")
    
    print(f"Files saved to: {output_dir}")
    
    # 確認
    print("\nVerification:")
    print(f"Train file first 5 lines:")
    with open(os.path.join(output_dir, 'train.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
    
    print(f"Val file first 5 lines:")
    with open(os.path.join(output_dir, 'val.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
    
    # 最終確認
    print(f"\nFinal verification:")
    print(f"Train file: {os.path.join(output_dir, 'train.txt')} ({len(train_samples)} lines)")
    print(f"Val file: {os.path.join(output_dir, 'val.txt')} ({len(val_samples)} lines)")

if __name__ == '__main__':
    create_synapse_split()