import os
from pathlib import Path

def create_cellmix_test_list():
    """CellMix テストデータのリストファイル作成"""
    
    # パス設定
    project_root = Path(__file__).parent.parent
    images_test_dir = project_root / "datasets" / "CellMix" / "imagesTs"
    labels_test_dir = project_root / "datasets" / "CellMix" / "labelsTs"
    lists_dir = project_root / "lists" / "CellMix"
    
    # ディレクトリ確認
    if not images_test_dir.exists():
        print(f"❌ Test images directory not found: {images_test_dir}")
        return
    
    if not labels_test_dir.exists():
        print(f"❌ Test labels directory not found: {labels_test_dir}")
        return
    
    # lists/CellMix ディレクトリ確保
    lists_dir.mkdir(parents=True, exist_ok=True)
    
    # テスト画像ファイル一覧取得
    test_image_files = sorted([
        f for f in images_test_dir.iterdir() 
        if f.suffix.lower() == '.png' and f.name.startswith('cell_test_')
    ])
    
    print(f"📁 Found {len(test_image_files)} test images")
    print(f"📂 Images directory: {images_test_dir}")
    print(f"📂 Labels directory: {labels_test_dir}")
    
    # ファイル名パターン分析
    print(f"\n📋 File naming analysis:")
    sample_files = test_image_files[:5]
    for f in sample_files:
        print(f"   {f.name}")
    
    # test.txt 作成用のケース名抽出
    test_cases = []
    
    for img_file in test_image_files:
        filename = img_file.name
        
        # cell_test_001_0000.png -> cell_test_001
        if filename.endswith('_0000.png'):
            case_name = filename.replace('_0000.png', '')
            test_cases.append(case_name)
    
    # ソート
    test_cases = sorted(test_cases)
    
    print(f"\n📊 Extracted {len(test_cases)} test cases:")
    for case in test_cases[:10]:
        print(f"   {case}")
    if len(test_cases) > 10:
        print(f"   ... and {len(test_cases) - 10} more")
    
    # test.txt ファイル作成
    test_txt_path = lists_dir / "test.txt"
    
    with open(test_txt_path, 'w') as f:
        for case in test_cases:
            f.write(f"{case}\n")
    
    print(f"\n✅ Created test.txt: {test_txt_path}")
    print(f"📊 Total test cases: {len(test_cases)}")
    
    # 対応するラベルファイルの確認
    print(f"\n🔍 Checking corresponding label files:")
    missing_labels = []
    existing_labels = []
    
    for case in test_cases[:5]:  # 最初の5ケースのみチェック
        # 想定されるラベルファイル名: cell_test_001.png
        label_name = f"{case}.png"
        label_path = labels_test_dir / label_name
        
        if label_path.exists():
            existing_labels.append(label_name)
            print(f"   ✅ {case} -> {label_name}")
        else:
            missing_labels.append(case)
            print(f"   ❌ {case} -> {label_name} not found")
    
    # 次のステップ案内
    print(f"\n💡 Next steps:")
    print(f"1. Verify test.txt content:")
    print(f"   head -10 {test_txt_path}")
    print(f"2. Test dataset loading:")
    print(f"   python3 -c \"from datasets.dataset_cellmix import CellMixDataset; d=CellMixDataset('datasets/CellMix', 'lists/CellMix', 'test'); print(f'Test samples: {{len(d)}}')\"")
    
    return test_txt_path, test_cases

if __name__ == '__main__':
    create_cellmix_test_list()