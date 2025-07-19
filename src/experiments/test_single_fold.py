import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_single_fold(fold_num, experiment_dir, dataset='CellMix'):
    """単一foldのテスト実行"""
    
    fold_dir = Path(experiment_dir) / f"fold_{fold_num}"
    model_path = fold_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    # テスト結果保存ディレクトリ
    test_output_dir = fold_dir / "test_results"
    test_output_dir.mkdir(exist_ok=True)
    
    # テスト実行コマンド
    cmd = [
        'python3', 'test.py',
        '--dataset', dataset,
        '--cfg', 'configs/swin_tiny_patch4_window7_224_lite.yaml',
        '--is_savenii', 'True',
        '--volume_path', str(test_output_dir / 'predictions'),
        '--test_save_dir', str(test_output_dir),
        '--snapshot', str(model_path),
        '--num_classes', '3',
        '--n_class', '3'
    ]
    
    print(f"🧪 Testing Fold {fold_num}...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Output: {test_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Fold {fold_num} test completed")
        
        # 結果ログ保存
        with open(test_output_dir / 'test_log.txt', 'w') as f:
            f.write(f"Fold {fold_num} Test Results\n")
            f.write("="*40 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Fold {fold_num} test failed: {e}")
        return False

def test_all_folds(experiment_dir, dataset='CellMix'):
    """全foldのテスト実行"""
    
    successful_tests = []
    failed_tests = []
    
    for fold in range(1, 6):
        success = test_single_fold(fold, experiment_dir, dataset)
        if success:
            successful_tests.append(fold)
        else:
            failed_tests.append(fold)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {failed_tests}")
    
    return successful_tests, failed_tests

def main():
    parser = argparse.ArgumentParser(description='Test K-fold Models')
    parser.add_argument('--fold', type=int, help='Test specific fold (1-5)')
    parser.add_argument('--experiment_dir', type=str, 
                       help='K-fold experiment directory')
    parser.add_argument('--dataset', type=str, default='CellMix',
                       help='Dataset name')
    
    args = parser.parse_args()
    
    if not args.experiment_dir:
        # 最新の実験ディレクトリを検索
        import glob
        kfold_dirs = glob.glob("experiments/cellmix_kfold_*")
        if kfold_dirs:
            args.experiment_dir = sorted(kfold_dirs)[-1]
            print(f"📁 Using latest experiment: {args.experiment_dir}")
        else:
            print("❌ No experiment directory found")
            return
    
    # プロジェクトルートに移動
    os.chdir(Path(__file__).parent.parent.parent)
    
    if args.fold:
        test_single_fold(args.fold, args.experiment_dir, args.dataset)
    else:
        test_all_folds(args.experiment_dir, args.dataset)

if __name__ == '__main__':
    main()