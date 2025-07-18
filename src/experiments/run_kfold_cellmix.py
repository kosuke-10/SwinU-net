import os
import subprocess
import sys
from datetime import datetime

# パス修正: プロジェクトルートからの相対パス
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_experiment_dir(base_name="cellmix_kfold"):
    """実験用ディレクトリの作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(PROJECT_ROOT, "experiments", f"{base_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"📁 Experiment directory: {exp_dir}")
    return exp_dir

def save_experiment_config(exp_dir, args):
    """実験設定の保存"""
    config_path = os.path.join(exp_dir, "experiment_config.txt")
    with open(config_path, 'w') as f:
        f.write("K-fold Cross Validation Experiment\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: CellMix\n")
        f.write(f"K-folds: {args.get('k_folds', 5)}\n")
        f.write(f"Max epochs: {args.get('max_epochs', 150)}\n")
        f.write(f"Batch size: {args.get('batch_size', 24)}\n")
        f.write(f"Base LR: {args.get('base_lr', 0.05)}\n")
        f.write(f"Config: configs/swin_tiny_patch4_window7_224_lite.yaml\n")


def run_single_fold(fold_num, exp_dir, max_epochs=150, batch_size=24, base_lr=0.05):
    """単一foldの学習実行（パス修正版）"""
    
    print(f"\n{'='*60}")
    print(f"Starting Fold {fold_num}")
    print(f"{'='*60}")
    
    # foldごとの出力ディレクトリ
    fold_output_dir = os.path.join(exp_dir, f"fold_{fold_num}")
    
    # プロジェクトルートに移動して実行
    cmd = [
        'python3', 'src/train.py',  # ✅ src/に移動したtrain.pyを指定
        '--dataset', 'CellMix',
        '--cfg', 'configs/swin_tiny_patch4_window7_224_lite.yaml',
        '--root_path', 'datasets/CellMix',
        '--list_dir', f'lists/CellMix_kfold/fold_{fold_num}',
        '--num_classes', '3',
        '--n_class', '3',
        '--max_epochs', str(max_epochs),
        '--output_dir', fold_output_dir,
        '--img_size', '224',
        '--base_lr', str(base_lr),
        '--batch_size', str(batch_size)
    ]
    
    print(f"📁 Output: {fold_output_dir}")
    print(f"📁 Working directory: {PROJECT_ROOT}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # プロジェクトルートで実行
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"✅ Fold {fold_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in Fold {fold_num}: {e}")
        return False

def run_all_folds(k_folds=5, max_epochs=150, batch_size=24, base_lr=0.05):
    """全foldの学習実行（整理された出力）"""
    
    # 実験ディレクトリ作成
    exp_dir = create_experiment_dir("cellmix_kfold")
    
    # 実験設定保存
    exp_config = {
        'k_folds': k_folds,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'base_lr': base_lr
    }
    save_experiment_config(exp_dir, exp_config)
    
    successful_folds = []
    failed_folds = []
    
    print(f"🚀 Starting {k_folds}-fold cross validation")
    print(f"📁 Results will be saved to: {exp_dir}")
    
    for fold in range(1, k_folds + 1):
        success = run_single_fold(fold, exp_dir, max_epochs, batch_size, base_lr)
        if success:
            successful_folds.append(fold)
        else:
            failed_folds.append(fold)
    
    # 結果サマリー保存
    summary_path = os.path.join(exp_dir, "kfold_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("K-fold Cross Validation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total folds: {k_folds}\n")
        f.write(f"Successful folds: {successful_folds}\n")
        f.write(f"Failed folds: {failed_folds}\n")
        f.write(f"Success rate: {len(successful_folds)}/{k_folds}\n")
    
    print(f"\n{'='*60}")
    print("K-fold Training Summary")
    print(f"{'='*60}")
    print(f"📁 Experiment directory: {exp_dir}")
    print(f"✅ Successful folds: {successful_folds}")
    print(f"❌ Failed folds: {failed_folds}")
    print(f"📊 Success rate: {len(successful_folds)}/{k_folds}")
    print(f"📄 Summary saved: {summary_path}")
    
    return exp_dir

def run_specific_fold(fold_num, max_epochs=150, batch_size=24, base_lr=0.05):
    """特定foldのみ実行（既存実験ディレクトリ使用または新規作成）"""
    
    # 最新の実験ディレクトリを検索
    experiments_dir = "experiments"
    if os.path.exists(experiments_dir):
        exp_dirs = [d for d in os.listdir(experiments_dir) 
                   if d.startswith("cellmix_kfold_") and os.path.isdir(os.path.join(experiments_dir, d))]
        if exp_dirs:
            # 最新の実験ディレクトリを使用
            latest_exp = sorted(exp_dirs)[-1]
            exp_dir = os.path.join(experiments_dir, latest_exp)
            print(f"📁 Using existing experiment: {exp_dir}")
        else:
            # 新規実験ディレクトリ作成
            exp_dir = create_experiment_dir("cellmix_kfold")
            exp_config = {
                'k_folds': 5,
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'base_lr': base_lr
            }
            save_experiment_config(exp_dir, exp_config)
    else:
        # 新規実験ディレクトリ作成
        exp_dir = create_experiment_dir("cellmix_kfold")
        exp_config = {
            'k_folds': 5,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'base_lr': base_lr
        }
        save_experiment_config(exp_dir, exp_config)
    
    # 指定foldを実行
    success = run_single_fold(fold_num, exp_dir, max_epochs, batch_size, base_lr)
    
    if success:
        print(f"✅ Fold {fold_num} completed in {exp_dir}")
    else:
        print(f"❌ Fold {fold_num} failed")
    
    return exp_dir

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='K-fold Cross Validation for CellMix')
    parser.add_argument('--fold', type=int, help='Run specific fold (1-5)')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--max_epochs', type=int, default=150, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=0.05, help='Base learning rate')
    
    args = parser.parse_args()
    
    if args.fold:
        # 特定のfoldのみ実行
        exp_dir = run_specific_fold(args.fold, args.max_epochs, args.batch_size, args.base_lr)
    else:
        # 全fold実行
        exp_dir = run_all_folds(args.k_folds, args.max_epochs, args.batch_size, args.base_lr)
    
    print(f"\n🎯 Final results location: {exp_dir}")