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
        f.write(f"Augmentation mode: {args.get('augmentation_mode', 'nnunet')}\n")
        f.write(f"Eval interval: {args.get('eval_interval', 1)}\n")  # 🆕 追加
        f.write(f"Config: configs/swin_tiny_patch4_window7_224_lite.yaml\n")


def get_eval_interval(max_epochs):
    """エポック数に応じた適切な検証間隔を計算 - 毎エポック検証推奨"""
    # 🔧 基本的に毎エポック検証を推奨（メモリが許す限り）
    if max_epochs <= 100:
        return 1      # 100エポック以下: 毎エポック検証
    elif max_epochs <= 500:
        return 1      # 500エポック以下: 毎エポック検証（推奨）
    else:
        # 超長期学習の場合のみ間隔を開ける
        return max(1, max_epochs // 100)  # 最低1、最大でも100回程度の検証


def run_single_fold(fold_num, exp_dir, max_epochs=150, batch_size=24, base_lr=0.05, augmentation_mode='standard', eval_interval=None):
    """単一foldの学習実行（毎エポック検証対応版）"""
    
    print(f"\n{'='*60}")
    print(f"Starting Fold {fold_num} (Augmentation: {augmentation_mode})")
    print(f"{'='*60}")
    
    # foldごとの出力ディレクトリ
    fold_output_dir = os.path.join(exp_dir, f"fold_{fold_num}")
    
    # 🆕 augmentation_modeを環境変数で渡す
    env = os.environ.copy()
    env['AUGMENTATION_MODE'] = augmentation_mode
    
    # 🔧 eval_intervalの決定（毎エポック検証を優先）
    if eval_interval is None:
        eval_interval = 1  # 🔧 デフォルトで毎エポック検証
    
    expected_validations = max_epochs if eval_interval == 1 else (max_epochs // eval_interval + 1)
    
    cmd = [
        'python3', 'src/train.py',
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
        '--batch_size', str(batch_size),
        '--eval_interval', str(eval_interval)  # 🔧 毎エポック検証
    ]
    
    print(f"🎨 Data Augmentation mode: {augmentation_mode}")
    print(f"📁 Output: {fold_output_dir}")
    if eval_interval == 1:
        print(f"📊 Validation: every epoch ({expected_validations} times total)")
    else:
        print(f"📊 Validation: every {eval_interval} epochs (~{max_epochs // eval_interval} times total)")
    print(f"📊 Progress plot will be saved to: {fold_output_dir}/progress.png")
    
    try:
        # 環境変数付きで実行
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
        print(f"✅ Fold {fold_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in Fold {fold_num}: {e}")
        return False


def run_all_folds(k_folds=5, max_epochs=150, batch_size=24, base_lr=0.05, augmentation_mode='nnunet', eval_interval=None):
    """全foldの学習実行（毎エポック検証対応版）"""
    
    exp_dir = create_experiment_dir("cellmix_kfold")
    
    # 🔧 eval_intervalの決定（毎エポック検証を優先）
    if eval_interval is None:
        eval_interval = 1  # 🔧 デフォルトで毎エポック検証
    
    exp_config = {
        'k_folds': k_folds,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'base_lr': base_lr,
        'augmentation_mode': augmentation_mode,
        'eval_interval': eval_interval  # 🆕 追加
    }
    save_experiment_config(exp_dir, exp_config)
    
    print(f"🚀 Starting {k_folds}-fold cross validation")
    print(f"🎨 Data Augmentation: {augmentation_mode}")
    if eval_interval == 1:
        print(f"📊 Validation: every epoch ({max_epochs} times per fold)")
    else:
        print(f"📊 Validation: every {eval_interval} epochs")
    print(f"📁 Results will be saved to: {exp_dir}")
    
    successful_folds = []
    failed_folds = []
    
    for fold in range(1, k_folds + 1):
        success = run_single_fold(fold, exp_dir, max_epochs, batch_size, base_lr, augmentation_mode, eval_interval)
        if success:
            successful_folds.append(fold)
        else:
            failed_folds.append(fold)
    
    # 🆕 結果サマリーの生成
    print(f"\n{'='*60}")
    print("K-fold Cross Validation Summary")
    print(f"{'='*60}")
    print(f"✅ Successful folds: {successful_folds} ({len(successful_folds)}/{k_folds})")
    if failed_folds:
        print(f"❌ Failed folds: {failed_folds}")
    
    # 結果をファイルに保存
    summary_path = os.path.join(exp_dir, "kfold_results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("K-fold Cross Validation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total folds: {k_folds}\n")
        f.write(f"Successful: {len(successful_folds)}\n")
        f.write(f"Failed: {len(failed_folds)}\n")
        f.write(f"Success rate: {len(successful_folds)/k_folds*100:.1f}%\n")
        f.write(f"Augmentation mode: {augmentation_mode}\n")
        f.write(f"Eval interval: {eval_interval}\n")
        if successful_folds:
            f.write(f"Successful folds: {', '.join(map(str, successful_folds))}\n")
        if failed_folds:
            f.write(f"Failed folds: {', '.join(map(str, failed_folds))}\n")
    
    return exp_dir


def run_specific_fold(fold_num, max_epochs=150, batch_size=24, base_lr=0.05, augmentation_mode='standard', eval_interval=None):
    """特定foldのみ実行（毎エポック検証対応版）"""
    
    # 🔧 eval_intervalの決定（毎エポック検証を優先）
    if eval_interval is None:
        eval_interval = 1  # 🔧 デフォルトで毎エポック検証
    
    # 最新の実験ディレクトリを検索
    experiments_dir = os.path.join(PROJECT_ROOT, "experiments")
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
                'base_lr': base_lr,
                'augmentation_mode': augmentation_mode,
                'eval_interval': eval_interval  # 🆕 追加
            }
            save_experiment_config(exp_dir, exp_config)
    else:
        # 新規実験ディレクトリ作成
        exp_dir = create_experiment_dir("cellmix_kfold")
        exp_config = {
            'k_folds': 5,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'base_lr': base_lr,
            'augmentation_mode': augmentation_mode,
            'eval_interval': eval_interval  # 🆕 追加
        }
        save_experiment_config(exp_dir, exp_config)
    
    # 🔧 eval_interval を追加
    success = run_single_fold(fold_num, exp_dir, max_epochs, batch_size, base_lr, augmentation_mode, eval_interval)
    
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
    parser.add_argument('--augmentation_mode', type=str, default='nnunet', 
                       choices=['standard', 'enhanced', 'nnunet'], 
                       help='Data augmentation mode')
    # 🆕 eval_interval引数を追加（デフォルト1=毎エポック）
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Validation interval (default: 1 = every epoch)')
    
    args = parser.parse_args()
    
    if args.fold:
        exp_dir = run_specific_fold(args.fold, args.max_epochs, args.batch_size, args.base_lr, 
                                   args.augmentation_mode, args.eval_interval)
    else:
        exp_dir = run_all_folds(args.k_folds, args.max_epochs, args.batch_size, args.base_lr, 
                               args.augmentation_mode, args.eval_interval)