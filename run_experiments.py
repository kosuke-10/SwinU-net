import os
import sys
import subprocess
import argparse
from datetime import datetime


def run_command(cmd, description):
    """コマンド実行の共通処理"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def train_cellmix(epochs=150, batch_size=24, output_name="cellmix_train"):
    """CellMix単発学習"""
    cmd = [
        'python3', 'src/train.py',
        '--dataset', 'CellMix',
        '--cfg', 'configs/swin_tiny_patch4_window7_224_lite.yaml',
        '--root_path', 'datasets/CellMix',
        '--list_dir', 'lists/CellMix',
        '--num_classes', '3',
        '--n_class', '3',
        '--max_epochs', str(epochs),
        '--output_dir', f'outputs/{output_name}',
        '--img_size', '224',
        '--base_lr', '0.05',
        '--batch_size', str(batch_size)
    ]
    
    return run_command(cmd, f"CellMix Training ({epochs} epochs, batch={batch_size})")


def train_kfold_single(fold=1, epochs=150, batch_size=24, augmentation_mode='nnunet', eval_interval=1):
    """K-fold単一fold学習"""
    cmd = [
        'python3', 'src/experiments/run_kfold_cellmix.py',
        '--fold', str(fold),
        '--max_epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--augmentation_mode', augmentation_mode,
        '--eval_interval', str(eval_interval)  # 🆕 追加
    ]
    
    return run_command(cmd, f"K-fold Fold {fold} ({epochs} epochs, batch={batch_size}, aug={augmentation_mode}, eval_interval={eval_interval})")


def train_kfold_all(epochs=150, batch_size=24, augmentation_mode='nnunet', eval_interval=1):
    """K-fold全fold学習"""
    cmd = [
        'python3', 'src/experiments/run_kfold_cellmix.py',
        '--max_epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--augmentation_mode', augmentation_mode,
        '--eval_interval', str(eval_interval)  # 🆕 追加
    ]
    
    return run_command(cmd, f"K-fold All Folds ({epochs} epochs, batch={batch_size}, aug={augmentation_mode}, eval_interval={eval_interval})")


def create_ensemble():
    """アンサンブルモデル作成"""
    cmd = ['python3', 'src/experiments/create_ensemble.py']  # 🔧 修正: ensemble → experiments
    return run_command(cmd, "Creating Ensemble Model")


def test_ensemble():
    """アンサンブルモデルテスト"""
    cmd = ['python3', 'src/experiments/test_ensemble_segmentation_cellmix.py']  # 🔧 修正: 正確なファイル名
    return run_command(cmd, "Testing Ensemble Model")


def analyze_results():
    """結果分析"""
    cmd = ['python3', 'src/experiments/analyze_results.py']  # 🔧 確認が必要
    return run_command(cmd, "Analyzing Results")


def interactive_mode():
    """インタラクティブモード"""
    print("🎯 Swin-Unet Experiments Menu")
    print("=" * 40)
    print("1. CellMix Training (Single)")
    print("2. K-fold Single Fold")
    print("3. K-fold All Folds")
    print("4. Create Ensemble")
    print("5. Test Ensemble")
    print("6. Analyze Results")
    print("0. Exit")
    
    choice = input("\nSelect option (0-6): ")
    
    if choice == '1':
        epochs = int(input("Epochs (default: 150): ") or 150)
        batch_size = int(input("Batch size (default: 24): ") or 24)
        output_name = input("Output name (default: cellmix_interactive): ") or "cellmix_interactive"
        train_cellmix(epochs, batch_size, output_name)
    elif choice == '2':
        fold = int(input("Fold number (1-5): "))
        epochs = int(input("Epochs (default: 150): ") or 150)
        batch_size = int(input("Batch size (default: 24): ") or 24)
        aug_mode = input("Augmentation mode (standard/enhanced/nnunet, default: nnunet): ") or "nnunet"
        eval_interval = int(input("Validation interval (default: 1 = every epoch): ") or 1)  # 🆕 追加
        train_kfold_single(fold, epochs, batch_size, aug_mode, eval_interval)
    elif choice == '3':
        epochs = int(input("Epochs (default: 150): ") or 150)
        batch_size = int(input("Batch size (default: 24): ") or 24)
        aug_mode = input("Augmentation mode (standard/enhanced/nnunet, default: nnunet): ") or "nnunet"
        eval_interval = int(input("Validation interval (default: 1 = every epoch): ") or 1)  # 🆕 追加
        estimated_time = epochs * 5 / 60
        print(f"⚠️ Estimated time: ~{estimated_time:.1f} hours")
        confirm = input("Continue? (y/N): ")
        if confirm.lower() == 'y':
            train_kfold_all(epochs, batch_size, aug_mode, eval_interval)
    elif choice == '4':
        create_ensemble()
    elif choice == '5':
        test_ensemble()
    elif choice == '6':
        analyze_results()
    elif choice == '0':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid option")
        interactive_mode()


def main():
    parser = argparse.ArgumentParser(description='Swin-Unet Experiments Runner (Unified)')
    parser.add_argument('--mode', choices=['train', 'kfold-single', 'kfold-all', 'ensemble', 'test-ensemble', 'analyze'], 
                       help='Execution mode')
    parser.add_argument('--fold', type=int, help='Specific fold for kfold modes')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size (default: 24)')
    parser.add_argument('--output_name', type=str, help='Output directory name')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--augmentation_mode', type=str, default='nnunet', 
                       choices=['standard', 'enhanced', 'nnunet'], 
                       help='Data augmentation mode (default: nnunet)')
    # 🆕 eval_interval引数を追加
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Validation interval (default: 1 = every epoch)')
    
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.interactive or not args.mode:
        interactive_mode()
    elif args.mode == 'train':
        output_name = args.output_name or f"cellmix_{args.epochs}ep"
        train_cellmix(epochs=args.epochs, batch_size=args.batch_size, output_name=output_name)
    elif args.mode == 'kfold-single':
        if not args.fold:
            print("⚠️  For single fold, please specify --fold (1-5)")
            return
        train_kfold_single(fold=args.fold, epochs=args.epochs, batch_size=args.batch_size, 
                          augmentation_mode=args.augmentation_mode, eval_interval=args.eval_interval)  # 🔧 修正
    elif args.mode == 'kfold-all':
        estimated_time = args.epochs * 5 / 60
        print(f"📊 K-fold All Folds Settings:")
        print(f"  • Epochs per fold: {args.epochs}")
        print(f"  • Batch size: {args.batch_size}")
        print(f"  • Augmentation mode: {args.augmentation_mode}")
        print(f"  • Validation: every {args.eval_interval} epochs")
        print(f"  • Estimated time: ~{estimated_time:.1f} hours")
        train_kfold_all(epochs=args.epochs, batch_size=args.batch_size, 
                       augmentation_mode=args.augmentation_mode, eval_interval=args.eval_interval)  # 🔧 修正
    elif args.mode == 'ensemble':
        create_ensemble()
    elif args.mode == 'test-ensemble':
        test_ensemble()
    elif args.mode == 'analyze':
        analyze_results()
    else:
        interactive_mode()


if __name__ == '__main__':
    main()