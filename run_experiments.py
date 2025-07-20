#!/usr/bin/env python3
"""
Swin-Unet実験実行スクリプト（統一化版）
シェルスクリプトを使わずにPythonで直接実行
"""

import subprocess
import sys
import os
import argparse

def run_command(cmd, description=""):
    """コマンドの実行"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def train_cellmix(epochs=150, batch_size=24, output_name=None):
    """CellMix学習（統一版）"""
    if output_name is None:
        output_name = f"cellmix_{epochs}ep"
    
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

def train_kfold_single(fold=1, epochs=150, batch_size=24):
    """K-fold単一fold学習"""
    cmd = [
        'python3', 'src/experiments/run_kfold_cellmix.py',
        '--fold', str(fold),
        '--max_epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    return run_command(cmd, f"K-fold Fold {fold} ({epochs} epochs, batch={batch_size})")

def train_kfold_all(epochs=150, batch_size=24):
    """K-fold全fold学習"""
    cmd = [
        'python3', 'src/experiments/run_kfold_cellmix.py',
        '--max_epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    return run_command(cmd, f"K-fold All Folds ({epochs} epochs, batch={batch_size})")

def create_ensemble():
    """アンサンブルモデル作成"""
    cmd = ['python3', 'src/experiments/create_ensemble.py']
    return run_command(cmd, "Creating Ensemble Model")

def test_ensemble():
    """アンサンブルモデルテスト"""
    cmd = ['python3', 'src/experiments/test_ensemble.py']
    return run_command(cmd, "Testing Ensemble Model")

def analyze_results():
    """結果分析"""
    cmd = ['python3', 'src/experiments/analyze_kfold_results.py']
    return run_command(cmd, "Results Analysis")

def show_menu():
    """実行メニューの表示（統一版）"""
    print("\n" + "="*60)
    print("🎯 Swin-Unet Experiments Menu")
    print("="*60)
    print("1. Single Model Training (fixed train/val split)")
    print("2. K-fold Single Fold Training (specify fold 1-5)")
    print("3. K-fold All Folds Training (full cross validation)")
    print("4. Create Ensemble Model")
    print("5. Test Ensemble Model")
    print("6. Analyze Results")
    print("7. Custom Command")
    print("0. Exit")
    print("="*60)

def get_user_params(default_epochs, default_batch_size, context=""):
    """学習パラメータの入力取得（統一化）"""
    print(f"\n📋 {context} Parameters:")
    print(f"  Default epochs: {default_epochs}")
    print(f"  Default batch size: {default_batch_size}")
    
    # エポック数の入力
    epochs_input = input(f"Epochs (Enter=default {default_epochs}): ").strip()
    epochs = int(epochs_input) if epochs_input and epochs_input.isdigit() else default_epochs
    
    # バッチサイズの入力
    batch_input = input(f"Batch size (Enter=default {default_batch_size}): ").strip()
    batch_size = int(batch_input) if batch_input and batch_input.isdigit() else default_batch_size
    
    return epochs, batch_size

def get_fold_number():
    """Fold番号の入力取得"""
    fold_input = input("Fold番号 (1-5, Enter=1): ").strip()
    if fold_input and fold_input.isdigit():
        fold = int(fold_input)
        fold = fold if 1 <= fold <= 5 else 1
    else:
        fold = 1
    return fold

def interactive_mode():
    """対話式実行モード（統一版）"""
    while True:
        show_menu()
        choice = input("\n選択してください (0-7): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
            
        elif choice == '1':
            print("\n🚀 CellMix Training")
            epochs, batch_size = get_user_params(150, 24, "CellMix Training")
            output_name = input(f"Output name (Enter=cellmix_{epochs}ep): ").strip() or f"cellmix_{epochs}ep"
            train_cellmix(epochs=epochs, batch_size=batch_size, output_name=output_name)
            
        elif choice == '2':
            print("\n🚀 K-fold Single Fold")
            fold = get_fold_number()
            epochs, batch_size = get_user_params(150, 24, f"K-fold Fold {fold}")
            train_kfold_single(fold=fold, epochs=epochs, batch_size=batch_size)
            
        elif choice == '3':
            print("\n🚀 K-fold All Folds")
            epochs, batch_size = get_user_params(150, 24, "K-fold All Folds")
            
            estimated_time = epochs * 5 / 60
            print(f"\n📊 Final Settings:")
            print(f"  • Epochs per fold: {epochs}")
            print(f"  • Batch size: {batch_size}")  
            print(f"  • Total folds: 5")
            print(f"  • Estimated time: ~{estimated_time:.1f} hours")
            
            confirm = input(f"\n⚠️  全5 fold実行を開始しますか？ (y/N): ")
            if confirm.lower() == 'y':
                train_kfold_all(epochs=epochs, batch_size=batch_size)
            else:
                print("❌ キャンセルしました")
                
        elif choice == '4':
            print("\n🎯 Create Ensemble Model")
            create_ensemble()
            
        elif choice == '5':
            print("\n🧪 Test Ensemble Model")
            test_ensemble()
            
        elif choice == '6':
            print("\n📊 Analyze Results")
            analyze_results()
            
        elif choice == '7':
            print("\n📝 Custom Command")
            print("例: python3 src/train.py --dataset CellMix --max_epochs 50")
            cmd_str = input("コマンドを入力: ").strip()
            if cmd_str:
                cmd = cmd_str.split()
                run_command(cmd, "Custom Command")
        else:
            print("❌ 無効な選択です")

def main():
    parser = argparse.ArgumentParser(description='Swin-Unet Experiments Runner (Unified)')
    parser.add_argument('--mode', choices=['train', 'kfold-single', 'kfold-all', 'ensemble', 'test-ensemble', 'analyze'], 
                       help='Execution mode')
    parser.add_argument('--fold', type=int, help='Specific fold for kfold modes')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size (default: 24)')
    parser.add_argument('--output_name', type=str, help='Output directory name')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
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
        train_kfold_single(fold=args.fold, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'kfold-all':
        estimated_time = args.epochs * 5 / 60
        print(f"📊 K-fold All Folds Settings:")
        print(f"  • Epochs per fold: {args.epochs}")
        print(f"  • Batch size: {args.batch_size}")
        print(f"  • Estimated time: ~{estimated_time:.1f} hours")
        train_kfold_all(epochs=args.epochs, batch_size=args.batch_size)
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