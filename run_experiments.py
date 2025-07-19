#!/usr/bin/env python3
"""
Swin-Unet実験実行スクリプト
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

def train_cellmix_basic(epochs=5, batch_size=8, output_name="cellmix_test"):
    """CellMix基本学習"""
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
    
    return run_command(cmd, f"CellMix Training ({epochs} epochs)")

def train_cellmix_full(epochs=150, batch_size=24, output_name="cellmix_main"):
    """CellMix本格学習"""
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
    
    return run_command(cmd, f"CellMix Full Training ({epochs} epochs)")

def train_kfold_single(fold=1, epochs=10, batch_size=8):
    """K-fold単一fold学習"""
    cmd = [
        'python3', 'src/experiments/run_kfold_cellmix.py',
        '--fold', str(fold),
        '--max_epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    return run_command(cmd, f"K-fold Fold {fold} ({epochs} epochs)")

def train_kfold_all(epochs=150, batch_size=24):
    """K-fold全fold学習"""
    cmd = [
        'python3', 'src/experiments/run_kfold_cellmix.py',
        '--max_epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    return run_command(cmd, f"K-fold All Folds ({epochs} epochs)")

def create_ensemble():
    """アンサンブルモデル作成"""
    cmd = [
        'python3', 'src/experiments/create_ensemble.py'
    ]
    
    return run_command(cmd, "Creating Ensemble Model")

def analyze_results():
    """結果分析"""
    cmd = [
        'python3', 'src/experiments/analyze_kfold_results.py'
    ]
    
    return run_command(cmd, "Results Analysis")

def show_menu():
    """実行メニューの表示"""
    print("\n" + "="*60)
    print("🎯 Swin-Unet Experiments Menu")
    print("="*60)
    print("1. CellMix Quick Test (customizable epochs)")
    print("2. CellMix Full Training (customizable epochs)")
    print("3. K-fold Single Fold Test (customizable epochs)")
    print("4. K-fold Single Fold Full (customizable epochs)")
    print("5. K-fold All Folds (customizable epochs)")
    print("6. Create Ensemble Model")
    print("7. Test Ensemble Model")
    print("8. Analyze Results")
    print("9. Custom Command")
    print("0. Exit")
    print("="*60)

def get_training_params():
    """学習パラメータの入力取得"""
    print("\n📋 Training Parameters:")
    
    # エポック数の入力
    epochs_input = input("Epochs (default: 5 for test, 150 for full): ").strip()
    if epochs_input:
        try:
            epochs = int(epochs_input)
        except ValueError:
            print("⚠️  Invalid epochs, using default")
            epochs = None
    else:
        epochs = None
    
    # バッチサイズの入力
    batch_input = input("Batch size (default: 8 for test, 24 for full): ").strip()
    if batch_input:
        try:
            batch_size = int(batch_input)
        except ValueError:
            print("⚠️  Invalid batch size, using default")
            batch_size = None
    else:
        batch_size = None
    
    return epochs, batch_size

def get_fold_number():
    """Fold番号の入力取得"""
    fold_input = input("Fold番号 (1-5, default: 1): ").strip()
    if fold_input:
        try:
            fold = int(fold_input)
            if fold < 1 or fold > 5:
                print("⚠️  Invalid fold number, using 1")
                fold = 1
        except ValueError:
            print("⚠️  Invalid fold number, using 1")
            fold = 1
    else:
        fold = 1
    return fold

def create_ensemble():
    """アンサンブルモデル作成"""
    cmd = [
        'python3', 'src/experiments/create_ensemble.py'
    ]
    
    return run_command(cmd, "Creating Ensemble Model")

def test_ensemble():
    """アンサンブルモデルテスト"""
    cmd = [
        'python3', 'src/experiments/test_ensemble.py'
    ]
    
    return run_command(cmd, "Testing Ensemble Model")

def interactive_mode():
    """対話式実行モード"""
    while True:
        show_menu()
        choice = input("\n選択してください (0-8): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            print("\n🚀 CellMix Quick Test")
            epochs, batch_size = get_training_params()
            epochs = epochs or 5
            batch_size = batch_size or 8
            
            output_name = input(f"Output name (default: cellmix_test_{epochs}ep): ").strip()
            output_name = output_name or f"cellmix_test_{epochs}ep"
            
            train_cellmix_basic(epochs=epochs, batch_size=batch_size, output_name=output_name)
            
        elif choice == '2':
            print("\n🚀 CellMix Full Training")
            epochs, batch_size = get_training_params()
            epochs = epochs or 150
            batch_size = batch_size or 24
            
            output_name = input(f"Output name (default: cellmix_main_{epochs}ep): ").strip()
            output_name = output_name or f"cellmix_main_{epochs}ep"
            
            train_cellmix_full(epochs=epochs, batch_size=batch_size, output_name=output_name)
            
        elif choice == '3':
            print("\n🚀 K-fold Single Fold Test")
            fold = get_fold_number()
            epochs, batch_size = get_training_params()
            epochs = epochs or 10
            batch_size = batch_size or 8
            
            train_kfold_single(fold=fold, epochs=epochs, batch_size=batch_size)
            
        elif choice == '4':
            print("\n🚀 K-fold Single Fold Full")
            fold = get_fold_number()
            epochs, batch_size = get_training_params()
            epochs = epochs or 150
            batch_size = batch_size or 24
            
            train_kfold_single(fold=fold, epochs=epochs, batch_size=batch_size)
            
        elif choice == '5':
            print("\n🚀 K-fold All Folds")
            epochs, batch_size = get_training_params()
            epochs = epochs or 150
            batch_size = batch_size or 24
            
            estimated_time = epochs * 5 / 60  # 大体の時間推定（時間）
            
            print(f"\n📊 Settings:")
            print(f"  Epochs per fold: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Estimated total time: ~{estimated_time:.1f} hours")
            
            confirm = input(f"⚠️  全5 fold実行を開始しますか？ (y/N): ")
            if confirm.lower() == 'y':
                train_kfold_all(epochs=epochs, batch_size=batch_size)
            else:
                print("キャンセルしました")
                
        elif choice == '6':
            print("\n🎯 Create Ensemble Model")
            create_ensemble()
            
        elif choice == '7':
            print("\n🧪 Test Ensemble Model")
            test_ensemble()
            
        elif choice == '8':
            print("\n📊 Analyze Results")
            analyze_results()
            
        elif choice == '9':
            print("\n📝 Custom Commands:")
            print("例: python3 src/train.py --dataset CellMix --max_epochs 50")
            cmd_str = input("コマンドを入力: ").strip()
            if cmd_str:
                cmd = cmd_str.split()
                run_command(cmd, "Custom Command")
        else:
            print("❌ 無効な選択です")

def main():
    parser = argparse.ArgumentParser(description='Swin-Unet Experiments Runner')
    parser.add_argument('--mode', choices=['quick', 'full', 'kfold-test', 'kfold-full', 'kfold-all', 'ensemble', 'analyze'], 
                       help='Execution mode')
    parser.add_argument('--fold', type=int, help='Specific fold for kfold modes')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--output_name', type=str, help='Output directory name')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.interactive:
        interactive_mode()
    elif args.mode == 'quick':
        epochs = args.epochs or 5
        batch_size = args.batch_size or 8
        output_name = args.output_name or f"cellmix_test_{epochs}ep"
        train_cellmix_basic(epochs=epochs, batch_size=batch_size, output_name=output_name)
    elif args.mode == 'full':
        epochs = args.epochs or 150
        batch_size = args.batch_size or 24
        output_name = args.output_name or f"cellmix_main_{epochs}ep"
        train_cellmix_full(epochs=epochs, batch_size=batch_size, output_name=output_name)
    elif args.mode == 'kfold-test':
        fold = args.fold or 1
        epochs = args.epochs or 10
        batch_size = args.batch_size or 8
        train_kfold_single(fold=fold, epochs=epochs, batch_size=batch_size)
    elif args.mode == 'kfold-full':
        if args.fold:
            epochs = args.epochs or 150
            batch_size = args.batch_size or 24
            train_kfold_single(fold=args.fold, epochs=epochs, batch_size=batch_size)
        else:
            print("⚠️  For single fold, please specify --fold")
    elif args.mode == 'kfold-all':
        epochs = args.epochs or 150
        batch_size = args.batch_size or 24
        train_kfold_all(epochs=epochs, batch_size=batch_size)
    elif args.mode == 'ensemble':
        create_ensemble()
    elif args.mode == 'analyze':
        analyze_results()
    else:
        interactive_mode()

if __name__ == '__main__':
    main()