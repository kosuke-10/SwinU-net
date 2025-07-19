import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# パス設定：プロジェクトルートをsys.pathに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ensemble_inference import create_ensemble_from_kfold, save_ensemble_model

def find_latest_kfold_experiment():
    """最新のK-fold実験ディレクトリを検索"""
    experiments_dir = Path("experiments")
    
    if not experiments_dir.exists():
        print("❌ experiments/ directory not found")
        return None
    
    kfold_dirs = [d for d in experiments_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("cellmix_kfold_")]
    
    if not kfold_dirs:
        print("❌ No K-fold experiments found")
        return None
    
    # 最新のディレクトリを取得
    latest_dir = sorted(kfold_dirs, key=lambda x: x.name)[-1]
    print(f"📁 Latest K-fold experiment: {latest_dir}")
    
    return str(latest_dir)

def check_kfold_completion(experiment_dir):
    """K-fold実験の完了状況確認"""
    exp_path = Path(experiment_dir)
    
    completed_folds = []
    missing_folds = []
    
    for fold in range(1, 6):
        fold_dir = exp_path / f"fold_{fold}"
        model_path = fold_dir / "best_model.pth"
        
        if model_path.exists():
            completed_folds.append(fold)
            # ファイルサイズも表示
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Fold {fold}: {size_mb:.1f}MB")
        else:
            missing_folds.append(fold)
            print(f"  ❌ Fold {fold}: Model not found")
    
    print(f"\n📊 Summary:")
    print(f"✅ Completed folds: {completed_folds}")
    print(f"❌ Missing folds: {missing_folds}")
    
    return completed_folds, missing_folds

def create_ensemble_workflow(experiment_dir=None):
    """アンサンブルモデル作成のワークフロー"""
    
    # 実験ディレクトリの決定
    if experiment_dir is None:
        experiment_dir = find_latest_kfold_experiment()
        if experiment_dir is None:
            return None
    
    print(f"\n{'='*60}")
    print(f"🎯 Creating Ensemble Model")
    print(f"{'='*60}")
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # 完了状況確認
    completed_folds, missing_folds = check_kfold_completion(experiment_dir)
    
    if len(completed_folds) < 3:
        print(f"⚠️  Warning: Only {len(completed_folds)} folds completed. Recommend at least 3 folds.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return None
    elif len(missing_folds) > 0:
        print(f"⚠️  Warning: {len(missing_folds)} folds are missing. Continue with {len(completed_folds)} folds?")
        response = input("Continue? (Y/n): ")
        if response.lower() == 'n':
            print("❌ Cancelled")
            return None
    
    # アンサンブルモデル作成
    try:
        print(f"\n🔄 Creating ensemble model...")
        ensemble_model, fold_model_paths = create_ensemble_from_kfold(experiment_dir)
        
        # アンサンブルモデル保存
        print(f"\n💾 Saving ensemble model...")
        ensemble_dir = save_ensemble_model(experiment_dir, fold_model_paths)
        
        print(f"\n🎉 Ensemble model created successfully!")
        print(f"📂 Location: {ensemble_dir}")
        print(f"📊 Models included: {len(fold_model_paths)} folds")
        
        # 使用例を表示
        print(f"\n📋 Usage example:")
        print(f"```python")
        print(f"from src.ensemble_inference import load_ensemble_model")
        print(f"ensemble_model, info, config = load_ensemble_model('{ensemble_dir}')")
        print(f"# Use ensemble_model for inference")
        print(f"```")
        
        return ensemble_dir
        
    except Exception as e:
        print(f"❌ Error creating ensemble: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Create Ensemble Model from K-fold Results')
    parser.add_argument('--experiment_dir', type=str, 
                       help='K-fold experiment directory (default: latest)')
    parser.add_argument('--force', action='store_true',
                       help='Force creation even with incomplete folds')
    
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    print(f"📁 Working directory: {os.getcwd()}")
    
    ensemble_dir = create_ensemble_workflow(args.experiment_dir)
    
    if ensemble_dir:
        print(f"\n✅ Ensemble creation completed!")
        print(f"📂 Next steps:")
        print(f"   • Test individual folds: python3 src/experiments/test_single_fold.py")
        print(f"   • Test ensemble: python3 src/experiments/test_ensemble.py")
        print(f"   • Analyze results: python3 src/experiments/analyze_results.py")

if __name__ == '__main__':
    main()