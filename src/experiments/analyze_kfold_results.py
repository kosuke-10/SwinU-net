import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_experiment():
    """最新の実験ディレクトリを検索"""
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("❌ No experiments directory found")
        return None
    
    exp_dirs = [d for d in os.listdir(experiments_dir) 
               if d.startswith("cellmix_kfold_") and os.path.isdir(os.path.join(experiments_dir, d))]
    
    if not exp_dirs:
        print("❌ No experiments found")
        return None
    
    latest_exp = sorted(exp_dirs)[-1]
    exp_dir = os.path.join(experiments_dir, latest_exp)
    print(f"📁 Analyzing experiment: {exp_dir}")
    return exp_dir

def analyze_kfold_results(exp_dir=None, k_folds=5):
    """K-fold結果の集計・分析（整理されたディレクトリ用）"""
    
    if exp_dir is None:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            return
    
    results = {}
    
    for fold in range(1, k_folds + 1):
        log_path = os.path.join(exp_dir, f"fold_{fold}", "log.txt")
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
            
            # 学習進行の抽出
            train_losses = re.findall(r'Train epoch: (\d+) : loss : ([0-9.]+)', content)
            val_losses = re.findall(r'Val epoch: (\d+) : loss : ([0-9.]+)', content)
            
            if train_losses and val_losses:
                train_epochs = [int(x[0]) for x in train_losses]
                train_loss_values = [float(x[1]) for x in train_losses]
                val_epochs = [int(x[0]) for x in val_losses]
                val_loss_values = [float(x[1]) for x in val_losses]
                
                results[fold] = {
                    'train_epochs': train_epochs,
                    'train_losses': train_loss_values,
                    'val_epochs': val_epochs,
                    'val_losses': val_loss_values,
                    'final_train_loss': train_loss_values[-1] if train_loss_values else None,
                    'final_val_loss': val_loss_values[-1] if val_loss_values else None,
                    'completed': train_epochs[-1] + 1 if train_epochs else 0
                }
                
                print(f"Fold {fold}: Epoch {results[fold]['completed']}/150, "
                      f"Train={results[fold]['final_train_loss']:.4f}, "
                      f"Val={results[fold]['final_val_loss']:.4f}")
        else:
            print(f"Fold {fold}: Not started yet")
    
    # 結果の可視化と保存
    if results:
        completed_folds = [fold for fold in results if results[fold]['completed'] >= 149]
        print(f"\nCompleted folds: {len(completed_folds)}/5")
        
        if len(completed_folds) >= 1:
            # 学習曲線の可視化
            plt.figure(figsize=(15, 5))
            
            # 各foldの学習曲線
            plt.subplot(1, 3, 1)
            for fold in results:
                epochs = range(len(results[fold]['train_losses']))
                plt.plot(epochs, results[fold]['train_losses'], 
                        label=f'Fold {fold} Train', alpha=0.7)
                plt.plot(epochs, results[fold]['val_losses'], '--', 
                        label=f'Fold {fold} Val', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Learning Curves by Fold')
            plt.legend()
            
            # 最終性能のボックスプロット
            if len(completed_folds) >= 2:
                plt.subplot(1, 3, 2)
                final_train = [results[fold]['final_train_loss'] for fold in completed_folds]
                final_val = [results[fold]['final_val_loss'] for fold in completed_folds]
                plt.boxplot([final_train, final_val], labels=['Train', 'Validation'])
                plt.ylabel('Final Loss')
                plt.title('Final Performance Distribution')
                
                # 統計情報
                plt.subplot(1, 3, 3)
                plt.text(0.1, 0.8, f"Train Loss: {np.mean(final_train):.4f} ± {np.std(final_train):.4f}", 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.text(0.1, 0.7, f"Val Loss: {np.mean(final_val):.4f} ± {np.std(final_val):.4f}", 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.text(0.1, 0.6, f"Completed: {len(completed_folds)}/5", 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                plt.title('Statistics')
            
            plt.tight_layout()
            
            # 結果画像を実験ディレクトリに保存
            results_path = os.path.join(exp_dir, 'kfold_results.png')
            plt.savefig(results_path, dpi=300, bbox_inches='tight')
            print(f"📊 Results saved to: {results_path}")
            plt.show()
        
        # 詳細結果をテキストファイルに保存
        analysis_path = os.path.join(exp_dir, 'detailed_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write("K-fold Cross Validation Detailed Analysis\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for fold in results:
                f.write(f"Fold {fold}:\n")
                f.write(f"  Completed epochs: {results[fold]['completed']}\n")
                f.write(f"  Final train loss: {results[fold]['final_train_loss']:.6f}\n")
                f.write(f"  Final val loss: {results[fold]['final_val_loss']:.6f}\n\n")
        
        print(f"📄 Detailed analysis saved: {analysis_path}")
    
    return results

if __name__ == '__main__':
    analyze_kfold_results()