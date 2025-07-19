import os
import glob

def check_kfold_results():
    """K-fold実験結果の確認"""
    
    # 実験ディレクトリの確認
    experiments_dir = "../experiments"
    if not os.path.exists(experiments_dir):
        print("❌ experiments ディレクトリが見つかりません")
        return
    
    # CellMix K-fold実験を探す
    kfold_dirs = glob.glob(os.path.join(experiments_dir, "cellmix_kfold_*"))
    if not kfold_dirs:
        print("❌ CellMix K-fold実験が見つかりません")
        return
    
    latest_exp = sorted(kfold_dirs)[-1]
    print(f"📁 最新の実験: {latest_exp}")
    
    # 各foldの結果確認
    for fold in range(1, 6):
        fold_dir = os.path.join(latest_exp, f"fold_{fold}")
        if os.path.exists(fold_dir):
            print(f"\n📂 Fold {fold}: {fold_dir}")
            
            # モデルファイル確認
            best_model = os.path.join(fold_dir, "best_model.pth")
            last_model = os.path.join(fold_dir, "last_model.pth")
            
            if os.path.exists(best_model):
                print(f"  ✅ best_model.pth ({os.path.getsize(best_model)/1024/1024:.1f}MB)")
            else:
                print(f"  ❌ best_model.pth not found")
                
            if os.path.exists(last_model):
                print(f"  ✅ last_model.pth ({os.path.getsize(last_model)/1024/1024:.1f}MB)")
            else:
                print(f"  ❌ last_model.pth not found")
            
            # ログファイル確認
            log_file = os.path.join(fold_dir, "log.txt")
            if os.path.exists(log_file):
                print(f"  ✅ log.txt ({os.path.getsize(log_file)/1024:.1f}KB)")
                # 最後のepochを確認
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        if "epoch" in last_line.lower():
                            print(f"  📊 最終ログ: {last_line.strip()}")
            else:
                print(f"  ❌ log.txt not found")
        else:
            print(f"❌ Fold {fold} ディレクトリが見つかりません")

if __name__ == "__main__":
    check_kfold_results()
    
    