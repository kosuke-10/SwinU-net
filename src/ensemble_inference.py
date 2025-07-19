import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
import sys
from datetime import datetime

# ✅ 確実なパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = current_dir

# プロジェクトルートをsys.pathに追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# srcディレクトリをsys.pathに追加
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# インポート
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config  # srcディレクトリから直接インポート

class EnsembleModel(nn.Module):
    """K-fold trained modelsのアンサンブル推論"""
    
    def __init__(self, fold_model_paths, config, num_classes=3, img_size=224):
        super(EnsembleModel, self).__init__()
        
        self.num_models = len(fold_model_paths)
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 各foldのモデルを読み込み
        self.models = nn.ModuleList()
        for i, model_path in enumerate(fold_model_paths):
            model = ViT_seg(config, img_size=img_size, num_classes=num_classes)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded fold {i+1} model: {os.path.basename(model_path)}")
            else:
                print(f"❌ Model not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model.eval()  # 推論モードに設定
            self.models.append(model)
        
        print(f"🎯 Ensemble created with {self.num_models} models")
    
    def forward(self, x):
        """アンサンブル推論"""
        outputs = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                # ソフトマックスで確率に変換
                output = torch.softmax(output, dim=1)
                outputs.append(output)
        
        # 平均アンサンブル
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        return ensemble_output

def create_ensemble_from_kfold(experiment_dir, config_path=None):
    """K-fold実験からアンサンブルモデルを作成"""
    
    experiment_path = Path(experiment_dir)
    
    # 各foldのモデルパスを収集
    fold_model_paths = []
    for fold in range(1, 6):  # fold_1 to fold_5
        fold_dir = experiment_path / f"fold_{fold}"
        model_path = fold_dir / "best_model.pth"
        
        if model_path.exists():
            fold_model_paths.append(str(model_path))
        else:
            print(f"⚠️  Model not found for fold {fold}: {model_path}")
    
    if len(fold_model_paths) == 0:
        raise ValueError("No fold models found!")
    
    print(f"📁 Found {len(fold_model_paths)} fold models")
    
    # 設定ファイルの読み込み
    if config_path is None:
        config_path = "configs/swin_tiny_patch4_window7_224_lite.yaml"
    
    # ダミーargs作成（config読み込み用）
    import argparse
    args = argparse.Namespace()
    args.cfg = config_path
    args.opts = None
    args.zip = False
    args.cache_mode = 'part'
    args.resume = ''
    args.accumulation_steps = None
    args.use_checkpoint = False
    args.amp_opt_level = 'O1'
    args.output = ''
    args.tag = None
    args.eval = False
    args.throughput = False
    args.batch_size = None
    
    config = get_config(args)
    
    # アンサンブルモデル作成
    ensemble_model = EnsembleModel(
        fold_model_paths=fold_model_paths,
        config=config,
        num_classes=3,
        img_size=224
    )
    
    return ensemble_model, fold_model_paths

def save_ensemble_model(experiment_dir, fold_model_paths):
    """アンサンブルモデル情報の保存"""
    
    experiment_path = Path(experiment_dir)
    ensemble_dir = experiment_path / "ensemble_model"
    ensemble_dir.mkdir(exist_ok=True)
    
    # モデルパス情報の保存
    model_paths_info = {
        "fold_model_paths": fold_model_paths,
        "num_models": len(fold_model_paths),
        "created_timestamp": str(datetime.now()),
        "model_type": "SwinUnet",
        "num_classes": 3,
        "img_size": 224
    }
    
    with open(ensemble_dir / "model_paths.json", 'w') as f:
        json.dump(model_paths_info, f, indent=2)
    
    # アンサンブル設定の保存
    ensemble_config = {
        "ensemble_method": "mean",
        "num_models": len(fold_model_paths),
        "config_path": "configs/swin_tiny_patch4_window7_224_lite.yaml",
        "ensemble_type": "kfold_crossvalidation"
    }
    
    with open(ensemble_dir / "ensemble_config.json", 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # アンサンブル用重みファイル
    ensemble_weights = {
        "model_paths": fold_model_paths,
        "weights": [1.0 / len(fold_model_paths)] * len(fold_model_paths),
        "ensemble_method": "weighted_average"
    }
    
    torch.save(ensemble_weights, ensemble_dir / "ensemble_weights.pth")
    
    print(f"💾 Ensemble model saved to: {ensemble_dir}")
    return ensemble_dir

def load_ensemble_model(ensemble_dir):
    """保存されたアンサンブルモデルの読み込み"""
    
    ensemble_path = Path(ensemble_dir)
    
    # モデルパス情報の読み込み
    with open(ensemble_path / "model_paths.json", 'r') as f:
        model_paths_info = json.load(f)
    
    # アンサンブル設定の読み込み
    with open(ensemble_path / "ensemble_config.json", 'r') as f:
        ensemble_config = json.load(f)
    
    # configの読み込み
    config_path = ensemble_config.get("config_path", "configs/swin_tiny_patch4_window7_224_lite.yaml")
    
    # アンサンブルモデル作成
    ensemble_model, _ = create_ensemble_from_kfold(
        experiment_dir=ensemble_path.parent,
        config_path=config_path
    )
    
    print(f"📂 Loaded ensemble model from: {ensemble_dir}")
    
    return ensemble_model, model_paths_info, ensemble_config