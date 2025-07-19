import torch
import torch.utils.data as data
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ensemble_inference import load_ensemble_model
from datasets.dataset_cellmix import CellMixDataset, RandomGenerator

class EnsembleSegmentationGenerator:
    """アンサンブルセグメンテーション画像生成専用クラス（元の場所に保存）"""
    
    def __init__(self, ensemble_dir, device='cuda'):
        self.ensemble_dir = ensemble_dir
        self.device = device
        
        # CellMixクラス定義（カラー可視化用）
        self.class_names = ['Background', 'Nucleus', 'Cytoplasm']
        self.class_colors = {
            0: [0, 0, 0],         # 背景: 黒 (0,0,0)
            1: [128, 0, 0],       # 核: 暗赤 (128,0,0)
            2: [0, 128, 0],       # 細胞質: 暗緑 (0,128,0)
        }
        
        # アンサンブルモデル読み込み
        print("📥 Loading CellMix ensemble model...")
        self.model, self.info, self.config = load_ensemble_model(ensemble_dir)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        print(f"✅ Ensemble model loaded with {self.info['num_models']} models")
        
        # 🔥 元の場所に戻す（cellmix_test_results）
        ensemble_parent = Path(ensemble_dir).parent  # experiments/cellmix_kfold_20250718_145719/
        self.results_dir = ensemble_parent / "cellmix_test_results"  # ← 元の場所
        self.results_dir.mkdir(exist_ok=True)
        
        # グレースケール用ディレクトリ（元の名前に戻す）
        self.prediction_dir = self.results_dir / "individual_predictions"      # ← 元の名前
        self.groundtruth_dir = self.results_dir / "individual_groundtruths"    # ← 元の名前
        
        # カラー画像用の新しいディレクトリ
        self.prediction_colored_dir = self.results_dir / "predictions_colored"
        self.groundtruth_colored_dir = self.results_dir / "groundtruths_colored"
        
        # 可視化用ディレクトリ（元の場所）
        self.visualization_dir = self.results_dir / "visualizations"           # ← 元の場所
        
        # 全ディレクトリ作成
        dirs_to_create = [
            self.prediction_dir, self.groundtruth_dir,                   # 元のディレクトリ
            self.prediction_colored_dir, self.groundtruth_colored_dir,   # 新規カラー用
            self.visualization_dir                                       # 元の可視化用
        ]
        
        for d in dirs_to_create:
            d.mkdir(exist_ok=True)
    
    def tensor_to_numpy_2d(self, tensor):
        """PyTorchテンソルを2次元numpy配列に変換"""
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            np_array = tensor.numpy()
        else:
            np_array = tensor
        
        # 形状を2次元に調整
        if len(np_array.shape) == 4:  # (1, 1, H, W) -> (H, W)
            np_array = np_array.squeeze()
        elif len(np_array.shape) == 3:  # (1, H, W) -> (H, W)
            if np_array.shape[0] == 1:
                np_array = np_array.squeeze(0)
            else:
                np_array = np_array[0]
        elif len(np_array.shape) == 2:  # (H, W) - そのまま
            pass
        else:
            raise ValueError(f"Unexpected array shape: {np_array.shape}")
        
        return np_array.astype(np.uint8)
    
    def mask_to_color(self, mask):
        """セグメンテーション マスクをカラー画像に変換"""
        mask_2d = self.tensor_to_numpy_2d(mask)
        h, w = mask_2d.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            color_mask[mask_2d == class_id] = color
            
        return color_mask
    
    def generate_segmentations(self, max_samples=None, save_grayscale=True, save_colored=True):
        """セグメンテーション画像生成（元の場所に保存）"""
        
        print("📁 Loading CellMix test dataset...")
        
        # テストデータセット準備
        test_transform = RandomGenerator(output_size=[224, 224])
        
        test_dataset = CellMixDataset(
            base_dir='datasets/CellMix',
            list_dir='lists/CellMix',
            split='test',
            transform=test_transform
        )
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        
        print(f"📊 Found {len(test_dataset)} CellMix test samples")
        
        if max_samples is None:
            max_samples = len(test_dataset)
        else:
            max_samples = min(max_samples, len(test_dataset))
        
        print(f"🔄 Generating segmentations for {max_samples} samples...")
        print(f"📂 Save location: {self.results_dir}")
        
        if save_grayscale:
            print("   ✅ Grayscale segmentations (0,1,2 values) will be saved")
        if save_colored:
            print("   ✅ Colored visualizations (RGB images) will be saved")
        
        # 生成されたファイル記録
        generated_files = {
            'individual_predictions': [],      # 元の名前に合わせる
            'individual_groundtruths': [],     # 元の名前に合わせる
            'predictions_colored': [],
            'groundtruths_colored': [],
            'case_names': [],
            'sample_ids': []
        }
        
        # セグメンテーション生成実行
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Generating Segmentations")):
                if i >= max_samples:
                    break
                
                # データ取得
                image = batch['image']
                label = batch['label']
                case_name = batch['case_name'][0]
                
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()
                
                # アンサンブル推論
                output = self.model(image)
                prediction = torch.argmax(output, dim=1)
                
                # CPU に移動
                pred_cpu = prediction.cpu()
                label_cpu = label.cpu()
                
                # 1. グレースケールセグメント画像保存（元の場所・名前）
                if save_grayscale:
                    pred_2d = self.tensor_to_numpy_2d(pred_cpu)
                    gt_2d = self.tensor_to_numpy_2d(label_cpu)
                    
                    pred_filename = f"{case_name}.png"
                    gt_filename = f"{case_name}.png"
                    
                    pred_path = self.prediction_dir / pred_filename      # individual_predictions/
                    gt_path = self.groundtruth_dir / gt_filename          # individual_groundtruths/
                    
                    Image.fromarray(pred_2d).save(pred_path)
                    Image.fromarray(gt_2d).save(gt_path)
                    
                    generated_files['individual_predictions'].append(str(pred_path))
                    generated_files['individual_groundtruths'].append(str(gt_path))
                
                # 2. カラー可視化画像保存（新規追加）
                if save_colored:
                    pred_color = self.mask_to_color(pred_cpu)
                    gt_color = self.mask_to_color(label_cpu)
                    
                    pred_color_filename = f"{case_name}.png"
                    gt_color_filename = f"{case_name}.png"
                    
                    pred_color_path = self.prediction_colored_dir / pred_color_filename
                    gt_color_path = self.groundtruth_colored_dir / gt_color_filename
                    
                    Image.fromarray(pred_color).save(pred_color_path)
                    Image.fromarray(gt_color).save(gt_color_path)
                    
                    generated_files['predictions_colored'].append(str(pred_color_path))
                    generated_files['groundtruths_colored'].append(str(gt_color_path))
                
                # 記録
                generated_files['case_names'].append(case_name)
                generated_files['sample_ids'].append(i)
                
                # 進行状況表示（10サンプルごと）
                if (i + 1) % 10 == 0:
                    print(f"📁 Generated [{i+1}/{max_samples}] segmentations")
        
        print(f"\n✅ Segmentation generation completed!")
        
        if save_grayscale:
            print(f"📊 Generated {len(generated_files['individual_predictions'])} grayscale prediction files")
            print(f"📊 Generated {len(generated_files['individual_groundtruths'])} grayscale ground truth files")
            print(f"📂 Grayscale predictions: {self.prediction_dir}")
            print(f"📂 Grayscale ground truths: {self.groundtruth_dir}")
        
        if save_colored:
            print(f"📊 Generated {len(generated_files['predictions_colored'])} colored prediction files")
            print(f"📊 Generated {len(generated_files['groundtruths_colored'])} colored ground truth files")
            print(f"📂 Colored predictions: {self.prediction_colored_dir}")
            print(f"📂 Colored ground truths: {self.groundtruth_colored_dir}")
            
            # 色情報を表示
            print(f"\n🎨 Color mapping:")
            for class_id, color in self.class_colors.items():
                class_name = self.class_names[class_id]
                print(f"   {class_name}: RGB{color}")
        
        return generated_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CellMix Ensemble Segmentations (Original Location)')
    parser.add_argument('--ensemble_dir', type=str,
                       default='experiments/cellmix_kfold_20250718_145719/ensemble_model',
                       help='Ensemble model directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all 78)')
    parser.add_argument('--no_grayscale', action='store_true',
                       help='Skip grayscale segmentation generation')
    parser.add_argument('--no_colored', action='store_true',
                       help='Skip colored visualization generation')
    
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    os.chdir(Path(__file__).parent.parent.parent)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # セグメンテーション生成器初期化
    generator = EnsembleSegmentationGenerator(
        ensemble_dir=args.ensemble_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # セグメンテーション生成実行
    generated_files = generator.generate_segmentations(
        max_samples=args.max_samples,
        save_grayscale=not args.no_grayscale,
        save_colored=not args.no_colored
    )

if __name__ == '__main__':
    main()