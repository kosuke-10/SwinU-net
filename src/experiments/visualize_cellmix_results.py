import numpy as np
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import argparse
from tqdm import tqdm
import pandas as pd

class CellMixVisualizationGenerator:
    """CellMix可視化生成専用クラス"""
    
    def __init__(self):
        # CellMixクラス定義
        self.class_names = ['Background', 'Nucleus', 'Cytoplasm']
        self.class_colors = {
            0: [0, 0, 0],         # 背景: 黒
            1: [128, 0, 0],       # 核: 暗赤
            2: [0, 128, 0],       # 細胞質: 暗緑
        }
    
    def mask_to_color(self, mask):
        """セグメンテーション マスクをカラー画像に変換"""
        mask_array = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        h, w = mask_array.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            color_mask[mask_array == class_id] = color
            
        return color_mask
    
    def create_comparison_visualization(self, original_img_path, gt_path, pred_path, 
                                      metrics_data, save_path):
        """詳細比較可視化を作成"""
        
        # 画像読み込み
        if original_img_path and os.path.exists(original_img_path):
            original = np.array(Image.open(original_img_path).convert('RGB'))
        else:
            # 原画像がない場合は白い画像を作成
            gt_img = Image.open(gt_path)
            original = np.ones((*gt_img.size[::-1], 3), dtype=np.uint8) * 255
        
        gt_mask = np.array(Image.open(gt_path))
        pred_mask = np.array(Image.open(pred_path))
        
        # カラー画像に変換
        gt_color = self.mask_to_color(gt_mask)
        pred_color = self.mask_to_color(pred_mask)
        
        # 比較画像作成
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 上段
        # 元画像
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # 正解ラベル
        axes[0, 1].imshow(gt_color)
        axes[0, 1].set_title('Ground Truth\n(Black:Background, Red:Nucleus, Green:Cytoplasm)', fontsize=12)
        axes[0, 1].axis('off')
        
        # 予測結果
        axes[0, 2].imshow(pred_color)
        axes[0, 2].set_title('Prediction\n(Black:Background, Red:Nucleus, Green:Cytoplasm)', fontsize=12)
        axes[0, 2].axis('off')
        
        # オーバーレイ
        if original.shape[:2] == pred_color.shape[:2]:
            overlay = cv2.addWeighted(original, 0.6, pred_color, 0.4, 0)
            axes[0, 3].imshow(overlay)
        else:
            axes[0, 3].imshow(pred_color)
        axes[0, 3].set_title('Prediction Overlay', fontsize=12)
        axes[0, 3].axis('off')
        
        # 下段
        # 差分画像
        diff_mask = np.abs(pred_mask.astype(int) - gt_mask.astype(int))
        diff_color = np.zeros((diff_mask.shape[0], diff_mask.shape[1], 3), dtype=np.uint8)
        diff_color[diff_mask > 0] = [255, 255, 0]  # 差分を黄色で表示
        axes[1, 0].imshow(diff_color)
        axes[1, 0].set_title('Difference (Yellow = Error)', fontsize=12)
        axes[1, 0].axis('off')
        
        # 核比較
        nucleus_pred = (pred_mask == 1).astype(np.uint8) * 255
        nucleus_gt = (gt_mask == 1).astype(np.uint8) * 255
        axes[1, 1].imshow(nucleus_gt, cmap='Reds', alpha=0.7)
        axes[1, 1].imshow(nucleus_pred, cmap='Blues', alpha=0.3)
        axes[1, 1].set_title('Nucleus Comparison\n(Red:GT, Blue:Pred)', fontsize=12)
        axes[1, 1].axis('off')
        
        # 細胞質比較
        cytoplasm_pred = (pred_mask == 2).astype(np.uint8) * 255
        cytoplasm_gt = (gt_mask == 2).astype(np.uint8) * 255
        axes[1, 2].imshow(cytoplasm_gt, cmap='Greens', alpha=0.7)
        axes[1, 2].imshow(cytoplasm_pred, cmap='Blues', alpha=0.3)
        axes[1, 2].set_title('Cytoplasm Comparison\n(Green:GT, Blue:Pred)', fontsize=12)
        axes[1, 2].axis('off')
        
        # メトリクス表示
        if metrics_data:
            metrics_text = f"""Evaluation Metrics

Overall Performance:
  Accuracy: {metrics_data.get('accuracy', 0):.4f}
  Mean Precision: {metrics_data.get('mean_precision', 0):.4f}
  Mean Recall: {metrics_data.get('mean_recall', 0):.4f}
  Mean F1-Score: {metrics_data.get('mean_f1', 0):.4f}

Class-wise Precision:
  Background: {metrics_data.get('precision_class0', 0):.4f}
  Nucleus: {metrics_data.get('precision_class1', 0):.4f}
  Cytoplasm: {metrics_data.get('precision_class2', 0):.4f}

Class-wise F1-Score:
  Background: {metrics_data.get('f1_class0', 0):.4f}
  Nucleus: {metrics_data.get('f1_class1', 0):.4f}
  Cytoplasm: {metrics_data.get('f1_class2', 0):.4f}

Dice Coefficient:
  Background: {metrics_data.get('dice_class0', 0):.4f}
  Nucleus: {metrics_data.get('dice_class1', 0):.4f}
  Cytoplasm: {metrics_data.get('dice_class2', 0):.4f}

IoU:
  Background: {metrics_data.get('iou_class0', 0):.4f}
  Nucleus: {metrics_data.get('iou_class1', 0):.4f}
  Cytoplasm: {metrics_data.get('iou_class2', 0):.4f}"""
        else:
            metrics_text = "No metrics data available"
        
        axes[1, 3].text(0.02, 0.98, metrics_text, transform=axes[1, 3].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 3].set_xlim(0, 1)
        axes[1, 3].set_ylim(0, 1)
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        # 保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_visualizations(self, gt_path, pred_path, save_dir, 
                              original_path=None, metrics_csv=None, max_samples=None):
        """可視化画像一括生成"""
        
        # ディレクトリ作成
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル一覧取得
        gt_files = sorted(glob.glob(os.path.join(gt_path, "*.png")))
        
        if max_samples:
            gt_files = gt_files[:max_samples]
        
        # メトリクスデータ読み込み
        metrics_data = {}
        if metrics_csv and os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            for _, row in df.iterrows():
                metrics_data[row['filename']] = row.to_dict()
        
        print(f"🎨 Generating visualizations for {len(gt_files)} images...")
        
        generated_files = []
        
        for gt_file in tqdm(gt_files, desc="Generating Visualizations"):
            filename = os.path.basename(gt_file)
            pred_file = os.path.join(pred_path, filename)
            
            # 予測ファイルチェック
            if not os.path.exists(pred_file):
                print(f"⚠️ Warning: {filename} prediction not found, skipping...")
                continue
            
            # 原画像パス
            original_file = None
            if original_path:
                original_file = os.path.join(original_path, filename)
                if not os.path.exists(original_file):
                    original_file = None
            
            # メトリクスデータ取得
            sample_metrics = metrics_data.get(filename, {})
            
            # 可視化画像保存パス
            save_name = filename.replace('.png', '_comparison.png')
            save_path = save_dir / save_name
            
            # 可視化生成
            self.create_comparison_visualization(
                original_file, gt_file, pred_file, sample_metrics, save_path
            )
            
            generated_files.append(str(save_path))
        
        print(f"\n✅ Visualization generation completed!")
        print(f"📊 Generated {len(generated_files)} visualization files")
        print(f"📂 Saved to: {save_dir}")
        
        return generated_files

def main():
    parser = argparse.ArgumentParser(description='Generate CellMix visualization comparisons')
    parser.add_argument('--gt_path', type=str, required=True, 
                       help='Ground truth images directory')
    parser.add_argument('--pred_path', type=str, required=True, 
                       help='Prediction images directory')
    parser.add_argument('--save_dir', type=str, required=True, 
                       help='Output directory for visualizations')
    parser.add_argument('--original_path', type=str, default=None,
                       help='Original images directory (optional)')
    parser.add_argument('--metrics_csv', type=str, default=None,
                       help='Metrics CSV file for displaying on visualizations (optional)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of visualizations to generate')
    
    args = parser.parse_args()
    
    # 可視化生成器初期化
    visualizer = CellMixVisualizationGenerator()
    
    # 可視化生成実行
    generated_files = visualizer.generate_visualizations(
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        save_dir=args.save_dir,
        original_path=args.original_path,
        metrics_csv=args.metrics_csv,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()