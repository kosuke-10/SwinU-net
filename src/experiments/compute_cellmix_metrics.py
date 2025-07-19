import os
import numpy as np
import pandas as pd
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

class CellMixMetricsCalculator:
    """CellMix用評価指標計算専用クラス"""
    
    def __init__(self):
        # CellMixクラス定義
        self.class_names = ['Background', 'Nucleus', 'Cytoplasm']
        self.num_classes = 3
    
    def calculate_dice_score(self, pred_flat, true_flat, class_id):
        """Dice係数計算"""
        pred_class = (pred_flat == class_id).astype(float)
        true_class = (true_flat == class_id).astype(float)
        
        intersection = (pred_class * true_class).sum()
        total = pred_class.sum() + true_class.sum()
        
        if total > 0:
            dice = (2 * intersection / total)
        else:
            dice = 1.0  # 両方とも空の場合は完全一致
        
        return dice
    
    def calculate_iou(self, pred_flat, true_flat, class_id):
        """IoU計算"""
        pred_class = (pred_flat == class_id).astype(float)
        true_class = (true_flat == class_id).astype(float)
        
        intersection = (pred_class * true_class).sum()
        union = pred_class.sum() + true_class.sum() - intersection
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0
        
        return iou
    
    def compute_metrics_for_images(self, gt_path, pred_path, save_path):
        """
        セグメンテーション結果と正解ラベルを比較して評価指標を計算
        """
        # ファイル一覧取得
        gt_files = sorted(glob.glob(os.path.join(gt_path, "*.png")))
        
        # 評価指標を格納するリスト
        all_metrics = []
        
        # クラス別スコア蓄積用
        class_dice_scores = [[] for _ in range(self.num_classes)]
        class_iou_scores = [[] for _ in range(self.num_classes)]
        all_accuracies = []
        all_precisions = [[] for _ in range(self.num_classes)]
        all_recalls = [[] for _ in range(self.num_classes)]
        all_f1s = [[] for _ in range(self.num_classes)]
        
        print(f"📊 Computing metrics for {len(gt_files)} images...")
        
        # 各ファイルに対して評価指標を計算
        for gt_file in tqdm(gt_files, desc="Computing Metrics"):
            filename = os.path.basename(gt_file)
            pred_file = os.path.join(pred_path, filename)
            
            # 予測ファイルが存在しない場合はスキップ
            if not os.path.exists(pred_file):
                print(f"⚠️ Warning: {filename} prediction not found, skipping...")
                continue
            
            # 画像読み込み
            gt_img = np.array(Image.open(gt_file))
            pred_img = np.array(Image.open(pred_file))
            
            # 画像サイズチェック
            if gt_img.shape != pred_img.shape:
                print(f"⚠️ Warning: {filename} size mismatch - GT: {gt_img.shape}, Pred: {pred_img.shape}")
                continue
            
            # 1次元に平坦化
            gt_flat = gt_img.flatten()
            pred_flat = pred_img.flatten()
            
            # 基本評価指標
            accuracy = accuracy_score(gt_flat, pred_flat)
            
            # クラス別評価指標
            precision_values = precision_score(gt_flat, pred_flat, average=None, 
                                             labels=[0, 1, 2], zero_division=0)
            recall_values = recall_score(gt_flat, pred_flat, average=None, 
                                       labels=[0, 1, 2], zero_division=0)
            f1_values = f1_score(gt_flat, pred_flat, average=None, 
                               labels=[0, 1, 2], zero_division=0)
            
            # Dice・IoU計算
            dice_scores = []
            iou_scores = []
            
            for class_id in range(self.num_classes):
                dice = self.calculate_dice_score(pred_flat, gt_flat, class_id)
                iou = self.calculate_iou(pred_flat, gt_flat, class_id)
                
                dice_scores.append(dice)
                iou_scores.append(iou)
                
                # クラス別蓄積
                class_dice_scores[class_id].append(dice)
                class_iou_scores[class_id].append(iou)
            
            # サンプル別メトリクス
            sample_metrics = {
                'filename': filename,
                'accuracy': accuracy,
                'precision_class0': precision_values[0] if len(precision_values) > 0 else 0,
                'precision_class1': precision_values[1] if len(precision_values) > 1 else 0,
                'precision_class2': precision_values[2] if len(precision_values) > 2 else 0,
                'recall_class0': recall_values[0] if len(recall_values) > 0 else 0,
                'recall_class1': recall_values[1] if len(recall_values) > 1 else 0,
                'recall_class2': recall_values[2] if len(recall_values) > 2 else 0,
                'f1_class0': f1_values[0] if len(f1_values) > 0 else 0,
                'f1_class1': f1_values[1] if len(f1_values) > 1 else 0,
                'f1_class2': f1_values[2] if len(f1_values) > 2 else 0,
                'dice_class0': dice_scores[0],
                'dice_class1': dice_scores[1],
                'dice_class2': dice_scores[2],
                'iou_class0': iou_scores[0],
                'iou_class1': iou_scores[1],
                'iou_class2': iou_scores[2],
                'mean_precision': np.mean(precision_values),
                'mean_recall': np.mean(recall_values),
                'mean_f1': np.mean(f1_values),
                'mean_dice': np.mean(dice_scores),
                'mean_iou': np.mean(iou_scores),
                'mean_precision_fg': np.mean(precision_values[1:]),  # 前景のみ
                'mean_recall_fg': np.mean(recall_values[1:]),
                'mean_f1_fg': np.mean(f1_values[1:]),
                'mean_dice_fg': np.mean(dice_scores[1:]),
                'mean_iou_fg': np.mean(iou_scores[1:])
            }
            
            all_metrics.append(sample_metrics)
            
            # 全体統計用蓄積
            all_accuracies.append(accuracy)
            for c in range(self.num_classes):
                if c < len(precision_values):
                    all_precisions[c].append(precision_values[c])
                    all_recalls[c].append(recall_values[c])
                    all_f1s[c].append(f1_values[c])
                else:
                    all_precisions[c].append(0)
                    all_recalls[c].append(0)
                    all_f1s[c].append(0)
        
        # 全体結果計算
        overall_results = self.calculate_overall_results(
            all_metrics, class_dice_scores, class_iou_scores,
            all_accuracies, all_precisions, all_recalls, all_f1s
        )
        
        # 結果保存
        self.save_results(overall_results, all_metrics, save_path)
        
        return overall_results, all_metrics
    
    def calculate_overall_results(self, all_metrics, class_dice_scores, class_iou_scores,
                                all_accuracies, all_precisions, all_recalls, all_f1s):
        """全体結果の集計"""
        
        overall_results = {
            'num_samples': len(all_metrics),
            'dataset': 'CellMix',
            'class_names': self.class_names,
            
            # 全体精度
            'overall_accuracy': np.mean(all_accuracies),
            'overall_accuracy_std': np.std(all_accuracies),
            
            # 全クラス含む平均
            'overall_mean_dice': np.mean([m['mean_dice'] for m in all_metrics]),
            'overall_mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
            'overall_mean_precision': np.mean([m['mean_precision'] for m in all_metrics]),
            'overall_mean_recall': np.mean([m['mean_recall'] for m in all_metrics]),
            'overall_mean_f1': np.mean([m['mean_f1'] for m in all_metrics]),
            
            # 前景クラス（核・細胞質）のみの平均
            'overall_mean_dice_fg': np.mean([m['mean_dice_fg'] for m in all_metrics]),
            'overall_mean_iou_fg': np.mean([m['mean_iou_fg'] for m in all_metrics]),
            'overall_mean_precision_fg': np.mean([m['mean_precision_fg'] for m in all_metrics]),
            'overall_mean_recall_fg': np.mean([m['mean_recall_fg'] for m in all_metrics]),
            'overall_mean_f1_fg': np.mean([m['mean_f1_fg'] for m in all_metrics]),
            
            'class_wise_dice': {},
            'class_wise_iou': {},
            'class_wise_precision': {},
            'class_wise_recall': {},
            'class_wise_f1': {}
        }
        
        # クラス別集計
        for class_id, class_name in enumerate(self.class_names):
            overall_results['class_wise_dice'][class_name] = {
                'mean': np.mean(class_dice_scores[class_id]),
                'std': np.std(class_dice_scores[class_id])
            }
            overall_results['class_wise_iou'][class_name] = {
                'mean': np.mean(class_iou_scores[class_id]),
                'std': np.std(class_iou_scores[class_id])
            }
            overall_results['class_wise_precision'][class_name] = {
                'mean': np.mean(all_precisions[class_id]),
                'std': np.std(all_precisions[class_id])
            }
            overall_results['class_wise_recall'][class_name] = {
                'mean': np.mean(all_recalls[class_id]),
                'std': np.std(all_recalls[class_id])
            }
            overall_results['class_wise_f1'][class_name] = {
                'mean': np.mean(all_f1s[class_id]),
                'std': np.std(all_f1s[class_id])
            }
        
        return overall_results
    
    def save_results(self, overall_results, all_metrics, save_path):
        """結果保存"""
        
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # サンプル別詳細CSV
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(save_path, index=False)
        
        # 平均値CSV（あなたの形式準拠）
        avg_metrics = {
            'accuracy': overall_results['overall_accuracy'],
            'precision_class0': overall_results['class_wise_precision']['Background']['mean'],
            'precision_class1': overall_results['class_wise_precision']['Nucleus']['mean'],
            'precision_class2': overall_results['class_wise_precision']['Cytoplasm']['mean'],
            'recall_class0': overall_results['class_wise_recall']['Background']['mean'],
            'recall_class1': overall_results['class_wise_recall']['Nucleus']['mean'],
            'recall_class2': overall_results['class_wise_recall']['Cytoplasm']['mean'],
            'f1_class0': overall_results['class_wise_f1']['Background']['mean'],
            'f1_class1': overall_results['class_wise_f1']['Nucleus']['mean'],
            'f1_class2': overall_results['class_wise_f1']['Cytoplasm']['mean'],
            'mean_precision': overall_results['overall_mean_precision'],
            'mean_recall': overall_results['overall_mean_recall'],
            'mean_f1': overall_results['overall_mean_f1'],
            'mean_dice': overall_results['overall_mean_dice'],
            'mean_iou': overall_results['overall_mean_iou']
        }
        
        avg_df = pd.DataFrame([avg_metrics])
        avg_path = str(save_path).replace('.csv', '_avg.csv')
        avg_df.to_csv(avg_path, index=False)
        
        # JSON詳細結果
        json_path = str(save_path).replace('.csv', '_detailed.json')
        overall_results['timestamp'] = str(datetime.now())
        
        with open(json_path, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Detailed CSV: {save_path}")
        print(f"   📊 Average CSV: {avg_path}")
        print(f"   📄 JSON: {json_path}")
        
        return save_path, avg_path, json_path
    
    def print_results_summary(self, overall_results):
        """結果サマリー表示"""
        
        print(f"\n{'='*80}")
        print(f"🧬 CellMix Evaluation Results")
        print(f"{'='*80}")
        print(f"📊 Total samples: {overall_results['num_samples']}")
        print(f"📈 Overall Accuracy: {overall_results['overall_accuracy']:.4f} ± {overall_results['overall_accuracy_std']:.4f}")
        
        print(f"\n🎯 Overall Performance (All Classes):")
        print(f"   Mean Precision: {overall_results['overall_mean_precision']:.4f}")
        print(f"   Mean Recall:    {overall_results['overall_mean_recall']:.4f}")
        print(f"   Mean F1-Score:  {overall_results['overall_mean_f1']:.4f}")
        print(f"   Mean Dice:      {overall_results['overall_mean_dice']:.4f}")
        print(f"   Mean IoU:       {overall_results['overall_mean_iou']:.4f}")
        
        print(f"\n🎯 Foreground Performance (Nucleus + Cytoplasm only):")
        print(f"   Mean Precision: {overall_results['overall_mean_precision_fg']:.4f}")
        print(f"   Mean Recall:    {overall_results['overall_mean_recall_fg']:.4f}")
        print(f"   Mean F1-Score:  {overall_results['overall_mean_f1_fg']:.4f}")
        print(f"   Mean Dice:      {overall_results['overall_mean_dice_fg']:.4f}")
        print(f"   Mean IoU:       {overall_results['overall_mean_iou_fg']:.4f}")
        
        print(f"\n📊 Class-wise Performance:")
        for class_name in self.class_names:
            print(f"\n   🧬 {class_name}:")
            print(f"      Precision: {overall_results['class_wise_precision'][class_name]['mean']:.4f} ± {overall_results['class_wise_precision'][class_name]['std']:.4f}")
            print(f"      Recall:    {overall_results['class_wise_recall'][class_name]['mean']:.4f} ± {overall_results['class_wise_recall'][class_name]['std']:.4f}")
            print(f"      F1-Score:  {overall_results['class_wise_f1'][class_name]['mean']:.4f} ± {overall_results['class_wise_f1'][class_name]['std']:.4f}")
            print(f"      Dice:      {overall_results['class_wise_dice'][class_name]['mean']:.4f} ± {overall_results['class_wise_dice'][class_name]['std']:.4f}")
            print(f"      IoU:       {overall_results['class_wise_iou'][class_name]['mean']:.4f} ± {overall_results['class_wise_iou'][class_name]['std']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Compute CellMix segmentation metrics')
    parser.add_argument('--gt_path', type=str, required=True, 
                       help='Ground truth images directory')
    parser.add_argument('--pred_path', type=str, required=True, 
                       help='Prediction images directory')
    parser.add_argument('--save_path', type=str, required=True, 
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # メトリクス計算器初期化
    calculator = CellMixMetricsCalculator()
    
    # メトリクス計算実行
    overall_results, all_metrics = calculator.compute_metrics_for_images(
        args.gt_path, args.pred_path, args.save_path
    )
    
    # 結果表示
    calculator.print_results_summary(overall_results)

if __name__ == "__main__":
    main()