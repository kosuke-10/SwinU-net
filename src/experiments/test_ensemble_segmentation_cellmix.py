import torch
import torch.utils.data as data
import numpy as np
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import sys
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

class CellMixEnsembleEvaluator:
    """CellMix用アンサンブルモデルによるセグメンテーション評価"""
    
    def __init__(self, ensemble_dir, device='cuda'):
        self.ensemble_dir = ensemble_dir
        self.device = device
        
        # CellMixクラス定義
        self.class_names = ['Background', 'Nucleus', 'Cytoplasm']
        self.class_colors = {
            0: [0, 0, 0],         # 背景: 黒
            1: [128, 0, 0],       # 核: 暗赤
            2: [0, 128, 0],       # 細胞質: 暗緑
        }
        
        # アンサンブルモデル読み込み
        print("📥 Loading CellMix ensemble model...")
        self.model, self.info, self.config = load_ensemble_model(ensemble_dir)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        print(f"✅ Ensemble model loaded with {self.info['num_models']} models")
        
        # 結果保存ディレクトリ
        self.results_dir = Path(ensemble_dir) / "cellmix_test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # サブディレクトリ
        self.segmentation_dir = self.results_dir / "segmentations"
        self.comparison_dir = self.results_dir / "comparisons"
        self.metrics_dir = self.results_dir / "metrics"
        
        for d in [self.segmentation_dir, self.comparison_dir, self.metrics_dir]:
            d.mkdir(exist_ok=True)
    
    def tensor_to_numpy_2d(self, tensor):
        """PyTorchテンソルを2次元numpy配列に変換（配列形状問題解決）"""
        if isinstance(tensor, torch.Tensor):
            # GPU -> CPU
            if tensor.is_cuda:
                tensor = tensor.cpu()
            # numpy変換
            np_array = tensor.numpy()
        else:
            np_array = tensor
        
        # 形状を2次元に調整
        if len(np_array.shape) == 4:  # (1, 1, H, W) -> (H, W)
            np_array = np_array.squeeze()
        elif len(np_array.shape) == 3:  # (1, H, W) or (C, H, W) -> (H, W)
            if np_array.shape[0] == 1:
                np_array = np_array.squeeze(0)
            else:
                np_array = np_array[0]  # 最初のチャンネルを取得
        elif len(np_array.shape) == 2:  # (H, W) - そのまま
            pass
        else:
            raise ValueError(f"Unexpected array shape: {np_array.shape}")
        
        return np_array.astype(np.uint8)
    
    def tensor_to_numpy_3d(self, tensor):
        """PyTorchテンソルを3次元numpy配列に変換（RGB画像用）"""
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            np_array = tensor.numpy()
        else:
            np_array = tensor
        
        # 形状を(H, W, C)に調整
        if len(np_array.shape) == 4:  # (1, C, H, W) -> (H, W, C)
            np_array = np_array.squeeze(0)
            if np_array.shape[0] in [1, 3]:  # チャンネル次元が最初
                np_array = np.transpose(np_array, (1, 2, 0))
        elif len(np_array.shape) == 3:
            if np_array.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
                np_array = np.transpose(np_array, (1, 2, 0))
        
        # 0-1範囲なら0-255に変換
        if np_array.max() <= 1.0:
            np_array = (np_array * 255).astype(np.uint8)
        else:
            np_array = np_array.astype(np.uint8)
        
        return np_array
    
    def calculate_dice_score(self, pred_mask, true_mask, class_id):
        """Dice係数計算"""
        pred_class = (pred_mask == class_id).float()
        true_class = (true_mask == class_id).float()
        
        intersection = (pred_class * true_class).sum()
        total = pred_class.sum() + true_class.sum()
        
        if total > 0:
            dice = (2 * intersection / total).item()
        else:
            dice = 1.0  # 両方とも空の場合は完全一致
        
        return dice
    
    def calculate_iou(self, pred_mask, true_mask, class_id):
        """IoU (Intersection over Union) 計算"""
        pred_class = (pred_mask == class_id).float()
        true_class = (true_mask == class_id).float()
        
        intersection = (pred_class * true_class).sum()
        union = pred_class.sum() + true_class.sum() - intersection
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 1.0
        
        return iou
    
    def calculate_sklearn_metrics(self, pred_mask, true_mask):
        """scikit-learnベースの評価指標計算"""
        # numpy配列に変換して平坦化
        pred_flat = self.tensor_to_numpy_2d(pred_mask).flatten()
        true_flat = self.tensor_to_numpy_2d(true_mask).flatten()
        
        # 基本指標
        accuracy = accuracy_score(true_flat, pred_flat)
        
        # クラス別指標（zero_divisionで0除算エラーを防ぐ）
        precision_values = precision_score(true_flat, pred_flat, average=None, 
                                         labels=[0, 1, 2], zero_division=0)
        recall_values = recall_score(true_flat, pred_flat, average=None, 
                                   labels=[0, 1, 2], zero_division=0)
        f1_values = f1_score(true_flat, pred_flat, average=None, 
                           labels=[0, 1, 2], zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision_values,
            'recall': recall_values,
            'f1': f1_values
        }
    
    def mask_to_color(self, mask):
        """セグメンテーション マスクをカラー画像に変換"""
        # 2次元配列に変換
        mask_2d = self.tensor_to_numpy_2d(mask)
        h, w = mask_2d.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            color_mask[mask_2d == class_id] = color
            
        return color_mask
    
    def save_individual_predictions(self, prediction, ground_truth, case_name, sample_id):
        """個別予測結果を保存（グレースケール）"""
        # ✅ 修正：2次元配列に変換
        pred_2d = self.tensor_to_numpy_2d(prediction)
        gt_2d = self.tensor_to_numpy_2d(ground_truth)
        
        # 予測結果をグレースケール画像として保存
        pred_path = self.segmentation_dir / f"sample_{sample_id:03d}_{case_name}_prediction.png"
        gt_path = self.segmentation_dir / f"sample_{sample_id:03d}_{case_name}_groundtruth.png"
        
        Image.fromarray(pred_2d).save(pred_path)
        Image.fromarray(gt_2d).save(gt_path)
        
        return pred_path, gt_path
    
    def save_comparison_visualization(self, image, prediction, ground_truth, 
                                    sample_id, case_name, metrics):
        """詳細比較可視化を保存"""
        
        # numpy配列に変換
        image_3d = self.tensor_to_numpy_3d(image)
        pred_2d = self.tensor_to_numpy_2d(prediction)
        gt_2d = self.tensor_to_numpy_2d(ground_truth)
        
        # カラー画像に変換
        pred_color = self.mask_to_color(pred_2d)
        gt_color = self.mask_to_color(gt_2d)
        
        # 比較画像作成
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 上段
        # 元画像
        axes[0, 0].imshow(image_3d)
        axes[0, 0].set_title(f'Original Image\n{case_name}', fontsize=12)
        axes[0, 0].axis('off')
        
        # 正解ラベル
        axes[0, 1].imshow(gt_color)
        axes[0, 1].set_title('Ground Truth\n(Black:Background, Red:Nucleus, Green:Cytoplasm)', fontsize=12)
        axes[0, 1].axis('off')
        
        # 予測結果
        axes[0, 2].imshow(pred_color)
        axes[0, 2].set_title('Ensemble Prediction\n(Black:Background, Red:Nucleus, Green:Cytoplasm)', fontsize=12)
        axes[0, 2].axis('off')
        
        # オーバーレイ（元画像 + 予測）
        overlay_pred = cv2.addWeighted(image_3d, 0.6, pred_color, 0.4, 0)
        axes[0, 3].imshow(overlay_pred)
        axes[0, 3].set_title('Prediction Overlay', fontsize=12)
        axes[0, 3].axis('off')
        
        # 下段
        # 差分画像
        diff_mask = np.abs(pred_2d.astype(int) - gt_2d.astype(int))
        diff_color = np.zeros((diff_mask.shape[0], diff_mask.shape[1], 3), dtype=np.uint8)
        diff_color[diff_mask > 0] = [255, 255, 0]  # 差分を黄色で表示
        axes[1, 0].imshow(diff_color)
        axes[1, 0].set_title('Difference (Yellow = Error)', fontsize=12)
        axes[1, 0].axis('off')
        
        # クラス別比較（核のみ）
        nucleus_pred = (pred_2d == 1).astype(np.uint8) * 255
        nucleus_gt = (gt_2d == 1).astype(np.uint8) * 255
        axes[1, 1].imshow(nucleus_gt, cmap='Reds', alpha=0.7)
        axes[1, 1].imshow(nucleus_pred, cmap='Blues', alpha=0.3)
        axes[1, 1].set_title('Nucleus Comparison\n(Red:GT, Blue:Pred)', fontsize=12)
        axes[1, 1].axis('off')
        
        # クラス別比較（細胞質のみ）
        cytoplasm_pred = (pred_2d == 2).astype(np.uint8) * 255
        cytoplasm_gt = (gt_2d == 2).astype(np.uint8) * 255
        axes[1, 2].imshow(cytoplasm_gt, cmap='Greens', alpha=0.7)
        axes[1, 2].imshow(cytoplasm_pred, cmap='Blues', alpha=0.3)
        axes[1, 2].set_title('Cytoplasm Comparison\n(Green:GT, Blue:Pred)', fontsize=12)
        axes[1, 2].axis('off')
        
        # メトリクス表示
        metrics_text = f"""CellMix Evaluation Metrics

Overall Performance:
  Accuracy: {metrics['accuracy']:.4f}
  Mean Precision: {metrics['mean_precision']:.4f}
  Mean Recall: {metrics['mean_recall']:.4f}
  Mean F1-Score: {metrics['mean_f1']:.4f}

Class-wise Precision:
  Background: {metrics['precision_class0']:.4f}
  Nucleus: {metrics['precision_class1']:.4f}
  Cytoplasm: {metrics['precision_class2']:.4f}

Class-wise Recall:
  Background: {metrics['recall_class0']:.4f}
  Nucleus: {metrics['recall_class1']:.4f}
  Cytoplasm: {metrics['recall_class2']:.4f}

Class-wise F1-Score:
  Background: {metrics['f1_class0']:.4f}
  Nucleus: {metrics['f1_class1']:.4f}
  Cytoplasm: {metrics['f1_class2']:.4f}

Dice Coefficient:
  Background: {metrics['dice_scores'][0]:.4f}
  Nucleus: {metrics['dice_scores'][1]:.4f}
  Cytoplasm: {metrics['dice_scores'][2]:.4f}
  Mean Dice: {metrics['mean_dice']:.4f}

IoU:
  Background: {metrics['iou_scores'][0]:.4f}
  Nucleus: {metrics['iou_scores'][1]:.4f}
  Cytoplasm: {metrics['iou_scores'][2]:.4f}
  Mean IoU: {metrics['mean_iou']:.4f}"""
        
        axes[1, 3].text(0.02, 0.98, metrics_text, transform=axes[1, 3].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 3].set_xlim(0, 1)
        axes[1, 3].set_ylim(0, 1)
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        # 保存
        comparison_path = self.comparison_dir / f"sample_{sample_id:03d}_{case_name}_detailed.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_path
    
    def test_on_cellmix(self, max_samples=None, save_visualizations=True):
        """CellMixテストデータセットでの評価実行"""
        
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
        
        print(f"🧪 Testing on {max_samples} samples...")
        
        # 結果記録用
        all_metrics = []
        class_dice_scores = [[] for _ in range(3)]
        class_iou_scores = [[] for _ in range(3)]
        
        # sklearn metrics用の蓄積
        all_accuracies = []
        all_precisions = [[] for _ in range(3)]
        all_recalls = [[] for _ in range(3)]
        all_f1s = [[] for _ in range(3)]
        
        # テスト実行
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Evaluating CellMix")):
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
                prediction_cpu = prediction.cpu()
                label_cpu = label.cpu()
                image_cpu = image.cpu()
                
                # Dice & IoU計算
                sample_metrics = {
                    'sample_id': i,
                    'case_name': case_name,
                    'dice_scores': [],
                    'iou_scores': []
                }
                
                for class_id in range(3):
                    dice = self.calculate_dice_score(prediction_cpu, label_cpu, class_id)
                    iou = self.calculate_iou(prediction_cpu, label_cpu, class_id)
                    
                    sample_metrics['dice_scores'].append(dice)
                    sample_metrics['iou_scores'].append(iou)
                    
                    class_dice_scores[class_id].append(dice)
                    class_iou_scores[class_id].append(iou)
                
                # sklearn metrics計算
                sklearn_metrics = self.calculate_sklearn_metrics(prediction_cpu, label_cpu)
                
                sample_metrics['accuracy'] = sklearn_metrics['accuracy']
                sample_metrics['precision_class0'] = sklearn_metrics['precision'][0]
                sample_metrics['precision_class1'] = sklearn_metrics['precision'][1]
                sample_metrics['precision_class2'] = sklearn_metrics['precision'][2]
                sample_metrics['recall_class0'] = sklearn_metrics['recall'][0]
                sample_metrics['recall_class1'] = sklearn_metrics['recall'][1]
                sample_metrics['recall_class2'] = sklearn_metrics['recall'][2]
                sample_metrics['f1_class0'] = sklearn_metrics['f1'][0]
                sample_metrics['f1_class1'] = sklearn_metrics['f1'][1]
                sample_metrics['f1_class2'] = sklearn_metrics['f1'][2]
                
                # 平均値計算
                sample_metrics['mean_dice'] = np.mean(sample_metrics['dice_scores'])
                sample_metrics['mean_iou'] = np.mean(sample_metrics['iou_scores'])
                sample_metrics['mean_precision'] = np.mean(sklearn_metrics['precision'])
                sample_metrics['mean_recall'] = np.mean(sklearn_metrics['recall'])
                sample_metrics['mean_f1'] = np.mean(sklearn_metrics['f1'])
                
                # 前景クラス（核・細胞質）のみの平均
                sample_metrics['mean_dice_fg'] = np.mean(sample_metrics['dice_scores'][1:])
                sample_metrics['mean_iou_fg'] = np.mean(sample_metrics['iou_scores'][1:])
                sample_metrics['mean_precision_fg'] = np.mean(sklearn_metrics['precision'][1:])
                sample_metrics['mean_recall_fg'] = np.mean(sklearn_metrics['recall'][1:])
                sample_metrics['mean_f1_fg'] = np.mean(sklearn_metrics['f1'][1:])
                
                all_metrics.append(sample_metrics)
                
                # 蓄積（全体統計用）
                all_accuracies.append(sklearn_metrics['accuracy'])
                for c in range(3):
                    all_precisions[c].append(sklearn_metrics['precision'][c])
                    all_recalls[c].append(sklearn_metrics['recall'][c])
                    all_f1s[c].append(sklearn_metrics['f1'][c])
                
                # 個別予測画像保存（全サンプル）
                self.save_individual_predictions(prediction_cpu, label_cpu, case_name, i)
                
                # ✅ 修正：全サンプルで詳細可視化保存
                if save_visualizations:  # max_samples <= 20 の条件を削除
                    self.save_comparison_visualization(
                        image_cpu, prediction_cpu, label_cpu, i, case_name, sample_metrics
                    )
                    
                # 進行状況の詳細表示（10サンプルごと）
                if (i + 1) % 10 == 0:
                    current_dice = np.mean([m['mean_dice_fg'] for m in all_metrics])
                    current_f1 = np.mean([m['mean_f1_fg'] for m in all_metrics])
                    print(f"\n📊 Progress [{i+1}/{max_samples}]: Current Avg Dice (FG): {current_dice:.4f}, F1 (FG): {current_f1:.4f}")
        
        
        # 全体結果計算
        overall_results = self.calculate_overall_results(
            all_metrics, class_dice_scores, class_iou_scores, 
            all_accuracies, all_precisions, all_recalls, all_f1s
        )
        
        # 結果保存
        self.save_results(overall_results, all_metrics)
        
        return overall_results, all_metrics
    
    def calculate_overall_results(self, all_metrics, class_dice_scores, class_iou_scores,
                                all_accuracies, all_precisions, all_recalls, all_f1s):
        """全体結果の集計"""
        
        # 全サンプル平均
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
                'std': np.std(class_dice_scores[class_id]),
                'min': np.min(class_dice_scores[class_id]),
                'max': np.max(class_dice_scores[class_id])
            }
            
            overall_results['class_wise_iou'][class_name] = {
                'mean': np.mean(class_iou_scores[class_id]),
                'std': np.std(class_iou_scores[class_id]),
                'min': np.min(class_iou_scores[class_id]),
                'max': np.max(class_iou_scores[class_id])
            }
            
            overall_results['class_wise_precision'][class_name] = {
                'mean': np.mean(all_precisions[class_id]),
                'std': np.std(all_precisions[class_id]),
                'min': np.min(all_precisions[class_id]),
                'max': np.max(all_precisions[class_id])
            }
            
            overall_results['class_wise_recall'][class_name] = {
                'mean': np.mean(all_recalls[class_id]),
                'std': np.std(all_recalls[class_id]),
                'min': np.min(all_recalls[class_id]),
                'max': np.max(all_recalls[class_id])
            }
            
            overall_results['class_wise_f1'][class_name] = {
                'mean': np.mean(all_f1s[class_id]),
                'std': np.std(all_f1s[class_id]),
                'min': np.min(all_f1s[class_id]),
                'max': np.max(all_f1s[class_id])
            }
        
        return overall_results
    
    def save_results(self, overall_results, all_metrics):
        """結果をJSON・CSVで保存"""
        
        # JSONで詳細結果保存
        results_json = self.results_dir / "cellmix_evaluation_results.json"
        overall_results['timestamp'] = str(datetime.now())
        overall_results['ensemble_info'] = self.info
        
        with open(results_json, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        # CSV形式でサンプル別結果保存
        df_metrics = pd.DataFrame(all_metrics)
        csv_path = self.results_dir / "cellmix_sample_metrics.csv"
        df_metrics.to_csv(csv_path, index=False)
        
        # 平均値のみのCSV（compute_metrics.py形式に近い形）
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
        avg_csv_path = self.results_dir / "cellmix_average_metrics.csv"
        avg_df.to_csv(avg_csv_path, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 Detailed JSON: {results_json}")
        print(f"   📊 Sample CSV: {csv_path}")
        print(f"   📊 Average CSV: {avg_csv_path}")
        print(f"   🖼️  Comparisons: {self.comparison_dir}")
        print(f"   🎯 Segmentations: {self.segmentation_dir}")
        
        return results_json, csv_path
    
    def print_results_summary(self, overall_results):
        """詳細結果サマリーの表示"""
        
        print(f"\n{'='*80}")
        print(f"🧬 CellMix Ensemble Segmentation Evaluation Results")
        print(f"{'='*80}")
        print(f"📊 Total samples tested: {overall_results['num_samples']}")
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
        
        print(f"\n📂 Results saved to: {self.results_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CellMix Ensemble Segmentation Evaluation')
    parser.add_argument('--ensemble_dir', type=str,
                       default='experiments/cellmix_kfold_20250718_145719/ensemble_model',
                       help='Ensemble model directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to test (default: all 78)')
    parser.add_argument('--no_vis', action='store_true',
                       help='Skip detailed visualization saving')
    
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    os.chdir(Path(__file__).parent.parent.parent)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # CellMix評価器初期化
    evaluator = CellMixEnsembleEvaluator(
        ensemble_dir=args.ensemble_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # テスト実行
    overall_results, all_metrics = evaluator.test_on_cellmix(
        max_samples=args.max_samples,
        save_visualizations=not args.no_vis
    )
    
    # 結果表示
    evaluator.print_results_summary(overall_results)

if __name__ == '__main__':
    main()