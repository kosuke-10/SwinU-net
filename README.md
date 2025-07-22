# Swin-Unet  
【ECCVW2022】本リポジトリは、「Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation」（[arXivリンク](https://arxiv.org/abs/2105.05537)）で提案された手法のコードです。  
本論文は ECCV 2022 MEDICAL COMPUTER VISION WORKSHOP（[公式サイト](https://mcv-workshop.github.io/)）に採択されました。  
再現性向上のための修正を行っています。結果の再現に役立つことを願っています。

---

## 1. Swin Transformer（Swin-T）の事前学習済みモデルのダウンロード  
* [こちらのリンクから事前学習済みモデルを取得](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing)  
  → ダウンロードしたファイルを `pretrained_ckpt/` フォルダに配置してください。

---

## 2. データの準備  

使用データセットは TransUnet の著者によって提供されています。以下のリンクから取得可能です：  
- Synapse/BTCV: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd  
- ACDC: https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4


細胞データセットをdatasetの内部に配置（名前はCellMix）
元画像はRGB・ラベル画像はL形式である必要あり

---

## 3. 実行環境 （修正版）

GPU：NVIDIA RTX A5000 VRAM 24G
nvidia driver：524.105.17
CUDA：11.0 ~ 11.8 （11.8使用）

Dockerfileの実行でOK（内容を確認して適宜修正すること）
```bash
cd SwinU-net/Docker
sh docker.sh サーバー番号
```

### Condaの有効化
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate swinunet
```

### conda仮想環境から抜ける
```bash
conda deactivate
```


---

## 4. 学習・テストの実行方法

### データ前処理
データ分割の実行(通常)
```bash
cd dataset_preprocessing
python3 make_dataset_cellmix.py
```

交差検定を行う場合
```bash
cd dataset_preprocessing
python3 make_dataset_cellmix_kfold.py
```

### 実験

バッチサイズは24推奨・500epochくらい？

### **🚀 Step 1: 学習実行**
```bash
cd /home/yoshida/Swin-Unet

# 5-fold交差検定で学習（推奨: 150-200エポック）
python3 run_experiments.py --mode kfold-all --epochs 400 --batch_size 24

# 実行結果
# → experiments/cellmix_kfold_YYYYMMDD_HHMMSS/ に保存
# → fold_0/, fold_1/, fold_2/, fold_3/, fold_4/ が作成される
```

```bash
# 基本学習
python3 run_experiments.py --mode train --epochs 150 --batch_size 24

# K-fold全実行
python3 run_experiments.py --mode kfold-all --epochs 150 --batch_size 24

# 特定fold実行
python3 run_experiments.py --mode kfold-single --fold 2 --epochs 100

# アンサンブル作成
python3 run_experiments.py --mode ensemble
```


### **🔗 Step 2: アンサンブルモデル作成**
```bash
# Step1で作成されたフォルダを指定
python3 run_experiments.py --mode ensemble \
  --experiment_dir experiments/cellmix_kfold_YYYYMMDD_HHMMSS

# 実行結果
# → experiments/cellmix_kfold_YYYYMMDD_HHMMSS/ensemble_model/ が作成
# → 5つのモデルを統合したアンサンブルモデル完成
```

### **🎯 Step 3: テスト実行（セグメント画像生成）**
```bash
# アンサンブルモデルでテストデータを推論
python3 src/experiments/generate_segmentations.py \
  --ensemble_dir experiments/cellmix_kfold_YYYYMMDD_HHMMSS/ensemble_model

# 実行結果（約3-4分）
# → experiments/cellmix_kfold_YYYYMMDD_HHMMSS/cellmix_test_results/ に保存
# → individual_predictions/ : グレースケール予測画像 (0,1,2値)
# → individual_groundtruths/ : グレースケール正解画像 (0,1,2値) 
# → predictions_colored/ : カラー予測画像 (RGB)
# → groundtruths_colored/ : カラー正解画像 (RGB)
```

### **📊 Step 4: 評価指標計算**
```bash
# 生成されたセグメント画像で評価指標を計算
python3 src/experiments/compute_cellmix_metrics.py \
  --gt_path experiments/cellmix_kfold_YYYYMMDD_HHMMSS/cellmix_test_results/individual_groundtruths \
  --pred_path experiments/cellmix_kfold_YYYYMMDD_HHMMSS/cellmix_test_results/individual_predictions \
  --save_path results/cellmix_ensemble_metrics.csv

# 実行結果（約30秒）
# → results/cellmix_ensemble_metrics.csv : サンプル別詳細評価
# → results/cellmix_ensemble_metrics_avg.csv : 平均評価指標
# → Dice, IoU, Precision, Recall, F1スコアなど
```

### **🎨 Step 5: 詳細可視化作成**
```bash
# 論文・発表用の詳細比較画像を作成
python3 src/experiments/visualize_cellmix_results.py \
  --gt_path experiments/cellmix_kfold_YYYYMMDD_HHMMSS/cellmix_test_results/individual_groundtruths \
  --pred_path experiments/cellmix_kfold_YYYYMMDD_HHMMSS/cellmix_test_results/individual_predictions \
  --save_dir experiments/cellmix_kfold_YYYYMMDD_HHMMSS/cellmix_test_results/visualizations \
  --metrics_csv results/cellmix_ensemble_metrics.csv

# 実行結果（約1-2分）
# → visualizations/ : 78個の8枚組み詳細比較画像
# → 原画像・正解・予測・差分・オーバーレイ・クラス別比較・メトリクス表示
```

---

## **📁 最終成果物の構造**

```
experiments/cellmix_kfold_YYYYMMDD_HHMMSS/
├── fold_0/                           # Fold 0 学習結果
├── fold_1/                           # Fold 1 学習結果  
├── fold_2/                           # Fold 2 学習結果
├── fold_3/                           # Fold 3 学習結果
├── fold_4/                           # Fold 4 学習結果
├── ensemble_model/                   # アンサンブルモデル
└── cellmix_test_results/            # テスト結果
    ├── individual_predictions/       # グレースケール予測 (78枚)
    ├── individual_groundtruths/      # グレースケール正解 (78枚)
    ├── predictions_colored/          # カラー予測 (78枚)
    ├── groundtruths_colored/         # カラー正解 (78枚) 
    └── visualizations/               # 8枚組み比較画像 (78枚)

results/
├── cellmix_ensemble_metrics.csv     # サンプル別詳細評価
├── cellmix_ensemble_metrics_avg.csv # 平均評価指標
└── cellmix_ensemble_metrics_detailed.json # JSON詳細結果
```

---

## **🔧 トラブルシューティング**

### **メモリ不足エラー**
```bash
# バッチサイズを下げる
python3 run_experiments.py --mode kfold-all --epochs 150 --batch_size 16

# テスト時のサンプル数を制限
python3 src/experiments/generate_segmentations.py --max_samples 20
```

### **パス指定間違い**
```bash
# 作成されたディレクトリを確認
ls -la experiments/

# 正しいパスを指定（例）
--ensemble_dir experiments/cellmix_kfold_20250719_123456/ensemble_model
```

## 再現性について

### コードについて
学習済みモデルは Huawei Cloud に保存されていますが、社内規定により外部にファイルを送信することはできません。

### セグメンテーション結果の再現について
論文で報告されたセグメンテーション結果を再現するには、使用する GPU の種類によって結果が異なる場合があることがわかっています。
本コードでは乱数シードを慎重に設定しており、同一種類の GPU 上であれば複数回の学習で一貫した結果が得られるはずです。
もし論文の結果と異なる場合は、学習率の調整を推奨します。
我々の実験では Tesla V100 を使用しています。
また、純粋なトランスフォーマーモデルでは事前学習が非常に重要です。
本研究では エンコーダとデコーダの両方に事前学習済み重みを適用しています（エンコーダのみに初期化するのではなく、デコーダも含めています）。

## 参考リンク
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)

## Citation

```bibtex
@InProceedings{swinunet,
author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
year = {2022}
}

@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
