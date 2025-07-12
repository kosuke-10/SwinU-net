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

### make_dataset_txt.pyの設定

1. **設定ファイルの修正**（Synapse用）:
   ```python
   # make_dataset_txt.py内のdatasets_configを以下に修正
   datasets_config = {
       'Synapse': {
           'data_dir': 'datasets/Synapse/train_npz',
           'num_classes': 9,  # Synapseは9クラス（0-8）
           'predict_head': 0
       },
   }

---

## 3. 実行環境 （修正版）

GPU：NVIDIA RTX A5000 VRAM 24G
nvidia driver：524.105.17
CUDA：11.0 ~ 11.8 （11.8使用）


Python 3.7 の環境を準備してください。その後、以下のコマンドで依存ライブラリをインストールします：
↓Dockerfileで直接読み込む形に変更

```bash
pip install -r requirements.txt
```

---

## 4. 学習・テストの実行方法

Synapseデータセットの使用

- Synapse データセットで学習スクリプトを実行します。バッチサイズは 24 を推奨していますが、GPU メモリの制約がある場合は 12 や 6 に変更可能です。

- Train

```bash
sh train.sh 
# または 
python3 train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh 
# or 
python3 test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

---

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
