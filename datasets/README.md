# データ準備

1. Synapse 多臓器データセットへのアクセス方法：
   1. [公式 Synapse サイト](https://www.synapse.org/#!Synapse:syn3193805/wiki/) に登録し、データセットをダウンロードしてください。次の処理を行います：データを NumPy 形式に変換し、画像の値を [-125, 275] の範囲にクリップし、各3D画像を [0, 1] の範囲に正規化します。学習用には3Dボリュームから2Dスライスを抽出し、テスト用には3Dボリュームを h5 形式で保持します。
   2. また、再現実験のための前処理済みデータを希望する場合は、jienengchen01 [アット] gmail.com にメールを送って直接リクエストすることも可能です。
2. プロジェクト全体のディレクトリ構成は以下のとおりです：

```bash
.
├── TransUNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──Synapse
        ├── test_vol_h5
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train_npz
            ├── case0005_slice000.npz
            └── *.npz
```
