FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 非対話モード＆タイムゾーン設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    apt-get install -y tzdata \
    wget \
    git \
    curl \
    unzip \
    nano \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Minicondaインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# GPUライブラリのパス追加
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
RUN ln -sf /usr/local/cuda-11.8/lib64/libcusolver.so.11 /usr/local/cuda-11.8/lib64/libcusolver.so.10

# SwinU-net用のPython 3.7環境作成
RUN conda create -n swinunet python=3.7 -y

# 作業ディレクトリを設定
WORKDIR /Swin-Unet

# バージョン指定のあるPyTorchをインストール
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# 基本的なパッケージをグループごとにインストール - バージョン指定に注意
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install 'numpy<1.23.0' 'scipy<1.8.0' h5py tqdm"

# 可視化・ロギング関連
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install 'tensorboard<2.11.0' tensorboardX"

# 医療画像処理関連 - バイナリインストール優先
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install --only-binary SimpleITK SimpleITK==2.0.2 && \
    pip install medpy"

# モデル関連
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install 'timm<0.6.0' 'einops<0.5.0' ml-collections"

# キャッシュ削除で軽量化
RUN conda clean -afy

# 起動時に自動でswinunet環境を有効化する設定
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate swinunet" >> ~/.bashrc

# デフォルトのシェルを設定
SHELL ["/bin/bash", "--login", "-c"]

# コンテナ起動時のコマンド
CMD ["/bin/bash", "--login"]