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
a
# 作業ディレクトリを設定
WORKDIR /Swin-Unet

# 一時ファイルをコピー（docker.shで作成）
COPY ./temp_requirements.txt /tmp/requirements.txt
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install --no-cache-dir -r /tmp/requirements.txt"

# PyTorch 1.10.0をCUDA 11.3でインストール（SwinU-netの推奨バージョン）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html"

# その他必要なライブラリをインストール
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate swinunet && \
    pip install SimpleITK medpy"

# キャッシュ削除で軽量化
RUN conda clean -afy

# 起動時に自動でswinunet環境を有効化する設定
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate swinunet" >> ~/.bashrc

# デフォルトのシェルを設定
SHELL ["/bin/bash", "--login", "-c"]

# コンテナ起動時のコマンド
CMD ["/bin/bash", "--login"]