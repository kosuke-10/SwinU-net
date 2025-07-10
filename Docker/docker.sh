# ----------------
# docker.sh
# dockerimageの作成
# dockerコンテナの作成
# docker.sh "GPU番号"
# ----------------

# -----------------------------
# 設定（必要に応じて書き換えてOK）
# -----------------------------

# GPU番号が指定されていなければエラーで終了
if [ -z "$1" ]; then
  echo "エラー: GPU番号を指定してください"
  echo "使い方: bash docker.sh <GPU番号>"
  exit 1
fi

GPU_ID=$1
IMAGE_NAME=yoshida/swin-unet
CONTAINER_NAME=yoshida_swinu-net${GPU_ID}

HOST_DIR=/home/yoshida/Swin-Unet
CONTAINER_DIR=/Swin-Unet

# シンボリックリンクの代わりに一時ファイルを作成
cp ../requirements.txt ./temp_requirements.txt

# Dockerビルド
docker build --force-rm -t ${IMAGE_NAME} .

# 一時ファイルを削除
rm -f ./temp_requirements.txt

# コンテナ実行
docker run --gpus "device=${GPU_ID}" \
   --shm-size=24g \
   -it -v ${HOST_DIR}:${CONTAINER_DIR} \
   --name ${CONTAINER_NAME} ${IMAGE_NAME}