# docker build -t bevtrt85 -f docker/Dockerfile .
docker run -it --rm \
    --gpus all \
    -v ./:/workspace/BEVFormer_tensorrt/ \
    -v /data/nuscenes/can_bus:/workspace/BEVFormer_tensorrt/data/can_bus \
    -v /data2/coco:/workspace/BEVFormer_tensorrt/data/coco \
    -v /data/nuscenes:/workspace/BEVFormer_tensorrt/data/nuscenes \
    --shm-size=16G \
    --privileged \
    --network=host \
    --user root \
    hadonga/bev_trt:1.0 /bin/bash