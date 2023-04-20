CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 \
train.py --cfg ./config/vq2d.yaml