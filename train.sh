CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 \
train_anchor.py --cfg ./config/train.yaml