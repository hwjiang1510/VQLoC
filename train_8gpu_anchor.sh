CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=8 \
train_anchor.py --cfg ./config/vq2d_all_transformer2_anchor_dinov2_hnm.yaml