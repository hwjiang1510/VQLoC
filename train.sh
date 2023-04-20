CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=4 \
train.py --cfg ./config/vq2d_all_transformer.yaml