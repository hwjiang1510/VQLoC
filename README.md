# Single-Stage Visual Query Localization in Egocentric Videos (NeurIPS 2023)

### [Project Page](https://hwjiang1510.github.io/VQLoC/) |  [Paper](https://arxiv.org/abs/2306.09324)
<br/>

> Single-Stage Visual Query Localization in Egocentric Videos

> [Hanwen Jiang](https://hwjiang1510.github.io/), [Santhosh Ramakrishnan](https://srama2512.github.io/), [Kristen Grauman](https://www.cs.utexas.edu/~grauman/)


## Installation
```
conda create --name vqloc python=3.8
conda activate vqloc

# Install pytorch or use your own torch version
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install -r requirements.txt 
```

## Pre-trained Weights
We provide the model weights trained on [here](https://utexas.box.com/shared/static/3j3q9qsc1kovpwfxtnsful7pvdy234q6.tar).


## Train LEAP

### Download Dataset
- Please follow [vq2d baseline](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D#preparing-data-for-training-and-inference) step 1/2/4/5 to process the dataset into video clips.

### Training
- Use `./train.sh` and change your training config accordingly.
- The default training configurations require about 200GB at most, e.g. 8 A40 GPUs with 40GB VRAM, each.


## Evaluate LEAP
- 1. Use `./inference_predict.sh` to inference on the target video clips. Change the path of your model checkpoint.
- 2. Use `python inference_results.py --cfg ./config/val.yaml` to format the results. Use `--eval` and `--cfg ./config/eval.yaml` for evaluation (submit to leaderboard).
- 3. Use `python evaluate.py` to get the numbers. Please change `--pred-file` and `--gt-file` accordingly.

## Known Issues
- The hard negative mining is not very steady. We set `use_hnm=False` by default.


## Citation
```bibtex
@article{jiang2023vqloc,
   title={Single-Stage Visual Query Localization in Egocentric Videos},
   author={Jiang, Hanwen and Ramakrishnan, Santhosh and Grauman, Kristen},
   journal={ArXiv},
   year={2023},
   volume={2306.09324}
}
```