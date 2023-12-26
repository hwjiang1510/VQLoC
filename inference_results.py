import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse
import json
import tqdm
from queue import Empty as QueueEmpty

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch import multiprocessing as mp

from config.config import config, update_config
from utils import exp_utils
from evaluation import eval_utils
from evaluation.task_inference_results import Task
# from model.corr_clip_spatial_transformer2_anchor_2heads import ClipMatcher
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher


class WorkerWithDevice(mp.Process):
    def __init__(self, config, task_queue, results_queue, worker_id, device_id):
        self.config = config
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))

    def work(self, task_queue, results_queue):

        device = torch.device(f"cuda:{self.device_id}")

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except QueueEmpty:
                break
            key_name = task.run(self.config, device)
            results_queue.put(key_name)
            del task


def get_results(annotations, config):
    num_gpus = torch.cuda.device_count()
    mp.set_start_method("forkserver")

    task_queue = mp.Queue()
    for _, annots in annotations.items():
        task = Task(config, annots)
        task_queue.put(task)
    # Results will be stored in this queue
    results_queue = mp.Queue()

    num_processes = 30 #num_gpus

    pbar = tqdm.tqdm(
        desc=f"Get RT results",
        position=0,
        total=len(annotations),
    )

    workers = [
        WorkerWithDevice(config, task_queue, results_queue, i, i % num_gpus)
        for i in range(num_processes)
    ]
    # Start workers
    for worker in workers:
        worker.start()
    # Update progress bar
    predicted_rts = {}
    n_completed = 0
    while n_completed < len(annotations):
        pred = results_queue.get()
        predicted_rts.update(pred)
        n_completed += 1
        pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    pbar.close()
    return predicted_rts


def format_predictions(annotations, predicted_rts):
    # print(predicted_rts)
    # Format predictions
    predictions = {}

    for clip_uid in annotations.keys():
        # print("clip_uid: ", clip_uid)
        if clip_uid not in predictions:
            predictions[clip_uid] = {"predictions":[]}
        for query_set in annotations[clip_uid]['annotations']:
            auid = query_set["annotation_uid"]
            apred = {
                "query_sets": {},
                # "annotation_uid": auid,
            }
            for qid in query_set["query_sets"].keys():
                if (auid, qid) in predicted_rts:
                    # print(predicted_rts[(auid, qid)])
                    rt_pred = predicted_rts[(auid, qid)][0].to_json()
                    apred["query_sets"][qid] = rt_pred
                else:
                    apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}
            predictions[clip_uid]["predictions"].append(apred)
        # video_predictions["clips"].append(clip_predictions)
    # predictions["results"]["videos"].append(video_predictions)
    # print(predictions)
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true",help="evaluate model")
    parser.add_argument(
        "--debug", dest="debug", action="store_true",help="evaluate model")
    parser.add_argument(
        "--gt-fg", dest="gt_fg", action="store_true",help="evaluate model")
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    mode = 'eval' if args.eval else 'val'
    config.inference_cache_path = os.path.join(output_dir, f'inference_cache_{mode}')
    os.makedirs(config.inference_cache_path, exist_ok=True)
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    mode = 'test_unannotated' if args.eval else 'val'
    annotation_path = os.path.join(config.data_dir, 'vq_{}.json'.format(mode))
    with open(annotation_path) as fp:
        annotations = json.load(fp)
    clipwise_annotations_list = eval_utils.convert_annotations_to_clipwise_list(annotations)

    if args.debug:
        clips_list = list(clipwise_annotations_list.keys())
        clips_list = sorted([c for c in clips_list if c is not None])
        clips_list = clips_list[: 20]
        clipwise_annotations_list = {
            k: clipwise_annotations_list[k] for k in clips_list
        }

    predictions_rt = get_results(clipwise_annotations_list, config)
    predictions = format_predictions(annotations, predictions_rt)
    if not args.debug:
        with open(config.inference_cache_path + '_results_vit_fpn.json', 'w') as fp:
            json.dump(predictions, fp, indent=4)