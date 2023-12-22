import warnings
import numpy as np
from utils.loss_utils import GiouLoss


def convert_annotations_to_clipwise_list(annotations):
    # for k,v in annotations.items():
    #     print(k,v)

    clipwise_annotations_list = {}
    # for v in annotations["videos"]:
    #     vuid = v["video_uid"]
        # for c in v["clips"]:
        # cuid = c["clip_uid"]
        # for a in c["annotations"]:
    for cuid, a in annotations.items():
        vuid = cuid
        aid = a["annotations"]
        for queries in aid:
            for qid, q in queries["query_sets"].items():
                if not q["is_valid"]:
                    continue
                curr_q = {
                    "metadata": {
                        "video_uid": vuid,
                        # "video_start_sec": c["video_start_sec"],
                        # "video_end_sec": c["video_end_sec"],
                        # "clip_fps": c["clip_fps"],
                        "query_set": qid,
                        "annotation_uid": queries['annotation_uid'],
                    },
                    "clip_uid": cuid,
                    "query_frame": q["query_frame"],
                    "visual_crop": q["visual_crop"],
                }
                if "response_track" in q:
                    curr_q["response_track"] = q["response_track"]
                if cuid not in clipwise_annotations_list:
                    clipwise_annotations_list[cuid] = []
                clipwise_annotations_list[cuid].append(curr_q)
    # print(clipwise_annotations_list["0a4eb349-689b-47fe-aaa2-a072976a7472"])
    # for _ in clipwise_annotations_list["385ac605-16e2-4ffa-9c43-1b2b859f0626"]:
    #     print(_)
    return clipwise_annotations_list


def format_predictions(annotations, predicted_rts):
    # Format predictions
    predictions = {
        "version": annotations["version"],
        "challenge": "ego4d_vq2d_challenge",
        "results": {"videos": []},
    }
    for v in annotations["videos"]:
        video_predictions = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                auid = a["annotation_uid"]
                apred = {
                    "query_sets": {},
                    "annotation_uid": auid,
                }
                for qid in a["query_sets"].keys():
                    if (auid, qid) in predicted_rts:
                        rt_pred = predicted_rts[(auid, qid)][0].to_json()
                        apred["query_sets"][qid] = rt_pred
                    else:
                        apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}
                clip_predictions["predictions"].append(apred)
            video_predictions["clips"].append(clip_predictions)
        predictions["results"]["videos"].append(video_predictions)
    return predictions
