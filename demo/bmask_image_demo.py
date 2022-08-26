# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import os, numpy as np, cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import pycocotools.mask as pm
from pycocotools.coco import COCO
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '---img_dir',
        default='/research/d3/bqyang/object_localization_network/data/project/20220819/test',
        help='Image file'
    )
    parser.add_argument(
        '--seq',
        default=None,
        help='Image file'
    )
    parser.add_argument('--config', default='/research/d3/bqyang/object_localization_network/work_dirs/bmask_project2/rcnn_mask.py', help='Config file')
    parser.add_argument('--checkpoint', default='/research/d3/bqyang/object_localization_network/work_dirs/bmask_project2/latest.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.95, help='bbox score threshold')
    parser.add_argument('--out_dir', default='/research/d3/bqyang/object_localization_network/project', help='output dir')
    args = parser.parse_args()
    return args

def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)

    test_name = args.img_dir.split('/')[-1]
    os.makedirs(os.path.join(args.out_dir), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, test_name), exist_ok=True)

    for seq in os.listdir(args.img_dir):
        if args.seq is not None:
            if seq != args.seq: continue

        os.makedirs(os.path.join(args.out_dir, test_name, seq), exist_ok=True)

        for k, file in enumerate(os.listdir(os.path.join(args.img_dir, seq, 'crop'))):
            if '.png' not in file: continue
            rgb = os.path.join(args.img_dir, seq, 'crop', file)
            print(rgb)
            result = inference_detector(model, rgb)
            
            bbox, mask = result[0][0], result[1][0]
            #bbox, mask = np.asarray(bbox), np.asarray(mask)
            # bbox, mask = bbox[bbox[:, -1] > args.score_thr], mask[bbox[:, -1] >= args.score_thr]
            
            show_result_pyplot(model, rgb, result, score_thr=args.score_thr, out_file=os.path.join(args.out_dir, test_name, seq, file))

if __name__ == '__main__':
    args = parse_args()
    main(args)
