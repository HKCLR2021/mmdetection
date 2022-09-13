# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default = "/home/xwchi/mmdetection/img_000.jpg", help='Image file')
    parser.add_argument('--config', default="/home/xwchi/mmdetection/configs/boundary_bq/configuration.py", help='Config file')
    parser.add_argument('--checkpoint', default = "/home/xwchi/exps_bound/latest.pth", help='Checkpoint file')
    parser.add_argument('--out-file', default="/home/xwchi/mmdetection/output.jpg", help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0., help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    x_min, x_max, y_min, y_max = 545, 955, 75, 345
    img = cv2.imread(args.img)
    img = img[y_min:y_max,x_min:x_max,:]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, img)
    # show the results
    show_result_pyplot(
        model,
        # args.img,
        img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
