#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="",
        help="paddle/trt_fp32/trt_fp16"
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="image path"
    )
    parser.add_argument(
        "--conf",
        default=None,
        type=float,
        help="test conf")
    parser.add_argument(
        "--nms",
        default=None,
        type=float,
        help="test nms threshold")
    parser.add_argument(
        "--tsize",
        default=None,
        type=int,
        help="test img size")
    parser.add_argument(
        "--num_classes",
        default=80,
        type=int,
        help="num classes")
    return parser.parse_args()


def init_predictor(args):
    config = Config(
        os.path.join(args.model_dir, 'model.pdmodel'),
        os.path.join(args.model_dir, 'model.pdiparams'))
    config.enable_use_gpu(200, 0)
    config.switch_ir_optim(True)
    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if args.run in precision_map:
        config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=3,
                precision_mode=precision_map[args.run_mode],
                use_static=False,
                use_calib_mode=False)
    
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor

def run(predictor, args):
    img = cv2.imread(args.path)
    img = ValTransform(legacy=legacy)(img, None, [args.tsize, args.tsize])
    img = img[None, :, :, :]
    img = img.astype(paddle.float32)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.copy_from_cpu(img)

    predictor.run()
    output_names = predictor.get_output_names()
    out_tensor = predictor.get_output_handle(output_names[0])
    out_boxes = out_tensor.copy_to_cpu()

    outputs = postprocess(
        out_boxes, args.num_classes, args.conf,
        args.nms, class_agnostic=True
    )

    for output in outputs:
        print(output)


if __name__ == '__main__':
    args = parse_args()
    predictor = init_predictor(args)
    run_fake(predictor, args)
