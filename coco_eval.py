#!/usr/bin/env python

import argparse
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.processing import preprocess, postprocess_coco
from utils.general import check_dataset, coco80_to_coco91_class

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]

jdict = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default=ROOT / 'data/coco.yaml',
                        help='dataset.yaml path')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='yolov7',
                        help='Inference model name, default yolov7')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input width, default 640')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input height, default 640')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')

    FLAGS = parser.parse_args()

    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    print("\n" + "="*20 + " TRITON SERVER " + "="*20)
    data = check_dataset(FLAGS.data)
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'val2017.txt')  # COCO dataset
    if not is_coco:
        raise Exception('Only Coco Dataset Eval is supported. ❌')

    val_data = open(data['val'], "r")
    val_files = val_data.readlines()
    val_data.close()
    class_map = coco80_to_coco91_class()
    
    print("Evaluating Model: " , FLAGS.model)
    for image_path in tqdm(val_files, desc="Inferencing images"):  # 2. Create an instance of tqdm
        image_path = image_path.rstrip("\n")
        image_path = image_path.lstrip("./")
        image_path =os.path.join(data['path'], image_path)

        if not os.path.exists(image_path):
            print(f"FAILED: missing {str(image_path)}")
            sys.exit(1)
        
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, FLAGS.width, FLAGS.height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))

        input_image = cv2.imread(str(image_path))
        if input_image is None:
            print(f"FAILED: could not load input image {str(image_path)}")
            sys.exit(1)

        
        input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

        inputs[0].set_data_from_numpy(input_image_buffer)

        results = triton_client.infer(model_name=FLAGS.model,
                                        inputs=inputs,
                                        outputs=outputs,
                                        client_timeout=FLAGS.client_timeout)

        for output in OUTPUT_NAMES:
            result = results.as_numpy(output)

        num_dets = results.as_numpy(OUTPUT_NAMES[0])
        det_boxes = results.as_numpy(OUTPUT_NAMES[1])
        det_scores = results.as_numpy(OUTPUT_NAMES[2])
        det_classes = results.as_numpy(OUTPUT_NAMES[3])
        detected_objects = postprocess_coco(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [FLAGS.width, FLAGS.height])

        image_id = int(Path(image_path).stem)

        for box in detected_objects:
            
            x1, y1, x2, y2 = box.box()
            x, y = float(x1), float(y1)
            w, h = float(x2 - x1), float(y2 - y1)
            bbox = [x, y, w, h]
            category_id=class_map[int(box.classID)] 
            score = float(box.confidence)

            jdict.append({
                "image_id": image_id,
                "category_id": category_id ,
                "bbox": [round(x, 2) for x in bbox],
                "score": score
            })

    if len(jdict):
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(f"./_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        eval.params.imgIds = [int(Path(x).stem) for x in val_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        print("\n" + "="*25 + " EVALUATION SUMMARY - "+ FLAGS.model.upper() +" "+"="*24)
        eval.summarize()
        map, map50, map75 = eval.stats[:3]  # update results (mAP@0.5:0.95, mAP@0.5)
        print("="*80)
        print(f"mAP@0.5:0.95: {round(map,3)} \nmAP@0.5:      {round(map50,3)} \nmAP@0.75:     {round(map75,3)} ")
        print("="*80)
