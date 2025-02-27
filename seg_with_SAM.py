import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import argparse
import boto3
import io
from PIL import Image
import re
import os
import json
sys.path.append("..")
sys.path.append("~")
from segment_anything import sam_model_registry, SamPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--weight_path", type=str, default='./weights/sam_vit_h_4b8939.pth')
    parser.add_argument("--model_type", type=str, default='vit_h')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--prefix", type=str, default='xxx/Datasets/SOC/Imgs/')
    parser.add_argument("--results_path", type=str, default='./results/SEG/LLaVA-1.5/SOC/')
    parser.add_argument("--detection_results_path", type=str, default='results/DET/LLaVA-1.5/SOC.jsonl')
    args = parser.parse_args()
    return args

def get_dataset_using_boto3(bucket, prefix):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    image_list = []
    for page in pages:
        temp = [x['Key'] for x in page['Contents'] if not x['Key'].endswith('/')]
        image_list = image_list + temp
 
    return(image_list)
    
def show_box(image, box):
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 0, 255), thickness=2)
    return image

def load_box_llava(file_path, category=None):
    err = []
    err_no_box = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        boxes_tree = {}
        if category is None:
            a = 0
        else:
            for line in lines[1:]:
                temp = line.split('@#@')
                image_path, detection_results = temp[0], temp[1]
                
                # image_name = image_path.split('/')[-1]
                
                ###mvtec
                image_name = image_path.split('.')[0]
                
                boxes_tree[image_name] = {}
                boxes_tree[image_name][category] = []
                detection_results = detection_results.replace(' ', '')
                pattern = r'<\d+><\d+><\d+><\d+>'
                matches = re.findall(pattern, detection_results)
                
                if len(matches)>0:
                    for bbox_string in matches:
                        integers = re.findall(r'\d+', bbox_string)
                        if len(integers) == 4:
                            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                            left = int(x0 / 100 * 448)
                            bottom = int(y0 / 100 * 448)
                            right = int(x1 / 100 * 448)
                            top = int(y1 / 100 * 448)
                            boxes_tree[image_name][category].append([left, bottom, right, top])
                        else:
                            err.append(line)
                else:
                    err_no_box.append(line)
          
        print('err not enough number:{}'.format(err))
        print('err no box number:{}'.format(err_no_box))
        return boxes_tree
    
def load_box_shikra(file_path, category=None):
    err = []
    err_no_box = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        boxes_tree = {}
        if category is None:
            a = 0
        else:
            for line in lines[1:]:
                temp = line.split('@#@')
                image_path, detection_results = temp[0], temp[1]
                
                # image_name = image_path.split('/')[-1]
                
                ###mvtec
                image_name = image_path.split('.')[0]
                
                boxes_tree[image_name] = {}
                boxes_tree[image_name][category] = []
                # detection_results = detection_results.replace(' ', '')
                pattern = r'\(\d+, \d+, \d+, \d+\)'
                matches = re.findall(pattern, detection_results)
                
                if len(matches)>0:
                    for bbox_string in matches:
                        integers = re.findall(r'\d+', bbox_string)
                        if len(integers) == 4:
                            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                            if x1>x0 and y1>y0:
                                boxes_tree[image_name][category].append([x0, y0, x1, y1])
                            else:
                                print(bbox_string)
                        else:
                            err.append(line)
                else:
                    err_no_box.append(line)
          
        print('err not enough number:{}'.format(err))
        print('err no box number:{}'.format(err_no_box))
        return boxes_tree
    
def load_box_minigptv_json(file_path, category=None):
    err = []
    err_no_box = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        boxes_tree = {}
        if category is None:
            a = 0
        else:
            for line in lines:
                record  = json.loads(line)
                image_path = record["img"]
                detection_results= record["answer"]
                
                ###mvtec
                image_name = image_path.split('.')[0]
                
                boxes_tree[image_name] = {}
                boxes_tree[image_name][category] = []
                # detection_results = detection_results.replace(' ', '')
                pattern = r'<\d+><\d+><\d+><\d+>'
                matches = re.findall(pattern, detection_results)
                
                if len(matches)>0:
                    for bbox_string in matches:
                        integers = re.findall(r'\d+', bbox_string)
                        if len(integers) == 4:
                            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                            if x1>x0 and y1>y0:
                                left = int(x0 / 100 * 448)
                                bottom = int(y0 / 100 * 448)
                                right = int(x1 / 100 * 448)
                                top = int(y1 / 100 * 448)
                                boxes_tree[image_name][category].append([left, bottom, right, top])
                            else:
                                print(bbox_string)
                        else:
                            err.append(line)
                else:
                    err_no_box.append(line)
          
        print('err not enough number:{}'.format(err))
        print('err no box number:{}'.format(err_no_box))
        return boxes_tree                        
                        
def evaluate(model, image_list, boxex_tree, save_path, dataset):
    
    if dataset=='MVTec' or dataset=='VisA':
        save_path = save_path
    else:
        ###########################################make floder#######################################
        # s3://mbz-hpc-aws-researchers/AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/Datasets/AD/MVTec/
        # s3://mbz-hpc-aws-researchers/AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/Datasets/AD/VisA/
        save_path_f = save_path
        if not os.path.exists(save_path_f):
            os.makedirs(save_path_f, exist_ok=True)

        box_path_f = save_path_f.replace('results', 'test')
        if not os.path.exists(box_path_f):
            os.makedirs(box_path_f, exist_ok=True)

    with torch.no_grad():
        for idx, img_path in enumerate(image_list):
            
            temp = img_path.replace(prefix, '')
            temp_name = temp.split('/')[-1]
            
            if dataset=='MVTec' or dataset=='VisA':
                ################################################make floder#################################
                # AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/Datasets/AD/MVTec/
                save_path_f = save_path + temp.replace(temp_name, '')
                # print(save_path)
                # save_path = args.results_path
                if not os.path.exists(save_path_f):
                    os.makedirs(save_path_f, exist_ok=True)
                    
                box_path_f = save_path_f.replace('results', 'test')
                if not os.path.exists(box_path_f):
                    os.makedirs(box_path_f, exist_ok=True)
             ###################################################load image################################
            
            key = temp.split('.')[0]
            print(idx, key)
            byte_io = io.BytesIO()
            s3_client.download_fileobj(bucket, img_path, byte_io)
            byte_io.seek(0)
            byteImg = Image.open(byte_io)
            byteImg = byteImg.convert('RGB')
            image = byteImg
            image = np.array(image)
            
            ######################################encode image###################################################
            predictor.set_image(image)
            # boxes = [[1, 23, 89, 86]]
            (h, w, c) = image.shape
            boxes = boxes_tree[key]['Anomaly']
            final_mask = np.zeros((h, w))
            final_mask = final_mask.astype(dtype=bool)
            box_image = image.copy()
            for i, box in enumerate(boxes):
                input_box = np.array(box)
                box_image = show_box(box_image, box)
                
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                    )
                
                final_mask = np.logical_or(final_mask, masks[0, :, :])

                
            box_image = cv2.cvtColor(box_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(box_path_f + temp_name , box_image)
            final_mask = final_mask.astype(np.uint8)
            cv2.imwrite(save_path_f + temp_name , final_mask*255)
    
    
if __name__=='__main__':
    print('Initializing SAM')
    args = parse_args()
    
    ###################################################get image list###################################################
    s3_client = boto3.client("s3", region_name="me-central-1")
    bucket = 'XXX'
    prefix = args.prefix
    image_list = get_dataset_using_boto3(bucket, prefix)
    print('Loading images：{}'.format(len(image_list)))
    
    
    ###################################################read detection results###################################################
    detection_results_path = args.detection_results_path
    boxes_tree = load_box_minigptv_json(file_path=detection_results_path, category='SOD')
    
    # print(boxes_tree)

    ###################################################Initializing SAM model#############################################
    sam_checkpoint = args.weight_path
    model_type = args.model_type
    device = args.device
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    evaluate(predictor, image_list, boxes_tree, args.results_path, dataset='SOC')