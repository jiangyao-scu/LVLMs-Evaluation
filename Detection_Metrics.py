import sys
import re
import json
sys.path.append('Object-Detection-Metrics-master/')
import _init_paths
from utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

def box_label_general(gt_path, typeCoordinates, classID, dataset):
    myBoundingBoxes = BoundingBoxes()
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            temp_split = line.split('@#@')
            image_name = temp_split[0]
            
            bboxes = re.findall(r'\[\d+, \d+, \d+, \d+\]', temp_split[1])
            for b in bboxes:
                temp = re.findall(r'-?\d+', b)
                x0, y0, x1, y1 = int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])
                if x1>=x0 and y1>=y0:
                    if dataset=='VisA':
                        temp = image_name.split('.')[0]
                        temp = temp.split('VisA_20220922/')[1]
                        temp = temp.replace('\\', '/')
                        temp = temp.replace('Masks', 'Images')

                    elif dataset == 'MVTec':    
                        temp = image_name.split('_mask.')[0]
                        temp = temp.split('MVTec/')[1]
                        temp_sp = temp.split('\\')
                        temp = temp_sp[0] + '/test/' + temp_sp[3] + '/' + temp_sp[4]

                    elif dataset=='Trans10K':
                        temp = image_name.split('_mask.')[0]
                    elif dataset=='ISIC':
                        temp = image_name.split('_segmentation.')[0]
                    else:
                        temp = image_name.split('.')[0]
                    
                    temp_boundingBox = BoundingBox(imageName=temp, classId=classID, classConfidence=0.8, 
                                                x=x0, y=y0, w=x1, h=y1, typeCoordinates=typeCoordinates,
                                                bbType=typeCoordinates, format=BBFormat.XYX2Y2, imgSize=(448,448))
                    myBoundingBoxes.addBoundingBox(temp_boundingBox)
    
    return myBoundingBoxes

def get_predBBoxes_general(pred_path, typeCoordinates, classID, dataset):
    myBoundingBoxes = BoundingBoxes()
    with open(pred_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if pred_path.find('json') == -1:
            lines = lines[1:]
        
        for line in lines:
            if pred_path.find('json') != -1:
                record  = json.loads(line)
                img_name = record['img']
                if dataset=='VisA':
                    img_name = img_name.split('VisA/VisA/')[-1]
                elif dataset=='MVTec':
                    img_name = img_name.split('MVTec/')[-1]
                else:
                    img_name = os.path.basename(img_name)
                detection_results = record['answer']

                bboxes = re.findall(r'\[\d+\.?\d+, \d+\.?\d+, \d+\.?\d+, \d+\.?\d+\]', detection_results)
                # bboxes = re.findall(r'<\d+><\d+><\d+><\d+>', detection_results)

                for bbox in bboxes:
                    temp = re.findall(r'\d+\.?\d+', bbox)
                    # temp = re.findall(r'-?\d+', bbox)
                    x0, y0, x1, y1 = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
                    left = int(x0 * 448)
                    bottom = int(y0 * 448)
                    right = int(x1 * 448)
                    top = int(y1 * 448)

                    if x1>=x0 and y1>=y0:
                        temp_boundingBox = BoundingBox(imageName=img_name.split('.')[0], classId=classID, classConfidence=0.8, 
                                                    x=left, y=bottom, w=right, h=top, typeCoordinates=typeCoordinates,
                                                    bbType=typeCoordinates, format=BBFormat.XYX2Y2, imgSize=(448,448))
                        myBoundingBoxes.addBoundingBox(temp_boundingBox)
                    else:
                        print(bbox)
                
            else:
                img_name, detection_results = line.split('@#@')

                bboxes = re.findall(r'<\d+><\d+><\d+><\d+>', detection_results)

                # if img_name=='ILSVRC2012_test_00000196.jpg':
                #     stop = 1
                for bbox in bboxes:
                    temp = re.findall(r'-?\d+', bbox)
                    x0, y0, x1, y1 = int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])
                    left = x0 / 100 * 448
                    bottom = y0 / 100 * 448
                    right = x1 / 100 * 448
                    top = y1 / 100 * 448
                    if x1>=x0 and y1>=y0:
                        temp_boundingBox = BoundingBox(imageName=img_name.split('.')[0], classId=classID, classConfidence=0.8, 
                                                    x=left, y=bottom, w=right, h=top, typeCoordinates=typeCoordinates,
                                                    bbType=typeCoordinates, format=BBFormat.XYX2Y2, imgSize=(448,448))
                        myBoundingBoxes.addBoundingBox(temp_boundingBox)
                    else:
                        print(bbox)
        f.close()
    
    return myBoundingBoxes

# MVTec
def load_list_visa_img(image_path):
    # F:\miniGPT-Datasets\AD\MVTec
    final_list = []
    object_lists = os.listdir(image_path)
    for object in object_lists:
        if os.path.isdir(os.path.join(image_path, object)):
            anomaly_types = os.listdir(os.path.join(image_path, object, 'Data\\Images\\'))
            for anomaly in anomaly_types:
                names = os.listdir(os.path.join(image_path, object, 'Data\\Images\\', anomaly))
                for name in names:
                    final_list.append(os.path.join(image_path, object, 'Data\\Images\\', anomaly, name))
    
    return final_list

# visa
def load_list_mvtec_img(image_path):
    # F:\miniGPT-Datasets\AD\MVTec
    final_list = []
    object_lists = os.listdir(image_path)
    for object in object_lists:
        if os.path.isdir(os.path.join(image_path, object)):
            anomaly_types = os.listdir(os.path.join(image_path, object, 'test'))
            for anomaly in anomaly_types:
                names = os.listdir(os.path.join(image_path, object, 'test', anomaly))
                for name in names:
                    final_list.append(os.path.join(image_path, object, 'test', anomaly, name))
    
    return final_list


def evaluate(gt_path, pred_path, classID, visual=False, dataset=None, data_lib=None):
    bboxes = get_predBBoxes_general(pred_path, typeCoordinates=BBType.Detected, classID=classID, dataset=dataset)
    temp = box_label_general(gt_path, typeCoordinates=BBType.GroundTruth, classID=classID, dataset=dataset)
    
    bboxes.union(temp)

    image_path = data_lib[dataset]
    save_path = 'visual_box_llava/{}/'.format(dataset)
    
    if visual:
        if dataset=='VisA':
            img_names = load_list_visa_img(image_path)
            for img_name in img_names:
                middle_path = img_name.replace(os.path.basename(img_name), '')
                middle_path = middle_path.replace(image_path, '')
                save_image_f = save_path + middle_path
                if not os.path.exists(save_image_f):
                    os.makedirs(save_image_f)

                img_path = img_name
                img = cv2.imread(img_path)
                img = cv2.resize(img, (448, 448))

                # #####################################visa
                key = img_name.split('.')[0]
                key = key.split('VisA/')[1]
                key = key.replace('\\', '/')
                key = key.replace('Masks', 'Images')

                print(key)
                img = bboxes.drawAllBoundingBoxes(img, key)
                cv2.imwrite(os.path.join(save_image_f, os.path.basename(img_name)), img)

        elif dataset=='MVTec':
            img_names = load_list_mvtec_img(image_path)
            for img_name in img_names:
                middle_path = img_name.replace(os.path.basename(img_name), '')
                middle_path = middle_path.replace(image_path, '')
                save_image_f = save_path + middle_path
                if not os.path.exists(save_image_f):
                    os.makedirs(save_image_f)

                img_path = img_name
                img = cv2.imread(img_path)
                img = cv2.resize(img, (448, 448))
                
                key = img_path.split('.')[0]
                key = key.split('MVTec/')[1]
                key = key.replace('\\', '/')

                print(key)
                img = bboxes.drawAllBoundingBoxes(img, key)
                cv2.imwrite(os.path.join(save_image_f, os.path.basename(img_name)), img)
                
        else:
            img_names = os.listdir(image_path)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            names = os.listdir(image_path)

            for name in names:
                img_path = image_path + name
                img = cv2.imread(img_path)
                img = cv2.resize(img, (448, 448))
                img = bboxes.drawAllBoundingBoxes(img, name.split('.')[0])
                cv2.imwrite(os.path.join(save_path, name), img)

    evaluator = Evaluator()
    metricsPerClass = evaluator.GetPascalVOCMetrics(bboxes, IOUThreshold=0.5)
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics 
    for mc in metricsPerClass:
    # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('precision         recall           average_precision           F1')
        print(' {}                {}                {}                        {}'.format(precision[-1], recall[-1], average_precision, ((2 * precision[-1] * recall[-1])/(precision[-1] + recall[-1]))))
        # print('%s: %f' % (c, average_precision))
        a = 0

if __name__=='__main__':
    import cv2
    import numpy as np
    import os
    from lib.Evaluator import Evaluator

    ground_truth_path = 'bbox_label/VisA_Bbox.lst'
    pred_path = 'results/DET_results/MiniGPT-v2/VisA.lst'

    data_lib = {'ISIC': 'XXX/ISIC2017/',
                'VisA': 'XXX/VisA/',
                'MVTec':'XXX/MVTec/',
                'Trans10K': 'XXX/Trans10K/',
                'SOC':'XXX/SOC/',
                'DUTS': 'XXX/DUTS/',
                'ColonDB': 'XXX/ColonDB/',
                'COD10K':'XXX/COD10K/',
                'ETIS':'XXX/ETIS/'
                }

    evaluate(ground_truth_path, pred_path, 'AD', dataset='VisA', visual=False, data_lib = data_lib)

