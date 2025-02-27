import os
from PIL import Image
import cv2
import math



def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x3, y3, x4, y4 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    if bbox1_area>bbox2_area:
        iou = intersection_area / bbox2_area
    else:
        iou = intersection_area / bbox2_area
    return iou

def box_union(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x3, y3, x4, y4 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    intersection_x1 = min(x1, x3)
    intersection_y1 = min(y1, y3)
    intersection_x2 = max(x2, x4)
    intersection_y2 = max(y2, y4)
    dis = math.sqrt(intersection_x1**2 + intersection_y1**2)

    return [intersection_x1, intersection_y1, intersection_x2-intersection_x1, intersection_y2-intersection_y1, dis]

def remove(bbox):
    new_bbox = [[x[0], x[1], x[2], x[3], math.sqrt(x[0]**2 + x[1]**2)] for x in bbox]
    new_bbox.sort(key=lambda x: x[0])

    temp = new_bbox
    final_bbox = []

    for i in range(len(new_bbox)-1):
        for j in range(len(new_bbox)):
            iou = computeIoU(new_bbox[i], new_bbox[j])
            if iou>0:
                union = box_union(new_bbox[i], new_bbox[j])
                new_bbox[i] = union
                new_bbox[j] = union
    
    for x in new_bbox:
        if x not in final_bbox:
            final_bbox.append(x)
    return final_bbox


def mask2bbox(mask):
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # bboxs = mask_find_bboxs(mask)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    bboxs = stats[:-1] # 排除最外层的连通图

    new_bboxs = remove(bboxs)

    final_bboxs = []
    for b in new_bboxs:
        if b[2]<=1 or b[3]<=1:
            continue
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        # print(x0, y0, x1, y1)
        x0 = int(x0 / 448 * 100)
        y0 = int(y0 / 448 * 100)
        x1 = int(x1 / 448 * 100)
        y1 = int(y1 / 448 * 100)
        if x0 > 0:
            x0 = x0 - 1
        if y0 > 0:
            y0 = y0 - 1
        if x1 < 100:
            x1 = x1 + 1
        if y1 < 100:
            y1 = y1 + 1
            
        final_bboxs.append([x0, y0, x1, y1])
    
    return final_bboxs

def draw_image_rectangle(img, bboxs):
    for b in bboxs:
        if b[0]<0 or b[1]<0 or b[2]>100 or b[3]>100:
        # if x0<=0 or y1<=0 or x1>=99 or x1>=99:
            print('err')
        img = cv2.rectangle(img, (int(b[0]/100*448), int(b[1]/100*448)), (int(b[2]/100*448), int(b[3]/100*448)), (0, 0, 255), 1)
    
    return img

if __name__=='__main__':
    mask_path = 'F:/miniGPT-Datasets/SOD/DUTS/DUTS/Test/GT/'
    result_path = 'DUTS_Bbox.lst'

    img_names = os.listdir(mask_path)

    for img_name in img_names:

        mask = cv2.imread(os.path.join(mask_path, img_name), cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (448, 448))

        bboxs = mask2bbox(mask)

        with open(result_path, 'a') as f:
            line = img_name
            for b in bboxs:
                line = line + '[{}, {}, {}, {},];'.format(b[0], b[1], b[2], b[3])
            
            f.writelines(line + '\n')

        # mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
        # mask_with_bbox = draw_image_rectangle(mask_BGR, bboxs=bboxs)

        # cv2.imshow('show_image', mask_with_bbox)
        # key = cv2.waitKey(0)
        