# client 端 api 调用案例---------------------------------
import os
import re
import base64
from io import BytesIO
from typing import Union

import torch
import requests
from PIL import Image
from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes


########################################
# helper
########################################

def pil_to_base64(pil_img):
    output_buffer = BytesIO()
    pil_img.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()
    encode_img = base64.b64encode(byte_data)
    return str(encode_img, encoding='utf-8')


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def draw_bounding_boxes(
        image,
        boxes,
        **kwargs,
):
    if isinstance(image, Image.Image):
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    return _draw_bounding_boxes(image, boxes, **kwargs)


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


########################################
#
########################################

def query(image: Union[Image.Image, str], text: str, boxes_value: list, boxes_seq: list, server_url='http://127.0.0.1:12345/shikra'):
    if isinstance(image, str):
        image = Image.open(image)
    pload = {
        "img_base64": pil_to_base64(image),
        "text": text,
        "boxes_value": boxes_value,
        "boxes_seq": boxes_seq,
    }
    resp = requests.post(server_url, json=pload)
    if resp.status_code != 200:
        raise ValueError(resp.reason)
    ret = resp.json()
    return ret


def postprocess(text, image):
    if image is None:
        return text, None

    image = expand2square(image)

    colors = ['#ed7d31', '#5b9bd5', '#70ad47', '#7030a0', '#c00000', '#ffff00', "olive", "brown", "cyan"]
    pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

    def extract_boxes(string):
        ret = []
        for bboxes_str in pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    extract_pred = extract_boxes(text)
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(extract_pred):
        color = colors[idx % len(colors)]
        for box in boxes:
            boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
            color_to_draw.append(color)
    if not boxes_to_draw:
        return text, None
    res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=8)
    res = ToPILImage()(res)

    # post process text color
    location_text = text
    edit_text = list(text)
    bboxes_str = pat.findall(text)
    for idx in range(len(bboxes_str) - 1, -1, -1):
        color = colors[idx % len(colors)]
        boxes = bboxes_str[idx]
        span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
        location_text = location_text[:span[0]]
        edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
    text = "".join(edit_text)
    return text, res


if __name__ == '__main__':
    server_url = 'http://127.0.0.1:12345' + "/shikra"


    def example(folder_path,img_path,prompt,answers):
        image_path = img_path
        text = prompt
        boxes_value = []
        boxes_seq = []
        print(img_path)
        answers.append(img_path+'@#@')
        response = query(folder_path+image_path, text, boxes_value, boxes_seq, server_url)
        _, image = postprocess(response['response'], image=Image.open(folder_path+image_path))
        print(_.replace('</s>',''))
        answers.append(_.replace('</s>','\n'))


    def get_dataset(prefix):
        image_list = os.listdir(prefix)
        return image_list


    def remove_cumputed(image_list, file_name,img_path):
        with open(file_name, 'r') as f:
            lines = f.readlines()
            print('Computed image number: {}'.format(len(lines)))
            for line in lines:
                temp = line.split('@#@')[0]
                if img_path + temp in image_list:
                    image_list.remove(img_path + temp)
        f.close()
        return image_list


    folder_path = 'xxx/Trans10K/test/Imgs/'
    results_path = './Trans10K.lst'
    image_list = get_dataset(folder_path)
    prompt = 'For this image <image>, I want a simple and direct answer to my question: Are there any transparent objects in this picture? Please answer Yes or No.'
    if os.path.exists(results_path):
        image_list = remove_cumputed( image_list, results_path,folder_path)
    # image_list = ['AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/COD10K/Imgs/COD10K-CAM-1-Aquatic-10-LeafySeaDragon-419.jpg']
    print('Total image number: {}'.format(len(image_list)))
    print('Examples: {}'.format(image_list[0]))
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            f.writelines(prompt+'\n')
        f.close()
    answers = []
    for img_path in image_list:
        example(folder_path,img_path,prompt,answers)
    with open(results_path, 'a') as f:
        f.writelines(answers)
    f.close()

