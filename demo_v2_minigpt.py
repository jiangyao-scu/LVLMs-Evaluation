import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat


from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt4.common.eval_utils import prepare_texts
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import boto3
import os
import io
from PIL import Image


''' ###################################################Use case###################################################
python demo_v2_my.py --prefix 'AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/Datasets/Polyp/ETIS-LaribPolypDB/images/' --results_path './results/ETIS.lst' --prompts '<s>[INST] <Img><ImageHere></Img> [detection] Polyps. [/INST]' '<s>[INST] <Img><ImageHere></Img> Are there any polyps in this picture? Please answer Yes or No. [/INST]' --seed 42; python demo_v2_my.py --prefix 'AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/Datasets/Polyp/ETIS-LaribPolypDB/images/' --results_path './results/ETIS_SEED2.lst' --prompts '<s>[INST] <Img><ImageHere></Img> [detection] Polyps. [/INST]' '<s>[INST] <Img><ImageHere></Img> Are there any polyps in this picture? Please answer Yes or No. [/INST]' --seed 2; python demo_v2_my.py --prefix 'AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/Datasets/Polyp/ETIS-LaribPolypDB/images/' --results_path './results/ETIS_SEED68.lst' --prompts '<s>[INST] <Img><ImageHere></Img> [detection] Polyps. [/INST]' '<s>[INST] <Img><ImageHere></Img> Are there any polyps in this picture? Please answer Yes or No. [/INST]' --seed 42
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--prefix", type=str, 
                        default='AROARU6TOWKRWGKMSM6E7:Yao.Jiang@mbzuai.ac.ae/COD10K/Imgs/')
    parser.add_argument("--results_path", type=str, 
                        default='./COD.lst')
    parser.add_argument("--prompts", type=str, nargs='*',
                        default=['<s>[INST] <Img><ImageHere></Img> [detection] The camouflaged objects. [/INST]',
                     '<s>[INST] <Img><ImageHere></Img> Are there any camouflaged objects in this picture? Please answer Yes or No. [/INST]'])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--options",nargs="+", help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True

        return False

def get_dataset_using_boto3():
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    image_list = []
    for page in pages:
        temp = [x['Key'] for x in page['Contents'] if not x['Key'].endswith('/')]
        image_list = image_list + temp
    
    return(image_list)

def remove_cumputed(image_list, file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        print('Computed image number: {}'.format(len(lines)))
        for line in lines:
            temp = line.split('@#@')[0]
            if prefix + temp in image_list:
                image_list.remove(prefix + temp)
                
    f.close()
    return image_list
        


print('Initializing Chat')
args = parse_args()
cfg = Config(args)
# print(args)
# print(cfg)

device = 'cuda:{}'.format(args.gpu_id)
print('device: {}'.format(device))

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

cudnn.benchmark = False
cudnn.deterministic = True

##########################################Initializing model and image processor###################################################
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100
model = model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


################################Get image list######################################################
s3_client = boto3.client("s3", region_name="me-central-1")
bucket = 'mbz-hpc-aws-researchers'
prefix = args.prefix
image_list = get_dataset_using_boto3()
if os.path.exists(args.results_path):
    image_list = remove_cumputed(image_list, args.results_path)
print('Total image number: {}'.format(len(image_list)))
print('Examples: {}'.format(image_list[0]))


################################Prepare record file######################################################
results_path = args.results_path
if not os.path.exists(args.results_path):
    lines = ''
    for i, prompt in enumerate(args.prompts):
        temp = prompt.replace('<s>[INST] <Img><ImageHere></Img>', '({})'.format(i+1))
        temp = temp.replace('[/INST]', '')
        lines  = lines + temp
    with open(results_path, 'w') as f:
        f.writelines(lines + '\n')
    f.close()


questions_messages = args.prompts

with torch.no_grad():
    for img_path in image_list:

        ###################################################load image###################################################
        byte_io = io.BytesIO()
        s3_client.download_fileobj(bucket, img_path, byte_io)
        byte_io.seek(0)
        byteImg = Image.open(byte_io)
        byteImg = byteImg.convert('RGB')
        

        answers = []
        #############################################Ask one question each time##################################
        for prompt in questions_messages:
            
            ########################################Enocde image########################################
            img = vis_processor(byteImg.copy())
            img = img.unsqueeze(0).to(device)
            img_embed, _ = model.encode_img(img)
            
            ########################################encode prompt########################################
            # prompt = '<s>[INST] <Img><ImageHere></Img> [detection] The camouflaged objects. [/INST]</s> [/INST]'
            embs = model.get_context_emb(prompt, [img_embed])

            ########################################Prepare the necessary variables########################################
            current_max_len = embs.shape[1] + 500
            if current_max_len - 2000 > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - 2000)
            embs = embs[:, begin_idx:]

            stop_words_ids = [torch.tensor([2]).to(device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

            generation_kwargs = dict(
                inputs_embeds=embs,
                max_new_tokens=500,
                stopping_criteria=stopping_criteria,
                num_beams=1,
                do_sample=True,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.05,
                length_penalty=1,
                temperature=float(1.0),
            )

            # streamer = TextIteratorStreamer(model.llama_tokenizer, skip_special_tokens=True)
            # generation_kwargs['streamer'] = streamer

            ################################################generate################################################
            with model.maybe_autocast():
                output = model.llama_model.generate(**generation_kwargs)[0]

            output_text = model.llama_tokenizer.decode(output, skip_special_tokens=True)

            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            answers.append(output_text)

        ################################################record################################################
        output_test = img_path.replace(prefix, '')
        for answer in answers:
            output_test = output_test + '@#@' + answer.replace('\n', ' ')

        print(output_test)
        with open(results_path, 'a') as f:
            f.writelines(output_test + '\n')
        f.close()

