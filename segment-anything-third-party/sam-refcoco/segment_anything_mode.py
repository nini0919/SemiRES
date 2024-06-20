import numpy as np
import torch
import torch.distributed as dist

import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
import os
import sys
from operator import itemgetter
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import argparse


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def tag_masks(masks):
    H,W = masks[0]['segmentation'].shape
    ret_mask = np.zeros((H,W),dtype=np.int64)
    for idx,m in enumerate(masks):
        ret_mask[m['segmentation']]=idx+1
    return ret_mask

def rle2mask(rle_dict):
    height, width = rle_dict["size"]
    mask = np.zeros(height * width, dtype=np.uint8)

    rle_array = np.array(rle_dict["counts"])
    starts = rle_array[0::2]
    lengths = rle_array[1::2]

    current_position = -1
    for start, length in zip(starts, lengths):
     #   current_position += start
        mask[start-1:start-1 + length] = 1
      #  current_position += length

    mask = mask.reshape((height, width), order='F')
    return mask


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 
    1 - mask, 
    0 - background
    
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    
    runs[1::2] -= runs[::2]
    seg=[]
    
    for x in runs:
        
        seg.append(int(x))
    size=[]
    for x in img.shape:
         size.append(int(x))
    result=dict()
    result['counts']=seg
    result['size']=size
    return result

def save_masks(save_path,masks):
    out_list = []
    for idx,m in enumerate(masks):
        rle_result=mask2rle(m['segmentation'])
        out_list.append(rle_result)
    with open(save_path,'w') as f:
        json.dump(out_list,f)


img_path = "./refer/data/images/mscoco/images/train2014"
sam_checkpoint = "./segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
dataset='refcoco'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default='refcoco')
parser.add_argument("--annotation_path",type=str,default='./anns/refcoco.json')
args = parser.parse_args()

dataset=args.dataset
annotation_path = args.annotation_path

dist.init_process_group(backend="nccl", init_method='env://', world_size=-1, rank=-1, group_name='')

world_size = dist.get_world_size()
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.cuda()
mask_generator = SamAutomaticMaskGenerator(sam)
if not osp.exists(f'./anns/{dataset}/sam_rle_mask'):
    os.mkdirs(f'./anns/{dataset}/sam_rle_mask')

with open(annotation_path) as f:
    ref_data = json.load(f)

for idx,ref in tqdm(enumerate(ref_data['val']),total=len(ref_data['val'])):
    if idx % world_size!=local_rank:
        continue
    image_id = ref['iid'] # image_id
    ref_bbox = ref['bbox']
    cat_id = ref['cat_id']
    mask_id = ref['mask_id']
    if mask_id!=3745:
        continue
    save_path = "./anns/{}/sam_rle_mask/{}.json".format(dataset,mask_id)
    if os.path.exists(save_path):
        continue
    print(mask_id)
    img_path = os.path.join("./refer/data/images/mscoco/images/train2014",'COCO_train2014_%012d.jpg'%image_id)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    save_masks(save_path,masks)