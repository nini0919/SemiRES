import json
import random
import math
from tqdm import tqdm
import os

def pick_dataset(IMAGE_PATH, ANN_PATH):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(ANN_PATH[args.dataset],'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    print("The number of original annotations:", len(json_data['train']))

    json_data_new_label = {}
    json_data_new_unlabel = {}
    json_data_anns = json_data['train']
    id = list()
    for i in tqdm(range(len(json_data_anns))):
        if json_data_anns[i]['iid'] not in id:
            id.append(json_data_anns[i]['iid'])

    random.seed(0)  
    sample_num = math.floor(args.pick_percent*len(id))
    json_data_anns_label_id = random.sample(id, sample_num)
    
    json_data_new_label['train'] = list()
    json_data_new_unlabel['train'] = list()

    for i in tqdm(range(len(json_data_anns))):
        if json_data_anns[i]['iid'] in json_data_anns_label_id:
            json_data_new_label['train'].append(json_data_anns[i])
        else:
            json_data_new_unlabel['train'].append(json_data_anns[i])

    if args.dataset == 'refcocog':
        json_data_new_label['val'] = json_data['val']
        json_data_new_label['test'] = json_data['test']

        json_data_new_unlabel['val'] = json_data['val']
        json_data_new_unlabel['test'] = json_data['test']
    else:
        json_data_new_label['val'] = json_data['val']
        json_data_new_label['testA'] = json_data['testA']
        json_data_new_label['testB'] = json_data['testB']

        json_data_new_unlabel['val'] = json_data['val']
        json_data_new_unlabel['testA'] = json_data['testA']
        json_data_new_unlabel['testB'] = json_data['testB']

    print("The number of labeled annotations processed:", len(json_data_new_label['train']))
    print("The number of unlabeled annotations processed:", len(json_data_new_unlabel['train']))

    if args.pick_percent == 0.1:
        label = '_10%_image.json'
        unlabel = '_90%_image.json'
    elif args.pick_percent == 0.05:
        label = '_5%_image.json'
        unlabel = '_95%_image.json'
    elif args.pick_percent == 0.01:
        label = '_1%_image.json'
        unlabel = '_99%_image.json'
    elif args.pick_percent == 0.001:
        label = '_0.1%_image.json'
        unlabel = '_99.9%_image.json'

    json.dump(json_data_new_label, open(args.save_dir+args.dataset+label, 'w'))
    json.dump(json_data_new_unlabel, open(args.save_dir+args.dataset+unlabel, 'w'))

if __name__ == '__main__':
    ANN_PATH = {
            'refcoco': './anns/refcoco.json',
            'refcoco+': './anns/refcoco+.json', 
            'refcocog': './data/anns/refcocog.json', 
            'referit': './data/anns/refclef.json', 
            'flickr': './data/anns/flickr.json',
            'vg': './data/anns/vg.json',
            'merge':'./data/anns/merge.json'
            }
    IMAGE_PATH = {
              'refcoco': './refer/data/images/mscoco/images/train2014',
              'refcoco+': './refer/data/images/mscoco/images/train2014',
              'refcocog': './data/images/train2014',
              'referit': './data/images/refclef',
              'flickr': './data/images/flickr',
              'vg':'./data/images/VG',
              'merge':'./data/images/'
          }
    import argparse
    parser = argparse.ArgumentParser(description='Pick dataset of specific percentage')
    parser.add_argument("--dataset",type=str,default='refcoco+')
    parser.add_argument("--save-dir",type=str,default='./anns/refcoco+/')
    parser.add_argument("--pick-percent",type=float,default=0.1)
    args = parser.parse_args()
    pick_dataset(IMAGE_PATH,ANN_PATH)
