import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import json
import copy
from bert.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import h5py
from refer.refer import REFER
from args import get_parser
import transforms as T
from collections import namedtuple

SamMask = namedtuple("SamMask","mask")


def my_collate(batch):
     elem = batch[0]
     if isinstance(elem, tuple):  # Some custom condition
        sam_batch = []
        normal_batch = []
        for b in batch:
            per_batch =[]
            for e in b:
                if isinstance(e,SamMask):
                    sam_batch.append(e)
                else:
                    per_batch.append(e)
            normal_batch.append(per_batch)
        if sam_batch ==[]:
            return default_collate(batch)
        else:
            return default_collate(normal_batch)+[sam_batch]
     else:  # Fall back to `default_collate`
         return default_collate(batch)


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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 1. split the dataset
# 2. see the final json and check how to use in your dataset class
class ReferDataset_Semi(data.Dataset):
    def __init__(self,
                 args,
                image_transforms=None,
                target_transforms=None, 
                 split='train',
                 sup=True,label=True):
        

        super(ReferDataset_Semi, self).__init__()
        self.args = args
        self.split = split
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.max_tokens = 20
        # label = False
        self.label = label

        if sup == True :
            if args.dataset!='refcocog':
                refs_path = "./anns/{0}/{1}_{2}%_image.json".format(args.dataset,args.dataset,args.sup_percent)
            else:
                refs_path = "./anns/{0}/{1}/{2}_{3}%_image.json".format(args.dataset,args.splitBy,args.dataset,args.sup_percent)
            stat_refs_list=json.load(open(refs_path, 'r'))
        else:

            if label == True:
                percent = args.sup_percent
            else:
                percent = args.unsup_percent
            
            if args.dataset!='refcocog':
                refs_path = "./anns/{0}/{1}_{2}%_image.json".format(args.dataset,args.dataset,percent)
            else:
                refs_path = "./anns/{0}/{1}/{2}_{3}%_image.json".format(args.dataset,args.splitBy,args.dataset,percent)
            stat_refs_list=json.load(open(refs_path, 'r'))


        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]

        self.ref_ids = []
        self.getImgIds = {}
        for i in stat_refs_list['train']:
            self.getImgIds[i['mask_id']]=i['iid']
            self.ref_ids.append(i['mask_id']) 
        
        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        ref_ann = stat_refs_list['train']
        for r in ref_ann:
            ref = r['refs']

            sentences_for_ref = []
            attentions_for_ref = []

            for i,(el) in enumerate(ref):
                sentence_raw = el
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
        
    def get_classes(self):
        return self.classes
    
    def __len__(self):
        return len(self.ref_ids)

    def load_img_feats(self, idx):
        img_path=None
        if self.args.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join("./refer/data/images/mscoco/images/train2014",'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        img = Image.open(img_path).convert("RGB")

        if self.args.dataset in ['refcoco','refcoco+','refcocog']:
            if self.args.dataset !='refcocog':
                mask=np.load(os.path.join("./anns/{0}/masks".format(self.args.dataset,self.args.dataset),'%d.npy'%self.refs_anno[idx]['mask_id']))
            else:
                split_by = self.args.splitBy
                mask=np.load(os.path.join("./anns/{0}/{1}/masks".format(self.args.dataset,split_by,self.args.dataset),'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([img.shape[0],img.shape[1],1],dtype=np.float)

        return img,mask

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.getImgIds[this_ref_id]
        img, target = self.load_img_feats(index) 

        annot = np.zeros(target.shape)
        annot[target == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        
        if self.image_transforms is not None and self.label== True:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)
        elif self.image_transforms is not None and self.label==False:
            weak_aug =  self.image_transforms[0]
            strong_aug = self.image_transforms[1]

            img_w,target_w = weak_aug(img,annot)
            img_s,target_s = strong_aug(img_w,target_w)

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]

        if self.label ==True:
            return img, target, tensor_embeddings, attention_mask
        else:
            img,target = self.target_transform(img,annot)
            return img,target,img_w,target_w,img_s,target_s,tensor_embeddings,attention_mask


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
        self.length=None

    def __len__(self):
        return len(self.batch_sampler.sampler) if self.length is None else max(len(self.batch_sampler.sampler),self.length)

    def __iter__(self):
        for i in range(len(self) if self.length is None else max(len(self),self.length)):
        # while True:
            yield next(self.iterator)
    def set_length(self,length):
        self.length=length


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class ReferSAMOfflineRleDataset(data.Dataset):
    def __init__(self,
                args,
                image_transforms=None,
                target_transforms=None, 
                    split='train',
                    sup=True,label=True):
        

        super(ReferSAMOfflineRleDataset, self).__init__()
        self.args = args
        self.split = split
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.max_tokens = 20
        # label = False
        self.label = label

        if sup == True :
            if args.dataset!='refcocog':
                refs_path = "./anns/{0}/{1}_{2}%_image.json".format(args.dataset,args.dataset,args.sup_percent)
            else:
                refs_path = "./anns/{0}/{1}/{2}_{3}%_image.json".format(args.dataset,args.splitBy,args.dataset,args.sup_percent)
            stat_refs_list=json.load(open(refs_path, 'r'))
        else:

            if label == True:
                percent = args.sup_percent
            else:
                percent = args.unsup_percent
            
            if args.dataset!='refcocog':
                refs_path = "./anns/{0}/{1}_{2}%_image.json".format(args.dataset,args.dataset,percent)
            else:
                refs_path = "./anns/{0}/{1}/{2}_{3}%_image.json".format(args.dataset,args.splitBy,args.dataset,percent)
            stat_refs_list=json.load(open(refs_path, 'r'))

        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]

        self.ref_ids = []
        self.getImgIds = {}
        for i in stat_refs_list['train']:
            self.getImgIds[i['mask_id']]=i['iid']
            self.ref_ids.append(i['mask_id']) 
        
        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        ref_ann = stat_refs_list['train']
        for r in ref_ann:
            ref = r['refs']

            sentences_for_ref = []
            attentions_for_ref = []

            for i,(el) in enumerate(ref):
                sentence_raw = el
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
        
    def get_classes(self):
        return self.classes
    
    def __len__(self):
        return len(self.ref_ids)

    def load_img_feats(self, idx):
        img_path=None
        if self.args.dataset in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join("./refer/data/images/mscoco/images/train2014",'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        img = Image.open(img_path).convert("RGB")

        if self.args.dataset in ['refcoco','refcoco+','refcocog']:
            if self.args.dataset !='refcocog':
                mask=np.load(os.path.join("./anns/{0}/masks".format(self.args.dataset),'%d.npy'%self.refs_anno[idx]['mask_id']))
            else:
                split_by = self.args.splitBy
                mask=np.load(os.path.join("./anns/{0}/{1}/masks".format(self.args.dataset,split_by),'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([img.shape[0],img.shape[1],1],dtype=np.float)

        return img,mask

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.getImgIds[this_ref_id]
        img, target = self.load_img_feats(index) 
        mask_id = self.refs_anno[index]['mask_id']
        if self.args.dataset!='refcocog':
            sam_rle_mask = json.load(open('./anns/{0}/sam_rle_mask/{1}.json'.format(self.args.dataset,mask_id)))
        else:
            sam_rle_mask = json.load(open('./anns/{0}/{1}/sam_rle_mask/{2}.json'.format(self.args.dataset,self.args.splitBy,mask_id)))
        N = len(sam_rle_mask)
        H,W = sam_rle_mask[0]['size']
        sam_masks = np.zeros((N,H,W),dtype=np.uint8)
        for k in range(N):
            sam_masks[k] = rle2mask(sam_rle_mask[k])
        sam_mask = torch.nn.functional.interpolate(torch.tensor(sam_masks)[None,:,:,:],size=(self.args.img_size,self.args.img_size),mode='nearest').numpy()[0,:,:,:]
        h,w = target.shape

        annot = np.zeros(target.shape)
        annot[target == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        
        if self.image_transforms is not None and self.label== True:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)
        elif self.image_transforms is not None and self.label==False:
            weak_aug =  self.image_transforms[0]
            strong_aug = self.image_transforms[1]
            img_w,target_w,sam_mask = weak_aug(img,annot,point=None,sam_mask=sam_mask)
            img_s,target_s = strong_aug(img_w,target_w)

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]
        if self.label ==True:
            return img, target, tensor_embeddings, attention_mask
        else:
            img,target = self.target_transform(img,annot)
            return img,target,img_w,target_w,img_s,target_s,tensor_embeddings,attention_mask,SamMask(mask=sam_mask)

def loader(args,dataset: torch.utils.data.Dataset, rank: int,num_replicas, shuffle,drop_last=False):

    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                   num_replicas=num_replicas,
                                                                   shuffle=True,
                                                                   rank=rank)
    g = torch.Generator()
    g.manual_seed(args.seed)

    data_loader = InfiniteDataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                sampler=dist_sampler,
                                num_workers=0,
                                pin_memory=False,
                                drop_last=drop_last,
                                worker_init_fn=seed_worker,
                                generator=g,
                                collate_fn=my_collate)  
    return data_loader

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.sup_percent = 1
    args.unsup_percent= 99
    dataset = ReferDataset_Semi(args,split='train',image_transforms=get_transform(args))

    g = torch.Generator()
    g.manual_seed(args.seed)
    data_loader = InfiniteDataLoader(dataset,
                             batch_size=10,
                             shuffle=True,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=g)
    for _,ref_iter,image_iter, mask_iter in data_loader:
        print("1")
        # print(image_iter.size())
        # print(mask_iter.size())
        # print(box_iter.size())
        # print(ref_iter.size())
        # # cv2.imwrite('./test.jpg', image_iter.numpy()[0].transpose((1, 2, 0))*255)
        # # cv2.imwrite('./mask.jpg', mask_iter.numpy()[0].transpose((1, 2, 0))*255)
        # print(info_iter.size())
        # print(info_iter)





