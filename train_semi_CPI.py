import datetime
import os
import time
import math

import torch
import torch.utils.data
from torch import nn
import torch.distributed as dist

from functools import reduce
import operator
from bert.modeling_bert import BertModel
from dataset_semi  import ReferDataset_Semi,ReferSAMOfflineRleDataset,loader
import torchvision
from lib import segmentation
from test import computeIoU
from loss.dice_loss import DiceLoss
from loss.bce_loss import WeightedCrossEntropyLoss
import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

import wandb

import random

from PIL import Image,ImageFilter
import cv2

run = wandb.init(project="lavt_new")
config = run.config

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    # if image_set == 'val':
    #     mode = True
    # else:
    #     mode = False
    mode = False
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=mode
                      )
    num_classes = 2

    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return T.Compose(transforms)

def save_model(single_model,single_bert_model,optimizer,epoch,lr_scheduler,cls='teacher'):

    if single_bert_model is not None:
        dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                        'lr_scheduler': lr_scheduler.state_dict()}
    else:
        dict_to_save = {'model': single_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                        'lr_scheduler': lr_scheduler.state_dict()}
    utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                '{0}_model_latest_{1}.pth'.format(cls,args.model_id)))

def get_semi_transform(args,is_aug=False,mode='weak'):
    transforms = [T.Resize(args.img_size, args.img_size),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
    if is_aug:
        if mode=='weak':
            w_transform = [
                T.RandomGaussianBlur(p=0.5),
                T.RandomHorizontalFlip(p=0),
            ]
            return T.Compose(transforms+w_transform+[T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif mode=='strong':
            s_transform = [T.RandomColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25),p=0.5)]
            return T.Compose(s_transform+[T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return T.Compose(transforms+[T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_BERT_model(args):
    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None
    return bert_model,single_bert_model

def get_optim_param(single_model,single_bert_model):
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    return params_to_optimize

def freeze_params(single_model,single_bert_model):
    for p in single_model.parameters():
        p.requires_grad = False
    
    for p in single_bert_model.parameters():
        p.requires_grad = False

@torch.no_grad()
def EMA_update(ema_rate,student,teacher):
    keep_rate = ema_rate
    student_model_dict=student.state_dict()
    new_teacher_dict = OrderedDict()
    # EMA
    for key, value in teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    teacher.load_state_dict(new_teacher_dict)

def check_param(model_1,model_2):
    cnt = 0
    model_2_state_dict = model_2.state_dict()
    # EMA
    for key, value in model_1.state_dict().items():
        if key in model_2_state_dict.keys():
            if model_2_state_dict[key].sum()!=value.sum():
                cnt +=1
        else:
            raise Exception("{} is not found in student model".format(key))
    print(cnt)

def log_print():
    pass

# def criterion(input, target):
#     # target.shape:[8, 2, 480, 480]
#     #dice loss
#     dice = DiceLoss(use_sigmoid=False)
#     dice_loss = dice(input, target)
#     #ce loss
#     weight = torch.FloatTensor([0.9, 1.1]).cuda()
#     ce_loss = nn.functional.cross_entropy(input, target, weight=weight)
#     loss = dice_loss + ce_loss
#     return ce_loss,dice_loss,loss

def bce_weight(soft_label):
    prob = soft_label[:,1,:,:]
    #initialize Gaussian mean and variance
    u1 = 0.5  
    sigma1 = 0.1  
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
    right = np.exp(-(prob.detach().cpu().numpy() - u1)**2 / (2 * sigma1))
    weight_numpy =  1.3 - left*right
    weight = torch.from_numpy(weight_numpy).cuda()
    return weight

def softmatch_weight(soft_label):
    # pred: b,c,h,w
    prob = soft_label[:,1,:,:]

    u1 = prob.detach().cpu().numpy().mean()  
    sigma1 = prob.detach().cpu().numpy().std() 
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
    right = np.exp(-(prob.detach().cpu().numpy() - u1)**2 / (2 * sigma1))
    weight_numpy =  1.3 - left*right
    weight = torch.from_numpy(weight_numpy).cuda()
    return weight

def weighted_ce_loss(input,target,weight):
    loss = WeightedCrossEntropyLoss()
    return loss(input,target,weight)

def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    cross_entropy = nn.CrossEntropyLoss(weight=weight,reduction='none')
    return cross_entropy(input, target).mean()

# def criterion(input, target):
#     weight = torch.FloatTensor([0.9, 1.1]).cuda()
#     return nn.functional.cross_entropy(input, target, weight=weight)

def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)
    run.log({"Mean IoU":mIoU})
    return 100 * iou, 100 * cum_I / cum_U

def dist_evaluate(model,data_loader,bert_model,device):
    model.eval()
    bert_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_its = torch.tensor(0.,device=device)
    acc_ious = torch.tensor(0.,device=device)
    # evaluation variables
    cum_I, cum_U = torch.tensor(0,device=device), torch.tensor(0,device=device)
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list), dtype=torch.int32,device=device)
    seg_total = torch.tensor(0,device=device)
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            for j in range(sentences.size(-1)):
                total_its += 1
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                iou, I, U = IoU(output, target)
                acc_ious += iou
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output
            if bert_model is not None:
                del last_hidden_states, embedding
    dist.barrier()
    Iou_size = total_its.clone()
    gather_list = [torch.zeros(1,device=device) for k in range(dist.get_world_size())]
    dist.all_gather(gather_list,Iou_size)
    Iou_size = torch.concat(gather_list,dim=0).flatten()
    del gather_list
    dist.all_reduce(cum_I,dist.ReduceOp.SUM)
    dist.all_reduce(cum_U,dist.ReduceOp.SUM)
    dist.all_reduce(seg_correct,dist.ReduceOp.SUM)
    dist.all_reduce(seg_total,dist.ReduceOp.SUM)
    dist.all_reduce(total_its,dist.ReduceOp.SUM)
    dist.all_reduce(acc_ious,dist.ReduceOp.SUM)

    iou = acc_ious / total_its
    mean_IoU = torch.tensor(mean_IoU,device=device)

    gather_list = [torch.zeros(int(Iou_size[k].item())) for k in range(dist.get_world_size())]
    
    dist.all_gather_object(gather_list,mean_IoU)
    seg_correct = np.array(seg_correct.cpu().numpy())
    seg_total = seg_total.item()
    mean_IoU = []
    for k,ts in enumerate(gather_list):
        mean_IoU.append(ts.cpu())
    mean_IoU = torch.concat(mean_IoU)
    mean_IoU = mean_IoU.flatten().numpy()
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I.item() * 100. / cum_U.item())
    print(results_str)

    run.log({"Mean IoU":mIoU})

    del gather_list,seg_total,seg_correct,acc_ious,total_its
    gc.collect()
    torch.cuda.empty_cache()
    return 100 * iou, 100 * cum_I / cum_U

def unnormalize(img):
    return img * np.array([[[0.229,0.224,0.225]]])+np.array([[[0.485,0.456,0.406]]])

def train_one_epoch(args,teacher,teacher_bert,student,student_bert,optimizer,
                    lr_scheduler, train_loader_label,train_loader_unlabel,
                    criterion, epoch, print_freq,iterations,ema_rate):
    
    student.train()
    student_bert.train()
    # combine dataloader 
    train_loader_label.sampler.set_epoch(epoch)
    train_loader_unlabel.sampler.set_epoch(epoch)
    train_loader_label.set_length(max(len(train_loader_label),len(train_loader_unlabel)))
    train_loader_unlabel.set_length(max(len(train_loader_label),len(train_loader_unlabel)))
    length = len(train_loader_unlabel)
    loader = zip(train_loader_label,train_loader_unlabel)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    cnt_sam_miss = torch.tensor(0,dtype=torch.float32).cuda()
    total_samles = train_loader_unlabel.dataset.__len__()
    train_loss = 0
    total_its = 0
    
    for data in metric_logger.log_semi(loader,print_freq,length,header):
        total_its+=1
        (label_img,label_target,label_sentences,label_attentions),(unlabel_img,unlabel_target,unlabel_img_w,unlabel_target_w,unlabel_img_s,unlabel_target_s,unlabel_sentences,unlabel_attentions,sam_masks) = data
        label_img, label_target, label_sentences, label_attentions = label_img.cuda(non_blocking=True),\
                                               label_target.cuda(non_blocking=True),\
                                               label_sentences.cuda(non_blocking=True),\
                                               label_attentions.cuda(non_blocking=True)
        
        unlabel_img_w,unlabel_img_s, unlabel_sentences, unlabel_attentions = unlabel_img_w.cuda(non_blocking=True),\
                                        unlabel_img_s.cuda(non_blocking=True),\
                                        unlabel_sentences.cuda(non_blocking=True),\
                                        unlabel_attentions.cuda(non_blocking=True)
        unlabel_sentences, unlabel_attentions =  unlabel_sentences.cuda(non_blocking=True),\
                                unlabel_attentions.cuda(non_blocking=True)
        
        label_sentences = label_sentences.squeeze(1)
        label_attentions = label_attentions.squeeze(1)

        unlabel_sentences = unlabel_sentences.squeeze(1)
        unlabel_attentions = unlabel_attentions.squeeze(1)

        all_img = torch.concat([label_img,unlabel_img_s])
        all_sentences = torch.concat([label_sentences,unlabel_sentences])
        all_attentions = torch.concat([label_attentions,unlabel_attentions])

        if student_bert is not None:
            last_hidden_states = student_bert(all_sentences, attention_mask=all_attentions)[0]  # (6, 10, 768)
            all_embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            all_attentions = all_attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            stu_pred_all = student(all_img, all_embedding, l_mask=all_attentions)
        else:
            stu_pred_all = student(all_img, all_sentences, l_mask=all_attentions)    
        stu_pred_l, stu_pred_u = stu_pred_all[:len(label_img)], stu_pred_all[len(label_img):]
        #sup_ce,sup_dice,sup_loss = criterion(stu_pred_l, label_target)
        sup_loss = criterion(stu_pred_l, label_target)
        
        teacher.eval()
        teacher_bert.eval()
        with torch.no_grad():
            if teacher_bert is not None:
                unlabel_last_hidden_states = teacher_bert(unlabel_sentences, attention_mask=unlabel_attentions)[0]  # (6, 10, 768)
                unlabel_embedding = unlabel_last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                unlabel_attentions = unlabel_attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
                tea_pred_unl = teacher(unlabel_img_w, unlabel_embedding, l_mask=unlabel_attentions)
            else:
                tea_pred_unl = teacher(unlabel_img_w, unlabel_sentences, l_mask=unlabel_attentions) 
        tea_pred_unl = torch.softmax(tea_pred_unl,dim=1) 
        weight = bce_weight(tea_pred_unl)
        c= torch.argmax(tea_pred_unl,dim=1)
        
        # refine teacher prediction
        batch_valid_idx = []
        refine_tea_pred = torch.zeros_like(c,dtype=torch.int64,device=c.device)
        for b in range(args.batch_size):
            sam_mask = sam_masks[b].mask.cuda()
            cnt = 0
            for t in range(sam_mask.shape[0]):
                _foreground = sam_mask[t]
                inter_1 = (_foreground * c[b]).sum()/(c[b].sum()+1e-9)
                inter_2 = (_foreground * c[b]).sum()/(_foreground.sum())
                if inter_1>0.7 or inter_2 > 0.7:
                    refine_tea_pred[b][_foreground.bool()] = 1
                else:
                    cnt +=1
            if cnt ==sam_mask.shape[0]:
                cnt_sam_miss = cnt_sam_miss + 1
                refine_tea_pred[b] = c[b]
                batch_valid_idx.append(b)
        wo_weight_idx= list(set(range(args.batch_size))-set(batch_valid_idx))
        if len(wo_weight_idx) ==0:
            unsup_sam_loss = 0
        else:
            unsup_sam_loss = criterion(stu_pred_u[wo_weight_idx], refine_tea_pred[wo_weight_idx])

        if len(batch_valid_idx) ==0:
            unsup_ori_loss = 0
        else:
            unsup_ori_loss = weighted_ce_loss(stu_pred_u[batch_valid_idx], refine_tea_pred[batch_valid_idx],weight[batch_valid_idx])
        unsup_loss = unsup_ori_loss + unsup_sam_loss

        if len(wo_weight_idx) != 0 and len(batch_valid_idx)!=0:
            unsup_loss = unsup_loss / 2.0
    
        total_loss = sup_loss + args.unsup_lambda * unsup_loss
        metric_logger.update(loss=total_loss,sup_loss=sup_loss,unsup_loss= unsup_loss*args.unsup_lambda,lr=optimizer.param_groups[0]["lr"])           

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        total_loss.backward()

        optimizer.step()
        lr_scheduler.step()
        EMA_update(ema_rate,student,teacher)
        EMA_update(ema_rate,student_bert,teacher_bert) 
        
        torch.cuda.synchronize()
        train_loss += total_loss.item()
        iterations += 1


        run.log({"loss": total_loss,'lr':optimizer.param_groups[0]["lr"],'sup_loss':sup_loss,'unsup_loss':unsup_loss})

        # del image, target, sentences, attentions, loss, output, data
        # if bert_model is not None:
        #     del last_hidden_states, embedding
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    torch.distributed.all_reduce(cnt_sam_miss,torch.distributed.ReduceOp.SUM)
    return cnt_sam_miss/total_samles

def main(args):
    if not osp.exist(args.output_dir):
        os.mkdirs(args.output_dir)
    setup_seed(args.seed)


    ###########################
    # 1. load data
    ###########################
    # from algorithms.semi_sup.augmentation import augmentation_transform_Withlabel
    # load label dataset
    dataset_label = ReferDataset_Semi(args,split='train',image_transforms=get_transform(args),sup=False,label=True)
   
    semi_transform = (get_semi_transform(args,is_aug=True,mode='weak'),get_semi_transform(args,is_aug=True,mode='strong')) 
    # load unlabel dataset
    dataset_unlabel = ReferSAMOfflineRleDataset(args,split='train',image_transforms=semi_transform,target_transforms=get_transform(args),sup=False,label=False)

    # load val dataset
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()  

    # loader
    dataset_label_loader = loader(args, dataset_label, global_rank, num_tasks, shuffle=False, drop_last=True)
    dataset_unlabel_loader = loader(args, dataset_unlabel, global_rank, num_tasks, shuffle=False, drop_last=True)
    

    # val dist loader

    # test_sampler = torch.utils.data.DistributedSampler(dataset_test)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=0,pin_memory=False)


    ###########################
    # 2. model initialization
    ###########################
    print(args.model)
    # teacher model
    teacher_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)  
    teacher_model.cuda()  
    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank], find_unused_parameters=True)
    teacher_single_model = teacher_model.module

    # student model
    student_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                            args=args)
    student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)  
    student_model.cuda()    
    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank], find_unused_parameters=True)
    student_single_model = student_model.module  

    # bert model
    teacher_bert_model,teacher_single_bert_model = get_BERT_model(args)
    student_bert_model,student_single_bert_model = get_BERT_model(args)

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        student_single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            student_single_bert_model.load_state_dict(checkpoint['bert_model'])

    # load Burn-In Model
    Burn_In_checkpoint = torch.load(args.burn_in_ckpt,map_location='cpu')
    student_single_model.load_state_dict(Burn_In_checkpoint['model'])
    teacher_single_model.load_state_dict(Burn_In_checkpoint['model'])
    student_single_bert_model.load_state_dict(Burn_In_checkpoint['bert_model'])
    teacher_single_bert_model.load_state_dict(Burn_In_checkpoint['bert_model'])
    ##############################
    # 3. optimizer & scheduler
    ##############################   

    # parameters to optimize
    # student params
    student_params_to_optimize = get_optim_param(student_single_model,student_single_bert_model)
    # teacher params is frozen（without gradient）
    freeze_params(teacher_model,teacher_bert_model)
    
    # optimizer
    optimizer = torch.optim.AdamW(student_params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,   
                                                     lambda x: (1 - x / ((len(dataset_unlabel_loader)+len(dataset_label_loader)) * args.epochs)) ** 0.9)


    ######################################
    # 4. resume
    ######################################
    
    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    

    ######################################
    # 5. training loop
    ######################################
    # training loops
    mis_rate_avg = torch.tensor(0,dtype=torch.float32).cuda()
    patience = 0 # early stop signal
    iter_eopch = 0
    patience = torch.tensor(0).cuda() # early stop signal
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        mis_match_rate = train_one_epoch(args,teacher_model,teacher_bert_model,student_model,student_bert_model,optimizer,
                    lr_scheduler, dataset_label_loader,dataset_unlabel_loader,
                    criterion, epoch, args.print_freq,iterations,ema_rate=0.9996)
        print('sam miss matched rate in epoch [{0}]: {1}%'.format(iter_eopch,round(mis_match_rate.item(),4)*100))
        mis_rate_avg +=mis_match_rate
        iter_eopch+=1
        # iou, overallIoU = evaluate_from_test(teacher_model, data_loader_test, teacher_bert_model,next(teacher_bert_model.parameters()).device)
        # iou, overallIoU = 0,0
        iou, overallIoU = evaluate(teacher_model, data_loader_test, teacher_bert_model)
        # iou, overallIoU = dist_evaluate(teacher_model, data_loader_test, teacher_bert_model,device=patience.device)
        torch.cuda.empty_cache()
        dist.barrier()

        if overallIoU.item() == 0:
            save_model(teacher_single_model,teacher_single_bert_model,optimizer,epoch,lr_scheduler)
            save_model(student_single_model,student_single_bert_model,optimizer,epoch,lr_scheduler,'student')
            break
        
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        run.log({"Average object IoU":iou,"Overall IoU":overallIoU})

        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            patience = 0
            print('Better epoch: {}\n'.format(epoch))
            if teacher_single_bert_model is not None:
                dict_to_save = {'model': teacher_single_model.state_dict(), 'bert_model': teacher_single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': teacher_single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
        else:
            patience +=1
        if patience==5:
            print("Early stopping. The training terminated because there were no improvements for 5 consecutive epochs. ")
            break
    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('global avg sam miss matched rate: {0}%'.format(round(mis_rate_avg.item()/(iter_eopch+1),4)*100))

def evaluate_from_test(model,data_loader,bert_model,device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    acc_ious = 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                acc_ious+=this_iou
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding
    
    iou = acc_ious / seg_total
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)
    run.log({"Mean IoU":mIoU})
    return 100 * torch.tensor(iou), 100 * cum_I / cum_U

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
