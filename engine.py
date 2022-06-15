import math
import sys
import random
import time
import datetime
from typing import Iterable
from estimator import *
import numpy as np
import torch
import torchvision
import torch.nn.functional as Func
import PIL
import util.misc as utils
from inference import keep_largest_connected_components

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def Cutout_augment(x, l, device, beta=1):
    lams = []
    try:
        x=x.tensors
    except:
        pass
    lam = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    bboxs = []
    x_flip = torch.flip(x,(0,))
    l_flip = torch.flip(l,(0,))
    for index in range(x.shape[0]):
        bbx1, bby1, bbx2, bby2= rand_bbox(x.shape, lam[index,0,0,0])
        x[index,:,bbx1:bbx2,bby1:bby2] = 0
        l[index,:,bbx1:bbx2,bby1:bby2]= 0
        bboxs.append([bbx1, bby1, bbx2, bby2])
    return x, l, bboxs

def Cutout_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, bboxs = Cutout_augment(samples, target_masks, device)
    return aug_samples, aug_targets, bboxs

def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim5(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def rotate(imgs,labels):
    num = imgs.shape[0]
    imgs_out_list = []
    labels_out_list = []
    angles = []
    
    for i in range(num):
        img = imgs[i,:,:,:]
        label = labels[i,:,:,:]

        angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
        
        rotated_img = torchvision.transforms.functional.rotate(img, angle, PIL.Image.NEAREST, False, None)
        rotated_label = torchvision.transforms.functional.rotate(label, angle, PIL.Image.NEAREST, False, None)
        
        imgs_out_list.append(rotated_img)
        labels_out_list.append(rotated_label)
        
        angles.append(angle)
    
    imgs_out = torch.stack(imgs_out_list)
    labels_out = torch.stack(labels_out_list)
    return imgs_out, labels_out, angles

def flip(imgs, labels):
    imgs_list = []
    labels_list = []
    flips = []
    for i in range(imgs.shape[0]):
        img = imgs[i,:,:,:]
        label = labels[i,:,:,:]

        flipped_img = img
        flipped_label = label

        flip_choice = int(random.random()*4)
        if flip_choice == 0:
            pass

        if flip_choice == 1:
            flipped_img = torch.flip(flipped_img,[1])
            flipped_label = torch.flip(flipped_label,[1])

        if flip_choice == 2:
            flipped_img = torch.flip(flipped_img,[2])
            flipped_label = torch.flip(flipped_label,[2])

        if flip_choice == 3:
            flipped_img = torch.flip(flipped_img,[1,2])
            flipped_label = torch.flip(flipped_label,[1,2])

        flips.append(flip_choice)
        imgs_list.append(flipped_img)
        labels_list.append(flipped_label)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    return imgs_out, labels_out, flips

def flip_back(outputs, flips):
    outs = []
    for i in range(outputs["pred_masks"].shape[0]):
        output = outputs["pred_masks"][i,:,:,:]
        flip_choice = flips[i]
        flipped_img = output

        if flip_choice == 0:
            pass

        if flip_choice == 1:
            flipped_img = torch.flip(flipped_img,[1])

        if flip_choice == 2:
            flipped_img = torch.flip(flipped_img,[2])

        if flip_choice == 3:
            flipped_img = torch.flip(flipped_img,[1,2])

        outs.append(flipped_img)
        outs = torch.stack(outs)
        return {"pred_masks":outs}

def rotate_back(outputs,angles):
    num = outputs["pred_masks"].shape[0]
    outputs_out_list = []
    
    for i in range(num):
        output = outputs["pred_masks"][i,:,:,:]
        angle = -angles[i]
        
        rotated_output =  torchvision.transforms.functional.rotate(output, angle, PIL.Image.NEAREST, False, None)
        
        outputs_out_list.append(rotated_output)
    
    outputs_out = torch.stack(outputs_out_list) 
    return {"pred_masks":outputs_out}

def Cutout(imgs,labels, device, n_holes=1, length=32):
    labels = [t["masks"] for t in labels]
    labels = torch.stack(labels)

    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []
    masks_list = []

    for i in range(num):
        label = labels[i,:,:,:]
        img = imgs[i,:,:,:]
        mask = np.ones((1, h, w), np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask
        label = label * mask
        imgs_list.append(img)
        labels_list.append(label)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, masks_out


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, dataloader_hold_dict, estimate_alpha):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items()}
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()

    model.train()
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets] 

        ## apply cutout, rotate, flip
        samples_cut, targets_cut, masks_cut  = Cutout(samples.tensors, targets, device)
        samples_cut, targets_cut, angles = rotate(samples_cut, targets_cut)
        samples_cut, targets_cut, flips = flip(samples_cut, targets_cut)
        targets_cut = to_onehot_dim5(targets_cut,device)
        targets = convert_targets(targets, device)
        ##

        # estimate alpha
        model.eval()
        outputs = model(samples.tensors, task)
        with torch.no_grad():
            alpha_dict = {0:{"n":None,"p":None,"g":None}, 1:{"n":None,"p":None,"g":None}, 2:{"n":None,"p":None,"g":None}, 3:{"n":None,"p":None,"g":None}}

            step_dict = {}
            for class_label in range(4):
                n_probs = []
                p_probs = []
                g_probs = []
                for batch_idx in range(targets.shape[0]):
                    u_index = 1-targets[batch_idx,4,:,:]
                    pos_index = targets[batch_idx,class_label,:,:]
                    p_prob = outputs["pred_masks"][batch_idx,class_label,:,:][u_index==1]#[pos_index==1]
                    n_prob = outputs["pred_masks"][batch_idx,class_label,:,:][u_index==0]
                    g_prob = pos_index[u_index == 1]
                    
                    n_probs = np.concatenate((n_probs, n_prob.cpu().detach().numpy()), axis=0)
                    p_probs = np.concatenate((p_probs, p_prob.cpu().detach().numpy()), axis=0)
                    g_probs = np.concatenate((g_probs, g_prob.cpu().detach().numpy()), axis=0)

                n_probs = np.asarray(n_probs)
                p_probs = np.asarray(p_probs)
                g_probs = np.asarray(g_probs)

                step_dict.update({class_label:{"n":n_probs,"p":p_probs,"g":g_probs}})

            for class_label in range(4):
                alpha_dict[class_label]["p"] = step_dict[class_label]["p"].reshape((len(step_dict[class_label]["p"]),-1))
                alpha_dict[class_label]["n"] = step_dict[class_label]["n"].reshape((len(step_dict[class_label]["n"]),-1))
                alpha_dict[class_label]["g"] = step_dict[class_label]["g"].reshape((len(step_dict[class_label]["g"]),-1)) 
            ratio_dict = EM_estimate(alpha_dict)  
        ###

        # train model
        model.train()        
        outputs = model(samples.tensors, task)

        # rotate and fip back
        outputs_cut = model(samples_cut, task)
        outputs_back = rotate_back(outputs_cut,angles)
        outputs_back = flip_back(outputs_back, flips)
        ###

        ## Maximize marginal probability and calculate negative loss
        annotated_area = 1-targets[:,4:5,:,:]
        annotated_area = annotated_area.repeat(1,4,1,1)

        if estimate_alpha == True:
            pseudo_labels = torch.zeros_like(outputs["pred_masks"])
            for key, value in ratio_dict.items():
                if key != 0:
                    flat_labels = outputs["pred_masks"][:,key,:,:][targets[:,-1,:,:] == 1]
                    sorted_dices = np.argsort(flat_labels.cpu().detach().numpy())
                    sorted_labels = flat_labels[sorted_dices]
                    threshold_pseudo = sorted_labels[max(int(len(sorted_dices)*(1-ratio_dict[key]))-1,0)]
                    if step == 0:
                        print(key, ":", threshold_pseudo.item())
                    pseudo_labels[:,key,:,:][outputs["pred_masks"][:,key,:,:] < threshold_pseudo] = 1

            pseudo_labels = pseudo_labels*(1-annotated_area)
            label_pseudo = pseudo_labels.sum(1)
            label_pseudo[label_pseudo >= 1] = 1
            outputs_pseudo = outputs["pred_masks"]*(1-pseudo_labels)
            
            pseudo_loss = -label_pseudo * torch.log(outputs_pseudo.sum(1)+1e-12)
            if pseudo_loss.sum()>0:
                pseudo_loss = pseudo_loss.sum() / label_pseudo.sum()
            else:
                pseudo_loss = pseudo_loss.mean()
        ###

        #PCE loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in ['loss_CrossEntropy'])

        #PCE loss for augmented slice
        loss_cut_dict = criterion(outputs_cut, targets_cut)
        losses_cut = sum(loss_cut_dict[k] * weight_dict[k] for k in loss_cut_dict.keys() if k in ['loss_CrossEntropy'])

        #global consistency loss
        invariant_loss = 1- Func.cosine_similarity(outputs["pred_masks"], outputs_back["pred_masks"], dim=1)
        invariant_loss = invariant_loss*masks_cut
        invariant_loss = 0.05*invariant_loss.mean()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = { f'{k}_unscaled': v for k, v in loss_dict_reduced.items() }
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy']}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())


        if step == 0:
            print("Positive loss:", losses.item())
            print("cut loss:", losses_cut.item())
            print("invariant loss", invariant_loss.item())
            if estimate_alpha == True:
                print("Pseudo loss:", pseudo_loss.item())
                print("estimated ratio:", ratio_dict)


        final_losses = losses + losses_cut + invariant_loss
        if estimate_alpha == True:
            final_losses = final_losses + pseudo_loss


        optimizer.zero_grad()
        final_losses.backward()
        optimizer.step()

        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, epoch, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ] 
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot= convert_targets(targets,device)
        outputs = model(samples.tensors, task)

        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict.keys()}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:  
            ##original  
            sample_list.append(samples.tensors[0])
            ##
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            
            ##original
            target_list.append(targets[0]['masks'])
            ##
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    writer.add_scalar('avg_loss', stats['loss_CrossEntropy'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)

    f = open("/home/zhangke/EM_dice.txt","a")
    f.write("epoch"+str(epoch)+",Avg:"+str(stats["Avg"])+",Lv:"+str(stats["Lv"])+",Rv:"+str(stats["Rv"])+",Myo:"+str(stats["Myo"]))
    f.write("\r\n")
    f.close()
    
    return stats
