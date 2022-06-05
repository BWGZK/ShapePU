import math
import sys
import random
import time
import numpy as np
import datetime
from estimator import *
import torch
import util.misc as utils

def augment(x, l, device, beta=0.5):
    mixs = []
    try:
        x=x.tensors
    except:
        pass
    mix = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    mix = torch.maximum(mix, 1 - mix)
    mix = mix.to(device)
    mixs.append(mix)
    xmix = x * mix + torch.flip(x,(0,)) * (1 - mix)
    lmix = l * mix + torch.flip(l,(0,)) * (1 - mix)
    return xmix, lmix, mixs

def mix_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, rates = augment(samples, target_masks, device)
    return aug_samples, aug_targets, rates

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


    model.eval()
    # alpha_dict = {0:{"n":None,"p":None,"g":None}, 1:{"n":None,"p":None,"g":None}, 2:{"n":None,"p":None,"g":None}, 3:{"n":None,"p":None,"g":None}}
    iterats_val= { k : iter(v) for k, v in dataloader_dict.items()}
    numbers_val = { k : len(v) for k, v in dataloader_dict.items() }

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
        targets_onehot= convert_targets(targets,device)

        # estimate alpha
        if estimate_alpha == True:
            model.eval()
            outputs = model(samples.tensors, task)
            with torch.no_grad():
                alpha_dict = {0:{"n":None,"p":None,"g":None}, 1:{"n":None,"p":None,"g":None}, 2:{"n":None,"p":None,"g":None}, 3:{"n":None,"p":None,"g":None}}

                step_dict = {}
                for class_label in range(4):
                    n_probs = []
                    p_probs = []
                    g_probs = []
                    for batch_idx in range(targets_onehot.shape[0]):
                        u_index = 1-targets_onehot[batch_idx,4,:,:]
                        pos_index = targets_onehot[batch_idx,class_label,:,:]
                        p_prob = outputs["pred_masks"][batch_idx,class_label,:,:][u_index==1]
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

        # train model
        model.train()        
        outputs = model(samples.tensors, task)

        if estimate_alpha == True:
            pseudo_labels = torch.zeros_like(outputs["pred_masks"])
            for key, value in ratio_dict.items():
                if key != 0:
                    flat_labels = outputs["pred_masks"][:,key,:,:][targets_onehot[:,-1,:,:] == 1]
                    sorted_dices = np.argsort(flat_labels.cpu().detach().numpy())
                    sorted_labels = flat_labels[sorted_dices]
                    # sorted_labels = sorted_labels[::-1]
                    try:
                        threshold_pseudo = sorted_labels[max(int(len(sorted_dices)*(1-ratio_dict[key]))-1,0)]
                        if step == 0:
                            print(key, ":", threshold_pseudo.item())
                        pseudo_labels[:,key,:,:][outputs["pred_masks"][:,key,:,:] < threshold_pseudo] = 1
                    except:
                        pass

        annotated_area = 1-targets_onehot[:,4:5,:,:]
        annotated_area = annotated_area.repeat(1,4,1,1)

        #calculate negative loss (pseudo loss)
        if estimate_alpha == True:
            pseudo_labels = pseudo_labels*(1-annotated_area)
            label_pseudo = pseudo_labels.sum(1)
            label_pseudo[label_pseudo >= 1] = 1
            outputs_pseudo = outputs["pred_masks"]*(1-pseudo_labels)
            
            pseudo_loss = -label_pseudo * torch.log(outputs_pseudo.sum(1)+1e-12)
            if pseudo_loss.sum()>0:
                pseudo_loss = pseudo_loss.sum() / label_pseudo.sum()
            else:
                pseudo_loss = pseudo_loss.mean()

        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in ['loss_CrossEntropy'])

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = { f'{k}_unscaled': v for k, v in loss_dict_reduced.items() }
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy']}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        #print loss values (optional)
        if step == 0:
            print("Positive loss:", losses.item())
            if estimate_alpha == True:
                print("Pseudo loss:", pseudo_loss.item())
                print("estimated ratio:", ratio_dict)


        final_losses = losses
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
        if step % round(total_steps / 7.) == 0:  
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
    
    return stats