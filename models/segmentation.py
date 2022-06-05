import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
from .UNet import build_UNet

class MSCMR(nn.Module):
    def __init__(self, args, freeze_whst=False):
        super().__init__()

        if freeze_whst:
            for p in self.parameters():
                p.requires_grad_(False)
        self.tasks = args.tasks
        self.UNet = build_UNet(args)


    def forward(self, samples: torch.Tensor, task):
        seg_masks = self.UNet(samples)
        out = {"pred_masks": seg_masks}
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for MMSCMR.
    """
    def __init__(self, losses, weight_dict, args):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.args = args

    def Lv(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.argmax(1)
        targets_masks = targets.argmax(1)
        dice=(2*torch.sum((src_masks==3)*(targets_masks==3),(1, 2)).float())/(torch.sum(src_masks==3,(1, 2)).float()+torch.sum(targets_masks==3,(1, 2)).float()+1e-10)
        return {"Lv": dice.mean()}

    def Myo(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.argmax(1)
        targets_masks = targets.argmax(1)
        dice=(2*torch.sum((src_masks==2)*(targets_masks==2),(1, 2)).float())/(torch.sum(src_masks==2,(1, 2)).float()+torch.sum(targets_masks==2,(1, 2)).float()+1e-10)
        return {"Myo": dice.mean()}
    
    def Rv(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.argmax(1)
        targets_masks = targets.argmax(1)
        dice=(2*torch.sum((src_masks==1)*(targets_masks==1),(1, 2)).float())/(torch.sum(src_masks==1,(1, 2)).float()+torch.sum(targets_masks==1,(1, 2)).float()+1e-10)
        return {"Rv": dice.mean()}

    def Avg(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.argmax(1)
        targets_masks = targets.argmax(1)
        avg_dice = 0
        for i in range(1,4,1):
            dice=(2*torch.sum((src_masks==i)*(targets_masks==i),(1, 2)).float())/(torch.sum(src_masks==i,(1, 2)).float()+torch.sum(targets_masks==i,(1, 2)).float()+1e-10)
            avg_dice += dice.mean()
        return {"Avg": avg_dice/3}

    def multiDice(self, outputs, targets, num_classes):
        multidice = []
        weights = []
        dice = - (2*outputs*targets + 1e-10) / (outputs + targets + 1e-10)
        dice = dice * targets
        return dice.mean()

    def loss_multiDice(self, outputs, targets):
        """
    	Compute multi-dice
    	"""
        num_classes = 4
        src_masks = outputs["pred_masks"]
        ##mixup
        target_masks = targets.argmax(1, keepdim=True)
        shp_y = src_masks.shape
        y = target_masks.long()
        y_onehot = torch.zeros((shp_y[0],5,shp_y[2],shp_y[3]))
        if src_masks.device.type == "cuda":
            y_onehot = y_onehot.cuda(src_masks.device.index)
        y_onehot.scatter_(1, y, 1).float()
        y_labeled = y_onehot[:,0:4,:,:]


        multiDice = self.multiDice(src_masks, y_labeled, num_classes)
        losses = {
            "loss_multiDice": multiDice,
        }
        return losses
 
    def loss_CrossEntropy(self, outputs, targets, eps=1e-12):
        src_masks = outputs["pred_masks"]
        y_labeled = targets[:,0:4,:,:]
        cross_entropy = -y_labeled*torch.log(src_masks+eps)
        flat_loss_weighted= cross_entropy.sum()/y_labeled.sum()
        losses = {
                "loss_CrossEntropy": flat_loss_weighted.mean(),
        }
        return losses
            
    def get_loss(self, loss, outputs, targets):
        loss_map = {'multiDice': self.loss_multiDice,
                    'Rv': self.Rv, 
                    'Lv': self.Lv, 
                    'Myo': self.Myo, 
                    'Avg': self.Avg, 
                    'CrossEntropy': self.loss_CrossEntropy,
                    }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        return losses


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
        return results

class Visualization(nn.Module):
    def __init__(self):
        super().__init__()
        
    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        
    def forward(self, inputs, outputs, labels, epoch, writer):
        self.save_image(inputs, 'inputs', epoch, writer)
        self.save_image(outputs.float(), 'outputs', epoch, writer)
        self.save_image(labels.float(), 'labels', epoch, writer)


def build(args):
    device = torch.device(args.device)
    model = MSCMR(args, freeze_whst=(args.frozen_weights is not None))
    weight_dict = {'Rv': args.Rv,
                   'Lv': args.Lv,
                   'Myo': args.Myo,
                   'Avg': args.Avg,
                   'loss_CrossEntropy': args.CrossEntropy_loss_coef,
                   }
    losses = ['multiDice', 'CrossEntropy','Rv','Lv','Myo','Avg']
    criterion = SetCriterion(losses=losses, weight_dict=weight_dict, args=args)
    criterion.to(device)
    visualizer = Visualization()
    postprocessors = {'segm': PostProcessSegm()}

    return model, criterion, postprocessors, visualizer


