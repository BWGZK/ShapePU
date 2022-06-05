import os
import argparse
import datetime
import random
import json
import time
from pathlib import Path
from tensorboardX import SummaryWriter
from copy import deepcopy
import clr
from inference import infer
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader, DistributedSampler
import data
import util.misc as utils
from data import build
from engine import evaluate, train_one_epoch
from models import build_model
from estimator import *


def get_args_parser():
    # define task, label values, and output channels
    tasks = {
        #'MR': {'lab_values': [0, 200, 500, 600], 'out_channels': 4}
        'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}
        }
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--tasks', default=tasks, type=dict)
    parser.add_argument('--model', default='MSCMR', required=False)
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--in_channels', default=1, type=int)
    
    # * position emdedding
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * heads and tails
    parser.add_argument('--num_pool', default=4, type=int,
                        help="Number of pooling layers"
                        )
    parser.add_argument('--return_interm', action='store_true', default=True,
                        help='whether to return intermediate features'
                        )

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer"
                        )
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer"
                        )
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks"
                        )
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)"
                        )
    parser.add_argument('--embedding_size', default=16, type=int,
                        help='size of embeddings projected by head module'
                        )
    parser.add_argument('--patch_size', default=4, type=int,
                        help='size of cropped small patch'
                        )
    parser.add_argument('--num_queries', default=256, type=int,
                        help="Number of query slots"
                        )
    parser.add_argument('--dropout', default=0.5, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true', default = True)

    # * Loss coefficients
    parser.add_argument('--multiDice_loss_coef', default=0, type=float)
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--Rv', default=1, type=float)
    parser.add_argument('--Lv', default=1, type=float)
    parser.add_argument('--Myo', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)
    # dataset parameters
    
    parser.add_argument('--dataset', default='MSCMR_dataset', type=str,
                        help='multi-sequence CMR segmentation dataset')
    parser.add_argument('--output_dir', default='/data/zhangke/MSCMR_ShapePU/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default = '0', help = 'Ids of GPUs')    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/data/zhangke/MSCMR_EM_PU_resume/best_checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default = False, action='evaluate the model on TestSet')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, visualizer = build_model(args)
    model.to(device)
    print(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print('Building validation dataset...')
    dataset_val_dict = build(image_set='val', args=args)
    num_val = [len(v) for v in dataset_val_dict.values()]
    print('Number of validation images: {}'.format(sum(num_val)))

    sampler_val_dict = {k : torch.utils.data.SequentialSampler(v) for k, v in dataset_val_dict.items()}

    dataloader_val_dict = {
        k : DataLoader(v1, args.batch_size, sampler=v2, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
        for (k, v1), v2 in zip(dataset_val_dict.items(), sampler_val_dict.values())
        }

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.whst.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        infer(model, criterion, device)

    print("Start training")
    best_dic = None
    best_dice = None
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        # the negative loss of PU learning is evoked after 100 epoches
        estimate_alpha = True
        if epoch <= 99:
            estimate_alpha = False
        
        # dataset_hold_dict is optional in shapePU, you can use it to valid if the estimated ratio is accurate. In the training, we do not use it.
        dataset_train_dict, dataset_hold_dict = build(image_set='train', args=args)
        sampler_train_dict = {k : torch.utils.data.RandomSampler(v) for k, v in dataset_train_dict.items()}
        sampler_hold_dict = {k : torch.utils.data.RandomSampler(v) for k, v in dataset_hold_dict.items()}
        batch_sampler_train = { 
        k : torch.utils.data.BatchSampler(v, args.batch_size, drop_last=True) for k, v in sampler_train_dict.items()
        }
        batch_sampler_hold = { 
        k : torch.utils.data.BatchSampler(v, args.batch_size, drop_last=True) for k, v in sampler_hold_dict.items()
        }
        dataloader_train_dict = {
            k : DataLoader(v1, batch_sampler=v2, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
            for (k, v1), v2 in zip(dataset_train_dict.items(), batch_sampler_train.values())
            }
        dataloader_hold_dict = {
            k : DataLoader(v1, batch_sampler=v2, collate_fn=utils.collate_fn, num_workers=args.num_workers) 
            for (k, v1), v2 in zip(dataset_hold_dict.items(), batch_sampler_hold.values())
            }
        optimizer.param_groups[0]['lr'] = 1e-4
        # optimizer.param_groups[0]['lr'] = clr.cyclic_learning_rate(epoch, mode='exp_range', gamma=1)
        train_stats = train_one_epoch(model, criterion, dataloader_train_dict, optimizer, device, epoch, dataloader_hold_dict, estimate_alpha)
        test_stats = evaluate(model, criterion, postprocessors, dataloader_val_dict, device, args.output_dir, visualizer, epoch, writer)
        dice_score = test_stats["Avg"]
        print("dice score:", dice_score)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if best_dice == None or dice_score > best_dice:
                best_dice = dice_score
                best_dic = deepcopy(test_stats)
                print("Update best model!")
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
            if dice_score > 0.73:
                print("Update high dice score model!")
                file_name = str(dice_score)[0:6]+'high_checkpoint.pth'
                checkpoint_paths.append(output_dir / file_name)
                
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    print(torch.cuda.is_available())
    main(args)
