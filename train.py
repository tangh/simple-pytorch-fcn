import argparse
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fcn32s import fcn32s
from models.fcn16s import fcn16s
from models.fcn8s import fcn8s
from utils import dataset, trainer
from utils.misc import setup_logger


parser = argparse.ArgumentParser(description="FCN Training With Pytorch")
parser.add_argument("--model", type=str, default="fcn32s",
                    choices=["fcn32s", "fcn16s", "fcn8s"],
                    help="use which model (default: fcn32s)")
parser.add_argument('--pretrained-model',
                    default='./checkpoints/fcn32s/model_best.pth.tar',
                    help='pretrained model')
parser.add_argument('--max-iter', type=int, default=10000, help='max iter')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda device')
parser.add_argument('--dataset', type=str, default='voc',
                    choices=['voc', 'sbd'],
                    help='use which dataset (default: voc2012)')
parser.add_argument('--dataset-root', default='./datasets',
                    help='Directory for datasets')
parser.add_argument('--save-dir', default='./checkpoints',
                    help='Directory for saving checkpoints')
parser.add_argument('--resume', help='resume training from checkpoint path')
args = parser.parse_args()


# [0] prepare envs
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)
logger = setup_logger("simple-pytorch-fcn", args.save_dir)
logger.info(args)


# [1] dataset
if args.dataset == "voc":
    train_loader = DataLoader(
        dataset.VOCClassSegBase(root=args.dataset_root, split="train"),
        batch_size=1, shuffle=True
    )
    val_loader = DataLoader(
        dataset.VOCClassSegBase(root=args.dataset_root, split="val"),
        batch_size=1, shuffle=True
    )
elif args.dataset == "sbd":
    train_loader = DataLoader(
        dataset.SBDClassSeg(root=args.dataset_root, split="train"),
        batch_size=1, shuffle=True
    )
    val_loader = DataLoader(
        dataset.VOC2011ClassSeg(root=args.dataset_root, split="seg11valid"),
        batch_size=1, shuffle=True
    )


# [2] network
if args.model == "fcn32s":
    model = fcn32s(n_class=21)
elif args.model == "fcn16s":
    model = fcn16s(n_class=21)
elif args.model == "fcn8s":
    model = fcn8s(n_class=21)

start_epoch = 0
start_iteration = 0
best_mean_iou = 0

if args.resume:
    checkpoint = torch.load(args.resume)
    # we dont save parameters in convtranspose since it is fixed
    # and used as a bilinear upsample, so state_dict in checkpoint
    # will miss some keys
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    start_epoch = checkpoint["epoch"]
    start_iteration = checkpoint["iteration"]
    best_mean_iou = checkpoint["best_mean_iou"]
else:
    if args.model == "fcn32s":
        import torchvision
        vgg16 = torchvision.models.vgg16()
        checkpoint = torch.load(args.pretrained_model)
        checkpoint = OrderedDict(
            [
                (k.replace(k.split(".")[1], str(int(k.split(".")[1])-1)), v)
                if "classifier" in k 
                else (k, v) for k, v in checkpoint.items()
            ]
        )
        vgg16.load_state_dict(checkpoint, strict=False)
        model.copy_params_from_vgg16(vgg16)
    elif args.model == "fcn16s":
        fcn32s_model = fcn32s(n_class=21)
        checkpoint = torch.load(args.pretrained_model)
        fcn32s_model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        model.copy_params_from_fcn32s(fcn32s_model)
    elif args.model == "fcn8s":
        fcn16s_model = fcn16s(n_class=21)
        checkpoint = torch.load(args.pretrained_model)
        fcn16s_model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        model.copy_params_from_fcn16s(fcn16s_model)


# [3] optimizer
def get_parameters(model, bias=False):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight


optimizer = torch.optim.SGD(
    [
        {'params': get_parameters(model, bias=False)},
        {'params': get_parameters(model, bias=True),
            'lr': args.lr * 2, 'weight_decay': 0},
    ],
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
if args.resume:
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=-1)


# [4] training
fcn_trainer = trainer.Trainer(
    device=device,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir=args.save_dir,
    max_iter=args.max_iter,
    validate_interval=4000,
)
fcn_trainer.epoch = start_epoch
fcn_trainer.iteration = start_iteration
fcn_trainer.best_mean_iou = best_mean_iou

fcn_trainer.train()
