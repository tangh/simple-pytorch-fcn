import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from models.fcn32s import fcn32s
from models.fcn16s import fcn16s
from models.fcn8s import fcn8s
from utils.visualization import label2rgb


parser = argparse.ArgumentParser(description="FCN Training With Pytorch")
parser.add_argument("--model", default="fcn8s",
                    help='directory for test images')
parser.add_argument('--checkpoint', default='./checkpoints/fcn8s/model_best.pth.tar',
                    help='path for model parameters')
parser.add_argument('--input-dir', default='./demo',
                    help='directory for test images')
args = parser.parse_args()

if args.model == "fcn32s":
    model = fcn32s(n_class=21)
elif args.model == "fcn16s":
    model = fcn16s(n_class=21)
elif args.model == "fcn8s":
    model = fcn8s(n_class=21)
model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])

mean_bgr = np.array([104.00699, 116.66877, 122.67892])
for filename in os.listdir(args.input_dir):

    img = Image.open(os.path.join(args.input_dir, filename))
    img = np.array(img).astype(np.float)[:, :, ::-1] - mean_bgr
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    with torch.no_grad():
        output = model(img.unsqueeze(0))

    output = F.softmax(output, dim=1).squeeze()
    output = torch.argmax(output, dim=0)  # 2D HW
    output = label2rgb(output.numpy(), n_labels=21)

    output = Image.fromarray(output)
    output.save(os.path.join(
        args.input_dir,
        os.path.splitext(filename)[0] + "-" + args.model + ".png"
    ), "png")
