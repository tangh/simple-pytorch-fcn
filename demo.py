import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from models.fcn32s import fcn32s

parser = argparse.ArgumentParser(description='FCN Training With Pytorch')
parser.add_argument('--checkpoint', default='./checkpoints/model_best.pth.tar',
                    help='path for model parameters')
parser.add_argument('--input-dir', default='./demo',
                    help='directory for test images')
args = parser.parse_args()

model = fcn32s(n_class=21)
model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])

mean_bgr = np.array([104.00699, 116.66877, 122.67892])
for img in os.listdir(args.input_dir):

    img = Image.open(os.path.join(args.input_dir, img))
    img = np.array(img).astype(np.float)[:, :, ::-1] - mean_bgr
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    with torch.no_grad():
        output = model(img.unsqueeze(0))

    output = F.softmax(output, dim=1).squeeze()
    output = torch.argmax(output, dim=0)

    output = Image.fromarray(output.numpy().astype("uint8") * 100)
    output.show()
