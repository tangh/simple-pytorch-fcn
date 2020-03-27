import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import tqdm

from models.fcn32s import fcn32s
from models.fcn16s import fcn16s
from models.fcn8s import fcn8s
from utils.dataset import VOCConfigs
from utils.visualization import label2rgb, visualize_demo


parser = argparse.ArgumentParser(description="FCN Training With Pytorch")
parser.add_argument("--model", default="fcn8s",
                    help='directory for test images')
parser.add_argument('--checkpoint', help='path for model parameters')
parser.add_argument('--input-dir', default='./demo',
                    help='directory for test images')
args = parser.parse_args()

if args.model == "fcn32s":
    model = fcn32s(n_class=21)
elif args.model == "fcn16s":
    model = fcn16s(n_class=21)
elif args.model == "fcn8s":
    model = fcn8s(n_class=21)
if args.checkpoint is None:
    args.checkpoint = "./checkpoints/" + args.model + "/model_best.pth.tar"
model.load_state_dict(
	torch.load(args.checkpoint)["model_state_dict"], strict=False
)
model.eval()

configs = VOCConfigs()
image_filenames = os.listdir(args.input_dir)

for filename in tqdm.tqdm(
    image_filenames, total=len(image_filenames),
    desc="generating results...", ncols=80, ascii=True
):

    img = Image.open(os.path.join(args.input_dir, filename))
    img = np.array(img)
    input = img.astype(np.float)[:, :, ::-1] - configs.mean_bgr
    input = torch.from_numpy(input.transpose(2, 0, 1)).float()

    with torch.no_grad():
        output = model(input.unsqueeze(0))
    output = F.softmax(output, dim=1).squeeze()
    output = torch.argmax(output, dim=0).numpy()  # 2D HW

    rgb_output = label2rgb(output, n_labels=21)
    visualize_output = visualize_demo(
        img=img, prediction=output,
        n_class=21, label_names=configs.class_names
    )

    rgb_output = Image.fromarray(rgb_output)
    rgb_output.save(os.path.join(
        args.input_dir,
        os.path.splitext(filename)[0] + "-" + args.model + ".png"
    ), "png")

    visualize_output = Image.fromarray(visualize_output)
    visualize_output.save(os.path.join(
        args.input_dir,
        os.path.splitext(filename)[0] + "-" + args.model + "-v.png"
    ), "png")
