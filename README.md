A simple pytorch implement of FCN. It use common dependency and don't need to be built.


# Requirements

+ python3
+ pytorch[https://github.com/pytorch/pytorch] >= 1.0.0
+ numpy, scipy, Pillow(PIL), scikit-image, opencv, tqdm


# Usage

Run `python train.py --model fcn32s --dataset voc --max-iter 20000 --save-dir ./checkpoints/fcn32s`

**Arguments:**
+ `--model`: `fcn32s` or `fcn16s` or `fcn8s`.
+ `--pretrained-model`: Path to a pretrained checkpoint. fcn16s(fcn8s) need to initialize with fcn32s(fcn16s) pretrained model.  
For fcn32s, it will download pretrained VGG Net model by torchvision.
+ `--max-iter`: Max training iterations, max epoch will be auto calculate by `max-iter / batchsize`.
+ `--lr`, `--momentum`, `--weight-decay`: Optimizer configs, lr for bias will double.
+ `--cuda`: Whether to use GPU training.
+ `--dataset`: Use VOC dataset or SBD dataset.
+ `--dataset-root`: Path to dataset.
+ `--save-dir`: Path to save checkpoints, visulized results, and log file.
+ `--resume`: Path to a checkpoint file used to resume training process.


# Results
TODO


# References

+ wkentaro/pytorch-fcn[https://github.com/wkentaro/pytorch-fcn]
+ maskrcnn-benchmark[https://github.com/facebookresearch/maskrcnn-benchmark]
