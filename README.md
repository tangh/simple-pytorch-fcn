A simple pytorch implement of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org). It use common dependencies and don't need to be built.



# Requirements

+ python3
+ [pytorch](https://github.com/pytorch/pytorch) >= 1.1.0
+ numpy, scipy, pillow(pil), scikit-image, opencv, tqdm

`pip install -r requirements.txt`

Test on Python 3.8 and PyTorch 1.4.



# Usage

## Training

Run `python train.py --model fcn32s --dataset voc --max-iter 20000 --save-dir ./checkpoints/fcn32s`. *Or* `sh ./scripts/train_fcn_sbd.sh` or `sh ./scripts/train_fcn_voc.sh`.

**Arguments:**
+ `--model`: `fcn32s` or `fcn16s` or `fcn8s`.
+ `--pretrained-model`: Path to a pretrained checkpoint. fcn16s(fcn8s) need to initialize from fcn32s(fcn16s) pretrained model. fcn32s need to initialize from VGG pretrained model.
+ `--batch-size`: Batch size of train or val. If set to 1, image and label will be returned as is, otherwise image and label will be resized to 500×500.
+ `--max-iter`: Max training iterations, max epoch will be auto calculate by `max-iter / batchsize`.
+ `--lr`, `--momentum`, `--weight-decay`: Optimizer configs, lr for bias will double.
+ `--cuda`: Whether to use GPU training.
+ `--dataset`: Use [VOC 2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) (1464train/1449val) or [SBD Dataset](http://home.bharathh.info/home/sbd) (8498train/736val).
+ `--dataset-root`: Path to dataset.
+ `--save-dir`: Path to save checkpoints, visulized results, and log file.
+ `--resume`: Path to a checkpoint file used to resume training process.

**Note:**

For VGG pretrained model, torchvision auto downloaded model require input image value between `[0, 1]` and then normalized with `mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]` in RGB format, as in [pytorch/examples](https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L207); while [caffe pretrained model](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014) require input image value between `[0, 255]` and then substracted by `[103.939, 116.779, 123.68]` in BGR format.

PyTorch version caffe pretrained model can be found at [jcjohnson/pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg).

Direct download SBD Dataset [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).


## Demo

Run `python demo.py --model fcn32s --checkpoint ./checkpoints/fcn32s/model_best.pth.tar --input-dir ./demo`



# Results

| \[mIoU\] | FCN-32s | FCN-16s | FCN-8s |
| :------: | :-----: | :-----: | :----: |
|    VOC   |  57.86  |  59.74  |  60.14 |
|    SBD   |  62.64  |  64.48  |  64.87 |
| Original CVPR | 59.4 | 62.4 | 62.7 |
| Original PAMI | 63.6 | 65.0 | 65.5 |

See `./checkpoints` dir for more details.



# References

+ [shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)
+ [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
+ [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
