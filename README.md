A simple pytorch implement of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org). It use common dependency and don't need to be built.


# Requirements

+ python3
+ [pytorch](https://github.com/pytorch/pytorch) >= 1.0.0
+ numpy, scipy, Pillow(PIL), scikit-image, opencv, tqdm

Test on Python 3.8 and PyTorch 1.4.

# Usage

Run `python train.py --model fcn32s --dataset voc --max-iter 20000 --save-dir ./checkpoints/fcn32s`

**Arguments:**
+ `--model`: `fcn32s` or `fcn16s` or `fcn8s`.
+ `--pretrained-model`: Path to a pretrained checkpoint. fcn16s(fcn8s) need to initialize from fcn32s(fcn16s) pretrained model. fcn32s need to initialize from VGG pretrained model.
+ `--max-iter`: Max training iterations, max epoch will be auto calculate by `max-iter / batchsize`.
+ `--lr`, `--momentum`, `--weight-decay`: Optimizer configs, lr for bias will double.
+ `--cuda`: Whether to use GPU training.
+ `--dataset`: Use VOC dataset or SBD dataset.
+ `--dataset-root`: Path to dataset.
+ `--save-dir`: Path to save checkpoints, visulized results, and log file.
+ `--resume`: Path to a checkpoint file used to resume training process.

**Note:**

For VGG pretrained model, torchvision auto downloaded model require input image value between `[0, 1]` and then normalized with `mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]` in RGB format, as in [pytorch/examples](https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L207); while [caffe pretrained model](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014) require input image value between `[0, 255]` and then substracted by `[103.939, 116.779, 123.68]` in BGR format.

Caffe pretrained model can be found at [jcjohnson/pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg).


# Results
TODO


# References

+ [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
+ [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
