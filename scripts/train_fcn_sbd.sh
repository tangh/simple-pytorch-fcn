python ./train.py --model fcn32s --pretrained-model ./checkpoints/vgg16-00b39a1b.pth \
--lr 1e-10 --fix-deconv --cuda --gpu-id 0 --dataset sbd --save-dir ./checkpoints/fcn32s-sbd

python ./train.py --model fcn16s --pretrained-model ./checkpoints/fcn32s-sbd/model_best.pth.tar \
--lr 1e-12 --fix-deconv --cuda --gpu-id 0 --dataset sbd --save-dir ./checkpoints/fcn16s-sbd

python ./train.py --model fcn8s --pretrained-model ./checkpoints/fcn16s-sbd/model_best.pth.tar \
--lr 1e-14 --fix-deconv --cuda --gpu-id 0 --dataset sbd --save-dir ./checkpoints/fcn8s-sbd