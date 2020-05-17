python ./train.py --model fcn32s --pretrained-model ./checkpoints/vgg16-00b39a1b.pth \
--lr 1e-10 --fix-deconv --cuda --gpu-id 1 --dataset voc --save-dir ./checkpoints/fcn32s-voc

python ./train.py --model fcn16s --pretrained-model ./checkpoints/fcn32s-voc/model_best.pth.tar \
--lr 1e-12 --fix-deconv --cuda --gpu-id 1 --dataset voc --save-dir ./checkpoints/fcn16s-voc

python ./train.py --model fcn8s --pretrained-model ./checkpoints/fcn16s-voc/model_best.pth.tar \
--lr 1e-14 --fix-deconv --cuda --gpu-id 1 --dataset voc --save-dir ./checkpoints/fcn8s-voc